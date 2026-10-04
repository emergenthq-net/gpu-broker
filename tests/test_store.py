import threading

from gpu_broker.constants import JobState
from gpu_broker.store import Store


def test_events_paging_and_tail(tmp_path):
    st = Store(str(tmp_path / "b.db"), str(tmp_path / "e.jsonl"))
    for i in range(10):
        st.event("x", n=i)
    assert [e["data"]["n"] for e in st.events(-3, 100)] == [7, 8, 9]
    assert len(st.events(-1000, 5)) == 5 and len(st.events(2, 3)) == 3
    assert len((tmp_path / "e.jsonl").read_text().splitlines()) == 10


def test_downloads_are_idempotent_unless_failed(tmp_path):
    st = Store(str(tmp_path / "b.db"))
    assert st.upsert_download("s", "hf", "o/r", None) is True
    assert st.upsert_download("s", "hf", "o/r", None) is False
    st.set_download("s", "failed", "boom")
    assert st.upsert_download("s", "hf", "o/r", None) is True


def test_stats_counts_residency_and_errors_only_in_the_window(tmp_path):
    st = Store(str(tmp_path / "b.db"))
    st.event("residency.start")
    st.event("worker.error")
    st.event("job.received")
    assert st.stats(0)["events"] == {"residency.start": 1, "worker.error": 1}


def test_update_job_only_writes_known_columns(tmp_path):
    import pytest
    st = Store(str(tmp_path / "b.db"))
    jid = st.create_job("t", "m", {})
    with pytest.raises(ValueError, match="updatable"):
        st.update_job(jid, **{"state=?, requester": "x"})


def test_creates_missing_directories(tmp_path):
    Store(str(tmp_path / "a" / "b.db"), str(tmp_path / "logs" / "e.jsonl")).event("x")
    assert (tmp_path / "logs" / "e.jsonl").exists()



class CountingLock:
    def __init__(self) -> None:
        self.lock, self.taken = threading.Lock(), 0

    def __enter__(self) -> None:
        self.lock.acquire()
        self.taken += 1

    def __exit__(self, *exc: object) -> None:
        self.lock.release()


def test_a_state_change_and_its_event_are_written_under_one_lock(tmp_path):
    """Two acquisitions leave a gap in which a reader sees the job done with no `job.done`
    (a client paging /v1/events after /v1/jobs said done; seen as a flaky re-queue test)."""
    s = Store(str(tmp_path / "b.db"), str(tmp_path / "e.jsonl"))
    s._lock = lock = CountingLock()
    jid = s.create_job("t", "m", {})
    assert lock.taken == 1
    s.update_job(jid, state=JobState.DONE)
    assert lock.taken == 2
    assert s.job(jid)["state"] == JobState.DONE and s.events(0, 10)[-1]["kind"] == "job." + JobState.DONE
