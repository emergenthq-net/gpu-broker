"""Broker.wait (blocking job calls, chat via the queue) is woken by the store's job updates: it
returns as soon as the job ends, reads the job only when something changed, and still honours
its timeout. A broker that is not started, so nothing but the test touches the job."""
import threading
import time

from gpu_broker.broker import Broker
from gpu_broker.constants import JobState
from tests.helpers import FakeBackends, FakeDriver, make_settings

PROMPT_S = 0.1   # "as soon as": well under any polling interval worth having


def idle_broker(tmp_path):
    driver = FakeDriver()
    return Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))


def test_wait_returns_when_the_job_ends_without_polling(tmp_path, monkeypatch):
    b = idle_broker(tmp_path)
    jid = b.store.create_job("t", "m", {})
    reads, real_job = [], b.store.job
    monkeypatch.setattr(b.store, "job", lambda j: reads.append(j) or real_job(j))
    ended = []

    def finish():
        time.sleep(0.5)
        b.store.update_job(jid, state=JobState.RUNNING)
        time.sleep(0.2)
        ended.append(time.monotonic())
        b.store.update_job(jid, state=JobState.DONE)
    threading.Thread(target=finish).start()
    j = b.wait(jid, 10)
    assert j["state"] == JobState.DONE and time.monotonic() - ended[0] < PROMPT_S
    assert len(reads) <= 3   # once, then once per update: 0.7 s of polling would read far more


def test_wait_gives_up_at_its_timeout_and_returns_the_job(tmp_path):
    b = idle_broker(tmp_path)
    jid = b.store.create_job("t", "m", {})
    t0 = time.monotonic()
    assert b.wait(jid, 0.3)["state"] == JobState.RECEIVED
    assert 0.3 <= time.monotonic() - t0 < 0.3 + PROMPT_S


def test_wait_returns_at_once_for_an_ended_or_unknown_job(tmp_path):
    b = idle_broker(tmp_path)
    jid = b.store.create_job("t", "m", {})
    b.store.update_job(jid, state=JobState.FAILED)
    t0 = time.monotonic()
    assert b.wait(jid, 10)["state"] == JobState.FAILED and b.wait("nope", 10) is None
    assert time.monotonic() - t0 < PROMPT_S


def test_other_jobs_updates_do_not_end_the_wait(tmp_path):
    b = idle_broker(tmp_path)
    mine, other = b.store.create_job("t", "m", {}), b.store.create_job("t", "m", {})
    threading.Timer(0.1, b.store.update_job, (other,), {"state": JobState.DONE}).start()
    assert b.wait(mine, 0.4)["state"] == JobState.RECEIVED
