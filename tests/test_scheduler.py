"""The GPU thread: parallel calls to the resident LLM, drain before a switch, idle restore."""
import threading
import time

from gpu_broker.constants import JobState
from tests.helpers import WAIT_S, done


def submit(b, model, priority="", **body):
    return b.submit({"model": model, "messages": [], **body}, "test", priority=priority)[0]


def wait_inflight(broker, n):
    end = time.monotonic() + WAIT_S
    while len(broker.scheduler.pool.ids()) < n and time.monotonic() < end:
        time.sleep(0.005)
    time.sleep(0.05)   # long enough for a call that should not start to start
    return len(broker.scheduler.pool.ids())


def test_background_calls_overlap_up_to_the_unreserved_slots_and_interactive_ones_use_the_rest(broker):
    gate = broker.backends.chat_gate = threading.Event()
    m = broker.catalog.models["llama-8b"]
    slots = m["slots"] - m["reserved_interactive"]
    jids = [submit(broker, "llama-8b", "background") for _ in range(slots + 1)]
    try:
        assert wait_inflight(broker, slots) == slots
        assert broker.store.job(jids[-1])["state"] == JobState.QUEUED and broker.scheduler.position(jids[-1]) == 1
        chat = submit(broker, "llama-8b", "interactive")   # not stuck behind the waiting background call
        assert wait_inflight(broker, slots + 1) == slots + 1 and chat in broker.scheduler.pool.ids()
    finally:
        gate.set()
    assert all(done(broker, j)["state"] == JobState.DONE for j in [*jids, chat])


def test_a_switch_waits_for_in_flight_calls(broker):
    gate = broker.backends.chat_gate = threading.Event()
    first = submit(broker, "llama-8b")
    switch = submit(broker, "qwen-coder-32b")
    try:
        time.sleep(10 * broker.settings.intervals.worker_poll_s)
        assert "qwen-coder" not in broker.driver.active and broker.store.job(switch)["state"] == JobState.QUEUED
    finally:
        gate.set()
    assert done(broker, first)["state"] == done(broker, switch)["state"] == JobState.DONE
    assert broker.driver.active == {"qwen-coder"}


def test_idle_restore_brings_the_default_back(broker):
    done(broker, submit(broker, "sdxl-base", prompt="x"))
    assert "llama-8b" not in broker.driver.active
    broker.catalog.defaults["idle_restore_s"] = 0
    def restored():
        return "residency.idle_restore" in [e["kind"] for e in broker.store.events(0, 1000)]
    end = time.monotonic() + WAIT_S
    while not restored() and time.monotonic() < end:   # the event is the last step of a restore
        time.sleep(0.01)
    assert restored() and broker.residency.current == "llama-8b" and "llama-8b" in broker.driver.active


def test_a_failing_job_is_reported_and_the_thread_survives(broker):
    broker.driver.fail_start.add("qwen-coder")
    assert "could not start" in done(broker, submit(broker, "qwen-coder-32b"))["error"]
    assert done(broker, submit(broker, "llama-8b"))["state"] == JobState.DONE


def test_restart_fails_orphans(broker, tmp_path):
    jid = broker.store.create_job("t", "llama-8b", {})
    broker.store.update_job(jid, state=JobState.RUNNING)
    assert len(broker.store.fail_orphans()[0]) == 1
    assert broker.store.job(jid)["state"] == JobState.FAILED
