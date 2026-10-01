"""The GPU thread: parallel calls to the resident LLM, drain before a switch, idle restore."""
import threading
import time

from gpu_broker.constants import JobState
from tests.helpers import WAIT_S, done


def submit(b, model, **body):
    return b.submit({"model": model, "messages": [], **body}, "test")[0]


def test_background_calls_overlap_up_to_the_unreserved_slots(broker):
    gate = broker.backends.chat_gate = threading.Event()
    m = broker.catalog.models["llama-8b"]
    slots = m["slots"] - m["reserved_interactive"]
    jids = [submit(broker, "llama-8b") for _ in range(slots + 1)]
    try:
        end = time.monotonic() + WAIT_S
        while broker.scheduler.position(jids[-1]) != 0 and time.monotonic() < end:
            time.sleep(0.005)
        assert len(broker.scheduler.pool.ids()) == slots   # the extra call holds the GPU thread, waiting for a slot
        assert broker.store.job(jids[-1])["state"] == JobState.QUEUED
    finally:
        gate.set()
    assert all(done(broker, j)["state"] == JobState.DONE for j in jids)


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
    assert broker.store.fail_orphans() == 1
    assert broker.store.job(jid)["state"] == JobState.FAILED
