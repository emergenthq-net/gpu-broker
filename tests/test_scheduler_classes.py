"""The live `fair` queue, continued: no switch jumps a call waiting for a slot on the resident
model; who may claim interactive; re-queued jobs get a class and keep their order ahead of new
work; submit's position; expected run times refreshed only under `fair`."""
import dataclasses
import threading
import time

from gpu_broker.broker import Broker
from gpu_broker.constants import JobState
from gpu_broker.settings import Scheduling
from tests.helpers import WAIT_S, FakeBackends, FakeDriver, done, make_settings, wait_idle
from tests.test_scheduler_fair import HEAVY, call, interactive, make, started  # noqa: F401 — `make` is a fixture


def until(cond):
    end = time.monotonic() + WAIT_S
    while not cond() and time.monotonic() < end:
        time.sleep(0.005)
    return cond()


def test_a_switch_waits_for_the_call_the_resident_model_is_about_to_serve(make):
    b = make()
    b.backends.chat_gate = gate = threading.Event()
    m = b.catalog.models["llama-8b"]
    busy = [call(b, HEAVY) for _ in range(m["slots"] - m["reserved_interactive"])] + [call(b, "op1")]
    assert until(lambda: len(b.scheduler.pool.ids()) == m["slots"])   # every slot busy
    x = call(b, "op2")                                                  # interactive, waits for a slot
    y = b.submit({"model": "sdxl-base", "prompt": "y"}, "artist", priority="background")[0]
    time.sleep(0.1)
    assert b.store.job(y)["state"] == JobState.QUEUED and "comfyui" not in b.driver.active   # no switch yet
    gate.set()
    assert started(b, x) < started(b, y)                                # x ran on llama first, then the switch
    assert all(done(b, j)["state"] == JobState.DONE for j in busy)


def test_only_the_listed_requesters_may_claim_interactive(make, tmp_path):
    (d := tmp_path / "claims").mkdir()
    s = dataclasses.replace(make_settings(d), scheduler=Scheduling(may_claim_interactive=("op",)))
    driver = FakeDriver({"llama-8b"})
    b = Broker(s, env={}, driver=driver, backends=FakeBackends(driver))
    b.start()
    try:
        assert interactive(b, call(b, "op", "interactive")) is True
        assert interactive(b, call(b, "guest", "interactive")) is False   # not listed: the header cannot raise it
        assert interactive(b, call(b, "guest")) is True                   # no claim: its requester decides
    finally:
        assert wait_idle(b)
        b.stop()


def test_requeued_jobs_get_a_class_and_run_before_new_work_of_their_class(tmp_path):
    def new_broker(start):
        driver = FakeDriver({"llama-8b"})
        b = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))
        if start:
            b.start()
        return b
    old = new_broker(False)
    old_ids = [old.store.create_job(req, "sdxl-base", {"model": "sdxl-base", "prompt": p}) for req, p in
               (("light", "a"), (HEAVY, "b"), ("light", "c"))]   # queued by a version without classes
    for j in old_ids:
        old.store.update_job(j, resolved="sdxl-base", state=JobState.QUEUED)
    old.stop()
    new = new_broker(False)
    new.scheduler.paused.set()
    new.start()
    try:
        assert [new.store.job(j)["payload"]["interactive"] for j in old_ids] == [True, False, True]
        new.scheduler.paused.clear()
        assert [done(new, j)["state"] for j in old_ids] == [JobState.DONE] * 3
        assert sorted(old_ids, key=lambda j: started(new, j)) == [old_ids[0], old_ids[2], old_ids[1]]   # by class
    finally:
        assert wait_idle(new)
        new.stop()


def test_requeued_jobs_stay_ahead_of_new_work_in_their_class(tmp_path):
    driver = FakeDriver({"llama-8b"})
    b = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))
    try:
        for req in ("a", "a", "a"):   # fair alone would let b's new job in after a's first
            jid = b.store.create_job(req, "sdxl-base", {"model": "sdxl-base", "interactive": False})
            b.store.update_job(jid, resolved="sdxl-base", state=JobState.QUEUED)
        old = [j["id"] for j in sorted(b.store.jobs(10), key=lambda j: j["created"])]
        b.scheduler.requeue(old)
        new = b.submit({"model": "sdxl-base", "prompt": "x"}, "b", priority="background")[0]
        assert b.scheduler.snapshot()[0] == [*old, new]
    finally:
        b.stop()


def test_submit_returns_the_position_it_was_given(tmp_path):
    driver = FakeDriver({"llama-8b"})
    b = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))   # GPU thread not started
    try:
        heavy = [b.submit({"model": "sdxl-base", "prompt": "h"}, HEAVY)[1]["queue_position"] for _ in range(3)]
        light = b.submit({"model": "sdxl-base", "prompt": "l"}, "light", priority="background")[1]["queue_position"]
        assert heavy == [1, 2, 3] and light == 2   # light joins level with heavy's first: ahead of its backlog
    finally:
        b.stop()


def test_expected_run_times_are_refreshed_only_under_fair(make):
    names = {t.name for t in make()._threads}
    assert "Costs.loop" in names and "Costs.loop" not in {t.name for t in make("fifo")._threads}


def test_requeued_jobs_keep_their_order_across_requesters(tmp_path):
    driver = FakeDriver({"llama-8b"})
    b = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))
    try:
        old = []
        for req in ("a", "b", "a", "b"):   # fair alone would interleave new work from c after a's first
            jid = b.store.create_job(req, "sdxl-base", {"model": "sdxl-base", "interactive": False})
            b.store.update_job(jid, resolved="sdxl-base", state=JobState.QUEUED)
            old.append(jid)
        b.scheduler.requeue(old)
        new = b.submit({"model": "sdxl-base", "prompt": "x"}, "c", priority="background")[0]
        assert b.scheduler.snapshot()[0] == [*old, new]
    finally:
        b.stop()


def test_a_freed_slot_wakes_the_gpu_thread_without_waiting_for_its_poll(tmp_path):
    slow = dataclasses.replace(make_settings(tmp_path).intervals, worker_poll_s=30)
    driver = FakeDriver({"llama-8b"})
    b = Broker(make_settings(tmp_path, intervals=slow), env={}, driver=driver, backends=FakeBackends(driver))
    b.start()
    try:
        b.backends.chat_gate = gate = threading.Event()
        m = b.catalog.models["llama-8b"]
        busy = [call(b, "op") for _ in range(m["slots"])]
        assert until(lambda: len(b.scheduler.pool.ids()) == m["slots"])
        late = call(b, "op2")                  # waits for a slot (fair skips it, so the thread sleeps on the line)
        time.sleep(0.1)
        t0 = time.monotonic()
        gate.set()                             # slots free: touch() kicks the line
        assert done(b, late)["state"] == JobState.DONE and time.monotonic() - t0 < 5
        assert all(done(b, j)["state"] == JobState.DONE for j in busy)
    finally:
        assert wait_idle(b)
        b.stop()
