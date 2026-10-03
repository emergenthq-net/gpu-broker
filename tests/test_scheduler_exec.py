"""The GPU thread around a job's end: the terminal state is recorded first, then the job's
staged files are dropped, and a failure to drop them never changes the outcome. A recipe that
may still hold the GPU (GpuHeld) holds the GPU: no job runs and no model is restored until a
clean confirms the job gone or an operator clears the hold; resume and restarts keep it."""
import base64
import time

import pytest
import yaml

from gpu_broker.broker import Broker
from gpu_broker.constants import Event, JobState
from gpu_broker.drivers import GpuHeld
from gpu_broker.execjob import TimeoutTooShort
from tests.helpers import WAIT_S, FakeBackends, FakeDriver, done, leftover_staged, make_settings

B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64).decode()


def job(client, **body):
    return client.post("/v1/jobs", json=body)


def test_the_terminal_state_comes_before_the_files_are_dropped(client, broker, monkeypatch):
    states, real = [], broker.staging.discard

    def discard(jid):
        states.append(broker.store.job(jid)["state"])
        real(jid)
    monkeypatch.setattr(broker.staging, "discard", discard)
    ok = done(broker, job(client, model="sharp", image=B64).json()["id"])
    broker.driver.recipe_error = "boom"
    bad = done(broker, job(client, model="sharp", image=B64).json()["id"])
    assert (ok["state"], bad["state"]) == (JobState.DONE, JobState.FAILED)
    assert leftover_staged(broker) == [] and states == [JobState.DONE, JobState.FAILED]


def test_a_failed_drop_changes_nothing(client, broker, monkeypatch):
    def broken(jid):
        raise OSError(13, "Permission denied")
    monkeypatch.setattr(broker.staging, "discard", broken)
    ok = done(broker, job(client, model="sharp", image=B64).json()["id"])
    broker.driver.recipe_error = "recipe sharp exited 1: boom"
    bad = done(broker, job(client, model="sharp", image=B64).json()["id"])
    assert ok["state"] == JobState.DONE and ok["result"]["outputs"]
    assert bad["state"] == JobState.FAILED and bad["error"] == "recipe sharp exited 1: boom"
    after = done(broker, job(client, model="llama-8b", prompt="hi").json()["id"])   # the GPU thread lives on
    assert after["state"] == JobState.DONE
    assert Event.WORKER_ERROR not in [e["kind"] for e in broker.store.events(0, 1000)]   # swallowed, not escaped


def test_a_broker_whose_exec_timeout_is_too_short_does_not_start(tmp_path):
    driver = FakeDriver()
    driver.recipe_seconds = 640   # sharp's 660 needs a recipe of at most 605 (+ 10 kill, 15 reap, 30)
    with pytest.raises(TimeoutTooShort, match=r"sharp: exec\.timeout_s 660 must be at least 695"):
        Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))


def hold_gpu(client, broker, monkeypatch):
    """Run an exec job whose recipe may survive; the held job's clean keeps failing."""
    def held(*a, **k):
        raise GpuHeld("recipe sharp for job x may still be running on the host: still running")
    monkeypatch.setattr(broker.driver, "run_recipe", held)
    broker.driver.clean_error = "exec-clean of job x failed (7)"
    first = done(broker, job(client, model="sharp", image=B64).json()["id"])
    assert first["state"] == JobState.FAILED and "may still be running" in first["error"]
    return first


def test_a_recipe_that_may_still_run_holds_the_gpu_through_resume_until_cleared(client, broker, monkeypatch):
    first = hold_gpu(client, broker, monkeypatch)
    held = client.get("/v1/status").json()["gpu_held"]
    assert (held["job"], held["recipe"]) == (first["id"], "sharp") and "may still be running" in held["reason"]
    assert not broker.scheduler.paused.is_set()   # not a quiesce: its own persisted state
    nxt = job(client, model="llama-8b", prompt="hi").json()["id"]
    assert client.post("/v1/admin/resume").json() == {"paused": False, "gpu_held": True}
    time.sleep(0.3)
    assert broker.store.job(nxt)["state"] == JobState.QUEUED   # nothing takes the GPU meanwhile
    cleared = client.post("/v1/admin/gpu-held/clear").json()["cleared"]
    assert cleared["job"] == first["id"] and client.get("/v1/status").json()["gpu_held"] is None
    assert broker.wait(nxt, WAIT_S)["state"] == JobState.DONE
    kinds = [e["kind"] for e in broker.store.events(0, 1000)]
    assert Event.EXEC_GPU_HELD in kinds and Event.GPU_HELD_CLEARED in kinds
    assert client.post("/v1/admin/gpu-held/clear").json() == {"cleared": None}


def test_the_hold_survives_a_restart(client, broker, monkeypatch, tmp_path):
    first = hold_gpu(client, broker, monkeypatch)
    again = Broker(broker.settings, env={}, driver=broker.driver, backends=broker.backends)
    again.start()
    try:
        assert again.scheduler.hold.get()["job"] == first["id"]
        nxt, _ = again.submit({"model": "llama-8b", "prompt": "hi"}, "t")
        time.sleep(0.3)
        assert again.store.job(nxt)["state"] == JobState.QUEUED
    finally:
        again.scheduler.hold.clear("test")
        again.wait(nxt, WAIT_S)
        again.stop()


def test_a_clear_or_a_stop_wakes_the_held_gpu_thread_at_once(client, broker, monkeypatch):
    """held_retry_s is 60 s here: the GPU thread must not sleep that out after a clear or stop."""
    hold_gpu(client, broker, monkeypatch)
    time.sleep(0.2)   # the GPU thread is now waiting out the hold
    t0 = time.monotonic()
    nxt = job(client, model="llama-8b", prompt="hi").json()["id"]
    client.post("/v1/admin/gpu-held/clear")
    assert broker.wait(nxt, WAIT_S)["state"] == JobState.DONE and time.monotonic() - t0 < 2
    hold_gpu(client, broker, monkeypatch)
    time.sleep(0.2)
    gpu = next(t for t in broker._threads if t.name == broker.scheduler.loop.__qualname__)
    broker.stop()
    gpu.join(2)
    assert not gpu.is_alive()


def test_restarting_mid_exec_holds_the_gpu_before_any_job_runs(tmp_path):
    """The old process died while an exec job ran: its program may still hold the card."""
    driver = FakeDriver()
    driver.clean_error = "exec-clean of job x failed (7)"
    first = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))
    running, _ = first.submit({"model": "sharp", "image": B64}, "t")
    first.store.update_job(running, state=JobState.RUNNING, resolved="sharp")
    queued, _ = first.submit({"model": "sharp", "image": B64}, "t")   # never started: no hold
    first.store.update_job(queued, resolved="sharp")
    again = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))
    again.start()
    try:
        held = again.scheduler.hold.get()   # set in start(), before the GPU thread runs
        nxt, _ = again.submit({"model": "llama-8b", "prompt": "hi"}, "t")
        assert held["job"] == running and held["recipe"] == "sharp" and "restarted" in held["reason"]
        assert again.store.job(running)["state"] == JobState.FAILED
        time.sleep(0.3)
        assert again.store.job(nxt)["state"] == JobState.QUEUED   # nothing ran
        assert driver.cleans[:1] == [("sharp", running)]   # cleaned at once, in the background
        driver.clean_error = None
        again.scheduler.hold._next = 0   # the next background retry is due now...
        again.scheduler.hold.changed.set()   # ...and the GPU thread looks again
        assert again.wait(nxt, WAIT_S)["state"] == JobState.DONE   # the clean cleared it
        assert again.scheduler.hold.get() is None
    finally:
        again.stop()


def test_while_held_no_model_is_restored(client, broker, monkeypatch):
    hold_gpu(client, broker, monkeypatch)
    restores = []
    monkeypatch.setattr(broker.scheduler, "maybe_restore", lambda: restores.append(1))
    time.sleep(0.3)
    assert restores == []


def test_a_timeout_found_too_short_at_run_time_evicts_nothing(tmp_path):
    """Unreadable at startup (only logged), too short when the job comes: the check runs before
    residency changes, so the resident LLM keeps running."""
    driver = FakeDriver({"llama-server"})
    real, reads = driver.recipe_info, []

    def info(recipe):
        reads.append(recipe)
        if len(reads) == 1:
            raise OSError("ssh: no route to host")
        return real(recipe)
    driver.recipe_info = info
    b = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))
    b.start()
    try:
        driver.recipe_seconds = 640   # sharp's 660 needs at most 605
        jid, _ = b.submit({"model": "sharp", "image": B64}, "t")
        j = b.wait(jid, WAIT_S)
        assert j["state"] == JobState.FAILED and "must be at least" in j["error"]
        assert ("stop", "llama-server") not in driver.calls and "llama-server" in driver.active
        assert b.backends.frees == 0 and driver.recipes == []
    finally:
        b.stop()


def test_a_restart_cleans_an_orphan_whose_model_left_the_catalog_by_its_recorded_recipe(tmp_path):
    driver = FakeDriver()
    first = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))
    running, _ = first.submit({"model": "sharp", "image": B64}, "t")
    assert first.store.job(running)["exec_recipe"] == "sharp"   # recorded at submit
    first.store.update_job(running, state=JobState.RUNNING, resolved="sharp")
    pruned = yaml.safe_load((tmp_path / "catalog.yaml").read_text())
    del pruned["models"]["sharp"]                                          # the operator removed it
    (tmp_path / "pruned.yaml").write_text(yaml.safe_dump(pruned))
    again = Broker(make_settings(tmp_path, tmp_path / "pruned.yaml"), env={}, driver=driver,
                   backends=FakeBackends(driver))
    again.start()
    try:
        nxt, _ = again.submit({"model": "llama-8b", "prompt": "hi"}, "t")
        assert again.wait(nxt, WAIT_S)["state"] == JobState.DONE   # the clean confirmed it gone
        assert driver.cleans[:1] == [("sharp", running)] and again.scheduler.hold.get() is None
    finally:
        again.stop()
