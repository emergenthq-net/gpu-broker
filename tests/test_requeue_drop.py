"""Jobs re-queued at startup that can no longer run are failed with a reason and their staged
files removed, never left `queued` (and so never re-queued again): a model that left the
catalog in the deploy, or an interactive session nobody is watching. And a job the GPU thread
takes just as a quiesce lands goes back to the front, so the line keeps its order."""
from __future__ import annotations

import base64
import threading
import time

import pytest
import yaml

from gpu_broker.broker import Broker
from gpu_broker.constants import SESSION_KEY, Event, JobState
from gpu_broker.jobline import MODEL_GONE, SESSION_EXPIRED
from tests.helpers import FIX, WAIT_S, FakeBackends, FakeDriver, done, make_settings, wait_idle

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
IMAGE_JOB = {"model": "wan-i2v", "prompt": "a fox", "image": base64.b64encode(PNG).decode()}
GONE = "wan2.2-14b-i2v"


def broker(tmp_path, catalog=FIX / "catalog.yaml", start=True):
    driver = FakeDriver({"llama-8b"})
    b = Broker(make_settings(tmp_path, catalog), env={}, driver=driver, backends=FakeBackends(driver))
    if start:
        b.start()
    return b


def without(tmp_path, key):
    data = yaml.safe_load((FIX / "catalog.yaml").read_text())
    del data["models"][key]
    path = tmp_path / "pruned.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def kinds(b, kind):
    return [e["job_id"] for e in b.store.events(0, 1000) if e["kind"] == kind]


def staged(b):
    return sorted(p.name for p in b.staging.dir.iterdir()) if b.staging.dir.exists() else []


@pytest.fixture
def old(tmp_path):
    b = broker(tmp_path, start=False)
    yield b
    b.stop()


def test_a_requeued_job_whose_model_left_the_catalog_fails_once_and_the_next_runs(tmp_path, old):
    gone = old.submit(IMAGE_JOB, "t")[0]
    nxt = old.submit({"model": "sdxl-base", "prompt": "next"}, "t")[0]
    assert staged(old) == [f"{gone}-image.png"]
    pruned = without(tmp_path, GONE)
    new = broker(tmp_path, pruned)
    try:
        j = new.store.job(gone)
        assert j["state"] == JobState.FAILED and j["error"] == MODEL_GONE.format(key=GONE)
        assert staged(new) == [] and done(new, nxt)["state"] == JobState.DONE
        assert kinds(new, Event.JOB_REQUEUED) == [nxt]
    finally:
        assert wait_idle(new)
        new.stop()
    again = broker(tmp_path, pruned)   # a second restart: nothing left to re-queue
    try:
        assert kinds(again, Event.JOB_REQUEUED) == [nxt]
        assert again.store.job(gone)["error"] == MODEL_GONE.format(key=GONE)
    finally:
        again.stop()


def test_the_gpu_thread_fails_a_job_whose_model_is_unknown_instead_of_raising(tmp_path):
    """Defence in depth: whatever puts it in line, a job on an unknown model ends failed."""
    b = broker(tmp_path, start=False)
    jid = b.store.create_job("t", GONE, {})
    b.store.update_job(jid, resolved="not-a-model", state=JobState.QUEUED)
    b.staging.dir.mkdir(parents=True, exist_ok=True)
    (b.staging.dir / f"{jid}-image.png").write_bytes(PNG)
    b.scheduler.run(jid)
    j = b.store.job(jid)
    assert j["state"] == JobState.FAILED and j["error"] == MODEL_GONE.format(key="not-a-model")
    assert staged(b) == []
    b.stop()


def test_a_requeued_session_fails_as_expired_and_never_holds_the_gpu(tmp_path, old):
    jid = old.store.create_job("t", "sdxl-base", {SESSION_KEY: True})
    old.store.update_job(jid, resolved="sdxl-base", state=JobState.QUEUED)
    nxt = old.submit({"model": "sdxl-base", "prompt": "next"}, "t")[0]
    new = broker(tmp_path)
    try:
        j = new.store.job(jid)
        assert j["state"] == JobState.FAILED and j["error"] == SESSION_EXPIRED
        assert done(new, nxt)["state"] == JobState.DONE and kinds(new, Event.JOB_REQUEUED) == [nxt]
        assert new.store.job(jid)["state"] == JobState.FAILED   # still failed: the session never opened
    finally:
        assert wait_idle(new)
        new.stop()


def test_a_job_picked_as_a_quiesce_lands_stays_at_the_front(tmp_path):
    b = broker(tmp_path, start=False)
    jids = [b.submit({"model": "sdxl-base", "prompt": f"p{i}"}, "t")[0] for i in range(3)]
    real_pick, took = b.scheduler.line.pick, threading.Event()

    def pick(blocked, timeout):   # the quiesce lands while the GPU thread picks
        w = real_pick(blocked, timeout)
        if not took.is_set():
            b.scheduler.paused.set()
            took.set()
        return w

    b.scheduler.line.pick = pick
    stop = threading.Event()
    t = threading.Thread(target=b.scheduler.loop, args=(stop,), daemon=True)
    t.start()
    try:
        assert took.wait(WAIT_S)
        time.sleep(0.1)                                       # the job was never taken
        assert b.scheduler.line.order() == b.scheduler.snapshot()[0] == jids
        b.scheduler.paused.clear()                            # resume
        assert [done(b, j)["state"] for j in jids] == [JobState.DONE] * 3
        assert kinds(b, "job." + JobState.DONE) == jids
    finally:
        stop.set()
        t.join(WAIT_S)
        b.stop()
