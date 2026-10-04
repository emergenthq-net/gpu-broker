"""A restart re-queues the jobs the previous process had queued but never started, in their
original order, with their staged input files; whatever had started is still failed as orphaned.
(A quiesce pauses the GPU thread, so before this every job queued at a deploy was dropped.)"""
from __future__ import annotations

import base64

import pytest

from gpu_broker.broker import Broker
from gpu_broker.constants import Event, JobState
from gpu_broker.store import ORPHANED
from tests.helpers import FakeBackends, FakeDriver, done, leftover_staged, make_settings, wait_idle

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
IMAGE_JOB = {"model": "wan-i2v", "prompt": "a fox", "image": base64.b64encode(PNG).decode()}


def broker(tmp_path, start=True):
    driver = FakeDriver({"llama-8b"})
    b = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))
    if start:
        b.start()
    return b


@pytest.fixture
def old(tmp_path):
    """The previous process: its GPU thread never ran (quiesced, then stopped), so what it
    accepted is persisted as `queued`, exactly as on a deploy."""
    b = broker(tmp_path, start=False)
    yield b
    b.stop()


def restart(tmp_path):
    """A new Broker on the same database and staging folder."""
    return broker(tmp_path)


def kinds(b, kind):
    return [e["job_id"] for e in b.store.events(0, 1000) if e["kind"] == kind]


def test_three_queued_jobs_all_complete_after_a_restart_in_their_order(tmp_path, old):
    jids = [old.submit({"model": "sdxl-base", "prompt": f"p{i}"}, "t")[0] for i in range(3)]
    old.scheduler.paused.set()                                      # the quiesce
    assert [old.store.job(j)["state"] for j in jids] == [JobState.QUEUED] * 3
    new = restart(tmp_path)
    try:
        assert [done(new, j)["state"] for j in jids] == [JobState.DONE] * 3
        assert kinds(new, Event.JOB_REQUEUED) == jids
        assert kinds(new, "job." + JobState.DONE) == jids              # in their original order
        texts = [{n["inputs"].get("text") for n in g.values()} for g in new.backends.graphs]
        assert [next(p for p in ("p0", "p1", "p2") if p in t) for t in texts] == ["p0", "p1", "p2"]
    finally:
        assert wait_idle(new)
        new.stop()


def test_started_jobs_still_fail_as_orphans_and_queued_ones_go_first(tmp_path, old):
    queued = old.submit({"model": "sdxl-base", "prompt": "q"}, "t")[0]
    started = {s: old.store.create_job("t", "llama-8b", {}) for s in (JobState.RECEIVED, JobState.SWITCHING, JobState.RUNNING)}
    for state, jid in started.items():
        old.store.update_job(jid, state=state)
    new = restart(tmp_path)
    late = new.submit({"model": "sdxl-base", "prompt": "late"}, "t")[0]
    try:
        assert all(new.store.job(j)["state"] == JobState.FAILED and new.store.job(j)["error"] == ORPHANED
                   for j in started.values())
        assert done(new, queued)["state"] == done(new, late)["state"] == JobState.DONE
        assert kinds(new, "job." + JobState.DONE) == [queued, late]     # re-queued ahead of new work
        assert kinds(new, Event.JOB_REQUEUED) == [queued]
    finally:
        assert wait_idle(new)
        new.stop()


def test_a_requeued_job_keeps_its_staged_image(tmp_path, old):
    jid = old.submit(IMAGE_JOB, "t")[0]
    new = restart(tmp_path)
    try:
        assert done(new, jid)["state"] == JobState.DONE
        assert new.backends.uploads == [(f"broker-{jid}-image.png", PNG, "png")]
        assert leftover_staged(new) == []
    finally:
        new.stop()


def test_a_requeued_job_whose_image_is_gone_fails_cleanly_and_the_queue_moves_on(tmp_path, old):
    gone = old.submit(IMAGE_JOB, "t")[0]
    nxt = old.submit({"model": "sdxl-base", "prompt": "next"}, "t")[0]
    old.staging.discard(gone)
    new = restart(tmp_path)
    try:
        j = done(new, gone)
        assert j["state"] == JobState.FAILED and "input files missing" in j["error"]
        assert new.backends.uploads == [] and done(new, nxt)["state"] == JobState.DONE
    finally:
        assert wait_idle(new)
        new.stop()


def test_a_failed_orphans_files_are_cleared_but_a_requeued_jobs_are_kept(tmp_path, old):
    keep = old.submit(IMAGE_JOB, "t")[0]
    lost = old.submit(IMAGE_JOB, "t")[0]
    old.store.update_job(lost, state=JobState.RUNNING)
    new = broker(tmp_path, start=False)
    lost_rows, waiting = new.store.fail_orphans()
    new.staging.clear(keep=frozenset(waiting))
    assert waiting == [keep] and [r["id"] for r in lost_rows] == [lost]
    assert sorted(p.name for p in new.staging.dir.iterdir()) == [f"{keep}-image.png"]
    new.stop()


def test_a_direct_chat_is_never_requeued_whatever_state_it_was_left_in(tmp_path, old):
    """Its caller held the connection and is gone; replaying it would answer nobody."""
    jid = old.store.create_job("t", "llama-8b", {})
    old.store.event(Event.JOB_DIRECT, jid, model="llama-8b")
    old.store.update_job(jid, state=JobState.QUEUED)
    lost, waiting = old.store.fail_orphans()
    assert waiting == [] and [r["id"] for r in lost] == [jid] and old.store.job(jid)["state"] == JobState.FAILED


def test_orphans_are_failed_in_chunks_of_ids(tmp_path, old, monkeypatch):
    monkeypatch.setattr("gpu_broker.store.SQL_IN_MAX", 2)
    jids = [old.store.create_job("t", "llama-8b", {}) for _ in range(5)]
    for jid in jids:
        old.store.update_job(jid, state=JobState.RUNNING)
    lost, waiting = old.store.fail_orphans()
    assert waiting == [] and len(lost) == 5
    assert all(old.store.job(j)["error"] == ORPHANED for j in jids)
