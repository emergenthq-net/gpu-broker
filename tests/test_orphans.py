"""Restart with jobs in flight: only a job that may have been an exec run holds the GPU. The
recipe lives in its own jobs column, written by submit for exec models; it never enters the
caller's payload, so it is never forwarded upstream."""
import sqlite3
import time

import pytest
import yaml

from gpu_broker.broker import Broker
from gpu_broker.constants import JobState
from gpu_broker.store import Store
from tests.helpers import FakeBackends, FakeDriver, make_settings

MSGS = [{"role": "user", "content": "hi"}]


def restart(tmp_path, driver, catalog=None):
    b = Broker(make_settings(tmp_path, catalog) if catalog else make_settings(tmp_path), env={}, driver=driver,
               backends=FakeBackends(driver))
    b.start()
    return b


def recipe_column(store, jid):
    return store._all("SELECT exec_recipe FROM jobs WHERE id=?", (jid,))[0]["exec_recipe"]


def test_a_direct_chat_naming_exec_recipe_records_nothing_and_holds_nothing_on_restart(client, broker, tmp_path):
    r = client.post("/v1/chat/completions", json={"model": "llama-8b", "messages": MSGS, "exec_recipe": "sharp"})
    assert r.json()["x_broker"]["direct"] is True
    jid = broker.store.jobs(1)[0]["id"]
    assert recipe_column(broker.store, jid) == ""   # NOT_EXEC, whatever the body said
    broker.store.update_job(jid, state=JobState.RUNNING)   # as if the process died mid-chat
    again = restart(tmp_path, FakeDriver({"llama-8b"}))
    try:
        assert again.scheduler.hold.get() is None and again.store.job(jid)["state"] == JobState.FAILED
    finally:
        again.stop()


def test_a_queued_llm_jobs_upstream_body_has_no_exec_recipe(client, broker):
    jid = client.post("/v1/jobs", json={"model": "llama-8b", "messages": MSGS}).json()["id"]
    assert broker.wait(jid, 5)["state"] == JobState.DONE
    assert broker.backends.sent and all("exec_recipe" not in p for p in broker.backends.sent)


def test_an_llm_model_pruned_while_its_job_ran_holds_nothing_on_restart(tmp_path):
    driver = FakeDriver({"llama-8b"})
    first = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))
    jid, _ = first.submit({"model": "qwen-coder-32b", "prompt": "hi"}, "t")
    first.store.update_job(jid, state=JobState.RUNNING, resolved="qwen-coder-32b")
    pruned = yaml.safe_load((tmp_path / "catalog.yaml").read_text())
    del pruned["models"]["qwen-coder-32b"]
    (tmp_path / "pruned.yaml").write_text(yaml.safe_dump(pruned))
    again = restart(tmp_path, driver, tmp_path / "pruned.yaml")
    try:
        assert again.scheduler.hold.get() is None and again.store.job(jid)["state"] == JobState.FAILED
    finally:
        again.stop()


def test_an_old_database_gains_the_column_with_null_for_its_rows(tmp_path):
    db = sqlite3.connect(tmp_path / "b.db")
    db.executescript("""CREATE TABLE jobs (id TEXT PRIMARY KEY, created REAL, updated REAL, requester TEXT,
      requested TEXT, resolved TEXT, substitution TEXT, state TEXT, payload TEXT, result TEXT, error TEXT, download TEXT);
      CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, job_id TEXT, kind TEXT, data TEXT);
      INSERT INTO jobs(id, created, requested, resolved, state, payload) VALUES ('ab12cd34ef56', 1, 'x', 'x', 'running', '{}');
      INSERT INTO jobs(id, created, requested, resolved, state, payload) VALUES ('cd34ef56ab12', 1, 'y', 'y', 'running', '{}');
      INSERT INTO events(ts, job_id, kind, data) VALUES (1, 'cd34ef56ab12', 'job.direct', '{}');""")
    db.close()
    store = Store(str(tmp_path / "b.db"))
    old = {o["id"]: o for o in store.fail_orphans()[0]}
    assert all(o["exec_recipe"] is None for o in old.values())
    assert (old["ab12cd34ef56"]["direct"], old["cd34ef56ab12"]["direct"]) == (0, 1)


def test_every_job_this_code_creates_says_whether_it_is_exec(client, broker):
    queued = client.post("/v1/jobs", json={"model": "llama-8b", "prompt": "hi"}).json()["id"]
    client.post("/v1/chat/completions", json={"model": "llama-8b", "messages": MSGS})
    direct = broker.store.jobs(1)[0]["id"]
    assert broker.wait(queued, 5)["state"] == JobState.DONE
    assert (recipe_column(broker.store, queued), recipe_column(broker.store, direct)) == ("", "")


def null_exec_row_left_running(tmp_path, created):
    """A running sharp job written with exec_recipe NULL, as a24325a does after a rollback."""
    store = Store(str(tmp_path / "b.db"))   # the migration ran: the column exists
    jid = store.create_job("t", "sharp", {"model": "sharp"})
    store._exec("UPDATE jobs SET exec_recipe=NULL, resolved='sharp', state=?, created=? WHERE id=?",
                (JobState.RUNNING, created, jid))
    return jid


@pytest.mark.parametrize("created", [time.time() + 3600, time.time() - 3600],
                         ids=["after-the-migration", "clock-stepped-back"])
def test_a_null_row_on_an_exec_model_is_held_after_a_rollback(tmp_path, created):
    jid = null_exec_row_left_running(tmp_path, created)
    driver = FakeDriver({"llama-8b"})
    driver.clean_error = "still running"   # keep the hold up so it can be read
    again = restart(tmp_path, driver)
    try:
        held = again.scheduler.hold.get()
        assert held is not None and (held["job"], held["recipe"]) == (jid, "sharp")
    finally:
        again.scheduler.hold.clear("test")
        again.stop()
