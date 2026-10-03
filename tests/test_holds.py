"""GpuHold: the held job's clean runs off the GPU thread, one at a time, `retry_s` apart, and
clears only the hold it was started for; a clear wakes the waiting GPU thread at once."""
import threading
import time

import pytest

from gpu_broker.holds import UNKNOWN, GpuHold, orphan_recipe
from gpu_broker.store import Store


class Clean:
    def __init__(self):
        self.calls, self.fail, self.gate = [], True, threading.Event()
        self.gate.set()

    def __call__(self, recipe, jid):
        self.calls.append((recipe, jid))
        self.gate.wait(5)
        if self.fail:
            raise RuntimeError("still running")


def make(tmp_path, retry_s=60.0):
    now = [100.0]
    clean = Clean()
    hold = GpuHold(Store(str(tmp_path / "b.db")), clean, retry_s, clock=lambda: now[0])
    return hold, clean, now


def settle(hold):
    w = hold._worker
    if w is not None:
        w.join(5)


def test_a_hold_set_by_a_failed_run_waits_retry_s_before_its_first_clean(tmp_path):
    hold, clean, now = make(tmp_path)
    hold.set("j1", "sharp", "may still run")
    assert hold.held() and clean.calls == []
    now[0] += 60
    assert hold.held()
    settle(hold)
    assert clean.calls == [("sharp", "j1")] and hold.get() is not None   # failed: still held
    assert hold.held()
    settle(hold)
    assert len(clean.calls) == 1                                          # not again at once


def test_at_most_one_clean_is_in_flight(tmp_path):
    hold, clean, _ = make(tmp_path)
    clean.gate.clear()
    hold.set("j1", "sharp", "x", retry_now=True)
    for _ in range(5):
        assert hold.held()
    clean.gate.set()
    settle(hold)
    assert len(clean.calls) == 1


def test_a_successful_clean_clears_its_own_hold_and_wakes_the_gpu_thread(tmp_path):
    hold, clean, _ = make(tmp_path)
    clean.fail = False
    hold.set("j1", "sharp", "x", retry_now=True)
    woke = threading.Thread(target=hold.wait)
    hold.held()
    woke.start()
    settle(hold)
    woke.join(2)
    assert not woke.is_alive() and hold.get() is None and not hold.held()
    assert hold.store.events(-1, 1)[0]["data"] == {"by": "exec-clean"}


def test_a_clean_never_clears_a_hold_set_for_another_job_meanwhile(tmp_path):
    hold, clean, _ = make(tmp_path)
    clean.fail, clean.gate = False, threading.Event()
    hold.set("j1", "sharp", "x", retry_now=True)
    hold.held()
    hold.set("j2", "sharp", "y")      # a later job's run held it again while j1's clean ran
    clean.gate.set()
    settle(hold)
    assert hold.get()["job"] == "j2"


def test_an_operator_clear_wakes_the_waiting_gpu_thread_at_once(tmp_path):
    hold, _, _ = make(tmp_path)
    hold.set("j1", "sharp", "x")
    t0 = time.monotonic()
    waiter = threading.Thread(target=hold.wait)
    waiter.start()
    time.sleep(0.05)
    assert hold.clear("operator")["job"] == "j1"
    waiter.join(2)
    assert not waiter.is_alive() and time.monotonic() - t0 < 1



MODELS = {"sharp": {"runner": "exec", "exec": {"recipe": "sharp"}}, "llm": {"runner": "llm_unit"}}


def row(**kw):
    return {"id": "ab12cd34ef56", "requested": "gone", "resolved": "gone", "state": "running",
            "payload": {}, "exec_recipe": None, "direct": 0} | kw


@pytest.mark.parametrize(("job", "want"), [
    (row(exec_recipe="views"), "views"),                         # recorded: cleaned even with its model gone
    (row(exec_recipe="views", direct=1), None),                  # a direct chat is never exec
    (row(exec_recipe=""), None),                                 # NOT_EXEC, model gone
    (row(exec_recipe="", resolved="llm"), None),
    (row(exec_recipe="", resolved="sharp"), "sharp"),            # the catalog's exec beats NOT_EXEC
    (row(exec_recipe="", requested="sharp"), "sharp"),
    (row(resolved="sharp"), "sharp"),                            # NULL (unknown writer): the catalog decides
    (row(resolved="llm"), None),
    (row(requested="sharp"), "sharp"),                           # resolved name gone, requested one is exec
    (row(exec_recipe="../x; rm"), UNKNOWN),                      # invalid, model gone: heuristics
    (row(exec_recipe="views\n"), UNKNOWN),                       # the whole value must match
    (row(exec_recipe="../x", resolved="sharp"), "sharp"),        # invalid: the catalog decides
    (row(exec_recipe="../x", resolved="llm"), None),
    (row(exec_recipe="../x", payload={"messages": []}), None),
    (row(), UNKNOWN),                                            # NULL, model gone, nothing rules exec out
    (row(direct=1), None),
    (row(payload={"messages": []}), None),
    (row(payload={"kind": "llm", "prompt": "x"}), None),
    (row(payload=None), UNKNOWN),
])
def test_what_an_orphan_may_still_be_running(job, want):
    assert orphan_recipe(job, MODELS) == want


def test_a_known_recipe_is_cleaned_at_once(tmp_path):
    hold, clean, _ = make(tmp_path)
    hold.orphan(row(exec_recipe="views"), MODELS)
    assert hold.get()["recipe"] == "views" and hold.held()   # due at once
    settle(hold)
    assert clean.calls == [("views", "ab12cd34ef56")]


def test_an_unknown_recipe_is_held_until_an_operator_clears_it(tmp_path):
    hold, clean, _ = make(tmp_path)
    hold.orphan(row(), MODELS)
    held = hold.get()
    assert held["recipe"] == UNKNOWN and "gone" in held["reason"] and "gpu-held/clear" in held["reason"]
    hold.clock = lambda: 1e9               # long past any retry
    assert hold.held()
    settle(hold)
    assert clean.calls == []               # never cleaned automatically
    assert hold.clear("operator") is not None and not hold.held()


def test_a_job_that_cannot_have_been_exec_is_not_held(tmp_path):
    hold, _, _ = make(tmp_path)
    hold.orphan(row(direct=1), MODELS)
    assert hold.get() is None
