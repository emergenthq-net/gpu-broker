"""line.choose: FIFO returns its head; fair skips a call with no slot, never lets a requester's
later job overtake its skipped one, and never lets a switch jump calls the resident model is
about to serve (priority inversion). Line.pick: one pool read per look, woken by a kick.
Costs: from the store, refreshed on its own thread."""
import threading
import time

from gpu_broker.constants import JobState
from gpu_broker.line import Costs, Line, Snap, Waiting, choose, waiting
from gpu_broker.policy import Fair, Fifo
from gpu_broker.store import Store


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def w(jid, requester="r", model="x", interactive=False, pooled=True):
    return Waiting(jid, requester, model, interactive, pooled)


FULL_BG = Snap("x", 4, (4, 5))     # x resident: background slots full, one interactive slot free
FULL = Snap("x", 5, (4, 5))        # every slot busy


def line_of(*items, policy="fair", clock=None):
    p = (Fair if policy == "fair" else Fifo)(lambda _: 1.0, 1800.0, clock or Clock())
    for it in items:
        p.add(it)
    return p


def picked(p, snap, evict_wait_s=600.0):
    c = choose(p, snap, evict_wait_s)
    return c and c.id


def test_fifo_returns_its_head_even_when_it_cannot_start():
    assert picked(line_of(w("a"), w("b", model="img", pooled=False), policy="fifo"), FULL) == "a"


def test_fair_skips_a_call_with_no_slot_for_its_class():
    assert picked(line_of(w("bg"), w("chat", "op", interactive=True)), FULL_BG) == "chat"
    assert picked(line_of(w("bg"), w("chat", "op", interactive=True)), FULL) is None


def test_a_switch_does_not_jump_a_call_waiting_for_a_slot_on_the_resident_model():
    """X resident, every slot busy, an interactive X call waiting: a background Y job behind it
    must not evict X; the X call starts when a slot frees, with no switch."""
    clock = Clock()
    p = line_of(w("x-call", "op", interactive=True), w("y-job", "bot", model="y", pooled=False), clock=clock)
    assert picked(p, FULL) is None                       # held, not switched
    assert picked(p, Snap("x", 4, (4, 5))) == "x-call"   # a slot freed
    p2 = line_of(w("x-call", "op", interactive=True), w("y-int", "op2", model="y", pooled=False, interactive=True))
    assert picked(p2, FULL) is None                      # same class: still held
    p3 = line_of(w("x-bg", "bot"), w("y-int", "op", model="y", pooled=False, interactive=True))
    assert picked(p3, FULL) == "y-int"                   # a higher class may evict
    p4 = line_of(w("x-call", "op", interactive=True), w("y-llm", "bot", model="y", interactive=True))
    assert picked(p4, FULL) is None                      # a call to another LLM evicts x just the same
    clock.t = 600
    assert picked(p, FULL) == "y-job"                    # the held call (and the job) waited too long


def test_the_eviction_bound_counts_either_side_waiting_too_long():
    clock = Clock()
    p = line_of(w("x-old", "op", interactive=True), clock=clock)
    clock.t = 600
    p.add(w("y-new", "bot", model="y", pooled=False, interactive=True))
    assert picked(p, FULL) == "y-new"                    # the call it would jump has waited too long already
    clock2 = Clock()
    q = line_of(w("y-old", "bot", model="y", pooled=False, interactive=True), clock=clock2)
    clock2.t = 600
    q.add(w("x-new", "op", interactive=True))
    assert picked(q, FULL) == "y-old"                    # the switch itself has waited too long


def test_a_requesters_later_job_never_overtakes_its_skipped_one():
    clock = Clock()
    p = Fair(lambda x: 0.0 if x.id == "r-call" else 1.0, 1800.0, clock)   # r-img and o-img both start at 1
    p.add(o0 := w("o0", "o", model="img", pooled=False))
    p.remove(o0)
    for it in (w("r-call", "r"), w("r-img", "r", model="img", pooled=False), w("o-img", "o", model="img", pooled=False)):
        p.add(it)
    clock.t = 600                                         # past evict_wait_s: a switch may go now
    assert [x.id for x in p.candidates()] == ["r-call", "r-img", "o-img"]
    assert picked(p, FULL) == "o-img"                     # not r-img: r's call is still waiting for a slot


def test_pick_reads_the_pool_once_per_look_and_waits_for_a_kick():
    line, reads = Line("fair", lambda _: 1.0), []
    line.put(w("a"))

    def snap():
        reads.append(1)
        return FULL if len(reads) < 3 else Snap("x", 0, (4, 5))
    assert line.pick(snap, 0.05) is None and len(reads) == 1   # one read; nothing changed, so no second
    threading.Timer(0.05, line.kick).start()
    t0 = time.monotonic()
    assert line.pick(snap, 5).id == "a" and time.monotonic() - t0 < 2


def test_a_change_while_the_pool_is_read_is_not_missed():
    line = Line("fair", lambda _: 1.0)
    line.put(w("a"))
    calls = []

    def snap():
        calls.append(1)
        if len(calls) == 1:
            line.kick()   # a slot frees right after this read
            return FULL
        return Snap("x", 0, (4, 5))
    t0 = time.monotonic()
    assert line.pick(snap, 5).id == "a" and time.monotonic() - t0 < 1


def test_a_picked_job_stays_until_taken():
    line = Line("fair", lambda _: 1.0)
    line.put(a := w("a"))
    assert line.pick(lambda: Snap(None, 0, (1, 1)), 0) is a and line.order() == ["a"]
    line.take(a)
    assert line.order() == [] and len(line) == 0


def test_waiting_and_snap_read_the_job_and_the_catalog():
    models = {"llm": {"runner": "llm_unit", "slots": 5, "reserved_interactive": 1}, "img": {"runner": "comfy"}}
    job = {"resolved": "llm", "requester": "op", "payload": {"interactive": True}}
    assert vars(waiting(models, "j", job)) == vars(Waiting("j", "op", "llm", True, True))
    assert not waiting(models, "j", {**job, "payload": {"session": True}}).pooled
    assert not waiting(models, "j", {**job, "resolved": "img"}).pooled
    assert waiting(models, "j", job, requeued=True).requeued
    assert vars(waiting(models, "gone", None)) == vars(Waiting("gone", "", "", False, False))
    assert Snap.of("llm", 2, models) == Snap("llm", 2, (4, 5))


def ran(store, seconds):
    jid = store.create_job("r", "img", {})
    store.update_job(jid, resolved="img", state=JobState.RUNNING)
    time.sleep(seconds)
    store.update_job(jid, state=JobState.DONE)


def test_costs_use_the_median_measured_run_and_refresh_on_their_own_thread(tmp_path):
    store = Store(str(tmp_path / "b.db"))
    costs = Costs(store, lambda: {"img": {"runner": "comfy"}}, 3600, 0.05)
    item = w("x", model="img")
    assert costs(item) == 60.0                       # nothing measured yet: the default
    ran(store, 0.2)
    stop = threading.Event()
    t = threading.Thread(target=costs.loop, args=(stop,), daemon=True)
    t.start()
    end = time.monotonic() + 5
    while costs(item) == 60.0 and time.monotonic() < end:
        time.sleep(0.01)
    assert 0.2 <= costs(item) < 1
    stop.set()
    t.join(5)
    assert not t.is_alive()


def test_a_failing_refresh_keeps_the_last_estimate_and_the_thread(tmp_path, monkeypatch):
    costs = Costs(Store(str(tmp_path / "b.db")), dict, 3600, 0.01)
    calls = []
    monkeypatch.setattr(costs, "refresh", lambda: calls.append(1) or (_ for _ in ()).throw(OSError("db gone")))
    stop = threading.Event()
    t = threading.Thread(target=costs.loop, args=(stop,), daemon=True)
    t.start()
    end = time.monotonic() + 5
    while len(calls) < 3 and time.monotonic() < end:
        time.sleep(0.01)
    stop.set()
    t.join(5)
    assert len(calls) >= 3 and costs(w("x")) == 60.0


def test_swapping_in_new_costs_reprices_the_line(tmp_path):
    store = Store(str(tmp_path / "b.db"))
    costs = Costs(store, lambda: {"img": {"runner": "comfy"}}, 3600, 60)
    line = Line("fair", costs)
    costs.changed = line.reprice
    seen = []
    line.policy.invalidate = lambda: seen.append("invalidated")
    costs.refresh()
    assert seen == ["invalidated"]
