"""policy.Fair: interactive (or aged) first, then re-queued jobs in their original order, then the
requester with the least expected GPU time used; monotonic virtual time, no banked credit,
bounded bookkeeping. gpu_cost: median run time per model."""
import random
from dataclasses import dataclass

import pytest

from gpu_broker.policy import DEFAULT_RUN_S, IDLE_MAX, Fair, Fifo, gpu_cost, make


@dataclass(eq=False)
class W:
    id: str
    requester: str
    model: str = "m"
    interactive: bool = False
    pooled: bool = False
    requeued: bool = False


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def fair(costs=None, max_wait_s=1800.0, clock=None):
    return Fair(lambda w: (costs or {}).get(w.id, 10.0), max_wait_s, clock or Clock())


def order(p):
    return [w.id for w in p.candidates()]


def start(p, jid):
    p.remove(next(w for w in p.candidates() if w.id == jid))


def drain(p):
    ids = []
    while len(p):
        w = next(p.candidates())
        ids.append(w.id)
        p.remove(w)
    return ids


def test_interactive_jobs_come_first_whatever_was_used():
    p = fair()
    for w in (W("b1", "bot"), W("i1", "op", interactive=True), W("b2", "bot")):
        p.add(w)
    assert order(p) == ["i1", "b1", "b2"]


def test_the_requester_that_used_least_goes_first_and_each_keeps_its_own_order():
    p = fair({"h1": 100.0})
    for i in range(1, 4):
        p.add(W(f"h{i}", "heavy"))
    start(p, "h1")                          # heavy ran a 100 s job
    p.add(W("l1", "light"))
    p.add(W("l2", "light"))
    assert order(p) == ["l1", "l2", "h2", "h3"]


def test_one_long_job_is_charged_like_many_short_ones():
    p = fair({"long": 90.0})
    p.add(W("long", "media"))
    p.add(W("next", "media"))
    for i in range(10):
        p.add(W(f"s{i}", "llm"))
    assert drain(p) == ["long", *(f"s{i}" for i in range(9)), "next", "s9"]   # nine 10 s calls = one 90 s job


def test_virtual_time_never_goes_back_so_a_newcomer_cannot_take_credit():
    """The idle-credit scenario: an interactive job of a requester far behind starts after a
    heavy background requester's; starting it must not pull virtual time back to its level."""
    p, seen = fair({f"h{i}": 100.0 for i in range(5)}), []
    p.add(W("i", "early", interactive=True))   # joins at 0 and waits (say, blocked on a slot)
    for i in range(5):
        p.add(W(f"h{i}", "heavy"))
    for jid in ("h0", "h1", "h2"):              # heavy runs meanwhile
        start(p, jid)
        seen.append(p._v)
    p.add(W("n", "new"))
    assert p._used["new"] == 0                  # virtual time is the lowest tag waiting (early's), not heavy's
    for jid in ("i", "h3"):                     # the early job starts at its old level
        start(p, jid)
        seen.append(p._v)
    assert seen == sorted(seen)


def test_virtual_time_is_monotonic_whatever_starts_and_newcomers_join_at_it():
    rng = random.Random(7)  # noqa: S311 — a reproducible sequence, not a secret
    p = Fair(lambda w: (1.0, 10.0, 300.0)[int(w.id[1:]) % 3], 1e9, Clock())
    last, n = 0.0, 0
    for _ in range(3000):
        if len(p) and rng.random() < 0.5:
            start(p, rng.choice(order(p)))           # any job may start (skips, classes)
            assert p._v >= last
            last = p._v
        else:
            r = f"r{rng.randrange(40)}"
            fresh = r not in p._q
            p.add(W(f"j{(n := n + 1)}", r, interactive=rng.random() < 0.2))
            assert not fresh or p._used[r] >= p._v


def test_an_idle_requester_banks_no_credit():
    p = fair()
    for i in range(5):
        p.add(w := W(f"h{i}", "heavy"))
        p.remove(w)                             # heavy used 50 s while light was away
    p.add(W("h5", "heavy"))
    p.add(W("h6", "heavy"))
    p.add(W("l1", "light"))
    p.add(W("l2", "light"))
    assert order(p) == ["l1", "h5", "l2", "h6"]   # light leads by at most heavy's job in service, then they alternate


def test_bookkeeping_stays_bounded_with_many_one_off_requesters():
    p = fair()
    for i in range(10_000):
        p.add(w := W(f"j{i}", f"r{i}"))
        p.remove(w)
    assert len(p._used) <= IDLE_MAX + 1 and len(p._idle) <= 3 * (IDLE_MAX + 1) and not p._added


def test_a_background_job_ages_into_the_interactive_class():
    clock = Clock()
    p = fair(max_wait_s=100, clock=clock)
    p.add(W("bg", "bot"))
    served = []
    for i in range(50):                         # a steady stream of interactive work, one every 10 s
        p.add(W(f"i{i}", "op", interactive=True))
        w = next(p.candidates())
        served.append(w.id)
        p.remove(w)
        clock.t += 10
    assert "bg" in served and served.index("bg") <= 11   # promoted once it had waited 100 s


def test_requeued_jobs_come_first_in_their_class_in_their_original_order():
    p = fair({"x": 0.0})
    p.add(W("x", "b"))
    start(p, "x")
    for w in (W("r1", "b", requeued=True), W("r2", "a", requeued=True), W("new", "c"),
              W("ri", "a", interactive=True, requeued=True), W("ni", "c", interactive=True)):
        p.add(w)
    assert order(p) == ["ri", "ni", "r1", "r2", "new"]


def test_the_order_is_cached_until_a_change_or_an_aging_deadline():
    clock = Clock()
    p = fair(max_wait_s=100, clock=clock)
    p.add(W("b", "bot"))
    p.add(W("i", "op", interactive=True))
    first = p._order if order(p) else None
    assert order(p) == ["i", "b"] and p._order is first      # not re-sorted
    clock.t = 100
    assert order(p) == ["b", "i"] and p._order is not first  # b aged: re-sorted, now level with i and earlier


def test_fifo_is_arrival_order_and_stalls():
    p = Fifo()
    for w in (W("b1", "bot"), W("i1", "op", interactive=True)):
        p.add(w)
    assert order(p) == ["b1", "i1"] and Fifo.stall_on_blocked and not Fair.stall_on_blocked


def test_make_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="fair"):
        make("lifo", lambda w: 1.0)


def test_gpu_cost_is_the_median_run_and_an_llm_call_costs_a_slots_share():
    models = {"llm": {"runner": "llm_unit", "slots": 4}, "img": {"runner": "comfy"}}
    cost = gpu_cost(models, {"llm": [100.0, 200.0, 400.0, 0.0], "img": [30.0]})
    assert cost(W("a", "r", "llm")) == 50.0 and cost(W("b", "r", "img")) == 30.0
    assert cost(W("c", "r", "new")) == DEFAULT_RUN_S


def test_new_costs_re_sort_the_cached_order():
    costs = {"a1": 10.0, "b1": 10.0}
    p = Fair(lambda w: costs.get(w.id, 10.0), 1800.0, Clock())
    for w in (W("a1", "a"), W("a2", "a"), W("b1", "b"), W("b2", "b")):
        p.add(w)
    assert order(p) == ["a1", "b1", "a2", "b2"]
    costs["a1"] = 100.0                      # a's first job turns out to be long
    assert order(p) == ["a1", "b1", "a2", "b2"]   # still the cached order...
    p.invalidate()
    assert order(p) == ["a1", "b1", "b2", "a2"]   # ...until the costs are swapped in
