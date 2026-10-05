"""replay.sim: the scheduler's rules in simulated time (pool slots, drain before a switch, the
GPU thread, idle restore, closed-loop arrivals) under a queue policy."""
from gpu_broker.policy import Fifo
from gpu_broker.replay.sim import Sim
from gpu_broker.replay.trace import Job, Trace
from tests.replaylog import CAT

SWITCH = dict.fromkeys(CAT.models, 20.0)   # every switch costs 20 s
IDLE = CAT.defaults["idle_restore_s"]


def sim(jobs, policy=None):
    t = Trace(jobs, dict(SWITCH), jobs[0].arrival if jobs else 0, 0)
    return Sim(t, CAT, Fifo() if policy is None else policy).run()


def job(jid, at, model, run_s, after=None, gap=0.0, requester="a", interactive=False):
    return Job(jid, at, requester, model, run_s, after=after, gap=gap, interactive=interactive,
               pooled=CAT.models[model]["runner"] == "llm_unit")


class Skipping(Fifo):
    """FIFO order, but a blocked job does not hold up the ones behind it."""
    stall_on_blocked = False


def test_background_calls_get_slots_minus_the_reserved_ones():
    r = sim([job(f"j{i}", 0, "llama-8b", 10) for i in range(5)])   # slots 5, reserved_interactive 1
    assert [r.waits[f"j{i}"] for i in range(5)] == [0, 0, 0, 0, 10]
    assert r.residency == [] and r.end == 20


def test_a_switch_drains_the_pool_then_pays_the_switch_and_runs_on_the_gpu_thread():
    r = sim([job("llm", 0, "llama-8b", 50), job("img", 1, "sdxl-base", 30), job("back", 2, "llama-8b", 5)])
    assert r.waits == {"llm": 0, "img": 49, "back": 48 + 20 + 30}   # drain to 50, +20 switch, +30 run
    assert r.residency == [(50, "sdxl-base"), (100, "llama-8b")]


def test_fifo_holds_the_line_behind_a_full_pool_and_a_skipping_policy_does_not():
    jobs = [job(f"b{i}", 0, "llama-8b", 100) for i in range(5)] + [job("chat", 1, "llama-8b", 5, requester="op", interactive=True)]
    assert sim(jobs).waits["chat"] == 99    # behind b4, which waits for one of the four background slots
    assert sim(jobs, Skipping()).waits["chat"] == 0   # takes the slot kept for interactive calls


def test_a_job_that_needs_the_pool_drained_is_taken_so_new_calls_cannot_starve_it():
    jobs = [job("llm", 0, "llama-8b", 100), job("img", 1, "sdxl-base", 10), job("chat", 2, "llama-8b", 5)]
    for policy in (None, Skipping()):
        r = sim(jobs, policy)
        assert r.waits["img"] == 99 and r.waits["chat"] == 98 + 20 + 10   # after the drain, the switch, the image


def test_the_default_model_comes_back_after_idle_restore_s():
    r = sim([job("img", 0, "sdxl-base", 10)])
    assert r.residency == [(0, "sdxl-base"), (30 + IDLE, "llama-8b")]
    assert r.end == 30 + IDLE + 20


def test_a_job_for_the_default_model_brings_it_back_without_waiting_for_the_idle_restore():
    r = sim([job("img", 0, "sdxl-base", 10), job("llm", 60, "llama-8b", 5)])
    assert r.residency == [(0, "sdxl-base"), (60, "llama-8b")] and r.waits["llm"] == 0


def test_a_non_resident_llm_switches_once_and_its_calls_share_its_slots():
    r = sim([job("q1", 0, "qwen-coder-32b", 10), job("q2", 0, "qwen-coder-32b", 10)])   # one slot
    assert r.waits == {"q1": 0, "q2": 30} and r.residency == [(0, "qwen-coder-32b"), (40 + IDLE, "llama-8b")]


def test_a_chained_job_arrives_gap_after_its_cause_ends_in_the_simulation():
    jobs = [job("img", 0, "sdxl-base", 100), job("a", 1, "llama-8b", 10),
            job("b", 13, "llama-8b", 1, after="a", gap=2)]   # logged at 13, but `a` ends later here
    r = sim(jobs)
    a_end = 20 + 100 + 20 + 10   # image switch + run, switch back, run
    assert r.waits["a"] == 119 and r.waits["b"] == 0 and r.end == a_end + 2 + 1


def test_a_gpu_job_releases_its_chained_job_too():
    r = sim([job("i1", 0, "sdxl-base", 10), job("i2", 99, "sdxl-base", 10, after="i1", gap=5)])
    assert r.waits["i2"] == 0 and r.residency[0] == (0, "sdxl-base") and len(r.residency) == 2


def test_an_empty_trace_ends_at_once():
    r = sim([])
    assert (r.waits, r.residency, r.end) == ({}, [], 0)


def test_fair_serves_a_light_requester_before_a_heavy_ones_backlog():
    from gpu_broker.policy import Fair
    jobs = [job(f"h{i}", 0, "sdxl-base", 10, requester="heavy") for i in range(5)] + [job("l", 1, "sdxl-base", 10, requester="light")]
    assert sim(jobs).waits["l"] == 20 + 5 * 10 - 1        # FIFO: behind the switch and heavy's five images
    assert sim(jobs, Fair(lambda j: 10.0)).waits["l"] == 20 + 10 - 1   # fair: right after heavy's first
