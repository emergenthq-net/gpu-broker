"""replay.trace: an event log → the jobs that queued, their run times, closed-loop chains and
per-model switch costs."""
import pytest

from gpu_broker.replay import trace as tr
from tests.replaylog import CAT, Log


def build(log):
    return tr.build(log.sorted(), CAT)


def test_a_job_has_its_arrival_requester_model_run_and_observed_wait():
    t = build(Log().job("j1", 100, "llama", 30, requester="r", wait_s=4))
    (j,) = t.jobs
    assert (j.id, j.arrival, j.requester, j.model, j.run_s, j.observed_wait) == ("j1", 100, "r", "llama-8b", 30, 4)
    assert (t.start, t.end) == (100, 134)


def test_the_substitution_names_the_model_and_variants_map_to_their_parent():
    log = Log().job("j1", 0, "nope", 5).ev(0, "job.substituted", "j1", requested="nope", resolved="sdxl-base")
    log.job("j2", 1, "llama-8b-precise", 5)
    assert [j.model for j in build(log).jobs] == ["sdxl-base", "llama-8b"]


def test_direct_lost_unknown_and_pre_log_jobs_are_counted_not_replayed():
    log = Log().ev(0, "job.received", "d", requester="r", requested="llama").ev(0, "job.direct", "d")
    log.ev(1, "job.received", "lost", requester="r", requested="llama").ev(1, "job.queued", "lost")
    log.job("u", 2, "no-such-model", 5)
    log.ev(3, "job.running", "early").ev(4, "job.done", "early")   # received before the log starts
    log.ev(5, "job.received", "rej", requester="r", requested="llama").ev(5, "job.rejected", "rej")
    t = build(log)
    assert (t.jobs, t.direct, t.lost, t.unknown_model) == ([], 1, 1, 1)


def test_a_job_that_ended_without_running_replays_with_no_run_time():
    log = Log().ev(0, "job.received", "f", requester="r", requested="sdxl").ev(0, "job.queued", "f")
    log.ev(1, "job.switching", "f").ev(9, "job.failed", "f")
    (j,) = build(log).jobs
    assert (j.run_s, j.observed_wait) == (0.0, 1)


def test_switch_cost_is_the_median_real_switch_then_the_load_time_then_the_overall_median():
    log = Log().job("a", 0, "sdxl", 1, switch_s=10).job("b", 50, "sdxl", 1, switch_s=30)
    log.job("c", 100, "sdxl", 1, switch_s=20).job("d", 200, "flux-schnell", 1, switch_s=0.5)   # a state step, not a switch
    log.ev(300, "residency.ready", model="qwen-coder-32b", load_s=7.0)
    s = build(log).switch_s
    assert (s["sdxl-base"], s["qwen-coder-32b"], s["flux-schnell"], s["llama-8b"]) == (20, 7.0, 20, 20)
    assert set(s) == set(CAT.models)


def test_with_no_switch_measured_the_default_applies():
    assert build(Log().job("a", 0, "llama", 1)).switch_s["sdxl-base"] == tr.DEFAULT_SWITCH_S


def test_a_job_sent_soon_after_its_requesters_previous_end_is_chained_to_it():
    log = Log().job("a", 0, "llama", 10, requester="r").job("b", 10 + 2, "llama", 10, requester="r")
    log.job("c", 22 + tr.CHAIN_S + 1, "llama", 1, requester="r")   # too late: a fresh request
    log.job("x", 12.5, "llama", 1, requester="other")             # another requester never chains
    jobs = {j.id: j for j in build(log).jobs}
    assert (jobs["b"].after, jobs["b"].gap) == ("a", 2)
    assert jobs["c"].after is None and jobs["x"].after is None and jobs["a"].after is None


def test_one_end_releases_at_most_one_job():
    log = Log().job("a", 0, "llama", 10).job("b", 0, "llama", 9)
    log.job("c", 11, "llama", 1).job("d", 11.5, "llama", 1).job("e", 11.8, "llama", 1)
    jobs = {j.id: j for j in build(log).jobs}
    assert (jobs["c"].after, jobs["d"].after, jobs["e"].after) == ("a", "b", None)


def test_observed_residency_keeps_only_changes():
    log = Log().ev(1, "residency.resident", model="llama-8b").ev(2, "residency.resident", model="llama-8b")
    log.ev(3, "residency.resident", model="sdxl-base").job("a", 4, "llama", 1)
    assert build(log).residency == [(1, "llama-8b"), (3, "sdxl-base")]


def test_jobs_come_out_in_arrival_order_and_blank_lines_are_skipped():
    log = Log().job("late", 50, "llama", 1).job("early", 5, "llama", 1)
    lines = ["\n"] + [__import__("json").dumps(e) for e in log.sorted()]
    assert [j.id for j in tr.build(tr.read(lines), CAT).jobs] == ["early", "late"]


@pytest.mark.parametrize("name, key", [("llama", "llama-8b"), ("LLAMA-3.1-8B", "llama-8b"), ("missing", None)])
def test_model_of(name, key):
    assert tr.model_of(CAT, name) == key


def test_a_jobs_class_follows_the_catalogs_background_requesters():
    t = build(Log().job("a", 0, "llama-8b", 1, requester="batch-agent").job("b", 1, "llama-8b", 1, requester="op"))
    assert {j.id: j.interactive for j in t.jobs} == {"a": False, "b": True}
