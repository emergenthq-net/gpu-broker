"""replay.report and `gpu-broker replay`: waits per requester, churn, and the command line."""
import json

from gpu_broker import cli
from gpu_broker.replay import report as rp
from gpu_broker.replay.sim import Result
from gpu_broker.replay.trace import Job, Trace
from tests.replaylog import FIX, Log

SW = {"a": 10.0, "b": 5.0, "c": 1.0}


def test_churn_counts_switches_per_day_aba_within_the_window_and_switch_seconds():
    res = [(0, "a"), (100, "b"), (200, "a"), (5000, "b"), (5000 + rp.ABA_WINDOW_S, "a"), (9000, "c")]
    assert rp.churn(res, rp.DAY_S / 2, SW) == {"switches": 6, "per_day": 12.0, "aba_15m": 1, "switch_s": 41.0}
    assert rp.churn([(0, "a")], 60, SW)["per_day"] == 24.0   # a span under an hour counts as an hour


def test_waits_are_percentiles_of_the_group():
    assert rp.waits([3, 1, 2]) == {"n": 3, "p50": 2, "p90": 3, "p99": 3, "max": 3}
    assert rp.waits([])["p50"] is None


def test_the_busiest_requesters_are_their_own_group_and_the_rest_are_pooled():
    jobs = [Job(f"{r}{i}", i, r, "m", observed_wait=1) for r, n in (("big", 9), *((f"r{k}", 2) for k in range(rp.TOP_REQUESTERS)))
            for i in range(n)]
    trace = Trace(jobs, {}, 0, 3600)
    out = rp.report(trace, Result(waits={j.id: 5.0 for j in jobs}), "fifo")
    assert "big" in out["wait"] and rp.OTHER in out["wait"] and len(out["wait"]) == rp.TOP_REQUESTERS + 1
    assert out["observed_wait"]["big"]["p50"] == 1 and out["wait"]["big"]["p50"] == 5
    by_model = rp.report(trace, Result(waits={j.id: 5.0 for j in jobs}), "fifo", by=lambda j: j.model)
    assert list(by_model["wait"]) == ["m"]


def test_text_shows_every_group_and_both_residency_lines():
    out = rp.text({"policy": "fifo", "jobs": 2, "span_h": 1.0, "chained": 1, "lost": 0, "direct": 0, "unknown_model": 0,
                   "wait": {"x": rp.waits([1.0, 2.0])}, "per_h": {"x": 2.0}, "behind_others": {"x": rp.waits([0.0])}, "observed_wait": {},
                   "residency": rp.churn([], 3600, SW), "observed_residency": rp.churn([(0, "a")], 3600, SW)})
    assert "policy fifo: 2 queued jobs" in out and "x" in out and "   (-)" in out
    assert "residency: 0 switches" in out and "observed:  1 switches" in out


def log(tmp_path):
    return Log().job("j1", 0, "llama", 10, requester="r").job("j2", 1, "sdxl", 5, requester="s").write(tmp_path / "e.jsonl")


def test_the_replay_command_needs_only_a_catalog(tmp_path, capsys):
    assert cli.main(["replay", str(log(tmp_path)), "--catalog", str(FIX / "catalog.yaml")], env={}) == 0
    assert "policy fifo: 2 queued jobs" in capsys.readouterr().out


def test_the_replay_command_takes_the_catalog_from_the_config_and_prints_json(tmp_path, capsys):
    (tmp_path / "c.yaml").write_text(f"catalog: {FIX / 'catalog.yaml'}\n")
    assert cli.main(["-c", str(tmp_path / "c.yaml"), "replay", str(log(tmp_path)), "--json", "--policy", "fifo"], env={}) == 0
    (r,) = json.loads(capsys.readouterr().out)
    assert r["policy"] == "fifo" and r["wait"]["s"]["p50"] == 9   # sent at 1, starts once the 10 s call drains


def test_behind_others_leaves_out_the_time_a_requesters_own_jobs_ran():
    jobs = [Job("a1", 0, "a", "m", 10), Job("a2", 0, "a", "m", 10), Job("b", 5, "b", "m", 10)]
    res = Result(waits={"a1": 0, "a2": 10, "b": 15}, starts={"a1": 0, "a2": 10, "b": 20},
                 ends={"a1": 10, "a2": 20, "b": 30}, end=30)
    got = rp.behind_others(Trace(jobs, {"m": 0}, 0, 30), res)
    assert got == {"a1": 0, "a2": 0, "b": 15}   # a2 waited only for a1; b waited for both of a's
