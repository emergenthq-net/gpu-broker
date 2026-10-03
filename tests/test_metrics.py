import time

import pytest

from gpu_broker import gpu
from gpu_broker.metrics import GpuSampler, job_metrics, parse_sample, percentile, summarize
from gpu_broker.store import Store


def test_parse_sample_sums_per_group():
    s = parse_sample("20679,24564,11,172.3,55,2520| llm:10908 comfyui:596 comfyui:100\n", 1.0)
    assert s["used_mib"] == 20679 and s["power_w"] == 172.3
    assert s["by_group"] == {"llm": 10908, "comfyui": 696}
    assert parse_sample("garbage", 1.0) is None


def test_job_metrics_phases_and_tokens(tmp_path):
    st = Store(str(tmp_path / "b.db"))
    jid = st.create_job("t", "llama-8b", {})
    st.update_job(jid, resolved="llama-8b", state="switching")
    st.update_job(jid, state="running")
    st.update_job(jid, state="done", result={"timings": {
        "prompt_n": 437, "cache_n": 655, "prompt_ms": 405.0, "prompt_per_second": 1078.0,
        "predicted_n": 6731, "predicted_per_second": 86.4}})
    (r,) = job_metrics(st, 0, 0)
    assert r["gen_tps"] == 86.4 and r["gen_tokens"] == 6731 and r["ttft_s"] >= 0.405
    assert r["switch_s"] is not None and r["run_s"] is not None
    sm = summarize([r])
    assert sm["gen_tps"]["p50"] == 86.4 and sm["gen_tokens_total"] == 6731 and sm["jobs"] == 1


def test_latest_is_the_newest_sample_only_while_fresh(monkeypatch):
    s = GpuSampler(lambda: iter(()), keep=10, retry_s=1)
    assert s.latest(5) is None
    s._buf.append(parse_sample("8000,24564,37,120,50,2500|", 100.0))
    monkeypatch.setattr(time, "time", lambda: 104.0)
    assert s.latest(5) == (8000, 24564, 37)
    monkeypatch.setattr(time, "time", lambda: 106.0)
    assert s.latest(5) is None


def test_percentile():
    assert percentile([None, 3, 1, 2], 0.5) == 2 and percentile([], 0.5) is None


def test_latest_ignores_a_sample_taken_before_after():
    s = GpuSampler(lambda: iter(()), 10, 1)
    s._buf.append({"t": time.time() - 1, "used_mib": 1, "total_mib": 2, "util_pct": 3})
    assert s.latest(5) == (1, 2, 3)
    assert s.latest(5, after=time.time() - 0.5) is None
    assert s.latest(5, after=time.time() - 2) == (1, 2, 3)


def test_unknown_power_temp_and_clock_are_none_and_the_rest_must_be_numbers():
    s = parse_sample("100,200,5,,,|", 1.0)
    assert (s["used_mib"], s["power_w"], s["temp_c"], s["clock_mhz"]) == (100, None, None, None)
    assert parse_sample("100,200,5,12.34,60,2500|", 1.0)["clock_mhz"] == 2500
    assert parse_sample(",200,5,1,1,1|", 1.0) is None and parse_sample("1,2,3,x,1,1|", 1.0) is None
    assert parse_sample("1,2,3,1,1|", 1.0) is None   # a field short


@pytest.mark.parametrize("line", [",2,3,,,|", "1,,3,,,|", "[N/A],2,3,,,|", "1,2,3|", "1,2,3,4,5,6,7|", "a,2,3,,,|"])
def test_a_line_without_memory_or_with_the_wrong_shape_is_dropped(line):
    assert parse_sample(line, 0) is None


def test_metrics_keep_sm_mhz_as_a_deprecated_alias_and_flag_unreadable_processes():
    s = parse_sample(f"1,2,3,,,2500|ct:5 {gpu.PROCS_UNREADABLE}", 0)
    assert s["sm_mhz"] == s["clock_mhz"] == 2500 and s["procs_unreadable"] is True and s["by_group"] == {"ct": 5}
    assert parse_sample("1,2,3,,,|ct:5", 0)["procs_unreadable"] is False
