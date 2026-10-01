from gpu_broker.metrics import job_metrics, parse_sample, percentile, summarize
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


def test_percentile():
    assert percentile([None, 3, 1, 2], 0.5) == 2 and percentile([], 0.5) is None
