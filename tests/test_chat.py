"""Interactive chat skips the queue and streams; background work never takes the reserved slot;
a switch in progress, a background caller or a non-resident model fall back to the queue."""
import json
import threading
import time

from gpu_broker.constants import JobState
from tests.helpers import WAIT_S, done

MSGS = [{"role": "user", "content": "hi"}]


def chat(c, model, headers=None, **extra):
    return c.post("/v1/chat/completions", json={"model": model, "messages": MSGS, **extra}, headers=headers or {})


def words(text):
    return [json.loads(f.removeprefix("data: "))["choices"][0]["delta"]["content"]
            for f in text.split("\n\n") if f.startswith("data: {") and '"delta"' in f]


def last_job(broker):
    return broker.store.job(broker.store.jobs(1)[0]["id"])


def test_interactive_stream_is_relayed_token_by_token_and_recorded(client, broker):
    r = chat(client, "llama-8b", stream=True)
    assert r.headers["content-type"].startswith("text/event-stream")
    assert words(r.text) == ["llama", "3.1", "8b", "instruct", "q4_k_m"]   # chunk by chunk, not one blob
    j = last_job(broker)
    assert j["state"] == JobState.DONE and j["resolved"] == "llama-8b"
    assert j["result"] == {"streamed": True, "timings": {"predicted_per_second": 90.0}}
    assert not broker.scheduler.pool.busy()   # slot released
    assert "job.direct" in [e["kind"] for e in broker.store.events(0, 1000)]


def test_interactive_plain_json_is_direct(client, broker):
    r = chat(client, "llama").json()
    assert r["x_broker"]["direct"] is True and r["x_broker"]["used"] == "llama-8b"
    assert broker.backends.streamed == []


def test_variant_maps_to_its_parent_with_overrides(client, broker):
    chat(client, "llama-8b-precise", stream=True)
    sent = broker.backends.streamed[0]
    assert sent["model"] == "llama-8b" and sent["temperature"] == 0.1


def test_chat_overtakes_a_saturated_background_queue(client, broker):
    gate = broker.backends.chat_gate = threading.Event()
    m = broker.catalog.models["llama-8b"]
    background = m["slots"] - m["reserved_interactive"]
    try:
        jids = [broker.submit({"model": "llama", "messages": MSGS}, "batch-agent")[0] for _ in range(background + 2)]
        end = time.monotonic() + WAIT_S
        while len(broker.scheduler.pool.ids()) < background and time.monotonic() < end:
            time.sleep(0.005)
        assert len(broker.scheduler.pool.ids()) == background   # background is capped below the slot count
        broker.backends.chat_gate = None                         # the interactive call itself is fast
        t0 = time.monotonic()
        r = chat(client, "llama-8b", stream=True)
        assert "llama" in words(r.text) and time.monotonic() - t0 < 1   # did not wait behind the queue
    finally:
        gate.set()
    assert all(done(broker, j)["state"] == JobState.DONE for j in jids)


def test_background_callers_are_queued(client, broker):
    for headers in ({"x-requester": "batch-agent"}, {"x-priority": "background"}):
        r = chat(client, "llama-8b", headers=headers)
        assert r.status_code == 200 and "direct" not in r.json()["x_broker"]
    assert broker.backends.streamed == []
    r = chat(client, "llama-8b", headers={"x-requester": "batch-agent", "x-priority": "interactive"})
    assert r.json()["x_broker"]["direct"] is True   # the header wins


def test_switch_in_progress_or_other_model_falls_back_to_the_queue(client, broker):
    broker.scheduler.pool.reopen(None)   # what close_and_drain leaves while residency changes
    r = chat(client, "llama-8b")
    assert r.status_code == 200 and "direct" not in r.json()["x_broker"]
    r = chat(client, "qwen-32b")         # not resident: the queue switches models
    assert r.json()["x_broker"]["used"] == "qwen-coder-32b" and "direct" not in r.json()["x_broker"]
    assert broker.scheduler.pool.resident == "qwen-coder-32b"   # reopened on the new resident
    assert all(j["state"] == JobState.DONE for j in broker.store.jobs(10))   # a fallback leaves no failed job


def test_a_direct_caller_waiting_for_a_slot_gives_up_when_the_model_switches(broker):
    pool, m = broker.scheduler.pool, broker.catalog.models["llama-8b"]
    held = [broker.store.create_job("t", "llama-8b", {}) for _ in range(m["slots"])]
    for jid in held:
        assert pool.acquire(jid, "llama-8b", m, interactive=True, direct=True)
    waiter = broker.store.create_job("t", "llama-8b", {})
    got = []
    t = threading.Thread(target=lambda: got.append(pool.acquire(waiter, "llama-8b", m, interactive=True, direct=True)))
    t.start()
    time.sleep(0.05)
    pool.reopen(None)
    t.join(WAIT_S)
    assert got == [False]
    for jid in held:
        broker.store.update_job(jid, state=JobState.DONE)
        pool.release(jid)


def test_a_failed_switch_reopens_the_pool(client, broker):
    broker.driver.fail_start.add("qwen-coder")
    assert chat(client, "qwen-32b").status_code == 502
    assert broker.scheduler.pool.resident == broker.residency.current


def test_quiesce_closes_the_direct_path_and_resume_reopens_it(client, broker):
    client.post("/v1/admin/quiesce", params={"wait_s": 1})
    assert broker.scheduler.pool.resident is None
    client.post("/v1/admin/resume")
    assert broker.scheduler.pool.resident == "llama-8b"
    assert chat(client, "llama-8b").json()["x_broker"]["direct"] is True


def test_a_failing_stream_is_recorded_and_frees_its_slot(client, broker):
    def boom(model, payload, summary):
        yield "data: {}\n\n"
        raise OSError("connection reset")
    broker.backends.llm_stream = boom
    chat(client, "llama-8b", stream=True)
    j = last_job(broker)
    assert j["state"] == JobState.FAILED and "reset" in j["error"] and not broker.scheduler.pool.busy()
