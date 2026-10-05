"""End to end through the HTTP API, with the driver and backends faked (see conftest)."""
import json
import time

from gpu_broker.constants import JobState
from tests.helpers import TOKEN, WAIT_S, done


def chat(c, model, **extra):
    return c.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "hi"}], **extra})


def test_every_data_route_needs_the_token(client):
    for path in ("/v1/status", "/v1/events", "/v1/models", "/v1/catalog", "/v1/gpu", "/v1/ui", "/v1/metrics", "/v1/stats"):
        for header in ("Bearer nope", f"Bearer{TOKEN}", f"Basic {TOKEN}", TOKEN, "Bearer ", ""):
            assert client.get(path, headers={"Authorization": header}).status_code == 401, (path, header)
        for header in (f"bearer {TOKEN}", f"Bearer {TOKEN} ", f"BEARER  {TOKEN}"):   # the scheme is case-insensitive (RFC 7235)
            assert client.get(path, headers={"Authorization": header}).status_code == 200, (path, header)
        assert client.get(path).status_code == 200, path
    assert client.post("/v1/jobs", json={}, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_no_configured_token_refuses_everything(broker):
    from fastapi.testclient import TestClient

    from gpu_broker.web.app import create_app
    with TestClient(create_app(broker, "", start=False)) as c:
        assert c.get("/v1/status", headers={"Authorization": "Bearer "}).status_code == 401
        assert c.get("/health").json()["ok"] is True


def test_security_headers_and_no_api_docs(client):
    r = client.get("/dash")
    assert "script-src 'self'" in r.headers["content-security-policy"] and r.headers["x-frame-options"] == "DENY"
    assert "<script>" not in r.text and "onclick" not in r.text   # CSP-compatible page
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_dashboard_serves_only_its_own_scripts(client):
    for name in ("dash", "live", "index", "imagejob"):
        assert client.get(f"/dash/{name}.js", headers={"Authorization": ""}).status_code == 200
    for name in ("../app", "dash.html", "..%2Fapp", "nope"):
        assert client.get(f"/dash/{name}.js").status_code == 404


def test_video_job_stops_the_llm_then_a_chat_restarts_it(client, broker):
    driver, backends = broker.driver, broker.backends
    j = done(broker, client.post("/v1/jobs", json={"model": "wan14b", "prompt": "fox"}).json()["id"])
    assert j["state"] == JobState.DONE and j["using"] == "wan2.2-14b-t2v" and j["result"]["outputs"]
    assert ("stop", "llama-8b") in driver.calls and "llama-8b" not in driver.active
    r = chat(client, "llama")
    assert r.status_code == 200 and r.json()["x_broker"]["used"] == "llama-8b"
    assert backends.frees >= 1 and "llama-8b" in driver.active


def test_switch_between_llms_stops_the_other(client, broker):
    assert chat(client, "qwen-32b").json()["x_broker"]["used"] == "qwen-coder-32b"
    assert broker.driver.active == {"qwen-coder"}


def test_substitution_is_reported_and_the_download_queued(client, broker):
    r = client.post("/v1/jobs", json={"model": "ltx-video", "caps": ["t2v", "style"], "prompt": "x"}).json()
    assert r["resolved"] == "wan2.2-14b-style" and "no runner" in r["substitution"]
    assert r["download"]["ref"] == "Lightricks/LTX-Video"
    done(broker, r["id"])


def test_unknown_repo_downloads_once_and_registers(client, broker):
    body = {"model": "someone/New-Video", "kind": "video", "caps": ["t2v"], "prompt": "x"}
    r = client.post("/v1/jobs", json=body).json()
    assert r["resolved"] == "wan2.2-14b-t2v"
    entry = broker.catalog.models["someone-new-video"]
    end = time.monotonic() + WAIT_S
    while not entry.get("downloaded") and time.monotonic() < end:   # marked after the download finishes
        time.sleep(0.01)
    assert broker.driver.downloads == [("hf", "someone/New-Video", "someone-new-video")]
    assert entry["status"] == "needs_integration" and entry["downloaded"] and "endpoint" not in entry
    done(broker, r["id"])
    done(broker, client.post("/v1/jobs", json=body).json()["id"])
    assert len(broker.driver.downloads) == 1


def test_impossible_and_malformed_requests(client):
    r = client.post("/v1/jobs", json={"model": "nope-model"}).json()
    assert r["job"]["state"] == JobState.REJECTED and "no kind" in r["error"]
    assert client.post("/v1/jobs", json={"model": 5}).status_code == 400
    assert client.post("/v1/jobs", json={"model": "llama", "caps": "chat"}).status_code == 400
    assert client.get("/v1/jobs/nope").status_code == 404
    assert chat(client, "nope-at-all", kind="audio").status_code == 503   # nothing can stand in


def test_dashboard_data(client, broker):
    done(broker, client.post("/v1/jobs", json={"model": "wan14b", "prompt": "fox"}).json()["id"])
    s = client.get("/v1/stats").json()
    assert s["models"]["wan2.2-14b-t2v"]["done"]["n"] == 1 and s["events"]
    assert len(client.get("/v1/events", params={"since": -3}).json()) == 3
    assert client.get("/v1/gpu").json() == {"used_mib": 8000, "total_mib": 24564, "util_pct": 37, "probe": "fake"}
    ui = client.get("/v1/ui").json()
    assert ui["comfy_url"] == broker.settings.comfy.url and ui["power_max_w"] == broker.settings.ui.power_max_w
    m = client.get("/v1/metrics").json()
    assert m["jobs"] and m["summary"]["jobs"] == len(m["jobs"])


def test_openai_model_list_is_ready_llms_and_their_variants(client):
    ids = {d["id"] for d in client.get("/v1/models").json()["data"]}
    cat = client.get("/v1/catalog").json()
    ready = {k: v for k, v in cat.items() if v["kind"] == "llm" and v["status"] == "ready"}
    assert ids == set(ready) | {n for v in ready.values() for n in v.get("variants") or {}}
    assert "llama-8b-precise" in ids


def test_queued_chat_streams_one_sse_chunk_when_asked(client):
    client.headers["x-priority"] = "background"   # the queued path; the direct path is in test_chat.py
    plain = chat(client, "llama-8b")
    assert plain.headers["content-type"].startswith("application/json")
    r = chat(client, "llama-8b", stream=True)
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = [f for f in r.text.split("\n\n") if f]
    assert frames[-1] == "data: [DONE]"
    first = json.loads(frames[0].removeprefix("data: "))
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["content"] == plain.json()["choices"][0]["message"]["content"]


def test_session_holds_the_gpu_and_rejects_non_comfy_models(client, broker):
    r = client.post("/v1/sessions", json={"model": "wan2.2-5b", "idle_min": 0.0001}).json()
    assert r["comfy_template"] == "video_wan2_2_5B_ti2v"
    j = done(broker, r["id"])
    assert j["state"] == JobState.DONE and j["result"]["ended"] == "idle"
    assert client.post("/v1/sessions", json={"model": "llama-8b"}).status_code == 400
    assert client.post("/v1/sessions", json={"model": "wan2.2-5b", "idle_min": -1}).status_code == 400
    assert client.post("/v1/sessions/end").json() == {"ended": False}


def test_quiesce_refuses_new_jobs_with_503_and_resume_takes_them_again(client, broker):
    assert client.post("/v1/admin/quiesce", params={"wait_s": 1}).json()["drained"] is True
    r = client.post("/v1/jobs", json={"model": "llama-8b", "messages": []})
    assert r.status_code == 503 and r.headers["retry-after"] == "5" and broker.store.jobs(limit=10) == []
    client.post("/v1/admin/resume")
    jid = client.post("/v1/jobs", json={"model": "llama-8b", "messages": []}).json()["id"]
    assert done(broker, jid)["state"] == JobState.DONE


def test_a_downloaded_catalog_model_becomes_ready(client, broker):
    m = broker.catalog.models["sdxl-base"]
    m["status"] = "downloadable"
    m["source"] = {"hf": m["source"]["hf"]}          # no explicit slug: the catalog key is the slug
    client.post("/v1/jobs", json={"model": "sdxl-base", "prompt": "x"})
    end = time.monotonic() + WAIT_S
    while m.get("status") != "ready" and time.monotonic() < end:
        time.sleep(0.01)
    assert m["downloaded"] is True and m["status"] == "ready"
    assert broker.driver.downloads == [("hf", "stabilityai/stable-diffusion-xl-base-1.0", "sdxl-base")]


def test_a_session_only_model_can_be_borrowed_but_not_queued(client, broker):
    key = next(k for k, m in broker.catalog.models.items() if m.get("session_only") and m.get("status") == "ready")
    assert client.post("/v1/jobs", json={"model": key, "prompt": "x"}).json()["job"]["state"] == JobState.REJECTED
    r = client.post("/v1/sessions", json={"model": key, "idle_min": 0.0001}).json()
    assert r["resolved"] == key and done(broker, r["id"])["state"] == JobState.DONE
