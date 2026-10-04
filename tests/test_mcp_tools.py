"""The MCP tools through the SDK's client against the broker on the fake driver: schemas, the
job flow, the wait cap, inline images, inputs, chat and who sees which job."""
import base64
import dataclasses
import json
import threading

import anyio
import pytest
from mcp import Client

from gpu_broker.mcp_server import core, server
from tests.mcp_fakes import COMFY, PNG, PNG_B64, call, mcp_broker, report

__all__ = ["mcp_broker"]   # the fixture, imported for use here

TOOLS = {"list_models", "generate_image", "generate_video", "edit_image", "image_to_video", "make_3d",
         "job_status", "job_result", "chat_local", "gpu_status"}


def tools(broker):
    async def go():
        async with Client(server.build(broker, default_caller="main")) as c:
            return {t.name: t for t in (await c.list_tools()).tools}
    return anyio.run(go)


def test_the_tool_list_and_schemas(mcp_broker):
    t = tools(mcp_broker)
    assert set(t) == TOOLS
    assert t["generate_image"].input_schema["required"] == ["prompt"]
    assert t["edit_image"].input_schema["required"] == ["image", "prompt"]
    assert t["image_to_video"].input_schema["required"] == ["image", "prompt"]
    assert set(t["make_3d"].input_schema["properties"]) >= {"image", "images", "video"}
    assert t["job_status"].input_schema["required"] == ["job_id"]
    for name in ("list_models", "job_status", "job_result", "gpu_status"):
        assert t[name].annotations.read_only_hint is True, name
    for name in ("generate_image", "generate_video", "edit_image", "image_to_video", "make_3d"):
        assert t[name].annotations.read_only_hint is False and t[name].annotations.destructive_hint is False, name
        assert "wait_s" in t[name].input_schema["properties"], name
    assert "ctx" not in json.dumps(t["generate_image"].input_schema)   # the context is not a parameter


def test_list_models_is_what_can_run_now(mcp_broker):
    names = {m["name"]: m for m in json.loads(call(mcp_broker, "list_models").content[0].text)}
    assert "sdxl-base" in names and names["sdxl-base"]["kind"] == "image"
    assert "flux-schnell" not in names                 # needs_integration
    assert not any(m["kind"] == "ui" for m in names.values())
    assert names["llama-8b"]["loaded"] is True


def test_generate_image_waits_and_returns_the_image_inline(mcp_broker, monkeypatch):
    mcp_broker.backends.view_data = PNG
    r = call(mcp_broker, "generate_image", {"prompt": "a fox", "width": 512, "height": 512})
    out = report(r)
    assert out["state"] == "done" and out["model"] == "sdxl-base" and "hint" not in out
    assert out["outputs"][0]["url"].startswith(COMFY + "/view?")
    assert r.content[1].type == "image" and base64.b64decode(r.content[1].data) == PNG
    assert r.content[1].mime_type == "image/png"
    assert mcp_broker.backends.views == [(out["job_id"] + ".png", "broker", "output")]   # through the broker's ComfyUI client
    job = mcp_broker.store.job(out["job_id"])
    assert job["requester"] == core.MAIN_REQUESTER and job["payload"]["width"] == 512


def test_an_image_larger_than_the_cap_stays_a_url(mcp_broker, monkeypatch):
    r = call(mcp_broker, "generate_image", {"prompt": "a fox"})
    assert report(r)["state"] == "done" and len(r.content) == 1


def test_a_long_job_returns_its_id_and_a_hint_then_job_result_waits(mcp_broker, monkeypatch):
    mcp_broker.backends.view_data = PNG
    mcp_broker.backends.gate = threading.Event()
    out = report(call(mcp_broker, "generate_video", {"prompt": "waves", "wait_s": 0.2}))
    assert out["state"] in ("queued", "running") and out["hint"] == core.POLL_HINT
    assert report(call(mcp_broker, "job_status", {"job_id": out["job_id"]}))["state"] in ("queued", "running")
    mcp_broker.backends.gate.set()
    done = report(call(mcp_broker, "job_result", {"job_id": out["job_id"], "wait_s": 5}))
    assert done["state"] == "done" and done["outputs"]


def test_the_wait_is_capped_by_the_broker(mcp_broker, monkeypatch):
    mcp_broker.settings = dataclasses.replace(mcp_broker.settings, mcp=dataclasses.replace(mcp_broker.settings.mcp, wait_s=0.2))
    mcp_broker.backends.gate = threading.Event()
    out = report(call(mcp_broker, "generate_video", {"prompt": "waves", "wait_s": 600}))
    assert out["state"] in ("queued", "running") and "job_id" in out


def test_edit_image_takes_base64_through_the_inputs_layer(mcp_broker, monkeypatch):
    mcp_broker.backends.view_data = PNG
    out = report(call(mcp_broker, "edit_image", {"image": PNG_B64, "prompt": "make it blue"}))
    assert out["state"] == "done" and out["model"] == "qwen-image-edit"
    assert mcp_broker.store.job(out["job_id"])["payload"]["inputs"]["image"]["type"] == "png"


def test_input_problems_are_tool_errors(mcp_broker):
    r = call(mcp_broker, "edit_image", {"image": base64.b64encode(b"not an image").decode(), "prompt": "x"})
    assert r.is_error and "image" in r.content[0].text
    r = call(mcp_broker, "edit_image", {"image": "https://example.com/a.png", "prompt": "x"})
    assert r.is_error and "image_url" in r.content[0].text        # inputs.allow_urls is off
    r = call(mcp_broker, "make_3d", {"image": PNG_B64, "video": "https://example.com/v.mp4"})
    assert r.is_error and "exactly one" in r.content[0].text
    r = call(mcp_broker, "generate_image", {"prompt": "x", "model": "someone/new-model"})
    assert r.is_error and "can run now" in r.content[0].text      # never a download


def test_image_to_video_and_make_3d_pick_a_model_that_takes_the_input(mcp_broker, monkeypatch):
    mcp_broker.backends.view_data = PNG
    out = report(call(mcp_broker, "image_to_video", {"image": PNG_B64, "prompt": "turn"}))
    assert out["state"] == "done" and "i2v" in mcp_broker.catalog.models[out["model"]].get("caps", []) + \
        mcp_broker.catalog.models[out["model"]].get("image_caps", [])


def test_chat_local_answers_from_the_loaded_model(mcp_broker):
    out = json.loads(call(mcp_broker, "chat_local", {"prompt": "hi", "system": "be brief"}).content[0].text)
    assert out["text"] == "llama-3.1-8b-instruct-q4_k_m" and out["model"] == "llama-8b"
    sent = mcp_broker.backends.sent[-1]
    assert sent["messages"][0] == {"role": "system", "content": "be brief"} and sent["stream"] is False


def test_a_client_key_sees_only_its_own_jobs(mcp_broker, monkeypatch):
    mine = report(call(mcp_broker, "generate_image", {"prompt": "a"}, caller="key:k1:laptop"))
    assert mcp_broker.store.job(mine["job_id"])["requester"] == "laptop"
    theirs = report(call(mcp_broker, "generate_image", {"prompt": "b"}))
    r = call(mcp_broker, "job_status", {"job_id": theirs["job_id"]}, caller="key:k1:laptop")
    assert r.is_error and core.HIDDEN in r.content[0].text
    assert report(call(mcp_broker, "job_status", {"job_id": mine["job_id"]}, caller="key:k1:laptop"))["state"] == "done"
    assert report(call(mcp_broker, "job_status", {"job_id": mine["job_id"]}))["state"] == "done"   # the main token sees all


def test_gpu_status_shows_a_client_key_its_own_jobs_and_a_count_of_the_rest(mcp_broker):
    mcp_broker.backends.gate = threading.Event()
    theirs = report(call(mcp_broker, "generate_video", {"prompt": "w", "wait_s": 0}))
    mine = report(call(mcp_broker, "generate_video", {"prompt": "v", "wait_s": 0}, caller="key:k1:laptop"))
    seen = json.loads(call(mcp_broker, "gpu_status", caller="key:k1:laptop").content[0].text)
    assert {"loaded_llm", "last_comfy_model", "vram"} <= set(seen)   # the video evicted the LLM
    assert [r["job_id"] for r in seen["running"] + seen["queue"]] == [mine["job_id"]]
    assert seen["others_running"] + seen["others_queued"] == 1
    main = json.loads(call(mcp_broker, "gpu_status").content[0].text)
    assert {theirs["job_id"], mine["job_id"]} <= {r["job_id"] for r in main["running"] + main["queue"]}
    assert "others_running" not in main


@pytest.mark.parametrize("header", [None, "", "key:", "key:k1", "key::laptop", "admin", "Main"])
def test_without_the_guards_header_every_tool_refuses(mcp_broker, header):
    async def go():
        async with Client(server.build(mcp_broker, default_caller=header)) as c:
            return await c.call_tool("gpu_status", {})
    r = anyio.run(go)
    assert r.is_error and "not authenticated" in r.content[0].text
