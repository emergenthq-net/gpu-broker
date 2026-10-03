"""Drop-in pieces below the SDKs: name mapping, settings, both auth headers, error shapes, the
embeddings 404, a broken stream, and the translation edge cases."""
from __future__ import annotations

import json

import pytest

from gpu_broker import settings
from gpu_broker.modelmap import joined, map_name, parse_map
from gpu_broker.web.anthropic_req import to_openai
from gpu_broker.web.anthropic_resp import message, stop_reason
from gpu_broker.web.anthropic_sse import events
from tests.dropin_fakes import EMBED_MODEL, MODEL_MAP, ToolBackends, app_client, catalog_with_embedder, dropin_broker  # noqa: F401
from tests.helpers import ROOT, TOKEN, FakeDriver, make_settings

HI = [{"role": "user", "content": "hi"}]
ANTHROPIC = {"model": "claude-x", "max_tokens": 8, "messages": HI}


def test_map_name_first_match_wins_ignores_case_and_skips_known_names():
    m = map_name(MODEL_MAP, False, "GPT-4o-Mini", "llama-8b")
    assert m is not None and m.target == "llama-8b-precise"
    assert map_name(MODEL_MAP, False, "gpt-5", "llama-8b").target == "llama-8b"   # @default -> resident
    assert map_name(MODEL_MAP, True, "gpt-5", "llama-8b") is None                  # the catalog knows it
    assert map_name(MODEL_MAP, False, "mistral-large", "llama-8b") is None         # no pattern: unchanged
    assert joined(None, "a", None, "b") == "a; b" and joined(None) is None


def test_model_map_from_file_and_env(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model_map: {'gpt-*': '@default'}\n")
    assert settings.load(str(p), {}).model_map == {"gpt-*": "@default"}
    env = settings.load(str(p), {"BROKER_MODEL_MAP": json.dumps({"claude-*": "qwen-32b"})})
    assert env.model_map == {"claude-*": "qwen-32b"}
    with pytest.raises(ValueError, match="model_map"):
        parse_map({"gpt-*": 3})


def test_shipped_example_config_maps_hosted_names():
    s = settings.load(str(ROOT / "examples" / "config.yaml"), {})
    for name in ("gpt-4o", "o3-mini", "chatgpt-4o-latest", "claude-sonnet-4-5"):
        assert map_name(s.model_map, False, name, "resident") is not None, name


def test_unmapped_unknown_name_keeps_the_resolver_substitution(app_client):
    r = app_client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {TOKEN}"},
                        json={"model": "mistral-large", "messages": HI})
    x = r.json()["x_broker"]
    assert "not in the catalog" in x["substitution"] and "model_map" not in x["substitution"]


@pytest.mark.parametrize("headers", [{"Authorization": f"Bearer {TOKEN}"}, {"x-api-key": TOKEN}])
def test_either_auth_header_is_accepted(app_client, headers):
    assert app_client.get("/v1/models", headers=headers).status_code == 200


@pytest.mark.parametrize("headers", [{}, {"x-api-key": "nope"}, {"Authorization": TOKEN}, {"x-api-key": f"Bearer {TOKEN}"}])
def test_anything_else_is_refused(app_client, headers):
    assert app_client.get("/v1/models", headers=headers).status_code == 401


def test_empty_broker_token_refuses_an_empty_api_key(dropin_broker):
    from fastapi.testclient import TestClient

    from gpu_broker.web.app import create_app
    with TestClient(create_app(dropin_broker, "", start=False)) as c:
        assert c.get("/v1/models", headers={"x-api-key": ""}).status_code == 401


def test_openai_errors_keep_detail_and_native_routes_are_unchanged(app_client):
    auth = {"Authorization": f"Bearer {TOKEN}"}
    r = app_client.post("/v1/chat/completions", headers=auth, json={"model": 3, "messages": HI})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error" and "string" in r.json()["detail"]
    native = app_client.post("/v1/jobs", headers=auth, json={"model": 3})
    assert native.status_code == 400 and set(native.json()) == {"detail"}


def test_anthropic_validation_errors_use_anthropic_shape(app_client):
    r = app_client.post("/v1/messages", headers={"x-api-key": TOKEN}, json={**ANTHROPIC, "messages": [{"role": "bot", "content": "x"}]})
    assert r.status_code == 400 and r.json()["error"] == {"type": "invalid_request_error", "message": "messages[0]: role must be 'user' or 'assistant'"}
    bad_json = app_client.post("/v1/messages", headers={"x-api-key": TOKEN}, content=b"[1]")
    assert bad_json.json()["type"] == "error"


def test_embeddings_without_an_embedding_model_is_a_404(tmp_path):
    from fastapi.testclient import TestClient

    from gpu_broker.broker import Broker
    from gpu_broker.web.app import create_app
    driver = FakeDriver({"llama-8b"})
    b = Broker(make_settings(tmp_path, catalog=catalog_with_embedder(tmp_path, embed=False)), env={}, driver=driver,
               backends=ToolBackends(driver))
    with TestClient(create_app(b, TOKEN, start=False)) as c:
        r = c.post("/v1/embeddings", headers={"Authorization": f"Bearer {TOKEN}"}, json={"model": "text-embedding-3-small", "input": "x"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    assert "caps: [embed]" in r.json()["error"]["message"]


def test_embeddings_by_catalog_name_or_a_chat_model(app_client):
    auth = {"Authorization": f"Bearer {TOKEN}"}
    direct = app_client.post("/v1/embeddings", headers=auth, json={"model": EMBED_MODEL, "input": "x"}).json()
    assert direct["x_broker"]["used"] == EMBED_MODEL and direct["x_broker"]["substitution"] is None
    chat = app_client.post("/v1/embeddings", headers=auth, json={"model": "llama", "input": "x"}).json()
    assert chat["x_broker"]["used"] == EMBED_MODEL and "'llama' is not an embedding model" in chat["x_broker"]["substitution"]


def test_queued_call_records_the_mapping_on_the_job(app_client, dropin_broker):
    r = app_client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {TOKEN}", "x-priority": "background"},
                        json={"model": "gpt-4o", "messages": HI}).json()
    assert "direct" not in r["x_broker"]
    assert "model_map pattern 'gpt-*'" in dropin_broker.store.job(r["x_broker"]["job"])["substitution"]


def test_a_broken_stream_ends_with_an_error_event():
    lines = ['data: {"choices": [{"delta": {"content": "par"}}]}\n\n']
    out = "".join(events(lines, "claude-x", False, {}, {"message": "server went away"}))
    assert '"text_delta"' in out and out.rstrip().endswith('"message": "server went away"}}')
    assert "event: error" in out and "message_stop" not in out


def test_stop_reasons():
    assert stop_reason("stop", None) == ("end_turn", None)
    assert stop_reason("length", None) == ("max_tokens", None)
    assert stop_reason("tool_calls", None) == ("tool_use", None)
    assert stop_reason("stop", "END") == ("stop_sequence", "END")
    m = message({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "stopping_word": "END"}, "c", False, {})
    assert (m["stop_reason"], m["stop_sequence"]) == ("stop_sequence", "END")


def test_translation_edge_cases():
    body = {**ANTHROPIC, "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
            "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                                                      "content": [{"type": "text", "text": "boom"}]}]}]}
    out = to_openai(body)
    assert out["messages"][0] == {"role": "system", "content": "a\nb"}
    assert out["messages"][1] == {"role": "tool", "tool_call_id": "t1", "content": "Error: boom"}
    assert out["tool_choice"] == "required" and out["parallel_tool_calls"] is False
    with pytest.raises(ValueError, match="document"):
        to_openai({**ANTHROPIC, "messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]})


def test_non_json_tool_arguments_survive():
    m = message({"choices": [{"message": {"tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "not json"}}]},
                              "finish_reason": "tool_calls"}]}, "c", False, {})
    assert m["content"][0]["input"] == {"_raw_arguments": "not json"}
