"""Fakes for the drop-in (OpenAI / Anthropic SDK) tests: a model server that can call a tool,
answer a tool result, reason, stream all of it the way llama-server does, and embed."""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
import yaml

from gpu_broker.broker import Broker
from tests.helpers import FIX, TOKEN, FakeBackends, FakeDriver, make_settings, wait_idle

TOOL = "get_weather"
TOOL_ARGS = '{"city": "Paris"}'
CALL_ID = "call_1"
ANSWER = "hello there friend"
REASONING = "the user said hi"
USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
VECTOR = [0.25, -0.5]
EMBED_MODEL = "nomic-embed"
MODEL_MAP = {"gpt-4o-mini": "llama-8b-precise",   # an override before the general pattern
             "gpt-*": "@default", "o1*": "@default", "claude-*": "@default", "text-embedding-*": EMBED_MODEL}


def answer(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """(assistant message, finish_reason) for a chat payload."""
    last = payload["messages"][-1]
    if last["role"] == "tool":
        return {"role": "assistant", "content": f"It is {last['content']}"}, "stop"
    if payload.get("tools"):
        call = {"id": CALL_ID, "type": "function", "function": {"name": TOOL, "arguments": TOOL_ARGS}}
        return {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
    return {"role": "assistant", "content": ANSWER, "reasoning_content": REASONING}, "stop"


def frame(chunk: dict[str, Any]) -> str:
    return "data: " + json.dumps({"object": "chat.completion.chunk", "model": "served", **chunk}) + "\n\n"


class ToolBackends(FakeBackends):
    def __init__(self, driver: FakeDriver) -> None:
        super().__init__(driver)
        self.embeds: list[dict[str, Any]] = []

    def llm_chat(self, model, payload) -> dict[str, Any]:
        self.sent.append(dict(payload))
        msg, finish = answer(payload)
        return {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": model["served_name"],
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": USAGE}

    def llm_stream(self, model, payload, summary) -> Iterator[str]:
        """Like llama-server: role, reasoning, content word by word, tool arguments in pieces."""
        self.streamed.append(dict(payload))
        msg, finish = answer(payload)
        yield frame({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        if msg.get("reasoning_content"):
            yield frame({"choices": [{"index": 0, "delta": {"reasoning_content": msg["reasoning_content"]}}]})
        for word in (msg.get("content") or "").split(" "):
            if word:
                yield frame({"choices": [{"index": 0, "delta": {"content": word + " "}}]})
        for call in msg.get("tool_calls") or []:
            head = {"index": 0, "id": call["id"], "type": "function", "function": {"name": TOOL, "arguments": ""}}
            yield frame({"choices": [{"index": 0, "delta": {"tool_calls": [head]}}]})
            for piece in (TOOL_ARGS[:5], TOOL_ARGS[5:]):
                yield frame({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}}]})
        yield frame({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        yield frame({"choices": [], "usage": USAGE})
        yield "data: [DONE]\n\n"

    def llm_embed(self, model, payload) -> dict[str, Any]:
        self.embeds.append(dict(payload))
        return {"object": "list", "model": model["served_name"], "usage": {"prompt_tokens": 2, "total_tokens": 2},
                "data": [{"object": "embedding", "index": 0, "embedding": VECTOR}]}


def catalog_with_embedder(tmp_path, embed: bool = True):
    data = yaml.safe_load((FIX / "catalog.yaml").read_text())
    if embed:
        data["models"][EMBED_MODEL] = {"kind": "llm", "runner": "llm_unit", "unit": EMBED_MODEL,
                                       "endpoint": "http://127.0.0.1:8090", "served_name": "nomic-embed-text-v1.5",
                                       "vram_mib": 600, "caps": ["embed"], "quality": 50, "status": "ready"}
    path = tmp_path / "src-catalog.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def dropin_broker(tmp_path):
    driver = FakeDriver({"llama-8b"})
    settings = make_settings(tmp_path, catalog=catalog_with_embedder(tmp_path), model_map=MODEL_MAP)
    b = Broker(settings, env={}, driver=driver, backends=ToolBackends(driver))
    b.start()
    yield b
    assert wait_idle(b), "test left a broker job in flight"
    b.stop()


@pytest.fixture
def app_client(dropin_broker):
    """A TestClient with NO default auth header: each SDK sends its own."""
    from fastapi.testclient import TestClient

    from gpu_broker.web.app import create_app
    with TestClient(create_app(dropin_broker, TOKEN, start=False)) as c:
        yield c
