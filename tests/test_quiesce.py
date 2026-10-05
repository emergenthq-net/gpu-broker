"""While quiesced, new work gets HTTP 503 with Retry-After on every route that would queue or
run it, in each API's own error shape, and nothing is recorded; the real OpenAI and Anthropic
SDKs retry it and succeed once the broker resumes. A job that slips in between the broker's
check and the queue is refused there too, so nothing is ever queued for a restart to orphan."""
from __future__ import annotations

import dataclasses
import threading
import time

import anthropic
import openai
import pytest

from gpu_broker.constants import JobState
from gpu_broker.quiesce import QUIESCED, Quiesced
from tests.dropin_fakes import ANSWER, EMBED_MODEL, app_client, dropin_broker  # noqa: F401
from tests.helpers import TOKEN, done

AUTH = {"Authorization": f"Bearer {TOKEN}"}
HI = [{"role": "user", "content": "hi"}]
ROUTES = [  # (path, body, extra headers, where the message is in the 503 body)
    ("/v1/jobs", {"model": "llama-8b", "messages": HI}, {}, ("detail",)),
    ("/v1/sessions", {"model": "sdxl-base", "idle_min": 0.0001}, {}, ("detail",)),
    ("/v1/chat/completions", {"model": "gpt-4o", "messages": HI}, {}, ("error", "message")),
    ("/v1/chat/completions", {"model": "gpt-4o", "messages": HI}, {"x-priority": "background"}, ("error", "message")),
    ("/v1/embeddings", {"model": EMBED_MODEL, "input": "x"}, {}, ("error", "message")),
    ("/v1/messages", {"model": "claude-x", "max_tokens": 8, "messages": HI}, {}, ("error", "message")),
    ("/v1/responses", {"model": "gpt-4o", "input": "hi"}, {}, ("error", "message")),
]


def quiesce(c):
    assert c.post("/v1/admin/quiesce", params={"wait_s": 1}, headers=AUTH).json()["drained"] is True


@pytest.mark.parametrize(("path", "body", "headers", "where"), ROUTES)
def test_every_route_answers_503_with_retry_after_and_records_nothing(app_client, dropin_broker, path, body, headers, where):
    quiesce(app_client)
    r = app_client.post(path, json=body, headers={**AUTH, **headers})
    assert r.status_code == 503 and r.headers["retry-after"] == "5" and r.headers["x-should-retry"] == "true"
    msg = r.json()
    for k in where:
        msg = msg[k]
    assert msg == QUIESCED
    assert dropin_broker.store.jobs(limit=10) == []          # no job row (a closed pool never opens a direct one)
    app_client.post("/v1/admin/resume", headers=AUTH)
    ok = app_client.post(path, json=body, headers={**AUTH, **headers})
    assert ok.status_code == 200
    if path in ("/v1/jobs", "/v1/sessions"):                # a native job: let it finish
        assert done(dropin_broker, ok.json()["id"])["state"] == JobState.DONE


def test_the_error_shapes_are_each_apis_own(app_client):
    quiesce(app_client)
    oai = app_client.post("/v1/chat/completions", json=ROUTES[2][1], headers=AUTH).json()
    assert oai["error"]["type"] == "server_error"
    ant = app_client.post("/v1/messages", json=ROUTES[5][1], headers=AUTH).json()
    assert ant["type"] == "error" and ant["error"]["type"] == "overloaded_error"
    app_client.post("/v1/admin/resume", headers=AUTH)


def test_a_submit_that_races_the_quiesce_is_refused_at_the_queue_and_recorded(dropin_broker):
    """The broker's own check passed; the quiesce lands before the job reaches the queue."""
    b = dropin_broker
    real_admit = b.scheduler.admit
    b.scheduler.admit = lambda: (real_admit(), b.scheduler.paused.set())[0]   # quiesce right after the check
    discarded: list[str] = []
    real_discard = b.staging.discard
    b.staging.discard = lambda jid: (discarded.append(jid), real_discard(jid))[1]
    with pytest.raises(Quiesced):
        b.submit({"model": "llama-8b", "messages": HI}, "t")
    b.scheduler.admit = real_admit
    (j,) = b.store.jobs(limit=10)
    assert j["state"] == JobState.REJECTED and j["error"] == QUIESCED and discarded == [j["id"]]   # inputs dropped
    assert b.scheduler.snapshot()[0] == [] and b.scheduler.position(j["id"]) is None
    b.scheduler.paused.clear()


@pytest.mark.parametrize("sdk", ["openai", "anthropic"])
def test_the_sdks_retry_the_503_and_succeed_after_resume(app_client, dropin_broker, sdk):
    dropin_broker.scheduler.i = dataclasses.replace(dropin_broker.scheduler.i, quiesced_retry_s=1)
    quiesce(app_client)
    if sdk == "openai":
        client = openai.OpenAI(base_url="http://testserver/v1", api_key=TOKEN, http_client=app_client, max_retries=3)
        call = lambda: client.chat.completions.create(model="gpt-4o", messages=HI).choices[0].message.content  # noqa: E731
    else:
        client = anthropic.Anthropic(base_url="http://testserver", api_key=TOKEN, http_client=app_client, max_retries=3)
        call = lambda: client.messages.create(model="claude-x", max_tokens=8, messages=HI).content[0].text  # noqa: E731
    out: list[str] = []
    t = threading.Thread(target=lambda: out.append(call()))
    t.start()
    time.sleep(0.3)                                     # the first attempt has been refused by now
    app_client.post("/v1/admin/resume", headers=AUTH)
    t.join(10)
    assert out and out[0].strip() == ANSWER
    states = [j["state"] for j in dropin_broker.store.jobs(limit=10)]
    assert states == [JobState.DONE]                    # the refused attempts left no job behind
