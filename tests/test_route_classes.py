"""Chat and embeddings derive a call's class like /v1/jobs: the header, else the body's own
`interactive` claim, else the requester. A class the route worked out is never re-read as a
claim (with `may_claim_interactive` set, that would demote a plain call from an unlisted
requester); a body may lower its class on every route."""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from gpu_broker.broker import Broker
from gpu_broker.catalog import Catalog
from gpu_broker.classes import Classes
from gpu_broker.settings import Scheduling
from gpu_broker.web.app import create_app
from tests.dropin_fakes import EMBED_MODEL, ToolBackends, catalog_with_embedder
from tests.helpers import TOKEN, FakeDriver, make_settings, wait_idle

AUTH = {"Authorization": f"Bearer {TOKEN}"}
MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(params=[None, ("alice",)], ids=["default-claims", "alice-may-claim"])
def broker(tmp_path, request):
    s = dataclasses.replace(make_settings(tmp_path, catalog=catalog_with_embedder(tmp_path)),
                            scheduler=Scheduling(may_claim_interactive=request.param))
    driver = FakeDriver({"llama-8b"})
    b = Broker(s, env={}, driver=driver, backends=ToolBackends(driver))
    b.start()
    yield b
    assert wait_idle(b)
    b.stop()


@pytest.fixture
def client(broker):
    with TestClient(create_app(broker, TOKEN, start=False)) as c:
        yield c


def last_payload(broker):
    return broker.store.job(broker.store.jobs(1)[0]["id"])["payload"]


def chat(client, model="llama-8b", requester="bob", headers=None, **body):
    return client.post("/v1/chat/completions", json={"model": model, "messages": MSGS, **body},
                       headers={**AUTH, "x-requester": requester, **(headers or {})})


def test_classes_derive_the_default_and_a_claim_only_raises_for_the_listed(tmp_path):
    cat = Catalog(str(make_settings(tmp_path).catalog))
    c = Classes.of("fair", ["alice"])
    assert c.of_request(cat, {}, "", "bob") is True                       # no claim: by requester
    assert c.of_request(cat, {"interactive": True}, "", "bob") is False   # bob may not raise
    assert c.of_request(cat, {"interactive": True}, "", "alice") is True
    assert c.of_request(cat, {"interactive": False}, "", "alice") is False   # anyone may lower
    assert c.for_queue({"interactive": False, "x": 1}, True) == {"interactive": False, "x": 1}   # the claim only
    assert Classes.of("fifo", None).for_queue({"x": 1}, True) == {"x": 1, "interactive": True}


def test_a_plain_call_from_an_unlisted_requester_stays_interactive_on_both_paths(client, broker):
    r = chat(client)
    assert r.status_code == 200 and r.json()["x_broker"]["direct"] is True        # direct path
    r = chat(client, model="qwen-32b")                                              # not resident: queued
    assert r.status_code == 200 and "direct" not in r.json()["x_broker"]
    assert last_payload(broker)["interactive"] is True
    r = client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input": "x"}, headers={**AUTH, "x-requester": "bob"})
    assert r.status_code == 200 and last_payload(broker)["interactive"] is True     # embeddings too


def test_a_body_can_lower_its_class_on_chat_and_embeddings(client, broker):
    r = chat(client, interactive=False)
    assert r.status_code == 200 and "direct" not in r.json()["x_broker"]           # queued, not direct
    assert last_payload(broker)["interactive"] is False
    r = client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input": "x", "interactive": False},
                    headers={**AUTH, "x-requester": "bob"})
    assert r.status_code == 200 and last_payload(broker)["interactive"] is False


def test_the_header_still_wins_over_the_body(client, broker):
    r = chat(client, headers={"x-priority": "background"}, interactive=True)
    assert "direct" not in r.json()["x_broker"] and last_payload(broker)["interactive"] is False


def test_under_fifo_the_route_writes_the_class_and_a_body_still_lowers_it(tmp_path):
    s = dataclasses.replace(make_settings(tmp_path, catalog=catalog_with_embedder(tmp_path)), scheduler=Scheduling(policy="fifo"))
    driver = FakeDriver({"llama-8b"})
    b = Broker(s, env={}, driver=driver, backends=ToolBackends(driver))
    b.start()
    try:
        with TestClient(create_app(b, TOKEN, start=False)) as c:
            for interactive in (False, None):
                body = {"model": EMBED_MODEL, "input": "x", **({} if interactive is None else {"interactive": interactive})}
                assert c.post("/v1/embeddings", json=body, headers={**AUTH, "x-requester": "bob"}).status_code == 200
                assert last_payload(b)["interactive"] is (interactive is None)
            assert chat(c, model="qwen-32b", interactive=False).status_code == 200
            assert last_payload(b)["interactive"] is False
    finally:
        assert wait_idle(b)
        b.stop()
