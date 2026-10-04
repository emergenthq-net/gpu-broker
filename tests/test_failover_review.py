"""Review round on failover (PR #28): one or more tests per finding.

1 the client's provider key goes only to providers with pass_client_key (official hosts by default)
2 every credential header is checked: a client key beside the main token is never forwarded
3 quota comes from error.type/code, never the message: a plain 429 mentioning billing is CLIENT
4 an Anthropic Bearer (OAuth / auth-token) client key goes back out as Bearer
5 previous_response_id unknown here goes to the cloud chain, never local; null is absent
6 client keys are not kept; the prober uses operator keys only, else the next request is the trial
7 breaker changes are reported after the lock; failovers are one event per provider per minute
8 quota breakers are dropped once closed and bounded (LRU)
9 the prober waits for the earliest due probe, probes providers independently, with a short timeout
10 an unrelayed stream is closed; pings and comments do not hide a first error event
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from gpu_broker.constants import Event
from gpu_broker.failover import breaker as breaker_mod
from gpu_broker.failover import transport
from gpu_broker.failover.breaker import Board, State
from gpu_broker.failover.classify import Kind, classify, first_event
from gpu_broker.failover.config import Upstreams, UpTimeouts
from gpu_broker.failover.creds import Cred
from gpu_broker.failover.router import Router
from gpu_broker.failover.tally import Tally
from gpu_broker.web.failover import FALLBACK, SERVED_BY
from tests.failover_fakes import ANT, BREAKER, ERRORS, FAST, LOCAL, OAI, OPENAI_PLANTED, PLANTED, FakeCloud, build
from tests.helpers import TOKEN, wait_idle

HI = [{"role": "user", "content": "hi"}]
MSG = {"model": "claude-sonnet-4-5", "max_tokens": 50, "messages": HI}
CHAT = {"model": "gpt-4o", "messages": HI}


@pytest.fixture
def cloud():
    c = FakeCloud()
    yield c
    c.close()


@pytest.fixture
def setup(tmp_path, cloud):
    made = []

    def make(**kw):
        client, b, router, clock = build(tmp_path, cloud, **kw)
        made.append(b)
        return client, b, router, clock
    yield make
    for b in made:
        assert wait_idle(b)
        b.stop()


def events(b, kind):
    return [e["data"] for e in b.store.events(0, 1000) if e["kind"] == kind]


def seen_keys(cloud):
    return json.dumps([s["headers"] for s in cloud.seen])


# ---- 1 --------------------------------------------------------------------------------

@pytest.mark.parametrize(("url", "passes"), [
    ("https://api.openai.com", True), ("https://api.anthropic.com", True),
    ("https://openrouter.ai/api", False), ("http://api.openai.com", False), ("http://127.0.0.1:9", False)])
def test_pass_client_key_defaults_on_only_for_the_official_hosts(url, passes):
    up = Upstreams(providers={"p": {"url": url, "api": "openai"}}, routes={"x": ["p"]})
    assert up.providers["p"].pass_client_key is passes
    assert Upstreams(providers={"p": {"url": url, "api": "openai", "pass_client_key": not passes}},
                     routes={"x": ["p"]}).providers["p"].pass_client_key is not passes
    with pytest.raises(ValueError, match="pass_client_key"):
        Upstreams(providers={"p": {"url": url, "api": "openai", "pass_client_key": "yes"}}, routes={"x": ["p"]})


def proxy_chain(cloud, key_env=""):
    return {"providers": {"anthropic": {"url": cloud.url, "api": "anthropic", "pass_client_key": True},
                          "openai": {"url": cloud.url, "api": "openai", "pass_client_key": True},
                          "proxy": {"url": cloud.url, "api": "openai", **({"key_env": key_env} if key_env else {})}},
            "routes": {"gpt-*": ["openai", "proxy", LOCAL]}}


def test_a_client_key_never_reaches_a_provider_without_pass_client_key(setup, cloud):
    client, *_ = setup(**proxy_chain(cloud))
    cloud.mode = "500"
    r = client.post("/v1/chat/completions", json=CHAT, headers=OAI)
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and "proxy: no credential" in r.headers[FALLBACK]
    assert len(cloud.posts()) == 1   # the official provider only


SLACK = UpTimeouts(connect_s=5, probe_s=5, first_byte_s=10, response_s=10, idle_s=10)   # never hit: 500s answer at once


def test_a_provider_without_pass_client_key_uses_its_operator_key(setup, cloud):
    client, *_ = setup(env={"UP_PROXY": "sk-or-operator"}, timeouts=SLACK, **proxy_chain(cloud, "UP_PROXY"))
    cloud.mode = "500"
    client.post("/v1/chat/completions", json=CHAT, headers=OAI)
    first, second = cloud.posts()
    assert first["headers"]["authorization"] == f"Bearer {OPENAI_PLANTED}"
    assert second["headers"]["authorization"] == "Bearer sk-or-operator" and OPENAI_PLANTED not in json.dumps(second)


# ---- 2 --------------------------------------------------------------------------------

def test_a_client_key_beside_the_main_token_is_a_broker_credential(setup, cloud):
    client, *_ = setup()
    key = client.post("/v1/keys", json={"name": "laptop"}, headers={"Authorization": f"Bearer {TOKEN}"}).json()["key"]
    r = client.post("/v1/messages", json=MSG, headers={"Authorization": f"Bearer {TOKEN}", "x-api-key": key})
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and "no credential" in r.headers[FALLBACK]
    assert cloud.seen == [] and key not in seen_keys(cloud)


def test_the_main_token_in_any_header_is_never_forwarded(setup, cloud):
    client, *_ = setup()
    r = client.post("/v1/messages", json=MSG, headers={"x-gpu-broker-key": TOKEN, "x-api-key": TOKEN})
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and TOKEN not in seen_keys(cloud)


# ---- 3 --------------------------------------------------------------------------------

RATE = json.dumps({"error": {"message": "Rate limit reached for gpt-4o on requests per min. Limit: 500. "
                                        "Visit https://platform.openai.com/account/billing to increase it.",
                             "type": "requests", "code": "rate_limit_exceeded"}}).encode()


@pytest.mark.parametrize(("status", "data", "kind"), [
    (429, RATE, Kind.CLIENT),
    (400, json.dumps({"error": {"type": "invalid_request_error", "message": "credit balance billing quota"}}).encode(), Kind.CLIENT),
    (429, json.dumps({"error": {"type": "insufficient_quota", "message": "x"}}).encode(), Kind.QUOTA),
    (429, json.dumps({"error": {"type": "requests", "code": "insufficient_quota", "message": "x"}}).encode(), Kind.QUOTA),
    (400, json.dumps({"type": "error", "error": {"type": "billing_error", "message": "x"}}).encode(), Kind.QUOTA)])
def test_quota_is_read_from_the_error_type_or_code_only(status, data, kind):
    assert classify("p", status, {}, data).kind is kind


def test_a_plain_openai_rate_limit_is_returned_and_opens_nothing(setup, cloud, monkeypatch):
    client, b, router, _ = setup()
    monkeypatch.setitem(ERRORS, "ratebilling", (429, json.loads(RATE), {"retry-after": "2"}))
    cloud.mode = "ratebilling"
    r = client.post("/v1/chat/completions", json=CHAT, headers=OAI)
    assert r.status_code == 429 and r.headers[SERVED_BY] == "openai"
    assert router.view()["providers"]["openai"]["keys_out_of_quota"] == 0 and not events(b, Event.UPSTREAM_QUOTA)


# ---- 4 --------------------------------------------------------------------------------

def test_an_anthropic_bearer_client_key_goes_upstream_as_bearer(setup, cloud):
    client, *_ = setup()
    r = client.post("/v1/messages", json=MSG, headers={"x-gpu-broker-key": TOKEN, "Authorization": "Bearer sk-ant-oat01-OAUTH"})
    assert r.headers[SERVED_BY] == "anthropic"
    sent = cloud.posts()[-1]["headers"]
    assert sent["authorization"] == "Bearer sk-ant-oat01-OAUTH" and "x-api-key" not in sent


def test_an_anthropic_x_api_key_goes_upstream_as_x_api_key(setup, cloud):
    client, *_ = setup()
    client.post("/v1/messages", json=MSG, headers=ANT)
    sent = cloud.posts()[-1]["headers"]
    assert sent["x-api-key"] == PLANTED and "authorization" not in sent


# ---- 5 --------------------------------------------------------------------------------

def test_an_unknown_previous_response_goes_to_the_cloud(setup, cloud):
    client, *_ = setup()
    r = client.post("/v1/responses", json={"model": "gpt-5", "input": "more", "previous_response_id": "resp_cloud"}, headers=OAI)
    assert r.status_code == 200 and r.headers[SERVED_BY] == "openai"
    assert cloud.posts()[-1]["body"]["previous_response_id"] == "resp_cloud"


def test_a_cloud_continuation_never_falls_back_to_local(setup, cloud):
    client, b, *_ = setup()
    cloud.mode = "500"
    r = client.post("/v1/responses", json={"model": "gpt-5", "input": "more", "previous_response_id": "resp_cloud"}, headers=OAI)
    assert r.status_code == 503 and "only the provider can continue" in r.json()["error"]["message"]
    assert not events(b, Event.UPSTREAM_FAILOVER)


def test_a_null_previous_response_is_routed_like_none(setup, cloud):
    client, *_ = setup()
    r = client.post("/v1/responses", json={"model": "gpt-5", "input": "hi", "previous_response_id": None}, headers=OAI)
    assert r.headers[SERVED_BY] == "openai" and "previous_response_id" not in cloud.posts()[-1]["body"]


def test_a_turn_the_local_fallback_served_continues_locally(setup, cloud):
    client, *_ = setup()
    cloud.mode = "500"
    first = client.post("/v1/responses", json={"model": "gpt-5", "input": "hi"}, headers=OAI)
    assert first.headers[SERVED_BY] == f"local:{LOCAL}"
    n = len(cloud.posts())
    r = client.post("/v1/responses", json={"model": "gpt-5", "input": "more", "previous_response_id": first.json()["id"]}, headers=OAI)
    assert r.status_code == 200 and len(cloud.posts()) == n
    assert r.json()["model"] == first.json()["model"] == LOCAL   # the chain's local model, as before


# ---- 6 --------------------------------------------------------------------------------

def test_client_keys_are_not_kept_after_their_request(setup, cloud):
    client, _, router, _ = setup()
    cloud.mode = "500"
    for _ in range(BREAKER.failures):
        client.post("/v1/messages", json=MSG, headers=ANT)
    held = json.dumps({k: repr(v) for k, v in vars(router.board).items()}) + repr(router.board.links)
    assert PLANTED not in held and not hasattr(router.board, "last_cred")


def test_without_an_operator_key_nothing_is_probed_and_the_next_request_is_the_trial(setup, cloud):
    client, _, router, clock = setup()
    cloud.mode = "500"
    for _ in range(BREAKER.failures):
        client.post("/v1/messages", json=MSG, headers=ANT)
    cloud.mode = "ok"
    clock.t += BREAKER.probe_s
    n = len(cloud.seen)
    router.probe_once()
    assert len(cloud.seen) == n   # no probe with a client's key
    r = client.post("/v1/messages", json=MSG, headers=ANT)
    assert r.headers[SERVED_BY] == "anthropic" and router.view()["providers"]["anthropic"]["state"] == "closed"


# ---- 7 --------------------------------------------------------------------------------

def test_breaker_changes_are_reported_outside_the_lock():
    held = []
    board: Board

    def notify(*_):
        held.append(board.lock.locked())
    board = Board(BREAKER, ["p"], notify)
    for _ in range(BREAKER.failures):
        board.failure("p", 0, "down")
    board.exhausted("p", "k", 0, "q", 0)
    board.success("p", "k")
    board.failure("p", 1, "x")
    board.reachable("p")
    assert held and not any(held)


def test_failovers_are_one_event_per_provider_per_minute():
    out, t = [], [0.0]
    tally = Tally(lambda kind, **f: out.append((kind, f)), lambda: t[0])
    for _ in range(5):
        tally.add("anthropic", "local:m", "claude-x", "down")
    assert len(out) == 1 and out[0][1]["count"] == 1
    t[0] = 30
    tally.flush()   # the window is still open: nothing recorded, nothing dropped
    assert len(out) == 1 and tally.next_flush() == 60
    t[0] = 60
    tally.flush()
    assert len(out) == 2 and out[1][0] == Event.UPSTREAM_FAILOVER and out[1][1]["count"] == 4
    tally.flush()
    assert len(out) == 2 and tally.next_flush() is None


def test_a_failover_outage_writes_one_event_not_one_per_request(setup, cloud):
    client, b, *_ = setup()
    cloud.mode = "500"
    for _ in range(6):
        client.post("/v1/messages", json=MSG, headers=ANT)
    assert len(events(b, Event.UPSTREAM_FAILOVER)) == 1


# ---- 8 --------------------------------------------------------------------------------

def test_a_closed_quota_breaker_is_dropped():
    board = Board(BREAKER, ["p"])
    board.exhausted("p", "k", 0, "q", 0)
    assert len(board.keys) == 1
    board.success("p", "k")
    assert board.keys == {}
    board.allow("p", "fresh", 0)
    board.success("p", "fresh")
    assert board.keys == {}   # a key that never ran out is never kept


def test_quota_breakers_are_bounded_least_recently_used_first(monkeypatch):
    monkeypatch.setattr(breaker_mod, "KEYS_MAX", 3)
    board = Board(BREAKER, ["p"])
    for k in ("a", "b", "c"):
        board.exhausted("p", k, 0, "q", 0)
    board.allow("p", "a", 1)   # a is used again: b is now the oldest
    board.exhausted("p", "d", 0, "q", 0)
    assert len(board.keys) == 3
    assert board.allow("p", "b", 1) == "" and "quota" in board.allow("p", "a", 1)


# ---- 9 --------------------------------------------------------------------------------

def up_router(send, providers=("a", "b")):
    up = Upstreams(providers={p: {"url": f"http://{p}.test", "api": "openai", "key_env": f"K_{p}"} for p in providers},
                   routes={"m": list(providers)}, breaker=BREAKER, timeouts=FAST)
    clock = [1000.0]
    r = Router(up, lambda *a, **k: None, {f"K_{p}": "op" for p in providers}, clock=lambda: clock[0], send=send)
    return r, clock


def test_the_prober_waits_until_the_earliest_probe_is_due():
    r, clock = up_router(lambda *a: transport.Answer(200, {}, b"{}"))
    assert r.prober.delay() is None   # nothing open: sleep until a breaker changes
    for _ in range(BREAKER.failures):
        r.board.failure("a", clock[0], "down")
    assert r.prober.delay() == BREAKER.probe_s and r.prober.wake.is_set()
    clock[0] += 4
    assert r.prober.delay() == BREAKER.probe_s - 4


def test_probes_use_the_short_timeout_and_one_hung_provider_holds_up_no_other():
    release, timeouts, done = threading.Event(), [], []

    def send(url, method, headers, body, stream, t):
        timeouts.append(t.response_s)
        if url.startswith("http://a."):
            release.wait(5)
        done.append(url)
        return transport.Answer(200, {}, b"{}")
    r, clock = up_router(send)
    for p in ("a", "b"):
        for _ in range(BREAKER.failures):
            r.board.failure(p, clock[0], "down")
    clock[0] += BREAKER.probe_s
    threads = r.prober.probe_once(block=False)
    deadline = time.monotonic() + 2
    while r.view()["providers"]["b"]["state"] != "closed" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert r.view()["providers"]["b"]["state"] == "closed" and r.view()["providers"]["a"]["state"] == "half-open"
    release.set()
    for t in threads:
        t.join()
    assert set(timeouts) == {FAST.probe_s} and UpTimeouts().probe_s < UpTimeouts().response_s


def test_the_prober_sleeps_while_nothing_is_open():
    r, _ = up_router(lambda *a: transport.Answer(200, {}, b"{}"))
    looked = []
    due = r.board.due
    r.board.due = lambda *a: looked.append(1) or due(*a)
    worker = threading.Thread(target=r.prober.run, daemon=True)
    worker.start()
    time.sleep(1.5)   # a poll would have run by now
    r.stop()
    worker.join(2)
    assert looked == []   # no wake-up: nothing was due, so nothing was looked at
    assert not worker.is_alive()


def test_the_prober_thread_wakes_on_a_breaker_change():
    calls = []
    r, clock = up_router(lambda url, *a: calls.append(url) or transport.Answer(200, {}, b"{}"))
    worker = threading.Thread(target=r.prober.run, daemon=True)
    worker.start()
    try:
        for _ in range(BREAKER.failures):
            r.board.failure("a", clock[0], "down")
        clock[0] += BREAKER.probe_s   # due now; the change woke the prober and it re-reads the delay
        r.prober.wake.set()
        deadline = time.monotonic() + 2
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls == ["http://a.test/v1/models"]
    finally:
        r.stop()
        worker.join(2)
    assert not worker.is_alive()


# ---- 10 -------------------------------------------------------------------------------

def test_a_ping_before_an_error_event_does_not_hide_it(setup, cloud):
    client, *_ = setup()
    cloud.mode = "stream_ping_error"
    r = client.post("/v1/messages", json=MSG | {"stream": True}, headers=ANT)
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and "stream error overloaded_error" in r.headers[FALLBACK]


@pytest.mark.parametrize(("buf", "want"), [
    (b": ok\n\nevent: ping\ndata: {}\n\nevent: error\ndata: {}\n\n", b"event: error\ndata: {}"),
    (b"data: {\"a\":1}\r\n\r\n", b"data: {\"a\":1}"),
    (b": ok\n\nevent: ping\ndata: {}\n\nevent: message_start\ndata: {", None),   # not complete yet
    (b"", None)])
def test_first_real_event(buf, want):
    assert first_event(buf) == want


def test_an_unrelayed_stream_is_closed():
    closed = []
    error = b"event: error\ndata: {\"type\": \"error\", \"error\": {\"type\": \"overloaded_error\"}}\n\n"

    def send(*a):
        return transport.Answer(200, {}, b": hi\n\n", iter([error]), stream=True, close=lambda: closed.append(1))
    r, _ = up_router(send, providers=("a",))
    got = r.route(r.cfg.providers["a"].api, "/v1/chat/completions", "m", {"model": "m"}, None, {}, True)
    assert type(got).__name__ == "Exhausted" and closed == [1]


def test_a_stream_with_no_event_in_its_opening_is_closed_and_failed_over():
    closed = []

    def send(*a):
        return transport.Answer(200, {}, b": pad\n\n" * 20000, iter(()), stream=True, close=lambda: closed.append(1))
    r, _ = up_router(send, providers=("a",))
    got = r.route(r.cfg.providers["a"].api, "/v1/chat/completions", "m", {"model": "m"}, None, {}, True)
    assert type(got).__name__ == "Exhausted" and "no event" in got.reason and closed == [1]


def test_a_redirect_to_a_stream_request_is_read_whole_not_left_open(cloud):
    cloud.mode = "redirect"
    a = transport.send(cloud.url + "/v1/messages", "POST", {"content-type": "application/json"}, b"{}", True, FAST)
    assert a.status == 302 and not a.stream and a.body == b"moved"


def test_creds_never_show_their_value():
    assert "secret" not in repr(Cred("secret", bearer=True))


def test_a_board_state_reads_closed_after_success():
    board = Board(BREAKER, ["p"])
    board.failure("p", 0, "x")
    board.success("p", "k")
    assert board.links["p"].state is State.CLOSED


# ---- verification round: nothing broker-shaped reaches a provider ---------------------

def issue_key(client, name="laptop"):
    return client.post("/v1/keys", json={"name": name}, headers={"Authorization": f"Bearer {TOKEN}"}).json()


def test_the_main_token_with_odd_bearer_spacing_is_never_forwarded(setup, cloud):
    client, *_ = setup()
    r = client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": f"Bearer  {TOKEN}"})
    assert r.status_code == 200 and r.headers[SERVED_BY] == f"local:{LOCAL}" and cloud.seen == []


def test_a_revoked_client_key_is_never_forwarded(setup, cloud):
    client, *_ = setup()
    k = issue_key(client)
    assert client.delete(f"/v1/keys/{k['id']}", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    r = client.post("/v1/messages", json=MSG, headers={"x-gpu-broker-key": TOKEN, "x-api-key": k["key"]})
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and cloud.seen == []


def test_an_unknown_client_key_is_never_forwarded(setup, cloud):
    client, *_ = setup()
    fake = "gbk_" + "A" * 43
    r = client.post("/v1/chat/completions", json=CHAT, headers={"x-gpu-broker-key": TOKEN, "Authorization": f"Bearer {fake}"})
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and cloud.seen == []


def variants(value):
    return [value, f" {value}", f"{value} ", f"\t{value}"]


def bearers(value):
    return [f"Bearer {value}", f"Bearer  {value}", f"bearer {value}", f"BEARER {value}", f" Bearer {value} "]


def test_nothing_resembling_a_broker_credential_reaches_a_provider(setup, cloud):
    """Every credential header, every broker secret (main token; client keys valid, revoked,
    unknown), every spacing and case: the fake provider sees none of them."""
    client, *_ = setup()
    valid, revoked = issue_key(client, "a"), issue_key(client, "b")
    client.delete(f"/v1/keys/{revoked['id']}", headers={"Authorization": f"Bearer {TOKEN}"})
    secrets = [TOKEN, valid["key"], revoked["key"], "gbk_" + "Z" * 43, "GBK_" + "q" * 43]
    sent = 0
    for secret in secrets:
        cases = [{"Authorization": b} for b in bearers(secret)] + [{"x-api-key": v} for v in variants(secret)]
        for case in cases:
            for path, body in (("/v1/messages", MSG), ("/v1/chat/completions", CHAT)):
                auth = {} if secret in (TOKEN, valid["key"]) and "x-api-key" not in case else {"x-gpu-broker-key": TOKEN}
                client.post(path, json=body, headers=auth | case)
                sent += 1
    assert sent == 90
    control = client.post("/v1/messages", json=MSG, headers={"x-gpu-broker-key": TOKEN, "x-api-key": " sk-ant-own "})
    assert control.status_code == 200   # the provider does see a real key, so silence above means something
    seen = json.dumps(cloud.seen).lower()
    assert "sk-ant-own" in seen
    for secret in secrets:
        assert secret.lower() not in seen
    assert "gbk_" not in seen


def bare_request(headers, token=TOKEN):
    """A request auth recorded nothing on, so only client_key's own guard stands in the way."""
    from starlette.requests import Request

    from gpu_broker.web.auth import MAIN_CHECK
    req = Request({"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]})
    setattr(req.state, MAIN_CHECK, lambda v: v.strip() == token)
    return req


@pytest.mark.parametrize("headers", [{"x-api-key": f" {TOKEN}"}, {"Authorization": f"bearer   {TOKEN} "},
                                     {"x-api-key": "gbk_" + "x" * 43}, {"Authorization": "Bearer GBK_revoked"}])
def test_client_key_refuses_broker_shaped_values_even_if_auth_missed_them(headers):
    from gpu_broker.failover.config import Api
    from gpu_broker.web.failover import client_key
    assert client_key(bare_request(headers), Api.ANTHROPIC) is None


def test_client_key_still_passes_a_real_provider_key():
    from gpu_broker.failover.config import Api
    from gpu_broker.web.failover import client_key
    cred = client_key(bare_request({"Authorization": "bearer  sk-own "}), Api.OPENAI)
    assert cred is not None and cred.value == "sk-own" and cred.bearer


def test_a_padded_main_token_in_x_api_key_authenticates(setup):
    client, *_ = setup()
    assert client.post("/v1/messages", json=MSG, headers={"x-api-key": f" {TOKEN} "}).status_code == 200


def test_auth_leaves_a_working_main_token_check_for_the_final_guard():
    from gpu_broker.web.auth import MAIN_CHECK, make_auth
    req = bare_request({"Authorization": f"Bearer {TOKEN}"}, token="")   # auth must replace this check
    make_auth(TOKEN)(req)
    check = getattr(req.state, MAIN_CHECK)
    assert check(f"  {TOKEN}\t") and not check("sk-own") and not check("")
