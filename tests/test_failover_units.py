"""Failover pieces below the routes: the error classifier, wait headers, the `upstreams:` config,
breaker transitions on a fake clock, the transport's timeouts, and the no-leak guarantee for the
client's provider key (logs, events, the database, the JSONL log, the dashboard's views)."""
from __future__ import annotations

import json
import logging
import pathlib
import time

import pytest

from gpu_broker.failover import transport
from gpu_broker.failover.breaker import Board, Breaker, State
from gpu_broker.failover.classify import Kind, classify, first_event_failed, retry_after
from gpu_broker.failover.config import BreakerCfg, Upstreams, UpTimeouts
from gpu_broker.settings import load
from gpu_broker.web.failover import header_value, make_router
from tests.failover_fakes import ANT, ERRORS, OAI, OPENAI_PLANTED, PLANTED, FakeCloud, build
from tests.helpers import TOKEN, wait_idle

CFG = BreakerCfg(failures=3, probe_s=10, probe_max_s=40, quota_probe_s=900, trial_s=60)
HI = [{"role": "user", "content": "hi"}]


def body(mode: str) -> bytes:
    return json.dumps(ERRORS[mode][1]).encode()


# ---- classify -----------------------------------------------------------------------

@pytest.mark.parametrize(("status", "mode", "kind"), [
    (529, "529", Kind.FAIL), (500, "500", Kind.FAIL), (503, None, Kind.FAIL), (408, None, Kind.FAIL),
    (429, "quota429", Kind.QUOTA), (402, "credit402", Kind.QUOTA), (402, None, Kind.QUOTA),
    (429, "rate429", Kind.CLIENT), (400, "400", Kind.CLIENT), (401, "401", Kind.CLIENT), (404, None, Kind.CLIENT),
    (200, None, Kind.OK)])
def test_classification(status, mode, kind):
    assert classify("p", status, {}, body(mode) if mode else b"").kind is kind


def test_reasons_carry_the_error_type_never_the_message():
    leaky = json.dumps({"error": {"type": "server_error", "message": f"key {PLANTED} broke"}}).encode()
    v = classify("anthropic", 500, {}, leaky)
    assert v.reason == "anthropic 500 server_error" and PLANTED not in v.reason
    weird = json.dumps({"error": {"type": f"{PLANTED}"}}).encode()   # not an identifier: dropped
    assert classify("p", 500, {}, weird).reason == "p 500"


def test_quota_honours_retry_after():
    v = classify("p", 429, {"Retry-After": "1200"}, body("quota429"))
    assert v.kind is Kind.QUOTA and v.retry_after_s == 1200


@pytest.mark.parametrize(("headers", "seconds"), [
    ({"retry-after": "30"}, 30), ({"Retry-After": "Thu, 01 Jan 1970 00:10:00 GMT"}, 600),
    ({"anthropic-ratelimit-tokens-reset": "1970-01-01T00:05:00Z"}, 300),
    ({"x-ratelimit-reset-requests": "6m0s"}, 360), ({"x-ratelimit-reset-tokens": "1.5s"}, 1.5),
    ({"x-ratelimit-reset-tokens": "200ms"}, 0.2), ({"retry-after": "30", "x-ratelimit-reset-requests": "2m"}, 120),
    ({"retry-after": "soon"}, 0), ({}, 0)])
def test_wait_headers(headers, seconds):
    assert retry_after(headers, now=0) == pytest.approx(seconds)


@pytest.mark.parametrize(("chunk", "reason"), [
    (b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error"}}\n\n', "a stream error overloaded_error"),
    (b'event: error\ndata: not json\n\n', "a stream error"),
    (b'data: {"error": {"type": "server_error"}}\n\n', "a stream error server_error"),
    (b'data: {"type": "error", "error": {"type": "api_error"}}\n\n', "a stream error api_error")])
def test_a_stream_opening_with_an_error_event_is_a_failure(chunk, reason):
    v = first_event_failed("a", chunk)
    assert v is not None and v.kind is Kind.FAIL and v.reason == reason


@pytest.mark.parametrize("chunk", [b'event: message_start\ndata: {"type":"message_start"}\n\n',
                                   b'data: {"choices": [{"delta": {"content": "error"}}]}\n\n', b": ping\n\n"])
def test_a_normal_first_event_is_not(chunk):
    assert first_event_failed("a", chunk) is None


def test_header_values_are_printable_and_short():
    assert header_value("a\r\nX-Evil: 1\u00e9" + "z" * 400) == "a??X-Evil: 1?" + "z" * 287


# ---- config -------------------------------------------------------------------------

def test_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text('upstreams:\n  providers:\n    anthropic: {url: "https://api.anthropic.com/"}\n'
                 '  routes:\n    "claude-*": [anthropic, my-local-llm]\n  breaker: {failures: 5}\n')
    u = load(str(p), {}).upstreams
    assert u.enabled and u.chain("claude-opus-4") == ("anthropic", "my-local-llm") and u.chain("gpt-4o") is None
    assert u.providers["anthropic"].url == "https://api.anthropic.com" and u.breaker.failures == 5
    assert not Upstreams().enabled   # off by default


@pytest.mark.parametrize(("providers", "routes", "msg"), [
    ({"a": {"url": "ftp://x", "api": "openai"}}, {}, "is not an http"),
    ({"a": {"url": "https://x", "api": "grpc"}}, {}, "api"),
    ({"a": {"url": "https://x", "api": "openai", "key": "sk"}}, {}, "unknown keys"),   # never a key in the file
    ({"a": {"url": "https://x", "api": "openai"}}, {"m*": ["local", "a"]}, "start with a provider"),
    ({"a": {"url": "https://x", "api": "openai"}}, {"m*": ["a", "l1", "l2"]}, "at most one local"),
    ({"a": {"url": "https://x", "api": "openai"}, "b": {"url": "https://y", "api": "openai"}},
     {"m*": ["a", "l1", "b"]}, "must come last"),
    ({"a": {"url": "https://x", "api": "openai"}}, {"m*": []}, "list")])
def test_config_errors(providers, routes, msg):
    with pytest.raises(ValueError, match=msg):
        Upstreams(providers=providers, routes=routes)


def test_an_unknown_local_fallback_is_refused_at_startup(tmp_path):
    cloud = FakeCloud()
    try:
        _, b, _, _ = build(tmp_path, cloud, routes={"claude-*": ["anthropic", "no-such-model"]})
        try:
            with pytest.raises(ValueError, match="neither providers nor catalog models"):
                make_router(b)
        finally:
            b.stop()
    finally:
        cloud.close()


# ---- breaker ------------------------------------------------------------------------

def test_breaker_opens_after_n_failures_and_lets_one_trial_through():
    b = Breaker(CFG, CFG.failures)
    for _ in range(2):
        b.failure(0, "x")
    assert b.state is State.CLOSED and b.allow(0)
    assert b.failure(0, "x") and b.state is State.OPEN
    assert not b.allow(9.9)
    assert b.allow(10) and b.state is State.HALF_OPEN
    assert not b.allow(10)   # only one trial at a time
    assert b.success() and b.state is State.CLOSED and b.failures == 0


def test_failed_trials_double_the_wait_up_to_the_cap():
    b = Breaker(CFG, CFG.failures)
    for _ in range(3):
        b.failure(0, "x")
    t = 0.0
    for want in (20, 40, 40):
        t = b.next_probe
        assert b.allow(t)
        b.failure(t, "x")
        assert b.next_probe - t == want


def test_a_trial_that_never_reports_back_is_given_up():
    b = Breaker(CFG, 1)
    b.failure(0, "x")
    assert b.allow(10) and not b.allow(69) and b.allow(70)


def test_quota_opens_at_once_for_the_longer_wait():
    b = Breaker(CFG, 1)
    assert b.exhausted(0, "q", 30) and b.next_probe == 900
    b = Breaker(CFG, 1)
    b.exhausted(0, "q", 3600)
    assert b.next_probe == 3600 and not b.probe_due(4000)   # quota trials are real requests, not probes


def test_a_quota_answer_proves_the_provider_reachable():
    board = Board(CFG, ["p"])
    board.failure("p", 0, "x")
    board.failure("p", 0, "x")
    board.exhausted("p", "k", 0, "q", 0)
    board.failure("p", 0, "x")   # the run of failures was broken by an answer
    assert board.allow("p", "k2", 0) == ""


def test_board_keeps_reachability_and_quota_apart():
    seen = []
    board = Board(CFG, ["p"], lambda *a: seen.append(a))
    board.exhausted("p", "k1", 0, "q", 0)
    assert "quota exhausted" in board.allow("p", "k1", 1) and board.allow("p", "k2", 1) == ""
    for _ in range(3):
        board.failure("p", 1, "down")
    assert "circuit open" in board.allow("p", "k2", 2)
    assert board.due(11, set()) == []        # no operator key: nothing to probe with
    assert board.due(11, {"p"}) == ["p"]
    board.reachable("p")
    assert board.allow("p", "k2", 12) == "" and "quota" in board.allow("p", "k1", 12)
    assert [s for _, s, _ in seen] == ["quota", State.OPEN, State.CLOSED]


# ---- transport ----------------------------------------------------------------------

T = UpTimeouts(connect_s=0.5, first_byte_s=0.3, response_s=0.3, idle_s=0.3)


def test_transport_refused_and_dns():
    with pytest.raises(transport.Unreachable, match="connection failed"):
        transport.send("http://127.0.0.1:9/v1/x", "GET", {}, None, False, T)
    with pytest.raises(transport.Unreachable, match="DNS"):
        transport.send("http://gpu-broker-test.invalid/v1/x", "GET", {}, None, False, T)


def test_transport_stream_that_goes_quiet_after_starting():
    cloud = FakeCloud()
    try:
        cloud.mode = "stream_stall"
        slow_start = UpTimeouts(connect_s=0.5, first_byte_s=2.5, response_s=2.5, idle_s=0.3)   # idle is its own limit
        a = transport.send(cloud.url + "/v1/chat/completions", "POST", {}, b'{"stream": true}', True, slow_start)
        start = time.monotonic()
        with pytest.raises(transport.Unreachable, match="went quiet"):
            list(a.rest)
        assert time.monotonic() - start < 2
    finally:
        cloud.close()


def test_transport_stream_idle_after_start():
    cloud = FakeCloud()
    try:
        cloud.mode = "stream_die"
        a = transport.send(cloud.url + "/v1/chat/completions", "POST", {"content-type": "application/json"},
                           b'{"stream": true}', True, T)
        assert a.status == 200 and b"chatcmpl-cloud" in a.body
        with pytest.raises(transport.Unreachable, match=r"broke off|went quiet"):
            list(a.rest)
    finally:
        cloud.close()


# ---- no leaks -----------------------------------------------------------------------

def test_the_clients_provider_keys_are_never_logged_or_stored(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    cloud = FakeCloud()
    client, b, router, clock = build(tmp_path, cloud)
    try:
        msg = {"model": "claude-x", "max_tokens": 9, "messages": HI}
        chat = {"model": "gpt-x", "messages": HI}
        for mode in ("ok", "529", "quota429", "drop", "400", "stream_die", "500", "500", "500"):
            cloud.mode = mode
            client.post("/v1/messages", json=msg | {"stream": mode == "stream_die"}, headers=ANT)
            client.post("/v1/chat/completions", json=chat, headers=OAI)
        clock.t += 10_000
        cloud.mode = "ok"
        router.probe_once()
        main = {"Authorization": f"Bearer {TOKEN}"}
        views = [client.get(p, headers=main).text for p in ("/v1/upstreams", "/v1/events?since=0&limit=1000", "/v1/jobs")]
        assert any(s["headers"].get("x-api-key") == PLANTED for s in cloud.seen)   # it was passed through
        assert wait_idle(b)
    finally:
        b.stop()
        cloud.close()
    stored = b"".join(pathlib.Path(p).read_bytes() for p in (tmp_path / "b.db", tmp_path / "e.jsonl")
                      if pathlib.Path(p).exists())
    stored += b"".join(f.read_bytes() for f in tmp_path.glob("b.db-*"))   # WAL / journal
    for secret in (PLANTED, OPENAI_PLANTED, "PLANTED"):
        assert secret.encode() not in stored
        assert secret not in caplog.text
        assert all(secret not in v for v in views)


def test_check_prints_the_routes_and_refuses_an_unknown_local_model(tmp_path, capsys):
    from gpu_broker import cli
    from tests.test_cli import EX, conf
    path = conf(tmp_path, EX / "catalog.yaml")
    with open(path, "a") as f:
        f.write('upstreams:\n  providers:\n    anthropic: {url: "https://api.anthropic.com"}\n'
                '  routes:\n    "claude-*": [anthropic, no-such-model]\n')
    assert cli.main(["-c", path, "check"], env={}) == cli.EXIT_PROBLEMS
    out = capsys.readouterr().out
    assert "cloud:   claude-* -> anthropic -> no-such-model" in out and "'no-such-model' is neither" in out


def test_every_script_the_dashboard_loads_is_served(tmp_path):
    import re

    from gpu_broker.web.dash import STATIC
    cloud = FakeCloud()
    client, b, _, _ = build(tmp_path, cloud)
    try:
        page = (STATIC / "dash.html").read_text()
        scripts = re.findall(r'<script src="(/dash/[a-z]+\.js)"', page)
        assert "/dash/upstreams.js" in scripts
        for src in scripts:
            r = client.get(src)
            assert r.status_code == 200 and "javascript" in r.headers["content-type"], src
    finally:
        b.stop()
        cloud.close()
