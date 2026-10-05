"""`connect` against a broker with cloud failover: tools that can send a custom header keep their
own provider key and add the broker key as `x-gpu-broker-key` (Claude Code through
ANTHROPIC_CUSTOM_HEADERS, Codex through `http_headers`); every other client, and every broker
without a pass-through provider for the tool's API, keeps today's single broker key. `clients`
says which a client has (or would get), and disconnect restores every file byte for byte, also
after a reconnect switched modes and the user edited the file since."""
from __future__ import annotations

import json
import tomllib

import pytest

from gpu_broker import constants
from gpu_broker.connect import CLIENTS, claude_code, codex, core, engine, upstreams
from tests.failover_fakes import FakeCloud, build
from tests.helpers import TOKEN as BROKER_TOKEN
from tests.helpers import wait_idle
from tests.test_connect_cli import TOKEN, FakeBroker, run
from tests.test_connect_clients import URL, home, snapshot, target, without_state  # noqa: F401

OPENAI_KEY = "sk-proj-own-account-key"
OWN = {core.OWN_KEY_OPT["anthropic"]: "1", core.OWN_KEY_OPT["openai"]: "1"}
CLAUDE = claude_code.SETTINGS


class Cloudy(FakeBroker):
    """A FakeBroker whose pass-through route reports failover on, passing these APIs' keys through."""

    def __init__(self, apis: list[str], enabled: bool = True) -> None:
        super().__init__()
        self.view = {"enabled": enabled, "apis": apis}

    def __call__(self, method, url, credential, body):
        if url.endswith(core.PASSTHROUGH_PATH):
            self.calls.append((method, url, credential, body))
            return self.view
        return super().__call__(method, url, credential, body)


BOTH = ["anthropic", "openai"]


def env_of(h):
    return json.loads((h / CLAUDE).read_text())["env"]


def test_the_header_name_and_path_match_the_broker():
    assert core.BROKER_KEY_HEADER == constants.BROKER_KEY_HEADER
    assert core.PASSTHROUGH_PATH == constants.PASSTHROUGH_PATH


# ---- what the broker says -------------------------------------------------------------

@pytest.mark.parametrize(("view", "want"), [
    ({"enabled": True, "apis": BOTH}, {"anthropic", "openai"}),
    ({"enabled": True, "apis": []}, set()),
    ({"enabled": True, "apis": ["openai", "gemini"]}, {"openai"}),   # an API connect has no tool for
    ({"enabled": False, "apis": BOTH}, set()),
    ({"revoked": True}, set()),   # an older broker's answer to an unknown route
])
def test_passthrough_apis(view, want):
    assert upstreams.passthrough_apis(URL, "cred", lambda *a: view) == want


def test_passthrough_apis_without_a_credential_or_a_broker_counts_as_off():
    def down(*a):
        raise OSError("refused")
    assert upstreams.passthrough_apis(URL, "", lambda *a: {"enabled": True, "apis": BOTH}) == set()
    assert upstreams.passthrough_apis(URL, "cred", down) == set()


def test_a_client_key_can_ask_the_real_broker(tmp_path):
    """connect asks with the key it was just issued, not the main token: the operator view
    (/v1/upstreams) refuses that key, so the answer must come from the model-scope route."""
    cloud = FakeCloud()
    client, b, *_ = build(tmp_path, cloud, env={"UP": "sk-operator-secret"},
                          providers={"anthropic": {"url": cloud.url, "api": "anthropic", "key_env": "UP", "pass_client_key": True},
                                     "openai": {"url": cloud.url, "api": "openai"}})
    try:
        key = client.post("/v1/keys", headers={"Authorization": f"Bearer {BROKER_TOKEN}"}, json={"name": "m"}).json()["key"]
        assert client.get("/v1/upstreams", headers={"Authorization": f"Bearer {key}"}).status_code == 403
        r = client.get(constants.PASSTHROUGH_PATH, headers={"Authorization": f"Bearer {key}"})
        assert r.json() == {"enabled": True, "apis": ["anthropic"]}   # no names, URLs, breakers or keys
        assert upstreams.passthrough_apis("http://testserver", key,
                                          lambda m, u, c, _b: client.get(u.removeprefix("http://testserver"),
                                                                         headers={"Authorization": f"Bearer {c}"}).json()) == {"anthropic"}
    finally:
        assert wait_idle(b)
        b.stop()
        cloud.close()


# ---- Claude Code ----------------------------------------------------------------------

def test_claude_code_keeps_its_own_key_and_adds_the_broker_header(home):
    (home / CLAUDE).write_text(json.dumps({"env": {"ANTHROPIC_CUSTOM_HEADERS": "X-Team: blue\nx-gpu-broker-key: stale"}}))
    before = snapshot(home)
    p = claude_code.plan(target(home, claude_code="1", **OWN))
    engine.connect(home, [p])
    env = env_of(home)
    assert "ANTHROPIC_AUTH_TOKEN" not in env and env["ANTHROPIC_BASE_URL"] == URL
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Team: blue\nx-gpu-broker-key: gbk_testkey"
    assert p.keys == upstreams.DUAL and "own login" in p.notes[0]
    engine.disconnect(home)
    assert without_state(snapshot(home)) == without_state(before)


def test_claude_code_single_key_without_pass_through(home):
    p = claude_code.plan(target(home, claude_code="1", **{core.OWN_KEY_OPT["openai"]: "1"}))   # OpenAI only
    engine.connect(home, [p])
    env = env_of(home)
    assert env["ANTHROPIC_AUTH_TOKEN"] == "gbk_testkey" and "ANTHROPIC_CUSTOM_HEADERS" not in env
    assert p.keys == upstreams.SINGLE


def test_a_reconnect_that_switches_mode_then_an_edit_still_disconnects_cleanly(home):
    original = json.loads((home / CLAUDE).read_text())
    engine.connect(home, [claude_code.plan(target(home, claude_code="1"))])           # single
    engine.connect(home, [claude_code.plan(target(home, claude_code="1", **OWN))])    # failover turned on since
    data = json.loads((home / CLAUDE).read_text())
    # the broker key leaves ANTHROPIC_AUTH_TOKEN: Claude Code would send it instead of its own login
    assert "ANTHROPIC_AUTH_TOKEN" not in data["env"] and "ANTHROPIC_CUSTOM_HEADERS" in data["env"]
    (home / CLAUDE).write_text(json.dumps(data | {"theme": "dark"}))              # the user edits the file
    engine.disconnect(home)
    assert json.loads((home / CLAUDE).read_text()) == original | {"theme": "dark"}   # every key of ours is gone


def test_a_switch_to_own_key_leaves_a_token_the_user_set_alone(home):
    (home / CLAUDE).write_text(json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "sk-ant-oat-theirs"}}))
    engine.connect(home, [claude_code.plan(target(home, claude_code="1", **OWN))])
    assert env_of(home)["ANTHROPIC_AUTH_TOKEN"] == "sk-ant-oat-theirs"   # not a broker key: theirs to keep


def test_switching_back_and_forth_then_disconnect_restores_the_file(home):
    original = (home / CLAUDE).read_bytes()
    for opts in ({}, OWN, {}, OWN):
        engine.connect(home, [claude_code.plan(target(home, claude_code="1", **opts))])
    engine.disconnect(home)
    assert (home / CLAUDE).read_bytes() == original


def test_the_broker_key_cannot_inject_a_header_line(home):
    with pytest.raises(ValueError, match="characters"):
        core.Target(URL, "gbk_x\nX-Evil: 1", "m", home)


# ---- Codex ----------------------------------------------------------------------------

def codex_env(h, **extra):
    return core.Target(URL, "gbk_testkey", "llama-8b", h, {"HOME": str(h), **extra}, OWN)


def test_codex_keeps_its_openai_key_and_adds_the_broker_header(home):
    p = codex.plan(codex_env(home, OPENAI_API_KEY=OPENAI_KEY))
    prof = tomllib.loads(p.files[0].new.decode())["model_providers"]["gpu-broker"]
    assert prof["env_key"] == "OPENAI_API_KEY" and prof["http_headers"] == {"x-gpu-broker-key": "gbk_testkey"}
    assert "experimental_bearer_token" not in prof and OPENAI_KEY not in p.files[0].new.decode()   # the key stays in the env
    assert p.keys == upstreams.DUAL


@pytest.mark.parametrize(("extra", "opts", "why"), [
    ({}, OWN, "OPENAI_API_KEY is not set"),
    ({"OPENAI_API_KEY": "gbk_" + "A" * 43}, OWN, "OPENAI_API_KEY is not set"),   # a broker key is not the user's own
    ({"OPENAI_API_KEY": OPENAI_KEY}, {}, "does not pass OpenAI keys"),
])
def test_codex_falls_back_to_the_broker_key_and_says_why(home, extra, opts, why):
    t = core.Target(URL, "gbk_testkey", "llama-8b", home, {"HOME": str(home), **extra}, opts)
    p = codex.plan(t)
    prof = tomllib.loads(p.files[0].new.decode())["model_providers"]["gpu-broker"]
    assert prof["experimental_bearer_token"] == "gbk_testkey" and "http_headers" not in prof and "env_key" not in prof
    assert p.keys == upstreams.SINGLE and why in p.notes[0]


# ---- the command line -----------------------------------------------------------------

def test_connect_asks_the_broker_and_clients_shows_each_mode(home):
    before = snapshot(home)
    fb = Cloudy(BOTH)
    rc, _ = run(home, "connect", "--url", URL, "--claude-code", call=fb,
                  env={"BROKER_TOKEN": TOKEN, "OPENAI_API_KEY": OPENAI_KEY})
    assert rc == 0 and ("GET", f"{URL}{core.PASSTHROUGH_PATH}", "gbk_issued1", None) in fb.calls   # asked with the client key
    assert "ANTHROPIC_CUSTOM_HEADERS" in env_of(home)
    _, listed = run(home, "clients", "--url", URL, call=fb)
    line = {ln.split()[0]: ln for ln in listed}
    assert f"[keys: {upstreams.DUAL}]" in line["claude-code"] and f"[keys: {upstreams.DUAL}]" in line["codex"]
    assert f"[keys: {upstreams.SEPARATE}]" in line["continue"] and f"[keys: {upstreams.SHELL}]" in line["shell"]
    assert "[keys:" not in line["claude-code-mcp"]
    run(home, "disconnect", "--url", URL, call=fb)
    assert without_state(snapshot(home)) == without_state(before)


def test_a_broker_without_failover_keeps_todays_setup(home):
    rc, _ = run(home, "connect", "--url", URL, "--claude-code", call=FakeBroker(),
                env={"BROKER_TOKEN": TOKEN, "OPENAI_API_KEY": OPENAI_KEY})
    assert rc == 0 and env_of(home)["ANTHROPIC_AUTH_TOKEN"] == "gbk_issued1"
    _, listed = run(home, "clients", "--url", URL)
    assert f"[keys: {upstreams.SINGLE}]" in next(ln for ln in listed if ln.startswith("claude-code "))


def test_every_model_client_says_what_it_sends():
    model_clients = {n for n in CLIENTS if not n.endswith("-mcp") and n != "claude-desktop"}
    assert {n for n, m in CLIENTS.items() if getattr(m, "KEY_MODE", "")} == model_clients

