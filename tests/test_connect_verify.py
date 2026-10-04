"""Verification round on connect (PR #16): an issued key is never orphaned, a user's own Open
WebUI connection is never touched, key-bearing temp files are private from the first byte,
and a key needs a real body after its prefix."""
from __future__ import annotations

import json
import os
import stat

from gpu_broker.connect import engine, shell, state
from gpu_broker.connect.core import KEY_BODY_LEN, KEY_PREFIX, Target, is_client_key
from gpu_broker.keys import KeyStore
from tests.test_connect_cli import TOKEN, FakeBroker, FakeWebui, run
from tests.test_connect_clients import KEY, URL, home, snapshot  # noqa: F401

ENV = {"BROKER_TOKEN": TOKEN}
OWUI = ["--openwebui-url", "http://owui:3000", "--openwebui-token", "owui-admin"]


class NoModel(FakeBroker):
    """Issues a key, then cannot say which model to use (so connect stops before any connector)."""

    def __init__(self, revoke_fails: bool = False) -> None:
        super().__init__()
        self.revoke_fails = revoke_fails

    def __call__(self, method, url, credential, body):
        if url.endswith(("/v1/status", "/v1/models")):
            self.calls.append((method, url, credential, body))
            raise OSError("broker went away")
        if method == "DELETE" and self.revoke_fails:
            self.calls.append((method, url, credential, body))
            raise OSError("still away")
        return super().__call__(method, url, credential, body)


def test_a_key_issued_before_a_failure_is_revoked_and_forgotten(home):
    fb, before = NoModel(), snapshot(home)
    rc, out = run(home, "connect", "--url", URL, call=fb, env=ENV)
    assert rc == 1 and any("--model" in o for o in out) and any("revoked the key" in o for o in out)
    assert ("DELETE", f"{URL}/v1/keys/k1", TOKEN, None) in fb.calls
    assert "key" not in engine.load_manifest(home) and snapshot(home) == before


def test_a_key_that_cannot_be_revoked_stays_recorded_with_instructions(home):
    fb = NoModel(revoke_fails=True)
    rc, out = run(home, "connect", "--url", URL, call=fb, env=ENV)   # no traceback: run() would raise
    assert rc == 1 and ("DELETE", f"{URL}/v1/keys/k1", TOKEN, None) in fb.calls
    assert engine.load_manifest(home)["key"]["id"] == "k1"
    assert any("still recorded" in o and "disconnect --revoke" in o for o in out)
    ok = FakeBroker()   # the advice works: disconnect --revoke takes it out
    assert run(home, "disconnect", "--revoke", call=ok, env=ENV)[0] == 0
    assert ("DELETE", f"{URL}/v1/keys/k1", TOKEN, None) in ok.calls and "key" not in engine.load_manifest(home)


def test_the_key_is_recorded_before_the_model_is_asked_for(home):
    seen = {}

    class Peek(FakeBroker):
        def __call__(self, method, url, credential, body):
            if url.endswith("/v1/status"):
                seen["recorded"] = engine.load_manifest(home).get("key", {}).get("id")
            return super().__call__(method, url, credential, body)
    run(home, "connect", "--url", URL, call=Peek(), env=ENV)
    assert seen["recorded"] == "k1"


def test_a_reused_key_is_not_revoked_when_connect_fails(home):
    run(home, "connect", "--url", URL, call=FakeBroker(), env=ENV)
    fb = NoModel()
    rc, _ = run(home, "connect", "--url", URL, call=fb, env=ENV)
    assert rc == 1 and not any(c[0] == "DELETE" for c in fb.calls) and engine.load_manifest(home)["key"]["id"] == "k1"


def test_a_users_own_openwebui_connection_is_left_alone_and_not_recorded(home):
    ui = FakeWebui()
    ui.config["OPENAI_API_BASE_URLS"].append(f"{URL}/v1")
    ui.config["OPENAI_API_KEYS"].append("users-own")
    ui.config["OPENAI_API_CONFIGS"]["1"] = {"enable": False, "name": "mine"}
    before = json.loads(json.dumps(ui.config))
    _, out = run(home, "connect", "--url", URL, "--key", KEY, "--model", "m", "--only", "openwebui", *OWUI, webui=ui)
    assert ui.posts == [] and ui.config == before and any("configured by you" in o for o in out)
    assert "openwebui" not in engine.connected(home)


def test_our_own_openwebui_connection_is_refreshed_on_reconnect(home):
    ui = FakeWebui()
    args = ["connect", "--url", URL, "--model", "m", "--only", "openwebui", *OWUI]
    run(home, *args, "--key", KEY, webui=ui)
    run(home, *args, "--key", KEY + "x", webui=ui)
    assert len(ui.posts) == 2 and ui.config["OPENAI_API_KEYS"][-1] == KEY + "x"


def test_temp_files_are_private_while_written(tmp_path, monkeypatch):
    modes = []
    real = os.fdopen

    def spy(fd, *a, **k):
        modes.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return real(fd, *a, **k)
    monkeypatch.setattr(state.os, "fdopen", spy)
    old = os.umask(0)                            # even with an open umask
    try:
        target = tmp_path / "rc"
        target.write_text("x")
        os.chmod(target, 0o644)
        state.write(target, b"KEY=secret", 0o600)
        state.backup(tmp_path, target, b"KEY=secret")
    finally:
        os.umask(old)
    assert modes == [0o600, 0o600]
    assert stat.S_IMODE(target.stat().st_mode) == 0o644   # the file itself keeps its own mode


def test_a_key_needs_its_whole_body():
    real = KeyStore(":memory:").issue("x")["key"]
    assert is_client_key(real) and len(real) == len(KEY_PREFIX) + KEY_BODY_LEN
    for bad in (KEY_PREFIX, KEY_PREFIX + "short", real + "x", real[:-1] + "$"):
        assert not is_client_key(bad)
    ks = KeyStore(":memory:")
    assert ks.check(KEY_PREFIX) is None


def test_a_bare_prefix_in_the_environment_is_someone_elses(home):
    t = Target(URL, KEY, "m", home, {"SHELL": "/bin/zsh", "OPENAI_API_KEY": KEY_PREFIX})
    assert any("already set" in n for n in shell.plan(t).notes)
