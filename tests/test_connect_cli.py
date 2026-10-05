"""`gpu-broker connect | disconnect | clients`: key issuing and reuse, model discovery, Open WebUI
through its API, --only, --dry-run, --revoke, and the installer zipapp run with a real python3."""
from __future__ import annotations

import json
import subprocess
import sys

from gpu_broker.cli import main as broker_main
from gpu_broker.connect import cli, engine
from tests.test_connect_clients import URL, home, snapshot, without_state  # noqa: F401

TOKEN = "main-token"


class FakeBroker:
    """The broker's /v1/keys, /v1/status and /v1/models, recorded."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, object]] = []
        self.issued = 0

    def __call__(self, method, url, credential, body):
        self.calls.append((method, url, credential, body))
        if url.endswith("/v1/keys") and method == "POST":
            self.issued += 1
            return {"id": f"k{self.issued}", "name": body["name"], "key": f"gbk_issued{self.issued}"}
        if url.endswith("/v1/status"):
            return {"resident_llm": "llama-8b"}
        return {"revoked": True}


class FakeWebui:
    def __init__(self) -> None:
        self.config = {"ENABLE_OPENAI_API": True, "OPENAI_API_BASE_URLS": ["https://api.openai.com/v1"],
                       "OPENAI_API_KEYS": ["sk-real"], "OPENAI_API_CONFIGS": {"0": {"enable": True}}}
        self.posts: list[dict] = []

    def __call__(self, method, url, token, body):
        assert token == "owui-admin"
        if method == "GET":
            return json.loads(json.dumps(self.config))
        self.posts.append(body)
        self.config = body
        return body


def run(home, *args, call=None, webui=None, env=None):
    out: list[str] = []
    rc = cli.main(list(args), env={"HOME": str(home), "SHELL": "/bin/zsh", **(env or {})}, home=home,
                  call=call or FakeBroker(), webui=webui or FakeWebui(), out=out.append)
    return rc, out


def test_connect_issues_a_key_named_for_the_machine_and_reuses_it(home):
    fb = FakeBroker()
    run(home, "connect", "--url", URL, "--name", "laptop", call=fb, env={"BROKER_TOKEN": TOKEN})
    assert fb.calls[0] == ("POST", f"{URL}/v1/keys", TOKEN, {"name": "laptop"})
    assert "export OPENAI_API_KEY='gbk_issued1'" in (home / ".zshrc").read_text()
    assert ("GET", f"{URL}/v1/status", TOKEN, None) in fb.calls   # /v1/status is main-token only now
    assert "model: \"llama-8b\"" in (home / ".continue/models/gpu-broker.yaml").read_text()
    run(home, "connect", "--url", URL, call=fb, env={"BROKER_TOKEN": TOKEN})
    assert fb.issued == 1                                                     # second connect reuses the key


def test_connect_without_key_or_token_refuses(home):
    try:
        run(home, "connect", "--url", URL)
    except SystemExit as e:
        assert "--key" in str(e)
    else:
        raise AssertionError("expected SystemExit")


def test_dry_run_needs_no_key_and_issues_none(home):
    fb = FakeBroker()
    rc, out = run(home, "connect", "--url", URL, "--model", "m", "--dry-run", call=fb)
    assert rc == 0 and fb.issued == 0 and any("would update" in line for line in out)


def test_only_and_dry_run(home):
    before = snapshot(home)
    rc, out = run(home, "connect", "--url", URL, "--key", "gbk_x", "--model", "m", "--only", "shell", "--dry-run")
    assert rc == 0 and snapshot(home) == before
    assert [line.split(":")[0] for line in out] == ["shell"] * len(out)
    rc, out = run(home, "connect", "--only", "nope")
    assert rc == 1 and "unknown clients: nope" in out[0]


def test_disconnect_revokes_the_issued_key_and_restores(home):
    before = snapshot(home)
    fb = FakeBroker()
    run(home, "connect", "--url", URL, call=fb, env={"BROKER_TOKEN": TOKEN})
    run(home, "disconnect", "--revoke", call=fb, env={"BROKER_TOKEN": TOKEN})
    assert ("DELETE", f"{URL}/v1/keys/k1", TOKEN, None) in fb.calls
    assert without_state(snapshot(home)) == before


def test_openwebui_connection_added_then_restored(home):
    ui = FakeWebui()
    original = json.loads(json.dumps(ui.config))
    opts = ["--openwebui-url", "http://owui:3000", "--openwebui-token", "owui-admin"]
    run(home, "connect", "--url", URL, "--key", "gbk_x", "--model", "m", "--only", "openwebui", *opts, webui=ui)
    run(home, "connect", "--url", URL, "--key", "gbk_x", "--model", "m", "--only", "openwebui", *opts, webui=ui)   # again
    sent = ui.posts[-1]
    assert sent["OPENAI_API_BASE_URLS"] == ["https://api.openai.com/v1", f"{URL}/v1"]   # added once, not twice
    assert sent["OPENAI_API_KEYS"] == ["sk-real", "gbk_x"] and sent["OPENAI_API_CONFIGS"]["0"] == {"enable": True}
    assert engine.connected(home) == {"openwebui"}
    run(home, "disconnect", *opts, webui=ui)
    assert ui.config == original and engine.connected(home) == set()


def test_clients_lists_what_is_installed(home):
    rc, out = run(home, "clients")
    assert rc == 0
    rows = {line.split()[0]: line for line in out}
    assert "found" in rows["cline"] and "not found" in rows["openwebui"] and "found" in rows["codex"]


def test_the_broker_cli_hands_connect_commands_over(home, monkeypatch):
    monkeypatch.setenv("HOME", str(home))
    assert broker_main(["clients", "--only", "shell"], env={"HOME": str(home)}) == 0


def test_the_zipapp_runs_with_a_plain_python(home, tmp_path):
    """What connect.sh downloads: the connect package alone, run by a separate python3."""
    from gpu_broker.web.connect import pyz
    app = tmp_path / "c.pyz"
    app.write_bytes(pyz())
    r = subprocess.run([sys.executable, "-I", str(app), "clients", "--only", "shell,cline"], capture_output=True, text=True,
                       env={"HOME": str(home), "SHELL": "/bin/zsh"}, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "shell" in r.stdout and "cline" in r.stdout
