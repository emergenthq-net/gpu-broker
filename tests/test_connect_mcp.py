"""connect registers the broker's MCP tools with Claude Code (~/.claude.json), Claude Desktop
(claude_desktop_config.json, a `gpu-broker mcp` command) and Codex (config.toml), with the
backup/manifest machinery: disconnect restores byte for byte, connecting twice changes
nothing, an edited file loses only our entry. Nothing is registered when the broker does not
serve /mcp, or (Claude Desktop) when this machine cannot run `gpu-broker mcp`."""
from __future__ import annotations

import json
import stat
import tomllib

import pytest

from gpu_broker.connect import CLIENTS, claude_desktop, codex_mcp, engine, mcpinfo
from gpu_broker.connect.core import MCP_COMMAND_OPT, MCP_OPT, Target
from tests.test_connect_clients import KEY, MODEL, URL, home, snapshot, without_state

__all__ = ["home"]
MCP_CLIENTS = ("claude-code-mcp", "claude-desktop", "codex-mcp")
COMMAND = json.dumps(["/opt/py/bin/python", "-m", "gpu_broker", "mcp"])


def target(h, **opts):
    return Target(URL, KEY, MODEL, h, {"HOME": str(h)}, opts)


def served(h, command=COMMAND):
    return target(h, **{MCP_OPT: "1", **({MCP_COMMAND_OPT: command} if command else {})})


@pytest.fixture
def desktop(home):
    d = claude_desktop.folder(target(home))
    d.mkdir(parents=True)
    (d / claude_desktop.FILE).write_text('{\n    "globalShortcut": "Alt+Space"\n}\n')
    return d / claude_desktop.FILE


def plans(t):
    return [CLIENTS[n].plan(t) for n in MCP_CLIENTS]


def test_registered_entries(home, desktop):
    engine.connect(home, plans(served(home)))
    code = json.loads((home / ".claude.json").read_text())["mcpServers"]["gpu-broker"]
    assert code == {"type": "http", "url": f"{URL}/mcp", "headers": {"Authorization": f"Bearer {KEY}"}}
    app = json.loads(desktop.read_text())
    assert app["globalShortcut"] == "Alt+Space"
    assert app["mcpServers"]["gpu-broker"] == {"command": "/opt/py/bin/python", "args": ["-m", "gpu_broker", "mcp", "--url", URL],
                                               "env": {"GPU_BROKER_API_KEY": KEY}}
    toml = tomllib.loads((home / ".codex/config.toml").read_text())
    assert toml["mcp_servers"]["gpu-broker"] == {"url": f"{URL}/mcp", "http_headers": {"Authorization": f"Bearer {KEY}"},
                                                 "tool_timeout_sec": codex_mcp.TOOL_TIMEOUT_S}
    assert toml["model"] == "gpt-5"   # the user's own settings stay
    for p in (home / ".claude.json", desktop, home / ".codex/config.toml"):
        assert stat.S_IMODE(p.stat().st_mode) == 0o600, p   # each now holds the key


def test_disconnect_restores_byte_for_byte_and_twice_is_idempotent(home, desktop):
    before = snapshot(home)
    engine.connect(home, plans(served(home)))
    once = snapshot(home)
    report = engine.connect(home, plans(served(home)))
    assert snapshot(home) == once and all("already connected" in line for line in report if "note:" not in line)
    engine.disconnect(home, set(MCP_CLIENTS))
    assert without_state(snapshot(home)) == before
    assert not (home / ".claude.json").exists()   # it did not exist before: connect created it


def test_an_edited_file_loses_only_our_entry(home, desktop):
    engine.connect(home, plans(served(home)))
    data = json.loads(desktop.read_text())
    data["mcpServers"]["mine"] = {"command": "my-server"}
    desktop.write_text(json.dumps(data))
    toml = home / ".codex/config.toml"
    toml.write_text(toml.read_text() + '\n[mcp_servers.other]\nurl = "http://other/mcp"\n')
    engine.disconnect(home, set(MCP_CLIENTS))
    assert json.loads(desktop.read_text())["mcpServers"] == {"mine": {"command": "my-server"}}
    left = tomllib.loads(toml.read_text())
    assert left["mcp_servers"] == {"other": {"url": "http://other/mcp"}} and "gpu-broker connect" not in toml.read_text()


def test_nothing_is_registered_when_the_broker_does_not_serve_mcp(home, desktop):
    for p in plans(target(home)):
        assert p.skip and "does not serve MCP" in p.skip, p.client


def test_claude_desktop_needs_a_local_gpu_broker_mcp(home, desktop):
    (p,) = [CLIENTS["claude-desktop"].plan(served(home, command=""))]
    assert p.skip and "gpu-broker[mcp]" in p.skip


def test_codex_leaves_a_gpu_broker_server_it_did_not_write(home):
    toml = home / ".codex/config.toml"
    toml.write_text(toml.read_text() + '\n[mcp_servers.gpu-broker]\ncommand = "theirs"\n')
    p = CLIENTS["codex-mcp"].plan(served(home))
    assert p.skip and "defines its own" in p.skip
    toml.write_text("model = \n")
    p = CLIENTS["codex-mcp"].plan(served(home))
    assert p.skip and "not TOML" in p.skip


@pytest.mark.parametrize(("platform", "env", "where"), [
    ("darwin", {}, "Library/Application Support/Claude"),
    ("linux", {}, ".config/Claude"),
    ("linux", {"XDG_CONFIG_HOME": "/x/cfg"}, "/x/cfg/Claude"),
    ("win32", {"APPDATA": "/x/roaming"}, "/x/roaming/Claude"),
])
def test_claude_desktop_config_folder(home, platform, env, where):
    t = Target(URL, KEY, MODEL, home, env, {})
    got = claude_desktop.folder(t, platform)
    assert str(got) == where if where.startswith("/") else got == home / where


def test_options_say_what_the_broker_and_this_machine_can_do(monkeypatch):
    assert mcpinfo.options(False) == {}
    monkeypatch.setattr(mcpinfo, "local_mcp_command", lambda: "")
    assert mcpinfo.options(True) == {MCP_OPT: "1"}
    monkeypatch.setattr(mcpinfo, "local_mcp_command", lambda: COMMAND)
    assert mcpinfo.options(True) == {MCP_OPT: "1", MCP_COMMAND_OPT: COMMAND}
    assert mcpinfo.serves_mcp(URL, lambda *a: {"ok": True, "mcp": True}) is True
    assert mcpinfo.serves_mcp(URL, lambda *a: {"ok": True}) is False

    def down(*a):
        raise OSError("refused")
    assert mcpinfo.serves_mcp(URL, down) is False


def test_this_python_can_run_gpu_broker_mcp():
    argv = json.loads(mcpinfo.local_mcp_command())   # the test venv has the mcp extra
    assert argv[1:] == ["-m", "gpu_broker", "mcp"]


def test_codex_refuses_a_config_it_cannot_extend(home):
    toml = home / ".codex/config.toml"
    toml.write_text('mcp_servers = { other = { url = "http://o/mcp" } }\n')   # an inline table cannot gain a [table]
    p = CLIENTS["codex-mcp"].plan(served(home))
    assert p.skip and "not TOML connect can extend" in p.skip


def test_an_existing_claude_json_becomes_private(home):
    path = home / ".claude.json"
    path.write_text('{"numStartups": 3}\n')
    path.chmod(0o644)
    engine.connect(home, [CLIENTS["claude-code-mcp"].plan(served(home))])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and json.loads(path.read_text())["numStartups"] == 3


def test_connect_asks_the_broker_whether_it_serves_mcp(home):
    from gpu_broker.connect import cli as connect_cli
    from tests.test_connect_cli import FakeBroker

    class Serving(FakeBroker):
        def __call__(self, method, url, credential, body):
            if url.endswith("/health"):
                return {"ok": True, "mcp": self.mcp}
            return super().__call__(method, url, credential, body)
    for mcp, registered in ((True, True), (False, False)):
        broker = Serving()
        broker.mcp = mcp
        out: list[str] = []
        connect_cli.main(["connect", "--url", URL, "--only", "claude-code-mcp"], env={"HOME": str(home), "BROKER_TOKEN": "t"},
                         home=home, call=broker, out=out.append)
        assert ((home / ".claude.json").exists()) is registered, out
        if registered:
            entry = json.loads((home / ".claude.json").read_text())["mcpServers"]["gpu-broker"]
            assert entry["headers"]["Authorization"] == "Bearer gbk_issued1"
            engine.disconnect(home)
