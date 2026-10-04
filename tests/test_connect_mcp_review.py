"""connect and the MCP clients, review of #27: a `gpu-broker` server the user defined is never
replaced (only connect's own entry is), a symlinked config is written through, a running
Claude Code that drops the entry is reported, and connect says which files it changed."""
import json
from pathlib import Path

from gpu_broker.connect import claude_code_mcp, claude_desktop, engine
from tests.test_connect_cli import FakeBroker, run
from tests.test_connect_clients import URL, home, snapshot, without_state
from tests.test_connect_mcp import desktop, served

__all__ = ["desktop", "home"]
THEIRS = {"command": "my-own-gpu-broker-wrapper"}


def entry(path):
    return json.loads(path.read_text())["mcpServers"]["gpu-broker"]


def test_a_user_defined_server_is_left_alone(home, desktop):
    for path in (home / ".claude.json", desktop):
        path.write_text(json.dumps({"mcpServers": {"gpu-broker": THEIRS}}))
    report = engine.connect(home, [claude_code_mcp.plan(served(home)), claude_desktop.plan(served(home))])
    assert all("defines its own 'gpu-broker' MCP server" in line for line in report), report
    assert entry(home / ".claude.json") == THEIRS and entry(desktop) == THEIRS


def test_connects_own_entry_is_replaced_on_reconnect(home, desktop):
    engine.connect(home, [claude_code_mcp.plan(served(home)), claude_desktop.plan(served(home))])
    first = served(home)
    again = type(first)(first.url, "gbk_otherkey", first.model, home, first.env, first.options)
    engine.connect(home, [claude_code_mcp.plan(again), claude_desktop.plan(again)])
    assert entry(home / ".claude.json")["headers"]["Authorization"] == "Bearer gbk_otherkey"
    assert entry(desktop)["env"]["GPU_BROKER_API_KEY"] == "gbk_otherkey"


def test_a_symlinked_config_is_written_through_and_restored_through(home):
    real = home / "dotfiles" / "claude.json"
    real.parent.mkdir()
    real.write_text('{"theme": "dark"}\n')
    link = home / ".claude.json"
    link.symlink_to(real)
    engine.connect(home, [claude_code_mcp.plan(served(home))])
    assert link.is_symlink() and link.resolve() == real.resolve()
    assert entry(real)["type"] == "http" and json.loads(real.read_text())["theme"] == "dark"
    engine.disconnect(home)
    assert link.is_symlink() and real.read_text() == '{"theme": "dark"}\n'


def test_a_running_claude_code_that_drops_the_entry_is_reported(home):
    path = home / ".claude.json"

    def claude_rewrites(_s):   # Claude Code writing its in-memory copy back over ours
        path.write_text('{"numStartups": 9}\n')
    report = engine.connect(home, [claude_code_mcp.plan(served(home))], settle_s=1.5, sleep=claude_rewrites)
    assert any(line.startswith("claude-code-mcp: warning:") and "Quit Claude Code" in line for line in report), report
    quiet = engine.connect(home, [claude_code_mcp.plan(served(home))], sleep=lambda s: None)
    assert not any("warning" in line for line in quiet)


def test_connect_prints_exactly_which_files_changed(home, desktop):
    before = without_state(snapshot(home))
    rc, out = run(home, "connect", "--url", URL, "--key", "gbk_x", "--model", "m", "--only", "claude-code-mcp,shell",
                  call=McpBroker())
    assert rc == 0 and out[-1].startswith("files changed: "), out
    listed = [Path(p).resolve() for p in out[-1].removeprefix("files changed: ").split(", ")]
    after = without_state(snapshot(home))
    differ = {(home / k).resolve() for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert set(listed) == differ and len(listed) == len(differ) and (home / ".claude.json").resolve() in differ
    rc, out = run(home, "connect", "--url", URL, "--key", "gbk_x", "--model", "m", "--only", "claude-code-mcp,shell",
                  call=McpBroker())
    assert out[-1] == "files changed: none"


class McpBroker(FakeBroker):
    """A broker whose /health says it serves /mcp."""

    def __call__(self, method, url, credential, body):
        if url.endswith("/health"):
            return {"ok": True, "mcp": True}
        return super().__call__(method, url, credential, body)
