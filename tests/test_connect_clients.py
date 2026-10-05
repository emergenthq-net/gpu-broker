"""Connectors against a temporary HOME holding real-shaped config files for every client:
connect, then disconnect, restores every file byte for byte; a dry run writes nothing;
connecting twice changes nothing more; a file edited since connect loses only our entries."""
from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest

from gpu_broker.connect import CLIENTS, edits, engine, state
from gpu_broker.connect.core import Target
from tests.helpers import FIX

URL, KEY, MODEL = "http://gpu-host:8095", "gbk_testkey", "llama-8b"
ROO_EXT = Path(".vscode") / "extensions" / "rooveterinaryinc.roo-cline-3.53.0"
VSCODE = Path(".config") / "Code" / "User" / "settings.json"


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    shutil.copytree(FIX / "connect_home", h)
    (h / ROO_EXT).mkdir(parents=True)
    return h


def snapshot(h: Path) -> dict[str, bytes]:
    return {str(p.relative_to(h)): p.read_bytes() for p in sorted(h.rglob("*")) if p.is_file()}


def target(h: Path, **opts: str) -> Target:
    return Target(URL, KEY, MODEL, h, {"HOME": str(h), "SHELL": "/bin/zsh"}, opts)


def plans(h: Path, **opts: str):
    return [m.plan(target(h, **opts)) for m in CLIENTS.values()]


def without_state(snap: dict[str, bytes]) -> dict[str, bytes]:
    return {k: v for k, v in snap.items() if not k.startswith(str(engine.STATE_DIR))}


def test_connect_then_disconnect_restores_every_file_exactly(home):
    before = snapshot(home)
    engine.connect(home, plans(home, claude_code="1"))
    after = snapshot(home)
    assert after != before
    zsh = (home / ".zshrc").read_text()
    assert f"export OPENAI_BASE_URL='{URL}/v1'" in zsh and f"export ANTHROPIC_API_KEY='{KEY}'" in zsh
    assert zsh.startswith(before[".zshrc"].decode())                              # appended, nothing else touched
    assert f"set -gx OPENAI_API_BASE '{URL}/v1'" in (home / ".config/fish/config.fish").read_text()
    assert (home / ".continue/config.yaml").read_bytes() == before[".continue/config.yaml"]   # never edited
    cline = json.loads((home / ".cline/data/settings/providers.json").read_text())
    assert cline["lastUsedProvider"] == "openai-compatible" and "anthropic" in cline["providers"]
    assert cline["providers"]["openai-compatible"]["settings"]["baseUrl"] == f"{URL}/v1"
    claude = json.loads((home / ".claude/settings.json").read_text())
    assert claude == {"env": {"FOO": "1", "ANTHROPIC_BASE_URL": URL, "ANTHROPIC_AUTH_TOKEN": KEY, "ENABLE_TOOL_SEARCH": "true"},
                      "model": "opus"}
    assert (home / ".codex/config.toml").read_bytes() == before[".codex/config.toml"]   # never edited
    codex = tomllib.loads((home / ".codex/gpu-broker.config.toml").read_text())
    assert codex == {"model": MODEL, "model_provider": "gpu-broker", "model_providers": {"gpu-broker": {
        "name": "gpu-broker", "base_url": f"{URL}/v1", "experimental_bearer_token": KEY, "wire_api": "responses"}}}
    assert (home / ".codex/gpu-broker.config.toml").stat().st_mode & 0o777 == 0o600
    engine.disconnect(home)
    assert without_state(snapshot(home)) == before                               # byte-identical, created files gone
    assert not (home / engine.STATE_DIR / engine.MANIFEST).exists()


def test_dry_run_writes_nothing(home):
    before = snapshot(home)
    report = engine.connect(home, plans(home, claude_code="1"), dry=True)
    assert snapshot(home) == before
    assert any(line.startswith("shell: would update") for line in report)
    engine.connect(home, plans(home))
    after = snapshot(home)
    engine.disconnect(home, dry=True)
    assert snapshot(home) == after


def test_connecting_again_changes_nothing_and_keeps_the_first_backup(home):
    before = snapshot(home)
    engine.connect(home, plans(home))
    once = snapshot(home)
    report = engine.connect(home, plans(home))
    assert snapshot(home) == once and all("already connected" in r or "skipped" in r or "note" in r for r in report)
    engine.disconnect(home)
    assert without_state(snapshot(home)) == before


def test_a_file_edited_since_connect_loses_only_our_entries(home):
    engine.connect(home, plans(home))
    zshrc, settings = home / ".zshrc", home / VSCODE
    zshrc.write_text(zshrc.read_text() + "alias g=git\n")
    data = json.loads(settings.read_text())
    settings.write_text(json.dumps({**data, "editor.tabSize": 2}, indent=4) + "\n")
    report = engine.disconnect(home)
    assert "OPENAI_BASE_URL" not in zshrc.read_text() and zshrc.read_text().endswith("alias g=git\n")
    assert json.loads(settings.read_text()) == {"editor.fontSize": 14, "editor.tabSize": 2}
    assert sum("removed only our entries" in r for r in report) == 2


def test_disconnect_one_client_leaves_the_others(home):
    engine.connect(home, plans(home))
    engine.disconnect(home, {"shell"})
    assert "OPENAI_BASE_URL" not in (home / ".zshrc").read_text()
    assert (home / ".continue/models/gpu-broker.yaml").exists()
    assert engine.connected(home) == {"continue", "cline", "roo", "codex"}


def test_skips_say_why():
    t = Target(URL, KEY, MODEL, Path("/nonexistent-home"), {})
    for name, m in CLIENTS.items():
        p = m.plan(t)
        assert p.skip and not p.files and p.api is None, name


def test_claude_code_needs_the_opt_in(home):
    p = CLIENTS["claude-code"].plan(target(home))
    assert p.skip and "--claude-code" in p.skip


def test_json_with_comments_is_not_edited(home):
    (home / VSCODE).write_text('{\n  // my font\n  "editor.fontSize": 14\n}\n')
    p = CLIENTS["roo"].plan(target(home))
    assert p.skip and "comments" in p.skip


def test_login_shell_rc_is_created_and_then_deleted(tmp_path):
    h = tmp_path / "h"
    h.mkdir()
    t = Target(URL, KEY, MODEL, h, {"SHELL": "/usr/bin/fish"})
    engine.connect(h, [CLIENTS["shell"].plan(t)])
    rc = h / ".config/fish/config.fish"
    assert rc.exists() and rc.stat().st_mode & 0o777 == 0o600
    engine.disconnect(h)
    assert not rc.exists()


def test_backups_are_private(home):
    engine.connect(home, plans(home))
    backups = home / engine.STATE_DIR / state.BACKUPS
    assert backups.stat().st_mode & 0o777 == 0o700 and any(backups.iterdir())
    assert (home / engine.STATE_DIR / engine.MANIFEST).stat().st_mode & 0o777 == 0o600


def test_block_edits_round_trip():
    for text in ("", "a\n", "a", "a\n\nb\n"):
        assert edits.strip_block(edits.put_block(text, ["x=1"])) in (text, text + "\n")
    twice = edits.put_block(edits.put_block("a\n", ["x=1"]), ["x=2"])
    assert twice.count(edits.BEGIN) == 1 and "x=2" in twice and "x=1" not in twice
    with pytest.raises(edits.Unsupported):
        edits.put_block(f"# {edits.BEGIN}\nno end\n", ["x"])


def test_reconnecting_with_a_new_key_still_restores_the_original(home):
    before = snapshot(home)
    engine.connect(home, plans(home, claude_code="1"))
    second = [m.plan(Target(URL, "gbk_other", MODEL, home, {"SHELL": "/bin/zsh"}, {"claude_code": "1"})) for m in CLIENTS.values()]
    engine.connect(home, second)
    assert "OPENAI_API_KEY='gbk_other'" in (home / ".zshrc").read_text()
    engine.disconnect(home)
    assert without_state(snapshot(home)) == before


def test_json_undo_restores_old_values_and_drops_objects_we_created(home):
    settings = home / ".claude/settings.json"
    settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://old"}, "model": "opus"}, indent=2) + "\n")
    engine.connect(home, [CLIENTS["claude-code"].plan(target(home, claude_code="1"))])
    settings.write_text(settings.read_text().replace('"opus"', '"sonnet"'))   # the user edits it since
    engine.disconnect(home)
    assert json.loads(settings.read_text()) == {"env": {"ANTHROPIC_BASE_URL": "https://old"}, "model": "sonnet"}
    settings.write_text('{\n  "model": "opus"\n}\n')
    engine.connect(home, [CLIENTS["claude-code"].plan(target(home, claude_code="1"))])
    settings.write_text(settings.read_text().replace('"opus"', '"sonnet"'))
    engine.disconnect(home)
    assert json.loads(settings.read_text()) == {"model": "sonnet"}               # the env object we created is gone


def test_edits_keep_the_files_own_style(home):
    engine.connect(home, plans(home))
    assert '\n    "roo-cline.autoImportSettingsPath"' in (home / VSCODE).read_text()   # 4-space file stays 4-space
    zsh = (home / ".zshrc").read_text()
    assert f"export GPU_BROKER_URL='{URL}'\n" in zsh and f"export OPENAI_API_BASE='{URL}/v1'\n" in zsh


def test_cline_is_not_rewritten_when_nothing_changed(home, monkeypatch):
    from gpu_broker.connect import cline
    engine.connect(home, [CLIENTS["cline"].plan(target(home))])
    first = (home / ".cline/data/settings/providers.json").read_bytes()
    monkeypatch.setattr(cline.time, "strftime", lambda *a: "2099-01-01T00:00:00.000Z")
    engine.connect(home, [CLIENTS["cline"].plan(target(home))])
    assert (home / ".cline/data/settings/providers.json").read_bytes() == first


def test_a_dry_run_never_writes_the_manifest(home):
    engine.connect(home, [], dry=True, key={"broker": URL, "id": "k", "name": "n", "key": KEY})
    assert not (home / engine.STATE_DIR).exists()


def test_cline_file_is_kept_at_600_and_checked_whole(home):
    providers = home / ".cline/data/settings/providers.json"
    providers.chmod(0o644)
    engine.connect(home, [CLIENTS["cline"].plan(target(home))])
    assert providers.stat().st_mode & 0o777 == 0o600
    engine.disconnect(home)
    assert providers.stat().st_mode & 0o777 == 0o644                 # disconnect restores the original mode too
    data = json.loads(providers.read_text())
    data["providers"]["anthropic"]["updatedAt"] = "yesterday"
    providers.write_text(json.dumps(data))
    p = CLIENTS["cline"].plan(target(home))
    assert p.skip and "not an ISO time" in p.skip and not p.files
    data["providers"]["anthropic"]["updatedAt"] = "2026-09-01T00:00:00.000Z"
    data["providers"]["anthropic"]["settings"]["baseUrl"] = "not a url"
    providers.write_text(json.dumps(data))
    assert "not an http(s) URL" in (CLIENTS["cline"].plan(target(home)).skip or "")
    data["providers"]["anthropic"]["settings"].pop("baseUrl")
    providers.write_text(json.dumps(data))
    assert not CLIENTS["cline"].plan(target(home)).skip
    for url in ("gpu-host:8095", "http://"):
        with pytest.raises(ValueError, match="not an http"):   # refused before any connector sees it
            Target(url, KEY, MODEL, home, {}, {})


def test_claude_code_keeps_mcp_tool_search_on(home):
    engine.connect(home, [CLIENTS["claude-code"].plan(target(home, claude_code="1"))])
    assert json.loads((home / ".claude/settings.json").read_text())["env"]["ENABLE_TOOL_SEARCH"] == "true"
