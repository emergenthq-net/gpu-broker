"""/v1/responses with client keys, and the Codex connector.

- a client key (MODEL scope) can call /v1/responses on either header; one key cannot continue
  another key's stored response, nor can the main token pose as that key with x-requester;
- the Codex connector adds a `gpu-broker` provider (wire_api = "responses") and profile as
  one block, keeps the user's own settings and the file private, survives reconnects, and
  leaves alone a file it cannot edit safely.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

from gpu_broker.connect import codex, engine
from gpu_broker.connect.core import Target
from gpu_broker.constants import REQUESTER_HEADER
from tests.test_connect_clients import KEY, MODEL, URL, home, snapshot, without_state  # noqa: F401
from tests.test_connect_web import MAIN, issue, plain_broker, web  # noqa: F401

EP = "/v1/responses"
HI = {"model": "gpt-4o", "input": "hi"}


def bearer(k):
    return {"Authorization": f"Bearer {k['key']}"}


def test_a_client_key_can_call_responses(web):
    k = issue(web)
    for h in (bearer(k), {"x-api-key": k["key"]}):
        r = web.post(EP, json=HI, headers=h)
        assert r.status_code == 200 and r.json()["object"] == "response"


def test_a_client_key_cannot_continue_another_keys_response(web):
    a, b = issue(web, "alpha"), issue(web, "beta")
    first = web.post(EP, json=HI, headers=bearer(a)).json()
    follow = {**HI, "input": "more", "previous_response_id": first["id"]}
    for h in (bearer(b), {**MAIN, REQUESTER_HEADER: "alpha"}, {**bearer(b), REQUESTER_HEADER: "alpha"}):
        r = web.post(EP, json=follow, headers=h)
        assert r.status_code == 400 and r.json()["error"]["code"] == "previous_response_not_found"
    assert web.post(EP, json=follow, headers=bearer(a)).status_code == 200


def test_a_revoked_key_loses_responses(web):
    k = issue(web)
    web.delete(f"/v1/keys/{k['id']}", headers=MAIN)
    assert web.post(EP, json=HI, headers=bearer(k)).status_code == 401


# ---- the connector -------------------------------------------------------------
PROFILE = Path(".codex/gpu-broker.config.toml")


def target(h: Path, **env: str) -> Target:
    return Target(URL, KEY, MODEL, h, {"HOME": str(h), **env})


def test_codex_gets_a_responses_profile_and_config_toml_is_untouched(home):
    before = snapshot(home)
    engine.connect(home, [codex.plan(target(home))])
    cfg = tomllib.loads((home / PROFILE).read_text())
    assert cfg["model_provider"] == "gpu-broker" and cfg["model"] == MODEL
    assert cfg["model_providers"]["gpu-broker"]["wire_api"] == "responses"
    assert cfg["model_providers"]["gpu-broker"]["base_url"] == f"{URL}/v1"
    assert os.stat(home / PROFILE).st_mode & 0o777 == 0o600
    assert (home / ".codex/config.toml").read_bytes() == before[".codex/config.toml"]
    engine.connect(home, [codex.plan(target(home))])     # reconnecting rewrites our own file
    engine.disconnect(home)
    assert without_state(snapshot(home)) == before       # the profile is gone again


def test_codex_home_moves_the_folder(home, tmp_path):
    elsewhere = tmp_path / "codex-home"
    elsewhere.mkdir()
    p = codex.plan(target(home, CODEX_HOME=str(elsewhere)))
    assert [f.path for f in p.files] == [elsewhere / "gpu-broker.config.toml"]


def test_no_config_toml_is_fine(home):
    (home / ".codex/config.toml").unlink()
    assert codex.plan(target(home)).files


@pytest.mark.parametrize(("config", "why"), [
    ("model = \n", "not valid TOML"),
    ('[model_providers.gpu-broker]\nbase_url = "http://mine"\n', "its own 'gpu-broker' provider"),
    ('[profiles.gpu-broker]\nmodel = "x"\n', "legacy 'gpu-broker' profile"),
    ('profile = "gpu-broker"\n', "legacy 'gpu-broker' profile"),
])
def test_a_config_it_would_clash_with_is_left_alone(home, config, why):
    (home / ".codex/config.toml").write_text(config)
    p = codex.plan(target(home))
    assert p.skip and why in p.skip and not p.files


def test_a_profile_file_someone_else_wrote_is_left_alone(home):
    (home / PROFILE).write_text('model = "mine"\n')
    p = codex.plan(target(home))
    assert p.skip and "not written by gpu-broker connect" in p.skip


def test_a_reconnected_profile_is_made_private_again(home):
    engine.connect(home, [codex.plan(target(home))])
    os.chmod(home / PROFILE, 0o644)
    engine.connect(home, [codex.plan(target(home))])
    assert os.stat(home / PROFILE).st_mode & 0o777 == 0o600


def test_disconnect_says_removed_for_the_profile_it_created(home):
    engine.connect(home, [codex.plan(target(home))])
    assert engine.disconnect(home, dry=True) == [f"codex: would remove {home / PROFILE}"]
    assert engine.disconnect(home) == [f"codex: removed {home / PROFILE}"]


def test_a_restored_file_still_says_restored(home):
    from gpu_broker.connect import shell
    engine.connect(home, [shell.plan(target(home, SHELL="/bin/zsh"))])
    assert f"shell: restored {home / '.zshrc'}" in engine.disconnect(home)
