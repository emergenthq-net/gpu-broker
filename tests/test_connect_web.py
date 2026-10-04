"""Client keys and the server side of connect: keys on both SDK headers, admin routes refuse them,
revocation, the dashboard's connect/disconnect on the broker's own machine, the single-use
installer, and the built-in model map with no config."""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from gpu_broker import cli
from gpu_broker.broker import Broker
from gpu_broker.keys import KeyStore
from gpu_broker.web.app import create_app
from tests.dropin_fakes import ToolBackends, catalog_with_embedder
from tests.helpers import TOKEN, FakeDriver, make_settings, wait_idle
from tests.test_connect_clients import home, snapshot, without_state  # noqa: F401

MAIN = {"Authorization": f"Bearer {TOKEN}"}
HI = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def plain_broker(tmp_path):
    """No model_map in the settings: the built-in default applies."""
    driver = FakeDriver({"llama-8b"})
    b = Broker(make_settings(tmp_path, catalog=catalog_with_embedder(tmp_path)), env={}, driver=driver, backends=ToolBackends(driver))
    b.start()
    yield b
    assert wait_idle(b)
    b.stop()


@pytest.fixture
def web(plain_broker, home, monkeypatch):
    monkeypatch.setenv("HOME", str(home))   # the broker's "own machine" for the dashboard connect
    with TestClient(create_app(plain_broker, TOKEN, start=False)) as c:
        yield c


def issue(c, name="laptop"):
    return c.post("/v1/keys", headers=MAIN, json={"name": name}).json()


def test_client_key_works_on_both_headers_but_not_admin_routes(web):
    k = issue(web)
    assert k["key"].startswith("gbk_") and k["shown"] == k["key"][:8]
    for h in ({"Authorization": f"Bearer {k['key']}"}, {"x-api-key": k["key"]}):
        assert web.post("/v1/chat/completions", headers=h, json=HI).status_code == 200
        assert web.get("/v1/keys", headers=h).status_code == 403
        assert web.post("/v1/admin/resume", headers=h).status_code == 403
    listed = web.get("/v1/keys", headers=MAIN).json()
    assert listed[0]["name"] == "laptop" and "key" not in listed[0] and listed[0]["last_used"]


def test_revoked_key_is_refused_and_stays_listed(web):
    k = issue(web)
    assert web.delete(f"/v1/keys/{k['id']}", headers=MAIN).json() == {"revoked": True}
    assert web.post("/v1/chat/completions", headers={"x-api-key": k["key"]}, json=HI).status_code == 401
    assert web.get("/v1/keys", headers=MAIN).json()[0]["revoked"]
    assert web.delete(f"/v1/keys/{k['id']}", headers=MAIN).status_code == 404


def test_keys_are_stored_hashed(tmp_path):
    ks = KeyStore(str(tmp_path / "k.db"))
    k = ks.issue("x")
    raw = (tmp_path / "k.db").read_bytes()
    assert k["key"].encode() not in raw and ks.check(k["key"]) == "x" and ks.check("gbk_wrong") is None
    with pytest.raises(ValueError):
        ks.issue("  ")


def test_no_main_token_refuses_client_keys_too(plain_broker, tmp_path):
    ks = KeyStore(str(tmp_path / "k.db"))
    k = ks.issue("x")
    with TestClient(create_app(plain_broker, "", start=False, keystore=ks)) as c:
        assert c.post("/v1/chat/completions", headers={"x-api-key": k["key"]}, json=HI).status_code == 401


def test_built_in_model_map_applies_with_no_config(web):
    r = web.post("/v1/chat/completions", headers=MAIN, json={**HI, "model": "o3-mini"}).json()
    assert r["model"] == "o3-mini" and "model_map pattern 'o[0-9]*'" in r["x_broker"]["substitution"]
    assert r["x_broker"]["served_by"] == "local"


def test_check_reports_the_name_mapping(tmp_path, capsys):
    from tests.test_cli import EX, conf  # the fixture amdgpu sysfs: check reads the GPU once
    assert cli.main(["-c", conf(tmp_path, EX / "catalog.yaml"), "check"], env={}) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "names:   built-in default: gpt-* -> @default" in out and "claude-* -> @default" in out
    assert "hosted:  fallback off" in out


def test_dashboard_connects_and_disconnects_this_machine(web, home):
    before = snapshot(home)
    rows = {r["name"]: r for r in web.get("/v1/connect/clients", headers=MAIN).json()}
    assert rows["shell"]["found"] and not rows["shell"]["connected"]
    report = web.post("/v1/connect/shell", headers=MAIN).json()["report"]
    assert any("updated" in r for r in report)
    zsh = (home / ".zshrc").read_text()
    assert "export OPENAI_BASE_URL='http://testserver/v1'" in zsh
    key = zsh.split("OPENAI_API_KEY='")[1].split("'")[0]
    assert web.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=HI).status_code == 200
    assert web.get("/v1/connect/clients", headers=MAIN).json()[0]["connected"]
    web.post("/v1/disconnect/shell", headers=MAIN)
    assert without_state(snapshot(home)) == before
    assert web.post("/v1/connect/nope", headers=MAIN).status_code == 404


def test_skipped_client_does_not_leave_a_key_behind(web):
    web.post("/v1/connect/claude-code", headers=MAIN)   # opt-in only: skipped from the web
    assert all(k["revoked"] for k in web.get("/v1/keys", headers=MAIN).json())


def test_installer_is_single_use_and_carries_a_fresh_key(web):
    inv = web.post("/v1/connect/invite", headers=MAIN, json={"name": "laptop"}).json()
    code = inv["command"].split("invite=")[1].split("'")[0]
    script = web.get(f"/connect.sh?invite={code}").text
    assert script.startswith("#!/bin/sh") and "--url 'http://testserver'" in script and "--model 'llama-8b'" in script
    key = script.split("printf '%s\\n' '")[1].split("'")[0]
    assert key.startswith("gbk_") and "--key-stdin" in script and key not in script.split("connect --url")[1]
    assert web.get("/v1/models", headers={"x-api-key": key}).status_code == 200
    assert web.get(f"/connect.sh?invite={code}").status_code == 404           # used up
    assert web.get("/connect.sh?invite=guess").status_code == 404
    assert web.post("/v1/connect/invite", headers={"x-api-key": key}).status_code == 403


def test_zipapp_holds_only_the_connectors(web):
    names = zipfile.ZipFile(io.BytesIO(web.get("/connect/gpu-broker-connect.pyz").content)).namelist()
    assert "__main__.py" in names and "gpu_broker/connect/cli.py" in names
    assert not [n for n in names if n.startswith("gpu_broker/") and not n.startswith("gpu_broker/connect/")
                and n != "gpu_broker/__init__.py"]


def test_zipapp_takes_only_python_files(tmp_path, monkeypatch):
    from gpu_broker.web import connect as web_connect
    pkg = tmp_path / "pkg"
    (pkg / "__pycache__.py").mkdir(parents=True)   # a directory, even one named like a module
    (pkg / "cli.py").write_text("x = 1\n")
    (pkg / "notes.txt").write_text("not code")
    monkeypatch.setattr(web_connect, "files", lambda _: pkg)
    names = zipfile.ZipFile(io.BytesIO(web_connect.pyz())).namelist()
    assert sorted(names) == ["__main__.py", "gpu_broker/__init__.py", "gpu_broker/connect/cli.py"]
