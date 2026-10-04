"""mcp_server.core without the SDK, the stdio side's choice of broker and credential, the
settings section, and the broker with no MCP SDK installed."""
import json

import pytest

from gpu_broker import cli, mcp_server, settings
from gpu_broker.connect import engine
from gpu_broker.mcp_server import core, stdio
from tests.mcp_fakes import COMFY, PNG, mcp_broker

__all__ = ["mcp_broker"]
CFG = settings.Mcp(inline_max_bytes=100, inline_max_images=2)


def test_the_caller_header():
    assert core.caller("main") == core.Caller(core.MAIN_REQUESTER, None)
    assert core.caller("key:k1:remote machine") == core.Caller("remote machine", "k1")
    assert core.caller("key:k1:a:b") == core.Caller("a:b", "k1")   # a name may hold colons; the id cannot
    for bad in (None, "", "key:", "key:k1", "key::laptop", "Main", "key", "admin"):
        with pytest.raises(PermissionError):
            core.caller(bad)
    assert core.Caller("laptop", "k1").sees({"owner": "k1", "requester": "laptop"})
    assert not core.Caller("laptop", "k1").sees({"owner": "k2", "requester": "laptop"})   # same name, other key
    assert not core.Caller("mcp", "k1").sees({"owner": None, "requester": "mcp"})          # the main token's job
    assert core.Caller("mcp").sees({"owner": "k9", "requester": "anyone"})


def test_inputs_by_url_or_inline():
    assert core.input_field("image", "https://h/a.png") == {"image_url": "https://h/a.png"}
    assert core.input_field("video", "http://h/v.mp4") == {"video_url": "http://h/v.mp4"}
    assert core.input_field("image", "data:image/png;base64,AAAA") == {"image": "data:image/png;base64,AAAA"}
    assert core.input_field("image", "file:///etc/passwd") == {"image": "file:///etc/passwd"}   # inline: then not an image


def test_pick_names_catalog_models_only(mcp_broker):
    assert core.pick(mcp_broker, "image", ["t2i"], frozenset(), None) == "sdxl-base"
    assert core.pick(mcp_broker, "image", ["t2i"], frozenset(), "sdxl") == "sdxl-base"   # an alias
    for name in ("org/some-repo", "flux-schnell"):   # unknown (a download), or in the catalog but not runnable
        with pytest.raises(ValueError, match="can run now"):
            core.pick(mcp_broker, "image", ["t2i"], frozenset(), name)
    with pytest.raises(ValueError, match="no ready video model can do"):
        core.pick(mcp_broker, "video", ["teleport"], frozenset(), None)


def test_inline_takes_small_images_with_their_own_type(mcp_broker):
    outs = [{"file": "broker/a.png", "url": f"{COMFY}/view?filename=a.png&subfolder=broker&type=output"},
            {"file": "broker/b.mp4", "url": f"{COMFY}/view?y"},
            {"file": "broker/c.webp"},                                   # no URL: not in ComfyUI's output
            {"file": "d.JPG", "url": f"{COMFY}/view?filename=d.JPG&subfolder=&type=temp"},   # a preview: type temp
            {"file": "broker/e.png", "url": f"{COMFY}/view?w"}]
    asked = []

    def fetch(name, sub, kind, cap):
        asked.append((name, sub, kind, cap))
        return PNG
    got = core.inline(mcp_broker, {"outputs": outs}, CFG, fetch)
    assert got == [("image/png", PNG), ("image/jpeg", PNG)]             # capped at inline_max_images
    assert asked == [("a.png", "broker", "output", CFG.inline_max_bytes), ("d.JPG", "", "temp", CFG.inline_max_bytes)]
    mcp_broker.backends.view_data = PNG                                  # by default, through broker.backends
    assert core.inline(mcp_broker, {"outputs": outs[:1]}, CFG) == [("image/png", PNG)]


def test_inline_leaves_unreadable_or_large_images_to_their_url(mcp_broker):
    outs = [{"file": "a.png", "url": "u"}, {"file": "b.png", "url": "u"}]

    def flaky(name, sub, kind, cap):
        if name == "a.png":
            raise OSError("refused")
    assert core.inline(mcp_broker, {"outputs": outs}, CFG, flaky) == []


LAN = {"gpu": "192.0.2.26", "other": "198.51.100.9", "public.example": "93.184.216.34", "mixed.example": "192.0.2.1"}


def lan(host, port):
    if host not in LAN:
        raise OSError("no such host")
    return [(None, None, None, "", (LAN[host], 0))] + ([(None, None, None, "", ("8.8.8.8", 0))] if host == "mixed.example" else [])


def test_stdio_finds_the_broker_and_credential(tmp_path):
    with pytest.raises(ValueError, match="GPU_BROKER_API_KEY"):
        stdio.target(None, {}, tmp_path, lan)
    env = {"GPU_BROKER_URL": "http://gpu:8095/", "GPU_BROKER_API_KEY": "k", "BROKER_TOKEN": "t"}
    assert stdio.target(None, env, tmp_path, lan) == ("http://gpu:8095/mcp", "k", ["192.0.2.26"])
    assert stdio.target("http://other:1", env, tmp_path, lan) == ("http://other:1/mcp", "k", ["198.51.100.9"])
    engine.remember_key(tmp_path, {"broker": "http://gpu:8095", "id": "k1", "name": "me", "key": "gbk_x"})
    assert stdio.target(None, {}, tmp_path, lan) == ("http://gpu:8095/mcp", "gbk_x", ["192.0.2.26"])   # what connect set up
    with pytest.raises(ValueError, match="no credential"):
        stdio.target("http://other:1", {}, tmp_path, lan)                           # that key is for another broker
    with pytest.raises(ValueError, match="characters"):
        stdio.target("http://h/$(x)", {"BROKER_TOKEN": "t"}, tmp_path, lan)


def test_the_main_token_goes_only_to_the_broker_on_this_machine(tmp_path, monkeypatch):
    monkeypatch.setenv("BROKER_CONFIG", str(tmp_path / "c.yaml"))
    (tmp_path / "c.yaml").write_text("server: {port: 9000}\n")
    env = {"BROKER_TOKEN": "t", "BROKER_CONFIG": str(tmp_path / "c.yaml")}
    assert stdio.target("http://127.0.0.1:9000", env, tmp_path, lan) == ("http://127.0.0.1:9000/mcp", "t", [])
    assert stdio.target("http://localhost:9000", env, tmp_path, lan)[1] == "t"
    for elsewhere in ("http://127.0.0.1:8095", "http://gpu:9000", "https://public.example:9000"):
        with pytest.raises(ValueError, match="main token: it goes only to the broker on this machine"):
            stdio.target(elsewhere, env, tmp_path, lan)


def test_no_credential_goes_over_plain_http_to_a_public_host(tmp_path):
    env = {"GPU_BROKER_API_KEY": "k"}
    for url in ("http://public.example:8095", "http://mixed.example:8095", "http://unknown.example:8095", "http://8.8.8.8:8095"):
        with pytest.raises(ValueError, match="plain http"):
            stdio.target(url, env, tmp_path, lan)
    for url in ("https://public.example", "http://gpu:8095", "http://10.1.2.3:8095", "http://[::1]:8095", "http://100.100.1.2:8095"):
        assert stdio.target(url, env, tmp_path, lan)[1] == "k", url


def test_the_mcp_settings_section(tmp_path):
    (tmp_path / "c.yaml").write_text("mcp: {wait_s: 30, inline_max_bytes: 0}\n")
    s = settings.load(str(tmp_path / "c.yaml"), env={"BROKER_MCP": "false"})
    assert (s.mcp.enabled, s.mcp.wait_s, s.mcp.inline_max_bytes) == (False, 30, 0)
    with pytest.raises(ValueError, match="mcp"):
        settings.Mcp(wait_s=-1)
    with pytest.raises(ValueError, match="mcp"):
        settings.Mcp(max_body_bytes=0)


@pytest.mark.parametrize("why", ["no sdk", "disabled"])
def test_without_the_sdk_or_when_disabled_there_is_no_mcp_route(mcp_broker, monkeypatch, why):
    import dataclasses

    from fastapi.testclient import TestClient

    from gpu_broker.web.app import create_app
    if why == "no sdk":
        monkeypatch.setattr(mcp_server, "available", lambda: False)
    else:
        mcp_broker.settings = dataclasses.replace(mcp_broker.settings, mcp=settings.Mcp(enabled=False))
    with TestClient(create_app(mcp_broker, "tok", start=False)) as c:
        assert c.get("/health").json() == {"ok": True, "mcp": False}
        assert c.post("/mcp", json={}, headers={"Authorization": "Bearer tok"}).status_code == 404


def test_gpu_broker_mcp_without_the_sdk_says_what_to_install(monkeypatch, capsys):
    monkeypatch.setattr(mcp_server, "available", lambda: False)
    assert cli.main(["mcp"], env={}) == cli.EXIT_PROBLEMS
    assert "gpu-broker[mcp]" in capsys.readouterr().err


def test_tool_reports_are_json(mcp_broker):
    out = core.status(mcp_broker, core.Caller("mcp"))
    assert json.loads(json.dumps(out)) == out


def test_a_job_the_broker_rejects_is_a_tool_error(mcp_broker):
    with pytest.raises(ValueError, match="no installed image model covers"):
        core.submit(mcp_broker, core.Caller("mcp"), {"model": "flux-schnell", "kind": "image", "caps": ["teleport"], "prompt": "x"})


def test_chat_runs_as_the_caller(mcp_broker):
    for who, requester in ((core.Caller("mcp"), "mcp"), (core.Caller("laptop", "k1"), "laptop")):
        before = {j["id"] for j in mcp_broker.store.jobs(100)}
        core.chat(mcp_broker, who, {"messages": [{"role": "user", "content": "hi"}]})
        (new,) = [j for j in mcp_broker.store.jobs(100) if j["id"] not in before]
        assert new["requester"] == requester
