"""`gpu-broker demo` end to end, on fast timings: the real broker, API and dashboard over the
simulation, its ComfyUI stand-ins, its traffic script, and the CLI entry point."""
from __future__ import annotations

import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from types import MappingProxyType
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from gpu_broker import cli, metrics
from gpu_broker.constants import RESIDENCY_EVENT_PREFIX, Event
from gpu_broker.demo import content, driver, run
from gpu_broker.demo.tuning import SimTimings, Step, Traffic
from tests.helpers import FAST, WAIT_S, done, wait_idle

QUICK = SimTimings(llm_stop_s=0.02, llm_load_s=0.03, comfy_load_s=0.02, chat_s=(0.01, 0.02), stream_word_s=0,
                   run_s=MappingProxyType({"image": 0.03, "video": 0.05, "3d": 0.03}), sample_s=0.01)
TOKEN = "demo-test"


@pytest.fixture
def demo(tmp_path):
    d = run.build(tmp_path, "127.0.0.1", 8199, TOKEN, run.Options(timings=QUICK, intervals=FAST, seed=7))
    d.broker.catalog.defaults["idle_restore_s"] = 0.2
    d.broker.start()
    yield d
    d.traffic.close()
    assert wait_idle(d.broker), "test left a demo job in flight"
    d.broker.stop()


def kinds(demo) -> list[str]:
    return [e["kind"] for e in demo.broker.store.events(0, 1000)]


def wait_for(cond, timeout=WAIT_S):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_a_chat_is_answered_by_the_resident_model_without_a_switch(demo):
    jid, _ = demo.broker.submit({"model": content.CHAT_MODEL, "messages": [{"role": "user", "content": "hi"}]}, "t")
    j = done(demo.broker, jid)
    assert j["state"] == "done" and j["result"]["choices"][0]["message"]["content"] in content.REPLIES
    assert not [k for k in kinds(demo) if k.startswith(RESIDENCY_EVENT_PREFIX)]   # it was loaded at start


def test_a_video_swaps_the_chat_model_out_and_idle_brings_it_back(demo):
    jid, _ = demo.broker.submit({"model": content.VIDEO_MODEL, "prompt": content.VIDEO_PROMPT}, "t")
    j = done(demo.broker, jid)
    assert j["state"] == "done", j
    url = j["result"]["outputs"][0]["url"]
    assert url.startswith(demo.comfy + "/view?")
    assert wait_for(lambda: Event.RES_IDLE_RESTORE in kinds(demo))
    ev = kinds(demo)
    assert ev.index(Event.RES_STOP) < ev.index(Event.RES_IDLE_RESTORE)
    assert demo.broker.residency.current == "qwen3-8b"
    with TestClient(demo.app) as c:   # the placeholder output is served like ComfyUI's /view
        r = c.get(url.removeprefix(demo.url))
        assert r.status_code == 200 and r.content.startswith(b"\x89PNG")


def test_the_api_and_dashboard_need_the_printed_token(demo):
    with TestClient(demo.app) as c:
        assert c.get("/v1/status").status_code == 401
        st = c.get("/v1/status", headers={"Authorization": f"Bearer {TOKEN}"}).json()
        assert st["resident_llm"] == "qwen3-8b"
        assert c.get("/v1/ui", headers={"Authorization": f"Bearer {TOKEN}"}).json()["groups"] == content.UI_GROUPS
        assert c.get("/dash").status_code == 200


@pytest.mark.parametrize(("query", "status"), [
    ("filename=x_00001_.png&subfolder=broker", 200),
    ("filename=x_00001_.png&subfolder=broker&type=input", 404),   # ComfyUI's input folder is not the demo's
    ("filename=broker.db&subfolder=..", 404),
    ("filename=../broker.db", 404),
    ("filename=broker.db", 404),
])
def test_view_serves_only_the_output_folder(demo, query, status):
    out = pathlib.Path(demo.broker.settings.comfy.output_dir, "broker")
    out.mkdir(parents=True)
    (out / "x_00001_.png").write_bytes(b"\x89PNG")
    with TestClient(demo.app) as c:
        assert c.get(demo.comfy.removeprefix(demo.url) + "/view?" + query).status_code == status


def test_the_comfyui_stand_ins_need_the_runs_secret_path(demo):
    out = pathlib.Path(demo.broker.settings.comfy.output_dir, "broker")
    out.mkdir(parents=True)
    (out / "x_00001_.png").write_bytes(b"\x89PNG")
    path = demo.comfy.removeprefix(demo.url)
    assert path.startswith("/comfy/") and len(path) > len("/comfy/") + 20
    query = "/view?filename=x_00001_.png&subfolder=broker"
    with TestClient(demo.app) as c:
        assert c.get(path + query).status_code == 200
        assert c.get("/comfy/wrong" + query).status_code == 404
        assert c.get(query).status_code == 404 and c.get("/").status_code == 404
        r = c.get(path + "/?template=video_wan2_2_5B_ti2v")
        assert r.status_code == 200 and 'href="/dash"' in r.text
        assert c.get("/comfy/wrong/").status_code == 404
        ui = c.get("/v1/ui", headers={"Authorization": f"Bearer {TOKEN}"}).json()   # how the dashboard learns it
        assert ui["comfy_url"] == demo.comfy and c.get("/v1/ui").status_code == 401


def test_the_traffic_script_chats_and_submits_its_jobs(demo):
    t = demo.traffic
    t.t = Traffic(chat_gap_s=(0.01, 0.02), start_s=0, steps=(
        Step(chat_s=0.05), Step(job=content.IMAGE_MODEL, prompt=content.IMAGE_PROMPT, chat_s=0.05, quiet_s=0.01)))
    for step in t.t.steps:
        t.run(step)
    asked = [demo.broker.store.job(j)["requested"] for j in t.submitted]
    assert content.IMAGE_MODEL in asked and asked.count(content.CHAT_MODEL) >= 2
    assert {demo.broker.store.job(j)["requester"] for j in t.submitted} <= {*content.REQUESTERS, content.JOB_REQUESTER}


def test_a_quiet_step_starts_its_quiet_once_its_job_is_done(demo, monkeypatch):
    t, wait, ended = demo.traffic, demo.broker.wait, []
    monkeypatch.setattr(demo.broker, "wait", lambda jid, timeout_s: ended.append(j := wait(jid, timeout_s)) or j)
    t.run(Step(job=content.VIDEO_MODEL, prompt=content.VIDEO_PROMPT, quiet_s=0.001))
    assert demo.broker.store.job(t.submitted[-1])["state"] == "done"
    assert ended[-1]["state"] == "done" and len(ended) < 20   # stopped waiting once it ended


def test_the_default_script_swaps_out_and_restores_once_per_cycle():
    steps = Traffic().steps
    jobs = [s.job for s in steps if s.job]
    assert jobs == [content.VIDEO_MODEL, content.IMAGE_MODEL]
    assert steps[-1].quiet_s > 0 and not steps[-1].chat_s   # the quiet that lets the chat model come back


def test_the_cli_runs_the_demo_without_config_or_token(monkeypatch):
    seen = {}
    monkeypatch.setattr(run, "main", lambda host, port, quiet, browser: seen.update(host=host, port=port, quiet=quiet,
                                                                                   browser=browser) or 0)
    assert cli.main(["demo", "--quiet", "--port", "9001"], env={}) == 0
    assert seen == {"host": None, "port": 9001, "quiet": True, "browser": True}
    assert cli.main(["demo", "--no-browser"], env={}) == 0
    assert seen == {"host": None, "port": None, "quiet": False, "browser": False}


def test_serve_and_check_never_load_the_demo(tmp_path):
    code = ("import sys, gpu_broker.cli as c; assert c.main(['serve'], env={}) == c.EXIT_NO_TOKEN; "
            "print(sorted(m for m in sys.modules if m.startswith('gpu_broker.demo')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=tmp_path).stdout
    assert out.strip() == "[]"


@pytest.mark.parametrize(("host", "url"), [
    ("127.0.0.1", "http://127.0.0.1:9"), ("0.0.0.0", "http://127.0.0.1:9"), ("::", "http://[::1]:9"),
    ("::1", "http://[::1]:9"), ("fe80::1", "http://[fe80::1]:9"), ("box.local", "http://box.local:9"),
])
def test_links_reach_the_bind_address_from_this_machine(host, url):
    assert run.base_url(host, 9) == url


def test_an_ipv6_bind_gives_urls_that_parse(tmp_path):
    d = run.build(tmp_path, "::", 8199, TOKEN)
    s = d.broker.settings.comfy
    assert urlsplit(d.url).hostname == "::1" and urlsplit(d.url).port == 8199
    assert urlsplit(s.url).hostname == "::1" and urlsplit(s.browser_url).path.startswith("/comfy/")


def taken() -> socket.socket:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    return s


def test_with_no_port_the_demo_takes_its_own_or_else_any_free_one(monkeypatch):
    with taken() as busy:
        monkeypatch.setattr(run, "DEMO_PORT", busy.getsockname()[1])
        with run.bind("127.0.0.1", None) as sock:
            assert sock.getsockname()[1] not in (0, busy.getsockname()[1])


@pytest.fixture
def started(monkeypatch):
    """run.main up to serving: what it served, opened and printed; the server returns at once."""
    seen = {"opened": [], "served": []}
    monkeypatch.setattr(run, "DEMO_PORT", 0)
    monkeypatch.setattr(run, "serve", lambda app, sock, graceful_s: seen["served"].append(sock.getsockname()[1]))
    monkeypatch.setattr(webbrowser, "open", seen["opened"].append)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    for k in run.SSH_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    return seen


def test_the_demo_binds_then_opens_its_dashboard_once(started, capsys):
    assert run.main(None, None, quiet=True) == 0
    (port,) = started["served"]
    (url,) = started["opened"]
    assert url.startswith(f"http://127.0.0.1:{port}/dash#token=")
    assert url in capsys.readouterr().out   # the link is printed as well


def test_no_browser_or_no_screen_only_prints_the_link(started, monkeypatch, capsys):
    assert run.main(None, None, quiet=True, browser=False) == 0
    monkeypatch.setenv("SSH_CONNECTION", "192.0.2.1 22 192.0.2.2 22")
    assert run.main(None, None, quiet=True) == 0
    assert started["opened"] == [] and len(started["served"]) == 2
    assert capsys.readouterr().out.count("/dash#token=") == 2


@pytest.mark.parametrize(("env", "platform", "ok"), [
    ({}, "darwin", True), ({}, "win32", True), ({}, "linux", False), ({"DISPLAY": ":0"}, "linux", True),
    ({"WAYLAND_DISPLAY": "wayland-0"}, "linux", True), ({"SSH_TTY": "/dev/pts/1"}, "darwin", False),
    ({"DISPLAY": ":0", "SSH_CONNECTION": "a"}, "linux", False),
])
def test_a_browser_is_opened_only_on_a_local_screen(env, platform, ok):
    assert run.can_open_browser(env, platform) is ok


def test_a_broken_browser_does_not_stop_the_demo():
    def boom(url):
        raise webbrowser.Error("no browser")
    run.open_browser("http://x", {}, "darwin", boom)


def test_a_taken_port_fails_before_anything_is_printed_opened_or_started(started, monkeypatch, capsys):
    monkeypatch.setattr(run, "build", lambda *a, **k: pytest.fail("built"))
    with taken() as busy:
        assert run.main(None, busy.getsockname()[1], quiet=False) == run.EXIT_NO_PORT
    out = capsys.readouterr()
    assert "in use" in out.err and "token" not in out.out + out.err
    assert started == {"opened": [], "served": []}


def test_close_returns_promptly_while_a_step_waits_for_its_job(demo, monkeypatch):
    t = demo.traffic
    t.t = Traffic(start_s=0, job_wait_s=600, wait_slice_s=0.02, steps=(Step(job=content.IMAGE_MODEL, quiet_s=0.01),))
    waits = []
    monkeypatch.setattr(demo.broker, "wait", lambda jid, timeout_s: waits.append(timeout_s) or time.sleep(timeout_s)
                        or {"state": "running"})
    t.start()
    assert wait_for(lambda: len(waits) >= 3)
    began = time.monotonic()
    t.close()
    assert time.monotonic() - began < 1 and not t._thread.is_alive()
    assert max(waits) == 0.02
    monkeypatch.undo()
    done(demo.broker, t.submitted[-1])


def test_close_gives_up_after_close_s_when_the_traffic_thread_does_not_stop(demo, monkeypatch):
    # A wait that ignores the stop flag (a stuck broker call): close() must still return, so the
    # demo can shut down; the daemon thread is left behind.
    t, release = demo.traffic, threading.Event()
    t.t = Traffic(start_s=0, close_s=0.1, steps=(Step(job=content.IMAGE_MODEL, quiet_s=0.01),))
    monkeypatch.setattr(demo.broker, "wait", lambda jid, timeout_s: release.wait(WAIT_S) and {"state": "done"})
    t.start()
    assert wait_for(lambda: t.submitted)
    began = time.monotonic()
    t.close()
    took = time.monotonic() - began
    stuck = t._thread.is_alive()
    release.set()
    assert 0.1 <= took < 1 and stuck
    t._thread.join(WAIT_S)
    monkeypatch.undo()
    done(demo.broker, t.submitted[-1])


def test_the_dashboard_reads_the_simulated_card_at_once_and_names_it(demo):
    # Not "finding the GPU…": the demo driver has no vendor to choose.
    with TestClient(demo.app) as c:
        g = c.get("/v1/gpu", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert g["probe"] == driver.SIM_PROBE and g["total_mib"] > g["used_mib"] > 0 and "state" not in g
    sample = metrics.parse_sample(demo.broker.driver.sim.sample(), 0)   # what gpu_stream yields
    assert sample is not None and sample["total_mib"] == g["total_mib"] and sample["temp_c"] is not None
