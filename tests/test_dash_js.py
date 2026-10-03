"""The dashboard's image-job view, run under node: it follows its job through the event stream
dash.js already fetches (no timers of its own), takes only real job states from it, fetches the
job once after submitting and once when it ends, and never draws an older answer over a newer
one or over a newer job."""
import json
import pathlib
import re
import shutil
import subprocess

import pytest

from gpu_broker.constants import JobState

ROOT = pathlib.Path(__file__).parents[1]
STATIC = ROOT / "gpu_broker/web/static"
NODE = shutil.which("node")
COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)   # good enough for these files: no "//" in strings


def code(name):
    """A script's source without comments, so a word in a comment proves nothing either way."""
    return COMMENTS.sub("", (STATIC / name).read_text())


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_image_job_follows_events_without_timers():
    out = subprocess.run([NODE, str(ROOT / "tests/js/imagejob_events.cjs"), str(STATIC / "imagejob.js")],
                         capture_output=True, text=True, timeout=20, check=True).stdout
    assert json.loads(out) == {"untouched": [True, 1], "running": [True, 1], "done": [True, 2], "after": [True, 2],
                               "terminalOnPost": [True, 3], "endedDuringPost": [True, 4], "race": [True],
                               "olderAnswer": [True, False], "retryOnEvent": [True, 1, True, 1],
                               "retryOnToken": [True, 1], "oneRetryAtATime": [1, True], "timers": 0}


def test_dash_hands_each_event_page_and_each_saved_token_to_the_hooks():
    assert "drawEvents(ev); EVENT_HOOKS.forEach(h => h(ev));" in code("dash.js")
    assert "tick(); TOKEN_HOOKS.forEach(h => h());" in code("dash.js")
    js = code("imagejob.js")
    assert "setInterval" not in js and "setTimeout" not in js


def test_the_job_states_are_the_brokers_in_order():
    (states,) = re.findall(r"const JOB_STATES = (\[[^\]]*\]);", code("imagejob.js"))
    assert json.loads(states) == [s.value for s in JobState]


def test_comments_do_not_count():
    assert code("imagejob.js").count("setTimeout") == 0 and "no timer" not in code("imagejob.js")
    assert COMMENTS.sub("", "a // setTimeout\nb /* setInterval */ c") == "a \nb  c"


def dash_at(url, status=401, saved=""):
    out = subprocess.run([NODE, str(ROOT / "tests/js/dash_token.cjs"), str(STATIC / "dash.js"), url, str(status), saved],
                         capture_output=True, text=True, timeout=20, check=True).stdout
    return json.loads(out)


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_a_token_in_the_link_fragment_is_used_then_saved_once_accepted_and_removed_from_the_address_bar():
    # `gpu-broker demo` prints its dashboard link with #token=...; the fragment never reaches the server.
    assert dash_at("http://h/dash?x=1#token=ab%2Bc", 200) == {"sent": "Bearer ab+c", "stored": "ab+c", "replaced": "/dash?x=1"}
    assert dash_at("http://h/dash#other") == {"sent": "Bearer ", "stored": None, "replaced": None}


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_a_link_token_the_broker_refuses_does_not_replace_the_saved_one():
    assert dash_at("http://h/dash#token=wrong", 401, "good") == {"sent": "Bearer wrong", "stored": "good", "replaced": "/dash"}
    assert dash_at("http://h/dash#token=new", 200, "old") == {"sent": "Bearer new", "stored": "new", "replaced": "/dash"}
    assert dash_at("http://h/dash#token=new", 500, "old")["stored"] == "old"   # not an answer to the token
    assert dash_at("http://h/dash", 200) == {"sent": "Bearer ", "stored": None, "replaced": None}   # nothing to save


@pytest.mark.skipif(NODE is None, reason="needs node")
@pytest.mark.parametrize("fragment", ["#token=", "#token=%E0%A4"])
def test_an_empty_or_malformed_link_token_is_dropped_and_the_saved_one_kept(fragment):
    assert dash_at("http://h/dash" + fragment, 200, "good") == {"sent": "Bearer good", "stored": "good", "replaced": "/dash"}
    assert dash_at("http://h/dash" + fragment) == {"sent": "Bearer ", "stored": None, "replaced": "/dash"}


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_the_model_index_says_who_has_the_gpu():
    out = subprocess.run([NODE, str(ROOT / "tests/js/index_gpu_line.cjs"), str(STATIC / "index.js")],
                         capture_output=True, text=True, timeout=20, check=True).stdout
    assert json.loads(out) == {
        "resident": "GPU is with qwen3-8b (the chat model).",
        "lent": "GPU is lent to wan2.2-5b (video); the chat model returns when it's done.",
        "lentUnknown": "GPU is lent to wan2.2-5b; the chat model returns when it's done.",
        "free": "GPU is free.",
    }
    assert "nothing" not in code("index.js")


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_live_panel_shows_an_unknown_reading_as_a_dash_and_charts_only_known_points():
    """Power, temperature and clock are null when the card does not report them (amdgpu without
    freq1_input, nvidia-smi's [N/A]): the label reads "—", never "null W" or "NaN"."""
    src = code("live.js")
    defs = "\n".join(re.findall(r"^const (?:UNKNOWN|reading|known) = .*$", src, re.M))
    probe = defs + """
const pts = [{t: 1, power_w: null}, {t: 2, power_w: 0}, {t: 3}, {t: 4, power_w: 287.5}];
console.log(JSON.stringify([reading(null, "W"), reading(undefined, "MHz"), reading(0, "W"), reading(64, "°C"),
                            known(pts, "power_w")]));"""
    out = subprocess.run([NODE, "-e", probe], capture_output=True, text=True, timeout=20, check=True).stdout
    assert json.loads(out) == ["—", "—", "0 W", "64 °C", [[2, 0], [4, 287.5]]]


def test_live_panel_labels_use_the_helpers_and_a_neutral_clock():
    src = code("live.js")
    assert 'reading(last.clock_mhz, "MHz")' in src and 'reading(last.power_w, "W")' in src
    assert 'reading(last.temp_c, "°C")' in src and "sm_mhz" not in src
    assert 'known(pts, "power_w")' in src and 'known(pts, "temp_c")' in src


def test_utilisation_may_be_unknown_and_unreadable_processes_are_named():
    """amdgpu while runtime-suspended has no gpu_busy_percent; the status and live panels say so
    instead of "null%", and the VRAM label carries the same hint as gpu.UNREADABLE_HINT."""
    from gpu_broker.gpu import UNREADABLE_HINT
    live, dash = code("live.js"), code("dash.js")
    assert 'reading(last.util_pct, "%")' in live and 'known(pts, "util_pct")' in live
    assert f'const PROCS_HINT = "{UNREADABLE_HINT}";' in live and "last.procs_unreadable" in live
    assert 'gpu.util_pct == null ? "util unknown"' in dash and "gpu.probe" in dash
