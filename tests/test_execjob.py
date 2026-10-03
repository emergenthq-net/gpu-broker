"""ExecJobs: staged inputs (streamed as open files, + params.json) go to the driver; output paths
become results; exec.timeout_s must outlast the recipe's own timeout plus the kill grace."""
import json
import pathlib
from typing import ClassVar

import pytest

from gpu_broker import execjob
from gpu_broker.constants import Event
from gpu_broker.media import InputFile
from gpu_broker.settings import Comfy, Timeouts
from gpu_broker.staging import Staging
from gpu_broker.store import Store
from tests.helpers import FakeDriver

MODEL = {"runner": "exec", "exec": {"recipe": "views", "timeout_s": 90, "params": ["frame_stride", "label"]}}
RECIPE_S = 35   # 35 + 10 kill grace + 15 reap + 30 margin = 90: just enough (FakeDriver.recipe_info)
T = Timeouts()
COMFY = Comfy(url="http://comfy:8188", public_url="https://comfy.example", output_dir="/srv/comfy/output/")


def jobs(tmp_path, driver=None, clock=None):
    staging = Staging(str(tmp_path))
    staging.put("j1", [InputFile("frames", b"f1", "png", "inline", 1), InputFile("frames", b"f0", "png", "inline", 0)])
    ticks = iter([10.0, 14.25])
    driver = driver or FakeDriver()
    driver.recipe_seconds = RECIPE_S
    return execjob.ExecJobs(driver, staging, COMFY, T, clock=clock or (lambda: next(ticks)))


def test_inputs_and_params_reach_the_recipe_and_outputs_get_view_urls(tmp_path):
    d = FakeDriver()
    d.recipe_outputs = ["/srv/comfy/output/broker/j1/scene.ply", "/elsewhere/log.ply"]
    out = jobs(tmp_path, d).run("j1", "splat", MODEL, {"frame_stride": 3, "prompt": "ignored"})
    (recipe, jid, files, timeout), = d.recipes
    assert (recipe, jid, timeout) == ("views", "j1", 90.0)
    assert files == [("frames-00.png", b"f0"), ("frames-01.png", b"f1"), ("params.json", b'{"frame_stride": 3}')]
    assert d.streamed == ["frames-00.png", "frames-01.png"]   # open files, not bytes read into memory
    assert out == {"model": "splat", "wall_s": 4.2, "outputs": [
        {"file": "broker/j1/scene.ply", "path": "/srv/comfy/output/broker/j1/scene.ply",
         "url": "https://comfy.example/view?filename=scene.ply&subfolder=broker%2Fj1&type=output"},
        {"file": "log.ply", "path": "/elsewhere/log.ply"}]}


def test_no_params_file_unless_the_catalog_declares_params(tmp_path):
    d = FakeDriver()
    jobs(tmp_path, d).run("j1", "s", {"exec": {"recipe": "sharp", "timeout_s": 90}}, {"frame_stride": 3})
    assert [n for n, _ in d.recipes[0][2]] == ["frames-00.png", "frames-01.png"]


def test_no_output_is_a_failure(tmp_path):
    d = FakeDriver()
    d.recipe_outputs = []
    with pytest.raises(RuntimeError, match="produced no output"):
        jobs(tmp_path, d).run("j1", "s", MODEL, {})


def test_outputs_get_urls_only_under_the_configured_output_dir(tmp_path):
    j = jobs(tmp_path)
    j.comfy = Comfy(output_dir="")
    assert j.output("/srv/comfy/output/x.ply") == {"file": "x.ply", "path": "/srv/comfy/output/x.ply"}
    j.comfy = COMFY
    assert "url" not in j.output("/srv/comfy/outputs-other/x.ply")


@pytest.mark.parametrize("value", [[1, 2], {"a": 1}, None, "x" * 201])
def test_params_must_be_short_scalars(value):
    with pytest.raises(ValueError, match="`label` must be"):
        execjob.params(MODEL, {"label": value})
    assert execjob.params(MODEL, {"label": "x" * 200, "frame_stride": 2.5, "other": [1]}) == {
        "label": "x" * 200, "frame_stride": 2.5}
    assert json.loads(json.dumps(execjob.params(MODEL, {"frame_stride": True}))) == {"frame_stride": True}


def test_params_with_choices_take_only_the_listed_values():
    m = {"exec": {**MODEL["exec"], "params": ["frames", "label"], "choices": {"frames": [81, 161]}}}
    assert execjob.params(m, {"frames": 161, "label": "x"}) == {"frames": 161, "label": "x"}
    for v in (100, 81.0, True, "81"):
        with pytest.raises(ValueError, match=r"`frames` must be one of \[81, 161\], got"):
            execjob.params(m, {"frames": v})


def test_the_staged_files_are_closed_after_the_run_even_when_it_fails(tmp_path, monkeypatch):
    opened = []
    real_open = pathlib.Path.open

    def track(self, *a, **k):
        f = real_open(self, *a, **k)
        opened.append(f)
        return f
    monkeypatch.setattr(pathlib.Path, "open", track)
    d = FakeDriver()
    d.recipe_error = "boom"
    with pytest.raises(RuntimeError, match="boom"):
        jobs(tmp_path, d).run("j1", "splat", MODEL, {})
    assert len(opened) == 2 and all(f.closed for f in opened)


def test_a_catalog_timeout_shorter_than_the_recipes_is_refused_before_running(tmp_path):
    d = FakeDriver()
    j = jobs(tmp_path, d)
    d.recipe_seconds = RECIPE_S + 1
    with pytest.raises(execjob.TimeoutTooShort, match=r"exec.timeout_s 90 must be at least 91 \(recipe views timeout_s 36"):
        j.run("j1", "splat", MODEL, {})
    assert d.recipes == []


def test_startup_check_refuses_a_short_timeout_and_logs_an_unreadable_recipe(tmp_path):
    class Cat:
        models: ClassVar = {"splat": MODEL, "llm": {"runner": "llm_unit"}}
    store = Store(str(tmp_path / "b.db"))
    d = FakeDriver()
    d.recipe_seconds = RECIPE_S

    def check(t=T):   # a fresh broker each time: the timings are cached per ExecJobs
        execjob.check_timeouts(Cat, execjob.ExecJobs(d, Staging(str(tmp_path / "s")), COMFY, t), store)
    check()                                                      # fits
    d.recipe_seconds = RECIPE_S + 1
    with pytest.raises(execjob.TimeoutTooShort, match="must be at least"):
        check()
    d.recipe_seconds = RECIPE_S
    with pytest.raises(execjob.TimeoutTooShort, match=r"timeouts\.exec_clean_s 19 must be at least 20"):
        check(Timeouts(exec_clean_s=19))                         # FakeDriver's clean: 10 s
    check(Timeouts(exec_clean_s=20))
    for err in (OSError("ssh: no route"), ValueError("recipe views, line 3: expected key=value")):
        d.recipe_info = lambda name, err=err: (_ for _ in ()).throw(err)
        check()                                                  # unreadable or unparsable now: logged, checked per run
    assert [e["kind"] for e in store.events(0, 10)] == [Event.EXEC_UNCHECKED] * 2


def test_recipe_timings_are_read_once_and_again_after_a_failed_run(tmp_path):
    d = FakeDriver()
    reads = []
    real = d.recipe_info
    d.recipe_info = lambda name: reads.append(name) or real(name)
    j = jobs(tmp_path, d, clock=lambda: 0.0)
    j.check("splat", MODEL)
    j.run("j1", "splat", MODEL, {})
    assert reads == ["views"]                       # cached from the first check
    d.recipe_error = "boom"
    with pytest.raises(RuntimeError, match="boom"):
        j.run("j1", "splat", MODEL, {})
    j.check("splat", MODEL)
    assert reads == ["views", "views"]              # the failure dropped the cache entry


def test_a_timeout_found_too_short_drops_the_cached_timings_so_a_fixed_host_is_read_again(tmp_path):
    d = FakeDriver()
    reads = []
    real = d.recipe_info
    d.recipe_info = lambda name: reads.append(name) or real(name)
    j = jobs(tmp_path, d, clock=lambda: 0.0)
    d.recipe_seconds = RECIPE_S + 1   # the host's recipe outgrew the catalog's timeout
    with pytest.raises(execjob.TimeoutTooShort):
        j.check("splat", MODEL)
    d.recipe_seconds = RECIPE_S       # the operator fixed the host
    j.check("splat", MODEL)
    assert reads == ["views", "views"]
