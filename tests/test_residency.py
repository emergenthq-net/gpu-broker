"""Residency transitions and recovery from state changed outside the broker."""
import pytest

from gpu_broker import settings
from gpu_broker.catalog import Catalog
from gpu_broker.residency import Residency
from gpu_broker.store import Store
from gpu_broker.units import unit_ref
from tests.helpers import FIX, FakeBackends, FakeDriver


@pytest.fixture
def res(tmp_path):
    driver = FakeDriver({"llama-8b"})
    backends = FakeBackends(driver)
    ticks = iter(range(10**6))
    r = Residency(Catalog(str(FIX / "catalog.yaml")), driver, backends, Store(str(tmp_path / "b.db")),
                  settings.Timeouts(llm_start_s=5, comfy_start_s=5), settings.Intervals(), unit_ref("comfyui"),
                  clock=lambda: next(ticks), sleep=lambda _: None)
    r.detect()
    return r


def kinds(r):
    return [e["kind"] for e in r.store.events(0, 1000)]


def test_detect_adopts_the_running_llm(res):
    assert res.current == "llama-8b"


def test_detect_survives_a_broken_driver(res):
    def boom(spec, verb):
        raise OSError("ssh: connect timed out")
    res.driver.unit = boom
    assert res.detect() is None and "residency.detect_failed" in kinds(res)


def test_llm_to_llm_frees_comfy_stops_the_old_starts_the_new(res):
    res.ensure("qwen-coder-32b")
    assert res.driver.active == {"qwen-coder"} and res.backends.frees == 1 and res.current == "qwen-coder-32b"


def test_resident_and_healthy_is_a_no_op(res):
    calls = len(res.driver.calls)
    res.ensure("llama-8b")
    assert len(res.driver.calls) == calls and res.backends.frees == 0


def test_llm_stopped_from_outside_is_restarted(res):
    res.driver.active.clear()
    res.ensure("llama-8b")
    assert "residency.lost" in kinds(res) and "llama-8b" in res.driver.active


def test_comfy_model_stops_the_llm_and_frees_on_model_change(res):
    res.ensure("sdxl-base")
    assert res.current is None and "llama-8b" not in res.driver.active and res.backends.frees == 0
    res.ensure("wan2.2-14b-t2v")
    assert res.backends.frees == 1 and res.last_comfy == "wan2.2-14b-t2v"
    res.release_comfy()
    assert res.backends.frees == 2 and res.last_comfy is None


def test_comfy_down_is_started_or_reported(res):
    res.backends.comfy_up = False
    real = res.driver.unit

    def unit(spec, verb):
        if unit_ref(spec).name == "comfyui":
            res.backends.comfy_up = True
        return real(spec, verb)
    res.driver.unit = unit
    res.ensure("sdxl-base")
    assert "residency.comfy_started" in kinds(res)
    res.backends.comfy_up, res.comfy_unit = False, None
    with pytest.raises(RuntimeError, match=r"no comfy\.unit"):
        res.ensure("wan2.2-14b-t2v")


def test_start_failures_and_health_timeouts_raise(res):
    res.driver.fail_start.add("qwen-coder")
    with pytest.raises(RuntimeError, match="could not start"):
        res.ensure("qwen-coder-32b")
    res.driver.fail_start.clear()
    res.backends.llm_healthy = lambda m: False
    with pytest.raises(RuntimeError, match="did not become healthy"):
        res.ensure("qwen-coder-32b")


def test_an_exec_model_waits_until_its_vram_is_free(res):
    """sharp needs 12000 MiB: the GPU shows 20000 used for two readings, then the freed memory."""
    readings = iter([(20000, 24564), (20000, 24564), (8000, 24564)])
    res.driver.gpu = lambda: (*next(readings), 0)
    slept = []
    res.sleep = slept.append
    res.ensure("sharp")
    assert len(slept) == 2 and next(readings, None) is None


def test_an_exec_model_gives_up_when_the_vram_never_frees(res):
    res.driver.vram = (20000, 24564)
    res.t = settings.Timeouts(exec_vram_s=5)
    calls = []
    real = res.driver.gpu
    res.driver.gpu = lambda: calls.append(1) or real()
    with pytest.raises(RuntimeError, match=r"sharp needs 12000 MiB of VRAM; only 4564 MiB free after 5s"):
        res.ensure("sharp")
    assert 1 < len(calls) < 10   # bounded by the deadline on the injected clock, one reading per tick


def test_the_vram_wait_reads_the_samplers_cache_and_ssh_only_when_it_is_stale(res):
    cache = iter([(20000, 24564, 0), None, (8000, 24564, 0)])   # fresh, stale, fresh
    res.cached_gpu = lambda after: next(cache)
    direct = []
    res.driver.gpu = lambda: direct.append(1) or (20000, 24564, 0)
    res.ensure("sharp")
    assert direct == [1] and next(cache, None) is None


def test_an_exec_model_without_a_size_does_not_read_the_gpu(res):
    res.catalog.models["sharp"].pop("vram_mib")
    res.driver.gpu = lambda: pytest.fail("no size, nothing to wait for")
    res.ensure("sharp")


def test_the_vram_wait_asks_for_readings_taken_after_the_evictions_returned(res):
    """A sample from before the LLM stop or ComfyUI's free says nothing about the card now."""
    order, afters = [], []
    res.backends.comfy_free = lambda: order.append("free")
    res.wall = lambda: order.append("stamp") or 1000.0
    res.cached_gpu = lambda after: afters.append(after) or None
    res.ensure("sharp")
    assert order == ["free", "stamp"] and afters and set(afters) == {1000.0}
