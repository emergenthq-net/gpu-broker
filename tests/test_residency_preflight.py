"""Capacity refusals preserve the LLM; only observed free memory counts."""
import pytest

from gpu_broker import settings
from tests.test_residency import res  # noqa: F401 — shared fixture


def unchanged(res, before):
    assert res.driver.calls == before
    assert res.driver.active == {"llama-8b", "unmanaged-tenant"}
    assert res.current == "llama-8b"


@pytest.mark.parametrize("reclaim", [None, 0, 7000, 30000])
def test_actual_card_capacity_rejects_before_any_unload_even_without_reclaim(res, reclaim):
    res.catalog.models["llama-8b"]["vram_mib"] = reclaim
    # Target alone fits; its reserve does not. Catalog default is a larger card.
    res.driver.vram = (8000, 21599)
    res.driver.active.add("unmanaged-tenant")
    before = list(res.driver.calls)
    with pytest.raises(RuntimeError, match="Insufficient estimated GPU headroom"):
        res.ensure("qwen-coder-32b")
    unchanged(res, before)
    assert res.backends.frees == 0


def test_missing_reclaim_still_requires_a_working_gpu_probe(res):
    res.catalog.models["llama-8b"].pop("vram_mib")
    res.driver.no_gpu = True
    before = list(res.driver.calls)
    with pytest.raises(RuntimeError, match="headroom observation unavailable"):
        res.ensure("qwen-coder-32b")
    assert res.driver.calls == before and res.backends.frees == 0
    assert res.current == "llama-8b"


@pytest.mark.parametrize("sample", [(-1, 24564), (25000, 24564), (0, 0)])
def test_missing_reclaim_does_not_skip_initial_observation_validation(res, sample):
    res.catalog.models["llama-8b"].pop("vram_mib")
    res.driver.vram = sample
    before = list(res.driver.calls)
    with pytest.raises(RuntimeError, match="Invalid GPU headroom observation"):
        res.ensure("qwen-coder-32b")
    assert res.driver.calls == before and res.backends.frees == 0
    assert res.current == "llama-8b"


def test_missing_reclaim_uses_total_as_upper_bound_not_zero(res):
    res.catalog.models["llama-8b"].pop("vram_mib")
    res.driver.vram = (23800, 24564)
    res.ensure("qwen-coder-32b")
    assert res.current == "qwen-coder-32b"


@pytest.mark.parametrize("delay", [False, True])
def test_comfy_release_is_observed_before_stopping_the_llm(res, delay):
    res.driver.active.add("unmanaged-tenant")
    res.last_comfy = "sdxl-base"
    observed = []
    readings = iter([23800, *([23800] if delay else []), 8000])

    def gpu():
        assert res.driver.active == {"llama-8b", "unmanaged-tenant"}
        assert res.backends.frees == bool(observed)
        used = next(readings)
        observed.append(used)
        return used, 24564, 0

    res.driver.gpu = gpu
    res.ensure("qwen-coder-32b")
    assert observed[-1] == 8000 and len(observed) >= 2
    assert res.backends.frees == 1 and res.last_comfy is None
    assert res.current == "qwen-coder-32b"
    assert res.driver.active == {"qwen-coder", "unmanaged-tenant"}


def test_successful_free_without_observed_capacity_never_evicts_the_llm(res):
    res.driver.vram = (23800, 24564)
    res.driver.active.add("unmanaged-tenant")
    res.t = settings.Timeouts(exec_vram_s=3)
    before = list(res.driver.calls)
    observations = []
    res.driver.gpu = lambda: observations.append(1) or (23800, 24564, 0)
    with pytest.raises(RuntimeError, match="Insufficient estimated GPU headroom"):
        res.ensure("qwen-coder-32b")
    unchanged(res, before)
    assert res.backends.frees == 1
    assert 2 <= len(observations) <= 6


@pytest.mark.parametrize("sample", [None, (-1, 24564), (25000, 24564), (0, 0)])
def test_failed_or_invalid_post_free_reading_preserves_the_llm(res, sample):
    res.driver.active.add("unmanaged-tenant")
    before = list(res.driver.calls)

    def gpu():
        if not res.backends.frees:
            return 23800, 24564, 0
        if sample is None:
            raise OSError("probe unavailable")
        return *sample, 0

    res.driver.gpu = gpu
    with pytest.raises(RuntimeError, match="headroom observation"):
        res.ensure("qwen-coder-32b")
    unchanged(res, before)
    assert res.backends.frees == 1


def test_observed_capacity_does_not_bypass_target_health_verification(res):
    res.driver.gpu = lambda: (8000 if res.backends.frees else 23800, 24564, 0)
    res.backends.llm_healthy = lambda model: False
    with pytest.raises(RuntimeError, match="did not become healthy"):
        res.ensure("qwen-coder-32b")
    assert res.current == "llama-8b"
    assert res.backends.frees == 1
