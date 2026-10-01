"""Catalog validation and the writes the API may make."""
import copy

import pytest
import yaml

from gpu_broker.catalog import Catalog, validate
from tests.helpers import FIX

DATA = yaml.safe_load((FIX / "catalog.yaml").read_text())


def bad(**model):
    d = copy.deepcopy(DATA)
    d["models"]["x"] = {"runner": "comfy", "status": "ready", **model}
    return d


def test_the_fixture_and_examples_validate():
    validate(copy.deepcopy(DATA))


@pytest.mark.parametrize("model", [{"runner": "shell"}, {"status": "maybe"}, {"unit": "../x"},
                                   {"endpoint": "file:///etc/passwd"}, {"open_url": "javascript:alert(1)"},
                                   {"health_path": "http://evil.example/health"}, {"metrics_path": "../metrics"},
                                   {"api_paths": ["/v1/chat/completions", "/arbitrary-proxy"]}])
def test_rejects_unsafe_entries(model):
    with pytest.raises(ValueError):
        validate(bad(**model))


def test_rejects_unknown_default_resident():
    d = copy.deepcopy(DATA)
    d["defaults"]["resident"] = "nope"
    with pytest.raises(ValueError, match="resident"):
        validate(d)


def test_register_never_adds_an_endpoint_and_save_round_trips(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text((FIX / "catalog.yaml").read_text())
    c = Catalog(str(p))
    with pytest.raises(ValueError):
        c.register("evil", {"kind": "llm", "runner": "external", "status": "needs_integration", "endpoint": "http://x"})
    c.register("new", {"kind": "image", "runner": "external", "status": "downloadable", "template": "sdxl"})
    c.mark_downloaded("new")
    again = Catalog(str(p))
    assert again.models["new"]["status"] == "ready" and again.models["new"]["downloaded"] is True
    assert not list(tmp_path.glob("*.tmp"))


def test_variants_and_reservations_are_validated():
    d = copy.deepcopy(DATA)
    d["models"]["llama-8b"]["variants"] = {"qwen-coder-32b": {}}
    with pytest.raises(ValueError, match="already a model"):
        validate(d)
    d["models"]["llama-8b"]["variants"] = {"v": {"stream": True}}
    with pytest.raises(ValueError, match="broker fields"):
        validate(d)
    d = copy.deepcopy(DATA)
    d["models"]["llama-8b"]["reserved_interactive"] = d["models"]["llama-8b"]["slots"]
    with pytest.raises(ValueError, match="reserved_interactive"):
        validate(d)


def test_runtime_contract_accepts_custom_health_and_declared_api_paths():
    d = copy.deepcopy(DATA)
    d["models"]["llama-8b"]["health_path"] = "/api/version"
    d["models"]["llama-8b"]["metrics_path"] = "/metrics"
    d["models"]["llama-8b"]["api_paths"] = ["/v1/chat/completions", "/v1/embeddings"]
    validate(d)



@pytest.mark.parametrize("mode", ["vllm_sleep", "ollama"])
def test_api_residency_requires_managed_llm_contract(mode):
    d = copy.deepcopy(DATA)
    d["models"]["llama-8b"]["residency"] = mode
    validate(d)

    bad = copy.deepcopy(DATA)
    bad["models"]["sdxl-base"]["residency"] = mode
    with pytest.raises(ValueError, match="requires runner llm_unit"):
        validate(bad)


def test_unknown_residency_mode_rejected():
    d = copy.deepcopy(DATA)
    d["models"]["llama-8b"]["residency"] = "magic"
    with pytest.raises(ValueError, match="unknown residency mode"):
        validate(d)
