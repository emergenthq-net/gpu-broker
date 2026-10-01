import copy
import pathlib

import pytest
import yaml

from gpu_broker.resolve import best_substitute, resolve

CAT = yaml.safe_load((pathlib.Path(__file__).parent / "fixtures/catalog.yaml").read_text())


@pytest.fixture
def cat():
    return copy.deepcopy(CAT)


def test_ready_model_runs_itself(cat):
    r = resolve(cat, "llama")
    assert r.resolved == "llama-8b" and r.substitution is None and r.download is None


def test_served_name_alias(cat):
    assert resolve(cat, "qwen2.5-coder-32b-instruct-q4_k_m").resolved == "qwen-coder-32b"


def test_downloadable_substitutes_and_queues_download(cat):
    cat["models"]["sdxl-base"]["status"] = "downloadable"
    r = resolve(cat, "sdxl")
    assert r.download.ref == "stabilityai/stable-diffusion-xl-base-1.0"
    # sdxl caps are [t2i, inpaint]; no other ready image model inpaints -> no substitute
    assert r.resolved is None and "no installed image model" in r.error


def test_needs_integration_falls_back_to_a_ready_model(cat):
    assert resolve(cat, "flux-schnell").resolved == "sdxl-base"


def test_substitute_picks_highest_quality_covering_caps(cat):
    r = resolve(cat, "ltx-video", caps=["t2v", "style"])
    assert r.resolved == "wan2.2-14b-style"
    assert "no runner" in r.substitution


def test_over_budget_model_substitutes(cat):
    cat["models"]["wan2.2-14b-t2v"]["vram_mib"] = 30000
    r = resolve(cat, "wan14b", caps=["t2v"])
    assert r.resolved == "wan2.2-14b-style" and "more than" in r.substitution


def test_unknown_repo_queues_download_and_substitutes_by_kind(cat):
    r = resolve(cat, "someone/Cool-Video-Model", kind="video", caps=["t2v"])
    assert r.download.as_dict() == {"kind": "hf", "ref": "someone/Cool-Video-Model",
                          "slug": "someone-cool-video-model", "include": []}
    assert r.register["status"] == "needs_integration"
    assert r.resolved == "wan2.2-14b-t2v"


def test_unknown_huggingface_url(cat):
    r = resolve(cat, "https://huggingface.co/someone/Pic-Model", kind="image")
    assert r.download.ref == "someone/Pic-Model" and r.resolved == "sdxl-base"


def test_unknown_github_url(cat):
    r = resolve(cat, "https://github.com/OpenRobotLab/AnySplat", kind="3d")
    assert r.download.kind == "gh" and r.resolved is None and r.error


def test_unknown_plain_name_without_kind_errors(cat):
    r = resolve(cat, "definitely-not-a-model")
    assert r.resolved is None and r.download is None and "no kind" in r.error


def test_substitute_never_returns_non_ready(cat):
    r = resolve(cat, "x/y", kind="3d", caps=["image_to_splat"])
    assert r.resolved is None


def test_session_only_model_is_never_a_job_target(cat):
    assert cat["models"]["triposplat"]["session_only"]
    r = resolve(cat, "triposplat")
    assert r.resolved is None and "session-only" in r.error
    assert best_substitute(cat, "3d", {"image_to_splat"}) is None


def test_frontend_entry_is_never_a_job_target():
    assert CAT["models"]["swarmui"]["session_only"]
    assert resolve(CAT, "swarmui").resolved != "swarmui"
    assert resolve(CAT, "sdxl-base", kind="image").resolved != "swarmui"


def test_known_model_must_match_requested_kind(cat):
    r = resolve(cat, "sdxl-base", kind="llm")
    assert r.resolved != "sdxl-base"
    assert r.substitution is not None and "requested kind" in r.substitution


def test_known_model_must_cover_explicit_capabilities(cat):
    r = resolve(cat, "sdxl-base", caps=["t2i", "nonexistent-cap"])
    assert r.resolved is None
    assert "does not provide capabilities" in r.error


def test_explicit_empty_capability_set_does_not_infer_all_model_caps(cat):
    r = resolve(cat, "sdxl-base", caps=[])
    assert r.resolved == "sdxl-base"


def test_auto_routes_by_kind_and_capability(cat):
    r = resolve(cat, "auto", kind="video", caps=["t2v", "style"])
    assert r.resolved == "wan2.2-14b-style"
    assert "auto-selected" in r.substitution


def test_auto_requires_kind_and_rejects_impossible_caps(cat):
    assert "requires `kind`" in resolve(cat, "auto", caps=["chat"]).error
    r = resolve(cat, "auto", kind="video", caps=["does-not-exist"])
    assert r.resolved is None and "no installed video model" in r.error
