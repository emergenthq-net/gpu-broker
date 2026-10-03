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


def test_image_jobs_only_substitute_models_that_take_the_image(cat):
    image = frozenset({"image"})
    assert best_substitute(cat, "video", set(), images=image) == "minimax-h3-i2v"
    assert best_substitute(cat, "video", set(), images=frozenset({"image", "end_image"})) == "minimax-h3-i2v"
    assert best_substitute(cat, "video", {"t2v"}, images=image) == "wan2.2-5b"   # optional start frame
    assert best_substitute(cat, "image", {"edit"}, images=image) == "qwen-image-edit"
    assert best_substitute(cat, "image", {"edit"}) is None                       # edit needs its image
    assert best_substitute(cat, "video", set()) == "wan2.2-14b-t2v"              # never a model needing one


def test_an_unavailable_image_model_is_replaced_by_one_that_takes_the_image(cat):
    cat["models"]["wan2.2-14b-i2v"]["status"] = "downloadable"
    r = resolve(cat, "wan2.2-14b-i2v", images=frozenset({"image"}))
    assert r.resolved == "minimax-h3-i2v" and "wan2.2-14b-i2v" in (r.substitution or "")


def test_image_caps_count_only_when_the_job_carries_an_image(cat):
    """wan2.2-5b is t2v, and i2v only given a start frame: a text job on it must not demand i2v
    of its stand-in (regression: [t2v, i2v] caps left text jobs with no substitute)."""
    cat["models"]["wan2.2-5b"]["status"] = "downloadable"
    r = resolve(cat, "wan2.2-5b")
    assert r.resolved == "wan2.2-14b-t2v" and "wan2.2-5b" in (r.substitution or "")
    r = resolve(cat, "wan2.2-5b", images=frozenset({"image"}))   # uses only i2v, so needs only i2v
    assert r.resolved == "minimax-h3-i2v" and "wan2.2-5b" in (r.substitution or "")
    cat["models"]["wan2.2-5b"]["status"] = "ready"
    assert best_substitute(cat, "video", {"i2v"}, images=frozenset({"image"})) == "minimax-h3-i2v"
    assert best_substitute(cat, "video", {"i2v"}) is None   # no image: 5b's i2v does not count
