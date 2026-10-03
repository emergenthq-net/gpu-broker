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
                                   {"endpoint": "file:///etc/passwd"}, {"open_url": "javascript:alert(1)"}])
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


@pytest.mark.parametrize("images", [{"img": "required"}, {"image": "maybe"}, {"end_image": True}])
def test_rejects_unknown_image_slots_or_needs(images):
    with pytest.raises(ValueError, match="inputs must map"):
        validate(bad(inputs=images))


@pytest.mark.parametrize(("exec_", "msg"), [
    ({}, "needs exec.recipe"), ({"recipe": "../x", "timeout_s": 5}, "needs exec.recipe"),
    ({"recipe": "x"}, "timeout_s"), ({"recipe": "x", "timeout_s": 0}, "timeout_s"),
    ({"recipe": "x", "timeout_s": True}, "timeout_s"), ({"recipe": "x", "timeout_s": 5, "params": ["Bad-Key"]}, "params"),
    ({"recipe": "x", "timeout_s": 5, "params": ["n"], "choices": {"m": [1]}}, "names listed in exec.params"),
    ({"recipe": "x", "timeout_s": 5, "params": ["n"], "choices": [1]}, "names listed in exec.params"),
    ({"recipe": "x", "timeout_s": 5, "params": ["n"], "choices": {"n": []}}, "non-empty list"),
    ({"recipe": "x", "timeout_s": 5, "params": ["n"], "choices": {"n": 81}}, "non-empty list"),
    ({"recipe": "x", "timeout_s": 5, "params": ["n"], "choices": {"n": [[81]]}}, "non-empty list")])
def test_exec_entries_need_a_recipe_and_a_timeout(exec_, msg):
    with pytest.raises(ValueError, match=msg):
        validate(bad(runner="exec", exec=exec_))
    validate(bad(runner="exec", exec={"recipe": "x", "timeout_s": 5, "params": ["frame_stride"]}))
    validate(bad(runner="exec", exec={"recipe": "x", "timeout_s": 5, "params": ["n"], "choices": {"n": [81, 161]}}))


@pytest.mark.parametrize("model", [{"runner": "llm_unit", "inputs": {"image": "optional"}},
                                   {"runner": "comfy", "inputs": {"image": "required"}},          # no template
                                   {"runner": "external", "inputs": {"image": "required"}}])
def test_only_comfy_templates_and_exec_models_take_inputs(model):
    with pytest.raises(ValueError, match="take inputs"):
        validate(bad(**model))


def test_comfy_templates_take_only_single_images():
    with pytest.raises(ValueError, match="take only"):
        validate(bad(template="wan5b", inputs={"image": "optional", "video": "optional"}))


def test_image_caps_need_inputs():
    with pytest.raises(ValueError, match="image_caps needs"):
        validate(bad(caps=["t2v"], image_caps=["i2v"]))


@pytest.mark.parametrize("frames", [{"need": "one_of", "min": 0, "max": 4}, {"need": "one_of", "min": 5, "max": 4},
                                    {"need": "one_of", "min": 2}, {"min": 2, "max": 4},
                                    {"need": "one_of", "min": 2, "max": 4, "step": 1},
                                    {"need": "one_of", "min": True, "max": 4}, {"need": "one_of", "min": 2, "max": "4"},
                                    {"need": "sometimes", "min": 2, "max": 4}])
def test_a_frame_range_is_need_min_max_with_1_le_min_le_max(frames):
    with pytest.raises(ValueError, match="inputs"):
        validate(bad(runner="exec", exec={"recipe": "x", "timeout_s": 5}, inputs={"frames": frames, "video": "one_of"}))


def test_a_frame_range_counts_as_its_need_and_only_frames_take_one():
    ok = bad(runner="exec", exec={"recipe": "x", "timeout_s": 5},
             inputs={"frames": {"need": "one_of", "min": 2, "max": 4}, "video": "one_of"})
    validate(ok)
    with pytest.raises(ValueError, match="one_of needs at least two"):
        validate(bad(runner="exec", exec={"recipe": "x", "timeout_s": 5},
                     inputs={"frames": {"need": "one_of", "min": 2, "max": 4}}))
    with pytest.raises(ValueError, match="inputs must map"):
        validate(bad(runner="exec", exec={"recipe": "x", "timeout_s": 5},
                     inputs={"image": {"need": "required", "min": 1, "max": 1}}))
