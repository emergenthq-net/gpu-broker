"""A catalog entry's `defaults`: request > entry `defaults` > template DEFAULTS, keys checked at load."""
import copy

import pytest

from gpu_broker import templates
from gpu_broker.catalog import validate
from gpu_broker.constants import JobState
from gpu_broker.templates.image import SDXL
from tests.helpers import done
from tests.test_catalog import DATA, bad

SDXL_MODEL = DATA["models"]["sdxl-base"]
ENTRY_STEPS, REQUEST_STEPS = SDXL["steps"] + 1, SDXL["steps"] + 2


def steps(graph):
    """The step count of the graph's sampler (the only node with a `steps` input)."""
    (value,) = [n["inputs"]["steps"] for n in graph.values() if "steps" in n["inputs"]]
    return value


@pytest.mark.parametrize(("entry", "request_", "want"), [
    ({}, {}, SDXL["steps"]),                                          # template DEFAULTS
    ({"steps": ENTRY_STEPS}, {}, ENTRY_STEPS),                        # the entry overrides the template
    ({"steps": ENTRY_STEPS}, {"steps": REQUEST_STEPS}, REQUEST_STEPS),  # the request overrides both
    ({}, {"steps": REQUEST_STEPS}, REQUEST_STEPS),
], ids=["template", "entry", "request-over-entry", "request-over-template"])
def test_precedence(entry, request_, want):
    model = {**SDXL_MODEL, **({templates.DEFAULTS_KEY: entry} if entry else {})}
    assert steps(templates.build(model, {"prompt": "x", **request_}, "p")) == want


def test_a_falsy_request_value_beats_a_truthy_entry_default():
    """The request wins by presence, not truthiness: enhance False over an entry's True."""
    model = {**DATA["models"]["qwen-image-edit"], templates.DEFAULTS_KEY: {"enhance": True}}
    req = {"prompt": "x", "image": "a.png"}
    classes = [{n["class_type"] for n in templates.build(model, r, "p").values()} for r in (req, {**req, "enhance": False})]
    assert ("TextGenerate" in classes[0], "TextGenerate" in classes[1]) == (True, False)


def test_entry_defaults_never_pick_files():
    """`params` still choose model files; a `defaults` key that is a file name is refused."""
    with pytest.raises(ValueError, match=r"\['ckpt'\] are not read by template 'sdxl'"):
        validate(bad(template="sdxl", params=SDXL_MODEL["params"], defaults={"ckpt": "other.safetensors"}))


@pytest.mark.parametrize(("model", "match"), [
    ({"template": "sdxl", "defaults": {"stpes": 20}}, r"\['stpes'\] are not read by template 'sdxl'"),
    ({"template": "qwen_image", "defaults": {"enhance": False}}, r"\['enhance'\] are not read"),   # another template's key
    ({"template": "sdxl", "defaults": [["steps", 20]]}, "must be a mapping"),
    ({"template": "no_such", "defaults": {"steps": 20}}, "needs a known ComfyUI template"),
    ({"defaults": {"steps": 20}}, "needs a known ComfyUI template"),
])
def test_unknown_keys_are_rejected_at_load(model, match):
    with pytest.raises(ValueError, match=match):
        validate(bad(**model))


def test_every_template_lists_the_keys_it_reads():
    assert set(templates.TEMPLATE_KEYS) == set(templates.TEMPLATES)
    assert all(templates.TEMPLATE_KEYS.values())


def test_a_job_runs_with_the_entrys_defaults_and_its_own_overrides(client, broker):
    data = copy.deepcopy(DATA)
    data["models"]["sdxl-base"][templates.DEFAULTS_KEY] = {"steps": ENTRY_STEPS}
    validate(data)
    broker.catalog.models["sdxl-base"][templates.DEFAULTS_KEY] = {"steps": ENTRY_STEPS}
    plain = client.post("/v1/jobs", json={"model": "sdxl-base", "prompt": "a fox"}).json()["id"]
    assert done(broker, plain)["state"] == JobState.DONE
    asked = client.post("/v1/jobs", json={"model": "sdxl-base", "prompt": "a fox", "steps": REQUEST_STEPS}).json()["id"]
    assert done(broker, asked)["state"] == JobState.DONE
    assert [steps(g) for g in broker.backends.graphs[-2:]] == [ENTRY_STEPS, REQUEST_STEPS]
