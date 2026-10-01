"""ComfyUI graphs: against recorded graphs, links resolve, params are checked.

The recordings are the output of the graph code the broker ran before each builder was
ported. The early ones were stored with sorted keys and are compared as values; those in
LIVE_ORDER kept the recorded key order and must serialise byte-for-byte."""
import json

import pytest
import yaml

from gpu_broker.templates import TEMPLATES
from tests.helpers import FIX, ROOT

GOLDEN = json.loads((FIX / "golden_graphs.json").read_text())
PREFIX = "broker/j1"
REQ = {"prompt": "a fox", "lora_strength": 0.7}
LIVE_ORDER = {"ltx25"}


def links_resolve(g):
    for node in g.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                assert v[0] in g, f"dangling link {v}"


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_graphs_match_the_recorded_ones(case):
    g = GOLDEN[case]
    built = TEMPLATES[case.split("/")[0]](g["req"], g["params"], PREFIX)
    assert json.loads(json.dumps(built)) == g["graph"]
    if case.split("/")[0] in LIVE_ORDER:
        assert json.dumps(built) == json.dumps(g["graph"])
    links_resolve(built)


def test_every_template_is_covered_by_a_recording():
    assert {c.split("/")[0] for c in GOLDEN} == set(TEMPLATES)


@pytest.mark.parametrize("cat", ["tests/fixtures/catalog.yaml", "examples/catalog.yaml"])
def test_every_catalog_template_builds(cat):
    for m in yaml.safe_load((ROOT / cat).read_text())["models"].values():
        if "template" in m:
            links_resolve(TEMPLATES[m["template"]](REQ, m.get("params", {}), PREFIX))


def test_wan14b_extra_lora_pair_is_chained_after_the_speed_pair():
    g = TEMPLATES["wan14b"](REQ, {"mode": "t2v", "extra_loras": ["h.safetensors", "l.safetensors"]}, PREFIX)
    assert g["20"]["inputs"] == {"model": ["3", 0], "lora_name": "h.safetensors", "strength_model": 0.7}
    assert g["5"]["inputs"]["model"] == ["20", 0] and g["6"]["inputs"]["model"] == ["21", 0]


def test_wan14b_rejects_unknown_params():
    with pytest.raises(ValueError, match="unknown catalog params"):
        TEMPLATES["wan14b"](REQ, {"mode": "t2v", "some_flag": True}, PREFIX)


def test_ltx25_frame_rate_is_a_float_for_conditioning_and_decode_and_an_int_for_audio():
    # A JSON client may send fps as 25.0; the audio latent node still needs an int.
    g = TEMPLATES["ltx25"]({"prompt": "x", "fps": 25.0}, GOLDEN["ltx25/0/0"]["params"], PREFIX)
    assert json.dumps(g["12"]["inputs"]["frame_rate"]) == "25.0" and json.dumps(g["4"]["inputs"]["fps"]) == "25.0"
    assert json.dumps(g["14"]["inputs"]["frame_rate"]) == "25"


def test_ltx25_requires_all_four_model_files():
    params = dict(GOLDEN["ltx25/0/0"]["params"])
    del params["audio_vae"]
    with pytest.raises(ValueError, match="missing catalog params"):
        TEMPLATES["ltx25"](REQ, params, PREFIX)
