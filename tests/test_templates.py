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
LIVE_ORDER = {"ltx25", "hunyuan_i2v", "qwen_edit", "flux2_klein_edit"}   # recorded in build order


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
            req = {**REQ, **{slot: f"{slot}.png" for slot in m.get("inputs", {})}}
            links_resolve(TEMPLATES[m["template"]](req, m.get("params", {}), PREFIX))


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


IMG = {"image": "broker-j1-image.png"}


@pytest.mark.parametrize(("template", "params"), [("wan14b", {"mode": "i2v"}), ("hunyuan_i2v", {"unet": "u.safetensors"}),
                                                   ("qwen_edit", {"unet": "u.safetensors"}),
                                                   ("flux2_klein_edit", {"unet": "u.gguf"})])
def test_image_templates_refuse_to_build_without_their_image(template, params):
    with pytest.raises(ValueError, match="needs an input image"):
        TEMPLATES[template](REQ, params, PREFIX)


def test_wan14b_i2v_loads_the_i2v_experts_and_samples_from_the_image_conditioning():
    g = TEMPLATES["wan14b"]({**REQ, **IMG}, {"mode": "i2v"}, PREFIX)
    assert [g[n]["inputs"]["unet_name"] for n in ("1", "2")] == [
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"]
    assert g["11"]["class_type"] == "WanImageToVideo" and g["11"]["inputs"]["start_image"] == ["17", 0]
    assert g["17"] == {"class_type": "LoadImage", "inputs": {"image": "broker-j1-image.png"}}
    hi = g["12"]["inputs"]
    assert (hi["positive"], hi["negative"], hi["latent_image"]) == (["11", 0], ["11", 1], ["11", 2])


def test_wan14b_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        TEMPLATES["wan14b"]({**REQ, **IMG}, {"mode": "v2v"}, PREFIX)


def test_wan5b_takes_the_image_as_an_optional_start_frame():
    assert "start_image" not in TEMPLATES["wan5b"](REQ, {}, PREFIX)["7"]["inputs"]
    assert TEMPLATES["wan5b"]({**REQ, **IMG}, {}, PREFIX)["7"]["inputs"]["start_image"] == ["12", 0]


def test_minimax_pins_the_last_frame_only_when_given():
    g = TEMPLATES["minimax"]({**REQ, **IMG}, {}, PREFIX)
    assert g["6"]["inputs"]["first_frame"] == ["16", 0] and "last_frame" not in g["6"]["inputs"] and "17" not in g
    g = TEMPLATES["minimax"]({**REQ, **IMG, "end_image": "e.png"}, {}, PREFIX)
    assert g["6"]["inputs"]["last_frame"] == ["17", 0] and g["17"]["inputs"]["image"] == "e.png"


def test_qwen_edit_enhancer_rewrites_the_prompt_unless_switched_off():
    g = TEMPLATES["qwen_edit"]({**REQ, **IMG}, {"unet": "u.safetensors"}, PREFIX)
    assert g["10"]["inputs"]["prompt"] == ["9", 0] and g["9"]["inputs"]["image"] == ["8", 0]
    assert g["9"]["inputs"]["sampling_mode.seed"] == g["11"]["inputs"]["seed"]
    g = TEMPLATES["qwen_edit"]({**REQ, **IMG, "enhance": False}, {"unet": "u.safetensors"}, PREFIX)
    assert g["10"]["inputs"]["prompt"] == "a fox" and not {"7", "8", "9"} & set(g)
    with pytest.raises(ValueError, match="enhance"):
        TEMPLATES["qwen_edit"]({**REQ, **IMG, "enhance": "false"}, {"unet": "u.safetensors"}, PREFIX)


def test_qwen_edit_output_takes_the_reference_size():
    g = TEMPLATES["qwen_edit"]({**REQ, **IMG}, {"unet": "u.safetensors"}, PREFIX)
    assert g["10"]["inputs"]["images.image_1"] == ["6", 0] and g["11"]["inputs"]["latent_image"] == ["10", 2]


def test_klein_edit_renders_at_the_scaled_source_size_with_the_source_as_reference():
    g = TEMPLATES["flux2_klein_edit"]({**REQ, **IMG}, {"unet": "u.gguf"}, PREFIX)
    assert g["16"]["inputs"]["width"] == ["6", 0] and g["13"]["inputs"]["height"] == ["6", 1]
    assert g["10"]["inputs"]["latent"] == g["11"]["inputs"]["latent"] == ["7", 0]
    assert g["12"]["inputs"]["model"] == ["1", 0]
    g = TEMPLATES["flux2_klein_edit"]({**REQ, **IMG}, {"unet": "u.gguf", "lora": "l.safetensors"}, PREFIX)
    assert g["12"]["inputs"]["model"] == ["20", 0] and g["20"]["inputs"]["model"] == ["1", 0]


def test_hunyuan_i2v_requires_its_model_file():
    with pytest.raises(ValueError, match="unet"):
        TEMPLATES["hunyuan_i2v"]({**REQ, **IMG}, {}, PREFIX)


def test_hunyuan_i2v_schedules_on_the_shifted_model():
    """BasicScheduler must see ModelSamplingSD3's shift, like the guider (review of PR #2)."""
    g = TEMPLATES["hunyuan_i2v"]({"prompt": "p", "image": "a.png"}, {"unet": "u.safetensors"}, PREFIX)
    by_class = {n["class_type"]: k for k, n in g.items()}
    shifted = [by_class["ModelSamplingSD3"], 0]
    assert g[by_class["BasicScheduler"]]["inputs"]["model"] == shifted
    assert g[by_class["CFGGuider"]]["inputs"]["model"] == shifted
