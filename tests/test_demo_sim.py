"""The demo's simulated parts: the card, the driver, the model servers and the placeholder
images. Every clock and sleep is a fake, so nothing here waits."""
from __future__ import annotations

import ipaddress
import random
import re
import struct
import zlib

import pytest
import yaml

from gpu_broker.catalog import Catalog
from gpu_broker.constants import Verb
from gpu_broker.demo import content, font, placeholder
from gpu_broker.demo.backends import SimBackends, prompt_of
from gpu_broker.demo.driver import SimDriver
from gpu_broker.demo.gpu import COMFY_GROUP, SimGpu
from gpu_broker.demo.run import memory
from gpu_broker.demo.tuning import CATALOG_FILE, SimCard, SimTimings
from gpu_broker.metrics import parse_sample
from tests.helpers import ROOT, load_leak_scan

CARD = SimCard()


def rng() -> random.Random:
    return random.Random(0)  # noqa: S311 — a seeded simulation, not security
DEMO_DIR = ROOT / "gpu_broker/demo"
DEMO_CATALOG = DEMO_DIR / CATALOG_FILE


class Clock:
    def __init__(self) -> None:
        self.t, self.slept = 0.0, []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def rig(tmp_path):
    clock, gpu = Clock(), SimGpu(CARD, rng())
    catalog = Catalog(str(DEMO_CATALOG))
    driver = SimDriver(gpu, memory(catalog), SimTimings(), CARD, str(tmp_path), clock=clock, sleep=clock.sleep)
    backends = SimBackends(catalog, driver, gpu, SimTimings(), CARD, str(tmp_path), "http://demo", rng(),
                           sleep=clock.sleep)
    return clock, gpu, driver, backends, catalog


def test_the_demo_catalog_offers_public_models_on_this_machine_only():
    data = yaml.safe_load(DEMO_CATALOG.read_text())
    assert set(data["models"]) == {"qwen3-8b", "llama-3.1-8b", "flux.2-klein-4b", "sdxl-base", "wan2.2-5b", "trellis"}
    assert all(m["endpoint"].startswith("http://127.0.0.1:") for m in data["models"].values() if "endpoint" in m)
    assert not any("target" in m for m in data["models"].values())   # no container ids
    Catalog(str(DEMO_CATALOG))   # loads and validates like any catalog


def test_the_demo_names_no_network_but_this_machine():
    """Every IPv4 address in the demo (catalog, scripted traffic, defaults) is loopback or
    "any": no private LAN, link-local or shared address can appear on its dashboard."""
    found = {(f.name, a) for f in DEMO_DIR.rglob("*") if f.suffix in {".py", ".yaml"}
             for a in re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", f.read_text())}
    assert found   # the check is looking at something
    assert all(ipaddress.ip_address(a).is_loopback or ipaddress.ip_address(a).is_unspecified for _, a in found), found


def test_the_demo_matches_nothing_on_the_private_denylist():
    """With GPU_BROKER_LEAK_DENYLIST naming a denylist file (CI holds it as a secret, so the
    list is never in this repository), nothing the demo ships or shows may match it."""
    leak_scan = load_leak_scan()
    pats = leak_scan.from_env()
    if pats is None:
        pytest.skip(f"{leak_scan.ENV} is not set")
    assert leak_scan.scan([DEMO_DIR], pats, set()) == []


def test_starting_a_chat_server_takes_its_load_time_and_claims_its_memory(tmp_path):
    clock, gpu, driver, backends, catalog = rig(tmp_path)
    qwen = catalog.models["qwen3-8b"]
    assert driver.unit("llama-server-qwen3", Verb.START)
    assert gpu.held("llama-server-qwen3") == qwen["vram_mib"]
    assert driver.unit("llama-server-qwen3", Verb.IS_ACTIVE) and not backends.llm_healthy(qwen)
    clock.t += SimTimings().llm_load_s
    assert backends.llm_healthy(qwen)


def test_stopping_a_chat_server_takes_its_stop_time_and_frees_its_memory(tmp_path):
    clock, gpu, driver, *_ = rig(tmp_path)
    driver.unit("llama-server-qwen3", Verb.START)
    driver.unit("llama-server-qwen3", Verb.STOP)
    assert clock.slept == [SimTimings().llm_stop_s]
    assert gpu.held("llama-server-qwen3") == 0 and not driver.unit("llama-server-qwen3", Verb.IS_ACTIVE)


def test_comfy_loads_a_model_once_then_renders_a_placeholder(tmp_path):
    clock, gpu, _, backends, catalog = rig(tmp_path)
    graph = {"5": {"class_type": "CLIPTextEncode", "inputs": {"text": content.IMAGE_PROMPT}}}
    out = backends.comfy_run("flux.2-klein-4b", graph, "abc123")
    t = SimTimings()
    assert clock.slept == [t.comfy_load_s, t.run_s["image"]]
    assert gpu.held(COMFY_GROUP) == CARD.comfy_idle_mib + catalog.models["flux.2-klein-4b"]["vram_mib"]
    assert out["outputs"][0]["url"] == "http://demo/view?filename=abc123_00001_.png&subfolder=broker&type=output"
    assert (tmp_path / "broker/abc123_00001_.png").read_bytes().startswith(placeholder.PNG_SIGNATURE)
    backends.comfy_run("flux.2-klein-4b", graph, "abc124")
    assert clock.slept[2:] == [t.run_s["image"]]   # already loaded: no second load
    backends.comfy_free()
    assert gpu.held(COMFY_GROUP) == CARD.comfy_idle_mib


def test_work_shows_as_utilisation_power_and_heat():
    gpu = SimGpu(CARD, rng())
    idle = parse_sample(gpu.sample(), 0)
    assert idle["util_pct"] == CARD.idle_util and idle["sm_mhz"] == CARD.idle_mhz
    with gpu.working(CARD.render_load):
        busy = [parse_sample(gpu.sample(), 0) for _ in range(3)]
    assert all(s["util_pct"] >= CARD.busy_util - CARD.jitter_util for s in busy)
    assert busy[0]["power_w"] > CARD.busy_w - CARD.jitter_w and busy[-1]["temp_c"] > idle["temp_c"]
    assert gpu.reading()[2] == CARD.idle_util   # the work ended


def test_samples_split_memory_by_owner():
    gpu = SimGpu(CARD, rng())
    gpu.hold("llama-server-qwen3", 8200)
    s = parse_sample(gpu.sample(), 0)
    assert s["by_group"] == {COMFY_GROUP: CARD.comfy_idle_mib, "llama-server-qwen3": 8200}
    assert s["used_mib"] == CARD.base_mib + CARD.comfy_idle_mib + 8200 and s["total_mib"] == CARD.total_mib


def test_chat_replies_carry_llama_cpp_timings(tmp_path):
    clock, _, _, backends, catalog = rig(tmp_path)
    r = backends.llm_chat(catalog.models["qwen3-8b"], {"messages": []})
    assert r["choices"][0]["message"]["content"] in content.REPLIES
    assert r["timings"]["predicted_per_second"] > 0 and r["timings"]["prompt_n"] > 0
    lo, hi = SimTimings().chat_s
    assert lo <= clock.slept[0] <= hi


def test_the_placeholder_is_a_valid_png_with_the_prompt_drawn_on_it():
    data = placeholder.render(["AB"], placeholder.KIND_RGB["image"])
    assert data[:8] == placeholder.PNG_SIGNATURE
    w, h = struct.unpack(">II", data[16:24])
    assert (w, h) == (placeholder.WIDTH, placeholder.HEIGHT)
    raw = zlib.decompress(data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8])
    stride = 1 + w * 3
    top = placeholder.MARGIN   # first row of the "A": its top bar is lit at glyph x 1..3
    x = placeholder.MARGIN + 2 * placeholder.SCALE
    assert raw[top * stride + 1 + x * 3:top * stride + 4 + x * 3] == bytes(placeholder.TEXT_RGB)
    assert raw[1:4] != bytes(placeholder.TEXT_RGB)   # the corner is background


def test_every_character_the_demo_draws_has_a_glyph():
    text = " ".join([content.PLACEHOLDER_TITLE, content.PLACEHOLDER_NOTE, content.IMAGE_PROMPT, content.VIDEO_PROMPT,
                     *placeholder.lines("wan2.2-5b", "video", "x")])
    assert {c for c in text.upper() if c not in font.GLYPHS} == set()
    assert all(len(g) == font.HEIGHT and max(g) < 1 << font.WIDTH for g in font.GLYPHS.values())


def test_prompt_of_reads_the_first_text_encoder():
    assert prompt_of({"1": {"class_type": "UNETLoader", "inputs": {}},
                      "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "a boat"}}}) == "a boat"
    assert prompt_of({}) == ""


def test_the_simulated_3d_recipe_writes_under_comfys_output_folder(tmp_path):
    clock, gpu, driver, *_ = rig(tmp_path)
    paths = driver.run_recipe("trellis", "abcdef12", [], 60)
    assert paths == [str(tmp_path / "broker/abcdef12/abcdef12_00001_.png")]
    assert clock.slept == [SimTimings().run_s["3d"]] and gpu.held("trellis") == 0
    assert driver.download("hf", "a/b", "b", []).returncode != 0   # the demo fetches nothing


def test_embeddings_are_one_stable_vector_per_input(tmp_path):
    _, _, _, backends, catalog = rig(tmp_path)
    qwen = catalog.models["qwen3-8b"]
    one = backends.llm_embed(qwen, {"input": "hello"})
    two = backends.llm_embed(qwen, {"input": ["hello", "world"]})
    assert [d["index"] for d in two["data"]] == [0, 1]
    assert one["data"][0]["embedding"] == two["data"][0]["embedding"] != two["data"][1]["embedding"]
    assert all(0 <= x <= 1 for x in one["data"][0]["embedding"])
