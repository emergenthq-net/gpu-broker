"""Wan 2.2 graphs: 14B (two experts; text- or image-to-video) and 5B (text, or image, to video).

The image-to-video graphs follow ComfyUI's `video_wan2_2_14B_i2v` and `video_wan2_2_5B_ti2v`
templates: the start image conditions the 14B experts through WanImageToVideo, and seeds the
5B latent through Wan22ImageToVideoLatent.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._graph import Node, input_image, link, load_image, node, options, save_video
from .image import BATCH, DEVICE, DTYPE, FULL_DENOISE
from .video import NEGATIVE, UMT5, VIDEO

ENABLE, DISABLE = "enable", "disable"
T2V, I2V = "t2v", "i2v"
WAN14B: Mapping[str, Any] = {"width": 640, "height": 640, "frames": 81, "seed": 42, "negative": NEGATIVE, "lora_strength": 1.0,
          "steps": 4, "switch_step": 2, "cfg": 1, "shift": 5, "fps": 16, "sampler": "euler", "scheduler": "simple",
          "speed_strength": 1.0, "low_noise_seed": 0, "clip": UMT5, "clip_type": "wan", "vae": "wan_2.1_vae.safetensors"}
WAN14B_FILES: Mapping[str, Mapping[str, tuple[str, str]]] = {   # per mode: (high-noise, low-noise) expert files
    T2V: {"unets": ("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
          "speed_loras": ("wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
                          "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors")},
    I2V: {"unets": ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"),
          "speed_loras": ("wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                          "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors")},
}
WAN14B_PARAMS = frozenset({"mode", "speed_loras", "extra_loras"})
WAN5B: Mapping[str, Any] = {"width": 1280, "height": 704, "frames": 49, "seed": 42, "steps": 20, "negative": NEGATIVE, "cfg": 5, "shift": 8,
         "fps": 24, "sampler": "uni_pc", "scheduler": "simple", "unet": "wan2.2_ti2v_5B_fp16.safetensors", "clip": UMT5,
         "clip_type": "wan", "vae": "wan2.2_vae.safetensors"}
COND_POSITIVE, COND_NEGATIVE, COND_LATENT = 0, 1, 2   # WanImageToVideo outputs

# The request keys each builder reads; a catalog entry's `defaults` may set them too.
WAN14B_KEYS = (*VIDEO, "negative", "lora_strength")
WAN5B_KEYS = (*VIDEO, "negative", "steps")

def wan14b(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """High-noise expert for the first `switch_step` steps, low-noise expert for the rest, both
    behind a step-distillation LoRA pair (`speed_loras`) and an optional style pair
    (`extra_loras`, strength = request `lora_strength`). `mode` t2v (default) starts from an
    empty latent; i2v conditions on the request's `image`. Unknown params are an error: a
    stale catalog param would otherwise silently change the graph."""
    if bad := set(params) - WAN14B_PARAMS:
        raise ValueError(f"wan14b: unknown catalog params {sorted(bad)}")
    mode = params.get("mode", T2V)
    if mode not in WAN14B_FILES:
        raise ValueError(f"wan14b: mode must be one of {sorted(WAN14B_FILES)}, got {mode!r}")
    d, files, o = WAN14B, WAN14B_FILES[mode], options(req, WAN14B, WAN14B_KEYS)
    speed, extra = params.get("speed_loras") or files["speed_loras"], params.get("extra_loras")
    g = {
        "1": node("UNETLoader", unet_name=files["unets"][0], weight_dtype=DTYPE),
        "2": node("UNETLoader", unet_name=files["unets"][1], weight_dtype=DTYPE),
        "3": node("LoraLoaderModelOnly", model=link("1"), lora_name=speed[0], strength_model=d["speed_strength"]),
        "4": node("LoraLoaderModelOnly", model=link("2"), lora_name=speed[1], strength_model=d["speed_strength"]),
    }
    hi, lo = link("3"), link("4")
    if extra:
        g["20"] = node("LoraLoaderModelOnly", model=hi, lora_name=extra[0], strength_model=o["lora_strength"])
        g["21"] = node("LoraLoaderModelOnly", model=lo, lora_name=extra[1], strength_model=o["lora_strength"])
        hi, lo = link("20"), link("21")
    size = {"width": o["width"], "height": o["height"], "length": o["frames"], "batch_size": BATCH}
    if mode == I2V:
        g["17"] = load_image(input_image(req, "image", "wan14b"))
        latent_node = node("WanImageToVideo", positive=link("9"), negative=link("10"), vae=link("8"),
                           start_image=link("17"), **size)
        pos, neg, latent = link("11", COND_POSITIVE), link("11", COND_NEGATIVE), link("11", COND_LATENT)
    else:
        latent_node = node("EmptyHunyuanLatentVideo", **size)
        pos, neg, latent = link("9"), link("10"), link("11")
    common = {"steps": d["steps"], "cfg": d["cfg"], "sampler_name": d["sampler"], "scheduler": d["scheduler"],
              "positive": pos, "negative": neg}
    g.update({
        "5": node("ModelSamplingSD3", model=hi, shift=d["shift"]),
        "6": node("ModelSamplingSD3", model=lo, shift=d["shift"]),
        "7": node("CLIPLoader", clip_name=d["clip"], type=d["clip_type"], device=DEVICE),
        "8": node("VAELoader", vae_name=d["vae"]),
        "9": node("CLIPTextEncode", clip=link("7"), text=req["prompt"]),
        "10": node("CLIPTextEncode", clip=link("7"), text=o["negative"]),
        "11": latent_node,
        "12": node("KSamplerAdvanced", model=link("5"), add_noise=ENABLE, noise_seed=o["seed"], latent_image=latent,
                   start_at_step=0, end_at_step=d["switch_step"], return_with_leftover_noise=ENABLE, **common),
        "13": node("KSamplerAdvanced", model=link("6"), add_noise=DISABLE, noise_seed=d["low_noise_seed"],
                   latent_image=link("12"), start_at_step=d["switch_step"], end_at_step=d["steps"],
                   return_with_leftover_noise=DISABLE, **common),
        "14": node("VAEDecode", samples=link("13"), vae=link("8")),
        "15": node("CreateVideo", images=link("14"), fps=d["fps"]),
        "16": save_video(link("15"), prefix),
    })
    return g


def wan5b(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """Text-to-video, or image-to-video when the request carries an `image` (start frame)."""
    d, o = WAN5B, options(req, WAN5B, WAN5B_KEYS)
    g = {
        "1": node("UNETLoader", unet_name=d["unet"], weight_dtype=DTYPE),
        "2": node("CLIPLoader", clip_name=d["clip"], type=d["clip_type"], device=DEVICE),
        "3": node("VAELoader", vae_name=d["vae"]),
        "4": node("CLIPTextEncode", clip=link("2"), text=req["prompt"]),
        "5": node("CLIPTextEncode", clip=link("2"), text=o["negative"]),
        "6": node("ModelSamplingSD3", model=link("1"), shift=d["shift"]),
        "7": node("Wan22ImageToVideoLatent", vae=link("3"), width=o["width"], height=o["height"], length=o["frames"],
                  batch_size=BATCH),
        "8": node("KSampler", model=link("6"), positive=link("4"), negative=link("5"), latent_image=link("7"), seed=o["seed"],
                  steps=o["steps"], cfg=d["cfg"], sampler_name=d["sampler"], scheduler=d["scheduler"], denoise=FULL_DENOISE),
        "9": node("VAEDecode", samples=link("8"), vae=link("3")),
        "10": node("CreateVideo", images=link("9"), fps=d["fps"]),
        "11": save_video(link("10"), prefix),
    }
    if req.get("image"):
        g["12"] = load_image(input_image(req, "image", "wan5b"))
        g["7"]["inputs"]["start_image"] = link("12")
    return g
