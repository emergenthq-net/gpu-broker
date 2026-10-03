"""Still-image graphs: Qwen-Image 2.1, Chroma, FLUX.2 Klein, single-checkpoint SD/SDXL."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._graph import Node, link, node, options, save_image

GGUF_SUFFIX = ".gguf"
DTYPE = "default"
DEVICE = "default"
BATCH = 1
FULL_DENOISE = 1.0
SIZE = ("width", "height", "seed", "steps", "negative")

QWEN: Mapping[str, Any] = {"width": 1024, "height": 1024, "seed": 42, "steps": 25, "negative": "", "cfg": 1.0,
        "sampler": "euler", "scheduler": "simple", "clip": "qwen3vl_8b_int8_convrot.safetensors",
        "clip_type": "qwen_image", "vae": "qwen_image_2.1_vae_bf16.safetensors", "cache_device": "auto"}
CHROMA: Mapping[str, Any] = {"width": 1024, "height": 1024, "seed": 42, "steps": 26, "negative": "low quality, blurry, deformed",
          "cfg": 3.5, "shift": 1, "sampler": "euler", "scheduler": "beta", "clip": "t5xxl_fp8_e4m3fn_scaled.safetensors",
          "clip_type": "chroma", "vae": "ae.safetensors", "t5_min_padding": 0, "t5_min_length": 0}
KLEIN: Mapping[str, Any] = {"width": 1024, "height": 1024, "seed": 42, "steps": 20, "negative": "", "cfg": 5, "sampler": "euler",
         "clip": "qwen_3_8b_fp8mixed.safetensors", "clip_type": "flux2", "vae": "flux2-vae.safetensors",
         "lora_strength": 1.0}
SDXL: Mapping[str, Any] = {"width": 1024, "height": 1024, "seed": 42, "steps": 30, "negative": "low quality, blurry", "cfg": 7.0,
        "sampler": "dpmpp_2m", "scheduler": "karras"}

# The request keys each builder reads; a catalog entry's `defaults` may set them too.
QWEN_KEYS = SIZE
CHROMA_KEYS = SIZE
KLEIN_KEYS = (*SIZE, "cfg")
SDXL_KEYS = SIZE

def unet_loader(unet: str) -> Node:
    """GGUF files need the ComfyUI-GGUF loader; everything else uses the stock one."""
    if unet.endswith(GGUF_SUFFIX):
        return node("UnetLoaderGGUF", unet_name=unet)
    return node("UNETLoader", unet_name=unet, weight_dtype=DTYPE)


def qwen_image(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    d, o = QWEN, options(req, QWEN, QWEN_KEYS)
    return {
        "1": unet_loader(params["unet"]),
        "2": node("QwenImage21Cache", model=link("1"), device=d["cache_device"], dtype=DTYPE),
        "3": node("CLIPLoader", clip_name=d["clip"], type=d["clip_type"], device=DEVICE),
        "4": node("VAELoader", vae_name=d["vae"]),
        "5": node("TextEncodeQwenImage21", clip=link("3"), prompt=req["prompt"], negative_prompt=o["negative"],
                  resolution=max(o["width"], o["height"])),
        "6": node("EmptyLatentImage", width=o["width"], height=o["height"], batch_size=BATCH),
        "7": node("KSampler", model=link("2"), seed=o["seed"], steps=o["steps"], cfg=d["cfg"], sampler_name=d["sampler"],
                  scheduler=d["scheduler"], positive=link("5"), negative=link("5", 1), latent_image=link("6"),
                  denoise=FULL_DENOISE),
        "8": node("VAEDecode", samples=link("7"), vae=link("4")),
        "9": save_image(link("8"), prefix),
    }


def chroma(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    d, o = CHROMA, options(req, CHROMA, CHROMA_KEYS)
    return {
        "1": node("UNETLoader", unet_name=params["unet"], weight_dtype=DTYPE),
        "2": node("CLIPLoader", clip_name=d["clip"], type=d["clip_type"], device=DEVICE),
        "3": node("VAELoader", vae_name=d["vae"]),
        "4": node("T5TokenizerOptions", clip=link("2"), min_padding=d["t5_min_padding"], min_length=d["t5_min_length"]),
        "5": node("CLIPTextEncode", clip=link("4"), text=req["prompt"]),
        "6": node("CLIPTextEncode", clip=link("4"), text=o["negative"]),
        "7": node("ModelSamplingAuraFlow", model=link("1"), shift=d["shift"]),
        "8": node("CFGGuider", model=link("7"), positive=link("5"), negative=link("6"), cfg=d["cfg"]),
        "9": node("BasicScheduler", model=link("7"), scheduler=d["scheduler"], steps=o["steps"], denoise=FULL_DENOISE),
        "10": node("RandomNoise", noise_seed=o["seed"]),
        "11": node("KSamplerSelect", sampler_name=d["sampler"]),
        "12": node("EmptySD3LatentImage", width=o["width"], height=o["height"], batch_size=BATCH),
        "13": node("SamplerCustomAdvanced", noise=link("10"), guider=link("8"), sampler=link("11"), sigmas=link("9"),
                   latent_image=link("12")),
        "14": node("VAEDecode", samples=link("13"), vae=link("3")),
        "15": save_image(link("14"), prefix),
    }


def flux2_klein(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """Klein *base* (undistilled): real CFG and ~20 steps, as in the reference workflow.
    An optional catalog `lora` patches the model only."""
    d, o = KLEIN, options(req, KLEIN, KLEIN_KEYS)
    g = {
        "1": node("UnetLoaderGGUF", unet_name=params["unet"]),
        "2": node("CLIPLoader", clip_name=d["clip"], type=d["clip_type"], device=DEVICE),
        "3": node("VAELoader", vae_name=d["vae"]),
        "5": node("CLIPTextEncode", clip=link("2"), text=req["prompt"]),
        "6": node("CLIPTextEncode", clip=link("2"), text=o["negative"]),
        "8": node("CFGGuider", model=link("1"), positive=link("5"), negative=link("6"), cfg=o["cfg"]),
        "9": node("Flux2Scheduler", steps=o["steps"], width=o["width"], height=o["height"]),
        "10": node("RandomNoise", noise_seed=o["seed"]),
        "11": node("KSamplerSelect", sampler_name=d["sampler"]),
        "12": node("EmptyFlux2LatentImage", width=o["width"], height=o["height"], batch_size=BATCH),
        "13": node("SamplerCustomAdvanced", noise=link("10"), guider=link("8"), sampler=link("11"), sigmas=link("9"),
                   latent_image=link("12")),
        "14": node("VAEDecode", samples=link("13"), vae=link("3")),
        "15": save_image(link("14"), prefix),
    }
    if params.get("lora"):
        g["4"] = node("LoraLoaderModelOnly", model=link("1"), lora_name=params["lora"],
                      strength_model=params.get("lora_strength", d["lora_strength"]))
        g["8"]["inputs"]["model"] = link("4")
    return g


def sdxl(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """ComfyUI's stock default workflow around one checkpoint (`params.ckpt`)."""
    d, o = SDXL, options(req, SDXL, SDXL_KEYS)
    model, clip, vae = link("1"), link("1", 1), link("1", 2)
    return {
        "1": node("CheckpointLoaderSimple", ckpt_name=params["ckpt"]),
        "2": node("CLIPTextEncode", clip=clip, text=req["prompt"]),
        "3": node("CLIPTextEncode", clip=clip, text=o["negative"]),
        "4": node("EmptyLatentImage", width=o["width"], height=o["height"], batch_size=BATCH),
        "5": node("KSampler", model=model, positive=link("2"), negative=link("3"), latent_image=link("4"), seed=o["seed"],
                  steps=o["steps"], cfg=params.get("cfg", d["cfg"]), sampler_name=params.get("sampler", d["sampler"]),
                  scheduler=params.get("scheduler", d["scheduler"]), denoise=FULL_DENOISE),
        "6": node("VAEDecode", samples=link("5"), vae=vae),
        "7": save_image(link("6"), prefix),
    }
