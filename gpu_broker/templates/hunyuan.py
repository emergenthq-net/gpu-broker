"""HunyuanVideo 1.5 graphs: 480p text-to-video (distilled, 4-step LoRA) and 720p image-to-video.

The image-to-video graph follows ComfyUI's `video_hunyuan_video_1.5_720p_i2v` template
without its optional 1080p super-resolution stage: the start image conditions the model both
as the first frame and through a SigLIP vision embedding, sampled with real CFG.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._graph import Node, input_image, link, load_image, node, options, save_video
from .image import BATCH, DEVICE, DTYPE, FULL_DENOISE
from .video import VIDEO, custom_sampler

CLIP_TYPE = "hunyuan_video_15"
CLIPS = ("qwen_2.5_vl_7b_fp8_scaled.safetensors", "byt5_small_glyphxl_fp16.safetensors")
VAE = "hunyuanvideo15_vae_fp16.safetensors"
HUNYUAN: Mapping[str, Any] = {"width": 848, "height": 480, "frames": 61, "seed": 42, "steps": 4, "shift": 7, "fps": 24, "sampler": "euler",
           "scheduler": "simple", "lora_strength": 1.0, "clip_type": CLIP_TYPE,
           "unet": "hunyuanvideo1.5_480p_t2v_cfg_distilled_fp8_scaled.safetensors", "clips": CLIPS, "vae": VAE,
           "lora": "hunyuanvideo1.5_t2v_480p_lightx2v_4step_lora_rank_32_bf16.safetensors"}
HUNYUAN_I2V: Mapping[str, Any] = {"width": 1280, "height": 720, "frames": 61, "seed": 42, "steps": 20, "cfg": 6, "shift": 7,
               "fps": 24, "negative": "", "sampler": "euler", "scheduler": "simple", "clip_type": CLIP_TYPE,
               "clips": CLIPS, "vae": VAE, "clip_vision": "sigclip_vision_patch14_384.safetensors", "crop": "center"}
COND_POSITIVE, COND_NEGATIVE, COND_LATENT = 0, 1, 2   # HunyuanVideo15ImageToVideo outputs

# The request keys each builder reads; a catalog entry's `defaults` may set them too.
HUNYUAN_KEYS = VIDEO
HUNYUAN_I2V_KEYS = (*VIDEO, "steps", "cfg", "negative")

def hunyuan(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    d, o = HUNYUAN, options(req, HUNYUAN, HUNYUAN_KEYS)
    return {
        "1": node("UNETLoader", unet_name=d["unet"], weight_dtype=DTYPE),
        "2": node("DualCLIPLoader", clip_name1=d["clips"][0], clip_name2=d["clips"][1], type=d["clip_type"], device=DEVICE),
        "3": node("VAELoader", vae_name=d["vae"]),
        "4": node("LoraLoaderModelOnly", model=link("1"), lora_name=d["lora"], strength_model=d["lora_strength"]),
        "5": node("ModelSamplingSD3", model=link("4"), shift=d["shift"]),
        "6": node("CLIPTextEncode", clip=link("2"), text=req["prompt"]),
        "7": node("EmptyHunyuanVideo15Latent", width=o["width"], height=o["height"], length=o["frames"], batch_size=BATCH),
        **custom_sampler(link("5"), link("6"), link("7"), d, o, first=8),
        "13": node("VAEDecode", samples=link("12"), vae=link("3")),
        "14": node("CreateVideo", images=link("13"), fps=d["fps"]),
        "15": save_video(link("14"), prefix),
    }


def hunyuan_i2v(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """Catalog `params.unet` names the 720p image-to-video model file."""
    if "unet" not in params:
        raise ValueError("hunyuan_i2v: missing catalog param 'unet'")
    d, o = HUNYUAN_I2V, options(req, HUNYUAN_I2V, HUNYUAN_I2V_KEYS)
    return {
        "1": node("UNETLoader", unet_name=params["unet"], weight_dtype=DTYPE),
        "2": node("DualCLIPLoader", clip_name1=d["clips"][0], clip_name2=d["clips"][1], type=d["clip_type"], device=DEVICE),
        "3": node("VAELoader", vae_name=d["vae"]),
        "4": node("CLIPVisionLoader", clip_name=d["clip_vision"]),
        "5": load_image(input_image(req, "image", "hunyuan_i2v")),
        "6": node("CLIPVisionEncode", clip_vision=link("4"), image=link("5"), crop=d["crop"]),
        "7": node("CLIPTextEncode", clip=link("2"), text=req["prompt"]),
        "8": node("CLIPTextEncode", clip=link("2"), text=o["negative"]),
        "9": node("HunyuanVideo15ImageToVideo", positive=link("7"), negative=link("8"), vae=link("3"), start_image=link("5"),
                  clip_vision_output=link("6"), width=o["width"], height=o["height"], length=o["frames"], batch_size=BATCH),
        "10": node("ModelSamplingSD3", model=link("1"), shift=d["shift"]),
        "11": node("BasicScheduler", model=link("10"), scheduler=d["scheduler"], steps=o["steps"], denoise=FULL_DENOISE),
        "12": node("RandomNoise", noise_seed=o["seed"]),
        "13": node("KSamplerSelect", sampler_name=d["sampler"]),
        "14": node("CFGGuider", model=link("10"), positive=link("9", COND_POSITIVE), negative=link("9", COND_NEGATIVE),
                   cfg=o["cfg"]),
        "15": node("SamplerCustomAdvanced", noise=link("12"), guider=link("14"), sampler=link("13"), sigmas=link("11"),
                   latent_image=link("9", COND_LATENT)),
        "16": node("VAEDecode", samples=link("15"), vae=link("3")),
        "17": node("CreateVideo", images=link("16"), fps=d["fps"]),
        "18": save_video(link("17"), prefix),
    }
