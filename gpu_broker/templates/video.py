"""Video graphs with audio: MiniMax H3 (text or image to video) and LTX 2.5; shared video helpers.
Wan 2.2 is in wan.py, HunyuanVideo 1.5 in hunyuan.py."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._graph import Node, input_image, link, load_image, node, options, save_video
from .image import BATCH, DEVICE, DTYPE, FULL_DENOISE

VIDEO = ("width", "height", "frames", "seed")
NEGATIVE = "blurry, static, low quality, distorted, watermark, text"
CUSTOM_SAMPLER_NODES = 5
UMT5 = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"

MINIMAX: Mapping[str, Any] = {"width": 832, "height": 480, "frames": 56, "seed": 42, "steps": 8, "fps": 24, "sampler": "res_multistep",
           "scheduler": "simple", "lora_strength": 1.0, "clip_type": "minimax",
           "unet": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
           "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
           "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
           "vae": "minimax_h3_video_vae_fp16.safetensors", "audio_vae": "minimax_h3_audio_vae_fp32.safetensors"}
LTX25: Mapping[str, Any] = {"width": 768, "height": 512, "frames": 97, "seed": 42, "steps": 8, "cfg": 1.0, "fps": 24,
         "negative": "blurry, distorted, low quality, watermark", "sampler": "euler", "scheduler": "simple",
         "tile_size": 512, "tile_overlap": 64, "temporal_size": 64, "temporal_overlap": 16}
LTX25_PARAMS = ("unet", "clip", "video_vae", "audio_vae")
LTX25_MODEL, LTX25_CLIP, LTX25_VIDEO_VAE, LTX25_AUDIO_VAE = range(4)  # LTXV25ModelsLoader outputs
COND_POSITIVE, COND_NEGATIVE = 0, 1  # LTXVConditioning outputs

# The request keys each builder reads; a catalog entry's `defaults` may set them too.
MINIMAX_KEYS = VIDEO
LTX25_KEYS = (*VIDEO, "steps", "cfg", "fps", "negative")

def custom_sampler(model: list[Any], cond: list[Any], latent: list[Any], d: Mapping[str, Any],
                   o: Mapping[str, Any], first: int) -> dict[str, Node]:
    """Scheduler → noise → sampler → guider → SamplerCustomAdvanced, numbered from `first`."""
    sched, noise, sampler, guider, run = map(str, range(first, first + CUSTOM_SAMPLER_NODES))
    return {
        sched: node("BasicScheduler", model=model, scheduler=d["scheduler"], steps=d["steps"], denoise=FULL_DENOISE),
        noise: node("RandomNoise", noise_seed=o["seed"]),
        sampler: node("KSamplerSelect", sampler_name=d["sampler"]),
        guider: node("BasicGuider", model=model, conditioning=cond),
        run: node("SamplerCustomAdvanced", noise=link(noise), guider=link(guider), sampler=link(sampler),
                  sigmas=link(sched), latent_image=latent),
    }


def minimax(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """Video with a synchronised audio track (separate audio VAE). The model is first/last-frame
    to video: with no image it generates from text alone; a request `image` pins the first
    frame and `end_image` the last, as in ComfyUI's `video_minimax_h3_i2v` template."""
    d, o = MINIMAX, options(req, MINIMAX, MINIMAX_KEYS)
    g = {
        "1": node("UNETLoader", unet_name=d["unet"], weight_dtype=DTYPE),
        "2": node("LoraLoaderModelOnly", model=link("1"), lora_name=d["lora"], strength_model=d["lora_strength"]),
        "3": node("CLIPLoader", clip_name=d["clip"], type=d["clip_type"], device=DEVICE),
        "4": node("VAELoader", vae_name=d["vae"]),
        "5": node("VAELoader", vae_name=d["audio_vae"]),
        "6": node("MiniMaxH3ImageToVideo", clip=link("3"), vae=link("4"), prompt=req["prompt"], width=o["width"],
                  height=o["height"], length=o["frames"]),
        **custom_sampler(link("2"), link("6"), link("6", 1), d, o, first=7),
        "12": node("VAEDecode", samples=link("11"), vae=link("4")),
        "13": node("VAEDecodeAudio", samples=link("11"), vae=link("5")),
        "14": node("CreateVideo", images=link("12"), fps=d["fps"], audio=link("13")),
        "15": save_video(link("14"), prefix),
    }
    for node_id, slot, frame in (("16", "image", "first_frame"), ("17", "end_image", "last_frame")):
        if req.get(slot):
            g[node_id] = load_image(input_image(req, slot, "minimax"))
            g["6"]["inputs"][frame] = link(node_id)
    return g


def ltx25(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """LTX 2.5 text-to-video with a synchronised audio track, video and audio sampled as one
    concatenated latent. Catalog `params` name the four files: `unet`, `clip`, `video_vae`,
    `audio_vae`.

    Requires the ComfyUI custom node pack ComfyUI-GGUF-Loader
    (https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader, verified at commit 142c614) for the
    `LTXV25ModelsLoader` and `LTXV25AVDecode` nodes; the other nodes are stock ComfyUI. The
    defaults render 97 frames at 768x512 in 8 steps."""
    if missing := [k for k in LTX25_PARAMS if k not in params]:
        raise ValueError(f"ltx25: missing catalog params {missing}")
    d, o = LTX25, options(req, LTX25, LTX25_KEYS)
    loader, fps = "1", o["fps"]
    return {
        loader: node("LTXV25ModelsLoader", unet_name=params["unet"], clip_name=params["clip"],
                     video_vae_name=params["video_vae"], audio_vae_name=params["audio_vae"]),
        "10": node("CLIPTextEncode", text=req["prompt"], clip=link(loader, LTX25_CLIP)),
        "11": node("CLIPTextEncode", text=o["negative"], clip=link(loader, LTX25_CLIP)),
        # Conditioning and decode take the frame rate as a float, the audio latent as an int.
        "12": node("LTXVConditioning", positive=link("10"), negative=link("11"), frame_rate=float(fps)),
        "13": node("EmptyLTXVLatentVideo", width=o["width"], height=o["height"], length=o["frames"], batch_size=BATCH),
        "14": node("LTXVEmptyLatentAudio", frames_number=o["frames"], frame_rate=int(fps), batch_size=BATCH,
                   audio_vae=link(loader, LTX25_AUDIO_VAE)),
        "15": node("LTXVConcatAVLatent", video_latent=link("13"), audio_latent=link("14")),
        "3": node("KSampler", model=link(loader, LTX25_MODEL), seed=o["seed"], steps=o["steps"], cfg=o["cfg"],
                  sampler_name=d["sampler"], scheduler=d["scheduler"], positive=link("12", COND_POSITIVE),
                  negative=link("12", COND_NEGATIVE), latent_image=link("15"), denoise=FULL_DENOISE),
        "4": node("LTXV25AVDecode", latent=link("3"), vae=link(loader, LTX25_VIDEO_VAE),
                  audio_vae=link(loader, LTX25_AUDIO_VAE), fps=float(fps), tile_size=d["tile_size"],
                  overlap=d["tile_overlap"], temporal_size=d["temporal_size"], temporal_overlap=d["temporal_overlap"]),
        "5": save_video(link("4"), prefix),
    }
