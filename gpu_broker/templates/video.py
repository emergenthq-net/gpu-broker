"""Video graphs: Wan 2.2 14B (two experts) and 5B, HunyuanVideo 1.5, MiniMax H3 and LTX 2.5 (with audio)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._graph import Node, link, node, options, save_video
from .image import BATCH, DEVICE, DTYPE, FULL_DENOISE

VIDEO = ("width", "height", "frames", "seed")
NEGATIVE = "blurry, static, low quality, distorted, watermark, text"
ENABLE, DISABLE = "enable", "disable"
CUSTOM_SAMPLER_NODES = 5
UMT5 = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"

WAN14B: Mapping[str, Any] = {"width": 640, "height": 640, "frames": 81, "seed": 42, "negative": NEGATIVE, "lora_strength": 1.0,
          "steps": 4, "switch_step": 2, "cfg": 1, "shift": 5, "fps": 16, "sampler": "euler", "scheduler": "simple",
          "speed_strength": 1.0, "low_noise_seed": 0, "clip": UMT5, "clip_type": "wan", "vae": "wan_2.1_vae.safetensors",
          "unets": ("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
          "speed_loras": ("wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
                          "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors")}
WAN14B_PARAMS = frozenset({"mode", "speed_loras", "extra_loras"})
WAN5B: Mapping[str, Any] = {"width": 1280, "height": 704, "frames": 49, "seed": 42, "steps": 20, "negative": NEGATIVE, "cfg": 5, "shift": 8,
         "fps": 24, "sampler": "uni_pc", "scheduler": "simple", "unet": "wan2.2_ti2v_5B_fp16.safetensors", "clip": UMT5,
         "clip_type": "wan", "vae": "wan2.2_vae.safetensors"}
HUNYUAN: Mapping[str, Any] = {"width": 848, "height": 480, "frames": 61, "seed": 42, "steps": 4, "shift": 7, "fps": 24, "sampler": "euler",
           "scheduler": "simple", "lora_strength": 1.0, "clip_type": "hunyuan_video_15",
           "unet": "hunyuanvideo1.5_480p_t2v_cfg_distilled_fp8_scaled.safetensors",
           "clips": ("qwen_2.5_vl_7b_fp8_scaled.safetensors", "byt5_small_glyphxl_fp16.safetensors"),
           "vae": "hunyuanvideo15_vae_fp16.safetensors",
           "lora": "hunyuanvideo1.5_t2v_480p_lightx2v_4step_lora_rank_32_bf16.safetensors"}
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


def wan14b(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """High-noise expert for the first `switch_step` steps, low-noise expert for the rest, both
    behind a step-distillation LoRA pair (`speed_loras`) and an optional style pair
    (`extra_loras`, strength = request `lora_strength`). Unknown params are an error: a stale
    catalog param would otherwise silently change the graph."""
    if bad := set(params) - WAN14B_PARAMS:
        raise ValueError(f"wan14b: unknown catalog params {sorted(bad)}")
    d, o = WAN14B, options(req, WAN14B, (*VIDEO, "negative", "lora_strength"))
    speed, extra = params.get("speed_loras") or d["speed_loras"], params.get("extra_loras")
    g = {
        "1": node("UNETLoader", unet_name=d["unets"][0], weight_dtype=DTYPE),
        "2": node("UNETLoader", unet_name=d["unets"][1], weight_dtype=DTYPE),
        "3": node("LoraLoaderModelOnly", model=link("1"), lora_name=speed[0], strength_model=d["speed_strength"]),
        "4": node("LoraLoaderModelOnly", model=link("2"), lora_name=speed[1], strength_model=d["speed_strength"]),
    }
    hi, lo = link("3"), link("4")
    if extra:
        g["20"] = node("LoraLoaderModelOnly", model=hi, lora_name=extra[0], strength_model=o["lora_strength"])
        g["21"] = node("LoraLoaderModelOnly", model=lo, lora_name=extra[1], strength_model=o["lora_strength"])
        hi, lo = link("20"), link("21")
    common = {"steps": d["steps"], "cfg": d["cfg"], "sampler_name": d["sampler"], "scheduler": d["scheduler"],
              "positive": link("9"), "negative": link("10")}
    g.update({
        "5": node("ModelSamplingSD3", model=hi, shift=d["shift"]),
        "6": node("ModelSamplingSD3", model=lo, shift=d["shift"]),
        "7": node("CLIPLoader", clip_name=d["clip"], type=d["clip_type"], device=DEVICE),
        "8": node("VAELoader", vae_name=d["vae"]),
        "9": node("CLIPTextEncode", clip=link("7"), text=req["prompt"]),
        "10": node("CLIPTextEncode", clip=link("7"), text=o["negative"]),
        "11": node("EmptyHunyuanLatentVideo", width=o["width"], height=o["height"], length=o["frames"], batch_size=BATCH),
        "12": node("KSamplerAdvanced", model=link("5"), add_noise=ENABLE, noise_seed=o["seed"], latent_image=link("11"),
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
    d, o = WAN5B, options(req, WAN5B, (*VIDEO, "negative", "steps"))
    return {
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


def hunyuan(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    d, o = HUNYUAN, options(req, HUNYUAN, VIDEO)
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


def minimax(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """Image-to-video with a synchronised audio track (separate audio VAE)."""
    d, o = MINIMAX, options(req, MINIMAX, VIDEO)
    return {
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
    d, o = LTX25, options(req, LTX25, (*VIDEO, "steps", "cfg", "fps", "negative"))
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
