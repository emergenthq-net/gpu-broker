"""Image-edit graphs: Qwen-Image 2.1 and FLUX.2 Klein 9B base, each editing the request's `image`.

Qwen follows ComfyUI's `image_qwen_image_2_1_image_edit` template: by default a prompt
enhancer (the pe_i2i Qwen 3.5 text model, through TextGenerate) looks at the source image and
rewrites the instruction before TextEncodeQwenImage21 encodes it with the image as a
reference; the output latent takes the reference's size. Klein follows
`image_flux2_klein_image_edit_9b_base`: the source, scaled to about `megapixels`, is
VAE-encoded and attached to both prompts as a reference latent.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._graph import Node, input_image, link, load_image, node, options, save_image
from .image import BATCH, DEVICE, DTYPE, FULL_DENOISE, KLEIN, QWEN, unet_loader

ON = "on"            # TextGenerate sampling_mode with sampling (its other mode is greedy)
ENHANCED_TEXT = 0    # TextGenerate outputs: (text, thinking)
COND_POSITIVE, COND_NEGATIVE, COND_LATENT = 0, 1, 2   # TextEncodeQwenImage21 outputs
WIDTH, HEIGHT = 0, 1                                   # GetImageSize outputs
QWEN_EDIT: Mapping[str, Any] = {
    "seed": 42, "steps": 25, "cfg": 1.0, "negative": "", "sampler": QWEN["sampler"], "scheduler": QWEN["scheduler"],
    # Reference images are resized to about resolution^2 pixels (0 = as uploaded, unbounded).
    "resolution": 1024, "enhance": True, "clip": QWEN["clip"], "clip_type": QWEN["clip_type"], "vae": QWEN["vae"],
    "cache_device": QWEN["cache_device"], "lora_strength": 1.0,
    "pe_clip": "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors",
    # The template's prompt-enhancer settings (TextGenerate).
    "pe": {"max_length": 16256, "thinking": True, "use_default_template": False, "mtp": "auto"},
    "pe_sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.05, "repetition_penalty": 1.05,
                    "presence_penalty": 0.0},
}
KLEIN_EDIT: Mapping[str, Any] = {"seed": 42, "steps": 20, "cfg": 5, "negative": "", "sampler": KLEIN["sampler"],
               "clip": KLEIN["clip"], "clip_type": KLEIN["clip_type"], "vae": KLEIN["vae"],
               "lora_strength": KLEIN["lora_strength"], "megapixels": 1.0, "upscale_method": "lanczos",
               "resolution_steps": 1}

# The request keys each builder reads; a catalog entry's `defaults` may set them too.
QWEN_EDIT_KEYS = ("seed", "steps", "cfg", "negative", "resolution", "enhance")
KLEIN_EDIT_KEYS = ("seed", "steps", "cfg", "negative", "megapixels")

def _lora(g: dict[str, Node], node_id: str, model: list[Any], params: Mapping[str, Any], strength: float) -> list[Any]:
    """Patch the model with the catalog's optional `lora`; returns the model link to use."""
    if not params.get("lora"):
        return model
    g[node_id] = node("LoraLoaderModelOnly", model=model, lora_name=params["lora"],
                      strength_model=params.get("lora_strength", strength))
    return link(node_id)


def qwen_edit(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """Catalog `params.unet` is the Qwen-Image 2.1 model file; `lora` is optional. Request
    `enhance: false` skips the prompt enhancer and encodes the instruction as written."""
    d, o = QWEN_EDIT, options(req, QWEN_EDIT, QWEN_EDIT_KEYS)
    g: dict[str, Node] = {"1": unet_loader(params["unet"])}
    model = _lora(g, "2", link("1"), params, d["lora_strength"])
    g.update({
        "3": node("QwenImage21Cache", model=model, device=d["cache_device"], dtype=DTYPE),
        "4": node("CLIPLoader", clip_name=d["clip"], type=d["clip_type"], device=DEVICE),
        "5": node("VAELoader", vae_name=d["vae"]),
        "6": load_image(input_image(req, "image", "qwen_edit")),
    })
    if not isinstance(o["enhance"], bool):
        raise ValueError("qwen_edit: `enhance` must be true or false")
    prompt: Any = req["prompt"]
    if o["enhance"]:
        sampling = {f"sampling_mode.{k}": v for k, v in d["pe_sampling"].items()}
        g.update({
            "7": node("CLIPLoader", clip_name=d["pe_clip"], type=d["clip_type"], device=DEVICE),
            "8": node("BatchImagesNode", **{"images.image0": link("6")}),
            "9": node("TextGenerate", clip=link("7"), image=link("8"), prompt=req["prompt"], sampling_mode=ON,
                      **sampling, **{"sampling_mode.seed": o["seed"]}, **d["pe"]),
        })
        prompt = link("9", ENHANCED_TEXT)
    g.update({
        "10": node("TextEncodeQwenImage21", clip=link("4"), vae=link("5"), prompt=prompt, negative_prompt=o["negative"],
                   resolution=o["resolution"], **{"images.image_1": link("6")}),
        "11": node("KSampler", model=link("3"), seed=o["seed"], steps=o["steps"], cfg=o["cfg"], sampler_name=d["sampler"],
                   scheduler=d["scheduler"], positive=link("10", COND_POSITIVE), negative=link("10", COND_NEGATIVE),
                   latent_image=link("10", COND_LATENT), denoise=FULL_DENOISE),
        "12": node("VAEDecode", samples=link("11"), vae=link("5")),
        "13": save_image(link("12"), prefix),
    })
    return g


def flux2_klein_edit(req: Mapping[str, Any], params: Mapping[str, Any], prefix: str) -> dict[str, Node]:
    """Catalog `params.unet` is the Klein 9B base file (GGUF or safetensors); `lora` is optional.
    The output keeps the scaled source's size."""
    d, o = KLEIN_EDIT, options(req, KLEIN_EDIT, KLEIN_EDIT_KEYS)
    g: dict[str, Node] = {"1": unet_loader(params["unet"])}
    model = _lora(g, "20", link("1"), params, d["lora_strength"])
    width, height = link("6", WIDTH), link("6", HEIGHT)
    g.update({
        "2": node("CLIPLoader", clip_name=d["clip"], type=d["clip_type"], device=DEVICE),
        "3": node("VAELoader", vae_name=d["vae"]),
        "4": load_image(input_image(req, "image", "flux2_klein_edit")),
        "5": node("ImageScaleToTotalPixels", image=link("4"), upscale_method=d["upscale_method"],
                  megapixels=o["megapixels"], resolution_steps=d["resolution_steps"]),
        "6": node("GetImageSize", image=link("5")),
        "7": node("VAEEncode", pixels=link("5"), vae=link("3")),
        "8": node("CLIPTextEncode", clip=link("2"), text=req["prompt"]),
        "9": node("CLIPTextEncode", clip=link("2"), text=o["negative"]),
        "10": node("ReferenceLatent", conditioning=link("8"), latent=link("7")),
        "11": node("ReferenceLatent", conditioning=link("9"), latent=link("7")),
        "12": node("CFGGuider", model=model, positive=link("10"), negative=link("11"), cfg=o["cfg"]),
        "13": node("Flux2Scheduler", steps=o["steps"], width=width, height=height),
        "14": node("RandomNoise", noise_seed=o["seed"]),
        "15": node("KSamplerSelect", sampler_name=d["sampler"]),
        "16": node("EmptyFlux2LatentImage", width=width, height=height, batch_size=BATCH),
        "17": node("SamplerCustomAdvanced", noise=link("14"), guider=link("12"), sampler=link("15"), sigmas=link("13"),
                   latent_image=link("16")),
        "18": node("VAEDecode", samples=link("17"), vae=link("3")),
        "19": save_image(link("18"), prefix),
    })
    return g
