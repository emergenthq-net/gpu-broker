# The catalog

The catalog lists what may be requested and how each model runs. It is trusted configuration:
the only place backend URLs come from.

- Annotated example: [`examples/catalog.yaml`](../examples/catalog.yaml)
- `gpu-broker init` writes it to `/etc/gpu-broker/catalog.yaml`.
- `gpu-broker check` validates it.

## A minimal catalog

One LLM and one ComfyUI model:

```yaml
defaults:
  resident: my-llm              # held on the card whenever nothing else needs it
  idle_restore_s: 120           # empty queue for this long → make `resident` resident again
  session_idle_s: 900           # interactive ComfyUI sessions (dashboard)
  session_yield_s: 120
  session_max_s: 14400
  vram_total_mib: 24564         # your card
  vram_reserve_mib: 600         # headroom; models above total - reserve are never loaded
  background_requesters: [batch-agent]   # x-requester values that never skip the queue

models:
  my-llm:
    kind: llm
    runner: llm_unit            # an OpenAI-compatible server the driver starts and stops
    unit: llama-server          # systemd unit or container name
    endpoint: http://127.0.0.1:8080
    served_name: llama-3.1-8b-instruct
    vram_mib: 7500
    slots: 4                    # matches llama-server -np 4
    reserved_interactive: 1     # background callers get 3 slots; one stays free for people
    variants: {my-llm-precise: {temperature: 0.1}}   # extra model id with request overrides
    caps: [chat, code]
    quality: 60
    status: ready
    aliases: [llama]

  sdxl:
    kind: image
    runner: comfy               # runs a graph on the shared ComfyUI
    template: sdxl              # a builder in gpu_broker/templates/
    params: {ckpt: sd_xl_base_1.0.safetensors}
    vram_mib: 9000
    caps: [t2i]
    quality: 60
    status: ready
```

## Runners

| `runner` | the model is | started by |
|---|---|---|
| `llm_unit` | an OpenAI-compatible server (llama.cpp, vLLM, ...) | the driver starts and stops its `unit` |
| `comfy` | a graph on the shared ComfyUI, built by a `template` | ComfyUI is shared; the broker frees its weights |
| `exec` | a command-line program | a host recipe: see [exec-recipes.md](exec-recipes.md) |

## Substitution

When a request names a model that is unknown, not installed, or too big for the card:

- The highest-`quality` ready model of the same `kind`, whose `caps` cover the request, runs instead.
- The response names the substitute and the reason.
- If the requested model can be downloaded, the download is queued as well.

## Input files

An entry that takes files declares them in `inputs`:

| `inputs` | for |
|---|---|
| `{image: required}` | image-to-video, image editing |
| `{image: optional, end_image: optional}` | an optional start frame, and an optional last frame |
| `{frames: one_of, video: one_of}` | several views of a scene *or* one video of it |
| `frames: {need: one_of, min: 2, max: 32}` | the same, with the number of views the model accepts |

- `one_of`: exactly one of those slots must be filled.
- Only ComfyUI models with a `template` (single images) and `exec` models take inputs.
- `image_caps`: capabilities a model has only with an input.
  - Example: `caps: [t2v]`, `image_caps: [i2v]` for an optional start frame.
  - A text-only job never demands them of a substitute.

How a job sends files:

- `image`, `end_image`, `video`: base64 or a `data:` URL.
- `frames`: a list of base64 images.
- `<slot>_url`: only with `inputs.allow_urls` on.
  - Reaches only public addresses and the broker's own ComfyUI outputs.
  - `inputs.url_allow_networks` adds more networks.

What the broker checks at submit (a failure is a 400, never a queued job that cannot run):

- Size caps.
- Format by magic bytes: PNG, JPEG, WebP; MP4, MOV, WebM.
- The frame count.
- That the model takes those files.
  - Substitutes are only models that take them.
  - A model that requires a file is never picked for a job without it.

ComfyUI inputs:

- Uploaded as `broker-<job id>-<slot>.<ext>` just before the graph runs.
- ComfyUI cannot delete inputs over its API. Prune them with
  [`examples/systemd/comfyui-input-prune.path`](../examples/systemd/comfyui-input-prune.path).

## Bundled templates

A catalog `template` names one of these graph builders.

- "Stock" means the nodes ship with a current ComfyUI release.
- For the others, install the node pack on your ComfyUI first.

| template | model family | needs |
|---|---|---|
| `sdxl` | single-checkpoint SD / SDXL | stock ComfyUI |
| `qwen_image` | Qwen-Image 2.1 | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) when `unet` is a `.gguf` file |
| `chroma` | Chroma1-HD | stock ComfyUI |
| `flux2_klein` | FLUX.2 Klein (optional LoRA) | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (`UnetLoaderGGUF`) |
| `flux2_klein_edit` | FLUX.2 Klein 9B image edit (`image` required) | as `flux2_klein` |
| `qwen_edit` | Qwen-Image 2.1 image edit with prompt enhancer (`image` required) | stock ComfyUI (plus ComfyUI-GGUF for a `.gguf` `unet`) |
| `wan14b` | Wan 2.2 14B video; `params.mode: t2v` or `i2v` (`image` required) | stock ComfyUI |
| `wan5b` | Wan 2.2 5B video, optional start `image` | stock ComfyUI |
| `hunyuan` | HunyuanVideo 1.5 480p text-to-video | stock ComfyUI |
| `hunyuan_i2v` | HunyuanVideo 1.5 720p image-to-video (`image` required) | stock ComfyUI |
| `minimax` | MiniMax H3 video with audio; optional `image` / `end_image` frames | stock ComfyUI |
| `ltx25` | LTX 2.5 text-to-video with audio | [ComfyUI-GGUF-Loader](https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader) (`LTXV25ModelsLoader`, `LTXV25AVDecode`; verified at commit `142c614`) |

`ltx25` notes:

- It takes four files in `params`: `unet`, `clip`, `video_vae` and `audio_vae`.
- Its defaults (97 frames at 768x512, 24 fps, 8 steps) took about 5 minutes on an RTX 4090.

## Per-model defaults

An entry may retune a template's request defaults with `defaults:`.

- Examples: `defaults: {steps: 20}`, or `defaults: {enhance: false}` on a `qwen_edit` model.
- Precedence: request > entry `defaults` > template defaults.
- Only keys the template reads as request options are accepted. Any other key fails at load.
- `params` stay the only way to choose files.

## Config

The config file sits beside the catalog. It is [`examples/config.yaml`](../examples/config.yaml).

- Every key and its default: [`gpu_broker/settings.py`](../gpu_broker/settings.py) and
  [`gpu_broker/tuning.py`](../gpu_broker/tuning.py).
- Secrets never go in it: `BROKER_TOKEN` and `UPSTREAM_TOKEN_*` come from the environment.
