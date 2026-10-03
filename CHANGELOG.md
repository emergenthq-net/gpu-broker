# Changelog

## Unreleased

### Changed
- Internal: modules that did two things are split along that seam, with no behaviour change.
  `catalogschema` (entry types and validation) out of `catalog`; `settingsschema` (the
  dataclasses) out of `settings`; `admission` (job submit) out of `broker`;
  `drivers.systemd` and `drivers.docker` out of `drivers.local`; `demo.assemble` out of
  `demo.run`. The old modules re-export the moved names, except the two concrete drivers:
  import `SystemdDriver` and `DockerDriver` from `gpu_broker.drivers.systemd` and
  `gpu_broker.drivers.docker` (a re-export from `drivers.local` would be circular).

## 0.3.2

### Fixed
- The sdist is self-testing: unpack it, install it with `[dev]`, run `pytest`. `MANIFEST.in`
  ships `tests/` (with `tests/helpers.py` and the fixtures), `host/`, `examples/` and `scripts/`.
  The amdgpu fixtures' `/proc/<pid>/fd` symlinks, which an sdist drops because they dangle,
  are listed in `tests/fixtures/amdgpu/fd-links.txt` and recreated in a temp dir at test time.
  CI checks this on every change.

## 0.3.1

### Changed
- Tests and examples use neutral container ids and paths, and the demo's leak check reads its
  denylist from `GPU_BROKER_LEAK_DENYLIST` instead of spelling it out. New
  `scripts/leak_scan.py` scans the tree and the built sdist and wheel against that denylist;
  CI (job `leak-scan`) and the release workflow run it before anything is published.

## 0.3.0

### Added
- `gpu-broker demo`: the real broker, API and dashboard on a simulated GPU, with a simulated
  team sending chats, images and videos. No graphics card, model servers or downloads needed;
  it opens the dashboard already signed in (`#token=` link). See README, Try it.
- Drop-in replacement for the OpenAI and Anthropic APIs: `POST /v1/messages` (Anthropic
  Messages, plain and streaming, tools, images, thinking), `POST /v1/embeddings` (routed to a
  `caps: [embed]` model), `x-api-key` auth beside `Authorization: Bearer`, and `model_map`
  (`BROKER_MODEL_MAP`) mapping hosted names such as `gpt-*` and `claude-*` to catalog models.
  See docs/drop-in.md.
- Input images: jobs may carry `image` / `end_image` (base64 or a data: URL), or
  `image_url` / `end_image_url` when `images.allow_urls` is on (off by default; fetches are
  limited to public addresses unless `url_allow_networks` allows more). Catalog entries say
  which images a model takes (`images:`), and substitution only picks models that take them.
- Image-to-video and image-edit templates: Wan 2.2 14B i2v, Wan 2.2 5B with a start image,
  MiniMax first/last frame, HunyuanVideo 1.5 i2v, Qwen-Image edit and FLUX.2 Klein edit.
- Exec runner (`runner: exec`): a model can be a program run from a recipe file
  (`recipes_dir`) on the GPU's machine, with no shell, fixed paths and validated parameters
  (`exec.params`, `exec.choices`). `examples/recipes/sharp.recipe` turns one image into a 3D
  Gaussian splat. Recipe `outputs` may list several globs. Prune units for old job folders are
  in `examples/systemd/`.
- Per-model request defaults in the catalog (`defaults:`): request > catalog entry > template.
- The dashboard can run image-taking models with an uploaded image, and its model index says
  who has the GPU: the resident model, the job it is lent to, or nobody.
- AMD GPUs: the amdgpu driver's sysfs files and `/proc/*/fdinfo`, no ROCm tools needed
  (`gpu.vendor`, `gpu.index`; on Proxmox `host/gpu-broker-gpu`). See README, Hardware.
- `/v1/metrics` samples carry `procs_unreadable` (per-process memory could not be read: it
  needs root or `CAP_SYS_PTRACE`), and `/v1/gpu` names the GPU reader in use (`probe`).

### Changed
- GPU sample field `sm_mhz` is now `clock_mhz` (the vendor-neutral name).
- Only VRAM used and total are required in a GPU reading; utilisation, power, temperature and
  clock may be `null` (unknown) in `/v1/gpu` and `/v1/metrics`.
- `serve` starts when no GPU can be read yet, and keeps retrying, instead of refusing to start.
  `gpu.vendor: auto` is chosen in the background, within `timeouts.gpu_query_s` of wall-clock
  time; `/v1/gpu` answers `{"state": "probing"}` meanwhile and never waits on `nvidia-smi`.
- Proxmox host script: `GPU_BUDGET_S` (default 8) bounds a whole `gpu` reading, the vendor
  choice included; `NV_TIMEOUT_S` now bounds each `nvidia-smi` call only.

### Deprecated
- `sm_mhz` in `/v1/metrics` samples: kept as an alias of `clock_mhz` for this release only;
  it is removed in the next one.
