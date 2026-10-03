# Changelog

All notable changes to gpu-broker. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `gpu-broker setup`: one command from install to a running broker, asking no questions.
  - Finds the GPU, and llama.cpp, vLLM, Ollama and ComfyUI on their usual ports, with the
    systemd units or Docker containers that run them.
  - Writes the config and catalog for them (the starter files if it finds nothing) and a new
    API token in `broker.env` (mode 600). Existing files and the token are kept.
  - Installs and starts the `gpu-broker` service with root or passwordless sudo; otherwise
    prints the `serve` command, and runs it in the foreground at a terminal.
  - Runs `check`, waits for `/health`, and opens the dashboard (not over SSH).
  - `--dry-run`, `--yes`, `--dir`. See `docs/setup.md`.
- Catalog: `health_path` for LLM servers without `/health` (Ollama: `/api/version`).
- README: a TLDR block first.
- `gpu-broker init`: writes a starter `config.yaml` and `catalog.yaml` to `/etc/gpu-broker`
  (or `--dir`) and creates the folders they name.
  - Existing files are kept unless `--force`.
- Docs pages: `docs/catalog.md`, `docs/exec-recipes.md`, `docs/api.md`, `docs/hardware.md`,
  `docs/threat-model.md`, and `AGENTS.md` for automated reviewers.

### Changed
- README: leads with the one-person case; installs from PyPI; reference material moved to `docs/`.

## [0.3.2] - 2026-10-03

### Fixed
- The sdist is self-testing: unpack it, install it with `[dev]`, run `pytest`.
  - `MANIFEST.in` ships `tests/` (with helpers and fixtures), `host/`, `examples/` and `scripts/`.
  - The amdgpu fixtures' `/proc/<pid>/fd` symlinks are listed in
    `tests/fixtures/amdgpu/fd-links.txt` and recreated at test time.
  - CI checks this on every change.

## [0.3.1] - 2026-10-03

### Changed
- Tests and examples use neutral container ids and paths.

### Security
- `scripts/leak_scan.py` scans the tree and the built sdist and wheel against a private denylist.
  - CI (job `leak-scan`) and the release workflow run it before anything is published.
  - The demo's leak check reads its denylist from `GPU_BROKER_LEAK_DENYLIST`.

## [0.3.0] - 2026-10-03

### Added
- `gpu-broker demo`: the real broker, API and dashboard on a simulated GPU.
  - No graphics card, model servers or downloads needed.
  - Opens the dashboard already signed in (`#token=` link).
- Drop-in replacement for the OpenAI and Anthropic APIs (see `docs/drop-in.md`):
  - `POST /v1/messages`: Anthropic Messages, plain and streaming, tools, images, thinking.
  - `POST /v1/embeddings`, routed to a `caps: [embed]` model.
  - `x-api-key` auth beside `Authorization: Bearer`.
  - `model_map` (`BROKER_MODEL_MAP`): hosted names such as `gpt-*` and `claude-*` map to catalog models.
- Input files: `image`, `end_image`, `frames` and `video`, as base64 or a `data:` URL.
  - `<slot>_url` when `inputs.allow_urls` is on (off by default); public addresses only,
    unless `inputs.url_allow_networks` allows more.
  - Catalog entries declare what they take (`inputs:`); substitution picks only models that take them.
- Image-to-video and image-edit templates: Wan 2.2 14B i2v, Wan 2.2 5B with a start image,
  MiniMax first/last frame, HunyuanVideo 1.5 i2v, Qwen-Image edit, FLUX.2 Klein edit.
- Exec runner (`runner: exec`): a model can be a program run from a host recipe file.
  - No shell, fixed paths, validated parameters (`exec.params`, `exec.choices`).
  - `examples/recipes/sharp.recipe` turns one image into a 3D Gaussian splat.
  - Prune units for old job folders in `examples/systemd/`.
- Per-model request defaults in the catalog (`defaults:`): request > catalog entry > template.
- Dashboard: run image-taking models with an uploaded image; the model index says who holds the GPU.
- AMD GPUs, through amdgpu sysfs and `/proc/*/fdinfo`; no ROCm tools needed.
  - `gpu.vendor`, `gpu.index`; on Proxmox, `host/gpu-broker-gpu`.
- `/v1/metrics` samples carry `procs_unreadable`; `/v1/gpu` names the GPU reader in use (`probe`).

### Changed
- GPU sample field `sm_mhz` is now `clock_mhz`.
- Only VRAM used and total are required in a GPU reading; the rest may be `null`.
- `serve` starts when no GPU can be read yet, and keeps retrying.
  - `gpu.vendor: auto` is chosen in the background, within `timeouts.gpu_query_s`.
  - `/v1/gpu` answers `{"state": "probing"}` meanwhile.
- Proxmox host script: `GPU_BUDGET_S` (default 8) bounds a whole reading; `NV_TIMEOUT_S`
  bounds each `nvidia-smi` call.

### Deprecated
- `sm_mhz` in `/v1/metrics` samples: kept as an alias of `clock_mhz` for this release only.

## [0.2.0] - 2026-10-01

Initial public release.

### Added
- An HTTP broker that queues requests and swaps LLM servers and ComfyUI in and out of one GPU.
- OpenAI-compatible `/v1/chat/completions` and `/v1/models`.
  - Interactive chat on the resident model skips the queue and streams.
- The job API (`/v1/jobs`), substitution with the reason reported, and background downloads
  by Hugging Face repo or GitHub URL.
- Idle restore of the resident model, interactive ComfyUI sessions, and a web dashboard.
- Host drivers: systemd, Docker, and systemd units in Proxmox LXCs via a forced-command script.
- SQLite job store and JSONL event log.

[Unreleased]: https://github.com/emergenthq-net/gpu-broker/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/emergenthq-net/gpu-broker/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/emergenthq-net/gpu-broker/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/emergenthq-net/gpu-broker/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/emergenthq-net/gpu-broker/releases/tag/v0.2.0
