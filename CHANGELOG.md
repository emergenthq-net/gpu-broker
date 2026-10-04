# Changelog

All notable changes to gpu-broker. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-10-04

### Added
- `gpu-broker connect` against a broker with cloud failover keeps Claude Code's and Codex's own
  provider keys and adds the broker key as `x-gpu-broker-key`; other clients keep the broker key
  alone. `gpu-broker clients` shows which keys each client sends. Connect asks
  `GET /v1/upstreams/passthrough`, which a client key may read (only whether failover is on and
  which APIs pass a client's key through); `/v1/upstreams` stays main-token only.
- Failover: unstreamed requests go upstream as streams and are reassembled into the provider's
  exact unstreamed answer, so a down provider fails over after `first_byte_s` and a long answer
  is never cut off; a provider that refuses streams is asked again unstreamed.
- Cloud first, local on failure (`upstreams:`, docs/failover.md): `claude-*` / `gpt-*` go to the
  real provider with the client's own key and fall back to a local model on an outage, a hang,
  5xx/529 or used-up quota or credit; client errors are returned as-is. A circuit breaker per
  provider (and per key for quota) skips a failing provider at once and probes it back.
  Answers carry `x-gpu-broker-served-by` and `x-gpu-broker-fallback`; a dashboard card shows
  the breakers and failover events. The broker credential may be sent as `x-gpu-broker-key`.
  A client's own key goes only to providers with `pass_client_key` (the official hosts by default).
- An MCP server (the `mcp` extra): tools to list models, generate images and video, edit
  images, animate an image, make 3D splats, poll and fetch jobs, ask the local LLM and read the
  GPU. Streamable HTTP at `/mcp` behind the chat-route credentials (a client key sees only its
  own jobs), and `gpu-broker mcp` over stdio. Long jobs return their id after `mcp.wait_s`;
  small images come back inline. `gpu-broker connect` registers it with Claude Code, Claude
  Desktop and Codex (`claude-code-mcp`, `claude-desktop`, `codex-mcp`). `/health` reports `mcp`.
  A client key sees only jobs it owns (by key id) and sends files as base64 unless
  `mcp.client_url_inputs`; tools take only models that can run now.
- A video's length is the request param `num_frames`; `frames` is only the list of input views
  (a number there is refused with a pointer to `num_frames`).
- `comfy.auth_env`: a ComfyUI behind a Bearer token.
- `connect` writes symlinked configs through, never replaces a user's own `gpu-broker` MCP
  server, warns when a running Claude Code drops its entry, and lists the files it changed.
- The GPU thread's queue is `fair` by default: interactive jobs first, then the requester that
  has used the least expected GPU time, so one busy client no longer holds everyone else's jobs
  behind its backlog.
  - A call waiting for a pool slot no longer blocks the jobs behind it.
  - `scheduler.policy: fifo` (env `BROKER_SCHEDULER_POLICY`) restores strict arrival order.
  - Background jobs age into the interactive class after `scheduler.max_wait_s`.
  - A switch never jumps a call waiting for a slot on the resident model for longer than
    `scheduler.evict_wait_s` unless it is of a higher class.
  - `/v1/jobs` takes `x-priority`; without it, a job's class follows `defaults.background_requesters`.
- `gpu-broker replay <events.jsonl>`: replays a broker's event log through the queue rules in
  simulated time and reports waits per requester, jobs/hour and residency churn, beside what the
  broker did. Offline; the yardstick for scheduler changes.

### Fixed
- MCP `initialize` reports the installed gpu-broker version in `serverInfo` (it was empty).

### Changed
- With `fair`, a `/v1/jobs` LLM call from a requester not in `defaults.background_requesters`
  (and without `x-priority: background`) is interactive, so it may use the slots kept by
  `reserved_interactive`; before, only chat routes classified their calls.
- With `fair`, a requester in `defaults.background_requesters` can only lower its class with
  `x-priority` (or `interactive` in a job), not raise it. `scheduler.may_claim_interactive`
  lists who may claim interactive (default: everyone not in `background_requesters`). `fifo`
  keeps the old rule.
- While quiesced (`/v1/admin/quiesce`), new jobs, chats, embeddings and sessions get HTTP 503
  with `Retry-After` (`intervals.quiesced_retry_s`, default 5) and `x-should-retry: true`,
  instead of being queued and then failed as orphans by the restart.
- A restart re-queues jobs the previous process had queued but not started, in their original
  order and with their staged input files (event `job.requeued`).
  - Jobs that had started, and direct chats, are still failed as orphaned.
  - A re-queued job fails with "input files missing" if its files are gone, and with
    "model '<key>' no longer in catalog" if its model left the catalog; a re-queued interactive
    session fails with "session expired by restart".

## [0.4.0] - 2026-10-03

### Added
- `gpu-broker setup`: one command from install to a running broker, asking no questions.
  - Finds the GPU, and llama.cpp, vLLM, Ollama and ComfyUI on their usual ports, with the
    systemd units or Docker containers that run them.
  - Writes the config and catalog for them (the starter files if it finds nothing) and a new
    API token in `broker.env` (mode 600). Existing files and the token are kept.
  - Installs and starts the `gpu-broker` service, never as root: a user service (lingering)
    for `systemctl --user` model servers; else, with root or passwordless sudo, a system
    service run as the installation's owner or a dedicated `gpu-broker` account, with a
    visudo-checked sudoers rule for exactly the catalog's units. It refuses code anyone else
    could change. Otherwise it prints the `serve` command, and runs it in the foreground at a
    terminal.
  - Runs `check`, waits for `/health`, and opens the dashboard (not over SSH).
  - `--dry-run`, `--yes`, `--dir`. See `docs/setup.md`.
- Catalog: `health_path` for LLM servers without `/health` (Ollama: `/api/version`).
- README: a TLDR block first, and the logo (light and dark).
- `gpu-broker init`: writes a starter `config.yaml` and `catalog.yaml` to `/etc/gpu-broker`
  (or `--dir`) and creates the folders they name.
  - Existing files are kept unless `--force`.
- Docs pages: `docs/catalog.md`, `docs/exec-recipes.md`, `docs/api.md`, `docs/hardware.md`,
  `docs/threat-model.md`, and `AGENTS.md` for automated reviewers.

### Changed
- README: leads with the one-person case; installs from PyPI; reference material moved to `docs/`.
- Internal: modules that mixed two responsibilities are split, with no behaviour change.
  - `catalogschema` (entry types and validation) out of `catalog`.
  - `settingsschema` (the dataclasses) out of `settings`.
  - `admission` (job submit) out of `broker`.
  - `drivers.systemd` and `drivers.docker` out of `drivers.local`.
  - `demo.assemble` out of `demo.run`.
  - The old modules re-export the moved names, except the two concrete drivers: import
    `SystemdDriver` and `DockerDriver` from `gpu_broker.drivers.systemd` and `gpu_broker.drivers.docker`.

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

[Unreleased]: https://github.com/emergenthq-net/gpu-broker/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/emergenthq-net/gpu-broker/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/emergenthq-net/gpu-broker/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/emergenthq-net/gpu-broker/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/emergenthq-net/gpu-broker/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/emergenthq-net/gpu-broker/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/emergenthq-net/gpu-broker/releases/tag/v0.2.0
