# Architecture

gpu-broker currently owns one GPU and decides what is resident on it. Requests resolve from intent to a concrete
model, become durable jobs, and are ordered by a pure scheduling policy. A single GPU thread remains the only
residency writer; before each job it makes exactly the needed model resident.

## Layers

```
web/            HTTP: auth, routes, dashboard            (FastAPI; no logic of its own)
broker.py       composition root: submit / view / wait   (wires everything below)
chat.py         interactive chat straight to the resident LLM (priority, variants)
scheduler.py    single residency-writer GPU thread: dispatch + idle restore
policy.py       pure pending-job ordering: priority, aging, residency locality (or FIFO)
  llmpool.py      concurrent calls to the resident LLM
  sessions.py     interactive ComfyUI sessions
  residency.py    what is on the card; switch safely
downloads.py    model downloads on their own thread
resolve.py      intent → model | substitute | download | rejection; supports auto routing (pure)
inventory.py    catalog → capability/runtime/resource summary         (pure)
catalog.py      trusted model/runtime contracts + safe writes
templates/      ComfyUI graph builders                   (pure)
backends.py     allowlisted HTTP protocols to model servers + ComfyUI (the only HTTP client)
drivers/        start/stop units, read the GPU, fetch files  (the only subprocesses)
store.py        SQLite jobs/events/downloads + JSONL event log
metrics.py      GPU samples, job latency/throughput
settings.py     typed configuration;  constants.py: protocol vocabulary
```

Dependencies point downwards only. `resolve` and `templates` are pure functions over plain
data and are tested by table. Everything that touches the outside world sits behind a
small interface — `Driver` (processes, files), `Backends` (HTTP), `Store` (disk) — that the
tests replace with in-memory fakes.

## Invariants

1. **Only the GPU thread changes residency, and only between jobs.** No lock guards
   `Residency`; the single-writer rule does.
2. **Nothing is mid-call when a model is stopped.** Every LLM call — queued or direct —
   holds a pool slot. Before any switch the GPU thread *closes* the pool (no new direct
   calls) and drains it; afterwards it reopens it naming the new resident. A direct caller
   can only ever reach the model the pool names, and gives up if it changes while waiting.
3. **Queue policy is pure; execution stays single-writer.** `policy.py` may reorder only pending jobs. The default
   `balanced` policy uses interactive/normal/background bands, age promotion to prevent starvation, and resident
   locality as a lower-order preference. `fifo` preserves strict submission order. Interactive chat on the already
   resident model may still bypass the queue, and background direct work may never fill `reserved_interactive` slots.
4. **State is re-checked, not trusted.** Before reusing the resident LLM its health check
   must pass; ComfyUI is freed before every LLM start because it may have been used directly;
   a stopped ComfyUI is restarted if a unit is configured. At startup the running LLM is
   adopted and non-terminal jobs from the previous process are failed as orphans.
5. **Resolution is a contract, not a guess.** Explicit `kind`/`caps` requirements are enforced even for a known model.
   `auto` selects only runnable compatible models. A job that runs a different implementation carries the reason;
   `/v1/resolve` exposes the same decision without executing anything.
6. **The broker only reaches configured addresses and allowlisted paths.** Backend URLs come from trusted config/catalog,
   never from a request; registered downloads never get an endpoint. Compatibility routes map only to a fixed protocol
   allowlist, optionally narrowed per model by `api_paths`; callers cannot turn the broker into an arbitrary proxy.

## A request's path

```
POST /v1/chat/completions (interactive, model resident, pool open)
     ─► DirectChat.open ─► LlmPool.acquire ─► Backends.llm_stream | llm_chat ─► release
     (anything else falls through to the job path below)
POST /v1/resolve ─► auth ─► resolve ─► explanation only (no store / switch / download)
POST /v1/jobs ─► auth ─► Broker.submit ─► resolve ─► Store.create_job ─► Scheduler.submit
                                    └─► Downloader.request (if files are missing)
Scheduler policy ─► next pending job
GPU thread ─► resident LLM with free slots? ─► LlmPool.dispatch ─► Backends.llm_request
          └─► drain pool ─► Residency.ensure ─► session hold | llm_chat | template ─► comfy_run
idle ─► Scheduler.maybe_restore ─► Residency.ensure(defaults.resident)
```

## Threads

| thread | loop | stops on |
|---|---|---|
| GPU | `Scheduler.loop` | `Broker.stop()` (after the running job) |
| downloads | `Downloader.loop` | `Broker.stop()` |
| GPU sampler | `GpuSampler.loop` (when `gpu_stream`) | `Broker.stop()` |
| one per queued LLM call | `LlmPool._call` | the call returning |
| request thread (direct chat) | `web/chat.py` relay | the stream ending |

## Extending

- **A model** is a catalog entry: `runner` (`llm_unit`, `comfy` or `external`), what it needs (`vram_mib`),
  what it can do (`kind`, `caps`, `quality`) and how to run it. Managed HTTP runtimes can additionally declare
  `health_path`, `metrics_path` and `api_paths`; runtime brand is intentionally not part of the scheduler ontology.
- **A ComfyUI graph** is a builder in `templates/` taking `(request, params, output prefix)`.
- **A host** is a `Driver`: `unit`, `gpu`, `gpu_stream`, `download`, `link`.

## Direction

The single NVIDIA GPU is the first concrete resource implementation, not the final ontology. The next architectural seam is an explicit resource topology (`device → memory/residency → runtime`) so placement can expand to multiple GPUs, CPU/NPU and optional remote targets without changing capability resolution or the queue-policy interface.
