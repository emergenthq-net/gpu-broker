# Architecture

gpu-broker owns one GPU and decides what is resident on it. Requests become jobs; jobs run
in order on a single GPU thread; before each job the broker makes exactly the needed model
resident.

## Layers

```
web/            HTTP: auth, routes, dashboard            (FastAPI; no logic of its own)
broker.py       composition root: submit / view / wait   (wires everything below)
chat.py         interactive chat straight to the resident LLM (priority, variants)
scheduler.py    the GPU thread: FIFO, dispatch, idle restore
  llmpool.py      concurrent calls to the resident LLM
  sessions.py     interactive ComfyUI sessions
  residency.py    what is on the card; switch safely
downloads.py    model downloads on their own thread
resolve.py      request → model | substitute | download | rejection   (pure)
catalog.py      the model catalog (trusted config) + safe writes
templates/      ComfyUI graph builders                   (pure)
backends.py     HTTP to LLM servers and ComfyUI          (the only HTTP client)
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
3. **FIFO for queued work; a fast lane for people.** A queued call that finds its slots busy
   holds the GPU thread until one frees, so a later job never overtakes an earlier one.
   Interactive chat on the resident model skips the queue, and background work may never
   fill the model's `reserved_interactive` slots, so one is always free for a person.
4. **State is re-checked, not trusted.** Before reusing the resident LLM its health check
   must pass; ComfyUI is freed before every LLM start because it may have been used directly;
   a stopped ComfyUI is restarted if a unit is configured. At startup the running LLM is
   adopted and non-terminal jobs from the previous process are failed as orphans.
5. **Substitution is always reported.** A job that runs a different model than requested
   carries the reason; nothing is silently swapped.
6. **The broker only reaches configured addresses.** Backend URLs come from the config file
   and catalog, never from a request; registered (downloaded) models never get an endpoint.

## A request's path

```
POST /v1/chat/completions (interactive, model resident, pool open)
     ─► DirectChat.open ─► LlmPool.acquire ─► Backends.llm_stream | llm_chat ─► release
     (anything else falls through to the job path below)
POST /v1/jobs ─► auth ─► Broker.submit ─► resolve ─► Store.create_job ─► Scheduler.submit
                                    └─► Downloader.request (if files are missing)
GPU thread ─► resident LLM with free slots? ─► LlmPool.dispatch ─► Backends.llm_chat
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

- **A model** is a catalog entry: `runner` (`llm_unit`, `comfy` or `external`), what it needs
  (`vram_mib`), what it can do (`kind`, `caps`, `quality`) and how to run it.
- **A ComfyUI graph** is a builder in `templates/` taking `(request, params, output prefix)`.
- **A host** is a `Driver`: `unit`, `gpu`, `gpu_stream`, `download`, `link`.
