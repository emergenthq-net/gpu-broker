# Architecture

gpu-broker owns one GPU and decides what is resident on it. Requests become jobs; jobs run
in order on a single GPU thread; before each job the broker makes exactly the needed model
resident.

## Layers

```
web/            HTTP: auth, routes, dashboard            (FastAPI; no logic of its own)
mcp_server/     MCP tools (core.py, SDK-free) at /mcp, and `gpu-broker mcp` (stdio relay)   (the `mcp` extra)
broker.py       composition root: submit / view / wait   (wires everything below)
  admission.py    submit: check, resolve, record, stage, queue a job
chat.py         interactive chat straight to the resident LLM (priority, variants)
scheduler.py    the GPU thread: pick, dispatch, idle restore
  line.py         waiting jobs in the policy's order; which may start (choose); expected GPU-seconds
classes.py      a job's class: interactive or background, and who may claim which
  jobline.py      which re-queued jobs can still run
  llmpool.py      concurrent calls to the resident LLM
  sessions.py     interactive ComfyUI sessions
  residency.py    what is on the card; switch safely
downloads.py    model downloads on their own thread
failover/       cloud first, local on failure: upstreams config, error classes, breakers, HTTP client
resolve.py      request → model | substitute | download | rejection   (pure)
inputs.py       input slots: request shape, model fit (catalog `inputs`)
media.py        input file data: base64, format by magic bytes, size; <slot>_url fetch
netguard.py     <slot>_url connections: public addresses only, vetted per connection and hop
deadline.py     <slot>_url time limit: the fetch deadline, bounded name lookups
execjob.py      runner exec: staged inputs → driver.run_recipe → outputs
staging.py      input files on disk between submit and run; upload to ComfyUI
catalog.py      the loaded model catalog (trusted config) and its safe writes
  catalogschema.py  catalog entry types and validation rules
templates/      ComfyUI graph builders                   (pure)
backends.py     HTTP to LLM servers and ComfyUI          (the only client of configured addresses)
drivers/        start/stop units, read the GPU, fetch files, run exec recipes  (the only subprocesses of `serve`)
store.py        SQLite jobs/events/downloads/flags + JSONL event log
schema.py       the SQLite schema and its migrations
gpu/            GPU probes: nvidia-smi, amdgpu sysfs + DRM fdinfo; the sample-line format
metrics.py      GPU samples, job latency/throughput
settings.py     loading configuration: YAML file + environment
  settingsschema.py the configuration dataclasses (tuning.py: the tuning sections)
constants.py    protocol vocabulary
policy.py       queue policies: the order the GPU thread considers waiting jobs in
replay/         an events.jsonl through a policy in simulated time  (offline; `gpu-broker replay`)

cli.py          the console script: serve, check, init, setup, demo, replay
connect/        `gpu-broker connect | disconnect | clients`: point local tools at the broker
setup/          `gpu-broker setup`: detect servers, write config + token, install the service
                (runs once, at install; its own argv-only subprocesses: systemctl, sudo -n, uv,
                useradd, visudo, loginctl)
starter/        `gpu-broker init`: the starter config and catalog, and the writer setup uses
browser.py      open a link on this machine's screen, never over SSH (demo, setup)
demo/           `gpu-broker demo`: the real broker on a simulated card
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
   adopted; jobs the previous process had queued but not started are re-queued in order (unless
   their model left the catalog or they were sessions: those fail with a reason), and
   any other non-terminal job (it had started, or was a direct chat) is failed as an orphan.
5. **Substitution is always reported.** A job that runs a different model than requested
   carries the reason; nothing is silently swapped.
6. **The broker only reaches configured addresses.** Backend URLs come from the config file
   and catalog, never from a request; registered (downloaded) models never get an endpoint.
   The one exception is opt-in: with `inputs.allow_urls` set, `media.fetch` downloads a
   caller's `<slot>_url` (http(s) only, public addresses unless allowlisted, plus output
   views of the broker's own ComfyUI — checked on every connection and redirect hop by
   netguard — size-capped, with a total deadline from deadline.py).
7. **Input files are checked before a job exists.** Slot shape, model fit (the catalog's
   `inputs`), size and format (magic bytes) are all verified in `Broker.submit`; a failure is
   a 400 and nothing is recorded. Image data never enters the job payload: it is staged on
   disk, uploaded to ComfyUI by the GPU thread right before the graph is built, and deleted
   when the job ends. At startup the files of failed orphans are cleared; re-queued jobs keep theirs.

## A request's path

```
POST /v1/chat/completions (interactive, model resident, pool open)
     ─► DirectChat.open ─► LlmPool.acquire ─► Backends.llm_stream | llm_chat ─► release
     (anything else falls through to the job path below)
POST /v1/jobs ─► auth ─► Broker.submit ─► inputs.slots ─► resolve ─► inputs.check ─► media.read
                      ─► Store.create_job ─► Staging.put ─► Scheduler.submit
                                    └─► Downloader.request (if files are missing)
GPU thread ─► resident LLM with free slots? ─► LlmPool.dispatch ─► Backends.llm_chat
          └─► drain pool ─► Residency.ensure ─► session hold | llm_chat
                                             | Staging.upload ─► template ─► comfy_run
                                             | ExecJobs.run ─► Driver.run_recipe
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

8. **No command comes from a request.** An exec job names a recipe the host defines
   (drivers/recipes.py, host/gpu-broker-ctl); only the recipe name, job id and input file
   names reach the driver, and both ends validate them with the same grammars.

- **A model** is a catalog entry: `runner` (`llm_unit`, `comfy`, `exec` or `external`), what it needs
  (`vram_mib`), what it can do (`kind`, `caps`, `quality`) and how to run it.
- **A ComfyUI graph** is a builder in `templates/` taking `(request, params, output prefix)`.
  Input images arrive as request keys (`image`, `end_image`) holding the uploaded file name;
  load them with `_graph.load_image(input_image(req, slot, name))`. Declare them in the
  catalog entry's `inputs` so the broker checks and routes them.
- **A command-line model** is a recipe file on the host plus a catalog entry with
  `runner: exec` and `exec: {recipe, timeout_s}`; residency treats it like a ComfyUI job that
  needs the whole card, and waits until the GPU shows its `vram_mib` free. A recipe that may
  still be running after its job (drivers.GpuHeld) sets the persisted GPU hold (holds.py):
  the scheduler runs nothing until a clean, retried on its own thread, confirms the job gone
  or an operator clears it (either wakes the GPU thread at once). `Broker.start` sets the hold
  for an exec job the previous process left running. Recipe timings are checked (and cached
  per recipe) before residency evicts anything. A job's processes are found by the GPU_BROKER_JOB=<job id> tag in their
  environment (drivers/reap.py; `reap_job` in the host script), so workers that leave the
  recipe's process group are found too.
- **A host** is a `Driver`: `unit`, `gpu`, `gpu_stream`, `download`, `link`, `recipe_info`,
  `run_recipe`, `clean_recipe`.
