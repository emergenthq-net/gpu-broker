# API

Every route except `/health` and the dashboard page needs `Authorization: Bearer $BROKER_TOKEN`.

For the OpenAI and Anthropic SDKs, see [drop-in.md](drop-in.md).

## Routes

| endpoint | what it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat. Broker details are in `x_broker`. |
| `POST /v1/jobs` | Run any model as a job. |
| `GET /v1/jobs/{id}` | The job's state, the model it used, and its outputs. |
| `GET /v1/models` | Ready LLMs and their variants (OpenAI format). |
| `GET /v1/catalog` | The catalog. |
| `GET /v1/status` | Residency, queue, downloads, recent jobs, any GPU hold. |
| `GET /v1/events?since=N` | The event log. |
| `GET /v1/gpu`, `/v1/metrics`, `/v1/stats`, `/v1/ui` | Dashboard data: GPU reading, live samples (job latency, tok/s), per-model stats, labels. |
| `POST /v1/sessions`, `/v1/sessions/end` | Borrow the GPU for interactive ComfyUI, and give it back. |
| `POST /v1/admin/quiesce`, `/v1/admin/resume` | Drain in-flight calls before a restart, and undo that. |
| `POST /v1/admin/gpu-held/clear` | Lift a GPU hold ([exec-recipes.md](exec-recipes.md#gpu-hold-after-a-crash)). |
| `GET /health` | Liveness check. No auth. |
| `GET /dash` | The dashboard. |

## Chat

- An interactive caller on the resident model is served directly.
  - With `stream: true`, tokens stream as they are generated.
- Other calls are queued.
  - A queued call answered with `stream: true` arrives as a single SSE chunk.

Optional headers:

| header | effect |
|---|---|
| `x-requester` | Labels the caller (dashboard, event log). |
| `x-priority: interactive\|background` | Overrides the catalog's `background_requesters`. |

## Jobs

Request body:

| field | |
|---|---|
| `model` | Catalog model, alias or variant. Unknown names are substituted ([catalog.md](catalog.md#substitution)). |
| `kind`, `caps` | Optional: what a substitute must be able to do. |
| `prompt` or `messages` | The prompt. |
| `image`, `end_image`, `video` | Input files: base64 or `data:` URL. `<slot>_url` when enabled. |
| `frames` | A list of base64 images. |
| `wait`, `wait_s` | Block until the job ends, up to `wait_s`. |
| anything else | A parameter for the model's ComfyUI template. Exec models take only the keys in `exec.params`. |

The response gives:

- `requested`, `resolved` and `substitution`: what was asked for, what runs, and why.
- `queue_position`.
- `download`, when a download was queued.

A malformed file, or one the model cannot take, is a 400.

## Examples

```bash
T="Authorization: Bearer $BROKER_TOKEN"
# Chat. Served directly by the resident model when it is up; `"stream": true` streams tokens.
curl -s localhost:8095/v1/chat/completions -H "$T" -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"hi"}]}'
# Any model as a job: the LLM stops, ComfyUI renders, and the LLM returns once the queue is idle.
curl -s localhost:8095/v1/jobs -H "$T" -H 'Content-Type: application/json' \
  -d '{"model":"wan2.2-5b","prompt":"a fox in snow","wait":true}'
# Image-to-video or image edit: send the image as base64 (or a data: URL).
curl -s localhost:8095/v1/jobs -H "$T" -H 'Content-Type: application/json' \
  -d "{\"model\":\"wan2.2-5b\",\"prompt\":\"the fox runs\",\"image\":\"$(base64 < fox.png | tr -d '\n')\"}"
curl -s localhost:8095/v1/status -H "$T"       # residency, queue, downloads, recent jobs
```
