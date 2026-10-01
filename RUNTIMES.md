# Runtime support

gpu-broker separates **inference protocol** from **GPU residency lifecycle**.

A model server may speak an OpenAI-compatible HTTP API while using one of several ways to
give GPU memory back. The catalog's `residency` field selects that lifecycle:

| `residency` | server process | model/GPU state | use when |
|---|---|---|---|
| `unit` (default) | stopped / started by the host driver | process lifetime owns residency | llama.cpp, SGLang, TensorRT-LLM, vLLM, or any dedicated server |
| `vllm_sleep` | stays running | vLLM level-1 sleep / wake | a vLLM server configured for sleep mode |
| `ollama` | shared daemon stays running | catalog model is loaded / unloaded with Ollama's native API | several catalog models share one Ollama daemon |

The scheduler does not know these details. It asks residency to make a catalog model
resident; residency performs the configured lifecycle transition.

## Generic OpenAI-compatible servers

`llm_unit` is not a llama.cpp-specific runner. It means "an HTTP model server whose
lifecycle gpu-broker manages". The catalog declares the compatibility contract:

```yaml
models:
  coder:
    kind: llm
    runner: llm_unit
    residency: unit
    unit: coder-server
    endpoint: http://127.0.0.1:8000
    served_name: my-coder
    health_path: /health
    metrics_path: /metrics
    api_paths:
      - /v1/chat/completions
      - /v1/completions
    vram_mib: 12000
    caps: [chat, code]
    quality: 80
    status: ready
```

`health_path` and `metrics_path` must be absolute paths. `api_paths` may only contain
protocol paths gpu-broker itself knows; it can narrow support but can never create an
arbitrary proxy route.

### llama.cpp

Use `residency: unit`. The server is started/stopped through the configured systemd,
Docker or Proxmox driver. Chat/completions can use the broker's streaming fast path.

### SGLang

Use `residency: unit` for a dedicated SGLang server. Current SGLang exposes
OpenAI-compatible chat/completions and additional APIs such as Responses, rerank and score.
It exposes `/health` and `/ready`; Prometheus metrics are available at `/metrics` when
SGLang is launched with metrics enabled. Declare only the paths enabled by your deployment.

### TensorRT-LLM

Use `residency: unit` with `trtllm-serve`. Current TensorRT-LLM serving exposes
OpenAI-style chat/completions and, for supported encoder models, embeddings, along with
`/health` and `/metrics`.

## vLLM without process restart

vLLM sleep mode can release GPU memory while leaving the HTTP server alive:

```yaml
models:
  qwen-vllm:
    kind: llm
    runner: llm_unit
    residency: vllm_sleep
    unit: vllm-qwen
    endpoint: http://127.0.0.1:8000
    served_name: Qwen/Qwen3-8B
    health_path: /health
    metrics_path: /metrics
    api_paths:
      - /v1/chat/completions
      - /v1/completions
      - /v1/responses
      - /v1/embeddings
    vram_mib: 11000
    caps: [chat, reasoning]
    quality: 85
    status: ready
```

The server must be launched with vLLM sleep mode enabled. gpu-broker uses only these
fixed internal lifecycle calls:

- `GET /is_sleeping`
- `POST /sleep?level=1`
- `POST /wake_up`

Level 1 offloads weights to CPU and discards KV cache. The server remains alive, so waking
the same model does not require a process restart.

**Security:** vLLM exposes online sleep controls only when its development server controls
are enabled. vLLM explicitly warns that those endpoints should not be exposed to users.
Bind the vLLM server to a trusted/internal interface and expose gpu-broker, not the vLLM
development endpoints.

## Ollama shared-daemon residency

Ollama is different from a dedicated server per model: one daemon can own many model
definitions and dynamically load them. `residency: ollama` lets gpu-broker control the
model without restarting that daemon.

```yaml
models:
  qwen-ollama:
    kind: llm
    runner: llm_unit
    residency: ollama
    unit: ollama
    endpoint: http://127.0.0.1:11434
    served_name: qwen3:8b
    api_paths:
      - /v1/chat/completions
      - /v1/completions
      - /v1/responses
      - /v1/embeddings
    vram_mib: 7000
    slots: 2
    reserved_interactive: 1
    caps: [chat, reasoning, embeddings]
    quality: 75
    status: ready
```

For inference, gpu-broker uses Ollama's OpenAI-compatible routes. For residency only, it
uses fixed native operations:

- `GET /api/version` — default health check
- `GET /api/ps` — confirm that the catalog's exact `served_name` is loaded
- `POST /api/chat` with empty messages and `keep_alive: -1` — load/retain that model
- `POST /api/chat` with empty messages and `keep_alive: 0` — unload that model

The model name always comes from trusted catalog configuration, never directly from an HTTP
request. Switching between two `ollama` catalog entries leaves the shared daemon running.

## Compatibility surface

The broker's fixed JSON compatibility allowlist is:

- `/v1/chat/completions` (streaming and non-streaming)
- `/v1/completions`
- `/v1/responses`
- `/v1/embeddings`
- `/v1/rerank`
- `/v1/score`

Non-chat compatibility routes are JSON/non-streaming in the current release. A model's
`api_paths` can explicitly narrow that list.

Use `POST /v1/resolve` to test a routing decision without loading a model or consuming GPU
time.
