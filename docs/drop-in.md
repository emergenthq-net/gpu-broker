# Use gpu-broker as a drop-in for ChatGPT or Claude

An app, script or tool that already talks to the OpenAI or Anthropic API can use your local
models instead. Change two settings, the **base URL** and the **API key**, and nothing else.

- Base URL: your broker, for example `http://gpu-host:8095`. OpenAI clients add `/v1` to it;
  Anthropic clients do not.
- API key: the broker's token (`$BROKER_TOKEN`). It is accepted in either SDK's header,
  `Authorization: Bearer <token>` or `x-api-key: <token>`. (The broker's admin routes,
  `/v1/admin/*`, take only `Authorization: Bearer`.)

## The two-line change

### OpenAI SDK (Python)

```python
from openai import OpenAI
client = OpenAI(base_url="http://gpu-host:8095/v1", api_key="<broker token>")
client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
```

Or leave the code alone and set the environment:

```sh
export OPENAI_BASE_URL=http://gpu-host:8095/v1
export OPENAI_API_KEY=<broker token>
```

### OpenAI SDK (JavaScript)

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://gpu-host:8095/v1", apiKey: "<broker token>" });
```

The same `OPENAI_BASE_URL` / `OPENAI_API_KEY` variables work.

### Anthropic SDK (Python)

```python
import anthropic
client = anthropic.Anthropic(base_url="http://gpu-host:8095", api_key="<broker token>")
client.messages.create(model="claude-sonnet-4-5", max_tokens=1024, messages=[{"role": "user", "content": "hi"}])
```

Or set `ANTHROPIC_BASE_URL=http://gpu-host:8095` and `ANTHROPIC_API_KEY=<broker token>`.

### Anthropic SDK (JavaScript)

```js
import Anthropic from "@anthropic-ai/sdk";
const client = new Anthropic({ baseURL: "http://gpu-host:8095", apiKey: "<broker token>" });
```

### curl

```sh
# OpenAI shape
curl http://gpu-host:8095/v1/chat/completions -H "Authorization: Bearer $BROKER_TOKEN" \
  -H "Content-Type: application/json" -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}'

# Anthropic shape
curl http://gpu-host:8095/v1/messages -H "x-api-key: $BROKER_TOKEN" -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet-4-5", "max_tokens": 256, "messages": [{"role": "user", "content": "hi"}]}'
```

### Open WebUI

Settings → Connections → OpenAI API: URL `http://gpu-host:8095/v1`, key = the broker token.
The model picker lists the broker's ready models (`GET /v1/models`).

### Editor tools (Continue, Cline, Aider, ...)

Anything that accepts an "OpenAI-compatible" provider takes the same two values: base URL
`http://gpu-host:8095/v1` and the token as the API key. Aider reads
`OPENAI_API_BASE=http://gpu-host:8095/v1` and `OPENAI_API_KEY`; pass the model as
`--model openai/<name>`.

## Which model answers: name mapping

Apps ask for hosted models by name (`gpt-4o`, `claude-sonnet-4-5`). `model_map` in the
config file says which of your catalog models answers instead:

```yaml
model_map:
  "gpt-4o-mini": llama-3.1-8b-precise   # specific names first: first match wins
  "gpt-*": "@default"                   # "@default" = the catalog's resident LLM
  "chatgpt-*": "@default"
  "o1*": "@default"
  "o3*": "@default"
  "o4*": "@default"
  "claude-*": "@default"
  "text-embedding-*": my-embedder       # a catalog model with caps: [embed]
```

- Patterns are globs, matched case-insensitively, in order. A target may be a catalog key,
  alias or variant.
- A name your catalog already knows (key, alias, served name, Hugging Face id, variant) is
  never mapped.
- A name no pattern matches keeps the normal behaviour: the best installed model of the
  same kind stands in, and the response says so.
- Override from the environment with JSON: `BROKER_MODEL_MAP='{"gpt-*": "@default"}'`.

Every response tells you what really ran. For a name that `model_map` mapped, the `model`
field echoes the name you asked for, so clients that check it keep working; a catalog name
keeps the model server's own `model`, as it always has. `x_broker` holds the facts:

```json
"model": "gpt-4o",
"x_broker": {"used": "llama-3.1-8b", "substitution": "'gpt-4o' mapped to 'llama-3.1-8b' by model_map pattern 'gpt-*'"}
```

The mapping is also recorded on the job, so it shows in the dashboard and `GET /v1/jobs/{id}`.

## What works

| | OpenAI (`/v1/chat/completions`) | Anthropic (`/v1/messages`) |
|---|---|---|
| Plain and streamed replies | yes | yes, with Anthropic's event sequence |
| Tool / function calling | `tools`, `tool_choice`, `tool_calls` | `tools`, `tool_choice`, `tool_use`, `tool_result` |
| Images in the prompt | `image_url` parts | `image` blocks (base64 or URL) |
| JSON output | `response_format` (json_object, json_schema) | n/a |
| Reasoning | `reasoning_content`, if the server returns it | a `thinking` block when `thinking` is enabled |
| Sampling | `temperature`, `top_p`, `stop`, `seed`, `max_tokens` / `max_completion_tokens` | `temperature`, `top_p`, `top_k`, `stop_sequences`, `max_tokens` |
| Model list | `GET /v1/models` | the same, in Anthropic's shape when `anthropic-version` is sent |
| Embeddings | `POST /v1/embeddings` (needs a model with `caps: [embed]`) | n/a |

Errors come back in each API's own shape, so SDK exceptions (`AuthenticationError`,
`BadRequestError`, ...) work as usual.

## Honest limits

- **Local models are less capable than hosted frontier models.** An app tuned on GPT or
  Claude may need simpler prompts, and long agent loops may go wrong sooner.
- **Tool calling needs a model and a server that support it.** For llama.cpp run
  `llama-server --jinja` with a model whose chat template handles tools. Without that the
  model answers in plain text instead of calling the tool.
- **Images need a vision model** (llama.cpp: started with its `--mmproj` projector).
- **Embeddings need an embedding model** in the catalog (`caps: [embed]`, served by
  `llama-server --embeddings`). With none configured, `/v1/embeddings` answers 404
  `model_not_found`. An embedding model is never used as a stand-in for chat and is not
  listed by `/v1/models`.
- **Hosted-only features are not emulated:** prompt caching, the Batches API, the Files
  API, web search, code execution, computer use, citations, PDF/document blocks. A request
  that uses them gets a clear 400, not a silent fallback.
- `stop_sequence` is reported only when the server says which stop string matched;
  otherwise a stop string ends the reply with `end_turn`.
- Reasoning sent back to Anthropic clients carries an empty `signature` (there is nothing
  to sign locally). Images inside a `tool_result` reach the model in a user message right
  after the tool result (an OpenAI tool message holds text only).
- `n` greater than 1 depends on the model server; llama.cpp returns one choice.
