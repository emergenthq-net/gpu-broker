# Use gpu-broker as a drop-in for ChatGPT or Claude

Apps, scripts and tools written for the OpenAI or Anthropic API can use your local models.
You don't edit their settings: gpu-broker does it.

## One command

On the machine whose apps should use the broker:

```sh
gpu-broker connect --url http://gpu-host:8095
```

It finds the AI apps installed there and points each one at the broker:

| App | What connect changes |
|---|---|
| Shell (zsh, bash, fish) | a marked block in `~/.zshrc`, `~/.bashrc`, `config.fish` setting `GPU_BROKER_URL`, `GPU_BROKER_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_API_BASE` (Aider). If `OPENAI_API_KEY` already holds another key, the OpenAI variables are left alone. `ANTHROPIC_*` is set only with `--claude-code` |
| Continue | adds `~/.continue/models/gpu-broker.yaml` (your `config.yaml` is not touched) |
| Cline (4.x) | adds an OpenAI-compatible provider to `~/.cline/data/settings/providers.json` (checked whole and kept at mode 600, as Cline requires) |
| Roo Code | a profile file Roo imports at start, named in VS Code's `settings.json` |
| Open WebUI | adds the broker as a connection through its admin API (`--openwebui-url`, `--openwebui-token`); disconnect removes just that connection, keeping any added since |
| Claude Code | `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` (and `ENABLE_TOOL_SEARCH=true`, so MCP tool search keeps working) in `~/.claude/settings.json`; **only with `--claude-code`**, because it switches your main assistant. With [cloud failover](failover.md) on, Claude Code keeps its own key and the broker key goes in `ANTHROPIC_CUSTOM_HEADERS` instead |
| Codex CLI | adds a `gpu-broker` profile, `~/.codex/gpu-broker.config.toml` (`$CODEX_HOME` moves it; mode 600), with a provider using `wire_api = "responses"`; your `config.toml` is not touched, so run `codex --profile gpu-broker` (Codex 0.134 or later). With cloud failover on and `OPENAI_API_KEY` set, the profile sends that key and the broker key in `http_headers` |

- **Nothing is lost.** Every file is backed up (timestamped, in `~/.gpu-broker/connect/backups/`)
  before its first change, and listed in a manifest as it is changed. `gpu-broker disconnect`
  puts every file back exactly as it was. If you edited a file since (even between two
  connects), it removes only the broker's entries and keeps your edits. Anything it cannot
  undo stays listed, so running disconnect again retries it.
- **See first:** `gpu-broker connect --dry-run` prints the plan and changes nothing.
  `gpu-broker clients` lists what was found and what is connected.
- **Pick apps:** `--only shell,continue`. Running connect again changes nothing new.
- **Its own key.** Connect asks the broker for a key named after the machine (it needs the
  main token once, from `$BROKER_TOKEN` or `--token`), so the main token is never copied into
  config files. A client key reaches only the model routes (chat, messages, embeddings, the
  model list, the Responses API, whether failover passes its own key through) and its calls are recorded under its own name; jobs, status, events and stats
  stay main-token only. Keys are listed, and revocable, on the dashboard; `gpu-broker
  disconnect --revoke` also revokes it (it needs the main token, and with `--only` it refuses
  while other apps still use the key).
- **Only safe values are written.** The broker URL, key and model may contain letters, digits
  and `. : / @ + - _ [ ]`; anything else is refused before any file is touched, and shell
  values are single-quoted.
- Apps unknown to connect, or in a form it will not edit safely (VS Code settings with
  comments, say), are skipped with the reason. They can still be set up by hand (below).

### From the dashboard

The **Connect apps** card lists the apps on the broker's own machine with a Connect or
Disconnect button each. For another machine, **Connect another machine** gives a one-line
installer; run it there within 15 minutes (it works once). The key is made when the
installer is fetched, so an unused link leaves none behind. The installer checks the
downloaded program against a SHA-256 it carries, passes the key on standard input (never on a
command line), and warns if the broker is reached over plain `http` on a public address:

```sh
curl -fsSL 'http://gpu-host:8095/connect.sh?invite=...' | sh
```

It needs only `python3` (3.9+) on that machine. Undo it there with
`python3 ~/.gpu-broker/connect/gpu-broker-connect.pyz disconnect`.

## When the GPU is busy: hosted fallback (optional)

Off by default. With `fallback.enabled: true` (or `BROKER_FALLBACK=1`) and your own provider
key in `UPSTREAM_OPENAI_API_KEY` / `UPSTREAM_ANTHROPIC_API_KEY`, a request for a hosted model
name that the local side cannot serve goes to the real provider:

- the GPU is busy with another model (a video job, say) and the call would have to wait for a switch;
- the local call failed;
- the request uses a feature only the hosted API has (an Anthropic server tool, a document block).

The key is read from the environment only and never logged or returned. Every response says
which side answered: the header `x-broker-served-by: local` or `hosted`, and on JSON bodies
`x_broker.served_by`. A catalog model name is never sent upstream.

## When the cloud is down: local failover (optional)

The other direction: send `claude-…` and `gpt-…` to the real provider first, and answer locally
when it fails (no internet, an outage, quota used up), going back to the cloud when it
recovers. The client's own provider key is passed through. See [failover.md](failover.md).

## Which model answers: name mapping

Apps ask for hosted models by name (`gpt-4o`, `claude-sonnet-4-5`). Out of the box, with no
configuration, `gpt-*`, `chatgpt-*`, `o1`/`o3`/`o4`... and `claude-*` are answered by the
resident model (`gpu-broker check` prints the mapping in force). To change it, set
`model_map` in the config file:

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
- `model_map: {}` turns mapping off.

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

The Responses API (`/v1/responses`) covers the same ground: `instructions`; `input` as a
string or as items (messages with text and `input_image` parts, `function_call`,
`function_call_output`, `custom_tool_call`, `custom_tool_call_output`); function and custom
(freeform) `tools`, `tool_choice` (including `allowed_tools`), `parallel_tool_calls`;
`max_output_tokens`, `temperature`, `top_p`; `text.format` (JSON object or JSON schema);
the streamed event sequence (`response.created`, `output_item.added`, `output_text.delta`,
`function_call_arguments.delta`, `reasoning_summary_text.delta`, `response.completed`, and
`response.incomplete` / `response.failed`); and `previous_response_id`.

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
  that uses them gets a clear 400, or goes to the provider if the hosted fallback is on.
- `stop_sequence` is reported only when the server says which stop string matched;
  otherwise a stop string ends the reply with `end_turn`.
- Reasoning sent back to Anthropic clients carries an empty `signature` (there is nothing
  to sign locally). Images inside a `tool_result` reach the model in a user message right
  after the tool result (an OpenAI tool message holds text only).
- `n` greater than 1 depends on the model server; llama.cpp returns one choice.
- **Responses API:** hosted tools (`web_search`, `file_search`, `code_interpreter`,
  `image_generation`, `computer_use`, `mcp`) and `namespace` tool groups cannot run locally.
  Codex sends some of them on every request, so they are left out rather than refused, and
  listed in `x_broker.dropped_tools`. A custom (freeform) tool reaches the model as a function
  with one string argument, `input`, and its calls come back as `custom_tool_call` items.
  `local_shell` and unknown tool types are refused with a 400.
- `instructions` and every developer or system message, including those of a stored
  conversation, reach the model as one system message at the start: many chat templates
  accept nothing else.
- `previous_response_id` works for an hour after the response, until the broker restarts,
  only for the identity that created it: the same client key, or the main token with the same
  `x-requester`. The last 256 responses are kept in memory, within
  `limits.response_store_entry_bytes` each (a larger conversation is not kept) and
  `limits.response_store_bytes` in all (the oldest go first); `store: false` keeps nothing.
  A request body over `limits.responses_body_bytes` (32 MiB) is refused with a 413.
  There is no `GET /v1/responses/{id}`, background mode, prompt templates or input files.
  The model's reasoning comes back as a `reasoning` item's summary text, and reasoning items
  sent back in `input` are dropped (there is no encrypted reasoning to resume).

## Manual setup

If you would rather not run `gpu-broker connect`, every client takes the same two values: the
broker's URL and a key (a client key from the dashboard, or the main token).

- Base URL: your broker, for example `http://gpu-host:8095`. OpenAI clients add `/v1` to it;
  Anthropic clients do not.
- Key: accepted in either SDK's header, `Authorization: Bearer <key>` or `x-api-key: <key>`.
  The broker's admin routes (`/v1/admin/*`, keys, connect) take only the main token, and
  only as `Authorization: Bearer`.

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

### Codex CLI and the Responses API

Codex speaks only the OpenAI Responses API, which the broker serves at `/v1/responses`.
`gpu-broker connect --only codex` sets it up; by hand, write the profile file
`~/.codex/gpu-broker.config.toml` (Codex 0.134 and later read a profile from its own file,
not from `[profiles.*]` in `config.toml`):

```toml
model = "gpt-5-codex"            # mapped like any hosted name (see below)
model_provider = "gpu-broker"

[model_providers.gpu-broker]
name = "gpu-broker"
base_url = "http://gpu-host:8095/v1"
env_key = "GPU_BROKER_API_KEY"   # or experimental_bearer_token = "<a client key>"
wire_api = "responses"
```

and run `codex --profile gpu-broker`. A stored response (`previous_response_id`) belongs to
the key that created it; Codex itself sends `store: false` and the whole conversation each
turn.

The Responses SDK (`client.responses.create(...)`) works the same way, with the base URL and
key changed.
