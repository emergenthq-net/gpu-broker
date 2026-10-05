# MCP server

With the `mcp` extra (`pip install 'gpu-broker[mcp]'`) the broker is also an MCP server, so an
assistant can use your GPU as a tool: "make an image of...", "animate this photo", "turn these
views into a 3D scene", "ask the local model".

## Tools

| tool | |
|---|---|
| `list_models` | What can run now, with kind (llm, image, video, 3d), caps and the inputs each takes. |
| `generate_image`, `generate_video` | From a prompt. |
| `edit_image`, `image_to_video` | From an image (base64, or an http(s) URL; see below) and a prompt; `image_to_video` takes an optional last frame. Video length is `num_frames`. |
| `make_3d` | A Gaussian splat from one image, several views, or a video. |
| `job_status`, `job_result` | A job's state; its output URLs, and small images inline. |
| `chat_local` | Ask the local LLM. |
| `gpu_status` | What is loaded, running and queued, and VRAM. |

- A generate tool waits up to `mcp.wait_s` (45 s). A job still running then comes back with its
  `job_id` and a hint to call `job_status`, so a long video never holds the call open.
- Finished images up to `mcp.inline_max_bytes` come back inline as well as by URL.
- Inputs go through the same checks and limits as `POST /v1/jobs`.
- A tool names only a model that can run now (never one that would be downloaded).

## Transports

- **Streamable HTTP** at `/mcp` on the broker, with the same credentials as the chat routes
  (the main token or a client key, as a Bearer header), for remote clients and ChatGPT
  connectors.
  - It is stateless: no `mcp-session-id` is issued, and none is needed. Each POST stands alone
    (send the Bearer header on every one), answers come on that POST's response, and `GET`
    (server-initiated stream) and `DELETE` (end a session) get 405. There are no server-to-client
    requests or notifications; poll `job_status` for a long job.
  - A client key sees only the jobs it submitted (by key, not by name: keys may share a name),
    and in `gpu_status` only a count of everyone else's.
  - A client key sends files as base64: a URL input would have the broker fetch an address the
    caller chose, so it is for the main token unless `mcp.client_url_inputs: true`.
- **stdio**: `gpu-broker mcp` relays to a broker's `/mcp`, for apps that launch a command.
  - The broker is `--url`, `$GPU_BROKER_URL` or the one `connect` used.
  - The credential is `$GPU_BROKER_API_KEY`, connect's key for that broker, or `$BROKER_TOKEN`
    (the main token goes only to the broker on this machine).
  - No credential goes over plain http to a host that is not loopback or private, and over
    plain http the relay connects only to the addresses it checked.
  - When the broker is down or refuses, the app gets an error naming the HTTP status; a revoked
    key ends the relay (exit 2).

## Registering it

`gpu-broker connect` registers the tools when the broker serves `/mcp`:

- Claude Code: user-scope `mcpServers` in `~/.claude.json`, over HTTP.
- Claude Desktop: `claude_desktop_config.json`, running `gpu-broker mcp`; offered where this
  machine has the `mcp` extra.
- Codex: `[mcp_servers.gpu-broker]` in `config.toml`, over HTTP.

These change no model: the apps keep their own and gain the tools. A `gpu-broker` server you
defined yourself is left alone. A symlinked config is written through (the link stays). Claude
Code rewrites `~/.claude.json` while it runs, so connect reads it back and warns if the entry
was dropped (quit Claude Code and connect again). connect ends with the files it changed;
`gpu-broker disconnect` removes the entries.
