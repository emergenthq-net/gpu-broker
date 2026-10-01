# Security

## Reporting

Please report vulnerabilities privately through GitHub's "Report a vulnerability" (security
advisories) on this repository rather than in a public issue.

## Threat model

gpu-broker controls services on a GPU host, so the questions are: who can make it act, what
can they make it do, and what can a compromised broker reach.

**Assets.** The GPU and the model servers on it; the host(s) those servers run on; the API
token and model-server keys; model files on disk; the job log (prompts and outputs).

**Trusted inputs.** The config file, the catalog file, the environment, and (for the
Proxmox driver) the host script and its config. Whoever can write these controls the broker.

**Untrusted inputs.** Every HTTP request, including from token holders; anything a backend
returns (ComfyUI prompt ids, file names, LLM output); model ids and repository references
named in requests.

### What a token holder can do

Queue jobs on catalog models, claim interactive priority with `x-priority: interactive`
(the reserved slots are a courtesy between cooperating clients, not an access control),
cause model switches (a denial of service to other users of
the GPU, by design bounded by FIFO order), queue downloads of public Hugging Face or GitHub
repositories into the models root, and read job results and the event log. The token is a
single shared secret: give it only to clients you would let use the GPU freely.

### What they cannot do

- **Run commands.** Drivers execute fixed argv lists with no shell; unit names, verbs,
  repository ids, URLs, slugs, include globs and paths are checked against strict grammars
  (`drivers/validate.py`, `units.py`) before any process starts, and units must be on the
  allowlist (default: exactly the units the catalog and config name).
- **Touch files outside the roots.** Downloads land in `<models_root>/<slug>`; links are
  created only under ComfyUI's model directories; both ends are resolved (symlinks included)
  and must stay inside their root.
- **Make the broker fetch arbitrary URLs (SSRF).** The broker contacts only `comfy.url` and
  catalog `endpoint`s, http(s) only. Models registered from a request never carry an
  endpoint or unit. Values from backends are URL-escaped before reuse.
- **Read secrets.** The API token and `UPSTREAM_TOKEN_*` keys are read from the environment,
  never logged, never returned, and never placed in URLs. Only env vars with the
  `UPSTREAM_TOKEN_` prefix can be referenced by a catalog `auth_env`.

### Authentication

`Authorization: Bearer <token>` on every route except `/health` and the static dashboard
assets, compared with `hmac.compare_digest`. If `BROKER_TOKEN` is unset the server refuses
to start (`gpu-broker serve`) and every request is rejected. There are no user accounts,
roles or rate limits; put a reverse proxy in front if you need them.

### Network exposure

The API listens on `127.0.0.1:8095` by default. Binding elsewhere (`server.host`) is a
deliberate choice; then terminate TLS in front of it, since the bearer token travels in a
header. The OpenAPI docs are disabled. Responses carry `X-Content-Type-Options`,
`X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` and a Content-Security-Policy.

### The dashboard

A static page with no embedded data. It asks for the token once and keeps it only in that
browser's `localStorage` (under `gpu-broker-token`), sending it as a header on same-origin
requests. Scripts are separate files served from an allowlist; there is no inline script
and no inline event handler, so `script-src 'self'` holds. All dynamic text is HTML-escaped,
and only http(s) URLs are ever opened. Anyone with access to that browser profile has the
token: clear site data on shared machines.

### Proxmox driver

The broker's SSH key is pinned on the host by `authorized_keys` `command=` to
`host/gpu-broker-ctl`, which re-validates every argument, allows only listed
`<container>:<unit>` pairs, logs every call, and with no config allows no unit. A compromised
broker can therefore start/stop the listed units, read GPU figures, download public models
into the models root and link them into ComfyUI — nothing else on the hypervisor.

### API-managed residency

Some catalog models can keep their server process running while gpu-broker releases or
restores that model's GPU memory. These controls are still bounded by trusted catalog
configuration:

- `vllm_sleep` can call only vLLM's fixed `/is_sleeping`, level-1 `/sleep` and
  `/wake_up` paths on the model's configured `endpoint`.
- `ollama` can call only `/api/version`, `/api/ps` and `/api/chat`; load/unload
  bodies use the catalog's exact `served_name`, never a model name supplied directly by
  the HTTP requester.
- The general compatibility proxy remains a fixed allowlist. `health_path`,
  `metrics_path` and `api_paths` may narrow or describe the trusted runtime surface but
  cannot introduce an arbitrary request-controlled URL.

vLLM online sleep controls require its server development mode. vLLM documents those
endpoints as not suitable for exposure to users. Run that endpoint on a trusted/internal
interface and put gpu-broker or another access-controlled proxy in front.

### Docker driver

Access to the Docker socket is root-equivalent on the host. The broker only issues
`start`, `stop` and `inspect` for allowlisted container names, but anyone who compromises
the broker process inherits the socket. Prefer the systemd driver with a narrow sudoers rule
when that matters.

### Data at rest

The SQLite database and JSONL log hold full requests and results (prompts, chat messages,
output paths). Protect `/var/lib/gpu-broker` and `/var/log/gpu-broker` accordingly and rotate
the JSONL log.

### Out of scope

Denial of service by a token holder (queue flooding, forcing switches); the security of
the model servers and ComfyUI themselves; content generated by the models.
