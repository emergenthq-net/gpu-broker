# Security

## Reporting

Please report vulnerabilities privately through GitHub's "Report a vulnerability" (security
advisories) on this repository rather than in a public issue.

## Threat model

The full threat model, with each threat tied to the code and tests that mitigate it, is in
[docs/threat-model.md](docs/threat-model.md). This section is the user-facing summary.

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
- **Make the broker fetch arbitrary URLs (SSRF)** — unless the operator enables
  `inputs.allow_urls`. The broker contacts only `comfy.url` and catalog `endpoint`s, http(s)
  only. Models registered from a request never carry an endpoint or unit. Values from
  backends are URL-escaped before reuse. With `inputs.allow_urls` on, a token holder can make
  the broker GET `image_url`, `end_image_url` and `video_url`, but only at **public**
  addresses: every connection (the first request and each redirect hop) resolves the host,
  refuses unless every address is globally routable — loopback, private, link-local (cloud
  metadata), shared and unspecified ranges are refused, and an IPv6 address that carries an
  IPv4 one (IPv4-mapped, 6to4 `2002::/16`, Teredo (its client address), NAT64 `64:ff9b::/96` and `64:ff9b:1::/48`, IPv4-compatible
  `::a.b.c.d`) must pass for the embedded IPv4 address too — and then connects to that vetted
  address, so DNS rebinding between check and connect does not help. Environment proxies are
  ignored. Operators can allow specific networks with `inputs.url_allow_networks` (CIDRs);
  anything listed there becomes reachable by every token holder. One exception lets a
  job's output URL feed the next job: a `GET /view?type=output` (one `type`, no `..` in
  `filename` or `subfolder`) may reach the broker's own ComfyUI — the addresses the hosts of
  `comfy.url` and `comfy.public_url` resolve to, on their ports. Other paths, other jobs'
  inputs (`type=input`), and other ports on that host stay unreachable. `inputs.fetch_s`
  bounds the whole fetch — name resolution, each connection attempt, TLS, headers and body,
  across every redirect hop: each step gets only the time left, and when it runs out a timer
  shuts the sockets down, which ends a read however slowly the server drips bytes. Name
  lookups run on a fixed set of four daemon threads, so hanging resolvers cannot multiply
  threads; IP literals are not looked up. A redirect's body is never read. The body is read
  up to the slot's size cap, and one shorter than its Content-Length is refused. Every
  failure — network, refusal, timeout, size, format — gives the same `could not be fetched`
  error, so callers cannot map what the broker reaches.
  (`frames` are inline only.)
- **Upload arbitrary files.** Input images must be PNG, JPEG or WebP and videos MP4, MOV or
  WebM by their magic bytes (`inputs.types`, `inputs.video_types`), agree with any declared
  type, and fit the size caps. They are stored under names the broker derives from the job id
  and slot (`broker-<job id>-<slot>.<ext>` in ComfyUI, `<slot>[-NN].<ext>` for exec recipes),
  never a caller-supplied name.
- **Choose what an exec job runs.** A catalog `exec` entry names a recipe; the recipe file,
  written by the host's administrator, holds the whole command, its paths, output glob and
  timeout. A job contributes only the recipe name (fixed by the catalog), its job id and its
  input files. On Proxmox those three cross SSH as `exec-put <recipe> <job id> <file name>`
  (the file on stdin, capped by `MAX_PUT_BYTES`), `exec-run <recipe> <job id>`, and, when a
  run's outcome is unknown, `exec-clean <recipe> <job id>` (kill every process of the job,
  remove its inputs, and cancel an exec-run of it still in flight; `exec-info <recipe>` only
  reports timings); the host script re-checks each against the same grammars, reads the
  recipe without sourcing it, splits `argv` on spaces with globbing off, and runs it with `pct
  exec` and no shell as `env -- GPU_BROKER_JOB=<job id> argv...` (so argv[0] may not contain
  `=` or start with `-`). The job's processes are found by that tag in /proc/*/environ inside
  the container, which a worker keeps when it leaves the process group, and killed by pid and
  group (never group 1). The job's output folder is
  created inside an existing parent and given the parent's owner, never a new root-owned
  tree. The broker never hands the GPU on while a recipe may still run: if no scan confirms
  the job gone (a scan that cannot read a process of the job's user, or a /proc that hides
  a pid known to be alive, counts as "not confirmed"), it sets a persisted GPU hold that
  resume and restarts keep, and a broker restarted while an exec job ran sets it before
  taking any job; only a later successful clean or `POST /v1/admin/gpu-held/clear` lifts it.
  Request
  values in `exec.params` reach the program only as data in `params.json` (numbers, booleans
  and short strings). The program itself runs with the container's (or, on the systemd
  driver, the broker's) privileges, so only install recipes for programs you trust with
  attacker-chosen input files.
- **Read secrets.** The API token and `UPSTREAM_TOKEN_*` keys are read from the environment,
  never logged, never returned, and never placed in URLs. Only env vars with the
  `UPSTREAM_TOKEN_` prefix can be referenced by a catalog `auth_env`.

### Authentication

`Authorization: Bearer <token>` on every route except `/health` and the static dashboard
assets, compared with `hmac.compare_digest`. If `BROKER_TOKEN` is unset the server refuses
to start (`gpu-broker serve`) and every request is rejected. There are no user accounts,
roles or rate limits; put a reverse proxy in front if you need them.

### The MCP server

`/mcp` (with the `mcp` extra) takes the same credentials as the chat routes, checked before
the MCP SDK sees the request; with none it answers 401. The guard then names the caller in an
internal header and drops any copy the client sent, so a client key cannot pose as the main
token. The key lookup runs off the event loop, and a refusal is a JSON-RPC error naming the
status. Its tools submit jobs through the same path as `POST /v1/jobs` (input checks, limits,
`inputs.allow_urls`), name only models that can run now (never a repository to download, by
generation or chat), and show a client key only the jobs it owns, recorded by key id (names
are not unique): another caller's job reads as missing, and `gpu_status` gives only a count of
other callers' jobs. A client key cannot send `<slot>_url` inputs, which would have the broker
fetch a caller-chosen address (inside `url_allow_networks` too), unless the operator sets
`mcp.client_url_inputs`; the main token can, under `inputs.allow_urls`. A request body is capped at
`mcp.max_body_bytes`. It is stateless (POST only), so nothing persists between calls.
DNS-rebinding protection is off, since a browser page cannot add the bearer token. `gpu-broker
mcp` reads its credential from the environment or connect's manifest, never a command line,
and writes only MCP messages to stdout. It sends `$BROKER_TOKEN` only to the broker on this
machine its config names (loopback, `server.port`), and no credential over plain http to a host
that is not loopback or private; over plain http it resolves the name once and connects only to
those checked addresses for the life of the relay, so a name that rebinds to a public address
later never receives the key. Inline images are read through the broker's own ComfyUI client
(`comfy.url`, `comfy.auth_env`), never a URL from a job record. `connect` writes the client key into the MCP entries
it registers (`~/.claude.json`, `claude_desktop_config.json`, Codex's `config.toml`) and sets
those files to mode 600; it never replaces a `gpu-broker` server the user defined, and lists
every file it changed.

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

### Docker driver

Access to the Docker socket is root-equivalent on the host. The broker only issues
`start`, `stop` and `inspect` for allowlisted container names, but anyone who compromises
the broker process inherits the socket. Prefer the systemd driver with a narrow sudoers rule
when that matters.

### Data at rest

The SQLite database and JSONL log hold full requests and results (prompts, chat messages,
output paths), but not input files: those wait in `inputs.staging_dir` (mode 0600) until the
job runs, are then copied into ComfyUI's input folder (or an exec recipe's `in_dir`), and are
deleted from the staging directory when the job ends. Exec recipes delete their `in_dir`
after each run. ComfyUI has no API to delete its copy and the broker has no access to
ComfyUI's filesystem; install `examples/systemd/comfyui-input-prune.{path,service}` beside
ComfyUI: on each upload it deletes `broker-*` inputs older than the longest job. Protect `/var/lib/gpu-broker` and `/var/log/gpu-broker` accordingly and rotate
the JSONL log.

### Out of scope

Denial of service by a token holder (queue flooding, forcing switches); the security of
the model servers and ComfyUI themselves; content generated by the models.
