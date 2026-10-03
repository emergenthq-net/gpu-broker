# Threat model

gpu-broker decides which model server holds one GPU, and starts and stops those servers on a
host. This file is the reference for security review.

- **The questions:** who can make the broker act, what can they make it do, and what can a
  compromised broker reach.
- **For each threat:** the code that mitigates it, and the tests that hold it in place.
- **Rule:** a change that weakens a mitigation needs a matching change here, and a test that
  fails without it.
- **Reporting a vulnerability:** see [SECURITY.md](../SECURITY.md).

## Deployment assumed

- One broker per GPU host.
- Used by one person, or a small group of cooperating clients, on loopback or a LAN.
  - Clients: chat apps, agents, scripts, a dashboard user.
  - Every client holds the same token.
- Binds `127.0.0.1:8095` by default (`settings.Server`).
  - Binding elsewhere is a deliberate operator choice, and then belongs behind TLS.

## Assets

| Asset | Where it lives | Why it matters |
|---|---|---|
| `BROKER_TOKEN` | the broker's environment (`cli.py`, `web/app.py`) | the only credential: whoever holds it can use the GPU, read every job and quiesce the broker |
| Upstream model-server keys | env vars `UPSTREAM_TOKEN_*`, named by a catalog entry's `auth_env` (`broker.py`, `backends.py`) | sent to model servers; must never leave in a response, log or URL |
| Host control | the driver: the Proxmox SSH key (`drivers/proxmox.py`, default `/etc/gpu-broker/id_ed25519`), the systemd sudo rule, or the Docker socket (`drivers/local.py`) | the path from the broker to processes on the host |
| Model files | `<models_root>/<slug>` and links under ComfyUI's model folders (`drivers/validate.py`, `downloads.py`) | disk space, and what the servers load |
| Inputs and outputs | `inputs.staging_dir` (mode 0600, `staging.py`), ComfyUI's input and output folders, an exec recipe's `in_dir`/`out_dir` | callers' images and video, and generated results |
| Job log | SQLite `db` and `events_jsonl` (`store.py`) | full prompts, chat messages and results |
| The GPU itself | the scheduler's residency (`scheduler.py`, `residency.py`, `holds.py`) | a recipe still running after the broker moved on would corrupt the next job |

## Trust boundaries

| # | Boundary | Trust |
|---|---|---|
| 1 | HTTP clients to the API (`web/`) | Untrusted input, token holders included: bodies, headers, model names, repository references, prompts, image and video data, `<slot>_url`. |
| 2 | Broker to host control (`drivers/`) | Arguments cross as argv, never through a shell the broker controls. Proxmox: an SSH key the host pins with `command=` to `host/gpu-broker-ctl`. Locally: `systemctl` (optionally a narrow sudo rule) or `docker`. |
| 3 | Host administrator to the broker | Trusted: exec recipes, the config file, the catalog, `/etc/gpu-broker-ctl.conf`. Whoever can write these controls the broker. |
| 4 | Broker to backends (`backends.py`) | ComfyUI and LLM servers at URLs from config and catalog only. What they return (prompt ids, file names, LLM output) is untrusted data. |
| 5 | Broker to caller-chosen URLs (`media.py`, `netguard.py`, `deadline.py`) | Only when the operator sets `inputs.allow_urls`. |
| 6 | Browser to the dashboard (`web/dash.py`, `web/static/`) | The page keeps the token in that browser's `localStorage`. |

## Threats and mitigations

### T1. Calling the API without the token

- **Mitigations**
  - `web/app.py` `make_auth`: `Authorization: Bearer <token>` or `x-api-key: <token>`, each
    compared with `hmac.compare_digest`.
  - Every route needs it except `/health`, `/dash` and the allowlisted `/dash/<name>.js`
    (static, no data).
  - `/v1/admin/*` accepts the Bearer form only.
  - An empty token rejects every call; `gpu-broker serve` refuses to start without one (`cli.py`).
- **Tests:** `tests/test_api.py`, `tests/test_dropin_openai.py`, `tests/test_dropin_anthropic.py`, `tests/test_cli.py`.

### T2. A token holder runs commands or touches files on the host

- **Mitigations**
  - No shell anywhere: drivers run fixed argv lists (`drivers/__init__.py` `run`, `run_input`, `run_group`).
  - Every value reaching argv or the filesystem is checked against a strict grammar first:
    - unit names (`units.py`);
    - repository ids, slugs, include globs, link paths, recipe calls (`drivers/validate.py`);
    - `validate.inside` keeps paths under their root after resolving symlinks.
  - Units must be on the allowlist (`drivers.check`).
    - Default: exactly the catalog's units plus `comfy.unit`.
    - Models registered from a request never carry a unit or endpoint (`resolve.py`).
  - `host/gpu-broker-ctl` re-validates every argument (`safe`, per-verb regexes).
    - Verbs: `unit`, `gpu`, `gpustream`, `download`, `comfy-link`, `exec-put`, `exec-info`, `exec-run`, `exec-clean`.
    - Only `<container>:<unit>` pairs in `ALLOW_UNITS`.
    - Logs each call. With no config it allows no unit and knows no recipe.
- **Tests:** `tests/test_validation.py`, `tests/test_drivers.py`, `tests/test_host_ctl.py`,
  `tests/test_host_ctl_exec.py`, `tests/test_host_ctl_parity.py` (broker and host script reject the same inputs).

### T3. A token holder chooses what an exec job runs

- **Mitigations**
  - A catalog `exec` entry names a recipe. The recipe file (host configuration) holds the
    command, paths, output globs and timeout (`drivers/recipes.py`).
  - A job contributes only (`execjob.py`):
    - its job id;
    - its input files;
    - `params.json`, with only the keys the catalog lists in `exec.params`.
  - Both recipe parsers (`drivers/recipes.py`, `host/gpu-broker-ctl` `load_recipe`):
    - read `key=value` lines without sourcing;
    - reject CR, unknown keys, and a first argv word that starts with `-` or contains `=`;
    - require `out_dir` to end in `/{jid}` inside an existing parent;
    - split argv on spaces, with no shell and globbing off.
- **Residual risk:** the program runs with the container's (or broker's) privileges on
  caller-chosen files. Installing a recipe is trusting that program.
- **Tests:** `tests/test_recipes.py`, `tests/test_execjob.py`, `tests/test_proxmox_exec.py`, `tests/test_api_exec.py`.

### T4. The GPU is handed on while an exec job may still run

- **Mitigations**
  - Job processes are tagged `GPU_BROKER_JOB=<job id>` and reaped by that tag (`drivers/reap.py`, `exec-clean`).
  - If no scan confirms the job gone, `holds.py` sets a persisted GPU hold.
    - Resume and restarts keep it.
    - Only a confirmed clean or `POST /v1/admin/gpu-held/clear` lifts it.
- **Tests:** `tests/test_holds.py`, `tests/test_reap.py`, `tests/test_host_ctl_clean.py`,
  `tests/test_scheduler_exec.py`, `tests/test_orphans.py`.

### T5. Server-side request forgery through `<slot>_url`

- **Mitigations**
  - Off unless `inputs.allow_urls`. Then http(s) only.
  - `netguard.py` resolves every connection and redirect hop itself:
    - refuses unless every address is globally routable or inside `inputs.url_allow_networks`;
    - checks the embedded address of IPv4-mapped, NAT64 and IPv4-compatible IPv6;
    - connects to the vetted address, so DNS rebinding does not help;
    - ignores environment proxies.
  - One exception: `GET /view?type=output` on the broker's own ComfyUI host and port, for chaining jobs.
  - `deadline.py` bounds the whole fetch by `inputs.fetch_s` (resolution, connect, TLS, body),
    on a fixed pool of resolver threads.
  - The body is capped at the slot's size limit.
  - Every failure returns the same error, so a caller cannot map what the broker reaches.
- **Tests:** `tests/test_netguard.py`, `tests/test_netguard_http.py`, `tests/test_deadline.py`,
  `tests/test_fetch_deadline.py`, `tests/test_media_fetch.py`.

### T6. Malicious input files

- **Mitigations**
  - `media.py` checks:
    - decoded size against `inputs.max_bytes` / `inputs.video_max_bytes`;
    - format by magic bytes against `inputs.types` / `inputs.video_types`;
    - agreement with any declared type.
  - Frame counts are bounded (`inputs.py`).
  - Stored names derive from the job id and slot, never the caller (`staging.py`).
    - `Staging.clear` deletes only names matching that pattern.
  - Exec uploads are capped on the host by `MAX_PUT_BYTES`.
- **Tests:** `tests/test_media.py`, `tests/test_inputs.py`, `tests/test_staging.py`.

### T7. Untrusted values from backends

- **Mitigations**
  - ComfyUI prompt ids and file names are URL-encoded before reuse in a request or a browser
    URL (`backends.py`, `view_url`).
  - An exec job's outputs are not what the program reports. The driver lists them itself, by
    the recipe's `outputs` globs, in the job's `out_dir` (`drivers/local.py`, `exec-run` in `host/gpu-broker-ctl`).
- **Tests:** `tests/test_backends.py`, `tests/test_api_exec.py`, `tests/test_execjob.py`.

### T8. Secrets leaking

- **Mitigations**
  - `BROKER_TOKEN` and `UPSTREAM_TOKEN_*` come from the environment only.
    - Never logged, returned or put in URLs.
  - Only `UPSTREAM_TOKEN_*` variables are loaded as upstream keys (`broker.py`).
    - A catalog `auth_env` naming anything else sends no key.
  - `gpu-broker setup` (`setup/host.py`) writes a new token to `broker.env`, mode 600. Every file
    it writes goes to a new temporary file in the same folder (created 0600, never following a
    symlink) that is renamed into place; it refuses to replace a symlink. It prints only the
    first four characters of the token.
    - `broker.env` stays in a root-owned folder, since systemd reads it as root; the catalog,
      which the service rewrites, lives in the service account's data folder instead.
    - It opens the dashboard as `/dash#token=...`: the fragment never reaches a server or a log,
      but the browser's command line holds it while it runs. On a machine other people log in
      to, close that browser, or sign in by hand (the dashboard asks once).
    - It uses sudo only with `-n`: never a password prompt, never more than the user already has.
  - Release hygiene: `scripts/leak_scan.py` scans the tree, sdist and wheel against a private
    denylist (CI job `leak-scan`).
- **Tests:** `tests/test_backends.py`, `tests/test_leak_scan.py`, `tests/test_setup.py`.

### T9. Injection into storage or the dashboard

- **Mitigations**
  - SQL uses `?` placeholders for every value (`store.py`).
    - The few f-strings interpolate only column names from fixed sets and constant lists,
      each marked `noqa: S608` with the reason.
  - Dashboard (`web/app.py`, `web/static/*.js`):
    - no inline script or event handler; scripts served from an allowlist;
    - a strict Content-Security-Policy and `X-Frame-Options: DENY`;
    - all dynamic text HTML-escaped;
    - only http(s) URLs opened.
- **Tests:** `tests/test_api.py` (headers, the script allowlist), `tests/test_dash_js.py`, `tests/test_store.py`.

### T10. A compromised broker

| driver | what it can reach |
|---|---|
| Proxmox | Start and stop the listed units, read GPU figures, download public models into the models root, link them into ComfyUI, run the installed recipes on files it supplies. Nothing else on the hypervisor. |
| systemd | What the sudo rule allows. Keep it to `systemctl start\|stop` of the listed units. `gpu-broker setup` never runs the broker as root: the service runs as the installation's owner or a dedicated `gpu-broker` account, reaches the GPU through groups (`video`, `render`), and starts and stops system units only through the rule it writes (`/etc/sudoers.d/gpu-broker`, checked by `visudo`), which names `systemctl start\|stop\|is-active -- <unit>` for the catalog's units and nothing else. Setup refuses to install a service whose code anyone but root or that account could change (`setup/service.py`). |
| Docker | The socket is root-equivalent on the host. Prefer another driver where that matters. |

## What a token holder can do, by design

- Queue jobs on any catalog model, and cause model switches.
- Claim interactive priority.
  - `x-priority: interactive` is a courtesy between cooperating clients, not access control.
- Queue downloads of any public Hugging Face or GitHub repository into the models root (`resolve.py`, `downloads.py`).
  - Downloaded files are not run until an operator adds a runner to the catalog, but they use disk.
- Read every job, result and event.
- Quiesce or resume the broker (`web/admin.py`).

There are no per-client keys, scopes, roles or rate limits. Give the token only to clients you
would let use the GPU freely.

## Out of scope

- Exposure to the internet, or to untrusted, multi-tenant users.
  - Put an authenticating, rate-limiting reverse proxy in front if you must.
- Denial of service by a token holder: queue flooding, forced switches, filling the disk with downloads.
- The security of the model servers, ComfyUI, and the programs recipes run.
- Content the models generate.
- Anyone who can write the config, catalog, recipes, host script config or the broker's
  environment: they are the operator.
