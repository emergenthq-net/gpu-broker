# AGENTS.md

Guidance for automated agents working on this repository: code review, security review and
code changes. Read [CONTRIBUTING.md](CONTRIBUTING.md) and [ARCHITECTURE.md](ARCHITECTURE.md)
for the rules behind these checks, and [docs/threat-model.md](docs/threat-model.md) for what
each security mitigation protects and which tests hold it in place.

## Checks before review

```sh
ruff check .          # includes PLR2004 (magic numbers) and bandit (S) rules
mypy                  # strict
pytest -q             # never needs a GPU, network or host: tests/conftest.py blocks them
```

## Review guidelines

Flag each of these. Cite the file and line, and say what to do instead.

### Release hygiene

- **Private infrastructure.** Hostnames, internal domains, private or LAN IP addresses,
  container or VM ids from a real site, personal home-directory paths (`/Users/<name>`,
  `/home/<name>`), real usernames or email addresses. Examples, docs and tests use neutral
  values: `127.0.0.1`, `localhost`, documentation ranges (`192.0.2.0/24`, `2001:db8::/32`),
  container id `101`, paths like `/srv/...` or `/var/lib/gpu-broker`.
- **AI or tool attribution** in code, comments, docs, commit messages or PR bodies:
  "Generated with", "Co-Authored-By" trailers naming an assistant, mentions of Claude, Codex,
  Copilot or similar as an author. Commits are authored by the human maintainer.
- **Secrets.** Tokens, API keys, private keys, passwords, or anything that looks like one,
  including in test fixtures. Tests use obviously fake values. Secrets come from the
  environment (`BROKER_TOKEN`, `UPSTREAM_TOKEN_*`), never config files, logs, responses or
  URLs.

### Code quality

- **Hard-coded values that should be constants or config.** A literal that means something
  (timeout, port, path, size, limit, event name, colour) belongs in `gpu_broker/constants.py`,
  at the top of its module or JS file, in `gpu_broker/tuning.py`, or as a documented
  `Settings` field. Hosts, URLs, unit names and labels belong in config or the catalog.
- **Files over ~200 lines.** One thing per module; split instead of squeezing. This applies
  to Python, JS and the host scripts.
- **Layering** (ARCHITECTURE.md): routes in `web/` call the `Broker`; only the scheduler's GPU
  thread changes residency; only drivers run processes; only `backends.py` makes HTTP calls
  to model servers.

### Security

Check every change against [docs/threat-model.md](docs/threat-model.md). In particular:

- **Unparameterised SQL.** Every value goes through a `?` placeholder (`gpu_broker/store.py`).
  An f-string may interpolate only column names or constants from a fixed set, with a
  `noqa: S608` and the reason. Anything derived from a request, a backend or the catalog in
  the SQL text is a finding.
- **Shell injection and argument smuggling.** No `shell=True`, `os.system` or string commands
  in Python; drivers pass argv lists and validate every value with `gpu_broker/drivers/validate.py`
  or `gpu_broker/units.py` first. In `host/gpu-broker-ctl` and `host/gpu-broker-gpu`: every
  variable quoted, no `eval`, no sourcing of recipe files, every argument checked against the
  same grammar as the broker (`tests/test_host_ctl_parity.py`), a new verb added to the
  allowlist only with validation, arguments that cannot be read as options (`--`, no leading
  `-`), and anything run inside a container bounded by a timeout.
- **Weakened checks.** Auth bypasses or new unauthenticated routes (`gpu_broker/web/app.py`),
  wider allowlists, relaxed grammars, fetches that skip `gpu_broker/netguard.py` or the
  `inputs.fetch_s` deadline, file names or paths taken from a caller or backend, outputs
  trusted from a program instead of listed by the driver, a GPU handed on without a confirmed
  clean.
- **New trust boundary.** A new route, driver verb, outbound request or file a request can
  influence needs an entry in docs/threat-model.md.

### Tests

- **Every fix and every security property needs a test that fails when the change is
  reverted.** Check this: in the review, say which test would fail without the fix, or flag
  that none would. A test that passes on both sides of the fix is not a regression test.
- Tests never touch real hardware, network or host binaries; use the fakes in
  `tests/helpers.py` and friends.
- Host-script changes are tested by running the real script against fakes on `PATH`
  (`tests/ctlfake.py`, `tests/test_host_ctl*.py`), not by reading its text.

## Making changes

- Keep PRs small and say what behaviour changes.
- Update CHANGELOG.md, README.md and docs/threat-model.md when behaviour, configuration or a
  security property changes.
- Run the three checks above before proposing a change.
