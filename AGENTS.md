# AGENTS.md

Guidance for automated agents working on this repository: code review, security review and
code changes.

- Rules behind these checks: [CONTRIBUTING.md](CONTRIBUTING.md), [ARCHITECTURE.md](ARCHITECTURE.md).
- What each security mitigation protects, and which tests hold it: [docs/threat-model.md](docs/threat-model.md).

## Checks before review

```sh
ruff check .          # includes PLR2004 (magic numbers) and bandit (S) rules
mypy                  # strict
pytest -q             # never needs a GPU, network or host: tests/conftest.py blocks them
```

## Review guidelines

Flag each of these. Cite the file and line, and say what to do instead.

### Release hygiene

- **Private infrastructure.**
  - Hostnames, internal domains, private or LAN IP addresses.
  - Container or VM ids from a real site.
  - Personal home-directory paths (`/Users/<name>`, `/home/<name>`), real usernames or email addresses.
  - Use neutral values instead: `127.0.0.1`, `localhost`, documentation ranges
    (`192.0.2.0/24`, `2001:db8::/32`), container id `101`, paths like `/srv/...` or `/var/lib/gpu-broker`.
- **AI or tool attribution** in code, comments, docs, commit messages or PR bodies.
  - "Generated with", or `Co-Authored-By` trailers naming an assistant.
  - Claude, Codex, Copilot or similar named as an author.
  - Commits are authored by the human maintainer.
- **Secrets.** Tokens, API keys, private keys, passwords, or anything that looks like one.
  - Includes test fixtures: tests use obviously fake values.
  - Secrets come from the environment (`BROKER_TOKEN`, `UPSTREAM_TOKEN_*`), never config files,
    logs, responses or URLs.

### Code quality

- **Hard-coded values that should be constants or config.**
  - A literal that means something (timeout, port, path, size, limit, event name, colour)
    belongs in `gpu_broker/constants.py`, at the top of its module or JS file, in
    `gpu_broker/tuning.py`, or as a documented `Settings` field.
  - Hosts, URLs, unit names and labels belong in config or the catalog.
- **Structure.**
  - A module that mixes clearly unrelated responsibilities.
  - Code mangled to fit an arbitrary limit: crammed lines, stripped names or comments.
- **Layering** (ARCHITECTURE.md).
  - Routes in `web/` call the `Broker`.
  - Only the scheduler's GPU thread changes residency.
  - Only drivers run processes.
  - Only `backends.py` makes HTTP calls to model servers.

### Security

Check every change against [docs/threat-model.md](docs/threat-model.md). In particular:

- **Unparameterised SQL.**
  - Every value goes through a `?` placeholder (`gpu_broker/store.py`).
  - An f-string may interpolate only column names or constants from a fixed set, with a
    `noqa: S608` and the reason.
  - Anything derived from a request, a backend or the catalog in the SQL text is a finding.
- **Shell injection and argument smuggling in Python.**
  - No `shell=True`, `os.system` or string commands.
  - Drivers pass argv lists, and validate every value with `gpu_broker/drivers/validate.py`
    or `gpu_broker/units.py` first.
- **Shell injection in `host/gpu-broker-ctl` and `host/gpu-broker-gpu`.**
  - Every variable quoted; no `eval`; no sourcing of recipe files.
  - Every argument checked against the broker's grammar (`tests/test_host_ctl_parity.py`).
  - A new verb joins the allowlist only with validation.
  - Arguments cannot be read as options (`--`, no leading `-`).
  - Anything run inside a container is bounded by a timeout.
- **Weakened checks.**
  - Auth bypasses, or new unauthenticated routes (`gpu_broker/web/app.py`).
  - Wider allowlists or relaxed grammars.
  - Fetches that skip `gpu_broker/netguard.py` or the `inputs.fetch_s` deadline.
  - File names or paths taken from a caller or backend.
  - Outputs trusted from a program instead of listed by the driver.
  - A GPU handed on without a confirmed clean.
- **New trust boundary.** A new route, driver verb, outbound request, or file a request can
  influence needs an entry in docs/threat-model.md.

### Tests

- **Every fix and every security property needs a test that fails when the change is reverted.**
  - In the review, say which test would fail without the fix, or flag that none would.
  - A test that passes on both sides of the fix is not a regression test.
- Tests never touch real hardware, network or host binaries. Use the fakes in `tests/helpers.py` and friends.
- Host-script changes are tested by running the real script against fakes on `PATH`
  (`tests/ctlfake.py`, `tests/test_host_ctl*.py`), not by reading its text.

## Making changes

- Keep PRs small and say what behaviour changes.
- Update CHANGELOG.md, README.md and docs/ when behaviour, configuration or a security property changes.
  - Changelog entries go under `## [Unreleased]`, in Keep a Changelog groups.
- Run the three checks above before proposing a change.
