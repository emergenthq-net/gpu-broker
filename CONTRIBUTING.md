# Contributing

- **Tests first, and never against real hardware.** `pytest` must pass without a GPU,
  network or host: use the fakes in `tests/helpers.py` (`tests/conftest.py` fails any test
  that opens a URL or execs a host binary). A regression test for every bug fix.
- **`ruff check .` and `mypy` (strict) clean.** Python 3.12+.
- **Respect the layers** described in [ARCHITECTURE.md](ARCHITECTURE.md): routes in `web/`
  call the `Broker`; only the scheduler's GPU thread changes residency, and every LLM call
  holds an `LlmPool` slot; only drivers run
  processes; only `backends.py` makes HTTP calls.
- **One thing per module**, and no module over 200 lines — split it instead of squeezing it.
- **No magic values.** A literal that means something (a timeout, a port, a path, a size, an
  event name, a chart colour) is a named constant in `constants.py` or at the top of its
  module/JS file, or a `Settings` field with a documented default. Ruff's `PLR2004` enforces
  the numeric case. Hosts, URLs, unit names and labels belong in config or the catalog;
  examples, docs and tests use neutral public models.
- **New ComfyUI graph?** Add a builder to `gpu_broker/templates/` with its file names and
  tunables in a `DEFAULTS`-style mapping at the top, register it in `templates.TEMPLATES`,
  and record its graph in `tests/fixtures/golden_graphs.json`. Say which ComfyUI version and
  model files you verified it with.
- **New driver?** Implement `drivers.Driver`, validate every argument with
  `drivers.validate`, enforce the allowlist with `drivers.check`, take an injectable `run` so
  tests can record argv, and add it to `drivers.build`.
- Small PRs with a clear description of behaviour changes. By contributing you agree your
  work is licensed under Apache-2.0.
