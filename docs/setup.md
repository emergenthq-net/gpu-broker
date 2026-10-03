# `gpu-broker setup`

One command from a fresh install to a running broker. It asks no questions: everything has a
default, and it never overwrites a file you already have.

```bash
gpu-broker setup              # do it
gpu-broker setup --dry-run    # print every change it would make; change nothing
```

## Flags

| flag | effect |
|---|---|
| `--dry-run` | detect, then print each file it would write and command it would run. Nothing changes. |
| `--yes` | never ask, and never run `serve` in the foreground. For scripts and provisioning. |
| `--dir DIR` | put `config.yaml`, `catalog.yaml` and `broker.env` in `DIR` instead of the default. |

## What it does, in order

1. **Finds the GPU.** It uses the same reader as `serve`: `nvidia-smi` first, then the amdgpu
   driver's files. It records the card's name and memory. No GPU is not an error; the broker
   starts anyway and keeps looking.
2. **Finds model servers** (table below). Detection only reads: it starts and stops nothing.
3. **Writes the config and catalog** for what it found, through the same writer as `gpu-broker init`.
   - A file that already exists is kept, and setup says so.
   - No LLM server found: it writes the starter catalog instead, for you to edit.
   - Both files are loaded and validated, as `serve` would, before anything is written.
4. **Creates the API token** in `broker.env`, mode 600.
   - An existing token is kept.
   - Setup prints only its first four characters.
5. **Starts the broker.**
   - With systemd, and root or passwordless sudo: it installs `/etc/systemd/system/gpu-broker.service`,
     then enables and starts it. It restarts the service only if something changed.
   - Otherwise it prints the exact `serve` command. At a terminal (and without `--yes`) it also
     runs that command in the foreground; Ctrl-C stops it.
6. **Checks it.** It runs `gpu-broker check` first and stops if that finds problems. After starting,
   it waits up to 60 s for `/health`.
7. **Opens the dashboard** in your browser and prints its address.
   - Not over SSH, and not on Linux without a display: then it only prints the address.
   - The dashboard asks for the token once; it is in `broker.env`.

## What it detects

| server | asked on | identified by | runs under |
|---|---|---|---|
| llama.cpp `llama-server` | `127.0.0.1:8080/v1/models` | `owned_by: llamacpp` | a unit or container whose name contains `llama` |
| vLLM | `127.0.0.1:8000/v1/models` | `owned_by: vllm` | `vllm` |
| Ollama | `127.0.0.1:11434/api/tags` | its model list | `ollama` |
| ComfyUI | `127.0.0.1:8188/system_stats` | its system block | `comfy` |

- **Units:** installed systemd services, both system and `--user`, whose names contain one of
  those words. Template units (`name@.service`) are skipped.
- **Containers:** read from the Docker socket, only when it is readable. A container matches by
  its name or its image.
- **A unit beats a container.** The driver is `docker` only when every server found runs in a
  container.
- **Idle units:** a matching unit or container that did not answer is reported, not added. Start
  it, then add it to the catalog ([docs/catalog.md](catalog.md)).
- **No unit at all:** setup still adds the server, under its usual unit name, and tells you to
  create that unit. Without one the broker cannot stop the server to free the GPU.

## What it writes

| file | system install | without root |
|---|---|---|
| config, catalog, `broker.env` | `/etc/gpu-broker/` | `~/.config/gpu-broker/` |
| database, input staging, downloads | `/var/lib/gpu-broker/` | `~/.local/share/gpu-broker/` |
| event log | `/var/log/gpu-broker/` | `~/.local/state/gpu-broker/` |

The user folders follow `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `XDG_STATE_HOME` when they are set.

### Catalog entries

- **One entry per model a server reports.** For Ollama, the first six.
- **The first LLM found is the resident model:** the one loaded when nothing else is running.
- **Ollama entries** set `health_path: /api/version`, since Ollama has no `/health`.
- **ComfyUI checkpoints** whose names contain `xl` get an image entry using the stock `sdxl`
  graph (up to four). Other image and video models need the files their template expects; add
  them by hand ([docs/catalog.md](catalog.md)).
- **Every generated entry has a `notes` line** saying setup found it.

### Memory estimates

`vram_mib` decides when the broker must stop one model to make room for another. Setup errs on
the safe side, so a wrong guess costs a switch, never an out-of-memory crash.

| server reports | estimate |
|---|---|
| the weights' size (Ollama, recent llama.cpp) | size × 1.15 + 1 GiB for context, rounded up |
| nothing (vLLM) | 90% of the card: vLLM takes that much up front |
| nothing (others) | the whole card, less 600 MiB of headroom |

No estimate ever exceeds the card. Measure the real peak (`nvidia-smi` while the model answers)
and put it in the catalog to let models share the card.

## Running it again

Safe at any time. It keeps:

- `config.yaml` and `catalog.yaml`, even when you have edited them;
- the token in `broker.env`;
- `gpu-broker.service`, even if it differs from what setup would write now.

To regenerate a file, delete it and run setup again.

## Installed with `uvx`

`uvx` runs gpu-broker from a cache that uv may clean, so a service cannot point there. When setup
installs the service from a `uvx` run, it first runs `uv tool install gpu-broker==<this version>`
and points the service at that copy.

## Lower-level commands

- `gpu-broker init` writes the starter files only, for [manual setup](../README.md#manual-setup).
- `gpu-broker check` validates a config and catalog without starting anything.
- `gpu-broker serve` runs the API; it needs `BROKER_TOKEN` in its environment.
