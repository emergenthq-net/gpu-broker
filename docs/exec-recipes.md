# Exec recipes (`runner: exec`)

Some models are a program, not a ComfyUI graph, such as an image-to-3D tool. gpu-broker runs
them as **exec jobs**:

- It stops the resident LLM and frees ComfyUI's weights.
- It hands the job's input files to the program.
- It returns the files the program wrote.

The command is never in the catalog or the request. It lives in a **recipe** file that the
host's administrator writes. A working example is
[`examples/recipes/sharp.recipe`](../examples/recipes/sharp.recipe).

## Catalog entry

The catalog names the recipe and the broker's own time limit:

```yaml
  apple-sharp:
    kind: 3d
    runner: exec
    exec: {recipe: sharp, timeout_s: 660}   # >= recipe timeout_s + 10 s kill grace + 30 s
    inputs: {image: required}
    caps: [image_to_splat]
    vram_mib: 12000
    status: ready
```

| key | what it does |
|---|---|
| `exec.recipe` | Name of the recipe file, without `.recipe`. |
| `exec.timeout_s` | How long the broker waits for the job. Must exceed the recipe's `timeout_s` + 10 s + 30 s (see [Timeouts](#timeouts-and-cleanup)). |
| `exec.params` | Request keys passed to the program as `params.json` (scalars only). |
| `exec.choices` | `{param: [values]}`: the only values a param accepts. Anything else is a 400 at submit. |

## Recipe format

A recipe is `/etc/gpu-broker/recipes/<name>.recipe`: `key=value` lines, `#` comments.

```ini
argv=/opt/ml-sharp/.venv/bin/sharp predict -i {in_dir} -o {out_dir} -c {checkpoint} --no-render
checkpoint=/var/lib/gpu-broker/models/apple-sharp/sharp_2572gikvuh.pt
in_dir=/var/lib/gpu-broker/exec/in/{jid}
out_dir=/var/lib/gpu-broker/exec/out/{jid}
outputs=*.ply
timeout_s=600
```

| key | what it does |
|---|---|
| `argv` | The command. Split on spaces; no shell, no quoting. `{jid}`, `{in_dir}`, `{out_dir}` and `{checkpoint}` are substituted. |
| `checkpoint` | Optional path, substituted as `{checkpoint}`. |
| `in_dir` | Where input files are written. Absolute; must contain `{jid}`. |
| `out_dir` | Where the program writes results. Absolute; must end in `/{jid}`. |
| `outputs` | File name globs, space-separated (`outputs=result.mp4 run.log`). |
| `timeout_s` | The program gets SIGTERM after this many seconds. |
| `target` | Proxmox only: the container id to run in. Local drivers refuse it. |

The full rules are in [`gpu_broker/drivers/recipes.py`](../gpu_broker/drivers/recipes.py).

## Inputs and outputs

- **Inputs** land in `in_dir` as `<slot>[-NN].<ext>`: `image.png`, `frames-00.png`, ...
- **`out_dir`**: only the final `{jid}` folder is created. Create its parent once.
  - On Proxmox the folder takes the parent's owner, so ComfyUI's user can serve and prune it.
- **Output order**: each glob's files, sorted by name, in the order the globs are listed.
  - A file is listed once, so the first glob decides `outputs[0]`.
  - Files whose names start with `.` are never outputs.
- **ComfyUI links**: when `out_dir` is under `comfy.output_dir`, each output also gets a ComfyUI
  `/view` URL.

## Timeouts and cleanup

- The program runs with `GPU_BROKER_JOB=<job id>` in its environment.
- At the recipe's `timeout_s` it gets SIGTERM; 10 s later, SIGKILL.
- Afterwards, every process still carrying that job id is killed, including workers that left
  its process group.
- The broker refuses to start if:
  - a catalog `exec.timeout_s` is shorter than the recipe timeout + 10 s + 30 s, or
  - `timeouts.exec_clean_s` is shorter than the driver's clean time + 10 s.
- It checks the timeout again before an exec job evicts anything.
- Before the program starts, the broker waits (up to `timeouts.exec_vram_s`) until the card
  shows the entry's `vram_mib` free.

## GPU hold after a crash

If a recipe might still be running after its job ended, the broker **holds the GPU**: no job
runs until the hold is lifted.

- **When it happens**
  - The job's processes could not be confirmed gone after a timeout or failure.
  - The broker restarted while an exec job was running.
- **How it clears**
  - Automatically: a background clean retries every `intervals.held_retry_s` and lifts the hold
    once the processes are gone.
  - By hand: make sure the processes are gone, then `POST /v1/admin/gpu-held/clear`. It takes
    effect at once.
- **Where it shows**: `/v1/status` and the dashboard. The hold survives resume and restarts.
- A direct chat never holds the GPU.

## Driver support

| driver | runs recipes |
|---|---|
| `systemd` | yes, on the broker's machine |
| `proxmox` | yes, inside a container (`target=<ct>` in the recipe) |
| `docker` | no |

## Housekeeping

- **Job folders are not removed by the broker.** Install
  [`examples/systemd/gpu-broker-exec-prune@.{path,service}`](../examples/systemd/gpu-broker-exec-prune@.service)
  where they are written.
  - It deletes them two hours after the job, which is how long results stay downloadable.
  - Raise its `-mmin` to keep them longer.
- **Recipe files must use LF line ends.** CRLF files are refused.

## Edge cases

- Finding a job's processes needs a readable `/proc`.
  - On the local driver, run the recipe as the broker's own user. A recipe that switches user
    is not tracked.
  - With `/proc` mounted `hidepid`, the scan reports "unknown" while the program is visible.
- A restart still cleans an exec job whose model has since left the catalog: the recipe is
  recorded with the job at submit.
- A job recorded by a broker older than 0.3.0, whose model is gone and which is not a chat, is
  held until an operator clears it.
- The Proxmox driver adds its SSH connect timeout to the clean time it reports.
