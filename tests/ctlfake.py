"""Runs host/gpu-broker-ctl for real, with fakes on PATH: `pct` logs every call, copies pushed
files, answers `stat` with an owner (none when NO_PARENT is set), reports one output for
`find` (or, given `files`, those of them matching its -name glob, unsorted), and exits 3 for any command containing FAIL. The job scan (`pct exec <ct> -- sh -c
REAP_SH ...`) runs the script's own in-container code for real, against a fake /proc
(`procs`) and with `kill` replaced by a fake that removes /proc entries (by pid or process
group) unless they are marked unkillable. Programs run by exec-run leave `leave` behind
(workers that escaped). `timeout` runs its command; `flock -n` fails `busy` times (`always`: never succeeds)."""
import pathlib
import subprocess

SCRIPT = pathlib.Path(__file__).parents[1] / "host/gpu-broker-ctl"
OWNER = "1000:1001"
TAG = "GPU_BROKER_JOB"
FAKE_PCT = r"""#!/bin/bash
if [[ "$1 $4" == "exec sh" ]]; then   # the job scan: run it for real against the fake /proc
  echo "scan $2 $8" >> "$PCT_LOG"
  f="$FAKE_PROC/../scan_fails"   # "always", or how many scans still fail
  if [[ -e "$f" ]]; then
    n=$(<"$f"); [[ "$n" == always ]] && exit 1
    (( n > 0 )) && { echo $((n - 1)) > "$f"; exit 1; }
  fi
  script="${6//kill -s KILL --/$FAKE_KILL -s KILL --}"
  exec sh -c "$script" "$7" "$8" "${9/#\/proc/$FAKE_PROC}"
fi
echo "$*" >> "$PCT_LOG"
if [[ "$1" == push ]]; then cp "$3" "$PUSHED/$(basename "$4")"; exit 0; fi
if [[ "$4" == timeout && -d "$LEAVE" ]]; then cp -R "$LEAVE"/. "$FAKE_PROC"/; fi
if [[ "$*" == *FAIL* ]]; then echo "model crashed" >&2; exit 3; fi
case "$4" in
  stat) [[ -n "${NO_PARENT:-}" ]] && exit 1; echo "OWNER"; exit 0 ;;
  find) [[ -n "${FAKE_FILES:-}" ]] || { echo "$5/scene.ply"; exit 0; }
        for n in $FAKE_FILES; do   # in FAKE_FILES order; `! -name .*` honoured
          [[ "$n" == ${11} ]] && ! [[ "${12} ${13} ${14}" == "! -name .*" && "$n" == .* ]] && echo "$5/$n"
        done; exit 0 ;;
esac
echo "chatter on stdout"
""".replace("OWNER", OWNER)
FAKE_KILL = r"""#!/bin/bash
[[ "$1 $2 $3" == "-s KILL --" ]] || { echo "fakekill: $*" >&2; exit 2; }
shift 3
die(){ [[ -e "$1/unkillable" ]] || rm -rf "$1"; }
for t in "$@"; do
  echo "$t" >> "$KILL_LOG"
  if [[ "$t" == -* ]]; then
    for d in "$FAKE_PROC"/[0-9]*; do
      s=$(cat "$d/stat"); read -r -a f <<< "${s##*) }"; [[ "${f[2]}" == "${t#-}" ]] && die "$d"
    done
  elif [[ -d "$FAKE_PROC/$t" ]]; then die "$FAKE_PROC/$t"; fi
done
exit 0
"""
FAKE_TIMEOUT = r"""#!/bin/bash
echo "$1 $2 $3 $4 $5 $6 $7" >> "$TIMEOUT_LOG"   # options, duration, and the start of the command
while [[ "$1" == -* ]]; do [[ "$1" == -k ]] && shift; shift; done
shift
exec "$@"
"""
FAKE_FLOCK = r"""#!/bin/bash
[[ "$1" == -n ]] || exit 0
left=$(cat "$FLOCK_BUSY" 2>/dev/null || echo 0)
[[ "$left" == always ]] && exit 1
(( left > 0 )) || exit 0
echo $((left - 1)) > "$FLOCK_BUSY"; exit 1
"""


def proc(root: pathlib.Path, pid: int, pgid: int, jid: str | None = None, unkillable: bool = False) -> None:
    """A fake /proc/<pid>: environ (tagged with `jid` if given), stat with process group `pgid`."""
    d = root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    env = ["PATH=/usr/bin", "HOME=/root"] + ([f"{TAG}={jid}"] if jid else [])
    (d / "environ").write_bytes(b"\0".join(e.encode() for e in env) + b"\0")
    (d / "stat").write_text(f"{pid} (py thon) S 1 {pgid} {pgid} 0 -1\n")
    if unkillable:
        (d / "unkillable").touch()


def make(tmp_path, recipe_text, name="t"):
    bin_, pushed, rdir = tmp_path / "bin", tmp_path / "pushed", tmp_path / "recipes"
    for d in (bin_, pushed, rdir):
        d.mkdir(exist_ok=True)
    for tool, text in (("pct", FAKE_PCT), ("fakekill", FAKE_KILL), ("timeout", FAKE_TIMEOUT), ("flock", FAKE_FLOCK)):
        (bin_ / tool).write_text(text)
        (bin_ / tool).chmod(0o755)
    (rdir / f"{name}.recipe").write_text(recipe_text)

    def run(cmd, stdin=b"", conf="", recipe=None, procs=(), leave=(), busy=0, scan_fails: int | str = 0, no_parent=False,
            files=()):
        """`procs`/`leave`: (pid, pgid, jid or None, unkillable) tuples. Returns (result, pct log,
        pushed dir, pids killed, pids left in the fake /proc)."""
        if recipe is not None:
            (rdir / f"{name}.recipe").write_bytes(recipe.encode())
        fake_proc, leave_dir = tmp_path / "proc", tmp_path / "leave"
        for d in (fake_proc, leave_dir):
            subprocess.run(["rm", "-rf", str(d)], check=True)
        proc(fake_proc, 1, 1)   # init: readable, never tagged
        for p in procs:
            proc(fake_proc, *p)
        for p in leave:
            proc(leave_dir, *p)
        (tmp_path / "scan_fails").unlink(missing_ok=True)
        if scan_fails:
            (tmp_path / "scan_fails").write_text(str(scan_fails))
        (tmp_path / "flock_busy").write_text(str(busy))
        cf = tmp_path / "ctl.conf"
        cf.write_text(f"LOG={tmp_path}/ctl.log\nRECIPES={rdir}\nLOCKS={tmp_path}/locks\nCLEAN_POLL_S=0\n{conf}")
        kills = tmp_path / "kills.log"
        kills.unlink(missing_ok=True)
        env = {"PATH": f"{bin_}:/usr/bin:/bin", "SSH_ORIGINAL_COMMAND": cmd, "GPU_BROKER_CTL_CONF": str(cf),
               "PCT_LOG": str(tmp_path / "pct.log"), "PUSHED": str(pushed), "FAKE_PROC": str(fake_proc),
               "FAKE_KILL": str(bin_ / "fakekill"), "KILL_LOG": str(kills), "LEAVE": str(leave_dir),
               "FLOCK_BUSY": str(tmp_path / "flock_busy"),
               "TIMEOUT_LOG": str(tmp_path / "timeout.log"), **({"NO_PARENT": "1"} if no_parent else {}),
               **({"FAKE_FILES": " ".join(files)} if files else {})}
        r = subprocess.run(["bash", str(SCRIPT)], env=env, input=stdin, capture_output=True, timeout=30)
        log = (tmp_path / "pct.log").read_text().splitlines() if (tmp_path / "pct.log").exists() else []
        (tmp_path / "pct.log").unlink(missing_ok=True)
        killed = kills.read_text().split() if kills.exists() else []
        left = sorted(int(p.name) for p in fake_proc.iterdir() if p.name != "1")
        return r, log, pushed, killed, left
    return run
