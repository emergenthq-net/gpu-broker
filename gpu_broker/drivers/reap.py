"""Finding and stopping an exec job's processes by their environment tag.

Every exec recipe runs with GPU_BROKER_JOB=<job id> in its environment, and children inherit
it — including workers that start a new session or double-fork to leave the recipe's process
group. So "is anything of this job still running?" is answered by scanning
/proc/<pid>/environ, never by process group or command line (`pgrep -f` would also match the
scanner itself). host/gpu-broker-ctl does the same inside the container (`reap_job`).

The answer is three-valued: gone, still running, or unknown. Only "gone" lets the GPU go to
the next job. Unknown: no /proc here; a process of our own user whose environment we may not
read (EACCES/EPERM: a setuid or non-dumpable program); or /proc hiding processes (mounted with
hidepid), noticed when a pid we know is alive is missing from the listing.

Limits: a process of another user is not readable and is skipped, so a recipe that switches
user (sudo, setuid) is not tracked; and the hidepid check needs a pid we know is alive (the
recipe's own while it is unreaped), so a hiding /proc seen only after that reads as "gone".
Run the broker (or the host script) as root with an ordinary /proc for full coverage.
"""
from __future__ import annotations

import contextlib
import errno
import os
import pathlib
import signal
import time
from collections.abc import Callable

MARK = "GPU_BROKER_JOB"     # environment variable carrying the job id (gpu-broker-ctl: MARK)
PROC = "/proc"
POLL_S = 0.2


def tag(jid: str) -> bytes:
    return f"{MARK}={jid}".encode()


GONE = (errno.ENOENT, errno.ESRCH)    # the process exited during the scan


def tagged(jid: str, proc: str = PROC, alive: int | None = None) -> list[int] | None:
    """Pids whose environment holds this job's tag; None when the scan cannot tell (no /proc, a
    process of ours it may not read, or `alive` — a pid known to exist — missing from /proc)."""
    root, want, found, uid = pathlib.Path(proc), tag(jid), [], os.getuid()
    try:
        entries = [e for e in root.iterdir() if e.name.isdigit()]
    except OSError:
        return None
    if alive is not None and str(alive) not in {e.name for e in entries}:
        return None   # /proc hides processes (hidepid): an empty answer would mean nothing
    for e in entries:
        try:
            env = (e / "environ").read_bytes()
        except OSError as err:
            if err.errno in GONE:
                continue
            if err.errno in (errno.EACCES, errno.EPERM) and _owner(e) not in (uid, None):
                continue   # another user's process: not one of ours (see Limits above)
            return None
        if want in env.split(b"\0"):
            found.append(int(e.name))
    return found


def _owner(entry: pathlib.Path) -> int | None:
    try:
        return entry.stat().st_uid
    except OSError:
        return None


def _group(pid: int, proc: str) -> int | None:
    """The process group id from /proc/<pid>/stat (the field after `) state ppid`)."""
    try:
        stat = pathlib.Path(proc, str(pid), "stat").read_text()
        return int(stat.rpartition(") ")[2].split()[2])
    except (OSError, ValueError, IndexError):
        return None


def reap(jid: str, wait_s: float, proc: str = PROC, kill: Callable[[int, int], None] = os.kill,
         killpg: Callable[[int, int], None] = os.killpg, clock: Callable[[], float] = time.monotonic,
         sleep: Callable[[float], None] = time.sleep, alive: int | None = None) -> bool:
    """SIGKILL every process tagged with `jid` (and its process group) until none is left;
    True once a scan finds none, False if some survive `wait_s` or the scan cannot tell."""
    end = clock() + wait_s
    own = os.getpgrp()
    while True:
        pids = tagged(jid, proc, alive)
        if pids is None:
            return False
        if not pids:
            return True
        if clock() >= end:
            return False
        for pid in pids:
            group = _group(pid, proc)
            with contextlib.suppress(OSError):
                if group is not None and group > 1 and group != own:   # never our own group, never -1
                    killpg(group, signal.SIGKILL)
            with contextlib.suppress(OSError):
                kill(pid, signal.SIGKILL)
        sleep(POLL_S)
