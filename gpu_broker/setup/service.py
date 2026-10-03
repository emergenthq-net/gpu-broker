"""The service `gpu-broker setup` installs: who it runs as, whether its code is safe to run, and
the unit and sudoers files.

The broker never runs as root. A system service runs as the account that owns the
installation, or as a dedicated `gpu-broker` account when root owns it; it reaches the GPU
through group membership (`SupplementaryGroups=`), and starts and stops system units only
through a sudoers rule that names those units and nothing else. A user service (for model
servers that are `systemctl --user` units) runs as the user, kept alive after logout by
lingering.

Setup refuses to install a service whose code (the script, its interpreter, the package and
everything installed beside it) someone other than root or the service account could change,
or that the service account cannot read.
"""
from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .host import exec_word, path_value

DEDICATED = "gpu-broker"           # the account a root-owned installation runs as
GPU_GROUPS = ("video", "render")   # device access for AMD (/dev/kfd, /dev/dri) and some NVIDIA setups
DOCKER_GROUP = "docker"
SUDOERS = "/etc/sudoers.d/gpu-broker"
SUDOERS_MODE = 0o440
UNIT_SAFE = re.compile(r"^[A-Za-z0-9_.@-]+$")   # unit names a sudoers rule may name without escaping
SITE_PROBE = ("import gpu_broker, os, sys; "
              "print(os.path.dirname(os.path.dirname(os.path.realpath(gpu_broker.__file__))))")
PIP_TRAMPOLINE = re.compile(r"""^'''exec' "?([^"]+?)"? "\$0" "\$@"$""")   # pip's launcher for paths with spaces
WRITE_BITS = stat.S_IWGRP | stat.S_IWOTH


@dataclass(frozen=True)
class Account:
    name: str
    uid: int
    gid: int
    home: str
    groups: tuple[int, ...] = ()   # supplementary group ids


@dataclass(frozen=True)
class Install:
    """What the service would run: the script, the Python it runs under and the folder the
    package is installed in (None where they could not be found)."""
    exe: str
    interpreter: str | None
    site: str | None

    def files(self) -> list[str]:
        return [p for p in (self.exe, self.interpreter) if p]


def interpreter(exe: list[str], read: Callable[[str], str | None]) -> str | None:
    """The Python an entry-point script runs: its shebang, or pip's /bin/sh trampoline (used when
    the path has spaces). `python -m gpu_broker` runs exe[0] itself."""
    if len(exe) > 1 and exe[1] == "-m":
        return exe[0]
    head = (read(os.path.realpath(exe[0])) or "").splitlines()[:2]   # `uv tool` links its scripts
    if not head or not head[0].startswith("#!"):
        return None
    first = head[0][2:].strip()
    if first in ("/bin/sh", "/usr/bin/sh") and len(head) > 1 and (m := PIP_TRAMPOLINE.match(head[1].strip())):
        return m.group(1)
    return first.split()[0] if first else None


def inspect(exe: list[str], read: Callable[[str], str | None],
            site_of: Callable[[str], str | None]) -> Install:
    interp = interpreter(exe, read)
    return Install(exe[0], interp, site_of(interp) if interp else None)


def owner_uid(inst: Install, lstat: Callable[[str], os.stat_result]) -> int | None:
    """Who owns the installation: the package folder's owner, else the script's."""
    for p in (inst.site, inst.exe):
        if p:
            try:
                return lstat(os.path.realpath(p)).st_uid
            except OSError:
                continue
    return None


def _chain(path: str) -> list[str]:
    """`path` (symlinks resolved) and every folder above it."""
    real = os.path.realpath(path)
    out = [real]
    while (parent := os.path.dirname(out[-1])) != out[-1]:
        out.append(parent)
    return out


@dataclass(frozen=True)
class Checker:
    """Whether an account, and only it and root, controls a set of paths."""
    acct: Account
    lstat: Callable[[str], os.stat_result] = os.lstat
    private_group: Callable[[int, int], bool] = lambda gid, uid: False   # main passes private_group

    def writable_by_others(self, path: str, st: os.stat_result) -> str | None:
        if stat.S_ISLNK(st.st_mode):
            return None                       # a symlink's own mode means nothing; its target is checked
        if st.st_uid not in (0, self.acct.uid):
            return f"{path} is owned by uid {st.st_uid}"
        sticky_dir = stat.S_ISDIR(st.st_mode) and st.st_mode & stat.S_ISVTX
        if st.st_mode & stat.S_IWOTH and not sticky_dir:
            return f"{path} is writable by everyone"
        if st.st_mode & stat.S_IWGRP and st.st_gid != 0 and not self.private_group(st.st_gid, self.acct.uid) \
                and not sticky_dir:
            return f"{path} is writable by group {st.st_gid}"
        return None

    def can(self, st: os.stat_result, bits: int) -> bool:
        """`bits` (r=4, x=1) are granted to the account by owner, group or other permissions."""
        if self.acct.uid == 0:
            return True
        if st.st_uid == self.acct.uid:
            return st.st_mode >> 6 & bits == bits
        if st.st_gid == self.acct.gid or st.st_gid in self.acct.groups:
            return st.st_mode >> 3 & bits == bits
        return st.st_mode & bits == bits

    def problems(self, inst: Install, walk: Callable[[str], Iterable[str]]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()

        def look(path: str, need: int | None) -> None:
            """`need`: the permission bits the account needs (None: read, plus pass for a folder)."""
            if path in seen:
                return
            seen.add(path)
            try:
                st = self.lstat(path)
            except OSError as e:
                out.append(f"{path}: {e.strerror}")
                return
            if why := self.writable_by_others(path, st):
                out.append(why)
            if need is None:
                need = 0 if stat.S_ISLNK(st.st_mode) else 5 if stat.S_ISDIR(st.st_mode) else 4
            if need and not self.can(st, need):
                out.append(f"{path} is not readable by {self.acct.name}")

        for f in inst.files():
            chain = _chain(f)
            look(chain[0], 5)                 # read and execute the file
            for d in chain[1:]:
                look(d, 1)                    # pass through every folder above it
        if inst.site:
            chain = _chain(inst.site)
            for d in chain[1:]:
                look(d, 1)
            for p in walk(chain[0]):
                look(p, None)
        return out


def walk(root: str) -> Iterable[str]:
    """Every path under `root`, `root` included, without following symlinks."""
    yield root
    for d, dirs, files in os.walk(root):
        for name in (*dirs, *files):
            yield os.path.join(d, name)


def supplementary(group_exists: Callable[[str], bool], docker: bool) -> list[str]:
    return [g for g in (*GPU_GROUPS, *([DOCKER_GROUP] if docker else [])) if group_exists(g)]


SYSTEM_UNIT = """\
# Written by `gpu-broker setup`. Setup never overwrites it; delete it and run setup again to regenerate.
[Unit]
Description=gpu-broker: one GPU, many models
After=network-online.target
Wants=network-online.target

[Service]
User={user}
{groups}EnvironmentFile={env}
ExecStart={exe} -c {config} serve
Restart=on-failure
# Shutdown closes open HTTP connections after server.graceful_shutdown_s (default 10 s).
TimeoutStopSec=15

[Install]
WantedBy=multi-user.target
"""

USER_UNIT = """\
# Written by `gpu-broker setup`. Setup never overwrites it; delete it and run setup again to regenerate.
[Unit]
Description=gpu-broker: one GPU, many models

[Service]
EnvironmentFile={env}
ExecStart={exe} -c {config} serve
Restart=on-failure
# Shutdown closes open HTTP connections after server.graceful_shutdown_s (default 10 s).
TimeoutStopSec=15

[Install]
WantedBy=default.target
"""


def system_unit(exe: list[str], config: str, env_file: str, user: str, groups: list[str]) -> str:
    return SYSTEM_UNIT.format(user=exec_word(user), env=path_value(env_file),
                              groups=f"SupplementaryGroups={' '.join(groups)}\n" if groups else "",
                              exe=" ".join(map(exec_word, exe)), config=exec_word(config))


def user_unit(exe: list[str], config: str, env_file: str) -> str:
    return USER_UNIT.format(env=path_value(env_file), exe=" ".join(map(exec_word, exe)), config=exec_word(config))


def sudoers(user: str, systemctl: str, units: list[str]) -> tuple[str, list[str]]:
    """The rule letting `user` start, stop and query exactly these units, the way the systemd
    driver calls them (`sudo -n systemctl <verb> -- <unit>`). Returns (text, units left out
    because their names would need sudoers escaping)."""
    ok = sorted({u for u in units if UNIT_SAFE.match(u)})
    skipped = sorted({u for u in units if not UNIT_SAFE.match(u)})
    cmds = ", ".join(f"{systemctl} {verb} -- {u}" for u in ok for verb in ("start", "stop", "is-active"))
    text = ("# Written by `gpu-broker setup`: the broker may start, stop and query the model-server units in\n"
            "# its catalog, and nothing else. Setup never overwrites it; delete it and run setup again.\n")
    return text + (f"{user} ALL=(root) NOPASSWD: {cmds}\n" if ok else ""), skipped


# The real lookups.

def account(key: str | int) -> Account | None:
    import pwd
    try:
        pw = pwd.getpwnam(key) if isinstance(key, str) else pwd.getpwuid(key)
    except KeyError:
        return None
    groups = tuple(g for g in os.getgrouplist(pw.pw_name, pw.pw_gid) if g != pw.pw_gid)
    return Account(pw.pw_name, pw.pw_uid, pw.pw_gid, pw.pw_dir, groups)


def group_exists(name: str) -> bool:
    import grp
    try:
        grp.getgrnam(name)
    except KeyError:
        return False
    return True


def private_group(gid: int, uid: int) -> bool:
    """`gid` is the user's own group: their primary group, with no other members."""
    import grp
    import pwd
    try:
        g, me = grp.getgrgid(gid), pwd.getpwuid(uid)
    except KeyError:
        return False
    return me.pw_gid == gid and not [m for m in g.gr_mem if m != me.pw_name] and \
        all(p.pw_gid != gid or p.pw_uid == uid for p in pwd.getpwall())
