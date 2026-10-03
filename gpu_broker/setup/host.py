"""Changing the machine for `gpu-broker setup`: files, the token, the systemd service.

`Host` does every write and command, as this user, as root, or through `sudo -n` when setup
installs a system service without being root. In a dry run it only says what it would do.
Commands are argv lists, never shell strings.
"""
from __future__ import annotations

import os
import pathlib
import secrets
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

from ..constants import TOKEN_ENV

UNIT_NAME = "gpu-broker"
UNIT_PATH = f"/etc/systemd/system/{UNIT_NAME}.service"
TOKEN_BYTES = 24
SHOWN = 4                       # characters of the token setup prints; the rest stays in the file
CMD_TIMEOUT_S = 60              # systemctl, uv tool install
PRIVATE, PUBLIC = 0o600, 0o644
UNIT = """\
# Written by `gpu-broker setup`. Setup never overwrites it; delete it and run setup again to regenerate.
[Unit]
Description=gpu-broker: one GPU, many models
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile={env}
ExecStart={exe} -c {config} serve
Restart=on-failure
# Shutdown closes open HTTP connections after server.graceful_shutdown_s (default 10 s).
TimeoutStopSec=15

[Install]
WantedBy=multi-user.target
"""

Run = Callable[..., subprocess.CompletedProcess[str]]


def run_cmd(argv: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=CMD_TIMEOUT_S, check=False, **kw)  # type: ignore[call-overload,no-any-return]  # noqa: S603


def masked(token: str) -> str:
    return token[:SHOWN] + "…"


def systemd_running(root: str = "/") -> bool:
    """systemd is PID 1 here (containers and WSL often have systemctl without it)."""
    return shutil.which("systemctl") is not None and pathlib.Path(root, "run/systemd/system").is_dir()


def can_sudo(run: Run = run_cmd) -> bool:
    """Passwordless sudo: setup never asks for a password."""
    if shutil.which("sudo") is None:
        return False
    try:
        return run(["sudo", "-n", "true"]).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@dataclass
class Host:
    dry: bool
    sudo: bool                      # write and run through `sudo -n` (system layout, not root)
    run: Run = run_cmd
    say: Callable[[str], None] = print
    did: list[str] = field(default_factory=list)   # what was (or would be) done, for tests

    def _note(self, line: str) -> None:
        self.did.append(line)
        if self.dry:
            self.say(f"would    {line}")

    def _sudo(self, argv: list[str]) -> list[str]:
        return ["sudo", "-n", *argv] if self.sudo else argv

    def cmd(self, argv: list[str], as_root: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a command that changes something (skipped in a dry run). `as_root=False` runs it
        as this user even when setup is using sudo."""
        self._note("run      " + " ".join(argv))
        if self.dry:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return self.run(self._sudo(argv) if as_root else argv)

    def exists(self, path: str) -> bool:
        if not self.sudo:
            return os.path.exists(path)
        return self.run(["sudo", "-n", "test", "-e", path]).returncode == 0

    def read(self, path: str) -> str | None:
        """A file's text, or None if it is not there (read through sudo for root-only files)."""
        if not self.sudo:
            try:
                return pathlib.Path(path).read_text()
            except (FileNotFoundError, PermissionError):
                return None
        r = self.run(["sudo", "-n", "cat", "--", path])
        return r.stdout if r.returncode == 0 else None

    def mkdir(self, path: str) -> None:
        if os.path.isdir(path):
            return
        self._note(f"create   {path}/")
        if self.dry:
            return
        if self.sudo:
            self._check(self.run(["sudo", "-n", "install", "-d", "-m", "755", "--", path]), path)
        else:
            os.makedirs(path, exist_ok=True)

    def write(self, path: str | pathlib.Path, text: str, mode: int = PUBLIC) -> None:
        self._note(f"write    {path} (mode {mode:o})")
        if self.dry:
            return
        if self.sudo:
            self._check(self.run(["sudo", "-n", "install", "-m", f"{mode:o}", "--", "/dev/stdin", str(path)],
                                 input=text), str(path))
            return
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(path, mode)        # O_CREAT's mode is masked by umask; an existing file keeps its own

    @staticmethod
    def _check(r: subprocess.CompletedProcess[str], what: str) -> None:
        if r.returncode != 0:
            raise PermissionError(f"{what}: {(r.stderr or r.stdout).strip()}")


def token(host: Host, env_file: str) -> tuple[str, bool]:
    """The API token: the one already in `env_file`, or a new one written there (mode 0600).
    Returns (token, whether it is new)."""
    old = host.read(env_file)
    for line in (old or "").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == TOKEN_ENV and value.strip():
            return value.strip(), False
    new = secrets.token_urlsafe(TOKEN_BYTES)
    keep = [ln for ln in (old or "").splitlines() if ln.partition("=")[0].strip() != TOKEN_ENV]
    host.write(env_file, "\n".join([*keep, f"{TOKEN_ENV}={new}"]) + "\n", PRIVATE)
    return new, True


def _quoted(arg: str) -> str:
    """One ExecStart word: systemd splits on spaces and honours double quotes."""
    return f'"{arg}"' if " " in arg else arg


def unit_text(exe: list[str], config: str, env_file: str) -> str:
    return UNIT.format(exe=" ".join(map(_quoted, exe)), config=_quoted(config), env=env_file)
