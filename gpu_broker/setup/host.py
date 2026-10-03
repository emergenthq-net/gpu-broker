"""Changing the machine for `gpu-broker setup`: files, the token, commands, unit-file quoting.

`Host` does every write and command, as this user, as root, or through `sudo -n` when setup
installs a system service without being root. In a dry run it only says what it would do.
Commands are argv lists, never shell strings. A command that cannot run (missing, timed out)
comes back as a failed result, so callers report it like any other failure.

Files are written to a new temporary file in the same folder (created 0600, never following a
symlink) and renamed into place, so a reader never sees half a file and a planted symlink is
never written through: setup refuses to replace a symlink at all.
"""
from __future__ import annotations

import contextlib
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
CMD_TIMEOUT_S = 60              # systemctl, useradd, loginctl
INSTALL_TIMEOUT_S = 300         # `uv tool install` downloads the package and its dependencies
NO_RESULT = -1                  # the returncode of a command that could not run at all
PRIVATE, PUBLIC = 0o600, 0o644

Run = Callable[..., subprocess.CompletedProcess[str]]


class UnsafePath(Exception):
    """A path setup will not write: a symlink (dangling or not) or a folder."""


def run_cmd(argv: list[str], timeout: float = CMD_TIMEOUT_S, **kw: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False, **kw)  # type: ignore[call-overload,no-any-return]  # noqa: S603


def ran(run: Run, argv: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    """`run`, with a command that cannot start or times out as a failed result."""
    try:
        return run(argv, **kw)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(argv, NO_RESULT, "", f"{argv[0]}: timed out after {e.timeout:g} s")
    except (OSError, subprocess.SubprocessError) as e:
        return subprocess.CompletedProcess(argv, NO_RESULT, "", f"{argv[0]}: {e}")


def parse_env(text: str) -> dict[str, str]:
    """An EnvironmentFile the way systemd reads it: blank lines and #/; comments skipped, an
    optional `export `, surrounding quotes stripped, the last assignment of a key winning."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        line = line.removeprefix("export ").lstrip()
        key, eq, value = line.partition("=")
        if not eq:
            continue
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


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

    def cmd(self, argv: list[str], as_root: bool = True,
            timeout: float = CMD_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
        """Run a command that changes something (skipped in a dry run). `as_root=False` runs it
        as this user even when setup is using sudo."""
        self._note("run      " + " ".join(argv))
        if self.dry:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return ran(self.run, self._sudo(argv) if as_root else argv, timeout=timeout)

    def query(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        """Run a command that only reads (also in a dry run), as this user."""
        return ran(self.run, argv)

    def _test(self, flag: str, path: str) -> bool:
        return ran(self.run, ["sudo", "-n", "test", flag, path]).returncode == 0

    def exists(self, path: str) -> bool:
        """Something is at `path`, a dangling symlink included."""
        if not self.sudo:
            return os.path.lexists(path)
        return self._test("-e", path) or self._test("-L", path)

    def read(self, path: str) -> str | None:
        """A file's text, or None if it is not there or is a symlink (read through sudo for
        root-only files)."""
        if not self.sudo:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            except OSError:
                return None
            try:
                with os.fdopen(fd) as f:
                    return f.read()
            except (OSError, UnicodeDecodeError):
                return None
        if self._test("-L", path):
            return None
        r = ran(self.run, ["sudo", "-n", "cat", "--", path])
        return r.stdout if r.returncode == 0 else None

    def mkdir(self, path: str, owner: str | None = None) -> None:
        """Create a folder (and its parents); `owner` (a user, with their login group) gets it,
        new or not."""
        exists = os.path.isdir(path)
        if exists and (owner is None or _owned_by(path, owner)):
            return
        self._note(f"create   {path}/" + (f" (owner {owner})" if owner else "") if not exists
                   else f"owner    {path}/ -> {owner}")
        if self.dry:
            return
        if self.sudo:
            self._check(ran(self.run, ["sudo", "-n", "install", "-d", "-m", "755", "--", path]), path)
            if owner:
                self._check(ran(self.run, ["sudo", "-n", "chown", "--", f"{owner}:", path]), path)
            return
        os.makedirs(path, exist_ok=True)
        if owner:
            os.chown(path, *_ids(owner))

    def write(self, path: str | pathlib.Path, text: str, mode: int = PUBLIC, owner: str | None = None) -> None:
        path = str(path)
        self._note(f"write    {path} (mode {mode:o}" + (f", owner {owner})" if owner else ")"))
        if self.dry:
            return
        if self.sudo:
            self._write_sudo(path, text, mode, owner)
            return
        if os.path.islink(path) or os.path.isdir(path):
            raise UnsafePath(f"{path} is a symlink or a folder; setup does not write through it")
        d, base = os.path.split(path)
        tmp = os.path.join(d, f".{base}.{secrets.token_hex(6)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, PRIVATE)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
                f.flush()
                os.fchmod(f.fileno(), mode)
                if owner:
                    uid, gid = _ids(owner)
                    os.fchown(f.fileno(), uid, gid)
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    def _write_sudo(self, path: str, text: str, mode: int, owner: str | None) -> None:
        if self._test("-L", path) or self._test("-d", path):
            raise UnsafePath(f"{path} is a symlink or a folder; setup does not write through it")
        r = ran(self.run, ["sudo", "-n", "mktemp", "--", f"{os.path.dirname(path)}/.{os.path.basename(path)}.XXXXXXXX"])
        self._check(r, path)
        tmp = r.stdout.strip()
        steps = [(["tee", "--", tmp], text), (["chmod", f"{mode:o}", "--", tmp], None),
                 *([(["chown", "--", f"{owner}:", tmp], None)] if owner else []), (["mv", "-fT", "--", tmp, path], None)]
        for argv, stdin in steps:
            r = ran(self.run, ["sudo", "-n", *argv], **({"input": stdin} if stdin is not None else {}))
            if r.returncode != 0:
                ran(self.run, ["sudo", "-n", "rm", "-f", "--", tmp])
                self._check(r, path)

    @staticmethod
    def _check(r: subprocess.CompletedProcess[str], what: str) -> None:
        if r.returncode != 0:
            raise PermissionError(f"{what}: {(r.stderr or r.stdout).strip()}")


def _is_token_line(line: str) -> bool:
    return parse_env(line).keys() == {TOKEN_ENV}


def token(host: Host, env_file: str) -> tuple[str, bool]:
    """The API token: the one already in `env_file` (read as systemd reads it), or a new one
    written there (mode 0600) in place of any empty or blank assignment. Returns (token,
    whether it is new)."""
    old = host.read(env_file) or ""
    if value := parse_env(old).get(TOKEN_ENV):
        return value, False
    new = secrets.token_urlsafe(TOKEN_BYTES)
    keep = [ln for ln in old.splitlines() if not _is_token_line(ln)]
    host.write(env_file, "\n".join([*keep, f"{TOKEN_ENV}={new}"]) + "\n", PRIVATE)
    return new, True


def _ids(owner: str) -> tuple[int, int]:
    """A user's uid and login group."""
    import pwd
    pw = pwd.getpwnam(owner)
    return pw.pw_uid, pw.pw_gid


def _owned_by(path: str, owner: str) -> bool:
    try:
        return (os.stat(path).st_uid, os.stat(path).st_gid) == _ids(owner)
    except (OSError, KeyError):
        return False


UNIT_UNSAFE = "\n\r\0"
QUOTE_IF = "\"'\\;"   # a word with any of these, or whitespace, is double-quoted


def exec_word(arg: str) -> str:
    """One ExecStart word, quoted the way systemd unquotes it: `%` is a specifier (`%%`), `$`
    an environment variable (`$$`), and inside double quotes `\\` and `"` are escaped."""
    if any(c in arg for c in UNIT_UNSAFE):
        raise UnsafePath(f"{arg!r}: a unit file cannot hold a newline")
    esc = arg.replace("%", "%%").replace("$", "$$")
    if not arg or any(c.isspace() or c in QUOTE_IF for c in arg):
        esc = '"' + esc.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return esc


def path_value(path: str) -> str:
    """A path setting such as EnvironmentFile=: the whole value is the path (spaces included),
    with specifiers expanded, so only `%` needs escaping. It must not be quoted: systemd then
    calls it "not absolute" and ignores it, starting the service without its environment."""
    if any(c in path for c in UNIT_UNSAFE) or path != path.strip() or not path.startswith("/"):
        raise UnsafePath(f"{path!r}: not an absolute path a unit file can hold")
    return path.replace("%", "%%")

