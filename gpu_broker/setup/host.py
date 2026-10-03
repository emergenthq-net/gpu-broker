"""Changing the machine for `gpu-broker setup`: files, the token, commands, unit-file quoting.

`Host` does every write and command, as this user, as root, or through `sudo -n` when setup
installs a system service without being root. In a dry run it only says what it would do.
Commands are argv lists, never shell strings. A command that cannot run (missing, timed out)
comes back as a failed result, so callers report it like any other failure.

Files are written to a new temporary file in the same folder (created 0600, never following a
symlink) and renamed into place, so a reader never sees half a file and a planted symlink is
never written through: setup refuses to replace a symlink at all.

When setup has root (`strict`), the service account owns some of the folders it touches (the
data folder and what is inside it), and could swap anything there for a symlink at any moment.
So nothing root does may follow a link the account could have planted:
- as root, a path is opened one folder at a time with O_NOFOLLOW from the deepest folder only
  root can change, and owners and modes are set on the open descriptor, never on a name;
- through sudo, where a descriptor cannot be held across commands, folders and files inside a
  folder the account controls are made *as the account* (`sudo -u`), so root never acts there,
  and root's own `chown` never follows a link (`-h`).
"""
from __future__ import annotations

import contextlib
import errno
import os
import pathlib
import secrets
import shutil
import stat
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
DIR_MODE = 0o755
GROUP_OTHER_WRITE = 0o022
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
NO_LINK = (errno.ELOOP, errno.ENOTDIR)   # what O_NOFOLLOW (with O_DIRECTORY) gives for a symlink

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
    strict: bool = False            # setup has root: never follow a symlink the service account could plant
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
        new or not. A symlink in its place is refused."""
        if os.path.islink(path):
            raise UnsafePath(f"{path} is a symlink; setup does not create or change a folder through it")
        exists = os.path.isdir(path)
        if exists and (owner is None or _owned_by(path, owner)):
            return
        self._note(f"create   {path}/" + (f" (owner {owner})" if owner else "") if not exists
                   else f"owner    {path}/ -> {owner}")
        if self.dry:
            return
        if self.sudo:
            self._mkdir_sudo(path, owner)
        elif self.strict:
            fd, _ = open_dir(path, create=True)
            try:
                if owner and (os.fstat(fd).st_uid, os.fstat(fd).st_gid) != _ids(owner):
                    os.fchown(fd, *_ids(owner))
            finally:
                os.close(fd)
        else:
            os.makedirs(path, exist_ok=True)
            if owner:
                os.chown(path, *_ids(owner))

    def _mkdir_sudo(self, path: str, owner: str | None) -> None:
        if self._test("-L", path):
            raise UnsafePath(f"{path} is a symlink; setup does not create or change a folder through it")
        if owner and controls(path, owner):
            self._check(ran(self.run, ["sudo", "-n", "-u", owner, "mkdir", "-p", "-m", f"{DIR_MODE:o}", "--", path]), path)
            return
        self._check(ran(self.run, ["sudo", "-n", "install", "-d", "-m", f"{DIR_MODE:o}", "--", path]), path)
        if owner:
            self._check(ran(self.run, ["sudo", "-n", "chown", "-h", "--", f"{owner}:", path]), path)

    def write(self, path: str | pathlib.Path, text: str, mode: int = PUBLIC, owner: str | None = None) -> None:
        path = str(path)
        self._note(f"write    {path} (mode {mode:o}" + (f", owner {owner})" if owner else ")"))
        if self.dry:
            return
        if self.sudo:
            self._write_sudo(path, text, mode, owner)
            return
        d, base = os.path.split(path)
        dfd = open_dir(d)[0] if self.strict else os.open(d, DIR_FLAGS)
        try:
            self._write_at(dfd, base, path, text, mode, owner)
        finally:
            os.close(dfd)

    @staticmethod
    def _write_at(dfd: int, base: str, path: str, text: str, mode: int, owner: str | None) -> None:
        """Write `base` in the open folder `dfd`: every step names the folder's descriptor, so a
        symlink planted on the way to it cannot redirect the write."""
        with contextlib.suppress(FileNotFoundError):
            st = os.stat(base, dir_fd=dfd, follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode) or stat.S_ISDIR(st.st_mode):
                raise UnsafePath(f"{path} is a symlink or a folder; setup does not write through it")
        tmp = f".{base}.{secrets.token_hex(6)}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, PRIVATE, dir_fd=dfd)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
                f.flush()
                os.fchmod(f.fileno(), mode)
                if owner:
                    uid, gid = _ids(owner)
                    os.fchown(f.fileno(), uid, gid)
                os.fsync(f.fileno())
            os.replace(tmp, base, src_dir_fd=dfd, dst_dir_fd=dfd)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp, dir_fd=dfd)
            raise

    def _write_sudo(self, path: str, text: str, mode: int, owner: str | None) -> None:
        """mktemp, tee, chmod, (chown,) mv: separate commands on one name. In a folder the
        account controls, all of them run as the account, so swapping the temporary file for a
        link between two steps gains it nothing; elsewhere only root can touch the name."""
        if self._test("-L", path) or self._test("-d", path):
            raise UnsafePath(f"{path} is a symlink or a folder; setup does not write through it")
        as_owner = bool(owner) and controls(path, owner or "")
        who = ["sudo", "-n", "-u", owner or ""] if as_owner else ["sudo", "-n"]
        r = ran(self.run, [*who, "mktemp", "--", f"{os.path.dirname(path)}/.{os.path.basename(path)}.XXXXXXXX"])
        self._check(r, path)
        tmp = r.stdout.strip()
        chown = [(["chown", "-h", "--", f"{owner}:", tmp], None)] if owner and not as_owner else []
        steps = [(["tee", "--", tmp], text), (["chmod", f"{mode:o}", "--", tmp], None), *chown,
                 (["mv", "-fT", "--", tmp, path], None)]
        for argv, stdin in steps:
            r = ran(self.run, [*who, *argv], **({"input": stdin} if stdin is not None else {}))
            if r.returncode != 0:
                ran(self.run, [*who, "rm", "-f", "--", tmp])
                self._check(r, path)

    def check_root_dir(self, path: str, root_uid: int = 0) -> str | None:
        """Why an existing folder setup writes root's files into (the config folder) is not safe:
        a symlink, not root's, or writable by group or others. None if it is fine or absent."""
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            return f"{path} is a symlink or not a folder"
        if st.st_uid != root_uid:
            return f"{path} is not owned by root"
        if st.st_mode & GROUP_OTHER_WRITE:
            return f"{path} is writable by its group or others"
        return None

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
        st = os.lstat(path)
    except OSError:
        return False
    try:
        return (st.st_uid, st.st_gid) == _ids(owner)
    except KeyError:
        return False


def controls(path: str, owner: str) -> bool:
    """`owner` (not root) owns the nearest folder above `path` that exists, or what a symlink there
    points at: it could swap anything below it, so root must not act on names there."""
    d = os.path.dirname(os.path.abspath(path))
    while not os.path.lexists(d):
        d = os.path.dirname(d)
    try:
        uid = _ids(owner)[0]
    except KeyError:
        return False
    if uid == 0:
        return False
    with contextlib.suppress(OSError):
        return uid in (os.lstat(d).st_uid, os.stat(d).st_uid)
    return False


def _root_only(st: os.stat_result, root_uid: int) -> bool:
    return st.st_uid == root_uid and not st.st_mode & GROUP_OTHER_WRITE


def anchor(path: str, root_uid: int = 0) -> tuple[str, tuple[str, ...]]:
    """Split an absolute path into its deepest existing folder that only root can change (every
    folder on its real path is root's and not group/other-writable) and the names after it.
    Following links within that prefix is safe (a root-owned /var -> /private/var, say): nobody
    else can change it. Everything after it is opened without following a link."""
    parts = pathlib.PurePath(os.path.abspath(path)).parts
    base = parts[0]
    for i, name in enumerate(parts[1:], 1):
        nxt = os.path.realpath(os.path.join(base, name))
        real = pathlib.PurePath(nxt).parts
        try:
            ok = os.path.isdir(nxt) and all(
                _root_only(os.lstat(os.path.join(*real[:j])), root_uid) for j in range(1, len(real) + 1))
        except OSError:
            ok = False
        if not ok:
            return base, parts[i:]
        base = nxt
    return base, ()


def open_dir(path: str, create: bool = False, root_uid: int = 0) -> tuple[int, bool]:
    """A descriptor for folder `path`, reached one name at a time past `anchor` with O_NOFOLLOW
    (missing folders made 0755 if `create`), and whether the last one was new. A symlink on the
    way is refused."""
    base, rest = anchor(path, root_uid)
    fd, new = os.open(base, DIR_FLAGS), False
    try:
        for name in rest:
            new = False
            if create:
                try:
                    os.mkdir(name, DIR_MODE, dir_fd=fd)
                    new = True
                except FileExistsError:
                    pass
            try:
                nfd = os.open(name, DIR_FLAGS | os.O_NOFOLLOW, dir_fd=fd)
            except OSError as e:
                if e.errno in NO_LINK:
                    raise UnsafePath(f"{path}: {name} is a symlink or not a folder; setup does not "
                                     "follow it") from e
                raise
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd, new


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

