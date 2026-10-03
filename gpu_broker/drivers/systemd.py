"""The systemd driver: model servers are systemd units on the machine the broker runs on."""
from __future__ import annotations

from typing import Any

from ..constants import Verb
from .local import ACTIVE, SYSTEMD_MODELS_ROOT, LocalDriver


class SystemdDriver(LocalDriver):
    """Units started with systemctl. Needs root, `sudo: true` with a sudoers rule limited to
    `systemctl start|stop <unit>`, or `user: true` for user units."""

    def __init__(self, sudo: bool = False, user: bool = False, models_root: str = SYSTEMD_MODELS_ROOT,
                 **kw: Any) -> None:
        super().__init__(models_root=models_root, **kw)
        self.base = [*(["sudo", "-n"] if sudo else []), "systemctl", *(["--user"] if user else [])]

    def unit(self, spec: Any, verb: Verb) -> bool:
        r = self.run([*self.base, verb, "--", self._local(spec, verb)], timeout=self.t.unit_s)
        return r.stdout.strip() == ACTIVE if verb == Verb.IS_ACTIVE else r.returncode == 0
