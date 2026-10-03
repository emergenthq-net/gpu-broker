"""The Docker driver: model servers are containers on the machine the broker runs on.

It cannot run exec recipes: a recipe would run in the broker's own container.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..constants import Verb
from . import Input, RecipeInfo
from .local import DOCKER, DOCKER_MODELS_ROOT, LocalDriver

NO_EXEC = "the docker driver cannot run exec recipes (use the systemd or proxmox driver)"
RUNNING = "true"   # `docker inspect .State.Running` output


class DockerDriver(LocalDriver):
    """Containers started and stopped by name. They must already exist (`docker compose
    create`); the broker needs the Docker socket."""

    def __init__(self, docker: str = DOCKER, models_root: str = DOCKER_MODELS_ROOT, **kw: Any) -> None:
        super().__init__(models_root=models_root, **kw)
        self.docker = docker

    def unit(self, spec: Any, verb: Verb) -> bool:
        name = self._local(spec, verb)
        if verb == Verb.IS_ACTIVE:
            r = self.run([self.docker, "inspect", "-f", "{{.State.Running}}", "--", name], timeout=self.t.unit_s)
            return r.returncode == 0 and r.stdout.strip() == RUNNING
        argv = ([self.docker, "start", "--", name] if verb == Verb.START else
                [self.docker, "stop", "-t", str(self.t.container_stop_s), "--", name])
        return self.run(argv, timeout=self.t.unit_s).returncode == 0

    def recipe_info(self, recipe: str) -> RecipeInfo:
        raise RuntimeError(NO_EXEC)

    def clean_recipe(self, recipe: str, jid: str) -> None:
        raise RuntimeError(NO_EXEC)

    def run_recipe(self, recipe: str, jid: str, files: Sequence[tuple[str, Input]], timeout_s: float) -> list[str]:
        raise RuntimeError(NO_EXEC)
