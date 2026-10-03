"""Opening a link in this machine's browser, for `gpu-broker demo` and `gpu-broker setup`.

A browser can only be opened on this machine's own screen: not over SSH, and on Linux and the
BSDs only with a display server (without one, webbrowser may take over the terminal). Callers
print the link either way.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping

SSH_ENV = ("SSH_CONNECTION", "SSH_TTY")
DISPLAY_ENV = ("DISPLAY", "WAYLAND_DISPLAY")
NEEDS_DISPLAY = ("linux", "freebsd", "openbsd", "netbsd")


def can_open_browser(env: Mapping[str, str], platform: str) -> bool:
    if any(env.get(k) for k in SSH_ENV):
        return False
    return not platform.startswith(NEEDS_DISPLAY) or any(env.get(k) for k in DISPLAY_ENV)


def open_browser(url: str, env: Mapping[str, str], platform: str, opener: Callable[[str], object]) -> None:
    """Open `url` where there is a screen to open it on; otherwise (or if that fails) do nothing."""
    if not can_open_browser(env, platform):
        return
    with contextlib.suppress(Exception):   # a missing or broken browser must not stop the caller
        opener(url)
