"""Every number the demo simulation uses: timings, the simulated card, and the traffic script.

The timings are in the range a 24 GB consumer card shows with the models in the demo catalog,
so a switch on the dashboard looks like one on real hardware. Tests pass faster values.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..tuning import Intervals
from . import content

DEMO_HOST = "127.0.0.1"   # loopback only: the demo's token is printed to the terminal
DEMO_PORT = 8096          # tried first; if it is taken, any free port (the real broker's default is 8095)
COMFY_PATH = "/comfy/"    # + a per-run secret: where the demo's ComfyUI stand-ins are served
CATALOG_FILE = "catalog.yaml"   # shipped in this package
DATA_PREFIX = "gpu-broker-demo-"   # temporary directory for the run's database, inputs and outputs
TOKEN_BYTES = 18                   # secrets.token_urlsafe: a fresh token per run

# Kinds of job and how long their simulated render takes (s).
RUN_S: Mapping[str, float] = MappingProxyType({"image": 6.0, "video": 8.0, "3d": 8.0})


@dataclass(frozen=True)
class SimTimings:
    llm_stop_s: float = 3.0       # an LLM server stopping and handing its memory back
    llm_load_s: float = 6.0       # an LLM server loading weights until its health check answers
    comfy_load_s: float = 10.0    # ComfyUI loading an image/video model before the first render
    chat_s: tuple[float, float] = (2.0, 5.0)   # one chat completion, shortest to longest
    stream_word_s: float = 0.05   # between streamed words
    run_s: Mapping[str, float] = field(default_factory=lambda: RUN_S)
    recipe_timeout_s: float = 60  # the simulated 3D recipe's own limit (the catalog's exec.timeout_s exceeds it)
    sample_s: float = 1.0         # GPU sample period (the dashboard's live charts)


@dataclass(frozen=True)
class SimCard:
    """A 24 GB card. Load is a share of the card's compute (one chat call ~ a third, a render all of it)."""
    total_mib: int = 24576
    base_mib: int = 420           # CUDA context and desktop
    comfy_idle_mib: int = 640     # ComfyUI running with nothing loaded
    idle_util: int = 1
    busy_util: int = 97
    jitter_util: int = 3
    idle_w: float = 28.0
    busy_w: float = 335.0
    jitter_w: float = 12.0
    idle_c: float = 38.0
    hot_c: float = 71.0
    heat_step: float = 0.12       # each sample, temperature moves this share of the way to its target
    idle_mhz: int = 210
    busy_mhz: int = 2520
    chat_load: float = 0.35
    render_load: float = 1.0
    model_load: float = 0.08      # reading weights: mostly I/O


@dataclass(frozen=True)
class Step:
    """One step of the traffic script: optionally submit a job, then chat, then go quiet."""
    job: str = ""        # catalog model to submit at the start ("" = none)
    prompt: str = ""
    chat_s: float = 0    # simulated people chatting, for this long
    quiet_s: float = 0   # then, once the job is done, nobody for this long (long enough and the chat model comes back)


# One cycle (~3.5 min), repeated. With the default timings and the demo catalog's idle_restore_s:
#   90 s  people chat, one request every few seconds; each is answered straight away
#   then  a video: the chat model is stopped, ComfyUI loads, the next couple of chats queue
#         behind it; once the video is done they run on the chat model, started again for them
#   20 s  more chat, answered straight away again
#   then  an image, then 30 s of quiet once it is done: after the catalog's idle_restore_s with
#         nothing to do, the chat model is loaded again on its own
# Most chats arrive while the chat model is loaded, so a typical one starts in well under a
# second; only the few sent during a video wait for it.
SCENARIO = (
    Step(chat_s=90),
    Step(job=content.VIDEO_MODEL, prompt=content.VIDEO_PROMPT, chat_s=14, quiet_s=2),
    Step(chat_s=20),
    Step(job=content.IMAGE_MODEL, prompt=content.IMAGE_PROMPT, quiet_s=30),
)


@dataclass(frozen=True)
class Traffic:
    chat_gap_s: tuple[float, float] = (5.0, 10.0)  # between two chat requests: slower than one completion, so no backlog
    start_s: float = 2.0                           # before the first request
    max_tokens: int = 256
    job_wait_s: float = 600                        # longest a quiet step waits for its job to finish
    wait_slice_s: float = 0.5                      # ...checking this often whether the demo is stopping
    close_s: float = 5                             # longest close() waits for the script to notice
    steps: tuple[Step, ...] = SCENARIO


# The broker's own polling, shortened so a switch and an idle restore show up promptly.
INTERVALS = Intervals(worker_poll_s=1.0, health_poll_s=0.5, comfy_poll_s=1.0, gpu_sample_s=1.0,
                      gpu_cache_s=1.0, session_poll_s=2.0)
