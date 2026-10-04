"""Fixed protocol values: the vocabulary shared by the API, the event log, the catalog and
the host drivers. Anything a deployer may want to change lives in `settings.py` instead."""
from __future__ import annotations

from enum import StrEnum

APP_NAME = "gpu-broker"
TOKEN_ENV = "BROKER_TOKEN"  # noqa: S105 — the env var NAME holding the API token, never read from a file
UPSTREAM_TOKEN_PREFIX = "UPSTREAM_TOKEN_"  # noqa: S105 — prefix of env vars holding model-server keys (catalog auth_env)
OWNER = APP_NAME                          # `owned_by` in the OpenAI model list
SESSION_KEY = "session"                   # job payload flag: an interactive session, not a queued run
REQUESTER_HEADER = "x-requester"          # optional caller label on /v1/chat/completions
PRIORITY_HEADER = "x-priority"            # interactive | background; overrides defaults.background_requesters
INTERACTIVE_KEY = "interactive"           # job payload flag: a person is waiting (may use reserved slots)
EMBED_KEY = "x_broker_embed"              # job payload flag: an embeddings call, not a chat completion
EMBED_CAP = "embed"                       # catalog cap of a model that serves /v1/embeddings
API_KEY_HEADER = "x-api-key"              # the Anthropic SDK's token header (Bearer is the OpenAI SDK's)
PASSTHROUGH_PATH = "/v1/upstreams/passthrough"   # model scope: which APIs pass a client's own key through
BROKER_KEY_HEADER = "x-gpu-broker-key"    # a broker credential beside a cloud key the client passes through
ANTHROPIC_VERSION_HEADER = "anthropic-version"   # sent by Anthropic clients; selects Anthropic shapes
DEFAULT_TARGET = "@default"               # model_map target meaning the catalog's resident (default) LLM
# model_map when the config sets none: hosted chat names -> the resident LLM. `model_map: {}` turns it off.
DEFAULT_MODEL_MAP = {"gpt-*": DEFAULT_TARGET, "chatgpt-*": DEFAULT_TARGET, "o[0-9]*": DEFAULT_TARGET,
                     "claude-*": DEFAULT_TARGET}
SERVED_BY_HEADER = "x-broker-served-by"   # local | hosted, on every drop-in response
# Request fields the broker consumes itself; never forwarded to a model server.
BROKER_FIELDS = frozenset({"model", "stream", "kind", "caps", "requester", "wait", "wait_s",
                           INTERACTIVE_KEY, SESSION_KEY, EMBED_KEY})
AUTH_SCHEME = "Bearer"

# Input files. Single slots arrive inline (base64 or a data: URL) or as `<slot>_url`;
# `frames` is a list of inline images.
IMAGE_SLOTS = ("image", "end_image")    # start frame / edit source, and an optional end frame
FRAMES, VIDEO = "frames", "video"        # several views of a scene, or one video of it
# A video's length in frames is the request key `num_frames`: `frames` is the input slot above.
# Templates read it as their own `frames` option (templates.build maps it).
NUM_FRAMES, TEMPLATE_FRAMES = "num_frames", "frames"
INPUT_SLOTS = (*IMAGE_SLOTS, FRAMES, VIDEO)
URL_SLOTS = (*IMAGE_SLOTS, VIDEO)        # slots that may come as `<slot>_url`
URL_SUFFIX = "_url"
IMAGE_MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}   # image format -> MIME type
VIDEO_MIME = {"mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm"}
HTTP_SCHEMES = frozenset({"http", "https"})   # the only URL schemes the broker contacts
INPUTS_KEY = "inputs"                    # job payload: what was received per slot (no file data)


class InputNeed(StrEnum):
    """Catalog `inputs: {slot: need}`: whether a model takes a file in that slot."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    ONE_OF = "one_of"     # exactly one of the slots marked one_of must be given


class JobState(StrEnum):
    RECEIVED = "received"
    QUEUED = "queued"
    SWITCHING = "switching"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"


TERMINAL = frozenset({JobState.DONE, JobState.FAILED, JobState.REJECTED})


class DownloadState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


ACTIVE_DOWNLOADS = frozenset({DownloadState.QUEUED, DownloadState.RUNNING, DownloadState.DONE})


class Runner(StrEnum):
    LLM_UNIT = "llm_unit"   # an OpenAI-compatible server the driver starts and stops
    COMFY = "comfy"         # a graph run on the shared ComfyUI
    EXEC = "exec"           # a command-line program, run by a recipe the host defines
    EXTERNAL = "external"   # downloaded, but nothing can run it yet


class ModelStatus(StrEnum):
    READY = "ready"
    DOWNLOADABLE = "downloadable"
    NEEDS_INTEGRATION = "needs_integration"


class Priority(StrEnum):
    INTERACTIVE = "interactive"   # a person waiting in a chat UI: may skip the queue
    BACKGROUND = "background"     # agents and batch work: queued, kept off reserved slots


class Kind(StrEnum):
    LLM = "llm"
    IMAGE = "image"
    VIDEO = "video"
    THREE_D = "3d"
    UI = "ui"       # a front end used through sessions, never a job
    UNKNOWN = "unknown"     # kind of a registered repo when the request named none


class DownloadKind(StrEnum):
    HF = "hf"
    GH = "gh"


class Verb(StrEnum):
    START = "start"
    STOP = "stop"
    IS_ACTIVE = "is-active"


class Event(StrEnum):
    """Event log kinds. `job.<state>` and `download.<state>` are derived from the enums above."""
    BROKER_STARTED = "broker.started"
    ORPHANS_FAILED = "broker.orphans_failed"
    QUIESCE = "broker.quiesce"
    RESUME = "broker.resume"
    WORKER_ERROR = "worker.error"
    JOB_SUBSTITUTED = "job.substituted"
    JOB_REQUEUED = "job.requeued"   # queued when the previous process stopped; queued again at start
    JOB_DIRECT = "job.direct"   # an interactive chat served by the resident LLM without queueing
    DOWNLOAD_QUEUED = "download.queued"
    RES_STOP = "residency.stop"
    RES_START = "residency.start"
    RES_READY = "residency.ready"
    RES_RESIDENT = "residency.resident"
    RES_LOST = "residency.lost"
    RES_COMFY_DOWN = "residency.comfy_down"
    RES_COMFY_STARTED = "residency.comfy_started"
    RES_DETECT_FAILED = "residency.detect_failed"
    RES_IDLE_RESTORE = "residency.idle_restore"
    RES_RESTORE_FAILED = "residency.restore_failed"
    EXEC_UNCHECKED = "exec.recipe_unchecked"   # its recipe could not be read at startup (checked per job)
    EXEC_GPU_HELD = "exec.gpu_held"            # a recipe may still run: no job runs until the hold clears
    GPU_HELD_CLEARED = "exec.gpu_held_cleared"  # by a clean that confirmed the job gone, or an operator
    UPSTREAM_FAILOVER = "upstream.failover"   # a cloud request answered further down its chain
    UPSTREAM_OPEN = "upstream.open"           # a provider's breaker opened: requests skip it
    UPSTREAM_QUOTA = "upstream.quota"         # a credential ran out of quota or credit
    UPSTREAM_CLOSED = "upstream.closed"       # the provider answers again: back to the cloud


RESIDENCY_EVENT_PREFIX = "residency."
JOB_EVENT_PREFIX = "job."
DOWNLOAD_EVENT_PREFIX = "download."
STATS_EXTRA_EVENTS = (Event.WORKER_ERROR, Event.JOB_SUBSTITUTED)

# How much of an error message is kept, by where it is stored.
ERR_EVENT = 500     # event log rows
ERR_JOB = 2000      # job.error, returned to the requester
ERR_DETAIL = 800   # download errors, backend error bodies
ERR_SHORT = 300     # driver/stream errors shown on the dashboard
