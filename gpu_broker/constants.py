"""Fixed protocol values: the vocabulary shared by the API, the event log, the catalog and
the host drivers. Anything a deployer may want to change lives in `settings.py` instead."""
from __future__ import annotations

from enum import StrEnum

APP_NAME = "gpu-broker"
TOKEN_ENV = "BROKER_TOKEN"  # noqa: S105 — the env var NAME holding the API token, never read from a file
UPSTREAM_TOKEN_PREFIX = "UPSTREAM_TOKEN_"  # noqa: S105 — prefix of env vars holding model-server keys (catalog auth_env)
OWNER = APP_NAME                          # `owned_by` in the OpenAI model list
SESSION_KEY = "session"                   # job payload flag: an interactive session, not a queued run
REQUESTER_HEADER = "x-requester"          # optional caller label on inference requests
PRIORITY_HEADER = "x-priority"            # interactive | normal | background; overrides requester defaults
INTERACTIVE_KEY = "interactive"           # job payload flag: a person is waiting (may use reserved slots)
PRIORITY_KEY = "priority"                  # interactive | normal | background; queued scheduler policy
OPENAI_PATH_KEY = "_openai_path"           # internal route chosen by the compatibility API; never forwarded
# Request fields the broker consumes itself; never forwarded to a model server.
BROKER_FIELDS = frozenset({"model", "stream", "kind", "caps", "requester", "wait", "wait_s",
                           INTERACTIVE_KEY, PRIORITY_KEY, OPENAI_PATH_KEY, SESSION_KEY})
AUTH_SCHEME = "Bearer"
CHAT_PATH = "/v1/chat/completions"
OPENAI_JSON_PATHS = frozenset({CHAT_PATH, "/v1/completions", "/v1/responses", "/v1/embeddings", "/v1/rerank", "/v1/score"})


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


class ResidencyMode(StrEnum):
    UNIT = "unit"                 # start/stop the model server process/container
    VLLM_SLEEP = "vllm_sleep"     # keep vLLM server alive; use its sleep/wake dev endpoints
    OLLAMA = "ollama"             # keep shared Ollama daemon alive; load/unload this catalog model


class Runner(StrEnum):
    LLM_UNIT = "llm_unit"   # an OpenAI-compatible server the driver starts and stops
    COMFY = "comfy"         # a graph run on the shared ComfyUI
    EXTERNAL = "external"   # downloaded, but nothing can run it yet


class ModelStatus(StrEnum):
    READY = "ready"
    DOWNLOADABLE = "downloadable"
    NEEDS_INTEGRATION = "needs_integration"


class Priority(StrEnum):
    INTERACTIVE = "interactive"   # a person waiting in a UI: may skip the queue when already resident
    NORMAL = "normal"             # ordinary queued work
    BACKGROUND = "background"     # agents and batch work: queued, kept off reserved slots


class Kind(StrEnum):
    LLM = "llm"
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


RESIDENCY_EVENT_PREFIX = "residency."
JOB_EVENT_PREFIX = "job."
DOWNLOAD_EVENT_PREFIX = "download."
STATS_EXTRA_EVENTS = (Event.WORKER_ERROR, Event.JOB_SUBSTITUTED)

# How much of an error message is kept, by where it is stored.
ERR_EVENT = 500     # event log rows
ERR_JOB = 2000      # job.error, returned to the requester
ERR_DETAIL = 800   # download errors, backend error bodies
ERR_SHORT = 300     # driver/stream errors shown on the dashboard
