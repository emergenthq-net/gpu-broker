"""What the simulated team says and asks for. Plain, safe-for-work text only: it shows up on
the dashboard and in the placeholder images."""
from __future__ import annotations

REQUESTERS = ("open-webui:maya", "open-webui:sam", "docs-agent", "support-bot")
JOB_REQUESTER = "design-team"

CHAT_PROMPTS = (
    "Summarise this pull request in three bullet points.",
    "Write a SQL query that lists customers with no orders this year.",
    "What does HTTP status 409 mean, and when should an API return it?",
    "Turn these meeting notes into action items with owners.",
    "Explain the difference between a process and a thread.",
    "Draft a friendly reply declining a meeting invitation.",
    "Suggest five names for an internal tool that schedules GPU jobs.",
    "Rewrite this error message so a non-engineer understands it.",
)

REPLIES = (
    "Here is a short summary: the change adds a queue, moves the retry logic into one place, "
    "and adds tests for the timeout path.",
    "Sure. A 409 Conflict means the request clashes with the current state of the resource, "
    "for example two people editing the same record.",
    "Action items: Priya to update the roadmap by Friday; Tom to book the venue; Lee to send the survey.",
    "A process has its own memory; threads inside one process share it, which makes them cheaper "
    "to start but easier to get wrong.",
)

IMAGE_PROMPT = "a watercolor lighthouse on a rocky coast at sunrise"
VIDEO_PROMPT = "a paper boat drifting down a rainy city street, slow camera pan"

CHAT_MODEL = "qwen3-8b"
IMAGE_MODEL = "flux.2-klein-4b"
VIDEO_MODEL = "wan2.2-5b"

PLACEHOLDER_TITLE = "GPU-BROKER DEMO"
PLACEHOLDER_NOTE = "SIMULATED OUTPUT - NO MODEL RAN"

GPU_LABEL = "simulated 24 GB card"
RESIDENT_LABEL = "the chat model"
# The dashboard's VRAM-by-owner legend: driver group (unit or recipe name) -> label and colour.
UI_GROUPS = {
    "llama-server-qwen3": {"label": "Qwen3 8B (chat)", "color": "#3b5bdb"},
    "llama-server-llama31": {"label": "Llama 3.1 8B (chat)", "color": "#1098ad"},
    "comfyui": {"label": "ComfyUI (image / video)", "color": "#e8590c"},
    "trellis": {"label": "TRELLIS (3D)", "color": "#ae3ec9"},
}

BANNER = """
gpu-broker demo: a simulated GPU. No graphics card, no model servers, nothing downloaded.

  Dashboard:  {url}/dash#token={token}
  API token:  {token}
              (made up for this run; the demo keeps its data in a temporary folder and
               deletes it when you stop)
  Traffic:    {traffic}

Try the API:  curl -H "Authorization: Bearer {token}" {url}/v1/status
Press Ctrl+C to stop.
"""
TRAFFIC_ON = "a few simulated people chat, and now and then ask for an image or a video"
TRAFFIC_OFF = "none (--quiet): the card stays idle until you send something"
