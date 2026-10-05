"""The broker's MCP tools (MCP Python SDK, the `gpu-broker[mcp]` extra), wrapping core.py.

Generation tools submit a job and wait up to `mcp.wait_s` (or the caller's shorter `wait_s`)
for it; a job still running then comes back with its id and a hint to call `job_status`, so a
20-minute video never holds a tool call open. A finished image small enough
(`mcp.inline_max_bytes`) comes back inline as image content as well as by URL. Image and video
inputs are an http(s) URL or base64 (a data: URL is fine), checked by the inputs layer.

The caller is read from the header http.py sets after checking the credential; build() without
`default_caller` refuses a call that lacks it.
"""
from __future__ import annotations

import base64
import json
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import Field

from ..broker import Broker
from ..constants import APP_NAME, FRAMES, IMAGE_SLOTS, NUM_FRAMES, VIDEO, Kind
from ..inputs import slots
from ..quiesce import Quiesced
from . import core

CALLER_HEADER = "x-gpu-broker-caller"   # set by http.py, never taken from the client
INSTRUCTIONS = ("Runs models on a local GPU shared through gpu-broker: images, video, image edits, 3D splats and a local LLM. "
                "Call list_models to see what is ready. Generation can take minutes: a tool returns a job_id and a hint when "
                "the job is still running; then call job_status or job_result with that id.")
Content = list[TextContent | ImageContent]
READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)

Prompt = Annotated[str, Field(description="What to generate, in plain words")]
Model = Annotated[str | None, Field(description="A model name from list_models; omitted = the best ready model for the job")]
Media = Annotated[str, Field(description="An http(s) URL, or the file as base64 (a data: URL is fine)")]
Wait = Annotated[float | None, Field(ge=0, description="Seconds to wait for the result (capped by the broker; 0 = return the job id now)")]
Size = Annotated[int | None, Field(gt=0, le=8192, description="Pixels; omitted = the model's default")]
Seed = Annotated[int | None, Field(ge=0, description="Fixed seed for a repeatable result")]
Frames = Annotated[int | None, Field(gt=0, le=2000, description="Video length in frames; omitted = the model's default")]


def content(report: dict[str, Any], images: Sequence[tuple[str, bytes]] = ()) -> Content:
    out: Content = [TextContent(type="text", text=json.dumps(report, indent=1))]
    return out + [ImageContent(type="image", data=base64.b64encode(b).decode(), mime_type=mime) for mime, b in images]


def package_version() -> str:
    """The installed gpu-broker version, for serverInfo ("" when run from a tree without metadata)."""
    try:
        return version(APP_NAME)
    except PackageNotFoundError:
        return ""


def build(broker: Broker, default_caller: str | None = None) -> MCPServer:
    cfg = broker.settings.mcp
    mcp: MCPServer = MCPServer(APP_NAME, version=package_version(), instructions=INSTRUCTIONS)

    def who(ctx: Context) -> core.Caller:
        headers = ctx.headers or {}
        try:
            return core.caller(headers.get(CALLER_HEADER, default_caller))
        except PermissionError as e:
            raise ToolError(str(e)) from None

    async def run(fn: Callable[..., Any], *args: Any) -> Any:
        """`fn` off the event loop (the broker blocks); its expected failures reach the caller as text."""
        try:
            return await anyio.to_thread.run_sync(fn, *args)
        except Quiesced as e:
            raise ToolError(f"{e}; try again in {e.retry_after_s} s") from None
        except ValueError as e:   # a bad request or input, an unknown model, a hidden job
            raise ToolError(str(e)) from None

    def budget(wait_s: float | None) -> float:
        return cfg.wait_s if wait_s is None else min(wait_s, cfg.wait_s)

    async def finish(caller: core.Caller, jid: str, wait_s: float | None) -> Content:
        report = await run(core.report, broker, caller, jid, budget(wait_s))
        return content(report, await run(core.inline, broker, report, cfg) if report.get("outputs") else [])

    async def generate(ctx: Context, kind: Kind, caps: list[str], model: str | None, wait_s: float | None,
                       **request: Any) -> Content:
        caller = who(ctx)
        body = {k: v for k, v in request.items() if v is not None}
        key = await run(core.pick, broker, kind, caps, await run(slots, body), model)
        jid = await run(core.submit, broker, caller, {**body, "model": key, "kind": kind, "caps": caps})
        return await finish(caller, jid, wait_s)

    @mcp.tool(annotations=READ)
    async def list_models(ctx: Context) -> str:
        """The models this broker can run now, with kind (llm, image, video, 3d), caps and the inputs each takes."""
        who(ctx)
        return json.dumps(await run(core.models, broker), indent=1)

    @mcp.tool(annotations=WRITE, structured_output=False)
    async def generate_image(ctx: Context, prompt: Prompt, model: Model = None, width: Size = None, height: Size = None,
                             seed: Seed = None, negative: str | None = None, wait_s: Wait = None) -> Content:
        """Generate an image from a text prompt on the local GPU. Returns the image (inline when small) and its URL."""
        return await generate(ctx, Kind.IMAGE, ["t2i"], model, wait_s, prompt=prompt, width=width, height=height,
                              seed=seed, negative=negative)

    @mcp.tool(annotations=WRITE, structured_output=False)
    async def generate_video(ctx: Context, prompt: Prompt, model: Model = None, width: Size = None, height: Size = None,
                             num_frames: Frames = None, seed: Seed = None, wait_s: Wait = None) -> Content:
        """Generate a video from a text prompt. Usually takes minutes: expect a job_id and poll job_status."""
        return await generate(ctx, Kind.VIDEO, ["t2v"], model, wait_s, prompt=prompt, width=width, height=height,
                              seed=seed, **{NUM_FRAMES: num_frames})

    @mcp.tool(annotations=WRITE, structured_output=False)
    async def edit_image(ctx: Context, image: Media, prompt: Annotated[str, Field(description="The change to make")],
                         model: Model = None, seed: Seed = None, wait_s: Wait = None) -> Content:
        """Edit an image as the prompt says (restyle, replace, add or remove things)."""
        return await generate(ctx, Kind.IMAGE, ["edit"], model, wait_s, prompt=prompt, seed=seed, **core.input_field("image", image))

    @mcp.tool(annotations=WRITE, structured_output=False)
    async def image_to_video(ctx: Context, image: Annotated[str, Field(description="The first frame: an http(s) URL or base64")],
                             prompt: Prompt, end_image: Annotated[str | None, Field(description="Optional last frame")] = None,
                             model: Model = None, num_frames: Frames = None, seed: Seed = None, wait_s: Wait = None) -> Content:
        """Animate an image into a video. Usually takes minutes: expect a job_id and poll job_status."""
        extra = core.input_field(IMAGE_SLOTS[1], end_image) if end_image else {}
        return await generate(ctx, Kind.VIDEO, ["i2v"], model, wait_s, prompt=prompt, seed=seed,
                              **{NUM_FRAMES: num_frames}, **core.input_field("image", image), **extra)

    @mcp.tool(annotations=WRITE, structured_output=False)
    async def make_3d(ctx: Context, image: Annotated[str | None, Field(description="One image (URL or base64)")] = None,
                      images: Annotated[list[str] | None, Field(description="Several views of one scene, as base64")] = None,
                      video: Annotated[str | None, Field(description="A video walking around the scene (URL or base64)")] = None,
                      model: Model = None, wait_s: Wait = None) -> Content:
        """Make a 3D Gaussian splat (.ply) from one image, several views, or a video. Give exactly one of them."""
        given = [n for n, v in (("image", image), ("images", images), ("video", video)) if v]
        if len(given) != 1:
            raise ToolError("give exactly one of image, images or video")
        extra: dict[str, Any]
        if image:
            cap, extra = "image_to_splat", core.input_field("image", image)
        elif images:
            cap, extra = "images_to_splat", {FRAMES: images}
        else:
            cap, extra = "video_to_splat", core.input_field(VIDEO, video or "")
        return await generate(ctx, Kind.THREE_D, [cap], model, wait_s, **extra)

    @mcp.tool(annotations=READ, structured_output=False)
    async def job_status(ctx: Context, job_id: str) -> Content:
        """A job's state, queue position and, once done, its output URLs. Returns at once."""
        return content(await run(core.report, broker, who(ctx), job_id, 0))

    @mcp.tool(annotations=READ, structured_output=False)
    async def job_result(ctx: Context, job_id: str, wait_s: Wait = None) -> Content:
        """Wait (up to wait_s) for a job, then return its outputs: URLs, and small images inline."""
        return await finish(who(ctx), job_id, wait_s)

    @mcp.tool(annotations=WRITE)
    async def chat_local(ctx: Context, prompt: Annotated[str, Field(description="The message to the local model")],
                         system: str | None = None, model: Model = None,
                         max_tokens: Annotated[int | None, Field(gt=0, le=32768)] = None) -> str:
        """Ask the local LLM (the one loaded on the GPU unless model names another) and return its answer."""
        caller = who(ctx)
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {"messages": messages, **({"model": model} if model else {}),
                                **({"max_tokens": max_tokens} if max_tokens else {})}
        return json.dumps(await run(core.chat, broker, caller, body), indent=1)

    @mcp.tool(annotations=READ)
    async def gpu_status(ctx: Context) -> str:
        """What is loaded on the GPU, what is running and queued, and VRAM in use."""
        return json.dumps(await run(core.status, broker, who(ctx)), indent=1)

    return mcp
