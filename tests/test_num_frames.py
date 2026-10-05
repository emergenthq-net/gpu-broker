"""A video's length is `num_frames`; `frames` is the multi-view input slot. The MCP tools and
the HTTP API both send a count as `num_frames`, which templates.build maps to the template's
own `frames` option."""
import pytest

from gpu_broker import inputs, templates
from tests.mcp_fakes import PNG_B64, call, mcp_broker, report

__all__ = ["mcp_broker"]


def lengths(graph):
    return [n["inputs"]["length"] for n in graph.values() if "length" in n["inputs"]]


def test_generate_video_and_image_to_video_take_num_frames(mcp_broker):
    out = report(call(mcp_broker, "generate_video", {"prompt": "waves", "num_frames": 33}))
    assert out["state"] == "done" and lengths(mcp_broker.backends.graphs[-1]) == [33]
    assert mcp_broker.store.job(out["job_id"])["payload"]["num_frames"] == 33
    out = report(call(mcp_broker, "image_to_video", {"image": PNG_B64, "prompt": "turn", "num_frames": 17}))
    assert out["state"] == "done" and set(lengths(mcp_broker.backends.graphs[-1])) == {17}


def test_the_tools_offer_num_frames_not_frames(mcp_broker):
    import anyio
    from mcp import Client

    from gpu_broker.mcp_server import server

    async def go():
        async with Client(server.build(mcp_broker, default_caller="main")) as c:
            return {t.name: t.input_schema["properties"] for t in (await c.list_tools()).tools}
    props = anyio.run(go)
    for tool in ("generate_video", "image_to_video"):
        assert "num_frames" in props[tool] and "frames" not in props[tool], tool


def test_a_count_in_the_frames_slot_says_to_use_num_frames():
    with pytest.raises(ValueError, match="num_frames"):
        inputs.slots({"frames": 81})


def test_build_maps_num_frames_and_never_reads_a_requests_frames():
    model = {"template": "wan5b", "params": {}}
    default = lengths(templates.build(model, {"prompt": "x"}, "p"))
    assert lengths(templates.build(model, {"prompt": "x", "num_frames": 9}, "p")) == [9]
    assert lengths(templates.build(model, {"prompt": "x", "frames": ["view.png"]}, "p")) == default
    assert lengths(templates.build({**model, "defaults": {"frames": 25}}, {"prompt": "x"}, "p")) == [25]   # catalog default
    for bad in (0, -1, 2.5, True, "9"):
        with pytest.raises(ValueError, match="num_frames"):
            templates.build(model, {"prompt": "x", "num_frames": bad}, "p")
