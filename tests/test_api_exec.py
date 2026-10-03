"""Exec jobs (command-line models such as image -> 3D splat) through the HTTP API."""
import base64

from gpu_broker.constants import JobState
from tests.helpers import COMFY_OUTPUT, done, leftover_staged

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
B64, VIDEO = base64.b64encode(PNG).decode(), base64.b64encode(MP4).decode()


def job(c, **body):
    return c.post("/v1/jobs", json=body)


def test_exec_job_frees_the_gpu_runs_the_recipe_and_returns_view_urls(client, broker):
    driver, backends = broker.driver, broker.backends
    frees = backends.frees
    r = job(client, model="sharp", image=B64)
    assert r.status_code == 200 and r.json()["resolved"] == "sharp"
    jid = r.json()["id"]
    j = done(broker, jid)
    assert j["state"] == JobState.DONE, j
    assert ("stop", "llama-8b") in driver.calls and backends.frees > frees      # LLM evicted, ComfyUI freed
    assert ("start", "comfyui") not in driver.calls                             # ComfyUI not needed
    (recipe, rjid, files, timeout), = driver.recipes
    assert (recipe, rjid, files, timeout) == ("sharp", jid, [("image.png", PNG)], 660.0)
    out = j["result"]["outputs"][0]
    assert out["path"] == f"{COMFY_OUTPUT}/broker/{jid}/scene.ply" and out["file"] == f"broker/{jid}/scene.ply"
    assert out["url"].endswith(f"/view?filename=scene.ply&subfolder=broker%2F{jid}&type=output")
    assert broker.residency.last_comfy is None and backends.uploads == []
    assert leftover_staged(broker) == []


def test_views_model_takes_frames_or_a_video_and_its_params(client, broker):
    jid = job(client, model="splat-views", frames=[B64, B64, B64], frame_stride=2).json()["id"]
    assert done(broker, jid)["state"] == JobState.DONE
    names = [n for n, _ in broker.driver.recipes[-1][2]]
    assert names == ["frames-00.png", "frames-01.png", "frames-02.png", "params.json"]
    assert broker.driver.recipes[-1][2][-1][1] == b'{"frame_stride": 2}'
    frame = {"bytes": len(PNG), "type": "png", "source": "inline"}
    assert broker.store.job(jid)["payload"]["inputs"] == {"frames": [frame, frame, frame]}
    jid = job(client, model="splat-views", video=VIDEO).json()["id"]
    assert done(broker, jid)["state"] == JobState.DONE
    assert [n for n, _ in broker.driver.recipes[-1][2]] == ["video.mp4", "params.json"]
    assert broker.store.job(jid)["payload"]["inputs"] == {"video": {"bytes": len(MP4), "type": "mp4", "source": "inline"}}


def test_wrong_inputs_or_params_are_a_400_before_anything_is_queued(client, broker):
    before = len(broker.store.jobs(100))
    for body, msg in (({"model": "sharp"}, "needs an input: send image"),
                      ({"model": "splat-views"}, "needs exactly one of frames, video"),
                      ({"model": "splat-views", "frames": [B64], "video": VIDEO}, "needs exactly one of"),
                      ({"model": "splat-views", "video": B64}, "not an accepted file"),
                      ({"model": "splat-views", "frames": [B64]}, "'splat-views' takes 2 to 8 frames, got 1"),
                      ({"model": "splat-views", "frames": [B64] * 9}, "takes 2 to 8 frames, got 9"),
                      ({"model": "splat-views", "frames": [B64] * 2, "frame_stride": [2]}, "`frame_stride` must be"),
                      ({"model": "splat-views", "frames": [B64] * 2, "quality": "max"}, "`quality` must be one of ['low', 'high']")):
        r = job(client, **body)
        assert r.status_code == 400 and msg in r.json()["detail"], (body, r.json())
    assert len(broker.store.jobs(100)) == before and broker.driver.recipes == []


def test_a_failing_recipe_fails_the_job_and_drops_its_inputs(client, broker):
    broker.driver.recipe_error = "recipe sharp exited 1: CUDA out of memory"
    j = done(broker, job(client, model="sharp", image=B64).json()["id"])
    assert j["state"] == JobState.FAILED and "CUDA out of memory" in j["error"]
    assert leftover_staged(broker) == []


def test_substitutes_for_a_3d_job_take_its_inputs(client, broker):
    r = job(client, model="some/new-splatter", kind="3d", image=B64)
    assert r.json()["resolved"] == "sharp"
    done(broker, r.json()["id"])
    r = job(client, model="some/other-splatter", kind="3d", video=VIDEO)
    assert r.json()["resolved"] == "splat-views"
    done(broker, r.json()["id"])



def test_an_exec_job_missing_a_staged_frame_fails_before_the_recipe_runs(client, broker, monkeypatch):
    real = broker.staging.received
    monkeypatch.setattr(broker.staging, "received", lambda jid: {**real(jid), "frames": 2})   # one of 3 vanished
    j = done(broker, job(client, model="splat-views", frames=[B64, B64, B64]).json()["id"])
    assert j["state"] == JobState.FAILED and "input files missing" in j["error"]
    assert broker.driver.recipes == []


def test_the_recipe_is_recorded_in_its_own_column_and_never_from_the_caller(client, broker):
    jid = client.post("/v1/jobs", json={"model": "llama-8b", "prompt": "hi", "exec_recipe": "sharp"}).json()["id"]
    assert broker.store._all("SELECT exec_recipe FROM jobs WHERE id=?", (jid,)) == [{"exec_recipe": ""}]
