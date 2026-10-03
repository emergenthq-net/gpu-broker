"""Image-to-video and image-edit jobs through the HTTP API: the image is checked at submit,
staged, uploaded to ComfyUI under a job-derived name, and its file name reaches the graph."""
import base64
import dataclasses
import json

import pytest

from gpu_broker.broker import STAGING_FAILED, Broker, StagingError
from gpu_broker.constants import JobState
from tests.helpers import FakeBackends, FakeDriver, done, leftover_staged

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
B64 = base64.b64encode(PNG).decode()


def job(c, **body):
    return c.post("/v1/jobs", json={"prompt": "a fox", **body})


def staged(broker):
    """What is left in staging, once the job's files are dropped (just after its terminal state)."""
    return leftover_staged(broker)


def test_i2v_job_uploads_the_image_and_the_graph_loads_it(client, broker):
    r = job(client, model="wan-i2v", image=B64)
    assert r.status_code == 200 and r.json()["resolved"] == "wan2.2-14b-i2v"
    jid = r.json()["id"]
    j = done(broker, jid)
    assert j["state"] == JobState.DONE, j
    name = f"broker-{jid}-image.png"
    assert broker.backends.uploads == [(name, PNG, "png")]
    graph = broker.backends.graphs[-1]
    assert graph["17"] == {"class_type": "LoadImage", "inputs": {"image": name}}
    assert staged(broker) == []                      # removed once uploaded


def test_the_stored_payload_records_the_image_but_not_its_data(client, broker):
    jid = job(client, model="minimax-h3-i2v", image=B64,
              end_image="data:image/jpeg;base64," + base64.b64encode(JPEG).decode()).json()["id"]
    done(broker, jid)
    payload = broker.store.job(jid)["payload"]
    assert "image" not in payload and "end_image" not in payload
    assert payload["inputs"] == {"image": {"bytes": len(PNG), "type": "png", "source": "inline"},
                                 "end_image": {"bytes": len(JPEG), "type": "jpeg", "source": "inline"}}
    g = broker.backends.graphs[-1]
    assert g["6"]["inputs"]["last_frame"] == ["17", 0] and g["17"]["inputs"]["image"] == f"broker-{jid}-end_image.jpg"


def test_a_model_that_needs_an_image_is_a_400_at_submit_and_records_nothing(client, broker):
    before = len(broker.store.jobs(100))
    r = job(client, model="wan2.2-14b-i2v")
    assert r.status_code == 400 and "needs an input: send image" in r.json()["detail"]
    assert len(broker.store.jobs(100)) == before


def test_an_image_for_a_text_only_model_is_a_400(client):
    r = job(client, model="wan2.2-14b-t2v", image=B64)
    assert r.status_code == 400 and "takes no image input" in r.json()["detail"]
    r = job(client, model="wan2.2-14b-i2v", image=B64, end_image=B64)
    assert r.status_code == 400 and "takes no end_image input" in r.json()["detail"]


def test_bad_image_data_is_a_400(client, broker):
    broker.settings = dataclasses.replace(broker.settings, inputs=dataclasses.replace(broker.settings.inputs, max_bytes=4096))
    for bad, msg in ((base64.b64encode(b"GIF89a....").decode(), "not an accepted file"),
                     ("not base64!", "not valid base64"),
                     (base64.b64encode(PNG * 1000).decode(), "larger than")):
        r = job(client, model="wan2.2-14b-i2v", image=bad)
        assert r.status_code == 400 and msg in r.json()["detail"], r.json()
    assert staged(broker) == []


def test_image_urls_are_refused_by_default(client):
    r = job(client, model="wan2.2-14b-i2v", image_url="http://example.com/a.png")
    assert r.status_code == 400 and "inputs.allow_urls" in r.json()["detail"]


def test_optional_image_model_runs_with_or_without_one(client, broker):
    for body in ({}, {"image": B64}):
        j = done(broker, job(client, model="wan5b", **body).json()["id"])
        assert j["state"] == JobState.DONE
    with_image, without = broker.backends.graphs[-1], broker.backends.graphs[-2]
    assert "start_image" in with_image["7"]["inputs"] and "start_image" not in without["7"]["inputs"]


def test_substitutes_for_an_image_job_take_images(client, broker):
    r = job(client, model="some/unknown-i2v-model", kind="video", image=B64)
    assert r.json()["resolved"] == "minimax-h3-i2v"          # best video model that takes an image
    r = job(client, model="some/unknown-t2v-model", kind="video")
    assert r.json()["resolved"] == "wan2.2-14b-t2v"          # never one that needs an image
    done(broker, r.json()["id"])


def test_an_upload_failure_fails_the_job_and_drops_the_staged_image(client, broker):
    def refuse(name, data, kind):
        raise RuntimeError("ComfyUI rejected the input image: disk full")
    broker.backends.comfy_upload = refuse
    j = done(broker, job(client, model="wan-i2v", image=B64).json()["id"])
    assert j["state"] == JobState.FAILED and "disk full" in j["error"]
    assert staged(broker) == []


def test_restart_clears_images_left_by_the_previous_process(broker):
    broker.staging.dir.mkdir(parents=True, exist_ok=True)
    (broker.staging.dir / "0123456789ab-image.png").write_bytes(PNG)
    driver = FakeDriver({"llama-8b"})
    again = Broker(broker.settings, env={}, driver=driver, backends=FakeBackends(driver))
    again.start()
    again.stop()
    assert staged(broker) == []


def test_a_rejected_job_never_reads_its_image(client, broker):
    r = job(client, model="no-such-model", image="not base64!")   # unknown and no kind: rejected
    assert r.status_code == 200 and r.json()["job"]["state"] == JobState.REJECTED
    assert staged(broker) == []


def test_a_staging_failure_fails_the_job_instead_of_orphaning_it(broker, monkeypatch):
    real_put = broker.staging.put

    def half_then_fail(jid, files):
        real_put(jid, files[:1])           # a partial file is left behind ...
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(broker.staging, "put", half_then_fail)
    queued = []
    monkeypatch.setattr(broker.scheduler, "submit", queued.append)
    with pytest.raises(StagingError) as e:
        broker.submit({"model": "minimax-h3-i2v", "prompt": "p", "image": B64,
                       "end_image": base64.b64encode(JPEG).decode()}, "t")
    (j,) = broker.store.jobs(10)
    assert (e.value.jid, j["state"], j["error"]) == (j["id"], JobState.FAILED, STAGING_FAILED)
    assert staged(broker) == [] and queued == []   # ... and removed; nothing was queued


def test_staging_failure_answers_json_with_the_job_id_and_no_detail(client, broker, monkeypatch, caplog):
    def fail(jid, files):
        raise OSError(13, "Permission denied", "/var/lib/gpu-broker/inputs/secret-path")
    monkeypatch.setattr(broker.staging, "put", fail)
    monkeypatch.setattr(broker.staging, "discard", lambda jid: (_ for _ in ()).throw(OSError("busy")))
    r = job(client, model="wan-i2v", image=B64)
    (j,) = broker.store.jobs(10)
    assert r.status_code == 500 and r.json() == {"detail": STAGING_FAILED, "id": j["id"]}
    assert j["state"] == JobState.FAILED and j["error"] == STAGING_FAILED   # marked although discard failed
    assert "secret-path" not in r.text and "secret-path" not in json.dumps(broker.store.job(j["id"]))
    assert "secret-path" in caplog.text   # the detail goes to the server log


def test_a_job_whose_staged_image_vanished_fails_before_the_graph_runs(client, broker, monkeypatch):
    monkeypatch.setattr(broker.staging, "received", lambda jid: {})
    uploads = []
    monkeypatch.setattr(broker.staging, "upload", lambda jid, uploader: uploads.append(jid) or {})
    j = done(broker, job(client, model="wan-i2v", image=B64).json()["id"])
    assert j["state"] == JobState.FAILED and "input files missing" in j["error"]
    assert broker.backends.graphs == [] and uploads == []   # checked before anything is uploaded
