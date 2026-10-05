"""The live `fair` queue: each job's class (x-priority, else the requester), a light requester
ahead of a heavy one's backlog, a switch the pool cannot starve, and `fifo` as the rollback."""
import dataclasses
import threading
import time

import pytest

from gpu_broker.broker import Broker
from gpu_broker.settings import Scheduling
from tests.helpers import WAIT_S, FakeBackends, FakeDriver, done, make_settings, wait_idle

HEAVY = "batch-agent"   # the fixture catalog's background requester


def call(b, requester, priority="", model="llama-8b", **body):
    return b.submit({"model": model, "messages": [], **body}, requester, priority=priority)[0]


def started(b, jid):
    assert done(b, jid)["state"] == "done"
    return next(e["ts"] for e in b.store.events(0, 10000) if e["job_id"] == jid and e["kind"] == "job.running")


@pytest.fixture
def make(tmp_path):
    made = []

    def build(policy="fair"):
        (d := tmp_path / policy).mkdir()
        s = dataclasses.replace(make_settings(d), scheduler=Scheduling(policy=policy))
        driver = FakeDriver({"llama-8b"})
        b = Broker(s, env={}, driver=driver, backends=FakeBackends(driver))
        b.start()
        made.append(b)
        return b
    yield build
    for b in made:
        b.backends.chat_gate and b.backends.chat_gate.set()
        assert wait_idle(b)
        b.stop()


def interactive(b, jid):
    return b.store.job(jid)["payload"].get("interactive")


def test_the_class_is_the_header_else_the_requester(make):
    b = make()
    assert interactive(b, call(b, "operator")) is True
    assert interactive(b, call(b, HEAVY)) is False
    assert interactive(b, call(b, "operator", "background")) is False
    assert interactive(b, call(b, HEAVY, "interactive")) is False   # a background requester can only lower
    assert interactive(b, call(b, HEAVY, "", interactive=True)) is False   # ...whether by header or body
    assert interactive(b, call(b, "operator", "", interactive=False)) is False   # a route that already decided
    assert interactive(b, call(b, "operator", "background", interactive=True)) is False   # the header wins


def test_jobs_take_the_priority_header(client, broker):
    for header, want in (("background", False), ("interactive", True)):
        r = client.post("/v1/jobs", json={"model": "llama-8b", "messages": []}, headers={"x-priority": header})
        assert interactive(broker, r.json()["id"]) is want


def test_fifo_leaves_the_class_as_the_caller_sent_it(make):
    b = make("fifo")
    assert interactive(b, call(b, "operator")) is None and interactive(b, call(b, HEAVY, "interactive")) is None


def light_vs_heavy(b):
    b.backends.chat_gate = threading.Event()
    slots = b.catalog.models["llama-8b"]["slots"] - b.catalog.models["llama-8b"]["reserved_interactive"]
    heavy = [call(b, HEAVY) for _ in range(slots + 3)]
    end = time.monotonic() + WAIT_S
    while len(b.scheduler.pool.ids()) < slots and time.monotonic() < end:
        time.sleep(0.005)
    light = call(b, "light", "background")
    return heavy, light


def test_fair_puts_a_light_requester_ahead_of_a_heavy_ones_backlog(make):
    b = make()
    heavy, light = light_vs_heavy(b)
    assert b.scheduler.position(light) == 1 and b.scheduler.snapshot()[0] == [light, *heavy[-3:]]
    b.backends.chat_gate.set()
    assert started(b, light) <= min(started(b, j) for j in heavy[-3:])


def test_fifo_keeps_arrival_order_and_the_head_holds_the_gpu_thread(make):
    b = make("fifo")
    heavy, light = light_vs_heavy(b)
    chat = call(b, "operator", "interactive")
    time.sleep(0.1)
    assert chat not in b.scheduler.pool.ids()   # head-of-line: stuck behind the waiting background call
    assert b.scheduler.snapshot()[0][-2:] == [light, chat] and b.scheduler.snapshot()[0][:2] == heavy[-2:]


@pytest.mark.parametrize("model", ["sdxl-base", "qwen-coder-32b"])   # a ComfyUI job, a non-resident LLM
def test_calls_sent_during_a_drain_cannot_starve_the_switch(make, model):
    b = make()
    b.backends.chat_gate = gate = threading.Event()
    first = [call(b, HEAVY) for _ in range(2)]
    end = time.monotonic() + WAIT_S
    while len(b.scheduler.pool.ids()) < 2 and time.monotonic() < end:
        time.sleep(0.005)
    img = b.submit({"model": model, "prompt": "x", "messages": []}, "operator")[0]
    time.sleep(0.1)                                     # the GPU thread has taken the image and is draining
    late = [call(b, HEAVY) for _ in range(2)]
    time.sleep(0.1)
    assert set(b.scheduler.pool.ids()) == set(first)    # the late calls did not refill the pool
    gate.set()
    assert all(started(b, j) > started(b, img) for j in late)
