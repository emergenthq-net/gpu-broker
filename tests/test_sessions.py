"""Interactive sessions end on idle, on a waiting job (sooner), on request, or at the cap."""
from gpu_broker.sessions import End, Sessions

DEFAULTS = {"session_idle_s": 100, "session_yield_s": 10, "session_max_s": 1000}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make(queue=lambda: 0, queued=False):
    c = Clock()
    s = Sessions(DEFAULTS, queue, poll_s=5, clock=c, sleep=c.sleep)
    s.queued_jobs = lambda: queued
    return s, c


def test_ends_when_comfy_is_idle():
    s, _ = make()
    out = s.hold("j", "m", {})
    assert out["ended"] == End.IDLE and out["held_s"] == 100


def test_yields_sooner_when_a_job_waits():
    s, _ = make(queued=True)
    assert s.hold("j", "m", {})["ended"] == End.YIELDED


def test_rendering_or_unknown_counts_as_busy_until_the_cap():
    for q in (lambda: 3, lambda: None):
        s, _ = make(queue=q)
        assert s.hold("j", "m", {})["ended"] == End.MAX


def test_user_end_and_custom_idle():
    s, _ = make()
    ended = []
    s.queue_len = lambda: ended.append(s.end()) or 1
    assert s.hold("j", "m", {"idle_min": 1})["ended"] == End.USER and ended[0] is True
    assert s.view() is None and s.end() is False
    s2, _ = make()
    assert s2.hold("j", "m", {"idle_min": 1})["held_s"] == 60
