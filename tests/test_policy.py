from gpu_broker.constants import Priority
from gpu_broker.policy import BALANCED, FIFO, Candidate, normalize_priority, order


def c(jid, seq, priority, queued_at, resolved):
    return Candidate(jid, seq, priority, queued_at, resolved)


def test_fifo_is_strict_submission_order():
    xs = [
        c("late-interactive", 3, Priority.INTERACTIVE, 90, "resident"),
        c("first", 1, Priority.BACKGROUND, 0, "other"),
        c("second", 2, Priority.NORMAL, 50, "resident"),
    ]
    assert [x.jid for x in order(xs, FIFO, "resident", 100, 300)] == ["first", "second", "late-interactive"]


def test_balanced_priority_then_residency_locality():
    xs = [
        c("other", 1, Priority.NORMAL, 90, "other"),
        c("local", 2, Priority.NORMAL, 90, "resident"),
        c("background", 3, Priority.BACKGROUND, 0, "resident"),
    ]
    assert [x.jid for x in order(xs, BALANCED, "resident", 100, 300)] == ["local", "other", "background"]


def test_aging_promotes_waiting_work_and_prevents_locality_starvation():
    xs = [
        c("old-other", 1, Priority.NORMAL, 0, "other"),
        c("new-local", 2, Priority.INTERACTIVE, 599, "resident"),
    ]
    # Two 300-second age bands make the old normal job effective-interactive and older
    # than the new interactive job, so residency affinity cannot starve it forever.
    assert next(iter(order(xs, BALANCED, "resident", 600, 300))).jid == "old-other"


def test_priority_normalization():
    assert normalize_priority(None) == Priority.NORMAL
    assert normalize_priority("BACKGROUND") == Priority.BACKGROUND
    assert normalize_priority("normal", interactive=True) == Priority.INTERACTIVE
