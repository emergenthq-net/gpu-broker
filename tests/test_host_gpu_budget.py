"""The host's `gpu` reading, auto's vendor choice included, ends within GPU_BUDGET_S of wall-clock
time; and that budget plus the broker's SSH connect fits in the broker's own timeout for the
call, so a hung nvidia-smi reaches the user as "set GPU_VENDOR=amd", not as an SSH timeout."""
import re
import time

import pytest

from gpu_broker import settings
from tests.gpufake import CTL, HELPER, gpu  # noqa: F401 — the fixture

# A `timeout` that really times out (perl's alarm), so a hung nvidia-smi is cut off as on the host.
REAL_TIMEOUT = '#!/bin/bash\necho "$*" >> "$TIMEOUT_LOG"\n[[ $1 == -k ]] && shift 2\nexec perl -e \'alarm shift; exec @ARGV\' "$@"\n'


def default(script, name):
    return int(re.search(rf'(?:^|; ){name}=(?:"\$\{{{name}:-)?(\d+)', script.read_text(), re.M).group(1))


def test_the_budget_fits_in_the_brokers_timeout_for_the_call():
    budget = default(CTL, "GPU_BUDGET_S")
    assert budget == default(HELPER, "GPU_BUDGET_S")
    t = settings.Timeouts()
    assert budget + t.ssh_connect_s < t.gpu_query_s


@pytest.mark.parametrize("budget", [3, 5])
def test_a_hung_nvidia_smi_ends_the_reading_within_the_budget_with_the_override_hint(gpu, budget):
    (gpu.bin / "timeout").write_text(REAL_TIMEOUT)
    t0 = time.monotonic()
    r = gpu("read", nv="ok", SMI_HANG="1", SMI_L_TAKES="60", NV_TIMEOUT_S="60", GPU_BUDGET_S=str(budget))
    took = time.monotonic() - t0
    assert r.returncode == 6 and r.stdout == "" and took <= budget + 0.5
    assert f"did not answer within {budget} s" in r.stderr and "GPU_VENDOR=amd" in r.stderr
    calls = gpu.timeout_calls()
    assert calls[0].startswith(f"-k 1 {budget - 1} nvidia-smi -L")   # the first call gets the budget, not NV_TIMEOUT_S


def test_a_reading_after_the_choice_gets_only_what_is_left(gpu):
    gpu("read", nv="ok", NV_TIMEOUT_S="60", GPU_BUDGET_S="5")
    t = [int(c.split()[2]) for c in gpu.timeout_calls()]
    assert len(t) == 2 and t[0] == 4 and 3 <= t[1] <= 4   # -L, then the reading: both within 5 s, never NV_TIMEOUT_S


def test_a_choice_that_used_the_budget_up_leaves_the_reading_no_unbounded_call(gpu):
    """`-L` answers after 2 of 3 s; the reading then has no time left and must fail at once,
    never call `timeout 0` (which waits forever)."""
    (gpu.bin / "timeout").write_text(REAL_TIMEOUT)
    t0 = time.monotonic()
    r = gpu("read", nv="ok", SMI_HANG="1", SMI_L_TAKES="2", NV_TIMEOUT_S="60", GPU_BUDGET_S="3")
    assert r.returncode == 6 and time.monotonic() - t0 <= 3.5
    assert len(gpu.timeout_calls()) == 1


def test_stream_drops_the_budget_once_the_vendor_is_chosen(gpu):
    gpu.stream(nv="ok", NV_TIMEOUT_S="7", GPU_BUDGET_S="2", lines=2)
    calls = gpu.timeout_calls()
    assert calls[0].startswith("-k 1 1 nvidia-smi -L") and all(c.startswith("-k 1 7 ") for c in calls[1:])


def test_the_budget_must_be_a_positive_whole_number(gpu):
    for bad in ("0", "1.5", "soon"):
        assert gpu("read", nv="ok", GPU_BUDGET_S=bad).returncode == 2
