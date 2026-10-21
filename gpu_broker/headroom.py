"""Fresh negative capacity bounds; catalog reclaim is an estimate, never admission."""
from .drivers import DRIVER_ERRORS, Driver


def llm_headroom(driver: Driver, reclaim: int) -> tuple[int, int]:
    """Return (optimistic free MiB, total MiB). Unknown reclaim leaves only a card bound.

    No memory is credited to a planned ComfyUI unload: its effect must already
    appear in the fresh GPU reading. Aggregate readings do not identify tenants.
    """
    try:
        used, total, _ = driver.gpu()
    except DRIVER_ERRORS as exc:
        raise RuntimeError("GPU headroom observation unavailable; preserving resident LLM") from exc
    if total <= 0 or used < 0 or used > total:
        raise RuntimeError("Invalid GPU headroom observation; preserving resident LLM")
    possible = min(total, total - used + reclaim) if reclaim else total
    return possible, total
