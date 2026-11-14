"""Pure provider snapshot tests: no network, secrets or GPUs."""
import pytest

from gpu_broker.provider_quotes import estimate_compute_usd, parse_quote, parse_supercomputer_skus

QUOTE = {
    "provider": "example-remote",
    "sku": "accelerator-80g",
    "memory_gib": 80,
    "hourly_usd": 2.0,
    "currency": "USD",
    "source": "offline-fixture",
}


def test_snapshot_and_compute_estimate() -> None:
    quote = parse_quote(QUOTE, observed_at_s=100.0)
    assert quote.capability.remote
    assert quote.capability.memory_bytes == 80 * 1024**3
    assert estimate_compute_usd(quote, duration_s=1800) == 1.0


@pytest.mark.parametrize(
    "change",
    [
        {"currency": "EUR"},
        {"hourly_usd": -1},
        {"hourly_usd": float("nan")},
        {"memory_gib": 0},
        {"memory_gib": True},
        {"provider": ""},
        {"source": ""},
    ],
)
def test_invalid_quote_refused(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_quote(QUOTE | change, observed_at_s=100.0)


def test_invalid_duration_refused() -> None:
    quote = parse_quote(QUOTE, observed_at_s=100.0)
    with pytest.raises(ValueError):
        estimate_compute_usd(quote, duration_s=float("inf"))


def test_supercomputer_documented_sku_snapshot() -> None:
    payload = {"skus": [{
        "name": "h100", "label": "H100", "vram_gb": 80,
        "vcpus": 16, "ram_gb": 128, "cost_per_hr": 3.16,
        "availability": "live", "effective_concurrency": 1,
    }], "catalog_status": "ok"}
    quotes = parse_supercomputer_skus(payload, observed_at_s=100.0)
    assert len(quotes) == 1
    assert quotes[0].sku == "h100"
    assert quotes[0].capability.remote
    assert quotes[0].source == "GET /v1/skus"


def test_supercomputer_catalog_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError):
        parse_supercomputer_skus({"skus": {"name": "h100"}}, observed_at_s=100.0)
    with pytest.raises(ValueError):
        parse_supercomputer_skus({"skus": [{"name": "h100", "vram_gb": 80,
            "cost_per_hr": 3.16, "availability": "mystery"}]}, observed_at_s=100.0)


def test_unavailable_and_queued_skus_not_offered() -> None:
    catalog = {"skus": [
        {"name": "not-ready", "vram_gb": 80, "cost_per_hr": 0.01, "availability": "queued"},
        {"name": "offline", "vram_gb": 80, "cost_per_hr": 0.01, "availability": "unavailable"},
        {"name": "ready", "vram_gb": 80, "cost_per_hr": 2.0, "availability": "live"},
    ]}
    quotes = parse_supercomputer_skus(catalog, observed_at_s=100)
    assert tuple(q.sku for q in quotes) == ("ready",)
