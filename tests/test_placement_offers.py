"""Offline quote-to-offer regression tests."""
from gpu_broker.placement_offers import quote_to_offer
from gpu_broker.provider_quotes import parse_quote

QUOTE = parse_quote({
    "provider": "example", "sku": "gpu80", "memory_gib": 80,
    "hourly_usd": 2.0, "currency": "USD", "source": "fixture",
}, observed_at_s=100)


def test_total_cost_includes_non_compute_charges() -> None:
    offer = quote_to_offer(QUOTE, duration_s=1800, estimated_latency_ms=1000, additional_cost_usd=0.25)
    assert offer.estimated_total_usd == 1.25


def test_unknown_additional_cost_cannot_be_ranked_as_free() -> None:
    offer = quote_to_offer(QUOTE, duration_s=1800, estimated_latency_ms=1000, additional_cost_usd=None)
    assert offer.estimated_total_usd is None
