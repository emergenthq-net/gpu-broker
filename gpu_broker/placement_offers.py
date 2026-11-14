"""Pure conversion of provider quotes to bounded compute-only placement offers."""
from __future__ import annotations

from math import isfinite

from .placement_planner import PlacementOffer
from .provider_quotes import ProviderQuote, estimate_compute_usd


def quote_to_offer(
    quote: ProviderQuote,
    *,
    duration_s: float,
    estimated_latency_ms: float,
    additional_cost_usd: float | None,
) -> PlacementOffer:
    """Unknown non-compute charges are not silently assumed zero."""
    if additional_cost_usd is None or not isfinite(additional_cost_usd) or additional_cost_usd < 0:
        return PlacementOffer(quote.capability, None, estimated_latency_ms)
    compute = estimate_compute_usd(quote, duration_s=duration_s)
    total = compute + additional_cost_usd
    if not isfinite(total):
        return PlacementOffer(quote.capability, None, estimated_latency_ms)
    return PlacementOffer(quote.capability, total, estimated_latency_ms)
