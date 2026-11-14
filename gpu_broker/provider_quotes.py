"""Offline provider quote normalization for optional remote accelerator discovery.

No network access, credentials, provisioning, or execution authority.
Provider responses are untrusted observations, not execution permission.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from .placement_preflight import Capability

BYTES_PER_GIB = 1024 ** 3


@dataclass(frozen=True)
class ProviderQuote:
    capability: Capability
    sku: str
    hourly_usd: float
    currency: str
    source: str


def parse_quote(raw: Mapping[str, Any], *, observed_at_s: float) -> ProviderQuote:
    """Normalize a caller-supplied snapshot; reject incomplete or unsafe quotes."""
    provider = raw.get("provider")
    sku = raw.get("sku")
    source = raw.get("source")
    currency = raw.get("currency")
    memory_gib = raw.get("memory_gib")
    hourly = raw.get("hourly_usd")
    if not isinstance(provider, str) or not provider.strip() or not isinstance(sku, str) or not sku.strip() or not isinstance(source, str) or not source.strip():
        raise ValueError("missing provider identity or quote provenance")
    if currency != "USD":
        raise ValueError("unsupported quote currency")
    if isinstance(memory_gib, bool) or not isinstance(memory_gib, (int, float)):
        raise ValueError("invalid memory")
    if isinstance(hourly, bool) or not isinstance(hourly, (int, float)):
        raise ValueError("invalid hourly rate")
    if not isfinite(memory_gib) or memory_gib <= 0 or not isfinite(hourly) or hourly < 0:
        raise ValueError("invalid memory or hourly rate")
    if not isfinite(observed_at_s) or observed_at_s < 0:
        raise ValueError("invalid observation time")
    memory_bytes = memory_gib * BYTES_PER_GIB
    if not isfinite(memory_bytes) or int(memory_bytes) != memory_bytes:
        raise ValueError("memory must resolve to finite whole bytes")
    return ProviderQuote(
        capability=Capability(provider, int(memory_bytes), observed_at_s, True),
        sku=sku,
        hourly_usd=float(hourly),
        currency=currency,
        source=source,
    )


def estimate_compute_usd(quote: ProviderQuote, *, duration_s: float) -> float:
    """Compute-only estimate; excludes storage, transfers, taxes and idle time."""
    if not isfinite(duration_s) or duration_s < 0:
        raise ValueError("invalid duration")
    cost = quote.hourly_usd * duration_s / 3600
    if not isfinite(cost):
        raise ValueError("invalid estimated cost")
    return cost


def parse_supercomputer_skus(
    payload: Mapping[str, Any], *, observed_at_s: float,
) -> tuple[ProviderQuote, ...]:
    """Normalize documented GET /v1/skus response without network or effects.

    The published catalog is an observation, not proof of capacity or permission.
    Unknown/malformed SKU records are rejected rather than silently priced.
    """
    skus = payload.get("skus")
    if not isinstance(skus, list):
        raise ValueError("invalid SKU catalog")
    quotes: list[ProviderQuote] = []
    for item in skus:
        if not isinstance(item, dict):
            raise ValueError("invalid SKU entry")
        availability = item.get("availability")
        if availability not in ("live", "unavailable", "queued"):
            raise ValueError("unknown SKU availability")
        if availability != "live":
            continue  # Non-live SKUs cannot become placement candidates.
        quotes.append(parse_quote({
            "provider": "supercomputer.sh",
            "sku": item.get("name"),
            "memory_gib": item.get("vram_gb"),
            "hourly_usd": item.get("cost_per_hr"),
            "currency": "USD",
            "source": "GET /v1/skus",
        }, observed_at_s=observed_at_s))
    return tuple(quotes)
