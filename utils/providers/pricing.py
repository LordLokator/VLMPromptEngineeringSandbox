"""Estimated cost of one answer from the vendored list prices (`utils/prices.json`).

Rules (design: providers): exact model ids only; an unknown model, or tokens the snapshot cannot price (cache tokens
without a cache rate, input above the long-context limit), give None: "n/a", never $0. A local agent (the locality
rule) has no API fee. `compat` and remote or cloud Ollama are unpriced. A cost the API reports itself wins.
"""
import json
import math
from functools import cache
from typing import Any

from utils.file_management import BASE_DIR
from utils.providers.settings import is_local

PRICES_FILE = BASE_DIR / "utils" / "prices.json"
LOCAL = "local"  # cost_source of an agent that runs on this machine: no API fee
API = "api"  # cost_source when the provider reported the cost


@cache
def load_prices() -> dict[str, Any]:
    with open(PRICES_FILE, encoding="utf-8") as f:
        return json.load(f)


def price_date() -> str:
    return str(load_prices()["_date"])


def estimate(
    provider: str,
    model: str,
    *,
    base_url: str | None = None,
    uncached_input: int | None,
    cache_read: int = 0,
    cache_write: int = 0,
    output: int | None,
    reported_cost: Any = None,
) -> tuple[float | None, str | None]:
    """(cost in USD, its source) for one answer, or (None, None) when it cannot be told."""
    if is_local(provider, model, base_url):
        return 0.0, LOCAL
    if isinstance(reported_cost, (int, float)) and not isinstance(reported_cost, bool) and math.isfinite(
            reported_cost) and reported_cost >= 0:
        return float(reported_cost), API
    entry = load_prices()["models"].get(provider, {}).get(model)
    if entry is None or uncached_input is None or output is None:
        return None, None
    limit = entry.get("priced_up_to_input_tokens")
    if limit is not None and uncached_input + cache_read + cache_write > limit:
        return None, None
    cost = uncached_input * entry["input"] + output * entry["output"]
    for tokens, rate in ((cache_read, entry.get("cache_read")), (cache_write, entry.get("cache_write"))):
        if tokens:
            if rate is None:
                return None, None
            cost += tokens * rate
    return cost / 1_000_000, f"list price {price_date()}"
