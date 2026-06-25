"""Market-data provider abstraction.

The whole platform depends on the ``MarketData`` contract below, never on a specific
data source. That indirection is what lets the evals and the offline demo run on
deterministic fixtures while production uses live yfinance — same agents, same
scoring, swappable data.

Providers:
  * ``FixtureProvider`` — bundled JSON fixtures with a reproducible synthetic price
    path. Deterministic and offline; used by evals and as the ``auto`` fallback.
  * ``YFinanceProvider`` — live data from Yahoo Finance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class DataUnavailableError(Exception):
    """Raised when no provider can return usable data for a ticker."""


@dataclass
class MarketData:
    """Normalized market data — the single contract every agent reads from."""

    ticker: str
    name: str
    currency: str = "USD"
    source: str = "fixture"
    current_price: float | None = None
    closes: list[float] = field(default_factory=list)  # chronological daily closes
    fundamentals: dict[str, Any] = field(default_factory=dict)
    analyst: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    news: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class MarketDataProvider:
    def fetch(self, ticker: str) -> MarketData:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Fixture provider
# --------------------------------------------------------------------------- #
def _synthesize_closes(spec: dict[str, Any]) -> list[float]:
    """Build a reproducible daily-close series from compact fixture parameters.

    A log-linear trend from ``start`` to ``end`` plus seeded Gaussian noise. This
    keeps fixtures tiny while still producing realistic, testable momentum signals.
    """
    start = float(spec.get("start", 100.0))
    end = float(spec.get("end", 100.0))
    days = int(spec.get("days", 252))
    vol = float(spec.get("volatility", 0.012))
    seed = int(spec.get("seed", 0))

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, days)
    trend = start * (end / start) ** t  # smooth log-linear path
    noise = np.cumsum(rng.normal(0.0, vol, days))
    noise = noise - np.linspace(0.0, noise[-1], days)  # pin both ends to the trend
    closes = trend * np.exp(noise)
    return [round(float(c), 4) for c in closes]


class FixtureProvider(MarketDataProvider):
    def __init__(self, fixture_dir: Path = FIXTURE_DIR):
        self.fixture_dir = fixture_dir

    def available_tickers(self) -> list[str]:
        return sorted(p.stem.upper() for p in self.fixture_dir.glob("*.json"))

    def fetch(self, ticker: str) -> MarketData:
        path = self.fixture_dir / f"{ticker.upper()}.json"
        if not path.exists():
            raise DataUnavailableError(
                f"No fixture for '{ticker}'. Available: {', '.join(self.available_tickers())}"
            )
        spec = json.loads(path.read_text(encoding="utf-8"))
        closes = _synthesize_closes(spec["price_series"])
        return MarketData(
            ticker=ticker.upper(),
            name=spec.get("name", ticker.upper()),
            currency=spec.get("currency", "USD"),
            source="fixture",
            current_price=closes[-1],
            closes=closes,
            fundamentals={**spec.get("fundamentals", {}), "sector": spec.get("sector")},
            analyst=spec.get("analyst", {}),
            risk=spec.get("risk", {}),
            news=spec.get("news", []),
        )


# --------------------------------------------------------------------------- #
# Live yfinance provider
# --------------------------------------------------------------------------- #
class YFinanceProvider(MarketDataProvider):
    def fetch(self, ticker: str) -> MarketData:
        import yfinance as yf  # imported lazily so offline/fixture mode needs no network

        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y", interval="1d")
        if hist is None or hist.empty:
            raise DataUnavailableError(f"yfinance returned no price history for '{ticker}'.")
        closes = [round(float(c), 4) for c in hist["Close"].dropna().tolist()]
        if len(closes) < 30:
            raise DataUnavailableError(f"Too little price history for '{ticker}'.")

        info: dict[str, Any] = {}
        try:
            info = tk.info or {}
        except Exception:  # noqa: BLE001 - info is best-effort
            info = {}

        warnings: list[str] = []
        if not info:
            warnings.append("Fundamental/analyst data unavailable from source; relied on price action.")

        news = []
        try:
            for item in (tk.news or [])[:8]:
                title = item.get("title") or item.get("content", {}).get("title")
                if title:
                    news.append({"title": title})
        except Exception:  # noqa: BLE001
            pass

        return MarketData(
            ticker=ticker.upper(),
            name=info.get("longName") or info.get("shortName") or ticker.upper(),
            currency=info.get("currency", "USD"),
            source="live",
            current_price=closes[-1],
            closes=closes,
            fundamentals={
                "pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg": info.get("trailingPegRatio") or info.get("pegRatio"),
                "pb": info.get("priceToBook"),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "roe": info.get("returnOnEquity"),
                "market_cap": info.get("marketCap"),
                "eps": info.get("trailingEps"),
                "dividend_yield": info.get("dividendYield"),
                "sector": info.get("sector"),
            },
            analyst={
                "recommendation_mean": info.get("recommendationMean"),
                "target_mean": info.get("targetMeanPrice"),
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "num_analysts": info.get("numberOfAnalystOpinions"),
            },
            risk={"beta": info.get("beta")},
            news=news,
            warnings=warnings,
        )


def get_provider(source: str) -> MarketDataProvider:
    if source == "live":
        return YFinanceProvider()
    if source == "fixture":
        return FixtureProvider()
    return _AutoProvider()


class _AutoProvider(MarketDataProvider):
    """Try live data; on any failure transparently fall back to fixtures."""

    def __init__(self) -> None:
        self.live = YFinanceProvider()
        self.fixture = FixtureProvider()

    def fetch(self, ticker: str) -> MarketData:
        try:
            return self.live.fetch(ticker)
        except Exception as exc:  # noqa: BLE001 - fallback is the whole point
            try:
                data = self.fixture.fetch(ticker)
                data.warnings.append(
                    f"Live data unavailable ({type(exc).__name__}); used bundled fixture data."
                )
                return data
            except DataUnavailableError:
                raise DataUnavailableError(
                    f"Could not fetch '{ticker}' live, and no fixture exists for it."
                ) from exc
