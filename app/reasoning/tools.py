"""Dynamic tools for the reasoning agent. (v2 addition)

These are the *only* levers the LLM agent can pull. They expose RAW computed data —
never the deterministic scores — so the agent forms an independent view (per the
design decision that the experiment must not just parrot the deterministic engine).

Each tool is an OpenAI function-calling schema plus a Python executor that reads the
already-fetched ``MarketData`` and reuses ``app/indicators.py`` for the math. A tool
that cannot answer returns an ``{"error": ...}`` payload rather than raising — a tool
error is a normal signal the agent reflects/retries on.
"""

from __future__ import annotations

from typing import Any

from app import indicators as ind
from app.data.provider import MarketData


# --- OpenAI function-calling schemas -----------------------------------------
def _tool(name: str, description: str) -> dict:
    # All four tools take only the ticker; the agent chooses which lenses to pull
    # and in what order (that ordering is the "dynamic" part of the loop).
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The ticker under analysis."}
                },
                "required": ["ticker"],
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _tool("get_price_technicals",
          "Price momentum & trend: current price, SMA50, SMA200, RSI(14), MACD histogram, "
          "and 3-month / 6-month trailing returns."),
    _tool("get_fundamentals",
          "Valuation & quality: P/E, PEG, profit margin, revenue growth, ROE, market cap."),
    _tool("get_sentiment_inputs",
          "Market sentiment raw inputs: analyst recommendation mean (1=Strong Buy..5=Strong "
          "Sell), mean price target, number of analysts, and recent news headlines."),
    _tool("get_risk_metrics",
          "Risk profile: annualized volatility, max drawdown, beta, and market cap."),
]

TOOL_NAMES = {s["function"]["name"] for s in TOOL_SCHEMAS}


# --- Executors ----------------------------------------------------------------
def _get_price_technicals(data: MarketData) -> dict[str, Any]:
    closes = data.closes
    if len(closes) < 30:
        return {"error": "insufficient price history"}
    return {
        "current_price": round(closes[-1], 2),
        "sma50": ind.sma(closes, 50),
        "sma200": ind.sma(closes, 200),
        "rsi14": ind.rsi(closes, 14),
        "macd_histogram": ind.macd_histogram(closes),
        "return_3m": ind.pct_return(closes, 63),
        "return_6m": ind.pct_return(closes, 126),
    }


def _get_fundamentals(data: MarketData) -> dict[str, Any]:
    f = data.fundamentals or {}
    if not f:
        return {"error": "no fundamental data available"}
    return {
        "pe": f.get("pe") or f.get("forward_pe"),
        "peg": f.get("peg"),
        "profit_margin": f.get("profit_margin"),
        "revenue_growth": f.get("revenue_growth"),
        "roe": f.get("roe"),
        "market_cap": f.get("market_cap"),
    }


def _get_sentiment_inputs(data: MarketData) -> dict[str, Any]:
    a = data.analyst or {}
    news = [item.get("title", "") for item in (data.news or [])]
    if not a and not news:
        return {"error": "no analyst or news data available"}
    return {
        "recommendation_mean": a.get("recommendation_mean"),
        "target_mean": a.get("target_mean"),
        "num_analysts": a.get("num_analysts"),
        "current_price": data.current_price,
        "news_headlines": news,
    }


def _get_risk_metrics(data: MarketData) -> dict[str, Any]:
    closes = data.closes
    return {
        "annualized_volatility": ind.annualized_volatility(closes) if closes else None,
        "max_drawdown": ind.max_drawdown(closes) if closes else None,
        "beta": (data.risk or {}).get("beta"),
        "market_cap": (data.fundamentals or {}).get("market_cap"),
    }


_EXECUTORS = {
    "get_price_technicals": _get_price_technicals,
    "get_fundamentals": _get_fundamentals,
    "get_sentiment_inputs": _get_sentiment_inputs,
    "get_risk_metrics": _get_risk_metrics,
}


def dispatch(name: str, arguments: dict, data: MarketData) -> dict[str, Any]:
    """Execute a tool against the fetched data. Never raises — errors come back as data."""
    fn = _EXECUTORS.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        result = fn(data)
    except Exception as exc:  # noqa: BLE001 - surface as a reflectable signal, don't crash the loop
        return {"error": f"{type(exc).__name__}: {exc}"}
    # Round floats for compact, stable payloads back to the model.
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}
