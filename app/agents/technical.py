"""Technical agent — price momentum and trend.

Signals: moving-average trend (50/200), RSI, MACD histogram, and multi-horizon
returns. Each maps to a signed contribution; the agent score is their weighted mean.
"""

from __future__ import annotations

from app import indicators as ind
from app.agents.base import Agent, clamp, scale
from app.data.provider import MarketData
from app.schemas import AgentResult, Signal


class TechnicalAgent(Agent):
    name = "technical"

    def analyze(self, data: MarketData) -> AgentResult:
        closes = data.closes
        if len(closes) < 30:
            return self._empty("insufficient price history")

        price = closes[-1]
        signals: list[Signal] = []
        contributions: list[tuple[float, float]] = []  # (contribution, weight)

        # --- Moving-average trend ------------------------------------------
        sma50 = ind.sma(closes, 50)
        sma200 = ind.sma(closes, 200)
        if sma50 and sma200:
            trend = scale(price / sma50 - 1, 0.0, 0.10) * 0.5 + scale(sma50 / sma200 - 1, 0.0, 0.10) * 0.5
            cross = "golden cross (50>200)" if sma50 > sma200 else "death cross (50<200)"
            signals.append(Signal(
                name="Moving-average trend",
                value=f"price {price:.2f} vs SMA50 {sma50:.2f}, SMA200 {sma200:.2f}",
                interpretation=f"{cross}; price {'above' if price > sma50 else 'below'} short-term average.",
                contribution=round(trend, 3),
            ))
            contributions.append((trend, 0.35))
        elif sma50:
            trend = scale(price / sma50 - 1, 0.0, 0.10)
            signals.append(Signal(
                name="Short-term trend", value=f"price {price:.2f} vs SMA50 {sma50:.2f}",
                interpretation="Price relative to 50-day average (200-day unavailable).",
                contribution=round(trend, 3),
            ))
            contributions.append((trend, 0.30))

        # --- RSI ------------------------------------------------------------
        rsi = ind.rsi(closes, 14)
        if rsi is not None:
            # Overbought (>70) is a mild headwind for new entry; oversold (<30) a mild tailwind.
            rsi_contrib = clamp((55 - rsi) / 35) * 0.6
            if rsi > 70:
                interp = "Overbought — stretched, pullback risk."
            elif rsi < 30:
                interp = "Oversold — potential mean-reversion bounce."
            else:
                interp = "Neutral momentum range."
            signals.append(Signal(
                name="RSI (14)", value=f"{rsi:.1f}", interpretation=interp,
                contribution=round(rsi_contrib, 3),
            ))
            contributions.append((rsi_contrib, 0.15))

        # --- MACD -----------------------------------------------------------
        macd = ind.macd_histogram(closes)
        if macd is not None:
            macd_contrib = scale(macd / price, 0.0, 0.02)
            signals.append(Signal(
                name="MACD histogram", value=f"{macd:+.3f}",
                interpretation="Bullish momentum." if macd > 0 else "Bearish momentum.",
                contribution=round(macd_contrib, 3),
            ))
            contributions.append((macd_contrib, 0.20))

        # --- Multi-horizon returns -----------------------------------------
        r3m = ind.pct_return(closes, 63)
        r6m = ind.pct_return(closes, 126)
        if r3m is not None or r6m is not None:
            parts, mom = [], 0.0
            if r3m is not None:
                parts.append(f"3M {r3m:+.1%}")
                mom += scale(r3m, 0.0, 0.25) * 0.5
            if r6m is not None:
                parts.append(f"6M {r6m:+.1%}")
                mom += scale(r6m, 0.0, 0.40) * 0.5
            signals.append(Signal(
                name="Price momentum", value=", ".join(parts),
                interpretation="Positive trailing returns." if mom > 0 else "Negative trailing returns.",
                contribution=round(mom, 3),
            ))
            contributions.append((mom, 0.30))

        if not contributions:
            return self._empty("could not compute any technical signal")

        total_w = sum(w for _, w in contributions)
        score = clamp(sum(c * w for c, w in contributions) / total_w)
        # Confidence scales with how much history we have (full ~1y -> ~1.0).
        confidence = clamp(len(closes) / 200.0, 0.0, 1.0) * 0.9 + 0.1

        return AgentResult(
            agent=self.name, score=round(score, 3), confidence=round(confidence, 3),
            signals=signals,
            summary=self._summary(score),
            data={"sma50": sma50, "sma200": sma200, "rsi": rsi, "macd_hist": macd,
                  "return_3m": r3m, "return_6m": r6m},
        )

    @staticmethod
    def _summary(score: float) -> str:
        if score > 0.3:
            return "Technicals are constructive: trend and momentum favor the bulls."
        if score < -0.3:
            return "Technicals are weak: downtrend and negative momentum dominate."
        return "Technicals are mixed/neutral with no decisive trend."
