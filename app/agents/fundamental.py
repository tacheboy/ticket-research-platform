"""Fundamental agent — valuation and business quality.

Signals: P/E, PEG, P/B (valuation), plus margins, revenue growth, and ROE (quality).
Cheap + growing + profitable scores bullish; expensive + shrinking + unprofitable
scores bearish. Missing metrics lower confidence rather than crashing.
"""

from __future__ import annotations

from app.agents.base import Agent, clamp, scale
from app.data.provider import MarketData
from app.schemas import AgentResult, Signal


class FundamentalAgent(Agent):
    name = "fundamental"

    def analyze(self, data: MarketData) -> AgentResult:
        f = data.fundamentals or {}
        signals: list[Signal] = []
        contributions: list[tuple[float, float]] = []
        present = 0
        expected = 5

        pe = f.get("pe") or f.get("forward_pe")
        if isinstance(pe, (int, float)) and pe > 0:
            present += 1
            # ~15 P/E is fair; lower is cheaper (bullish), higher pricier (bearish).
            contrib = -scale(pe, 18.0, 22.0)
            signals.append(Signal(
                name="P/E ratio", value=f"{pe:.1f}",
                interpretation="Inexpensive vs market." if pe < 18 else
                               ("Rich valuation." if pe > 30 else "Roughly fair valuation."),
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.30))
        elif pe is not None and (not isinstance(pe, (int, float)) or pe <= 0):
            signals.append(Signal(
                name="P/E ratio", value="n/a (negative earnings)",
                interpretation="Company is unprofitable — no meaningful P/E.",
                contribution=-0.3,
            ))
            contributions.append((-0.3, 0.20))

        peg = f.get("peg")
        if isinstance(peg, (int, float)) and peg > 0:
            present += 1
            contrib = -scale(peg, 1.5, 1.5)  # PEG ~1 cheap-for-growth, >2 expensive
            signals.append(Signal(
                name="PEG ratio", value=f"{peg:.2f}",
                interpretation="Cheap relative to growth." if peg < 1.2 else
                               ("Expensive vs growth." if peg > 2.5 else "Reasonable vs growth."),
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.20))

        margin = f.get("profit_margin")
        if isinstance(margin, (int, float)):
            present += 1
            contrib = scale(margin, 0.08, 0.20)
            signals.append(Signal(
                name="Profit margin", value=f"{margin:.1%}",
                interpretation="Healthy profitability." if margin > 0.12 else
                               ("Loss-making." if margin < 0 else "Thin margins."),
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.20))

        growth = f.get("revenue_growth")
        if isinstance(growth, (int, float)):
            present += 1
            contrib = scale(growth, 0.05, 0.20)
            signals.append(Signal(
                name="Revenue growth", value=f"{growth:+.1%}",
                interpretation="Growing." if growth > 0.05 else
                               ("Contracting." if growth < 0 else "Flat."),
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.20))

        roe = f.get("roe")
        if isinstance(roe, (int, float)):
            present += 1
            contrib = scale(roe, 0.12, 0.25)
            signals.append(Signal(
                name="Return on equity", value=f"{roe:.1%}",
                interpretation="Strong capital efficiency." if roe > 0.15 else
                               ("Destroying equity value." if roe < 0 else "Modest returns."),
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.15))

        if not contributions:
            return self._empty("no fundamental metrics available")

        total_w = sum(w for _, w in contributions)
        score = clamp(sum(c * w for c, w in contributions) / total_w)
        confidence = clamp(present / expected, 0.0, 1.0) * 0.9 + 0.05

        return AgentResult(
            agent=self.name, score=round(score, 3), confidence=round(confidence, 3),
            signals=signals,
            summary=self._summary(score),
            data={"pe": pe, "peg": peg, "profit_margin": margin,
                  "revenue_growth": growth, "roe": roe, "metrics_present": present},
        )

    @staticmethod
    def _summary(score: float) -> str:
        if score > 0.3:
            return "Fundamentals are attractive: reasonable valuation and solid business quality."
        if score < -0.3:
            return "Fundamentals are poor: stretched valuation and/or weak profitability and growth."
        return "Fundamentals are mixed — some strengths offset by some concerns."
