"""Risk agent — volatility and downside exposure.

Unlike the other agents, a high reading here is *bad*: the score is bearish-signed,
so elevated volatility, beta, drawdown, or micro-cap/illiquidity all push the
composite toward caution. The orchestrator also uses this agent's magnitude to
temper overall confidence.
"""

from __future__ import annotations

from app import indicators as ind
from app.agents.base import Agent, clamp, scale
from app.data.provider import MarketData
from app.schemas import AgentResult, Signal


class RiskAgent(Agent):
    name = "risk"

    def analyze(self, data: MarketData) -> AgentResult:
        signals: list[Signal] = []
        contributions: list[tuple[float, float]] = []
        present = 0

        closes = data.closes
        vol = ind.annualized_volatility(closes) if closes else None
        if vol is not None:
            present += 1
            # ~25% annualized vol is typical; higher is riskier (negative contribution).
            contrib = -scale(vol, 0.30, 0.30)
            signals.append(Signal(
                name="Annualized volatility", value=f"{vol:.0%}",
                interpretation="Low/typical volatility." if vol < 0.30 else
                               ("Very high volatility." if vol > 0.55 else "Elevated volatility."),
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.35))

        dd = ind.max_drawdown(closes) if closes else None
        if dd is not None:
            present += 1
            contrib = -scale(-dd, 0.25, 0.30)  # dd is negative; deeper -> more risk
            signals.append(Signal(
                name="Max drawdown (1y)", value=f"{dd:.0%}",
                interpretation="Contained drawdown." if dd > -0.25 else "Severe peak-to-trough decline.",
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.25))

        beta = (data.risk or {}).get("beta")
        if isinstance(beta, (int, float)) and beta > 0:
            present += 1
            contrib = -scale(beta, 1.0, 1.0)  # beta 1 = market; >2 = twice as jumpy
            signals.append(Signal(
                name="Beta", value=f"{beta:.2f}",
                interpretation="Defensive vs market." if beta < 1 else
                               ("High market sensitivity." if beta > 1.5 else "Tracks the market."),
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.25))

        mcap = (data.fundamentals or {}).get("market_cap")
        if isinstance(mcap, (int, float)) and mcap > 0:
            present += 1
            # Small caps carry liquidity/idiosyncratic risk; mega caps are safer.
            if mcap < 2e9:
                contrib, interp = -0.6, "Small-cap — higher idiosyncratic/liquidity risk."
            elif mcap < 1e10:
                contrib, interp = -0.2, "Mid-cap — moderate risk."
            else:
                contrib, interp = 0.2, "Large/mega-cap — lower liquidity risk."
            signals.append(Signal(
                name="Market cap", value=f"${mcap/1e9:.1f}B",
                interpretation=interp, contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.15))

        if not contributions:
            return self._empty("no risk metrics available")

        total_w = sum(w for _, w in contributions)
        score = clamp(sum(c * w for c, w in contributions) / total_w)
        confidence = clamp(present / 4.0, 0.0, 1.0) * 0.9 + 0.1

        return AgentResult(
            agent=self.name, score=round(score, 3), confidence=round(confidence, 3),
            signals=signals,
            summary=self._summary(score),
            data={"volatility": vol, "max_drawdown": dd, "beta": beta, "market_cap": mcap},
        )

    @staticmethod
    def _summary(score: float) -> str:
        if score > 0.2:
            return "Risk profile is benign: contained volatility and downside."
        if score < -0.3:
            return "Risk profile is elevated: high volatility, beta, or drawdown warrant caution."
        return "Risk profile is moderate."
