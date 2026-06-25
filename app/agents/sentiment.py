"""Sentiment agent — analyst consensus + news tone.

Two grounded inputs:
  * Analyst consensus: recommendationMean (1=Strong Buy .. 5=Strong Sell) and the
    implied upside to the mean price target.
  * News tone: a transparent keyword lexicon over recent headlines. (Deterministic
    on purpose so evals are stable; the LLM layer can enrich the *narrative* but does
    not change this score.)
"""

from __future__ import annotations

from app.agents.base import Agent, clamp, scale
from app.data.provider import MarketData
from app.schemas import AgentResult, Signal

_POSITIVE = {
    "beat", "beats", "upgrade", "upgrades", "raise", "raised", "raises", "record",
    "strong", "growth", "optimism", "optimistic", "buyback", "surge", "rally",
    "outperform", "robust", "demand", "increase", "bullish", "gains", "boost", "wins",
}
_NEGATIVE = {
    "miss", "misses", "downgrade", "downgrades", "cut", "cuts", "weak", "weakness",
    "loss", "losses", "warn", "warns", "warning", "decline", "falls", "fall", "layoffs",
    "restructuring", "pressure", "risk", "lawsuit", "probe", "bearish", "slump", "plunge",
    "struggles", "struggling", "debt",
}


def _score_headline(title: str) -> int:
    words = {w.strip(".,:;!?'\"()").lower() for w in title.split()}
    return len(words & _POSITIVE) - len(words & _NEGATIVE)


class SentimentAgent(Agent):
    name = "sentiment"

    def analyze(self, data: MarketData) -> AgentResult:
        signals: list[Signal] = []
        contributions: list[tuple[float, float]] = []
        coverage = 0.0

        a = data.analyst or {}
        rec_mean = a.get("recommendation_mean")
        if isinstance(rec_mean, (int, float)) and rec_mean > 0:
            coverage += 0.5
            # 1=Strong Buy .. 5=Strong Sell -> map to [+1 .. -1] around 3 (Hold).
            contrib = clamp((3.0 - rec_mean) / 2.0)
            n = a.get("num_analysts")
            signals.append(Signal(
                name="Analyst consensus", value=f"{rec_mean:.1f}/5" + (f" ({n} analysts)" if n else ""),
                interpretation="Buy-leaning consensus." if rec_mean < 2.5 else
                               ("Sell-leaning consensus." if rec_mean > 3.5 else "Hold-leaning consensus."),
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.45))

        target = a.get("target_mean")
        price = data.current_price
        if isinstance(target, (int, float)) and target > 0 and price:
            coverage += 0.25
            upside = (target - price) / price
            contrib = scale(upside, 0.0, 0.25)
            signals.append(Signal(
                name="Price-target upside", value=f"{upside:+.1%} to {target:.2f}",
                interpretation="Implied upside." if upside > 0 else "Trading above mean target.",
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.25))

        news = data.news or []
        if news:
            coverage += 0.25
            raw = [_score_headline(item.get("title", "")) for item in news]
            avg = sum(raw) / len(raw)
            contrib = clamp(avg / 1.5)
            signals.append(Signal(
                name="News tone", value=f"{len(news)} headlines, net {('+' if avg>=0 else '')}{avg:.1f}",
                interpretation="Positive coverage." if avg > 0.2 else
                               ("Negative coverage." if avg < -0.2 else "Balanced coverage."),
                contribution=round(contrib, 3),
            ))
            contributions.append((contrib, 0.30))

        if not contributions:
            return self._empty("no analyst or news data available")

        total_w = sum(w for _, w in contributions)
        score = clamp(sum(c * w for c, w in contributions) / total_w)
        confidence = clamp(coverage, 0.0, 1.0) * 0.85 + 0.1

        return AgentResult(
            agent=self.name, score=round(score, 3), confidence=round(confidence, 3),
            signals=signals,
            summary=self._summary(score),
            data={"recommendation_mean": rec_mean, "target_mean": target,
                  "news_count": len(news)},
        )

    @staticmethod
    def _summary(score: float) -> str:
        if score > 0.3:
            return "Sentiment is positive: analysts and news flow lean supportive."
        if score < -0.3:
            return "Sentiment is negative: analyst consensus and headlines are cautious."
        return "Sentiment is neutral/mixed across analysts and news."
