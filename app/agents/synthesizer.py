"""Synthesizer — turns the committee's scores into one decision.

This is the orchestration core: it combines the specialist scores into a composite,
maps that to Buy/Sell/Hold, derives an overall confidence (which penalizes
disagreement), and produces a deterministic rationale that always works even with no
LLM. It returns plain values; the orchestrator assembles the final report and applies
the remaining guardrails.
"""

from __future__ import annotations

from statistics import pstdev

from app import config
from app.schemas import AgentResult, Recommendation, Sentiment


def synthesize(results: list[AgentResult]) -> dict:
    usable = [r for r in results if r.confidence > 0]

    if not usable:
        return {
            "composite_score": 0.0, "confidence": 0.0,
            "recommendation": Recommendation.HOLD, "sentiment": Sentiment.NEUTRAL,
            "rationale": "No agent could produce a usable read; defaulting to HOLD.",
        }

    # Composite = configured weight * agent confidence, normalized.
    num = den = 0.0
    for r in usable:
        w = config.AGENT_WEIGHTS.get(r.agent, 0.0) * r.confidence
        num += r.score * w
        den += w
    composite = num / den if den else 0.0

    # Overall confidence: average agent confidence, scaled down by disagreement.
    avg_conf = sum(r.confidence for r in usable) / len(usable)
    coverage = sum(config.AGENT_WEIGHTS.get(r.agent, 0.0) for r in usable)
    disagreement = pstdev([r.score for r in usable]) if len(usable) > 1 else 0.0
    agreement_factor = max(0.5, 1.0 - disagreement)  # wide spread -> lower confidence
    confidence = max(0.0, min(1.0, avg_conf * coverage * agreement_factor))

    if composite >= config.BUY_THRESHOLD:
        recommendation = Recommendation.BUY
    elif composite <= config.SELL_THRESHOLD:
        recommendation = Recommendation.SELL
    else:
        recommendation = Recommendation.HOLD

    if composite > 0.15:
        sentiment = Sentiment.BULLISH
    elif composite < -0.15:
        sentiment = Sentiment.BEARISH
    else:
        sentiment = Sentiment.NEUTRAL

    rationale = _deterministic_rationale(recommendation, composite, confidence, usable, disagreement)

    return {
        "composite_score": round(composite, 3),
        "confidence": round(confidence, 3),
        "recommendation": recommendation,
        "sentiment": sentiment,
        "rationale": rationale,
    }


def _deterministic_rationale(
    rec: Recommendation, composite: float, confidence: float,
    results: list[AgentResult], disagreement: float,
) -> str:
    by_name = {r.agent: r for r in results}
    order = ["technical", "fundamental", "sentiment", "risk"]
    lines = [
        f"Recommendation: {rec.value} (composite score {composite:+.2f}, "
        f"confidence {confidence:.0%}).",
    ]
    drivers = sorted(
        results,
        key=lambda r: abs(r.score) * config.AGENT_WEIGHTS.get(r.agent, 0.0),
        reverse=True,
    )
    top = drivers[0]
    lines.append(
        f"The call is driven most by the {top.agent} read ({top.score:+.2f}). "
    )
    for name in order:
        r = by_name.get(name)
        if r:
            lines.append(f"- {name.capitalize()}: {r.summary} (score {r.score:+.2f})")
    if disagreement > 0.5:
        lines.append(
            "Note: the specialists disagree meaningfully, which lowers conviction and "
            "argues for a more cautious stance."
        )
    return "\n".join(lines)
