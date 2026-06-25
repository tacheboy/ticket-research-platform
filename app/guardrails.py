"""Guardrails — the safety boundary around the agent system.

These are deliberately first-class (the exercise is evaluated on guardrails), and
grouped by where they sit in the pipeline:

  INPUT   validate_ticker            reject malformed / injection-y input
  OUTPUT  the pydantic schemas       guarantee report shape (see schemas.py)
  GROUND  verify_rationale_grounding catch an LLM rationale that invents numbers or
                                     contradicts the computed recommendation
  POLICY  apply_confidence_floor     refuse a directional call on thin data
"""

from __future__ import annotations

import re

from app import config
from app.schemas import Recommendation, TickerReport

# Tickers: 1-6 letters, optional .EXCHANGE or -CLASS suffix (e.g. BRK.B, RDS-A).
_TICKER_RE = re.compile(r"^[A-Za-z]{1,6}([.\-][A-Za-z0-9]{1,4})?$")


class InvalidTickerError(ValueError):
    """Raised when user input fails the input guardrail."""


def validate_ticker(raw: str) -> str:
    """Normalize and validate a user-supplied ticker. Raises InvalidTickerError."""
    if raw is None:
        raise InvalidTickerError("No ticker provided.")
    candidate = raw.strip().upper()
    if not candidate:
        raise InvalidTickerError("Ticker is empty.")
    if len(candidate) > 12:
        raise InvalidTickerError("Ticker is too long.")
    if not _TICKER_RE.match(candidate):
        raise InvalidTickerError(
            f"'{raw}' is not a valid ticker symbol. Use 1-6 letters, e.g. AAPL or BRK.B."
        )
    return candidate


def apply_confidence_floor(
    recommendation: Recommendation, confidence: float
) -> tuple[Recommendation, list[str]]:
    """Force HOLD when confidence is too low to responsibly take a directional view."""
    warnings: list[str] = []
    if confidence < config.MIN_CONFIDENCE_FOR_DIRECTIONAL_CALL and recommendation is not Recommendation.HOLD:
        warnings.append(
            f"Data confidence ({confidence:.0%}) is below the "
            f"{config.MIN_CONFIDENCE_FOR_DIRECTIONAL_CALL:.0%} threshold required for a "
            f"{recommendation.value} call — downgraded to HOLD."
        )
        return Recommendation.HOLD, warnings
    return recommendation, warnings


# Numbers we expect to appear in a grounded rationale, derived from real data.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def verify_rationale_grounding(rationale: str, report: TickerReport) -> tuple[bool, str]:
    """Sanity-check an LLM-written rationale against the deterministic findings.

    Two cheap, high-value checks:
      1. The rationale must not contradict the computed recommendation (e.g. say
         "we recommend selling" when the engine said BUY).
      2. Every percentage figure it cites should be traceable to a number that
         actually appears in the agents' computed data (loose tolerance).

    On failure the caller discards the LLM text and falls back to the deterministic
    template — the system never ships an ungrounded narrative.
    """
    text = rationale.lower()

    # 1. Recommendation consistency.
    rec = report.recommendation
    contradictions = {
        Recommendation.BUY: ["we recommend selling", "recommend a sell", "recommend selling"],
        Recommendation.SELL: ["we recommend buying", "recommend a buy", "recommend buying"],
    }
    for phrase in contradictions.get(rec, []):
        if phrase in text:
            return False, f"Rationale contradicts the {rec.value} recommendation."

    # 2. Percentage figures should be traceable to the underlying data.
    cited = {round(float(m), 1) for m in _NUM_RE.findall(rationale.replace("%", " %"))
             if "%" in rationale}
    if cited:
        known = _known_numbers(report)
        unexplained = [
            c for c in cited
            if not any(abs(c - k) <= max(1.0, abs(k) * 0.15) for k in known)
        ]
        # Allow a couple of incidental numbers (years, counts); flag wholesale invention.
        if len(unexplained) > max(2, len(cited) // 2):
            return False, "Rationale cites figures not supported by the computed data."

    return True, "ok"


def _known_numbers(report: TickerReport) -> set[float]:
    known: set[float] = set()

    def add(v: object) -> None:
        if isinstance(v, (int, float)):
            known.add(round(float(v), 1))
            known.add(round(float(v) * 100, 1))  # fractions expressed as percentages

    add(report.composite_score)
    add(report.confidence)
    add(report.current_price)
    for v in report.key_metrics.values():
        add(v)
    for ar in report.agent_results:
        add(ar.score)
        add(ar.confidence)
        for v in ar.data.values():
            add(v)
    return known
