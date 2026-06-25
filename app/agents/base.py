"""Base class and shared math for specialist agents.

Design choice: each specialist computes a **deterministic** score from real data.
The recommendation a user acts on must be reproducible and auditable — letting an
LLM free-hand the buy/sell number would be unevaluable and prone to hallucination.
The optional LLM layer (see ``app/llm.py``) only narrates these computed facts.
"""

from __future__ import annotations

from app.data.provider import MarketData
from app.schemas import AgentResult


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def scale(value: float, neutral: float, span: float) -> float:
    """Map ``value`` to roughly [-1, 1]: equals ``neutral`` -> 0, +/- ``span`` -> +/-1."""
    if span == 0:
        return 0.0
    return clamp((value - neutral) / span)


class Agent:
    """Interface: take normalized market data, return a structured result."""

    name: str = "agent"

    def analyze(self, data: MarketData) -> AgentResult:  # pragma: no cover - interface
        raise NotImplementedError

    def _empty(self, reason: str) -> AgentResult:
        return AgentResult(
            agent=self.name, score=0.0, confidence=0.0,
            summary=f"No usable data: {reason}",
        )
