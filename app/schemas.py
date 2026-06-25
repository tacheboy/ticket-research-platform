"""Typed schemas for agent outputs and the final report.

Every agent emits an ``AgentResult`` and the orchestrator emits a ``TickerReport``.
Using pydantic here is a deliberate guardrail: it means malformed agent output is
caught at the boundary rather than silently producing a garbage recommendation, and
the final report is guaranteed to match a stable, documented contract that the API,
the PDF generator, and the evals can all rely on.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Recommendation(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Sentiment(str, Enum):
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"


class Signal(BaseModel):
    """A single, human-readable factor an agent considered."""

    name: str
    value: str = Field(description="The observed value, formatted for display.")
    interpretation: str = Field(description="What this signal implies, in plain words.")
    contribution: float = Field(
        ge=-1.0, le=1.0,
        description="Signed contribution of this factor to the agent's score.",
    )


class AgentResult(BaseModel):
    """Structured output of one specialist agent."""

    agent: str
    score: float = Field(
        ge=-1.0, le=1.0,
        description="Bullish/bearish score: +1 strongly bullish, -1 strongly bearish.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How reliable this agent's read is, given data coverage/quality.",
    )
    signals: list[Signal] = Field(default_factory=list)
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class TickerReport(BaseModel):
    """The final investment report — the public contract of the platform."""

    ticker: str
    company_name: str
    currency: str = "USD"
    as_of: str = Field(description="UTC timestamp the report was generated.")

    current_price: float | None = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)

    recommendation: Recommendation
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=-1.0, le=1.0)

    rationale: str
    rationale_source: str = Field(
        default="deterministic",
        description="'llm' if narrated by Claude, else 'deterministic' template.",
    )

    agent_results: list[AgentResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_source: str = "fixture"
    disclaimer: str = ""
