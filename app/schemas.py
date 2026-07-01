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
        description="'llm' if narrated by OpenAI, else 'deterministic' template.",
    )

    agent_results: list[AgentResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_source: str = "fixture"
    disclaimer: str = ""


# === v2 schemas ===
# Contracts for the optional LLM *reasoning experiment* (app/reasoning/). This runs
# in parallel to the deterministic engine above and never alters it. The deterministic
# TickerReport remains the platform's ground-truth public contract.


class ToolCallTrace(BaseModel):
    """One dynamic tool call the reasoning agent made, with its result/ error."""

    step: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ReasoningStep(BaseModel):
    """A labeled phase of the agent loop (plan / reflect), for transparency."""

    phase: str = Field(description="'plan' | 'decide' | 'reflect'.")
    content: str = ""


class ExperimentComparison(BaseModel):
    """How the LLM's independent call lines up with the deterministic ground truth."""

    deterministic_recommendation: Recommendation
    reasoning_recommendation: Recommendation
    agree: bool
    deterministic_composite: float
    reasoning_score: float
    score_delta: float = Field(description="reasoning_score - deterministic_composite.")
    confidence_delta: float = Field(description="reasoning_conf - deterministic_conf.")
    note: str = ""


class ReasoningReport(BaseModel):
    """Output of the v2 reasoning pipeline — the LLM's independent, agentic analysis."""

    ticker: str
    company_name: str
    as_of: str

    # The agent's own independent call (NOT used for the deterministic recommendation).
    recommendation: Recommendation
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_score: float = Field(ge=-1.0, le=1.0)
    reasoning: str = ""

    # Agentic transparency: plan, the dynamic tool calls it chose, and its self-critique.
    plan: str = ""
    tool_trace: list[ToolCallTrace] = Field(default_factory=list)
    steps: list[ReasoningStep] = Field(default_factory=list)
    reflection: str = ""

    # Cost-aware routing telemetry.
    model_used: str = ""
    escalated: bool = False
    retry_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    # Side-by-side vs the deterministic engine.
    comparison: ExperimentComparison
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = ""
