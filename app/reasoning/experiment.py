"""Experiment runner — deterministic ground truth vs the LLM reasoning agent. (v2 addition)

Fetches data once, runs BOTH pipelines on it, and assembles a ``ReasoningReport`` that
puts the LLM's independent call side-by-side with the deterministic engine's verdict.
The deterministic ``Orchestrator`` is used unchanged and remains the source of truth.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app import config, guardrails
from app.data.provider import get_provider
from app.orchestrator import Orchestrator
from app.reasoning.agent import ReasoningAgent
from app.reasoning.openai_client import (
    CostMeter, LLMUnavailableError, OpenAIClient, is_available,
)
from app.schemas import (
    ExperimentComparison, ReasoningReport, ReasoningStep, ToolCallTrace,
)


def run_experiment(ticker: str, data_source: str | None = None, client=None) -> ReasoningReport:
    """Run the v2 reasoning experiment for ``ticker``.

    ``client`` is injectable so the eval suite can pass a scripted, offline FakeClient.
    In production it is created here (and raises LLMUnavailableError without a key).
    """
    ticker = guardrails.validate_ticker(ticker)  # reuse the input guardrail

    source = data_source or config.DATA_SOURCE
    data = get_provider(source).fetch(ticker)  # single fetch feeds both pipelines

    # Deterministic ground truth — unchanged engine, no LLM narrative needed here.
    deterministic = Orchestrator(data_source=source)._build_report(data, use_llm=False)

    if client is None:
        if not is_available():
            raise LLMUnavailableError("OPENAI_API_KEY is not set.")
        client = OpenAIClient(meter=CostMeter(budget_usd=config.REASONING_BUDGET_USD))

    result = ReasoningAgent(client).run(data)
    decision = result["decision"]

    comparison = ExperimentComparison(
        deterministic_recommendation=deterministic.recommendation,
        reasoning_recommendation=decision["recommendation"],
        agree=deterministic.recommendation == decision["recommendation"],
        deterministic_composite=deterministic.composite_score,
        reasoning_score=decision["score"],
        score_delta=round(decision["score"] - deterministic.composite_score, 3),
        confidence_delta=round(decision["confidence"] - deterministic.confidence, 3),
        note=_comparison_note(deterministic.recommendation, decision["recommendation"]),
    )

    steps = [ReasoningStep(phase="plan", content=result["plan"])]
    if result["reflection"].get("critique"):
        steps.append(ReasoningStep(phase="reflect", content=result["reflection"]["critique"]))

    return ReasoningReport(
        ticker=deterministic.ticker,
        company_name=deterministic.company_name,
        as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        recommendation=decision["recommendation"],
        sentiment=decision["sentiment"],
        confidence=decision["confidence"],
        reasoning_score=decision["score"],
        reasoning=decision["reasoning"],
        plan=result["plan"],
        tool_trace=[ToolCallTrace(**t) for t in result["tool_trace"]],
        steps=steps,
        reflection=result["reflection"].get("critique", ""),
        model_used=result["model_used"],
        escalated=result["escalated"],
        retry_count=result["retry_count"],
        tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"],
        cost_usd=result["cost_usd"],
        comparison=comparison,
        warnings=list(data.warnings),
        disclaimer=config.DISCLAIMER,
    )


def _comparison_note(det, llm) -> str:
    if det == llm:
        return "The LLM agent independently reached the same call as the deterministic engine."
    return (
        f"Divergence: deterministic engine says {det.value}, the LLM agent says {llm.value}. "
        "The deterministic engine remains the system's authoritative recommendation."
    )
