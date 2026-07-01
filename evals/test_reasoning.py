"""v2 reasoning-pipeline evals — fully offline via a scripted FakeClient.

No network and no API key: a FakeClient implements the same surface as OpenAIClient
(`.chat(...)` + `.meter`) and returns scripted plan/tool-call/decision/reflection
responses. This lets us assert the agentic loop, the cheap->strong escalation policy,
the cost meter + budget cap, and the comparison/schema — all deterministically.
"""

import json

import pytest

from app import config
from app.data.provider import FixtureProvider
from app.reasoning import tools
from app.reasoning.agent import ReasoningAgent
from app.reasoning.experiment import run_experiment
from app.reasoning.openai_client import (
    BudgetExceededError, CostMeter, LLMResponse, ToolCall, model_for_tier,
)
from app.schemas import ReasoningReport, Recommendation

AAPL = FixtureProvider().fetch("AAPL")

_TOOLSEQ = ["get_price_technicals", "get_fundamentals", "get_sentiment_inputs", "get_risk_metrics"]


class FakeClient:
    """Scripted, offline stand-in for OpenAIClient."""

    def __init__(self, decision: dict, cheap_conf: float = 0.9, strong_conf: float = 0.95,
                 budget_usd: float = 0.0):
        self.decision = decision
        self.cheap_conf = cheap_conf
        self.strong_conf = strong_conf
        self.meter = CostMeter(budget_usd=budget_usd)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, tier="cheap", force_json=False) -> LLMResponse:
        model = model_for_tier(tier)
        self.calls.append({"tier": tier, "tools": bool(tools), "json": force_json})
        self.meter.record(model, 100, 50)  # fixed usage per call -> deterministic cost

        last_user = next((m["content"] for m in reversed(messages)
                          if m.get("role") == "user"), "")
        system = messages[0].get("content", "")

        # Reflection phase (critic system + json)
        if force_json and "critical reviewer" in system.lower():
            conf = self.strong_conf if tier == "strong" else self.cheap_conf
            return self._json({"self_confidence": conf, "consistent": conf >= 0.55,
                               "critique": f"Reviewed at {tier} tier."}, model)

        # Decision phase
        if force_json:
            return self._json(self.decision, model)

        # Act phase: first tool-enabled call returns the batch of tool calls; once tool
        # results are present in the transcript, stop calling tools.
        if tools:
            if any(m.get("role") == "tool" for m in messages):
                return LLMResponse("Gathered enough.", [], model, 100, 50)
            tcs = [ToolCall(id=f"c{i}", name=name, arguments={"ticker": AAPL.ticker})
                   for i, name in enumerate(_TOOLSEQ)]
            return LLMResponse(None, tcs, model, 100, 50)

        # Plan phase
        return LLMResponse("Plan: weigh momentum, valuation, sentiment, and risk.",
                           [], model, 100, 50)

    @staticmethod
    def _json(obj: dict, model: str) -> LLMResponse:
        return LLMResponse(json.dumps(obj), [], model, 100, 50)


_BUY_DECISION = {
    "recommendation": "BUY", "sentiment": "Bullish", "confidence": 0.8,
    "score": 0.5, "reasoning": "Uptrend, healthy margins, supportive analysts.",
}


# --- Tools dispatch -----------------------------------------------------------
def test_tool_dispatch_returns_real_fixture_numbers():
    res = tools.dispatch("get_price_technicals", {"ticker": "AAPL"}, AAPL)
    assert "error" not in res
    assert res["current_price"] > 100          # AAPL fixture ends ~212
    assert res["sma50"] and res["sma200"]
    fund = tools.dispatch("get_fundamentals", {"ticker": "AAPL"}, AAPL)
    assert fund["pe"] == 28.5


def test_dispatch_unknown_tool_is_soft_error():
    res = tools.dispatch("nope", {}, AAPL)
    assert "error" in res


# --- Full loop / schema -------------------------------------------------------
def test_experiment_builds_valid_report_without_escalation():
    client = FakeClient(_BUY_DECISION, cheap_conf=0.9)
    report = run_experiment("AAPL", client=client)

    assert isinstance(report, ReasoningReport)
    ReasoningReport.model_validate(report.model_dump())  # schema round-trips
    assert report.recommendation is Recommendation.BUY
    assert len(report.tool_trace) == len(_TOOLSEQ)
    assert report.escalated is False and report.retry_count == 0
    assert report.model_used == config.LLM_TIER_CHEAP
    assert report.cost_usd > 0
    # Comparison is wired to the deterministic engine (AAPL -> BUY), so they agree.
    assert report.comparison.deterministic_recommendation is Recommendation.BUY
    assert report.comparison.agree is True


# --- Escalation policy --------------------------------------------------------
def test_low_reflection_confidence_triggers_escalation():
    client = FakeClient(_BUY_DECISION, cheap_conf=0.30, strong_conf=0.95)
    report = run_experiment("AAPL", client=client)
    assert report.escalated is True
    assert report.retry_count == 1
    assert report.model_used == config.LLM_TIER_STRONG
    assert any(c["tier"] == "strong" for c in client.calls)


def test_confident_cheap_pass_does_not_escalate():
    client = FakeClient(_BUY_DECISION, cheap_conf=0.85)
    report = run_experiment("AAPL", client=client)
    assert report.escalated is False
    assert all(c["tier"] == "cheap" for c in client.calls)


# --- Cost meter + budget cap --------------------------------------------------
def test_cost_meter_sums_tokens_and_usd():
    m = CostMeter()
    m.record("gpt-4o-mini", 1000, 1000)
    assert m.tokens_in == 1000 and m.tokens_out == 1000
    assert m.cost_usd == pytest.approx(0.00015 + 0.00060, rel=1e-6)


def test_budget_cap_raises():
    m = CostMeter(budget_usd=0.0001)
    with pytest.raises(BudgetExceededError):
        for _ in range(10):
            m.record("gpt-4o", 1000, 1000)


def test_run_aborts_cleanly_when_budget_exceeded():
    client = FakeClient(_BUY_DECISION, budget_usd=0.00001)  # blows almost immediately
    with pytest.raises(BudgetExceededError):
        ReasoningAgent(client).run(AAPL)
