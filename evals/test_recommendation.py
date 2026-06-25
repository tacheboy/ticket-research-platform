"""End-to-end recommendation evals — golden scenarios + invariants."""

from app.orchestrator import Orchestrator
from app.schemas import Recommendation, Sentiment

orch = Orchestrator(data_source="fixture")


def test_strong_name_is_buy():
    r = orch.analyze("AAPL", use_llm=False)
    assert r.recommendation is Recommendation.BUY
    assert r.sentiment is Sentiment.BULLISH
    assert r.composite_score > 0


def test_weak_name_is_sell():
    r = orch.analyze("WEAK", use_llm=False)
    assert r.recommendation is Recommendation.SELL
    assert r.sentiment is Sentiment.BEARISH
    assert r.composite_score < 0


def test_mixed_name_is_not_a_strong_buy():
    # TSLA: flat price, expensive, very high risk -> should not be a BUY.
    r = orch.analyze("TSLA", use_llm=False)
    assert r.recommendation in (Recommendation.HOLD, Recommendation.SELL)


def test_determinism_same_input_same_output():
    a = orch.analyze("AAPL", use_llm=False)
    b = orch.analyze("AAPL", use_llm=False)
    assert a.recommendation == b.recommendation
    assert a.composite_score == b.composite_score
    assert a.confidence == b.confidence


def test_report_contains_all_four_agents():
    r = orch.analyze("AAPL", use_llm=False)
    agents = {ar.agent for ar in r.agent_results}
    assert agents == {"technical", "fundamental", "sentiment", "risk"}


def test_rationale_is_deterministic_without_key():
    r = orch.analyze("AAPL", use_llm=False)
    assert r.rationale_source == "deterministic"
    assert r.recommendation.value in r.rationale
