"""Guardrail evals — input validation, confidence floor, schema, LLM grounding."""

import pytest

from app.guardrails import (
    InvalidTickerError, apply_confidence_floor, validate_ticker,
    verify_rationale_grounding,
)
from app.orchestrator import Orchestrator
from app.schemas import Recommendation, TickerReport


# --- Input guardrail ---------------------------------------------------------
@pytest.mark.parametrize("good", ["AAPL", "aapl", " msft ", "BRK.B", "RDS-A", "F"])
def test_valid_tickers_accepted(good):
    assert validate_ticker(good) == good.strip().upper()


@pytest.mark.parametrize("bad", ["", "   ", "TOOLONGSYMBOL", "A B", "123;DROP", "<script>", "$$$"])
def test_invalid_tickers_rejected(bad):
    with pytest.raises(InvalidTickerError):
        validate_ticker(bad)


def test_orchestrator_rejects_bad_input():
    orch = Orchestrator(data_source="fixture")
    with pytest.raises(InvalidTickerError):
        orch.analyze("not a ticker!!")


# --- Policy guardrail: confidence floor --------------------------------------
def test_confidence_floor_downgrades_to_hold():
    rec, warns = apply_confidence_floor(Recommendation.BUY, confidence=0.10)
    assert rec is Recommendation.HOLD
    assert warns and "downgraded to HOLD" in warns[0]


def test_confidence_floor_leaves_confident_calls_alone():
    rec, warns = apply_confidence_floor(Recommendation.BUY, confidence=0.90)
    assert rec is Recommendation.BUY
    assert warns == []


# --- Output guardrail: schema validity ---------------------------------------
def test_report_always_validates_against_schema():
    orch = Orchestrator(data_source="fixture")
    for ticker in ("AAPL", "TSLA", "WEAK"):
        report = orch.analyze(ticker, use_llm=False)
        # Re-validation round-trips cleanly -> output contract holds.
        TickerReport.model_validate(report.model_dump())
        assert -1.0 <= report.composite_score <= 1.0
        assert 0.0 <= report.confidence <= 1.0
        assert report.disclaimer  # compliance text always present


# --- Grounding guardrail -----------------------------------------------------
def _report():
    return Orchestrator(data_source="fixture").analyze("AAPL", use_llm=False)


def test_grounding_accepts_consistent_rationale():
    r = _report()
    text = f"We rate {r.ticker} a {r.recommendation.value}; the technical and fundamental reads support this."
    ok, _ = verify_rationale_grounding(text, r)
    assert ok


def test_grounding_rejects_contradictory_recommendation():
    r = _report()
    r.recommendation = Recommendation.BUY
    ok, reason = verify_rationale_grounding("Overall we recommend selling this name.", r)
    assert not ok and "contradict" in reason.lower()


def test_grounding_rejects_invented_numbers():
    r = _report()
    bad = "Revenue is up 999.0% with a 888.0% margin and 777.0% ROE and 666.0% upside."
    ok, _ = verify_rationale_grounding(bad, r)
    assert not ok
