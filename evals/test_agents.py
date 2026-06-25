"""Agent-level evals — each specialist points the right direction on known data."""

from app.agents import FundamentalAgent, RiskAgent, SentimentAgent, TechnicalAgent
from app.data.provider import FixtureProvider

provider = FixtureProvider()
AAPL = provider.fetch("AAPL")   # strong uptrend, fair value, positive sentiment, low risk
WEAK = provider.fetch("WEAK")   # downtrend, unprofitable, negative sentiment, high risk
TSLA = provider.fetch("TSLA")   # flat price, expensive, mixed sentiment, very high risk


def test_technical_bullish_on_uptrend_bearish_on_downtrend():
    up = TechnicalAgent().analyze(AAPL)
    down = TechnicalAgent().analyze(WEAK)
    assert up.score > 0.2, up
    assert down.score < -0.2, down
    assert up.confidence > 0.5


def test_fundamental_rewards_quality_penalizes_weakness():
    good = FundamentalAgent().analyze(AAPL)
    bad = FundamentalAgent().analyze(WEAK)
    expensive = FundamentalAgent().analyze(TSLA)
    assert good.score > bad.score
    assert bad.score < 0           # unprofitable + shrinking
    assert expensive.score < good.score  # rich valuation drags TSLA below AAPL


def test_sentiment_tracks_analyst_and_news():
    pos = SentimentAgent().analyze(AAPL)
    neg = SentimentAgent().analyze(WEAK)
    assert pos.score > 0.2
    assert neg.score < -0.2


def test_risk_flags_high_volatility_names():
    calm = RiskAgent().analyze(AAPL)
    wild = RiskAgent().analyze(TSLA)   # beta 2.25, high vol
    assert wild.score < calm.score
    assert wild.score < 0


def test_agents_degrade_gracefully_on_missing_fields():
    # WEAK has null P/E and PEG — agent must not crash and should lower confidence.
    res = FundamentalAgent().analyze(WEAK)
    assert res.confidence < 1.0
    assert res.signals  # still produced something usable
