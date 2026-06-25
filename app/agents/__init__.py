from app.agents.fundamental import FundamentalAgent
from app.agents.risk import RiskAgent
from app.agents.sentiment import SentimentAgent
from app.agents.technical import TechnicalAgent

# The committee of specialist agents the orchestrator dispatches to.
SPECIALIST_AGENTS = [
    TechnicalAgent(),
    FundamentalAgent(),
    SentimentAgent(),
    RiskAgent(),
]

__all__ = [
    "TechnicalAgent",
    "FundamentalAgent",
    "SentimentAgent",
    "RiskAgent",
    "SPECIALIST_AGENTS",
]
