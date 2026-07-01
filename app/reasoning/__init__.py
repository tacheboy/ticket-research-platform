"""v2 reasoning pipeline — an optional, OpenAI-backed agentic experiment. (v2 addition)

This package runs *alongside* the deterministic engine and never changes it. It exists
to compare an LLM that plans, calls tools, reflects, retries, and decides against the
platform's deterministic ground truth.
"""

from app.reasoning.agent import ReasoningAgent
from app.reasoning.experiment import run_experiment

__all__ = ["ReasoningAgent", "run_experiment"]
