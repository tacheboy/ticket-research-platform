"""Force deterministic, offline fixture data for the whole eval suite."""

import os

os.environ["TICKER_DATA_SOURCE"] = "fixture"
os.environ.pop("ANTHROPIC_API_KEY", None)  # evals never hit the network/LLM
