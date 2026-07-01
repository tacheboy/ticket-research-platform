"""Force deterministic, offline fixture data for the whole eval suite."""

import os

os.environ["TICKER_DATA_SOURCE"] = "fixture"
# v2: the LLM layer is OpenAI now — strip its key so evals never hit the network/LLM.
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)
