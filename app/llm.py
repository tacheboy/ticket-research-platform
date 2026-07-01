"""Optional LLM narrative layer. (narrator migrated Anthropic -> OpenAI, v2)

The platform is fully functional without this. When ``OPENAI_API_KEY`` is set, the
synthesizer asks OpenAI to turn the *already-decided* recommendation and the computed
signals into a readable rationale. The model does NOT make the call or the math — it
only narrates grounded facts, and its output is verified by
``guardrails.verify_rationale_grounding`` before being used.

v2 note: this used to call Anthropic Claude. It now routes through the shared OpenAI
client at the CHEAP tier — narration is an easy task, so paying for the strong model
here would be wasteful. Signature and silent-degrade behavior are unchanged.
"""

from __future__ import annotations

import json

from app import config


def is_available() -> bool:
    return config.HAS_API_KEY


_SYSTEM = (
    "You are a sell-side equity research writer. You are given a ticker, a "
    "pre-computed Buy/Sell/Hold recommendation, a confidence score, and the exact "
    "quantitative signals four specialist models produced. Write a concise, "
    "professional rationale (120-180 words) that EXPLAINS the given recommendation. "
    "Rules: (1) Do NOT change or second-guess the recommendation — it is final. "
    "(2) Only use numbers present in the provided data; never invent figures. "
    "(3) Reference the technical, fundamental, sentiment, and risk angles. "
    "(4) No hype, no guarantees, plain professional prose. Output only the rationale."
)


def generate_rationale(context: dict) -> str | None:
    """Return an LLM-written rationale, or None if unavailable / on any error."""
    if not is_available():
        return None
    try:
        # --- v2 --- route narration through the shared OpenAI client (cheap tier).
        from app.reasoning.openai_client import OpenAIClient

        client = OpenAIClient()
        resp = client.chat(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": (
                    "Write the rationale for this analysis. Use only these facts:\n\n"
                    + json.dumps(context, indent=2, default=str)
                )},
            ],
            tier="cheap",
        )
        text = (resp.content or "").strip()
        return text or None
        # --- end v2 ---
    except Exception:  # noqa: BLE001 - LLM is strictly optional; degrade silently
        return None
