"""Optional LLM narrative layer (Anthropic Claude).

The platform is fully functional without this. When ``ANTHROPIC_API_KEY`` is set,
the synthesizer asks Claude to turn the *already-decided* recommendation and the
computed signals into a readable rationale. Claude does NOT make the call or the
math — it only narrates grounded facts, and its output is verified by
``guardrails.verify_rationale_grounding`` before being used.
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
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=600,
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    "Write the rationale for this analysis. Use only these facts:\n\n"
                    + json.dumps(context, indent=2, default=str)
                ),
            }],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        text = "\n".join(parts).strip()
        return text or None
    except Exception:  # noqa: BLE001 - LLM is strictly optional; degrade silently
        return None
