"""The reasoning agent — plan → act → decide → reflect → retry/escalate. (v2 addition)

This is the genuinely *agentic* part of v2: an LLM that decides which tools to call and
in what order, forms its own Buy/Sell/Hold view, critiques that view, and — only when
the critique is weak — escalates from the cheap model to the strong one and tries again.

It is deliberately independent of the deterministic engine: it sees raw data tools, not
the deterministic scores. The comparison against ground truth happens in experiment.py.

The agent depends only on a duck-typed client exposing ``.chat(...)`` and ``.meter`` —
so the eval suite injects a scripted ``FakeClient`` and runs the whole loop offline.
"""

from __future__ import annotations

import json

from app import config
from app.reasoning.tools import TOOL_SCHEMAS, dispatch
from app.schemas import Recommendation, Sentiment

_SYSTEM = (
    "You are an autonomous equity-research reasoning agent. You analyze ONE ticker and "
    "must reach an INDEPENDENT Buy/Sell/Hold view from raw data you gather via tools. "
    "You have four tools: get_price_technicals, get_fundamentals, get_sentiment_inputs, "
    "get_risk_metrics. Decide for yourself which to call and in what order. Reason across "
    "momentum, valuation, sentiment, and risk. Be disciplined and avoid hype."
)

_PLAN_PROMPT = (
    "Ticker: {ticker}. In 2-3 sentences, outline your analysis plan: which lenses you will "
    "examine and what would make you bullish vs bearish. Do not call tools yet."
)

_ACT_PROMPT = (
    "Now gather the data you need by calling the tools. Call as many as you need, then stop "
    "calling tools when you have enough to decide."
)

_DECIDE_PROMPT = (
    "Using only the data you gathered, output your decision as JSON with EXACTLY these keys: "
    '{"recommendation": "BUY|SELL|HOLD", "sentiment": "Bullish|Neutral|Bearish", '
    '"confidence": 0.0-1.0, "score": -1.0-1.0 (your bullish/bearish score), '
    '"reasoning": "120-180 word rationale citing the numbers you saw"}.'
)

_REFLECT_SYSTEM = (
    "You are a critical reviewer. Judge whether a draft equity recommendation is well-"
    "supported by its own stated reasoning and internally consistent."
)

_REFLECT_PROMPT = (
    "Here is the draft decision:\n{decision}\n\nReturn JSON with EXACTLY these keys: "
    '{{"self_confidence": 0.0-1.0 (how well-supported the call is), '
    '"consistent": true|false (does the recommendation match the reasoning and score?), '
    '"critique": "one or two sentences"}}.'
)


def _assistant_toolcalls_msg(resp) -> dict:
    """Reconstruct an OpenAI-format assistant message carrying the model's tool calls."""
    return {
        "role": "assistant",
        "content": resp.content or None,
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in resp.tool_calls
        ],
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _parse_decision(content: str | None) -> tuple[dict, bool]:
    """Normalize the model's decision JSON into typed values. ok=False if unusable."""
    try:
        raw = json.loads(content or "{}")
    except json.JSONDecodeError:
        return _fallback_decision(), False

    rec_str = str(raw.get("recommendation", "")).strip().upper()
    sent_str = str(raw.get("sentiment", "")).strip().capitalize()
    try:
        recommendation = Recommendation(rec_str)
    except ValueError:
        return _fallback_decision(), False
    try:
        sentiment = Sentiment(sent_str)
    except ValueError:
        # Derive sentiment from the score rather than failing the whole decision.
        score_guess = _clamp(raw.get("score", 0.0), -1.0, 1.0)
        sentiment = (Sentiment.BULLISH if score_guess > 0.15 else
                     Sentiment.BEARISH if score_guess < -0.15 else Sentiment.NEUTRAL)

    return ({
        "recommendation": recommendation,
        "sentiment": sentiment,
        "confidence": _clamp(raw.get("confidence", 0.5), 0.0, 1.0),
        "score": _clamp(raw.get("score", 0.0), -1.0, 1.0),
        "reasoning": str(raw.get("reasoning", "")).strip(),
    }, True)


def _fallback_decision() -> dict:
    return {
        "recommendation": Recommendation.HOLD, "sentiment": Sentiment.NEUTRAL,
        "confidence": 0.0, "score": 0.0,
        "reasoning": "The reasoning agent could not produce a valid structured decision.",
    }


def _parse_reflection(content: str | None) -> dict:
    try:
        raw = json.loads(content or "{}")
    except json.JSONDecodeError:
        return {"self_confidence": 0.0, "consistent": False,
                "critique": "Reflection output was not valid JSON."}
    return {
        "self_confidence": _clamp(raw.get("self_confidence", 0.0), 0.0, 1.0),
        "consistent": bool(raw.get("consistent", False)),
        "critique": str(raw.get("critique", "")).strip(),
    }


class ReasoningAgent:
    def __init__(self, client):
        self.client = client

    def run(self, data) -> dict:
        """Run the full agentic loop; escalate cheap->strong if reflection is weak."""
        attempt = self._attempt(data, tier="cheap")
        reflection = self._reflect(attempt["decision"], tier="cheap")

        escalated = False
        retry_count = 0
        final_tier = "cheap"

        if self._needs_escalation(attempt, reflection) and config.MAX_ESCALATIONS > 0:
            escalated = True
            retry_count = 1
            attempt = self._attempt(data, tier="strong")
            reflection = self._reflect(attempt["decision"], tier="strong")
            final_tier = "strong"

        d = attempt["decision"]
        return {
            "plan": attempt["plan"],
            "tool_trace": attempt["tool_trace"],
            "decision": d,
            "reflection": reflection,
            "escalated": escalated,
            "retry_count": retry_count,
            "model_used": _model(final_tier),
            "tokens_in": self.client.meter.tokens_in,
            "tokens_out": self.client.meter.tokens_out,
            "cost_usd": self.client.meter.cost_usd,
        }

    # -- phases ------------------------------------------------------------
    def _attempt(self, data, tier: str) -> dict:
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _PLAN_PROMPT.format(ticker=data.ticker)}]
        plan_resp = self.client.chat(messages, tier=tier)
        plan = (plan_resp.content or "").strip()
        messages.append({"role": "assistant", "content": plan})
        messages.append({"role": "user", "content": _ACT_PROMPT})

        tool_trace: list[dict] = []
        steps_used = 0
        while steps_used < config.MAX_TOOL_STEPS:
            resp = self.client.chat(messages, tools=TOOL_SCHEMAS, tier=tier)
            if not resp.tool_calls:
                if resp.content:
                    messages.append({"role": "assistant", "content": resp.content})
                break
            messages.append(_assistant_toolcalls_msg(resp))
            for tc in resp.tool_calls:
                steps_used += 1
                result = dispatch(tc.name, tc.arguments, data)
                err = result.get("error") if isinstance(result, dict) else None
                tool_trace.append({
                    "step": steps_used, "tool": tc.name, "arguments": tc.arguments,
                    "result": {} if err else result, "error": err,
                })
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result)})

        messages.append({"role": "user", "content": _DECIDE_PROMPT})
        decide_resp = self.client.chat(messages, tier=tier, force_json=True)
        decision, ok = _parse_decision(decide_resp.content)
        return {"plan": plan, "tool_trace": tool_trace, "decision": decision, "decision_ok": ok}

    def _reflect(self, decision: dict, tier: str) -> dict:
        payload = {k: (v.value if hasattr(v, "value") else v) for k, v in decision.items()}
        messages = [
            {"role": "system", "content": _REFLECT_SYSTEM},
            {"role": "user", "content": _REFLECT_PROMPT.format(decision=json.dumps(payload))},
        ]
        resp = self.client.chat(messages, tier=tier, force_json=True)
        return _parse_reflection(resp.content)

    @staticmethod
    def _needs_escalation(attempt: dict, reflection: dict) -> bool:
        if not attempt["decision_ok"]:
            return True
        if not reflection["consistent"]:
            return True
        if reflection["self_confidence"] < config.REFLECTION_CONFIDENCE_FLOOR:
            return True
        # If every tool call errored, the cheap pass had nothing to stand on.
        trace = attempt["tool_trace"]
        if trace and all(t["error"] for t in trace):
            return True
        return False


def _model(tier: str) -> str:
    from app.reasoning.openai_client import model_for_tier
    return model_for_tier(tier)
