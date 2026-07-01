"""OpenAI client wrapper + cost-aware two-tier router. (v2 addition)

Everything that talks to OpenAI goes through here. Two responsibilities:

  1. A thin, normalized ``chat`` call (so the agent and the narrator share one path,
     and so tests can inject a ``FakeClient`` with the same surface — no network).
  2. Cost accounting. Every call's token usage is metered into dollars via the price
     table in ``config.MODEL_PRICES``; an optional per-run budget cap aborts cleanly
     when exceeded. The deterministic report never depends on any of this.

The "router" is deliberately simple (two tiers): callers pass ``tier="cheap"`` by
default and only the agent's reflection step escalates to ``tier="strong"``. Keeping
the policy in the caller (agent) rather than here means the routing decision is
explainable and testable on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from app import config


class BudgetExceededError(RuntimeError):
    """Raised when a run's accumulated LLM spend exceeds the configured cap."""


class LLMUnavailableError(RuntimeError):
    """Raised when an OpenAI call is attempted without a configured API key."""


@dataclass
class ToolCall:
    """A normalized tool/function call requested by the model."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Normalized chat result — provider-agnostic so the agent never sees the SDK."""

    content: str | None
    tool_calls: list[ToolCall]
    model: str
    tokens_in: int
    tokens_out: int


@dataclass
class CostMeter:
    """Accumulates token usage across an entire run and converts it to USD.

    One meter is created per ``/api/reason`` run and shared across every LLM call
    (plan, tool steps, decision, reflection, and any escalated retry), so the cost
    reported to the user is the true end-to-end spend.
    """

    budget_usd: float = 0.0  # 0 disables the cap
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    by_model: dict[str, float] = field(default_factory=dict)

    def record(self, model: str, tokens_in: int, tokens_out: int) -> None:
        in_price, out_price = config.MODEL_PRICES.get(model, (0.0, 0.0))
        cost = (tokens_in / 1000.0) * in_price + (tokens_out / 1000.0) * out_price
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd = round(self.cost_usd + cost, 6)
        self.calls += 1
        self.by_model[model] = round(self.by_model.get(model, 0.0) + cost, 6)
        if self.budget_usd and self.cost_usd > self.budget_usd:
            raise BudgetExceededError(
                f"Reasoning run exceeded the ${self.budget_usd:.4f} budget "
                f"(spent ${self.cost_usd:.4f} over {self.calls} calls)."
            )


def model_for_tier(tier: str) -> str:
    """Map a logical tier ('cheap' | 'strong') to a concrete configured model id."""
    return config.LLM_TIER_STRONG if tier == "strong" else config.LLM_TIER_CHEAP


def is_available() -> bool:
    return config.HAS_API_KEY


class OpenAIClient:
    """Live OpenAI client. Constructed lazily so offline/no-key paths never import the SDK."""

    def __init__(self, meter: CostMeter | None = None):
        self.meter = meter or CostMeter(budget_usd=config.REASONING_BUDGET_USD)
        self._client = None

    def _ensure(self):
        if self._client is None:
            if not is_available():
                raise LLMUnavailableError("OPENAI_API_KEY is not set.")
            import openai  # imported lazily — no key / offline never needs it

            self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        return self._client

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tier: str = "cheap",
        force_json: bool = False,
    ) -> LLMResponse:
        client = self._ensure()
        model = model_for_tier(tier)
        kwargs: dict = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in (choice.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = resp.usage
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        self.meter.record(model, tin, tout)  # may raise BudgetExceededError

        return LLMResponse(
            content=choice.content,
            tool_calls=tool_calls,
            model=model,
            tokens_in=tin,
            tokens_out=tout,
        )
