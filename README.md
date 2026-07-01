# 📈 Ticker Research Platform

A multi-agent system that takes a stock ticker and produces an investment
recommendation report — **Buy / Sell / Hold**, a confidence score, a sentiment read,
and a full rationale — viewable in a simple web UI and **downloadable as a PDF**.

Built for the technical exercise. The three things it is evaluated on —
**agent orchestration design, guardrails, and evals** — are the three things it
treats as first-class. This README explains the choices and why.

---

## TL;DR — run it

```bash
pip install -r requirements.txt
python run.py
# open http://127.0.0.1:8000  →  type AAPL, TSLA, or WEAK  →  Analyze  →  Download PDF
```

```bash
# run the eval suite (deterministic, offline)
python -m pytest evals/ -q          # 31 tests
```

No API key and no internet are required — the platform ships with bundled fixture
data and a deterministic engine. Set `OPENAI_API_KEY` to additionally get an
LLM-written rationale and the v2 reasoning experiment (see *The LLM layer* and
*v2 — Reasoning experiment* below).

---

## What you get

For any ticker the report includes everything the brief asked for:

- **Current price & key metrics** (P/E, PEG, margins, growth, ROE, beta, market cap,
  analyst consensus, mean target)
- **Market sentiment analysis** (analyst consensus + news tone)
- **Technical + fundamental insights** (per-agent signal breakdown)
- **Buy / Sell / Hold recommendation**
- **Confidence score & rationale**
- **Downloadable PDF report**

---

## 1. Agent orchestration design

### Pattern: orchestrator → parallel specialist agents → synthesizer

```
   ticker
     │  [INPUT GUARDRAIL: validate_ticker]
     ▼
 ┌──────────────┐      MarketData (provider abstraction: live yfinance OR fixtures)
 │ Orchestrator │◄────────────────────────────────────────────────────────────────
 └──────┬───────┘
        │  fan out (thread pool — agents are independent)
        ├──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
   ┌─────────┐  ┌────────────┐  ┌───────────┐  ┌─────────┐
   │Technical│  │Fundamental │  │ Sentiment │  │  Risk   │   each returns a
   │  agent  │  │   agent    │  │   agent   │  │  agent  │   deterministic
   └────┬────┘  └─────┬──────┘  └─────┬─────┘  └────┬────┘   score + confidence
        └─────────────┴───────┬───────┴─────────────┘        + signals + summary
                              ▼
                       ┌─────────────┐   weighted, confidence-scaled composite
                       │ Synthesizer │   → Buy/Sell/Hold + sentiment + confidence
                       └──────┬──────┘   → deterministic rationale
                              │  [POLICY GUARDRAIL: confidence floor]
                              ▼
                       TickerReport  [OUTPUT GUARDRAIL: pydantic schema]
                              │  [optional LLM rationale → GROUNDING GUARDRAIL]
                              ▼
                     JSON API · Web UI · PDF
```

**Four specialist agents** (the brief asks for ≥2; the recommendation logic asks for
four distinct lenses, so there is one agent per lens):

| Agent | Lens (from the brief) | What it scores |
|-------|------------------------|----------------|
| `TechnicalAgent`   | Price momentum / technical indicators | SMA 50/200 trend, RSI, MACD histogram, 3M/6M returns |
| `FundamentalAgent` | Valuation metrics | P/E, PEG, P/B, profit margin, revenue growth, ROE |
| `SentimentAgent`   | Sentiment (news + analyst consensus) | analyst `recommendationMean`, target upside, news-headline tone |
| `RiskAgent`        | Risk factors | annualized volatility, max drawdown, beta, market-cap/liquidity |

Each agent reads the same normalized `MarketData` and emits a typed `AgentResult`
(`score ∈ [-1,+1]`, `confidence ∈ [0,1]`, a list of explainable `Signal`s, and a
summary). The **synthesizer** combines them with configurable weights
(`app/config.py`), scaled by each agent's own confidence, into a composite score that
maps to the recommendation.

### Key design decision: deterministic core, LLM as narrator

The actual Buy/Sell/Hold decision is **rule-based on real data**, not produced by an
LLM. For a financial recommendation that has to be reproducible, auditable, and
testable, letting a model free-hand the number would be both un-evaluable and a
hallucination risk. So:

- **Numbers and the decision** → deterministic scoring on fetched data.
- **Prose / rationale** → optionally an LLM (OpenAI), *narrating the already-decided facts*,
  and only after passing a grounding guardrail.

This is what makes the eval suite meaningful and the guardrails real.

### Why a data-provider abstraction

Everything depends on the `MarketData` contract, never on a data source
(`app/data/provider.py`). `YFinanceProvider` is live; `FixtureProvider` serves bundled,
deterministic fixtures (with a reproducible synthetic price path); `auto` tries live
and transparently falls back to fixtures. This is what lets the **same agents and
scoring** power the live app *and* run fully offline in CI.

---

## 2. Guardrails

Guardrails are a dedicated module (`app/guardrails.py`) plus the schema layer, placed
at every stage of the pipeline:

| Stage | Guardrail | What it prevents |
|-------|-----------|------------------|
| **Input** | `validate_ticker` | Malformed / injection-style input (regex allowlist, length cap, normalization). Bad input → `400`. |
| **Data** | provider fallback + `DataUnavailableError` | Network/source failure degrades gracefully (fixture fallback) or returns a clean `404`; missing fields lower an agent's confidence instead of crashing. |
| **Output** | pydantic `TickerReport` / `AgentResult` | Guarantees the report shape and value ranges (`score ∈ [-1,1]`, `confidence ∈ [0,1]`). Malformed agent output is caught at the boundary. |
| **Policy** | `apply_confidence_floor` | Refuses a directional call on thin data — forces **HOLD** with an explicit warning when confidence < threshold. |
| **Grounding** | `verify_rationale_grounding` | Rejects an LLM rationale that **contradicts the computed recommendation** or **cites numbers not present in the data**; falls back to the deterministic rationale. |
| **Compliance** | mandatory `DISCLAIMER` | Every report (UI + PDF) carries a not-financial-advice disclaimer. |

The LLM is also sandboxed by construction: it is told the recommendation is final and
to use only supplied numbers, runs with a bounded `max_tokens`, and any error degrades
silently to the deterministic path.

---

## 3. Evals

`python -m pytest evals/` — 31 tests, deterministic and offline (fixtures forced,
API key stripped via `evals/conftest.py`). Four layers:

- **`test_agents.py`** — each specialist points the right way on known data
  (uptrend → bullish technicals, unprofitable+shrinking → bearish fundamentals,
  high-beta → risk flagged) and degrades gracefully on missing fields.
- **`test_recommendation.py`** — golden end-to-end scenarios (strong name → BUY,
  weak name → SELL, mixed name → not a BUY) plus a **determinism** test
  (same input → identical output).
- **`test_guardrails.py`** — input validation accept/reject, confidence-floor
  downgrade, schema round-trip validity, and the grounding guardrail catching
  contradictions and invented numbers.
- **Schema invariants** — every report re-validates against the public contract.

The fixtures are designed as **distinct regimes** so the assertions are meaningful:
`AAPL` (uptrend / fair value / positive / low risk), `TSLA` (flat / expensive /
mixed / very high risk), `WEAK` (downtrend / unprofitable / negative / high risk).

---

## The LLM layer (optional)

Set `OPENAI_API_KEY` and the synthesizer asks OpenAI (`gpt-4o-mini` by default) to
write the rationale from the structured findings. The output must pass
`verify_rationale_grounding` or it is discarded in favor of the deterministic
rationale (the report's `rationale_source` field records which was used). The
decision and all numbers are unchanged either way.

---

## v2 — Reasoning experiment & OpenAI cost router

v2 adds a **parallel, experimental reasoning pipeline** and migrates the entire LLM
layer from Anthropic to **OpenAI**. The deterministic engine above is **unchanged** and
remains the system's authoritative recommendation. All v2 code is marked with `v2`
docstrings / `# --- v2 ---` comment markers so before/after is obvious.

### What it is

`GET /api/reason?ticker=AAPL` runs a genuinely *agentic* loop **alongside** the
deterministic engine and shows the two side-by-side (also surfaced in the web UI via the
**🧪 Run LLM reasoning experiment** button):

```
            ┌─────────────── same MarketData ───────────────┐
            ▼                                                ▼
   Deterministic engine                          LLM reasoning agent (v2)
   (authoritative, unchanged)                    plan → act → decide → reflect → retry
            │                                                │
            └───────────────► ExperimentComparison ◄─────────┘
                         (agree? score / confidence deltas)
```

The agent (`app/reasoning/agent.py`) **plans**, then issues **dynamic tool calls**
(`get_price_technicals`, `get_fundamentals`, `get_sentiment_inputs`, `get_risk_metrics` —
raw data only, *not* the deterministic scores, so its view is independent), then
**decides** a structured Buy/Sell/Hold, then **reflects** on its own call. If the
reflection is weak (low self-confidence, inconsistent, or every tool errored) it
**retries with escalation**.

### Cost-aware two-tier router

Every call defaults to the **cheap** model (`gpt-4o-mini`); the agent escalates to the
**strong** model (`gpt-4o`) *only* when reflection fails. A `CostMeter`
(`app/reasoning/openai_client.py`) meters token usage into USD via the price table in
`app/config.py` and enforces an optional **per-run budget cap** (`TICKER_REASONING_BUDGET_USD`)
— exceeding it aborts the experiment cleanly (`402`) without touching the deterministic report.
The report returns `model_used`, `escalated`, `retry_count`, tokens, and `cost_usd`.

### Evals

`evals/test_reasoning.py` drives the whole loop with a scripted **offline `FakeClient`**
(no network/key): it verifies tool dispatch returns real fixture numbers, that a
low-confidence reflection triggers cheap→strong escalation, that the `CostMeter` sums
correctly and the budget cap raises, and that the comparison + schemas validate.

---

## Project layout

```
ticker-research-platform/
├── run.py                     # launch the web app
├── requirements.txt
├── .env.example
├── app/
│   ├── config.py              # weights, thresholds, disclaimer — all tunables in one place
│   ├── schemas.py             # pydantic contracts (output guardrail)
│   ├── guardrails.py          # input / policy / grounding guardrails
│   ├── indicators.py          # pure-numpy technical math
│   ├── llm.py                 # optional OpenAI narrative (v2: migrated off Anthropic)
│   ├── orchestrator.py        # the pipeline conductor
│   ├── report_pdf.py          # reportlab PDF
│   ├── main.py                # FastAPI: /api/analyze, /api/report, /api/reason, /api/health, /
│   ├── agents/                # technical, fundamental, sentiment, risk, synthesizer
│   ├── reasoning/             # v2: agentic LLM experiment + OpenAI cost router
│   │   ├── openai_client.py   #   OpenAI wrapper, two-tier router, CostMeter + budget cap
│   │   ├── tools.py           #   dynamic function-calling tools over MarketData
│   │   ├── agent.py           #   plan → act → decide → reflect → retry/escalate loop
│   │   └── experiment.py      #   runs deterministic + reasoning, builds comparison
│   ├── data/                  # provider abstraction + fixtures/
│   └── frontend/index.html    # single-page UI (+ v2 experiment panel)
└── evals/                     # pytest suite (agents, recommendation, guardrails, reasoning)
```

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Web UI |
| `GET /api/analyze?ticker=AAPL` | JSON report |
| `GET /api/report?ticker=AAPL` | PDF download |
| `GET /api/reason?ticker=AAPL` | **v2** — LLM reasoning experiment vs deterministic (needs `OPENAI_API_KEY`; `503` without) |
| `GET /api/health` | Status, data source, LLM availability, tiers, budget, fixture list |

## Configuration (env)

| Variable | Default | Meaning |
|----------|---------|---------|
| `TICKER_DATA_SOURCE` | `auto` | `auto` \| `live` \| `fixture` |
| `OPENAI_API_KEY` | _unset_ | Enables the LLM rationale layer + v2 reasoning experiment |
| `TICKER_LLM_TIER_CHEAP` | `gpt-4o-mini` | Default (cheap) model — used for everything |
| `TICKER_LLM_TIER_STRONG` | `gpt-4o` | Escalation (strong) model — only on reflection failure |
| `TICKER_REASONING_BUDGET_USD` | `0.05` | Per-run spend cap for `/api/reason` (`0` disables) |

---

*Educational/informational use only — not investment advice.*
