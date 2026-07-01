"""FastAPI app — JSON API, PDF download, and the single-page frontend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from app import __version__, config, llm
from app.data.provider import DataUnavailableError, FixtureProvider
from app.guardrails import InvalidTickerError
from app.orchestrator import Orchestrator
from app.report_pdf import build_pdf
# --- v2 ---
from app.reasoning.experiment import run_experiment
from app.reasoning.openai_client import BudgetExceededError, LLMUnavailableError
# --- end v2 ---

app = FastAPI(title="Ticker Research Platform", version=__version__)
_orchestrator = Orchestrator()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    from pathlib import Path
    return (Path(__file__).parent / "frontend" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "data_source": _orchestrator.data_source,
        "llm_enabled": llm.is_available(),
        # --- v2 --- OpenAI replaced Anthropic; expose the two-tier router + budget.
        "openai_enabled": llm.is_available(),
        "llm_tiers": {"cheap": config.LLM_TIER_CHEAP, "strong": config.LLM_TIER_STRONG}
        if llm.is_available() else None,
        "reasoning_budget_usd": config.REASONING_BUDGET_USD,
        # --- end v2 ---
        "fixture_tickers": FixtureProvider().available_tickers(),
    }


@app.get("/api/analyze")
def analyze(
    ticker: str = Query(..., description="Stock ticker, e.g. AAPL"),
    llm_narrative: bool = Query(True, description="Use the LLM narrative if available"),
) -> dict:
    report = _run(ticker, use_llm=llm_narrative)
    return report.model_dump()


@app.get("/api/report")
def report_pdf(ticker: str = Query(...), llm_narrative: bool = Query(True)) -> Response:
    report = _run(ticker, use_llm=llm_narrative)
    pdf = build_pdf(report)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{report.ticker}_research_report.pdf"'},
    )


def _run(ticker: str, use_llm: bool):
    try:
        return _orchestrator.analyze(ticker, use_llm=use_llm)
    except InvalidTickerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except DataUnavailableError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# --- v2 ---
@app.get("/api/reason")
def reason(ticker: str = Query(..., description="Stock ticker, e.g. AAPL")) -> dict:
    """Run the LLM reasoning experiment side-by-side with the deterministic engine.

    This never alters the deterministic recommendation; it returns the agent's own
    independent call plus a comparison. Requires OPENAI_API_KEY (503 if unset).
    """
    try:
        return run_experiment(ticker).model_dump()
    except InvalidTickerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except DataUnavailableError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except LLMUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail="LLM reasoning is unavailable: set OPENAI_API_KEY to enable it.",
        ) from e
    except BudgetExceededError as e:
        raise HTTPException(status_code=402, detail=str(e)) from e
# --- end v2 ---
