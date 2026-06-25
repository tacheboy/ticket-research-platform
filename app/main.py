"""FastAPI app — JSON API, PDF download, and the single-page frontend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from app import __version__, config, llm
from app.data.provider import DataUnavailableError, FixtureProvider
from app.guardrails import InvalidTickerError
from app.orchestrator import Orchestrator
from app.report_pdf import build_pdf

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
        "llm_model": config.LLM_MODEL if llm.is_available() else None,
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
