"""Orchestrator — the conductor of the multi-agent pipeline.

Flow:
    ticker
      -> [INPUT GUARDRAIL]  validate_ticker
      -> fetch MarketData   (provider abstraction: live or fixture)
      -> fan out to the 4 specialist agents (run concurrently)
      -> synthesize         composite score -> Buy/Sell/Hold + confidence
      -> [POLICY GUARDRAIL] confidence floor (force HOLD on thin data)
      -> assemble TickerReport  [OUTPUT GUARDRAIL: pydantic schema]
      -> [optional] LLM rationale -> [GROUNDING GUARDRAIL] verify -> use or fall back
      -> attach disclaimer

Agents are independent, so they run in a thread pool — this is the orchestration
pattern (parallel specialist workers feeding a synthesizer), and it keeps latency
flat as more agents are added.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app import config, guardrails, llm
from app.agents import SPECIALIST_AGENTS
from app.agents.synthesizer import synthesize
from app.data.provider import DataUnavailableError, MarketData, get_provider
from app.schemas import AgentResult, TickerReport


class Orchestrator:
    def __init__(self, data_source: str | None = None):
        self.data_source = data_source or config.DATA_SOURCE
        self.provider = get_provider(self.data_source)
        self.agents = SPECIALIST_AGENTS

    # -- public API ---------------------------------------------------------
    def analyze(self, raw_ticker: str, use_llm: bool = True) -> TickerReport:
        ticker = guardrails.validate_ticker(raw_ticker)  # INPUT guardrail
        data = self.provider.fetch(ticker)               # may raise DataUnavailableError
        return self._build_report(data, use_llm=use_llm)

    # -- internals ----------------------------------------------------------
    def _run_agents(self, data: MarketData) -> list[AgentResult]:
        # Specialist agents are independent -> run them concurrently.
        with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
            return list(pool.map(lambda a: a.analyze(data), self.agents))

    def _build_report(self, data: MarketData, use_llm: bool) -> TickerReport:
        results = self._run_agents(data)
        decision = synthesize(results)

        # POLICY guardrail: refuse a directional call when confidence is too low.
        recommendation, floor_warnings = guardrails.apply_confidence_floor(
            decision["recommendation"], decision["confidence"]
        )

        warnings = list(data.warnings) + floor_warnings
        rationale = decision["rationale"]
        if floor_warnings:
            rationale = floor_warnings[0] + "\n\n" + rationale

        # OUTPUT guardrail: constructing the pydantic model validates the whole shape.
        report = TickerReport(
            ticker=data.ticker,
            company_name=data.name,
            currency=data.currency,
            as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            current_price=data.current_price,
            key_metrics=self._key_metrics(data),
            recommendation=recommendation,
            sentiment=decision["sentiment"],
            confidence=decision["confidence"],
            composite_score=decision["composite_score"],
            rationale=rationale,
            rationale_source="deterministic",
            agent_results=results,
            warnings=warnings,
            data_source=data.source,
            disclaimer=config.DISCLAIMER,
        )

        # Optional LLM narrative, only if it passes the GROUNDING guardrail.
        if use_llm and llm.is_available():
            report = self._maybe_add_llm_rationale(report)

        return report

    def _maybe_add_llm_rationale(self, report: TickerReport) -> TickerReport:
        context = {
            "ticker": report.ticker,
            "company": report.company_name,
            "recommendation": report.recommendation.value,
            "confidence": report.confidence,
            "composite_score": report.composite_score,
            "current_price": report.current_price,
            "key_metrics": report.key_metrics,
            "agents": [
                {"agent": r.agent, "score": r.score, "confidence": r.confidence,
                 "summary": r.summary,
                 "signals": [{"name": s.name, "value": s.value,
                              "interpretation": s.interpretation} for s in r.signals]}
                for r in report.agent_results
            ],
        }
        text = llm.generate_rationale(context)
        if not text:
            return report
        ok, reason = guardrails.verify_rationale_grounding(text, report)
        if ok:
            report.rationale = text
            report.rationale_source = "llm"
        else:
            report.warnings.append(f"LLM rationale rejected by grounding guardrail: {reason}")
        return report

    @staticmethod
    def _key_metrics(data: MarketData) -> dict:
        f = data.fundamentals or {}
        a = data.analyst or {}
        return {
            "P/E": f.get("pe"),
            "PEG": f.get("peg"),
            "Profit margin": f.get("profit_margin"),
            "Revenue growth": f.get("revenue_growth"),
            "ROE": f.get("roe"),
            "Market cap": f.get("market_cap"),
            "Beta": (data.risk or {}).get("beta"),
            "Analyst consensus": a.get("recommendation_mean"),
            "Mean price target": a.get("target_mean"),
        }


__all__ = ["Orchestrator", "DataUnavailableError"]
