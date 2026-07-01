"""Render a TickerReport to a downloadable PDF using reportlab."""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from app.schemas import Recommendation, TickerReport

_REC_COLOR = {
    Recommendation.BUY: colors.HexColor("#1a7f37"),
    Recommendation.SELL: colors.HexColor("#cf222e"),
    Recommendation.HOLD: colors.HexColor("#9a6700"),
}


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 1e9:
            return f"${value/1e9:.1f}B"
        if -2 < value < 2:
            return f"{value:.2f}"
        return f"{value:,.2f}"
    return str(value)


def build_pdf(report: TickerReport) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        title=f"{report.ticker} Research Report",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, textColor=colors.grey)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14, alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=7.5, leading=10, textColor=colors.grey)

    story: list = []

    # Header
    story.append(Paragraph(f"{report.company_name} ({report.ticker})", h1))
    story.append(Paragraph(
        f"Automated Investment Research Report &nbsp;|&nbsp; Generated {report.as_of} "
        f"&nbsp;|&nbsp; Data source: {report.data_source}", sub))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.lightgrey))

    # Recommendation banner
    rec_color = _REC_COLOR.get(report.recommendation, colors.black)
    banner = Table([[
        Paragraph(f'<font color="white" size="16"><b>{report.recommendation.value}</b></font>', body),
        Paragraph(
            f'<font color="white">Sentiment: <b>{report.sentiment.value}</b><br/>'
            f'Confidence: <b>{report.confidence:.0%}</b><br/>'
            f'Composite score: <b>{report.composite_score:+.2f}</b></font>', body),
        Paragraph(
            f'<font color="white">Price<br/><b>{_fmt(report.current_price)} {report.currency}</b></font>', body),
    ]], colWidths=[1.7 * inch, 3.2 * inch, 1.6 * inch])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), rec_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(Spacer(1, 10))
    story.append(banner)

    # Rationale
    story.append(Paragraph("Rationale", h2))
    src = "OpenAI (LLM)" if report.rationale_source == "llm" else "deterministic engine"
    story.append(Paragraph(f'<i><font size="8" color="grey">Narrative source: {src}</font></i>', body))
    story.append(Spacer(1, 4))
    for line in report.rationale.split("\n"):
        if line.strip():
            story.append(Paragraph(line.replace("&", "&amp;"), body))

    # Key metrics
    story.append(Paragraph("Key Metrics", h2))
    metric_rows = [["Metric", "Value"]]
    for k, v in report.key_metrics.items():
        if k in ("Profit margin", "Revenue growth", "ROE") and isinstance(v, (int, float)):
            metric_rows.append([k, f"{v:.1%}"])
        else:
            metric_rows.append([k, _fmt(v)])
    mt = Table(metric_rows, colWidths=[2.6 * inch, 3.9 * inch])
    mt.setStyle(_table_style())
    story.append(mt)

    # Per-agent breakdown
    story.append(Paragraph("Specialist Agent Breakdown", h2))
    for r in report.agent_results:
        story.append(Paragraph(
            f"<b>{r.agent.capitalize()} agent</b> &nbsp; "
            f"score {r.score:+.2f} &nbsp;|&nbsp; confidence {r.confidence:.0%}", body))
        story.append(Paragraph(f'<font size="9" color="grey">{r.summary}</font>', body))
        if r.signals:
            rows = [["Signal", "Value", "Read", "Contribution"]]
            for s in r.signals:
                rows.append([s.name, s.value, s.interpretation, f"{s.contribution:+.2f}"])
            t = Table(rows, colWidths=[1.4 * inch, 1.5 * inch, 2.8 * inch, 0.8 * inch])
            t.setStyle(_table_style(small=True))
            story.append(Spacer(1, 2))
            story.append(t)
        story.append(Spacer(1, 8))

    # Warnings
    if report.warnings:
        story.append(Paragraph("Warnings & Guardrails", h2))
        for w in report.warnings:
            story.append(Paragraph(f"• {w}", small))

    # Disclaimer
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=colors.lightgrey))
    story.append(Spacer(1, 4))
    story.append(Paragraph(report.disclaimer, small))

    doc.build(story)
    return buf.getvalue()


def _table_style(small: bool = False) -> TableStyle:
    fs = 7.5 if small else 9
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24292f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ])
