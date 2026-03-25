# ruff: noqa: E501
"""Render standalone slice-dashboard HTML for verification evaluation reports."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .verification_report import VerificationEvaluationReport, VerificationSliceSummary

VERIFICATION_SLICE_DASHBOARD_HTML_NAME = "verification_slice_dashboard.html"

_SECTION_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "noise_slice",
        "Noise",
        "Corruption-aware breakdown for noise suites using category and severity.",
    ),
    (
        "reverb_slice",
        "Reverb",
        "Room and direct-to-reverb breakdown for reverberant suites.",
    ),
    (
        "channel_slice",
        "Channel",
        "Channel-style codec breakdown for `dev_channel`-like evaluations.",
    ),
    (
        "distance_slice",
        "Distance",
        "Near/mid/far style breakdown for far-field distance stress suites.",
    ),
    (
        "duration_bucket",
        "Duration",
        "Duration buckets derived from left/right trial metadata.",
    ),
    (
        "silence_slice",
        "Silence",
        "Silence/pause robustness breakdown using suite severity or candidate id.",
    ),
)


def render_verification_slice_dashboard_html(report: VerificationEvaluationReport) -> str:
    metrics = report.summary.metrics
    score_statistics = report.summary.score_statistics
    calibration = report.summary.calibration

    cards = [
        _metric_card("Trials", str(metrics.trial_count)),
        _metric_card("Positives", str(metrics.positive_count)),
        _metric_card("Negatives", str(metrics.negative_count)),
        _metric_card("EER", _format_optional(metrics.eer)),
        _metric_card("MinDCF", _format_optional(metrics.min_dcf)),
        _metric_card("Score Gap", _format_optional(score_statistics.score_gap)),
        _metric_card("Brier", _format_optional(calibration.brier_score)),
        _metric_card("Log Loss", _format_optional(calibration.log_loss)),
    ]

    sections = [
        _render_section(
            report.slice_breakdown,
            field_name=field_name,
            title=title,
            description=description,
        )
        for field_name, title, description in _SECTION_SPECS
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Verification Slice Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe8;
      --panel: #fffdf9;
      --border: #d8cec1;
      --ink: #1f2723;
      --muted: #6d756f;
      --accent: #0d5c63;
      --accent-soft: rgba(13, 92, 99, 0.10);
      --good: #156f43;
      --shadow: 0 22px 54px rgba(22, 35, 31, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(13, 92, 99, 0.12), transparent 24rem),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 2.5rem 1.25rem 4rem;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(13, 92, 99, 0.96), rgba(22, 111, 67, 0.88));
      color: #f8fbfa;
      border-radius: 1.5rem;
      padding: 1.75rem;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.1rem);
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 0.85rem 0 0;
      max-width: 58rem;
      color: rgba(248, 251, 250, 0.88);
      line-height: 1.55;
      font-size: 1.02rem;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 0.9rem;
      margin: 1.5rem 0 2rem;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 1rem;
      padding: 1rem;
      box-shadow: var(--shadow);
    }}
    .card .label {{
      color: var(--muted);
      font-size: 0.82rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .card .value {{
      margin-top: 0.5rem;
      font-size: 1.65rem;
      font-weight: 700;
      letter-spacing: -0.04em;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 1.35rem;
      padding: 1.25rem;
      margin-top: 1rem;
      box-shadow: var(--shadow);
    }}
    section h2 {{
      margin: 0;
      font-size: 1.45rem;
      letter-spacing: -0.03em;
    }}
    section p {{
      margin: 0.45rem 0 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    .empty {{
      margin-top: 1rem;
      padding: 1rem;
      background: var(--accent-soft);
      border-radius: 1rem;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
      font-family: "SF Mono", "Menlo", "Consolas", monospace;
      font-size: 0.86rem;
    }}
    th, td {{
      padding: 0.75rem 0.65rem;
      border-bottom: 1px solid rgba(216, 206, 193, 0.72);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 0.72rem;
    }}
    td:first-child {{
      font-weight: 700;
      color: var(--accent);
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      border-radius: 999px;
      padding: 0.28rem 0.65rem;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.78rem;
      font-family: "SF Mono", "Menlo", "Consolas", monospace;
    }}
    .footer {{
      margin-top: 1.5rem;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.5;
    }}
    @media (max-width: 760px) {{
      main {{ padding: 1rem 0.8rem 2.2rem; }}
      .hero {{ padding: 1.2rem; }}
      section {{ padding: 1rem; }}
      table, thead, tbody, th, td, tr {{ display: block; }}
      thead {{ display: none; }}
      tr {{
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(216, 206, 193, 0.72);
      }}
      td {{
        border: 0;
        padding: 0.2rem 0;
      }}
      td::before {{
        content: attr(data-label);
        display: block;
        color: var(--muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.15rem;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="hero">
      <span class="pill">verification_eval_report</span>
      <h1>Verification Slice Dashboard</h1>
      <p>Offline summary for corruption-aware and duration-aware verification slices. This dashboard is generated directly from the evaluation artifacts and is intended for fast run-to-run review.</p>
    </div>
    <div class="cards">
      {"".join(cards)}
    </div>
    {"".join(sections)}
    <div class="footer">
      Curves, histograms, and the full markdown report remain alongside this file in the same output directory.
    </div>
  </main>
</body>
</html>"""


def _render_section(
    slice_rows: tuple[VerificationSliceSummary, ...] | list[VerificationSliceSummary],
    *,
    field_name: str,
    title: str,
    description: str,
) -> str:
    rows = sorted(
        (row for row in slice_rows if row.field_name == field_name),
        key=lambda row: (-row.trial_count, row.field_value),
    )
    if not rows:
        return (
            "<section>"
            f"<h2>{escape(title)}</h2>"
            f"<p>{escape(description)}</p>"
            '<div class="empty">No slice rows were emitted for this dimension in the current run.</div>'
            "</section>"
        )

    header = ["Slice", "Trials", "Pos", "Neg", "EER", "MinDCF", "Gap", "Mean score"]
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f'<td data-label="{header[0]}">{escape(row.field_value)}</td>'
            f'<td data-label="{header[1]}">{row.trial_count}</td>'
            f'<td data-label="{header[2]}">{row.positive_count}</td>'
            f'<td data-label="{header[3]}">{row.negative_count}</td>'
            f'<td data-label="{header[4]}">{_format_optional(row.eer)}</td>'
            f'<td data-label="{header[5]}">{_format_optional(row.min_dcf)}</td>'
            f'<td data-label="{header[6]}">{_format_optional(row.score_gap)}</td>'
            f'<td data-label="{header[7]}">{_format_optional(row.mean_score)}</td>'
            "</tr>"
        )

    head = "".join(f"<th>{escape(label)}</th>" for label in header)
    return (
        "<section>"
        f"<h2>{escape(title)}</h2>"
        f"<p>{escape(description)}</p>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        "</section>"
    )


def _metric_card(label: str, value: str) -> str:
    return (
        '<div class="card">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>'
        "</div>"
    )


def _format_optional(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
