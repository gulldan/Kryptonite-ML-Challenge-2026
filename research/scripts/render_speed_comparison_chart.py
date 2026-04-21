"""Render an SVG chart comparing end-to-end speed across model families."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class SpeedRun:
    family: str
    label: str
    mode: str
    wall_total_s: float
    embedding_s: float
    search_s: float
    rerank_s: float
    validation_s: float
    public_lb: float | None
    validator_passed: bool | None
    submission_path: str


FAMILY_COLORS: dict[str, str] = {
    "organizer_baseline": "#0f766e",
    "campp": "#c2410c",
    "w2vbert2": "#16a34a",
    "eres_wavlm": "#2563eb",
}

STAGE_COLORS: dict[str, str] = {
    "embedding": "#475569",
    "search": "#94a3b8",
    "rerank": "#cbd5e1",
    "validation": "#e2e8f0",
    "other": "#f1f5f9",
}


def main() -> None:
    args = _parse_args()
    runs = _load_runs(Path(args.input_json))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_svg(runs), encoding="utf-8")
    print(f"wrote {output_path}")


def _load_runs(path: Path) -> list[SpeedRun]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_runs = payload.get("runs") if isinstance(payload, dict) else payload
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"{path} must contain a non-empty runs list.")
    runs = [_coerce_run(item) for item in raw_runs]
    if not runs:
        raise ValueError(f"{path} did not contain any plottable runs.")
    return runs


def _coerce_run(item: object) -> SpeedRun:
    if not isinstance(item, dict):
        raise ValueError("Each run must be a JSON object.")
    data = cast("dict[str, object]", item)
    embedding_s = _float(data.get("embedding_s", data.get("frontend_model_s", 0.0)))
    search_s = _float(data.get("search_s", data.get("exact_topk_s", 0.0)))
    rerank_s = _float(data.get("rerank_s", data.get("c4_rerank_s", 0.0)))
    validation_s = _float(data.get("validation_s", 0.0))
    wall_total_s = _float(
        data.get("wall_total_s", embedding_s + search_s + rerank_s + validation_s)
    )
    if wall_total_s <= 0.0:
        raise ValueError("Each speed run must define wall_total_s or positive stage seconds.")
    public_lb_raw = data.get("public_lb")
    return SpeedRun(
        family=str(data.get("family", "organizer_baseline")),
        label=str(data.get("label", data.get("experiment_id", "run"))),
        mode=str(data.get("mode", data.get("encoder_backend", ""))),
        wall_total_s=wall_total_s,
        embedding_s=embedding_s,
        search_s=search_s,
        rerank_s=rerank_s,
        validation_s=validation_s,
        public_lb=None if public_lb_raw is None or public_lb_raw == "" else _float(public_lb_raw),
        validator_passed=_optional_bool(data.get("validator_passed")),
        submission_path=str(data.get("submission_path", "")),
    )


def _render_svg(runs: list[SpeedRun]) -> str:
    width = 1000
    margin_left = 302
    margin_right = 28
    speedup_x = width - 120
    chart_right = speedup_x - 60
    margin_top = 178
    row_height = 118
    chart_width = chart_right - margin_left
    height = margin_top + row_height * len(runs) + 104
    max_seconds = max(run.wall_total_s for run in runs)
    scale = chart_width / max_seconds if max_seconds > 0 else 1.0

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>Kryptonite speed comparison</title>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        _text("End-to-end speed comparison", 36, 46, size=30, weight=800, fill="#0f172a"),
        _text(
            "Lower is faster. Wall time covers prepared-model public submission "
            "generation on remote GPU1.",
            36,
            78,
            size=16,
            fill="#475569",
        ),
    ]
    _append_legend(lines, width)

    axis_y = margin_top - 26
    lines.append(_line(margin_left, axis_y, chart_right, axis_y, "#cbd5e1"))
    lines.append(_text("vs baseline", speedup_x, axis_y - 10, size=14, weight=700, fill="#475569"))
    for tick in _ticks(max_seconds):
        x = margin_left + tick * scale
        lines.append(_line(x, axis_y - 6, x, height - 82, "#e2e8f0"))
        lines.append(
            _text(f"{tick:.0f}s", x, axis_y - 12, size=13, fill="#64748b", anchor="middle")
        )

    for row_index, run in enumerate(runs):
        y = margin_top + row_index * row_height
        family_color = FAMILY_COLORS.get(run.family, "#64748b")
        row_fill = "#ffffff" if row_index % 2 == 0 else "#f1f5f9"
        lines.append(
            f'<rect x="24" y="{y - 22}" width="{width - margin_right - 24}" '
            f'height="94" rx="8" fill="{row_fill}" stroke="#e2e8f0"/>'
        )
        lines.append(
            f'<circle cx="44" cy="{y + 18}" r="8" fill="{family_color}">'
            f"<title>{escape(run.family)}</title></circle>"
        )
        lines.append(
            _text(_display_label(run.label), 66, y + 14, size=18, weight=700, fill="#0f172a")
        )
        lines.append(_text(_compact_mode(run.mode), 66, y + 40, size=14, fill="#64748b"))
        detail_parts = [f"wall {run.wall_total_s:.1f}s"]
        if run.public_lb is not None:
            detail_parts.append(f"LB {run.public_lb:.4f}")
        if run.validator_passed is not None:
            detail_parts.append("validator OK" if run.validator_passed else "validator failed")
        lines.append(_text(" / ".join(detail_parts), 66, y + 64, size=14, fill="#64748b"))
        _append_stacked_bar(lines, run, margin_left, y + 2, scale, speedup_x)
        lines.append(
            _text(
                _speedup_text(run, runs),
                speedup_x,
                y + 33,
                size=15,
                weight=700,
                fill=family_color,
            )
        )

    lines.append(
        _text(
            "Generated by research/scripts/render_speed_comparison_chart.py",
            48,
            height - 34,
            size=12,
        )
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _append_stacked_bar(
    lines: list[str],
    run: SpeedRun,
    x: float,
    y: float,
    scale: float,
    speedup_x: float,
) -> None:
    stages = [
        ("embedding", run.embedding_s),
        ("search", run.search_s),
        ("rerank", run.rerank_s),
        ("validation", run.validation_s),
    ]
    accounted = sum(seconds for _, seconds in stages)
    if run.wall_total_s > accounted:
        stages.append(("other", run.wall_total_s - accounted))
    cursor = x
    for name, seconds in stages:
        if seconds <= 0.0:
            continue
        width = max(1.0, seconds * scale)
        lines.append(
            f'<rect x="{cursor:.1f}" y="{y:.1f}" width="{width:.1f}" height="36" '
            f'rx="8" fill="{STAGE_COLORS[name]}"><title>{escape(name)}: '
            f"{seconds:.3f}s</title></rect>"
        )
        cursor += width
    if cursor == x:
        width = max(1.0, run.wall_total_s * scale)
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="36" rx="8" fill="#475569"/>'
        )
    bar_end = x + run.wall_total_s * scale
    label = f"{run.wall_total_s:.1f}s"
    label_width = len(label) * 8
    outside_x = bar_end + 10
    label_limit = speedup_x - 42
    if outside_x + label_width <= label_limit:
        lines.append(_text(label, outside_x, y + 24, fill="#0f172a"))
    elif bar_end - x >= label_width + 22:
        lines.append(_text(label, bar_end - 10, y + 24, fill="#ffffff", weight=700, anchor="end"))
    else:
        lines.append(_text(label, label_limit, y + 24, fill="#0f172a", anchor="end"))


def _append_legend(lines: list[str], width: int) -> None:
    del width
    x = 36
    y = 106
    item_widths = {
        "embedding": 130,
        "search": 96,
        "rerank": 104,
        "validation": 132,
        "other": 84,
    }
    cursor = x
    for name, color in STAGE_COLORS.items():
        item_x = cursor
        lines.append(f'<rect x="{item_x}" y="{y}" width="18" height="12" rx="4" fill="{color}"/>')
        lines.append(_text(name, item_x + 26, y + 12, size=14, fill="#475569"))
        cursor += item_widths[name]


def _speedup_text(run: SpeedRun, runs: list[SpeedRun]) -> str:
    baseline = next((item for item in runs if item.family == "organizer_baseline"), None)
    if baseline is None or run is baseline:
        return "baseline"
    if run.wall_total_s <= 0.0:
        return ""
    return f"{baseline.wall_total_s / run.wall_total_s:.2f}x"


def _compact_mode(value: str) -> str:
    replacements = {
        "organizer ONNX full submit": "organizer ONNX full submit",
        "prepared TensorRT full submit": "prepared TensorRT full submit",
    }
    return replacements.get(value, value or "runtime")


def _display_label(value: str) -> str:
    replacements = {
        "CAM++ MS32 encoder + MS41 tail": "CAM++ MS32 + MS41",
        "w2v-BERT 2.0 W2V1j stage3": "w2v-BERT W2V1j",
        "Official ERes2Net H9": "ERes2Net H9",
    }
    return replacements.get(value, value)


def _ticks(max_seconds: float) -> list[float]:
    if max_seconds <= 0:
        return [0.0]
    step = _nice_step(max_seconds / 5.0)
    ticks = []
    current = 0.0
    while current <= max_seconds + step * 0.25:
        ticks.append(current)
        current += step
    return ticks


def _nice_step(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** int(len(str(int(value))) - 1)
    for multiplier in (1, 2, 5, 10):
        step = magnitude * multiplier
        if step >= value:
            return float(step)
    return float(magnitude * 10)


def _text(
    value: str,
    x: float,
    y: float,
    *,
    size: int = 13,
    fill: str = "#334155",
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" '
        'font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}">{escape(value)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="1"/>'
    )


def _float(value: object | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}.")


def _optional_bool(value: object) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed"}
    return bool(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default="artifacts/speed-family-comparison/speed_results.json",
        help="JSON file with a top-level runs list.",
    )
    parser.add_argument(
        "--output",
        default="research/docs/assets/speed-comparison.svg",
        help="SVG path to write.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
