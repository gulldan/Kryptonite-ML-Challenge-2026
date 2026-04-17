from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class ScorePoint:
    order: int
    date: str
    experiment: str
    score: float
    is_record: bool


@dataclass(frozen=True)
class FamilyStyle:
    title: str
    color: str
    soft_color: str
    dash: str | None = None


@dataclass(frozen=True)
class PointLabel:
    text: str
    dx: float
    dy: float


@dataclass(frozen=True)
class PlacedPointLabel:
    point: ScorePoint
    label: PointLabel
    family: FamilyStyle
    anchor_x: float
    anchor_y: float
    center_x: float
    center_y: float
    width: float
    height: float


FAMILY_STYLES: dict[str, FamilyStyle] = {
    "organizer_baseline": FamilyStyle("Organizer baseline", "#0f766e", "#ccfbf1"),
    "campp": FamilyStyle("CAM++", "#c2410c", "#ffedd5"),
    "w2vbert2": FamilyStyle("w2v-BERT 2.0", "#16a34a", "#dcfce7"),
    "eres_wavlm": FamilyStyle("ERes / WavLM", "#2563eb", "#dbeafe"),
}

FAMILY_ORDER = (
    "organizer_baseline",
    "campp",
    "w2vbert2",
    "eres_wavlm",
)

POINT_LABELS: dict[str, PointLabel] = {
    "Organizer baseline": PointLabel("Орг старт", 82, -42),
    "C4_b8_labelprop_mutual10": PointLabel("Graph tail", 74, -62),
    "P1_eres2netv2_h100_b128_public_c4": PointLabel("ERes jump", 78, -58),
    "P3_eres2netv2_g6_pseudo_ft_public_c4": PointLabel("ERes pseudo", 82, -66),
    "MS1_modelscope_campplus_voxceleb_default": PointLabel("CAM++ base", -92, -46),
    "MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z": PointLabel(
        "LowLR tune", -58, -72
    ),
    "MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z": PointLabel(
        "VoxBlink aug", 20, -70
    ),
    "MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z": PointLabel(
        "Pseudo boost", 102, -22
    ),
    "MS34_ms32_strict_consensus_pseudo_clean_20260414T1616Z": PointLabel("Strict miss", -56, 96),
    "MS36b_multiteacher_soft_pseudo_fast_public_c4_20260414T1922Z": PointLabel("Soft miss", 40, 96),
    "H9_official_eres_filtered_pseudo_public_c4_20260414T2113Z": PointLabel("ERes probe", -24, 78),
    "MS38_campp_weight_soup_public_c4_20260415T0531Z": PointLabel("Weight soup", -108, 70),
    "MS41_ms32_classaware_c4_weak_20260415T0530Z": PointLabel("Class bonus", 44, -18),
    "MS40_rowwise_tail_router_20260415T0611Z": PointLabel("Router miss", -42, 96),
    "MS39_campp_ms31_bn_adapter_pseudo_20260415T0639Z": PointLabel("BN pseudo", -78, 120),
    "W2V1k_stage1_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1": PointLabel(
        "w2vBERT s1", 96, -18
    ),
    "W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1": PointLabel(
        "w2vBERT s2", 40, -82
    ),
    "W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1": PointLabel(
        "w2vBERT s3", 90, -62
    ),
}


def _clean_markdown(value: str) -> str:
    return re.sub(r"`([^`]*)`", r"\1", value).strip()


def _short_label(experiment: str) -> str:
    cleaned = _clean_markdown(experiment)
    if cleaned == "Organizer baseline":
        return "Organizer"
    prefix = cleaned.split("_", maxsplit=1)[0]
    return prefix[:18]


def _parse_leaderboard_points(path: Path) -> list[ScorePoint]:
    rows: list[ScorePoint] = []
    best_score = -math.inf
    numeric_score_re = re.compile(r"(?<![\d.])(\d+\.\d+)(?![\d.])")

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 5 or cells[0] == "Date":
            continue

        score_match = numeric_score_re.search(cells[4])
        if score_match is None:
            continue

        score = float(score_match.group(1))
        is_record = score > best_score
        best_score = max(best_score, score)
        rows.append(
            ScorePoint(
                order=len(rows),
                date=_clean_markdown(cells[0]),
                experiment=_clean_markdown(cells[1]),
                score=score,
                is_record=is_record,
            )
        )

    if not rows:
        msg = f"no numeric public leaderboard scores found in {path}"
        raise ValueError(msg)
    return rows


def _scale(
    value: float, source_min: float, source_max: float, target_min: float, target_max: float
) -> float:
    if source_max == source_min:
        return (target_min + target_max) / 2
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def _score_ticks(min_score: float, max_score: float) -> list[float]:
    tick_min = math.floor(min_score * 10) / 10
    tick_max = math.ceil(max_score * 10) / 10
    ticks: list[float] = []
    current = tick_min
    while current <= tick_max + 0.0001:
        ticks.append(round(current, 1))
        current += 0.1
    return ticks


def _family_key(experiment: str) -> str:
    if experiment == "Organizer baseline" or experiment.startswith("organizer_baseline"):
        return "organizer_baseline"
    if experiment.startswith("baseline_fixed") or experiment.startswith(
        ("B2_", "B4_", "B7_", "B8_", "C4_")
    ):
        return "organizer_baseline"
    if experiment.startswith(("P1_", "P3_", "F", "G", "E", "H")):
        return "eres_wavlm"
    if experiment.startswith(("P2_", "MS")):
        return "campp"
    if experiment.startswith("W2V"):
        return "w2vbert2"
    return "organizer_baseline"


def _svg_text(
    text: str,
    x: float,
    y: float,
    *,
    size: int = 14,
    fill: str = "#334155",
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" '
        'font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}">{escape(text)}</text>'
    )


def _svg_rect(x: float, y: float, width: float, height: float, fill: str, stroke: str) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="8" fill="{fill}" stroke="{stroke}"/>'
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _label_box_width(text: str) -> float:
    return max(132.0, 34.0 + len(text) * 8.6)


def _label_overlap_area(
    first: PlacedPointLabel, second: PlacedPointLabel, *, padding: float = 10.0
) -> float:
    first_left = first.center_x - first.width / 2 - padding
    first_right = first.center_x + first.width / 2 + padding
    first_top = first.center_y - first.height / 2 - padding
    first_bottom = first.center_y + first.height / 2 + padding
    second_left = second.center_x - second.width / 2 - padding
    second_right = second.center_x + second.width / 2 + padding
    second_top = second.center_y - second.height / 2 - padding
    second_bottom = second.center_y + second.height / 2 + padding
    overlap_x = min(first_right, second_right) - max(first_left, second_left)
    overlap_y = min(first_bottom, second_bottom) - max(first_top, second_top)
    if overlap_x <= 0 or overlap_y <= 0:
        return 0.0
    return overlap_x * overlap_y


def _layout_point_labels(
    points: list[ScorePoint],
    coords: list[tuple[float, float]],
    *,
    width: float,
    margin_left: float,
    margin_top: float,
    plot_height: float,
) -> list[PlacedPointLabel]:
    content_left = margin_left + 20
    content_right = width - 36
    box_height = 54.0
    x_offsets = (0.0, -126.0, 126.0, -252.0, 252.0)
    y_offsets = (0.0, -66.0, 66.0, -132.0, 132.0, -198.0, 198.0)
    pending: list[PlacedPointLabel] = []

    for point in points:
        label = POINT_LABELS.get(point.experiment)
        if label is None or not point.is_record:
            continue
        anchor_x, anchor_y = coords[point.order]
        width_hint = _label_box_width(label.text)
        preferred_x = _clamp(
            anchor_x + label.dx,
            content_left + width_hint / 2,
            content_right - width_hint / 2,
        )
        preferred_y = _clamp(
            anchor_y + label.dy,
            margin_top + box_height / 2,
            margin_top + plot_height - box_height / 2,
        )
        pending.append(
            PlacedPointLabel(
                point=point,
                label=label,
                family=FAMILY_STYLES[_family_key(point.experiment)],
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                center_x=preferred_x,
                center_y=preferred_y,
                width=width_hint,
                height=box_height,
            )
        )

    pending.sort(
        key=lambda item: (
            not item.point.is_record,
            -item.point.score,
            -item.point.order,
        )
    )
    placed: list[PlacedPointLabel] = []

    for item in pending:
        best_candidate: PlacedPointLabel | None = None
        best_cost = math.inf
        direction_penalty_scale = 0.22 if item.label.dx >= 0 else 0.14

        for x_shift in x_offsets:
            for y_shift in y_offsets:
                candidate_x = _clamp(
                    item.center_x + x_shift,
                    content_left + item.width / 2,
                    content_right - item.width / 2,
                )
                candidate_y = _clamp(
                    item.center_y + y_shift,
                    margin_top + item.height / 2,
                    margin_top + plot_height - item.height / 2,
                )
                candidate = PlacedPointLabel(
                    point=item.point,
                    label=item.label,
                    family=item.family,
                    anchor_x=item.anchor_x,
                    anchor_y=item.anchor_y,
                    center_x=candidate_x,
                    center_y=candidate_y,
                    width=item.width,
                    height=item.height,
                )
                overlap_penalty = sum(_label_overlap_area(candidate, other) for other in placed)
                direction_penalty = 0.0
                if item.label.dx >= 0 and candidate.center_x < item.anchor_x:
                    direction_penalty = (
                        item.anchor_x - candidate.center_x
                    ) * direction_penalty_scale
                if item.label.dx < 0 and candidate.center_x > item.anchor_x:
                    direction_penalty = (
                        candidate.center_x - item.anchor_x
                    ) * direction_penalty_scale
                connector_penalty = math.dist(
                    (item.anchor_x, item.anchor_y),
                    (candidate.center_x, candidate.center_y),
                )
                preferred_penalty = abs(candidate.center_x - item.center_x) + abs(
                    candidate.center_y - item.center_y
                )
                cost = (
                    overlap_penalty * 1000
                    + connector_penalty * 0.2
                    + preferred_penalty
                    + direction_penalty
                )
                if cost < best_cost:
                    best_cost = cost
                    best_candidate = candidate

        if best_candidate is None:
            best_candidate = item
        placed.append(best_candidate)

    by_experiment = {item.point.experiment: item for item in placed}
    return [
        by_experiment[point.experiment] for point in points if point.experiment in by_experiment
    ]


def render_svg(points: list[ScorePoint]) -> str:
    width = 1100
    height = 960
    margin_left = 72
    margin_right = 34
    margin_top = 168
    margin_bottom = 196
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    min_score = min(point.score for point in points)
    max_score = max(point.score for point in points)
    y_min = max(0.0, math.floor((min_score - 0.025) * 20) / 20)
    y_max = math.ceil((max_score + 0.025) * 20) / 20

    def x_for(index: int) -> float:
        return _scale(index, 0, len(points) - 1, margin_left, margin_left + plot_width)

    def y_for(score: float) -> float:
        return _scale(score, y_min, y_max, margin_top + plot_height, margin_top)

    coords = [(x_for(index), y_for(point.score)) for index, point in enumerate(points)]
    chronological_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    best_point = max(points, key=lambda point: point.score)
    best_delta = best_point.score - points[0].score

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title desc">',
        "<title>Public leaderboard hypothesis map</title>",
        "<desc>Public Precision@10 chart split by model families with milestone labels.</desc>",
        "<defs>",
        '<linearGradient id="score-bg" x1="0" x2="0" y1="0" y2="1">',
        '<stop offset="0%" stop-color="#eff6ff"/>',
        '<stop offset="62%" stop-color="#f8fafc"/>',
        '<stop offset="100%" stop-color="#ffffff"/>',
        "</linearGradient>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="url(#score-bg)"/>',
        '<rect x="18" y="22" width="1064" height="902" rx="22" fill="#ffffff" stroke="#dbe4ee"/>',
        _svg_text("Public LB: карта гипотез", 42, 68, size=32, fill="#0f172a", weight=800),
        _svg_text(
            "Все public submissions видны точками; крупные бейджи отмечают новые рекорды.",
            42,
            104,
            size=17,
            fill="#475569",
        ),
        _svg_rect(790, 42, 260, 82, "#f8fafc", "#dbe4ee"),
        _svg_text("Лучший public", 812, 74, size=15, fill="#64748b", weight=700),
        _svg_text(f"{best_point.score:.4f}", 812, 111, size=34, fill="#c2410c", weight=800),
        _svg_text(f"+{best_delta:.4f}", 940, 102, size=17, fill="#166534", weight=700),
    ]

    high_band_y = y_for(0.7)
    elements.extend(
        [
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" '
            f'height="{high_band_y - margin_top:.1f}" fill="#dcfce7" opacity="0.38"/>',
            _svg_text(
                "зона сильных public веток",
                margin_left + 20,
                high_band_y - 12,
                size=15,
                fill="#166534",
                weight=700,
            ),
        ]
    )

    for tick in _score_ticks(y_min, y_max):
        y = y_for(tick)
        elements.extend(
            [
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" '
                f'y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>',
                _svg_text(
                    f"{tick:.1f}",
                    margin_left - 14,
                    y + 5,
                    size=15,
                    fill="#64748b",
                    anchor="end",
                ),
            ]
        )

    date_positions: dict[str, int] = {}
    for index, point in enumerate(points):
        date_positions.setdefault(point.date, index)

    for date, index in date_positions.items():
        x = x_for(index)
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" '
                f'y2="{margin_top + plot_height}" stroke="#e7edf4" stroke-width="1"/>',
                _svg_text(
                    date,
                    x,
                    margin_top + plot_height + 42,
                    size=15,
                    fill="#475569",
                    anchor="middle",
                ),
            ]
        )

    elements.extend(
        [
            f'<polyline points="{chronological_points}" fill="none" stroke="#94a3b8" '
            'stroke-width="2" stroke-dasharray="4 7" stroke-linecap="round" '
            'stroke-linejoin="round" opacity="0.58"/>',
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
            f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
            'stroke="#334155" stroke-width="1.6"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
            f'y2="{margin_top + plot_height}" stroke="#334155" stroke-width="1.6"/>',
        ]
    )

    for family_key in FAMILY_ORDER:
        family_points = [
            (index, point)
            for index, point in enumerate(points)
            if _family_key(point.experiment) == family_key
        ]
        if len(family_points) < 2:
            continue
        style = FAMILY_STYLES[family_key]
        dash = f' stroke-dasharray="{style.dash}"' if style.dash is not None else ""
        family_line = " ".join(
            f"{coords[index][0]:.1f},{coords[index][1]:.1f}" for index, _ in family_points
        )
        elements.append(
            f'<polyline points="{family_line}" fill="none" stroke="{style.color}" '
            f'stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round"{dash}/>'
        )

    for index, point in enumerate(points):
        x, y = coords[index]
        family = FAMILY_STYLES[_family_key(point.experiment)]
        ring_color = "#f59e0b" if point.is_record else "#ffffff"
        ring_width = 3.2 if point.is_record else 2.2
        radius = 8 if point.is_record else 6.2
        elements.extend(
            [
                "<g>",
                f"<title>{escape(point.experiment)}: {point.score:.4f}</title>",
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{family.color}" '
                f'stroke="{ring_color}" stroke-width="{ring_width}"/>',
                "</g>",
            ]
        )

    for placed_label in _layout_point_labels(
        points,
        coords,
        width=width,
        margin_left=margin_left,
        margin_top=margin_top,
        plot_height=plot_height,
    ):
        box_x = placed_label.center_x - placed_label.width / 2
        box_y = placed_label.center_y - placed_label.height / 2
        elements.extend(
            [
                (
                    f'<g class="point-label" '
                    f'data-experiment="{escape(placed_label.point.experiment)}">'
                ),
                (
                    f'<line x1="{placed_label.anchor_x:.1f}" y1="{placed_label.anchor_y:.1f}" '
                    f'x2="{placed_label.center_x:.1f}" y2="{placed_label.center_y:.1f}" '
                    f'stroke="{placed_label.family.color}" stroke-width="1.3" opacity="0.72"/>'
                ),
                _svg_rect(
                    box_x,
                    box_y,
                    placed_label.width,
                    placed_label.height,
                    "#ffffff",
                    placed_label.family.color,
                ),
                _svg_text(
                    placed_label.label.text,
                    placed_label.center_x,
                    placed_label.center_y - 4,
                    size=14,
                    fill="#0f172a",
                    weight=800,
                    anchor="middle",
                ),
                _svg_text(
                    f"{placed_label.point.score:.4f}",
                    placed_label.center_x,
                    placed_label.center_y + 16,
                    size=14,
                    fill=placed_label.family.color,
                    weight=800,
                    anchor="middle",
                ),
                "</g>",
            ]
        )

    legend_columns = 2
    legend_x_start = margin_left
    legend_y_start = height - 132
    legend_x_gap = 430
    legend_y_gap = 36
    for index, family_key in enumerate(FAMILY_ORDER):
        style = FAMILY_STYLES[family_key]
        dash = f' stroke-dasharray="{style.dash}"' if style.dash is not None else ""
        legend_column = index % legend_columns
        legend_row = index // legend_columns
        legend_x = legend_x_start + legend_column * legend_x_gap
        legend_y = legend_y_start + legend_row * legend_y_gap
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y:.1f}" x2="{legend_x + 42}" '
                f'y2="{legend_y:.1f}" stroke="{style.color}" stroke-width="5" '
                f'stroke-linecap="round"{dash}/>',
                _svg_text(style.title, legend_x + 52, legend_y + 6, size=16, fill="#334155"),
            ]
        )

    elements.extend(
        [
            f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 38}" text-anchor="middle" '
            'fill="#334155" font-family="Inter, Arial, sans-serif" font-size="16">'
            "Порядок public submissions; серый пунктир связывает все отправки во времени</text>",
            f'<text x="27" y="{margin_top + plot_height / 2:.1f}" text-anchor="middle" '
            'fill="#334155" font-family="Inter, Arial, sans-serif" font-size="16" '
            'transform="rotate(-90 27 '
            f'{margin_top + plot_height / 2:.1f})">Precision@10 public</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render public leaderboard score history as SVG.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/challenge-experiment-history.md"),
        help="Markdown file with the Leaderboard History table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/public-lb-score.svg"),
        help="Destination SVG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = _parse_leaderboard_points(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(points), encoding="utf-8")
    print(f"wrote {len(points)} points to {args.output}")


if __name__ == "__main__":
    main()
