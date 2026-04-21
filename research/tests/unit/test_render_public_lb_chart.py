from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _load_chart_module() -> Any:
    script_path = Path(__file__).parents[2] / "scripts" / "render_public_lb_chart.py"
    spec = importlib.util.spec_from_file_location("render_public_lb_chart", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _label_rectangles(svg: str) -> dict[str, tuple[float, float, float, float]]:
    root = ET.fromstring(svg)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    rectangles: dict[str, tuple[float, float, float, float]] = {}

    for group in root.findall(".//svg:g[@class='point-label']", namespace):
        experiment = group.attrib["data-experiment"]
        rect = group.find("svg:rect", namespace)
        assert rect is not None
        rectangles[experiment] = (
            float(rect.attrib["x"]),
            float(rect.attrib["y"]),
            float(rect.attrib["width"]),
            float(rect.attrib["height"]),
        )
    return rectangles


def _rectangles_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    return (
        first_x < second_x + second_width
        and second_x < first_x + first_width
        and first_y < second_y + second_height
        and second_y < first_y + first_height
    )


def test_render_public_lb_chart_separates_top_right_w2vbert_labels() -> None:
    module = _load_chart_module()
    history_path = Path(__file__).parents[2] / "docs" / "challenge-experiment-history.md"
    points = module._parse_leaderboard_points(history_path)

    rectangles = _label_rectangles(module.render_svg(points))
    stage1 = "W2V1k_stage1_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1"
    stage3 = "W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1"

    assert stage1 in rectangles
    assert stage3 in rectangles
    assert not _rectangles_overlap(rectangles[stage1], rectangles[stage3])


def test_render_public_lb_chart_uses_only_core_model_families_in_legend() -> None:
    module = _load_chart_module()
    history_path = Path(__file__).parents[2] / "docs" / "challenge-experiment-history.md"
    svg = module.render_svg(module._parse_leaderboard_points(history_path))

    for legend_title in ("Organizer baseline", "CAM++", "w2v-BERT 2.0", "ERes / WavLM"):
        assert legend_title in svg
    for removed_title in (
        ">Organizer<",
        ">Baseline<",
        "CAM++ safe branch (7.2M)",
        "CAM++ probes / tail (7.2M)",
    ):
        assert removed_title not in svg
