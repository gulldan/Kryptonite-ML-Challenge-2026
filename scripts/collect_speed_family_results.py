"""Collect prepared-model full-submit timings into the speed chart JSON schema."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASELINE_WALL_RE = re.compile(r"Inference \+ indexing wall time:\s*([0-9.]+)s")


def main() -> None:
    args = _parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = _resolve(project_root, args.config)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    runs = [
        _collect_model(model=model, config=config, project_root=project_root)
        for model in config.get("models", [])
    ]
    output_path = _resolve(project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                "config": str(config_path.relative_to(project_root)),
                "speed_metric_scope": config.get("speed_metric_scope", ""),
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}")


def _collect_model(
    *, model: dict[str, Any], config: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    family = str(model["family"])
    command_name = (
        "baseline_public_original" if family == "organizer_baseline" else "public_tensorrt_tail"
    )
    submission_path = _resolve(project_root, str(model["optimized_submission"]))
    output_dir = submission_path.parent
    timing = _load_json(output_dir / f"{command_name}_timing.json")
    summary = _load_optional_json(model.get("optimized_summary"), project_root=project_root)
    validation = _load_optional_json(
        submission_path.with_name(f"{submission_path.stem}_suite_validation.json"),
        project_root=project_root,
    )
    source_overlap = _load_optional_json(
        submission_path.with_name(f"{submission_path.stem}_source_overlap.json"),
        project_root=project_root,
    )

    if family == "organizer_baseline":
        wall_total_s = _baseline_submit_seconds(timing=timing, project_root=project_root)
        embedding_s = wall_total_s
        search_s = 0.0
        rerank_s = 0.0
        submit_write_s = 0.0
    else:
        wall_total_s = _summary_submit_generation_seconds(summary)
        embedding_s = _float(summary.get("embedding_s", 0.0))
        search_s = _float(summary.get("search_s", 0.0))
        rerank_s = _float(summary.get("rerank_s", summary.get("c4_rerank_s", 0.0)))
        submit_write_s = _float(
            summary.get("submit_write_s", summary.get("c4_submit_write_s", 0.0))
        )

    validator_passed = _validator_passed(summary=summary, validation=validation)
    row: dict[str, Any] = {
        "family": family,
        "label": model.get("label", family),
        "mode": "prepared TensorRT full submit"
        if family != "organizer_baseline"
        else "organizer ONNX full submit",
        "public_lb": model.get("public_lb"),
        "wall_total_s": round(wall_total_s, 6),
        "embedding_s": round(embedding_s, 6),
        "search_s": round(search_s, 6),
        "rerank_s": round(rerank_s, 6),
        "submit_write_s": round(submit_write_s, 6),
        "validation_s": 0.0,
        "validator_passed": validator_passed,
        "submission_path": str(submission_path.relative_to(project_root)),
        "timing_path": str((output_dir / f"{command_name}_timing.json").relative_to(project_root)),
        "log_path": timing.get("log_path", ""),
        "source_submission": model.get("source_submission", ""),
    }
    if summary:
        row["summary_path"] = str(
            _resolve(project_root, str(model["optimized_summary"])).relative_to(project_root)
        )
    if validation:
        row["suite_validation_error_count"] = validation.get("error_count")
    if source_overlap:
        comparison = source_overlap.get("comparison", {})
        row["source_overlap_mean_at_10"] = comparison.get("mean_overlap_at_k")
        row["source_overlap_median_at_10"] = comparison.get("median_overlap_at_k")
        row["source_overlap_top1_equal_share"] = comparison.get("top1_equal_share")
        row["source_overlap_ordered_cell_equal_share"] = comparison.get("ordered_cell_equal_share")
        row["source_overlap_row_exact_same_order_share"] = comparison.get(
            "row_exact_same_order_share"
        )
        row["source_overlap_row_same_set_share"] = comparison.get("row_same_set_share")
    return row


def _summary_submit_generation_seconds(summary: dict[str, Any]) -> float:
    for key in ("submit_generation_s", "c4_submit_generation_s", "exact_submit_generation_s"):
        if key in summary:
            return _float(summary[key])
    return (
        _float(summary.get("embedding_s", 0.0))
        + _float(summary.get("search_s", 0.0))
        + _float(summary.get("rerank_s", summary.get("c4_rerank_s", 0.0)))
        + _float(summary.get("submit_write_s", summary.get("c4_submit_write_s", 0.0)))
    )


def _baseline_submit_seconds(*, timing: dict[str, Any], project_root: Path) -> float:
    log_path = timing.get("log_path")
    if isinstance(log_path, str) and log_path:
        log_text = _resolve(project_root, log_path).read_text(encoding="utf-8", errors="replace")
        matches = BASELINE_WALL_RE.findall(log_text)
        if matches:
            return float(matches[-1])
    return _float(timing.get("wall_total_s", 0.0))


def _validator_passed(*, summary: dict[str, Any], validation: dict[str, Any]) -> bool | None:
    for key in ("validator_passed", "c4_validator_passed", "exact_validator_passed"):
        if key in summary:
            return bool(summary[key])
    if "passed" in validation:
        return bool(validation["passed"])
    return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_optional_json(value: object, *, project_root: Path) -> dict[str, Any]:
    if value in {None, ""}:
        return {}
    path = _resolve(project_root, str(value))
    if not path.is_file():
        return {}
    return _load_json(path)


def _float(value: object) -> float:
    if value in {None, ""}:
        return 0.0
    return float(value)


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/release/speed-family-comparison.toml",
        help="Speed-family TOML config.",
    )
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument(
        "--output",
        default="artifacts/speed-family-comparison/speed_results.json",
        help="JSON path consumed by render_speed_comparison_chart.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
