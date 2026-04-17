"""Run prepared-model full submission generation for selected model families."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = _resolve(project_root, args.config)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    run_id = args.run_id or f"speed_submit_{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"
    log_root = _resolve(project_root, args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    suite_rows = []
    for model in _selected_models(config, families=set(args.family)):
        row = _run_model(
            model=model,
            config=config,
            project_root=project_root,
            run_id=run_id,
            log_root=log_root,
        )
        suite_rows.append(row)
        if row["returncode"] != 0 and not args.keep_going:
            break

    suite_path = _resolve(
        project_root,
        f"artifacts/speed-family-comparison/{run_id}_suite.json",
    )
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": str(config_path),
                "speed_metric_scope": config.get("speed_metric_scope", ""),
                "rows": suite_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {suite_path}")


def _run_model(
    *,
    model: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
    run_id: str,
    log_root: Path,
) -> dict[str, Any]:
    family = str(model["family"])
    command_name, command = _submission_command(model)
    output_dir = _optimized_output_dir(model=model, project_root=project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{run_id}_{family}_{command_name}.log"
    timing_path = output_dir / f"{command_name}_timing.json"
    started_at = _now()
    started = time.perf_counter()
    returncode = _run_logged_command(command=command, cwd=project_root, log_path=log_path)
    wall_total_s = time.perf_counter() - started
    finished_at = _now()

    row: dict[str, Any] = {
        "family": family,
        "label": model.get("label", family),
        "command_name": command_name,
        "command": command,
        "returncode": returncode,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_total_s": round(wall_total_s, 6),
        "log_path": str(log_path.relative_to(project_root)),
        "speed_metric_scope": config.get("speed_metric_scope", ""),
    }
    if returncode == 0:
        row.update(
            _run_post_checks(
                model=model, config=config, project_root=project_root, log_path=log_path
            )
        )
    timing_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{family}: returncode={returncode} wall_total_s={wall_total_s:.3f} "
        f"log={log_path.relative_to(project_root)}"
    )
    return row | {"timing_path": str(timing_path.relative_to(project_root))}


def _run_post_checks(
    *,
    model: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
    log_path: Path,
) -> dict[str, Any]:
    submission_value = model.get("optimized_submission")
    if not submission_value:
        return {"post_check_status": "skipped_no_optimized_submission"}
    submission_path = _resolve(project_root, str(submission_value))
    if not submission_path.is_file():
        return {
            "optimized_submission": str(submission_path.relative_to(project_root)),
            "post_check_status": "skipped_missing_optimized_submission",
        }

    validation_path = submission_path.with_name(f"{submission_path.stem}_suite_validation.json")
    validation_command = (
        "uv run --group train python scripts/validate_submission.py "
        f"--template-csv {_quote(str(config['public_csv']))} "
        f"--submission-csv {_quote(str(submission_path.relative_to(project_root)))} "
        f"--output-json {_quote(str(validation_path.relative_to(project_root)))}"
    )
    validation_s, validation_returncode = _run_aux(
        command=validation_command,
        cwd=project_root,
        log_path=log_path,
        section="suite-validation",
    )
    result: dict[str, Any] = {
        "optimized_submission": str(submission_path.relative_to(project_root)),
        "suite_validation_json": str(validation_path.relative_to(project_root)),
        "suite_validation_s": round(validation_s, 6),
        "suite_validation_returncode": validation_returncode,
    }

    source_value = model.get("source_submission")
    if not source_value:
        return result
    source_path = _resolve(project_root, str(source_value))
    if not source_path.is_file():
        result["source_submission_status"] = "missing"
        return result

    comparison_path = submission_path.with_name(f"{submission_path.stem}_source_overlap.json")
    comparison_command = (
        "uv run --group train python scripts/compare_submission_overlap.py "
        f"--left-csv {_quote(str(source_path.relative_to(project_root)))} "
        f"--right-csv {_quote(str(submission_path.relative_to(project_root)))} "
        f"--template-csv {_quote(str(config['public_csv']))} "
        f"--output-json {_quote(str(comparison_path.relative_to(project_root)))}"
    )
    comparison_s, comparison_returncode = _run_aux(
        command=comparison_command,
        cwd=project_root,
        log_path=log_path,
        section="suite-source-overlap",
    )
    result.update(
        {
            "source_submission": str(source_path.relative_to(project_root)),
            "source_overlap_json": str(comparison_path.relative_to(project_root)),
            "source_overlap_s": round(comparison_s, 6),
            "source_overlap_returncode": comparison_returncode,
        }
    )
    return result


def _run_logged_command(*, command: str, cwd: Path, log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{_now()}] command start\n{command}\n\n")
        log_file.flush()
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_file.write(f"\n[{_now()}] command exit returncode={completed.returncode}\n")
    return int(completed.returncode)


def _run_aux(*, command: str, cwd: Path, log_path: Path, section: str) -> tuple[float, int]:
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{_now()}] {section} start\n{command}\n\n")
        log_file.flush()
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_file.write(f"\n[{_now()}] {section} exit returncode={completed.returncode}\n")
    return time.perf_counter() - started, int(completed.returncode)


def _selected_models(config: dict[str, Any], *, families: set[str]) -> list[dict[str, Any]]:
    models = config.get("models", [])
    if not isinstance(models, list):
        raise ValueError("Config must contain a [[models]] array.")
    selected = [model for model in models if not families or model.get("family") in families]
    if not selected:
        raise ValueError(f"No models selected for families={sorted(families)}.")
    return selected


def _submission_command(model: dict[str, Any]) -> tuple[str, str]:
    wanted = (
        "baseline_public_original"
        if model.get("family") == "organizer_baseline"
        else "public_tensorrt_tail"
    )
    for command in model.get("commands", []):
        if command.get("name") == wanted:
            return wanted, str(command["command"])
    raise ValueError(f"Model {model.get('family')} does not define command {wanted!r}.")


def _optimized_output_dir(*, model: dict[str, Any], project_root: Path) -> Path:
    submission = model.get("optimized_submission")
    if submission:
        return _resolve(project_root, str(submission)).parent
    return _resolve(project_root, f"artifacts/speed-family-comparison/{model['family']}")


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/release/speed-family-comparison.toml",
        help="Speed-family TOML config.",
    )
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--log-root", default="artifacts/logs", help="Log directory.")
    parser.add_argument("--run-id", default="", help="Stable run id for logs and suite JSON.")
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Family to run. Repeat to run a subset; default runs all families.",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed family.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
