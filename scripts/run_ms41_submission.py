"""Run the current MS41 final submission pipeline from one preset config."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    config = _load_config(Path(args.config))
    manifest_csv = Path(args.manifest_csv)
    template_csv = Path(args.template_csv) if args.template_csv else manifest_csv

    run_id = args.run_id or str(config["run"]["run_id"])
    data_root = args.data_root or str(config["run"]["data_root"])
    output_dir = Path(args.output_dir or config["run"]["output_dir"])
    base_run = f"{run_id}_ms32"
    base_out = output_dir / "ms32"
    classcache_id = f"{run_id}_classcache"

    output_dir.mkdir(parents=True, exist_ok=True)
    base_out.mkdir(parents=True, exist_ok=True)

    commands = _build_commands(
        config=config,
        manifest_csv=manifest_csv,
        template_csv=template_csv,
        data_root=data_root,
        output_dir=output_dir,
        base_out=base_out,
        run_id=run_id,
        base_run=base_run,
        classcache_id=classcache_id,
    )

    timings: dict[str, float] = {}
    for stage_name, command in commands:
        _print_command(stage_name, command)
        if args.dry_run:
            continue
        started = time.perf_counter()
        subprocess.run(command, check=True)
        timings[f"{stage_name}_seconds"] = round(time.perf_counter() - started, 3)

    if args.dry_run:
        return

    final_submission = output_dir / f"submission_{run_id}.csv"
    root_submission = Path("submission.csv")
    shutil.copyfile(final_submission, root_submission)

    validation_command = [
        sys.executable,
        "scripts/validate_submission.py",
        "--template-csv",
        str(template_csv),
        "--submission-csv",
        str(root_submission),
        "--output-json",
        str(output_dir / "submission_validation.json"),
    ]
    _print_command("validate", validation_command)
    started = time.perf_counter()
    subprocess.run(validation_command, check=True)
    timings["validation_seconds"] = round(time.perf_counter() - started, 3)

    checksum = hashlib.sha256(root_submission.read_bytes()).hexdigest()
    (output_dir / "submission.sha256").write_text(f"{checksum}  submission.csv\n", encoding="utf-8")
    timings["total_seconds"] = round(sum(timings.values()), 3)
    _write_speed_summary(output_dir / "speed_summary.txt", timings)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/release/ms41-private-full.toml",
        help="MS41 preset TOML config.",
    )
    parser.add_argument("--manifest-csv", required=True, help="Private or public manifest CSV.")
    parser.add_argument(
        "--template-csv",
        default="",
        help="Template CSV for validation. Defaults to --manifest-csv.",
    )
    parser.add_argument("--run-id", default="", help="Override run id from the preset.")
    parser.add_argument("--output-dir", default="", help="Override output dir from the preset.")
    parser.add_argument("--data-root", default="", help="Override data root from the preset.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved commands without executing them.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, dict[str, Any]]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _build_commands(
    *,
    config: dict[str, dict[str, Any]],
    manifest_csv: Path,
    template_csv: Path,
    data_root: str,
    output_dir: Path,
    base_out: Path,
    run_id: str,
    base_run: str,
    classcache_id: str,
) -> list[tuple[str, list[str]]]:
    run_cfg = config["run"]
    model_cfg = config["model"]
    official_cfg = config["official_tail"]
    classifier_cfg = config["classifier"]
    graph_cfg = config["graph"]

    official_command = [
        sys.executable,
        "scripts/run_official_campp_tail.py",
        "--checkpoint-path",
        str(model_cfg["checkpoint_path"]),
        "--manifest-csv",
        str(manifest_csv),
        "--template-csv",
        str(template_csv),
        "--data-root",
        data_root,
        "--output-dir",
        str(base_out),
        "--experiment-id",
        base_run,
        "--encoder-backend",
        "tensorrt",
        "--tensorrt-config",
        str(model_cfg["tensorrt_config"]),
        "--device",
        str(run_cfg["device"]),
        "--search-device",
        str(run_cfg["search_device"]),
        "--batch-size",
        str(official_cfg["batch_size"]),
        "--frontend-workers",
        str(official_cfg["frontend_workers"]),
        "--frontend-prefetch",
        str(official_cfg["frontend_prefetch"]),
        "--search-batch-size",
        str(official_cfg["search_batch_size"]),
        "--top-cache-k",
        str(official_cfg["top_cache_k"]),
        "--mode",
        str(official_cfg["mode"]),
        "--eval-chunk-seconds",
        str(official_cfg["eval_chunk_seconds"]),
        "--segment-count",
        str(official_cfg["segment_count"]),
        "--pad-mode",
        str(official_cfg["pad_mode"]),
        "--skip-c4",
    ]
    if official_cfg.get("force_embeddings", False):
        official_command.append("--force-embeddings")

    classifier_command = [
        sys.executable,
        "scripts/run_classifier_first_tail.py",
        "--checkpoint-path",
        str(model_cfg["checkpoint_path"]),
        "--embeddings-path",
        str(base_out / f"embeddings_{base_run}.npy"),
        "--manifest-csv",
        str(manifest_csv),
        "--output-dir",
        str(output_dir),
        "--experiment-id",
        classcache_id,
        "--device",
        str(run_cfg["device"]),
        "--class-batch-size",
        str(classifier_cfg["class_batch_size"]),
        "--class-top-k",
        str(classifier_cfg["class_top_k"]),
        "--class-scale",
        str(classifier_cfg["class_scale"]),
        "--class-cache-only",
    ]
    if classifier_cfg.get("force_classifier", False):
        classifier_command.append("--force-classifier")

    graph_command = [
        sys.executable,
        "scripts/run_class_aware_graph_tail.py",
        "--indices-path",
        str(base_out / f"indices_{base_run}_top{official_cfg['top_cache_k']}.npy"),
        "--scores-path",
        str(base_out / f"scores_{base_run}_top{official_cfg['top_cache_k']}.npy"),
        "--class-indices-path",
        str(output_dir / f"class_indices_{classcache_id}_top{classifier_cfg['class_top_k']}.npy"),
        "--class-probs-path",
        str(output_dir / f"class_probs_{classcache_id}_top{classifier_cfg['class_top_k']}.npy"),
        "--manifest-csv",
        str(manifest_csv),
        "--template-csv",
        str(template_csv),
        "--output-dir",
        str(output_dir),
        "--experiment-id",
        run_id,
        "--class-overlap-top-k",
        str(graph_cfg["class_overlap_top_k"]),
        "--class-overlap-weight",
        str(graph_cfg["class_overlap_weight"]),
        "--same-top1-bonus",
        str(graph_cfg["same_top1_bonus"]),
        "--same-query-topk-bonus",
        str(graph_cfg["same_query_topk_bonus"]),
        "--edge-top",
        str(graph_cfg["edge_top"]),
        "--reciprocal-top",
        str(graph_cfg["reciprocal_top"]),
        "--rank-top",
        str(graph_cfg["rank_top"]),
        "--iterations",
        str(graph_cfg["iterations"]),
        "--label-min-size",
        str(graph_cfg["label_min_size"]),
        "--label-max-size",
        str(graph_cfg["label_max_size"]),
        "--label-min-candidates",
        str(graph_cfg["label_min_candidates"]),
        "--shared-top",
        str(graph_cfg["shared_top"]),
        "--shared-min-count",
        str(graph_cfg["shared_min_count"]),
        "--reciprocal-bonus",
        str(graph_cfg["reciprocal_bonus"]),
        "--density-penalty",
        str(graph_cfg["density_penalty"]),
    ]
    return [
        ("official_tail", official_command),
        ("classifier_cache", classifier_command),
        ("class_aware_graph", graph_command),
    ]


def _print_command(stage_name: str, command: list[str]) -> None:
    rendered = " ".join(command)
    print(f"[ms41-submission] {stage_name}: {rendered}", flush=True)


def _write_speed_summary(path: Path, timings: dict[str, float]) -> None:
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in timings.items()),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
