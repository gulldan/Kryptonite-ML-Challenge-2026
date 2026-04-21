"""Resume-safe robustness benchmark for speaker embeddings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from kryptonite.eda.dense_audio import eval_crops, l2_normalize_rows, load_eval_waveform
from kryptonite.eval.verification_metrics import compute_verification_metrics_from_arrays
from kryptonite.models.scoring import cosine_score_pairs
from kryptonite.training.teacher_peft import load_teacher_peft_encoder_from_checkpoint

from .audio import (
    TARGET_SAMPLE_RATE_HZ,
    DistortionCondition,
    build_distorted_plan,
    build_frozen_clean_subset,
    collect_audio_stats,
    default_distortion_conditions,
    materialize_condition_audio,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class BenchmarkPaths:
    root: Path
    manifests: Path
    reports: Path
    state: Path
    distortions: Path
    embeddings: Path
    trials: Path


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    label: str
    kind: str
    checkpoint_path: Path
    feature_extractor_path: Path | None = None
    checkpoint_metadata_path: Path | None = None
    backbone_path: Path | None = None


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_offline_env(hf_home=Path(args.hf_home))
    paths = _build_paths(Path(args.runtime_root))
    _ensure_runtime_dirs(paths)
    conditions = default_distortion_conditions()
    models = _build_model_specs(args)
    _write_json(
        paths.state / "assets.json",
        {
            "run_key": args.run_key,
            "seed": args.seed,
            "clean_set_size": args.clean_set_size,
            "source_data_root": args.source_data_root,
            "source_dev_manifest": args.source_manifest_csv,
            "conditions": [asdict(condition) for condition in conditions],
            "offline_env": {
                "HF_HOME": os.environ.get("HF_HOME", ""),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
                "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", ""),
            },
            "models": [asdict(model) for model in models],
            "normalized_degradation_formula": (
                "0.5*max((eer-eer_clean)/eer_clean,0)+"
                "0.5*max((min_dcf-min_dcf_clean)/min_dcf_clean,0)"
            ),
            "robustness_protocol": {
                "clean_condition": "clean-clean",
                "distorted_condition": "clean-enroll vs distorted-test",
                "clean_source_pool": "participants_fixed/dev_manifest.csv",
            },
        },
    )

    source_manifest = Path(args.source_manifest_csv)
    stats = collect_audio_stats(
        manifest_path=source_manifest,
        data_root=Path(args.source_data_root),
        output_path=paths.state / "audio_stats_dev.csv",
        workers=args.audio_stats_workers,
    )
    frozen_manifest_path = paths.manifests / "clean_3000_frozen.csv"
    if frozen_manifest_path.is_file():
        clean_manifest = pl.read_csv(frozen_manifest_path)
    else:
        clean_manifest = build_frozen_clean_subset(
            stats=stats,
            target_size=args.clean_set_size,
            seed=args.seed,
        )
        clean_manifest.write_csv(frozen_manifest_path)
    distorted_plan = build_distorted_plan(
        clean_manifest=clean_manifest,
        runtime_root=paths.root,
        conditions=conditions,
    )
    distorted_plan.write_csv(paths.manifests / "distorted_variants_plan.csv")
    trial_rows = _load_or_build_trials(
        clean_manifest=clean_manifest,
        output_path=paths.trials / "clean_eval_trials.jsonl",
        negative_multiplier=args.negative_multiplier,
        seed=args.seed,
    )
    trial_arrays = {
        "left_indices": np.asarray([int(row["left_index"]) for row in trial_rows], dtype=np.int32),
        "right_indices": np.asarray(
            [int(row["right_index"]) for row in trial_rows],
            dtype=np.int32,
        ),
        "labels": np.asarray([int(row["label"]) for row in trial_rows], dtype=np.int8),
    }
    _write_json(
        paths.trials / "clean_eval_trials_summary.json",
        {
            "trial_count": int(trial_arrays["labels"].shape[0]),
            "positive_count": int((trial_arrays["labels"] == 1).sum()),
            "negative_count": int((trial_arrays["labels"] == 0).sum()),
            "negative_multiplier": args.negative_multiplier,
        },
    )

    model_results: dict[str, dict[str, Any]] = {}
    for model in models:
        print(f"[robustness] model={model.key} start", flush=True)
        progress = _read_json(paths.state / "progress.json", default={})
        model_rows: list[dict[str, Any]] = []
        drift_rows: list[dict[str, Any]] = []
        clean_embeddings = _extract_embeddings(
            model=model,
            condition=conditions[0],
            manifest=clean_manifest,
            condition_manifest_path=frozen_manifest_path,
            paths=paths,
            args=args,
        )
        clean_metrics, clean_scores = _score_condition(
            left_embeddings=clean_embeddings,
            right_embeddings=clean_embeddings,
            trial_arrays=trial_arrays,
            scores_path=paths.trials / model.key / "scores_clean.npy",
        )
        clean_row = {
            "model": model.key,
            "condition": "clean",
            "family": "clean",
            "severity": "clean",
            "trial_protocol": "clean-clean",
            **clean_metrics,
            "delta_eer_vs_clean": 0.0,
            "delta_min_dcf_vs_clean": 0.0,
            "normalized_degradation": 0.0,
        }
        model_rows.append(clean_row)
        for condition in conditions[1:]:
            print(
                f"[robustness] model={model.key} condition={condition.name} materialize/extract",
                flush=True,
            )
            condition_manifest_path = paths.manifests / "distorted" / f"{condition.name}.csv"
            condition_manifest = materialize_condition_audio(
                clean_manifest=clean_manifest,
                condition=condition,
                runtime_root=paths.root,
                source_data_root=Path(args.source_data_root),
                workers=args.distortion_workers,
                seed=args.seed,
                manifest_path=condition_manifest_path,
            )
            condition_embeddings = _extract_embeddings(
                model=model,
                condition=condition,
                manifest=condition_manifest,
                condition_manifest_path=condition_manifest_path,
                paths=paths,
                args=args,
            )
            metrics, scores = _score_condition(
                left_embeddings=clean_embeddings,
                right_embeddings=condition_embeddings,
                trial_arrays=trial_arrays,
                scores_path=paths.trials / model.key / f"scores_{condition.name}.npy",
            )
            row = {
                "model": model.key,
                "condition": condition.name,
                "family": condition.family,
                "severity": condition.severity,
                "trial_protocol": "clean-enroll_vs_condition-test",
                **metrics,
                "delta_eer_vs_clean": round(metrics["eer"] - clean_metrics["eer"], 6),
                "delta_min_dcf_vs_clean": round(metrics["min_dcf"] - clean_metrics["min_dcf"], 6),
                "normalized_degradation": round(
                    _normalized_degradation(
                        clean_eer=clean_metrics["eer"],
                        clean_min_dcf=clean_metrics["min_dcf"],
                        condition_eer=metrics["eer"],
                        condition_min_dcf=metrics["min_dcf"],
                    ),
                    6,
                ),
            }
            model_rows.append(row)
            drift_rows.append(
                {
                    "model": model.key,
                    "condition": condition.name,
                    "family": condition.family,
                    "severity": condition.severity,
                    **_compute_drift_metrics(
                        clean_embeddings=clean_embeddings,
                        distorted_embeddings=condition_embeddings,
                        clean_scores=clean_scores,
                        distorted_scores=scores,
                        trial_arrays=trial_arrays,
                    ),
                }
            )
            progress.setdefault(model.key, {})[condition.name] = "completed"
            _write_json(paths.state / "progress.json", progress)

        metrics_df = pl.DataFrame(model_rows)
        drift_df = pl.DataFrame(drift_rows)
        metrics_df.write_csv(paths.reports / f"{model.key}_main_metrics.csv")
        drift_df.write_csv(paths.reports / f"{model.key}_drift_metrics.csv")
        aggregate = _aggregate_model_summary(
            metrics_df=metrics_df,
            drift_df=drift_df,
            model=model,
        )
        _write_json(paths.reports / f"{model.key}_summary.json", aggregate)
        model_results[model.key] = {
            "main_metrics_path": str(paths.reports / f"{model.key}_main_metrics.csv"),
            "drift_metrics_path": str(paths.reports / f"{model.key}_drift_metrics.csv"),
            "summary_path": str(paths.reports / f"{model.key}_summary.json"),
            "summary": aggregate,
        }

    comparison_rows = _build_comparison_rows(model_results)
    pl.DataFrame(comparison_rows).write_csv(paths.reports / "model_comparison.csv")
    _write_json(
        paths.reports / "benchmark_summary.json",
        {
            "run_key": args.run_key,
            "completed": True,
            "source_manifest_csv": args.source_manifest_csv,
            "source_data_root": args.source_data_root,
            "frozen_clean_manifest": str(frozen_manifest_path),
            "distorted_plan": str(paths.manifests / "distorted_variants_plan.csv"),
            "trial_manifest": str(paths.trials / "clean_eval_trials.jsonl"),
            "model_results": model_results,
            "comparison_path": str(paths.reports / "model_comparison.csv"),
            "notes": [
                (
                    "Additive noise and reverb use deterministic synthetic distortions "
                    "because local noise/RIR banks were not materialized in the inspected "
                    "host paths."
                ),
                (
                    "CAM++ benchmark path uses the MS32 encoder checkpoint documented as "
                    "the embedding branch behind the final MS41 family pipeline."
                ),
                (
                    "Robustness metrics use clean enrollment embeddings against "
                    "condition-specific test embeddings."
                ),
            ],
        },
    )
    (paths.reports / "benchmark_summary.md").write_text(
        _render_markdown_summary(
            run_key=args.run_key,
            model_results=model_results,
            frozen_manifest_path=frozen_manifest_path,
            plan_path=paths.manifests / "distorted_variants_plan.csv",
        ),
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--source-data-root", required=True)
    parser.add_argument("--source-manifest-csv", required=True)
    parser.add_argument("--clean-set-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--negative-multiplier", type=int, default=5)
    parser.add_argument("--audio-stats-workers", type=int, default=8)
    parser.add_argument("--distortion-workers", type=int, default=8)
    parser.add_argument("--campp-checkpoint", required=True)
    parser.add_argument("--w2v-checkpoint", required=True)
    parser.add_argument("--w2v-backbone-path", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--campp-batch-size", type=int, default=64)
    parser.add_argument("--campp-frontend-workers", type=int, default=8)
    parser.add_argument("--campp-frontend-prefetch", type=int, default=128)
    parser.add_argument("--w2v-batch-size", type=int, default=1024)
    parser.add_argument("--w2v-num-workers", type=int, default=4)
    parser.add_argument("--w2v-prefetch-factor", type=int, default=1)
    parser.add_argument("--w2v-crop-seconds", type=float, default=6.0)
    parser.add_argument("--w2v-n-crops", type=int, default=3)
    return parser.parse_args(argv)


def _configure_offline_env(*, hf_home: Path) -> None:
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _build_paths(root: Path) -> BenchmarkPaths:
    return BenchmarkPaths(
        root=root,
        manifests=root / "manifests",
        reports=root / "reports",
        state=root / "state",
        distortions=root / "cache" / "distortions",
        embeddings=root / "cache" / "embeddings",
        trials=root / "cache" / "pairs_or_trials",
    )


def _ensure_runtime_dirs(paths: BenchmarkPaths) -> None:
    for path in (
        paths.root,
        paths.manifests,
        paths.manifests / "distorted",
        paths.reports,
        paths.state,
        paths.distortions,
        paths.embeddings,
        paths.trials,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _build_model_specs(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    w2v_checkpoint = Path(args.w2v_checkpoint)
    return (
        ModelSpec(
            key="campp_ms41_family",
            label="CAM++ MS41 family encoder",
            kind="campp",
            checkpoint_path=Path(args.campp_checkpoint),
        ),
        ModelSpec(
            key="w2v1j_teacher_peft_stage3",
            label="W2V1j teacher-PEFT stage3",
            kind="teacher_peft",
            checkpoint_path=w2v_checkpoint,
            feature_extractor_path=(
                w2v_checkpoint / "feature_extractor" / "preprocessor_config.json"
            ),
            checkpoint_metadata_path=w2v_checkpoint / "checkpoint_metadata.json",
            backbone_path=Path(args.w2v_backbone_path),
        ),
    )


def _extract_embeddings(
    *,
    model: ModelSpec,
    condition: DistortionCondition,
    manifest: pl.DataFrame,
    condition_manifest_path: Path,
    paths: BenchmarkPaths,
    args: argparse.Namespace,
) -> np.ndarray:
    output_dir = paths.embeddings / model.key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"embeddings_{condition.name}.npy"
    if output_path.is_file():
        return np.load(output_path)
    device = str(args.device)
    search_device = str(args.search_device)
    command = [
        sys.executable,
        "research/scripts/robustness/extract_embeddings.py",
        "--model-kind",
        model.kind,
        "--model-key",
        model.key,
        "--condition-name",
        condition.name,
        "--manifest-csv",
        str(condition_manifest_path),
        "--output-path",
        str(output_path),
        "--data-root",
        str(args.source_data_root),
        "--checkpoint-path",
        str(model.checkpoint_path),
        "--device",
        device,
        "--search-device",
        search_device,
        "--campp-batch-size",
        str(args.campp_batch_size),
        "--campp-frontend-workers",
        str(args.campp_frontend_workers),
        "--campp-frontend-prefetch",
        str(args.campp_frontend_prefetch),
        "--campp-frontend-cache-dir",
        str(paths.root / "cache" / "campp_frontend"),
        "--w2v-batch-size",
        str(args.w2v_batch_size),
        "--w2v-num-workers",
        str(args.w2v_num_workers),
        "--w2v-prefetch-factor",
        str(args.w2v_prefetch_factor),
        "--w2v-crop-seconds",
        str(args.w2v_crop_seconds),
        "--w2v-n-crops",
        str(args.w2v_n_crops),
    ]
    if model.backbone_path is not None:
        command.extend(["--backbone-path", str(model.backbone_path)])
    subprocess.run(
        command,
        check=True,
        cwd=PROJECT_ROOT,
    )
    return np.load(output_path)


def _extract_teacher_peft_embeddings(
    *,
    model: ModelSpec,
    manifest: pl.DataFrame,
    output_path: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    crop_seconds: float,
    n_crops: int,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    checkpoint_dir, _, feature_extractor, encoder = load_teacher_peft_encoder_from_checkpoint(
        checkpoint_path=model.checkpoint_path,
        trainable=False,
    )
    del checkpoint_dir
    encoder = encoder.to(device)
    encoder.eval()
    dataset = _TeacherPeftEvalDataset(
        paths=(
            manifest["resolved_path"].to_list()
            if "resolved_path" in manifest.columns
            else manifest["filepath"].to_list()
        ),
        row_indices=np.asarray(manifest["clean_index"], dtype=np.int64),
        crop_samples=int(round(crop_seconds * TARGET_SAMPLE_RATE_HZ)),
        n_crops=n_crops,
    )
    loader = DataLoader(
        cast(Any, dataset),
        batch_size=max(1, batch_size // max(1, n_crops)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        collate_fn=_TeacherPeftEvalCollator(
            feature_extractor=feature_extractor,
            sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
        ),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    sums: np.ndarray | None = None
    counts = np.zeros(manifest.height, dtype=np.int32)
    with (
        torch.inference_mode(),
        torch.amp.autocast(device, enabled=device == "cuda"),
    ):
        for batch in loader:
            inputs = {
                key: value.to(device, non_blocking=device == "cuda")
                for key, value in batch.model_inputs.items()
            }
            values = encoder(**inputs).detach().float().cpu().numpy().astype(np.float32, copy=False)
            values = l2_normalize_rows(values)
            owners = batch.owners.cpu().numpy().astype(np.int64, copy=False)
            if sums is None:
                sums = np.zeros((manifest.height, values.shape[1]), dtype=np.float32)
            np.add.at(sums, owners, values)
            np.add.at(counts, owners, 1)
    if sums is None:
        raise RuntimeError("Teacher-PEFT extraction produced no embeddings.")
    embeddings = l2_normalize_rows(sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    np.save(output_path, embeddings)
    return embeddings


def _load_or_build_trials(
    *,
    clean_manifest: pl.DataFrame,
    output_path: Path,
    negative_multiplier: int,
    seed: int,
) -> list[dict[str, Any]]:
    if output_path.is_file():
        return [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    rows = clean_manifest.to_dicts()
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_duration_bin: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)
        by_duration_bin[int(row["duration_bin"])].append(row)

    positive_trials: list[dict[str, Any]] = []
    for speaker_id in sorted(by_speaker):
        speaker_rows = sorted(by_speaker[speaker_id], key=lambda row: int(row["clean_index"]))
        for left, right in combinations(speaker_rows, 2):
            positive_trials.append(_trial_row(left=left, right=right, label=1))
            positive_trials.append(_trial_row(left=right, right=left, label=1))

    negative_trials: dict[tuple[int, int], dict[str, Any]] = {}
    for positive in positive_trials:
        left = rows[int(positive["left_index"])]
        right = rows[int(positive["right_index"])]
        pool = [
            candidate
            for candidate in by_duration_bin[int(right["duration_bin"])]
            if candidate["speaker_id"] != left["speaker_id"]
        ]
        if not pool:
            pool = [
                candidate for candidate in rows if candidate["speaker_id"] != left["speaker_id"]
            ]
        for index in range(negative_multiplier):
            candidate_index = (
                seed + int(left["clean_index"]) * 17 + int(right["clean_index"]) * 31 + index
            ) % len(pool)
            candidate = pool[candidate_index]
            key = (int(left["clean_index"]), int(candidate["clean_index"]))
            negative_trials.setdefault(key, _trial_row(left=left, right=candidate, label=0))
    all_trials = positive_trials + list(negative_trials.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_trials),
        encoding="utf-8",
    )
    return all_trials


def _trial_row(*, left: dict[str, Any], right: dict[str, Any], label: int) -> dict[str, Any]:
    return {
        "left_index": int(left["clean_index"]),
        "right_index": int(right["clean_index"]),
        "left_item_id": str(left["item_id"]),
        "right_item_id": str(right["item_id"]),
        "left_speaker_id": str(left["speaker_id"]),
        "right_speaker_id": str(right["speaker_id"]),
        "label": int(label),
    }


def _score_condition(
    *,
    left_embeddings: np.ndarray,
    right_embeddings: np.ndarray,
    trial_arrays: dict[str, np.ndarray],
    scores_path: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    scores = cosine_score_pairs(
        left_embeddings[trial_arrays["left_indices"]],
        right_embeddings[trial_arrays["right_indices"]],
        normalize=True,
    ).astype(np.float32, copy=False)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(scores_path, scores)
    metrics = compute_verification_metrics_from_arrays(
        labels=trial_arrays["labels"],
        scores=scores,
    )
    positive_scores = scores[trial_arrays["labels"] == 1]
    negative_scores = scores[trial_arrays["labels"] == 0]
    return (
        {
            "eer": metrics.eer,
            "min_dcf": metrics.min_dcf,
            "eer_threshold": metrics.eer_threshold,
            "min_dcf_threshold": metrics.min_dcf_threshold,
            "trial_count": int(metrics.trial_count),
            "positive_count": int(metrics.positive_count),
            "negative_count": int(metrics.negative_count),
            "mean_positive_score": round(float(positive_scores.mean()), 6),
            "mean_negative_score": round(float(negative_scores.mean()), 6),
            "score_gap": round(float(positive_scores.mean() - negative_scores.mean()), 6),
        },
        scores,
    )


def _compute_drift_metrics(
    *,
    clean_embeddings: np.ndarray,
    distorted_embeddings: np.ndarray,
    clean_scores: np.ndarray,
    distorted_scores: np.ndarray,
    trial_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    same_clip_cosine = np.clip(np.sum(clean_embeddings * distorted_embeddings, axis=1), -1.0, 1.0)
    same_clip_l2 = np.linalg.norm(clean_embeddings - distorted_embeddings, axis=1)
    labels = trial_arrays["labels"]
    clean_positive = clean_scores[labels == 1]
    clean_negative = clean_scores[labels == 0]
    distorted_positive = distorted_scores[labels == 1]
    distorted_negative = distorted_scores[labels == 0]
    clean_gap = float(clean_positive.mean() - clean_negative.mean())
    distorted_gap = float(distorted_positive.mean() - distorted_negative.mean())
    retrieval = distorted_embeddings @ clean_embeddings.T
    self_retrieval_at1 = float(
        (np.argmax(retrieval, axis=1) == np.arange(retrieval.shape[0])).mean()
    )
    return {
        "same_clip_cosine_mean": round(float(same_clip_cosine.mean()), 6),
        "same_clip_cosine_p95": round(float(np.quantile(same_clip_cosine, 0.95)), 6),
        "same_clip_l2_mean": round(float(same_clip_l2.mean()), 6),
        "same_clip_l2_p95": round(float(np.quantile(same_clip_l2, 0.95)), 6),
        "positive_mean_score": round(float(distorted_positive.mean()), 6),
        "negative_mean_score": round(float(distorted_negative.mean()), 6),
        "separation_gap": round(distorted_gap, 6),
        "separation_gap_delta_vs_clean": round(distorted_gap - clean_gap, 6),
        "self_retrieval_at1": round(self_retrieval_at1, 6),
    }


def _normalized_degradation(
    *,
    clean_eer: float,
    clean_min_dcf: float,
    condition_eer: float,
    condition_min_dcf: float,
) -> float:
    eer_term = max((condition_eer - clean_eer) / max(clean_eer, 1e-6), 0.0)
    dcf_term = max((condition_min_dcf - clean_min_dcf) / max(clean_min_dcf, 1e-6), 0.0)
    return 0.5 * eer_term + 0.5 * dcf_term


def _aggregate_model_summary(
    *,
    metrics_df: pl.DataFrame,
    drift_df: pl.DataFrame,
    model: ModelSpec,
) -> dict[str, Any]:
    clean_row = metrics_df.filter(pl.col("condition") == "clean").to_dicts()[0]
    distorted = metrics_df.filter(pl.col("condition") != "clean")
    normalized_degradation_values = np.asarray(
        distorted.get_column("normalized_degradation"),
        dtype=np.float64,
    )
    if normalized_degradation_values.size == 0:
        raise RuntimeError("Distorted metrics are empty; cannot aggregate degradation.")
    family_summary = (
        distorted.group_by("family")
        .agg(
            pl.mean("normalized_degradation").alias("mean_normalized_degradation"),
            pl.mean("delta_eer_vs_clean").alias("mean_delta_eer"),
            pl.mean("delta_min_dcf_vs_clean").alias("mean_delta_min_dcf"),
        )
        .sort("mean_normalized_degradation", descending=True)
        .to_dicts()
    )
    drift_summary = (
        drift_df.group_by("family")
        .agg(
            pl.mean("same_clip_cosine_mean").alias("mean_same_clip_cosine"),
            pl.mean("same_clip_l2_mean").alias("mean_same_clip_l2"),
            pl.mean("separation_gap_delta_vs_clean").alias("mean_separation_delta"),
        )
        .sort("mean_same_clip_l2", descending=True)
        .to_dicts()
    )
    return {
        "model": asdict(model),
        "clean": clean_row,
        "aggregated_normalized_degradation": round(
            float(normalized_degradation_values.mean()),
            6,
        ),
        "worst_condition": distorted.sort("normalized_degradation", descending=True).to_dicts()[0],
        "family_summary": family_summary,
        "drift_family_summary": drift_summary,
    }


def _build_comparison_rows(model_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, payload in sorted(model_results.items()):
        summary = payload["summary"]
        rows.append(
            {
                "model": model_key,
                "clean_eer": summary["clean"]["eer"],
                "clean_min_dcf": summary["clean"]["min_dcf"],
                "aggregated_normalized_degradation": summary["aggregated_normalized_degradation"],
                "worst_condition": summary["worst_condition"]["condition"],
                "worst_condition_degradation": summary["worst_condition"]["normalized_degradation"],
            }
        )
    return rows


def _render_markdown_summary(
    *,
    run_key: str,
    model_results: dict[str, dict[str, Any]],
    frozen_manifest_path: Path,
    plan_path: Path,
) -> str:
    lines = [
        f"# Robustness Benchmark Summary: {run_key}",
        "",
        f"- Frozen clean manifest: `{frozen_manifest_path}`",
        f"- Distorted plan: `{plan_path}`",
        "- Distorted protocol: clean enrollment embeddings vs condition-specific test embeddings.",
        (
            "- Noise and reverb are deterministic synthetic transforms because local "
            "noise/RIR banks were not materialized on the inspected host paths."
        ),
        "",
    ]
    for model_key, payload in sorted(model_results.items()):
        summary = payload["summary"]
        lines.extend(
            [
                f"## {model_key}",
                "",
                f"- Clean EER: `{summary['clean']['eer']}`",
                f"- Clean minDCF: `{summary['clean']['min_dcf']}`",
                (
                    "- Aggregated normalized degradation: "
                    f"`{summary['aggregated_normalized_degradation']}`"
                ),
                (
                    f"- Worst condition: `{summary['worst_condition']['condition']}` "
                    f"(`{summary['worst_condition']['normalized_degradation']}`)"
                ),
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return dict(default)
    return payload


@dataclass(frozen=True, slots=True)
class _EvalCropExample:
    row_index: int
    crops: list[np.ndarray]


@dataclass(frozen=True, slots=True)
class _EvalCropBatch:
    model_inputs: dict[str, Any]
    owners: Any


class _TeacherPeftEvalDataset:
    def __init__(
        self,
        *,
        paths: list[str],
        row_indices: np.ndarray,
        crop_samples: int,
        n_crops: int,
    ) -> None:
        self._paths = paths
        self._row_indices = row_indices.astype(np.int64, copy=False)
        self._crop_samples = crop_samples
        self._n_crops = n_crops

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> _EvalCropExample:
        waveform = load_eval_waveform(Path(self._paths[index]), trim=False)
        crops = eval_crops(
            waveform,
            crop_samples=self._crop_samples,
            n_crops=self._n_crops,
        )
        return _EvalCropExample(
            row_index=int(self._row_indices[index]),
            crops=crops,
        )


class _TeacherPeftEvalCollator:
    def __init__(self, *, feature_extractor: Any, sample_rate_hz: int) -> None:
        self._feature_extractor = feature_extractor
        self._sample_rate_hz = sample_rate_hz

    def __call__(self, batch: list[_EvalCropExample]) -> _EvalCropBatch:
        import torch

        waveforms: list[np.ndarray] = []
        owners: list[int] = []
        for example in batch:
            waveforms.extend(example.crops)
            owners.extend([example.row_index] * len(example.crops))
        encoded = self._feature_extractor(
            waveforms,
            sampling_rate=self._sample_rate_hz,
            padding=True,
            return_tensors="pt",
        )
        model_inputs: dict[str, Any] = {}
        for key, value in encoded.items():
            if key == "attention_mask":
                model_inputs[key] = value.to(dtype=torch.int32)
            else:
                model_inputs[key] = value.to(dtype=torch.float32)
        return _EvalCropBatch(
            model_inputs=model_inputs,
            owners=torch.tensor(owners, dtype=torch.long),
        )


__all__ = ["main"]
