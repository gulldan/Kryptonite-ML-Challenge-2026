"""Builder for reproducible TAS-norm experiment decisions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from kryptonite.deployment import resolve_project_path
from kryptonite.repro import fingerprint_path

from .cohort_bank import COHORT_SUMMARY_JSON_NAME, build_cohort_embedding_bank
from .tas_norm import (
    apply_tas_norm_model,
    fit_tas_norm_model,
    prepare_tas_norm_feature_batch,
)
from .tas_norm_experiment_config import TasNormExperimentConfig
from .tas_norm_experiment_models import (
    BuiltTasNormExperiment,
    TasNormArtifactRef,
    TasNormExperimentCheck,
    TasNormExperimentReport,
    TasNormExperimentSummary,
    TasNormMetricSnapshot,
    TasNormSplitSummary,
)
from .verification_data import load_verification_score_rows
from .verification_metrics import compute_verification_metrics, normalize_verification_score_rows

_IMPLEMENTATION_SCOPE = (
    "Repo-native TAS experiment that keeps a frozen cohort bank and trains a lightweight "
    "logistic head over raw-score, AS-norm, and cohort-stat features. This is a controlled "
    "go/no-go experiment, not a full learnable-impostor-embedding reproduction."
)

_ARTIFACT_DESCRIPTIONS = {
    "scores": "Raw verification score JSONL used as the experiment baseline.",
    "trials": "Verification trials JSONL used for deterministic cohort-bank exclusion.",
    "metadata": "Embedding-export metadata aligned with the verification embeddings.",
    "embeddings": "Verification embedding matrix used to derive AS/TAS cohort statistics.",
    "cohort_bank": "Frozen cohort bank consumed by AS-norm and the TAS experiment.",
}


def build_tas_norm_experiment_report(
    config: TasNormExperimentConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> BuiltTasNormExperiment:
    resolved_project_root = _resolve_project_root(project_root)
    source_config_file = None if config_path is None else Path(config_path).resolve()
    source_config_sha256 = None
    if source_config_file is not None:
        source_fingerprint = fingerprint_path(source_config_file)
        source_config_sha256 = (
            None if not bool(source_fingerprint["exists"]) else str(source_fingerprint["sha256"])
        )

    cohort_bank_root = resolve_project_path(
        str(resolved_project_root),
        config.artifacts.cohort_bank_output_root,
    )
    cohort_built_during_run = not (cohort_bank_root / COHORT_SUMMARY_JSON_NAME).exists()
    if cohort_built_during_run:
        build_cohort_embedding_bank(
            project_root=resolved_project_root,
            output_root=config.artifacts.cohort_bank_output_root,
            embeddings_path=config.artifacts.embeddings_path,
            metadata_path=config.artifacts.metadata_path,
            selection=config.cohort_selection,
        )

    score_rows = load_verification_score_rows(
        resolve_project_path(str(resolved_project_root), config.artifacts.scores_path)
    )
    batch = prepare_tas_norm_feature_batch(
        score_rows,
        embeddings_path=resolve_project_path(
            str(resolved_project_root), config.artifacts.embeddings_path
        ),
        metadata_path=resolve_project_path(
            str(resolved_project_root), config.artifacts.metadata_path
        ),
        cohort_bank_root=cohort_bank_root,
        top_k=config.runtime.top_k,
        std_epsilon=config.runtime.std_epsilon,
        exclude_matching_speakers=config.runtime.exclude_matching_speakers,
        embeddings_key=config.cohort_selection.embeddings_key,
        ids_key=config.cohort_selection.ids_key,
        point_id_field=config.cohort_selection.point_id_field,
    )

    train_indices, eval_indices = _build_stratified_split_indices(
        batch.rows,
        batch.labels,
        eval_fraction=config.runtime.eval_fraction,
        split_seed=config.runtime.split_seed,
    )

    train_features, train_labels, _ = batch.slice(train_indices)
    model = fit_tas_norm_model(
        train_features,
        train_labels,
        training_config=config.training,
        feature_names=batch.feature_names,
    )

    raw_train_rows = _build_raw_score_rows(batch, train_indices)
    raw_eval_rows = _build_raw_score_rows(batch, eval_indices)
    as_norm_train_rows = _build_as_norm_score_rows(batch, train_indices)
    as_norm_eval_rows = _build_as_norm_score_rows(batch, eval_indices)
    tas_norm_train_rows = apply_tas_norm_model(batch, model, indices=train_indices)
    tas_norm_eval_rows = apply_tas_norm_model(batch, model, indices=eval_indices)

    split_summary = _build_split_summary(
        labels=batch.labels,
        train_indices=train_indices,
        eval_indices=eval_indices,
        eval_fraction=config.runtime.eval_fraction,
        split_seed=config.runtime.split_seed,
    )
    raw_train = _build_metric_snapshot(raw_train_rows)
    raw_eval = _build_metric_snapshot(raw_eval_rows)
    as_norm_train = _build_metric_snapshot(as_norm_train_rows)
    as_norm_eval = _build_metric_snapshot(as_norm_eval_rows)
    tas_norm_train = _build_metric_snapshot(tas_norm_train_rows)
    tas_norm_eval = _build_metric_snapshot(tas_norm_eval_rows)

    artifacts = _build_artifact_refs(config=config, project_root=resolved_project_root)
    checks = tuple(
        _build_checks(
            config=config,
            split=split_summary,
            raw_eval=raw_eval,
            as_norm_eval=as_norm_eval,
            tas_norm_train=tas_norm_train,
            tas_norm_eval=tas_norm_eval,
        )
    )
    failed_checks = tuple(check.detail for check in checks if not check.passed)
    report = TasNormExperimentReport(
        title=config.title,
        report_id=config.report_id,
        candidate_label=config.candidate_label,
        summary_text=config.summary,
        output_root=str(resolve_project_path(str(resolved_project_root), config.output_root)),
        source_config_path=None if source_config_file is None else str(source_config_file),
        source_config_sha256=source_config_sha256,
        implementation_scope=_IMPLEMENTATION_SCOPE,
        cohort_bank_output_root=str(cohort_bank_root),
        cohort_built_during_run=cohort_built_during_run,
        feature_names=batch.feature_names,
        top_k=batch.top_k,
        effective_top_k=batch.effective_top_k,
        cohort_size=batch.cohort_size,
        embedding_dim=batch.embedding_dim,
        floored_std_count=batch.floored_std_count,
        excluded_same_speaker_count=batch.excluded_same_speaker_count,
        split=split_summary,
        raw_train=raw_train,
        raw_eval=raw_eval,
        as_norm_train=as_norm_train,
        as_norm_eval=as_norm_eval,
        tas_norm_train=tas_norm_train,
        tas_norm_eval=tas_norm_eval,
        model=model,
        training_config=config.training.to_dict(),
        gates=config.gates.to_dict(),
        artifacts=artifacts,
        checks=checks,
        validation_commands=config.validation_commands,
        notes=config.notes,
        summary=TasNormExperimentSummary(
            decision="go" if not failed_checks else "no_go",
            passed_check_count=sum(1 for check in checks if check.passed),
            failed_check_count=len(failed_checks),
            eval_winner=_select_eval_winner(
                raw_eval=raw_eval,
                as_norm_eval=as_norm_eval,
                tas_norm_eval=tas_norm_eval,
            ),
            key_blockers=failed_checks[:5],
        ),
    )
    return BuiltTasNormExperiment(
        report=report,
        as_norm_eval_score_rows=as_norm_eval_rows,
        tas_norm_eval_score_rows=tas_norm_eval_rows,
    )


def _resolve_project_root(project_root: Path | str | None) -> Path:
    if project_root is None:
        return resolve_project_path(".", ".")
    return resolve_project_path(str(project_root), ".")


def _build_stratified_split_indices(
    rows: list[dict[str, object]],
    labels: np.ndarray,
    *,
    eval_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for label_value in (0, 1):
        label_indices = [index for index, value in enumerate(labels) if int(value) == label_value]
        if len(label_indices) < 2:
            raise ValueError(
                "TAS-norm experiments require at least two trials per class to split train/eval."
            )
        label_indices.sort(
            key=lambda index: _stable_split_hash(
                rows[index],
                label=int(label_value),
                split_seed=split_seed,
            )
        )
        eval_count = max(1, int(round(len(label_indices) * eval_fraction)))
        eval_count = min(eval_count, len(label_indices) - 1)
        eval_indices.extend(label_indices[:eval_count])
        train_indices.extend(label_indices[eval_count:])

    return (
        np.asarray(sorted(train_indices), dtype=np.int64),
        np.asarray(sorted(eval_indices), dtype=np.int64),
    )


def _stable_split_hash(row: dict[str, object], *, label: int, split_seed: int) -> str:
    left_id = str(row.get("left_id", row.get("left_audio", ""))).strip()
    right_id = str(row.get("right_id", row.get("right_audio", ""))).strip()
    payload = f"{split_seed}:{label}:{left_id}:{right_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _build_raw_score_rows(
    batch,
    indices: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in [batch.rows[int(index)] for index in indices]:
        rows.append(
            {
                **row,
                "score": round(float(row["raw_score"]), 8),
                "score_normalization": "raw",
            }
        )
    return rows


def _build_as_norm_score_rows(
    batch,
    indices: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in [batch.rows[int(index)] for index in indices]:
        rows.append(
            {
                **row,
                "score": round(float(row["as_norm_score"]), 8),
                "score_normalization": "as-norm",
            }
        )
    return rows


def _build_split_summary(
    *,
    labels: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    eval_fraction: float,
    split_seed: int,
) -> TasNormSplitSummary:
    train_labels = labels[train_indices]
    eval_labels = labels[eval_indices]
    return TasNormSplitSummary(
        eval_fraction=round(float(eval_fraction), 6),
        split_seed=split_seed,
        train_trial_count=int(train_labels.shape[0]),
        train_positive_count=int(train_labels.sum()),
        train_negative_count=int(train_labels.shape[0] - train_labels.sum()),
        eval_trial_count=int(eval_labels.shape[0]),
        eval_positive_count=int(eval_labels.sum()),
        eval_negative_count=int(eval_labels.shape[0] - eval_labels.sum()),
    )


def _build_metric_snapshot(score_rows: list[dict[str, object]]) -> TasNormMetricSnapshot:
    metrics = compute_verification_metrics(score_rows)
    normalized_rows = normalize_verification_score_rows(score_rows)
    mean_score = float(
        np.asarray([row["score"] for row in normalized_rows], dtype=np.float64).mean()
    )
    return TasNormMetricSnapshot(
        trial_count=metrics.trial_count,
        positive_count=metrics.positive_count,
        negative_count=metrics.negative_count,
        eer=metrics.eer,
        min_dcf=metrics.min_dcf,
        mean_score=round(mean_score, 6),
    )


def _build_artifact_refs(
    *,
    config: TasNormExperimentConfig,
    project_root: Path,
) -> tuple[TasNormArtifactRef, ...]:
    specs = (
        ("scores", config.artifacts.scores_path),
        ("trials", config.artifacts.trials_path),
        ("metadata", config.artifacts.metadata_path),
        ("embeddings", config.artifacts.embeddings_path),
        ("cohort_bank", config.artifacts.cohort_bank_output_root),
    )
    refs: list[TasNormArtifactRef] = []
    for label, configured_path in specs:
        resolved_path = resolve_project_path(str(project_root), configured_path)
        fingerprint = fingerprint_path(resolved_path)
        refs.append(
            TasNormArtifactRef(
                label=label,
                configured_path=configured_path,
                resolved_path=str(resolved_path),
                exists=bool(fingerprint["exists"]),
                kind=str(fingerprint["kind"]),
                sha256=None if fingerprint["sha256"] is None else str(fingerprint["sha256"]),
                file_count=int(fingerprint["file_count"]),
                description=_ARTIFACT_DESCRIPTIONS[label],
            )
        )
    return tuple(refs)


def _build_checks(
    *,
    config: TasNormExperimentConfig,
    split: TasNormSplitSummary,
    raw_eval: TasNormMetricSnapshot,
    as_norm_eval: TasNormMetricSnapshot,
    tas_norm_train: TasNormMetricSnapshot,
    tas_norm_eval: TasNormMetricSnapshot,
) -> list[TasNormExperimentCheck]:
    checks = [
        TasNormExperimentCheck(
            name="Train split coverage",
            passed=(
                split.train_trial_count >= config.gates.min_train_trials
                and split.train_positive_count >= config.gates.min_train_positives
                and split.train_negative_count >= config.gates.min_train_negatives
            ),
            detail=(
                "Train split has "
                f"{split.train_trial_count} trials "
                f"({split.train_positive_count} pos / {split.train_negative_count} neg); "
                "required >= "
                f"{config.gates.min_train_trials} total, "
                f"{config.gates.min_train_positives} pos, "
                f"{config.gates.min_train_negatives} neg."
            ),
        ),
        TasNormExperimentCheck(
            name="Eval split coverage",
            passed=(
                split.eval_trial_count >= config.gates.min_eval_trials
                and split.eval_positive_count >= config.gates.min_eval_positives
                and split.eval_negative_count >= config.gates.min_eval_negatives
            ),
            detail=(
                "Eval split has "
                f"{split.eval_trial_count} trials "
                f"({split.eval_positive_count} pos / {split.eval_negative_count} neg); "
                "required >= "
                f"{config.gates.min_eval_trials} total, "
                f"{config.gates.min_eval_positives} pos, "
                f"{config.gates.min_eval_negatives} neg."
            ),
        ),
        _build_gain_check(
            name="Eval EER gain vs raw",
            baseline_value=raw_eval.eer,
            candidate_value=tas_norm_eval.eer,
            min_gain=config.gates.min_eer_gain_vs_raw,
            metric_name="EER",
            baseline_label="raw",
        ),
        _build_gain_check(
            name="Eval minDCF gain vs raw",
            baseline_value=raw_eval.min_dcf,
            candidate_value=tas_norm_eval.min_dcf,
            min_gain=config.gates.min_min_dcf_gain_vs_raw,
            metric_name="minDCF",
            baseline_label="raw",
        ),
        _build_gain_check(
            name="Eval EER gain vs AS-norm",
            baseline_value=as_norm_eval.eer,
            candidate_value=tas_norm_eval.eer,
            min_gain=config.gates.min_eer_gain_vs_as_norm,
            metric_name="EER",
            baseline_label="AS-norm",
        ),
        _build_gain_check(
            name="Eval minDCF gain vs AS-norm",
            baseline_value=as_norm_eval.min_dcf,
            candidate_value=tas_norm_eval.min_dcf,
            min_gain=config.gates.min_min_dcf_gain_vs_as_norm,
            metric_name="minDCF",
            baseline_label="AS-norm",
        ),
        _build_gap_check(
            name="Train/eval EER gap",
            train_value=tas_norm_train.eer,
            eval_value=tas_norm_eval.eer,
            max_gap=config.gates.max_train_eval_eer_gap,
            metric_name="EER",
        ),
        _build_gap_check(
            name="Train/eval minDCF gap",
            train_value=tas_norm_train.min_dcf,
            eval_value=tas_norm_eval.min_dcf,
            max_gap=config.gates.max_train_eval_min_dcf_gap,
            metric_name="minDCF",
        ),
    ]
    return checks


def _build_gain_check(
    *,
    name: str,
    baseline_value: float,
    candidate_value: float,
    min_gain: float,
    metric_name: str,
    baseline_label: str,
) -> TasNormExperimentCheck:
    if baseline_value <= 0.0:
        return TasNormExperimentCheck(
            name=name,
            passed=False,
            detail=(
                f"{baseline_label} eval {metric_name} is already {baseline_value:.6f}; "
                "there is no measurable headroom for TAS-norm on this split."
            ),
        )
    gain = baseline_value - candidate_value
    return TasNormExperimentCheck(
        name=name,
        passed=gain >= min_gain,
        detail=(
            f"{metric_name} gain vs {baseline_label} = {gain:.6f} "
            f"(required >= {min_gain:.6f}; baseline={baseline_value:.6f}, "
            f"tas={candidate_value:.6f})."
        ),
    )


def _build_gap_check(
    *,
    name: str,
    train_value: float,
    eval_value: float,
    max_gap: float | None,
    metric_name: str,
) -> TasNormExperimentCheck:
    gap = abs(eval_value - train_value)
    if max_gap is None:
        return TasNormExperimentCheck(
            name=name,
            passed=True,
            detail=f"{metric_name} gap = {gap:.6f}; gate disabled.",
        )
    return TasNormExperimentCheck(
        name=name,
        passed=gap <= max_gap,
        detail=f"{metric_name} gap = {gap:.6f} (max allowed {max_gap:.6f}).",
    )


def _select_eval_winner(
    *,
    raw_eval: TasNormMetricSnapshot,
    as_norm_eval: TasNormMetricSnapshot,
    tas_norm_eval: TasNormMetricSnapshot,
) -> str:
    candidates = {
        "raw": raw_eval,
        "as-norm": as_norm_eval,
        "tas-norm": tas_norm_eval,
    }
    winner, _ = min(
        candidates.items(),
        key=lambda item: (item[1].min_dcf, item[1].eer, -item[1].mean_score, item[0]),
    )
    return winner


__all__ = ["build_tas_norm_experiment_report"]
