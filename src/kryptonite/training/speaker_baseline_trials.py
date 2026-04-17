"""Trial generation and scoring helpers for manifest-backed speaker baselines."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.eval import CohortEmbeddingBankSelection, build_cohort_embedding_bank
from kryptonite.models import l2_normalize_embeddings

from .speaker_baseline_types import (
    EMBEDDINGS_FILE_NAME,
    SCORES_FILE_NAME,
    TRIALS_FILE_NAME,
    EmbeddingExportSummary,
    ScoredTrialsArtifacts,
    ScoreSummary,
    TrialManifestArtifacts,
)


def load_or_generate_trials(
    *,
    output_root: Path,
    configured_trials_manifest: str | None,
    metadata_rows: list[dict[str, Any]],
    project_root: Path,
    emit_progress: Callable[[str], None] | None = None,
) -> TrialManifestArtifacts:
    from kryptonite.deployment import resolve_project_path

    if configured_trials_manifest:
        trials_path = resolve_project_path(str(project_root), configured_trials_manifest)
        if emit_progress is not None:
            emit_progress(f"[trials] source=configured path={trials_path}")
        return TrialManifestArtifacts(
            trials_path=str(trials_path),
            trial_count=_count_trial_manifest_rows(trials_path),
        )

    enrollment_rows = [row for row in metadata_rows if row.get("role") == "enrollment"]
    test_rows = [row for row in metadata_rows if row.get("role") == "test"]
    trials_path = output_root / TRIALS_FILE_NAME
    written_trials = 0
    flush_interval = 100_000
    progress_interval = 500_000
    pending_lines: list[str] = []
    started_at = time.monotonic()
    estimated_trials = (
        len(enrollment_rows) * len(test_rows)
        if enrollment_rows and test_rows
        else math.comb(len(metadata_rows), 2)
    )
    with trials_path.open("w", encoding="utf-8") as handle:
        if enrollment_rows and test_rows:
            for left in enrollment_rows:
                for right in test_rows:
                    if left["trial_item_id"] == right["trial_item_id"]:
                        continue
                    pending_lines.append(_serialize_trial_row(left=left, right=right))
                    written_trials += 1
                    if emit_progress is not None and (
                        written_trials == 1 or written_trials % progress_interval == 0
                    ):
                        elapsed_seconds = max(time.monotonic() - started_at, 1e-6)
                        trial_per_sec = written_trials / elapsed_seconds
                        remaining_trials = max(estimated_trials - written_trials, 0)
                        eta_seconds = (
                            remaining_trials / trial_per_sec if trial_per_sec > 0.0 else 0.0
                        )
                        emit_progress(
                            "[trials] "
                            f"generated={written_trials}/{estimated_trials} "
                            f"trial_per_sec={trial_per_sec:.2f} "
                            f"elapsed={_format_runtime_duration(elapsed_seconds)} "
                            f"eta={_format_runtime_duration(eta_seconds)}"
                        )
                    if len(pending_lines) >= flush_interval:
                        handle.writelines(pending_lines)
                        pending_lines.clear()
        else:
            for left, right in combinations(metadata_rows, 2):
                if left["trial_item_id"] == right["trial_item_id"]:
                    continue
                pending_lines.append(_serialize_trial_row(left=left, right=right))
                written_trials += 1
                if emit_progress is not None and (
                    written_trials == 1 or written_trials % progress_interval == 0
                ):
                    elapsed_seconds = max(time.monotonic() - started_at, 1e-6)
                    trial_per_sec = written_trials / elapsed_seconds
                    remaining_trials = max(estimated_trials - written_trials, 0)
                    eta_seconds = remaining_trials / trial_per_sec if trial_per_sec > 0.0 else 0.0
                    emit_progress(
                        "[trials] "
                        f"generated={written_trials}/{estimated_trials} "
                        f"trial_per_sec={trial_per_sec:.2f} "
                        f"elapsed={_format_runtime_duration(elapsed_seconds)} "
                        f"eta={_format_runtime_duration(eta_seconds)}"
                    )
                if len(pending_lines) >= flush_interval:
                    handle.writelines(pending_lines)
                    pending_lines.clear()
        if pending_lines:
            handle.writelines(pending_lines)

    return TrialManifestArtifacts(
        trials_path=str(trials_path),
        trial_count=written_trials,
    )


def score_trials(
    *,
    output_root: Path,
    trials_path: Path,
    metadata_rows: list[dict[str, Any]],
    trial_rows: Sequence[dict[str, Any]] | None,
    embeddings_path: Path | None = None,
) -> ScoreSummary:
    return score_trials_detailed(
        output_root=output_root,
        trials_path=trials_path,
        metadata_rows=metadata_rows,
        trial_rows=trial_rows,
        embeddings_path=embeddings_path,
        trial_count_hint=(None if trial_rows is None else len(trial_rows)),
    ).summary


def score_trials_detailed(
    *,
    output_root: Path,
    trials_path: Path,
    metadata_rows: list[dict[str, Any]],
    trial_rows: Sequence[dict[str, Any]] | None,
    embeddings_path: Path | None = None,
    trial_count_hint: int | None = None,
    emit_progress: Callable[[str], None] | None = None,
    chunk_size: int = 250_000,
) -> ScoredTrialsArtifacts:
    resolved_embeddings_path = embeddings_path or (output_root / EMBEDDINGS_FILE_NAME)
    embeddings_payload = np.load(resolved_embeddings_path)
    embeddings = l2_normalize_embeddings(
        np.asarray(embeddings_payload["embeddings"], dtype=np.float32),
        field_name="embeddings",
    ).astype(np.float32, copy=False)

    id_to_index: dict[str, int] = {}
    for index, row in enumerate(metadata_rows):
        id_to_index[row["trial_item_id"]] = index
        audio_path = row.get("audio_path")
        if audio_path:
            id_to_index.setdefault(str(audio_path), index)
            id_to_index.setdefault(Path(str(audio_path)).name, index)
        utterance_id = row.get("utterance_id")
        if utterance_id:
            id_to_index.setdefault(utterance_id, index)

    resolved_trial_count = trial_count_hint
    if resolved_trial_count is None:
        resolved_trial_count = _count_trial_manifest_rows(trials_path)
    labels = np.empty((resolved_trial_count,), dtype=np.int8)
    scores = np.empty((resolved_trial_count,), dtype=np.float32)
    missing_embedding_count = 0
    written_rows = 0
    scores_path = output_root / SCORES_FILE_NAME
    processed_trials = 0
    progress_interval = min(max(10_000, resolved_trial_count // 50), 500_000)
    started_at = time.monotonic()
    with scores_path.open("w", encoding="utf-8") as handle:
        chunk_iterable = (
            _chunk_sequence(trial_rows, chunk_size)
            if trial_rows is not None
            else _iter_trial_rows_from_jsonl(trials_path, chunk_size=chunk_size)
        )
        for chunk in chunk_iterable:
            (
                chunk_lines,
                chunk_labels,
                chunk_scores,
                chunk_missing_count,
            ) = _score_trial_chunk(
                chunk=chunk,
                id_to_index=id_to_index,
                embeddings=embeddings,
            )
            processed_trials += len(chunk)
            missing_embedding_count += chunk_missing_count
            if chunk_lines:
                next_index = written_rows + len(chunk_lines)
                labels[written_rows:next_index] = chunk_labels
                scores[written_rows:next_index] = chunk_scores
                handle.writelines(chunk_lines)
                written_rows = next_index
            if emit_progress is not None and (
                processed_trials == resolved_trial_count
                or processed_trials <= len(chunk)
                or processed_trials % progress_interval == 0
            ):
                elapsed_seconds = max(time.monotonic() - started_at, 1e-6)
                trials_per_second = processed_trials / elapsed_seconds
                remaining_trials = max(resolved_trial_count - processed_trials, 0)
                eta_seconds = (
                    remaining_trials / trials_per_second if trials_per_second > 0.0 else 0.0
                )
                emit_progress(
                    "[score] "
                    f"trial {processed_trials}/{resolved_trial_count} "
                    f"written={written_rows} missing={missing_embedding_count} "
                    f"trial_per_sec={trials_per_second:.2f} "
                    f"elapsed={_format_runtime_duration(elapsed_seconds)} "
                    f"eta={_format_runtime_duration(eta_seconds)}"
                )

    labels = labels[:written_rows]
    scores = scores[:written_rows]
    positive_mask = labels == 1
    positive_scores = scores[positive_mask]
    negative_scores = scores[~positive_mask]

    mean_positive = mean_or_none(positive_scores.tolist())
    mean_negative = mean_or_none(negative_scores.tolist())
    score_gap = None
    if mean_positive is not None and mean_negative is not None:
        score_gap = round(mean_positive - mean_negative, 6)

    summary = ScoreSummary(
        trials_path=str(trials_path),
        scores_path=str(scores_path),
        trial_count=written_rows,
        positive_count=int(positive_mask.sum()),
        negative_count=int((~positive_mask).sum()),
        missing_embedding_count=missing_embedding_count,
        mean_positive_score=mean_positive,
        mean_negative_score=mean_negative,
        score_gap=score_gap,
    )
    return ScoredTrialsArtifacts(
        summary=summary,
        labels=labels,
        scores=scores,
    )


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _serialize_trial_row(*, left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            {
                "left_id": left["trial_item_id"],
                "right_id": right["trial_item_id"],
                "left_speaker_id": left["speaker_id"],
                "right_speaker_id": right["speaker_id"],
                "label": int(left["speaker_id"] == right["speaker_id"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _count_trial_manifest_rows(trials_path: Path) -> int:
    row_count = 0
    with trials_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Trials manifest must contain object JSONL rows: {trials_path}:{line_number}"
                )
            row_count += 1
    if row_count <= 0:
        raise ValueError(f"No trial rows found in {trials_path}")
    return row_count


def _chunk_sequence(
    values: Sequence[dict[str, Any]],
    chunk_size: int,
) -> Iterable[list[dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])


def _iter_trial_rows_from_jsonl(
    trials_path: Path,
    *,
    chunk_size: int,
) -> Iterable[list[dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    pending_rows: list[dict[str, Any]] = []
    with trials_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSONL rows in {trials_path}:{line_number}")
            pending_rows.append(payload)
            if len(pending_rows) >= chunk_size:
                yield pending_rows
                pending_rows = []
    if pending_rows:
        yield pending_rows


def _score_trial_chunk(
    *,
    chunk: Sequence[dict[str, Any]],
    id_to_index: Mapping[str, int],
    embeddings: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    left_indices: list[int] = []
    right_indices: list[int] = []
    normalized_rows: list[dict[str, Any]] = []
    labels: list[int] = []
    missing_embedding_count = 0
    for row in chunk:
        left_id = str(row.get("left_id", row.get("left_audio", "")))
        right_id = str(row.get("right_id", row.get("right_audio", "")))
        left_index = id_to_index.get(left_id)
        right_index = id_to_index.get(right_id)
        if left_index is None or right_index is None:
            missing_embedding_count += 1
            continue
        left_indices.append(left_index)
        right_indices.append(right_index)
        labels.append(int(row["label"]))
        normalized_rows.append({**row, "left_id": left_id, "right_id": right_id})

    if not left_indices:
        return (
            [],
            np.empty((0,), dtype=np.int8),
            np.empty((0,), dtype=np.float32),
            missing_embedding_count,
        )

    left_embeddings = embeddings[np.asarray(left_indices, dtype=np.int32)]
    right_embeddings = embeddings[np.asarray(right_indices, dtype=np.int32)]
    chunk_scores = np.einsum("ij,ij->i", left_embeddings, right_embeddings, optimize=True).astype(
        np.float32,
        copy=False,
    )
    chunk_labels = np.asarray(labels, dtype=np.int8)
    chunk_lines = [
        json.dumps(
            {**row, "score": round(float(score_value), 8)},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row, score_value in zip(normalized_rows, chunk_scores, strict=True)
    ]
    return chunk_lines, chunk_labels, chunk_scores, missing_embedding_count


def _format_runtime_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def build_default_cohort_bank(
    *,
    output_root: Path,
    embedding_summary: EmbeddingExportSummary,
    train_manifest_path: str,
    trials_path: Path,
    project_root: Path,
):
    return build_cohort_embedding_bank(
        project_root=project_root,
        output_root=output_root,
        embeddings_path=embedding_summary.embeddings_path,
        metadata_path=embedding_summary.metadata_parquet_path,
        selection=CohortEmbeddingBankSelection(
            trial_paths=(str(trials_path),),
            validation_manifest_paths=(train_manifest_path,),
            strict_speaker_disjointness=False,
            allow_trial_overlap_fallback=True,
            point_id_field="atlas_point_id",
            embeddings_key="embeddings",
            ids_key="point_ids",
        ),
    )
