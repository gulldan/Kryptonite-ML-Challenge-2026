"""Shared runtime helpers for manifest-backed speaker baseline recipes."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

import polars as pl
import torch
import torch.nn.functional as torch_functional
from torch import nn

from kryptonite.config import ChunkingConfig, ProjectConfig
from kryptonite.data import AudioLoadRequest, ManifestRow, load_manifest_audio
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.deployment import resolve_project_path
from kryptonite.eval import (
    CohortEmbeddingBankSelection,
    WrittenVerificationEvaluationReport,
    build_cohort_embedding_bank,
)
from kryptonite.features import (
    FbankExtractionRequest,
    FbankExtractor,
    UtteranceChunkingRequest,
    chunk_utterance,
    pool_chunk_tensors,
)
from kryptonite.models import cosine_score_pairs

from .baseline_reporting import relative_to_project, render_markdown_report
from .manifest_speaker_data import TrainingBatch
from .optimization_runtime import (
    TrainingOptimizationRuntime,
    run_classification_batches,
)

EMBEDDINGS_FILE_NAME = "dev_embeddings.npz"
EMBEDDING_METADATA_JSONL_NAME = "dev_embedding_metadata.jsonl"
EMBEDDING_METADATA_PARQUET_NAME = "dev_embedding_metadata.parquet"
TRAINING_SUMMARY_FILE_NAME = "training_summary.json"
SCORES_FILE_NAME = "dev_scores.jsonl"
TRIALS_FILE_NAME = "dev_trials.jsonl"
SCORE_SUMMARY_FILE_NAME = "score_summary.json"
REPRODUCIBILITY_FILE_NAME = "reproducibility_snapshot.json"


class EpochAwareDataset(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


class EpochAwareBatchSampler(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@dataclass(frozen=True, slots=True)
class EpochSummary:
    epoch: int
    mean_loss: float
    accuracy: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    device: str
    train_manifest: str
    dev_manifest: str
    provenance_ruleset: str
    provenance_initialization: str
    speaker_count: int
    train_row_count: int
    dev_row_count: int
    checkpoint_path: str
    epochs: tuple[EpochSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "train_manifest": self.train_manifest,
            "dev_manifest": self.dev_manifest,
            "provenance_ruleset": self.provenance_ruleset,
            "provenance_initialization": self.provenance_initialization,
            "speaker_count": self.speaker_count,
            "train_row_count": self.train_row_count,
            "dev_row_count": self.dev_row_count,
            "checkpoint_path": self.checkpoint_path,
            "epochs": [asdict(epoch) for epoch in self.epochs],
        }


@dataclass(frozen=True, slots=True)
class EmbeddingExportSummary:
    manifest_path: str
    embedding_dim: int
    utterance_count: int
    speaker_count: int
    embeddings_path: str
    metadata_jsonl_path: str
    metadata_parquet_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    trials_path: str
    scores_path: str
    trial_count: int
    positive_count: int
    negative_count: int
    missing_embedding_count: int
    mean_positive_score: float | None
    mean_negative_score: float | None
    score_gap: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpeakerBaselineRunArtifacts:
    output_root: str
    checkpoint_path: str
    training_summary_path: str
    embeddings_path: str
    embedding_metadata_jsonl_path: str
    embedding_metadata_parquet_path: str
    trials_path: str
    scores_path: str
    score_summary_path: str
    reproducibility_path: str
    report_path: str
    training_summary: TrainingSummary
    embedding_summary: EmbeddingExportSummary
    score_summary: ScoreSummary
    verification_report: WrittenVerificationEvaluationReport | None = None
    tracking_run_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "checkpoint_path": self.checkpoint_path,
            "training_summary_path": self.training_summary_path,
            "embeddings_path": self.embeddings_path,
            "embedding_metadata_jsonl_path": self.embedding_metadata_jsonl_path,
            "embedding_metadata_parquet_path": self.embedding_metadata_parquet_path,
            "trials_path": self.trials_path,
            "scores_path": self.scores_path,
            "score_summary_path": self.score_summary_path,
            "reproducibility_path": self.reproducibility_path,
            "report_path": self.report_path,
            "training_summary": self.training_summary.to_dict(),
            "embedding_summary": self.embedding_summary.to_dict(),
            "score_summary": self.score_summary.to_dict(),
            "verification_report": (
                None if self.verification_report is None else self.verification_report.to_dict()
            ),
            "tracking_run_dir": self.tracking_run_dir,
        }


def prepare_demo_artifacts_if_needed(
    *,
    project: ProjectConfig,
    train_manifest: str,
    dev_manifest: str,
    enabled: bool,
) -> None:
    if not enabled:
        return
    project_root = resolve_project_path(project.paths.project_root, ".")
    resolved_train_manifest = resolve_project_path(str(project_root), train_manifest)
    resolved_dev_manifest = resolve_project_path(str(project_root), dev_manifest)
    if resolved_train_manifest.exists() and resolved_dev_manifest.exists():
        return
    generate_demo_artifacts(config=project)


def resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(normalized)


def build_fixed_train_chunking_request(
    *,
    chunking: ChunkingConfig,
    baseline_name: str,
) -> UtteranceChunkingRequest:
    if chunking.train_num_crops != 1:
        raise ValueError(f"{baseline_name} baseline requires chunking.train_num_crops=1.")
    if not math.isclose(
        chunking.train_min_crop_seconds,
        chunking.train_max_crop_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"{baseline_name} baseline requires fixed-size train crops; set "
            "chunking.train_min_crop_seconds == chunking.train_max_crop_seconds."
        )
    return UtteranceChunkingRequest.from_config(chunking)


def train_epochs(
    *,
    model: nn.Module,
    classifier: nn.Module,
    criterion: nn.Module,
    training_runtime: TrainingOptimizationRuntime,
    loader: Iterable[TrainingBatch],
    dataset: EpochAwareDataset,
    sampler: EpochAwareBatchSampler | None,
    device: torch.device,
    max_epochs: int,
    tracker_run: Any | None,
) -> list[EpochSummary]:
    summaries: list[EpochSummary] = []
    for epoch in range(max_epochs):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        total_loss, total_correct, total_examples = run_classification_batches(
            model=model,
            classifier=classifier,
            criterion=criterion,
            training_runtime=training_runtime,
            loader=loader,
            device=device,
        )

        if total_examples == 0:
            raise ValueError("Training loader produced zero examples.")

        learning_rate = round(training_runtime.current_learning_rate(), 8)
        summary = EpochSummary(
            epoch=epoch + 1,
            mean_loss=round(total_loss / total_examples, 6),
            accuracy=round(total_correct / total_examples, 6),
            learning_rate=learning_rate,
        )
        summaries.append(summary)
        if tracker_run is not None:
            tracker_run.log_metrics(
                {
                    "train_loss": summary.mean_loss,
                    "train_accuracy": summary.accuracy,
                    "learning_rate": summary.learning_rate,
                },
                step=summary.epoch,
            )
        training_runtime.step_scheduler(mean_loss=summary.mean_loss)
    return summaries


def write_checkpoint(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    classifier: nn.Module,
    model_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    speaker_to_index: Mapping[str, int],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "model_config": dict(model_config),
            "baseline_config": dict(baseline_config),
            "speaker_to_index": dict(speaker_to_index),
        },
        checkpoint_path,
    )


def export_dev_embeddings(
    *,
    output_root: Path,
    model: nn.Module,
    rows: Sequence[ManifestRow],
    manifest_path: str,
    project_root: Path,
    audio_request: AudioLoadRequest,
    feature_request: FbankExtractionRequest,
    chunking: ChunkingConfig,
    device: torch.device,
    embedding_source: str,
) -> tuple[EmbeddingExportSummary, list[dict[str, Any]]]:
    model.eval()
    extractor = FbankExtractor(request=feature_request)
    manifest_metadata_lookup = _load_manifest_metadata_lookup(
        manifest_path=manifest_path,
        project_root=project_root,
    )
    metadata_rows: list[dict[str, Any]] = []
    embeddings: list[torch.Tensor] = []
    point_ids: list[str] = []

    with torch.no_grad():
        for index, row in enumerate(rows):
            loaded = load_manifest_audio(row, project_root=project_root, request=audio_request)
            eval_chunks = chunk_utterance(
                loaded.audio.waveform,
                sample_rate_hz=loaded.audio.sample_rate_hz,
                stage="eval",
                request=UtteranceChunkingRequest.from_config(chunking),
            )
            chunk_embeddings: list[torch.Tensor] = []
            for chunk in eval_chunks.chunks:
                features = extractor.extract(
                    chunk.waveform,
                    sample_rate_hz=loaded.audio.sample_rate_hz,
                )
                embedding = model(
                    features.unsqueeze(0).to(device=device, dtype=torch.float32)
                ).squeeze(0)
                chunk_embeddings.append(embedding.detach().to(device="cpu"))
            pooled = pool_chunk_tensors(chunk_embeddings, pooling_mode=eval_chunks.pooling_mode)
            normalized = torch_functional.normalize(pooled, dim=0)
            trial_item_id = row.utterance_id or row.audio_path
            point_id = f"utt-{index:05d}"

            embeddings.append(normalized)
            point_ids.append(point_id)
            metadata_rows.append(
                {
                    **_lookup_manifest_metadata_row(
                        row=row,
                        trial_item_id=trial_item_id,
                        manifest_metadata_lookup=manifest_metadata_lookup,
                    ),
                    "atlas_point_id": point_id,
                    "trial_item_id": trial_item_id,
                    "speaker_id": row.speaker_id,
                    "utterance_id": row.utterance_id,
                    "audio_path": row.audio_path,
                    "split": row.split,
                    "role": row.role,
                    "channel": row.channel,
                    "dataset": row.dataset,
                    "source_dataset": row.source_dataset,
                    "duration_seconds": loaded.audio.duration_seconds,
                    "embedding_device": str(device),
                    "embedding_source": embedding_source,
                }
            )

    import numpy as np

    embeddings_matrix = torch.stack(embeddings, dim=0).to(dtype=torch.float32).numpy()
    npz_path = output_root / EMBEDDINGS_FILE_NAME
    jsonl_path = output_root / EMBEDDING_METADATA_JSONL_NAME
    parquet_path = output_root / EMBEDDING_METADATA_PARQUET_NAME
    np.savez(npz_path, embeddings=embeddings_matrix, point_ids=np.asarray(point_ids, dtype=str))
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )
    pl.DataFrame(metadata_rows).write_parquet(parquet_path)

    summary = EmbeddingExportSummary(
        manifest_path=manifest_path,
        embedding_dim=int(embeddings_matrix.shape[1]),
        utterance_count=int(embeddings_matrix.shape[0]),
        speaker_count=len({row["speaker_id"] for row in metadata_rows}),
        embeddings_path=str(npz_path),
        metadata_jsonl_path=str(jsonl_path),
        metadata_parquet_path=str(parquet_path),
    )
    return summary, metadata_rows


def _load_manifest_metadata_lookup(
    *,
    manifest_path: str,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    manifest_file = resolve_project_path(str(project_root), manifest_path)
    lookup: dict[str, dict[str, Any]] = {}
    for raw_line in manifest_file.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object JSONL rows in {manifest_file}")
        for key in _manifest_lookup_keys(payload):
            lookup.setdefault(key, payload)
    return lookup


def _lookup_manifest_metadata_row(
    *,
    row: ManifestRow,
    trial_item_id: str,
    manifest_metadata_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for key in (trial_item_id, row.utterance_id, row.audio_path, Path(row.audio_path).name):
        if not key:
            continue
        payload = manifest_metadata_lookup.get(key)
        if payload is not None:
            return dict(payload)
    return {}


def _manifest_lookup_keys(payload: Mapping[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    for field_name in ("trial_item_id", "utterance_id", "audio_path"):
        value = payload.get(field_name)
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized:
            continue
        keys.append(normalized)
        if field_name == "audio_path":
            keys.append(Path(normalized).name)
    return tuple(dict.fromkeys(keys))


def load_or_generate_trials(
    *,
    output_root: Path,
    configured_trials_manifest: str | None,
    metadata_rows: list[dict[str, Any]],
    project_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    if configured_trials_manifest:
        trials_path = resolve_project_path(str(project_root), configured_trials_manifest)
        trial_rows = [
            json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()
        ]
        if not all(isinstance(row, dict) for row in trial_rows):
            raise ValueError("Trials manifest must contain object JSONL rows.")
        return trials_path, list(trial_rows)

    enrollment_rows = [row for row in metadata_rows if row.get("role") == "enrollment"]
    test_rows = [row for row in metadata_rows if row.get("role") == "test"]
    if enrollment_rows and test_rows:
        pairs = ((left, right) for left in enrollment_rows for right in test_rows)
    else:
        pairs = combinations(metadata_rows, 2)

    trial_rows: list[dict[str, Any]] = []
    for left, right in pairs:
        if left["trial_item_id"] == right["trial_item_id"]:
            continue
        trial_rows.append(
            {
                "left_id": left["trial_item_id"],
                "right_id": right["trial_item_id"],
                "left_speaker_id": left["speaker_id"],
                "right_speaker_id": right["speaker_id"],
                "label": int(left["speaker_id"] == right["speaker_id"]),
            }
        )

    trials_path = output_root / TRIALS_FILE_NAME
    trials_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in trial_rows),
        encoding="utf-8",
    )
    return trials_path, trial_rows


def score_trials(
    *,
    output_root: Path,
    trials_path: Path,
    metadata_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    embeddings_path: Path | None = None,
) -> ScoreSummary:
    import numpy as np

    resolved_embeddings_path = embeddings_path or (output_root / EMBEDDINGS_FILE_NAME)
    embeddings_payload = np.load(resolved_embeddings_path)
    embeddings = embeddings_payload["embeddings"]
    point_id_to_embedding = {
        row["trial_item_id"]: embeddings[index] for index, row in enumerate(metadata_rows)
    }
    for index, row in enumerate(metadata_rows):
        audio_path = row.get("audio_path")
        if not audio_path:
            continue
        normalized_audio_path = str(audio_path)
        point_id_to_embedding[normalized_audio_path] = embeddings[index]
        point_id_to_embedding.setdefault(Path(normalized_audio_path).name, embeddings[index])
    point_id_to_embedding.update(
        {
            row["utterance_id"]: embeddings[index]
            for index, row in enumerate(metadata_rows)
            if row.get("utterance_id")
        }
    )

    scored_rows: list[dict[str, Any]] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    scored_trial_rows: list[dict[str, Any]] = []
    left_embeddings: list[np.ndarray] = []
    right_embeddings: list[np.ndarray] = []
    missing_embedding_count = 0
    for row in trial_rows:
        left_id = str(row.get("left_id", row.get("left_audio", "")))
        right_id = str(row.get("right_id", row.get("right_audio", "")))
        left_embedding = point_id_to_embedding.get(left_id)
        right_embedding = point_id_to_embedding.get(right_id)
        if left_embedding is None or right_embedding is None:
            missing_embedding_count += 1
            continue
        scored_trial_rows.append(
            {
                **row,
                "left_id": left_id,
                "right_id": right_id,
            }
        )
        left_embeddings.append(np.asarray(left_embedding, dtype=np.float64))
        right_embeddings.append(np.asarray(right_embedding, dtype=np.float64))

    pairwise_scores: np.ndarray
    if left_embeddings:
        pairwise_scores = cosine_score_pairs(
            np.stack(left_embeddings, axis=0),
            np.stack(right_embeddings, axis=0),
            normalize=True,
        )
    else:
        pairwise_scores = np.empty((0,), dtype=np.float64)

    for row, score in zip(scored_trial_rows, pairwise_scores, strict=True):
        label = int(row["label"])
        score_value = float(score)
        if label == 1:
            positive_scores.append(score_value)
        else:
            negative_scores.append(score_value)
        scored_rows.append(
            {
                **row,
                "score": round(score_value, 8),
            }
        )

    scores_path = output_root / SCORES_FILE_NAME
    scores_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored_rows),
        encoding="utf-8",
    )

    mean_positive = mean_or_none(positive_scores)
    mean_negative = mean_or_none(negative_scores)
    score_gap = None
    if mean_positive is not None and mean_negative is not None:
        score_gap = round(mean_positive - mean_negative, 6)
    return ScoreSummary(
        trials_path=str(trials_path),
        scores_path=str(scores_path),
        trial_count=len(scored_rows),
        positive_count=len(positive_scores),
        negative_count=len(negative_scores),
        missing_embedding_count=missing_embedding_count,
        mean_positive_score=mean_positive,
        mean_negative_score=mean_negative,
        score_gap=score_gap,
    )


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


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


__all__ = [
    "EMBEDDINGS_FILE_NAME",
    "EMBEDDING_METADATA_JSONL_NAME",
    "EMBEDDING_METADATA_PARQUET_NAME",
    "REPRODUCIBILITY_FILE_NAME",
    "SCORE_SUMMARY_FILE_NAME",
    "SCORES_FILE_NAME",
    "TRAINING_SUMMARY_FILE_NAME",
    "TRIALS_FILE_NAME",
    "EmbeddingExportSummary",
    "EpochAwareBatchSampler",
    "EpochAwareDataset",
    "EpochSummary",
    "ScoreSummary",
    "SpeakerBaselineRunArtifacts",
    "TrainingSummary",
    "build_default_cohort_bank",
    "build_fixed_train_chunking_request",
    "export_dev_embeddings",
    "load_or_generate_trials",
    "mean_or_none",
    "prepare_demo_artifacts_if_needed",
    "relative_to_project",
    "render_markdown_report",
    "resolve_device",
    "score_trials",
    "train_epochs",
    "write_checkpoint",
]
