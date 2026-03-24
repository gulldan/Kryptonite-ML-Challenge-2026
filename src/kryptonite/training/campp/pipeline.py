"""End-to-end CAM++ baseline training, embedding export, and cosine scoring."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torch.nn.functional as torch_functional
from torch import nn
from torch.utils.data import DataLoader

from kryptonite.data import AudioLoadRequest, load_manifest_audio
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.deployment import resolve_project_path
from kryptonite.features import (
    FbankExtractionRequest,
    FbankExtractor,
    UtteranceChunkingRequest,
    chunk_utterance,
    pool_chunk_tensors,
)
from kryptonite.models import ArcMarginLoss, CAMPPlusEncoder, CosineClassifier
from kryptonite.repro import build_reproducibility_snapshot, set_global_seed
from kryptonite.tracking import build_tracker, create_run_id

from .config import CAMPPlusBaselineConfig
from .data import (
    ManifestSpeakerDataset,
    TrainingBatch,
    build_speaker_index,
    collate_training_examples,
    load_manifest_rows,
)

EMBEDDINGS_FILE_NAME = "dev_embeddings.npz"
EMBEDDING_METADATA_JSONL_NAME = "dev_embedding_metadata.jsonl"
EMBEDDING_METADATA_PARQUET_NAME = "dev_embedding_metadata.parquet"
TRAINING_SUMMARY_FILE_NAME = "training_summary.json"
SCORES_FILE_NAME = "dev_scores.jsonl"
TRIALS_FILE_NAME = "dev_trials.jsonl"
SCORE_SUMMARY_FILE_NAME = "score_summary.json"
REPRODUCIBILITY_FILE_NAME = "reproducibility_snapshot.json"
REPORT_FILE_NAME = "campp_baseline_report.md"


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
class CAMPPlusRunArtifacts:
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
    tracking_run_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
            "tracking_run_dir": self.tracking_run_dir,
        }
        return payload


def run_campp_baseline(
    config: CAMPPlusBaselineConfig,
    *,
    config_path: Path | str,
    device_override: str | None = None,
) -> CAMPPlusRunArtifacts:
    _prepare_demo_artifacts_if_needed(config)
    _validate_training_precision(config.project.training.precision)
    if config.project.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive for CAM++ baseline runs.")
    seed_state = set_global_seed(
        config.project.runtime.seed,
        deterministic=config.project.reproducibility.deterministic,
        pythonhashseed=config.project.reproducibility.pythonhashseed,
    )
    del seed_state

    device = _resolve_device(device_override or config.project.runtime.device)
    project_root = resolve_project_path(config.project.paths.project_root, ".")

    train_rows = load_manifest_rows(
        config.data.train_manifest,
        project_root=project_root,
        limit=config.data.max_train_rows,
    )
    dev_rows = load_manifest_rows(
        config.data.dev_manifest,
        project_root=project_root,
        limit=config.data.max_dev_rows,
    )
    speaker_to_index = build_speaker_index(train_rows)

    audio_request = AudioLoadRequest.from_config(
        config.project.normalization,
        vad=config.project.vad,
    )
    feature_request = FbankExtractionRequest.from_config(config.project.features)
    chunking_request = _build_fixed_train_chunking_request(config)

    train_dataset = ManifestSpeakerDataset(
        rows=train_rows,
        speaker_to_index=speaker_to_index,
        project_root=project_root,
        audio_request=audio_request,
        feature_request=feature_request,
        chunking_request=chunking_request,
        seed=config.project.runtime.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.project.training.batch_size,
        shuffle=True,
        num_workers=config.project.runtime.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_training_examples,
        drop_last=False,
    )

    tracker_run = None
    if config.project.tracking.enabled:
        tracker = build_tracker(config=config.project)
        tracker_run = tracker.start_run(kind="campp-baseline", config=config.to_dict())
        run_id = tracker_run.run_id
    else:
        run_id = create_run_id()

    output_root = resolve_project_path(str(project_root), config.data.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    model = CAMPPlusEncoder(config.model).to(device)
    classifier = CosineClassifier(
        config.model.embedding_size,
        num_classes=len(speaker_to_index),
        num_blocks=config.objective.classifier_blocks,
        hidden_dim=config.objective.classifier_hidden_dim,
    ).to(device)
    criterion = ArcMarginLoss(
        scale=config.objective.scale,
        margin=config.objective.margin,
        easy_margin=config.objective.easy_margin,
    )
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(classifier.parameters()),
        lr=config.optimization.learning_rate,
        momentum=config.optimization.momentum,
        nesterov=True,
        weight_decay=config.optimization.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_build_lr_lambda(config),
    )

    epoch_summaries = _train_epochs(
        model=model,
        classifier=classifier,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=train_loader,
        dataset=train_dataset,
        device=device,
        max_epochs=config.project.training.max_epochs,
        grad_clip_norm=config.optimization.grad_clip_norm,
        tracker_run=tracker_run,
    )
    checkpoint_path = output_root / config.data.checkpoint_name
    _write_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        classifier=classifier,
        config=config,
        speaker_to_index=speaker_to_index,
    )

    training_summary = TrainingSummary(
        device=str(device),
        train_manifest=config.data.train_manifest,
        dev_manifest=config.data.dev_manifest,
        speaker_count=len(speaker_to_index),
        train_row_count=len(train_rows),
        dev_row_count=len(dev_rows),
        checkpoint_path=str(checkpoint_path),
        epochs=tuple(epoch_summaries),
    )
    training_summary_path = output_root / TRAINING_SUMMARY_FILE_NAME
    training_summary_path.write_text(
        json.dumps(training_summary.to_dict(), indent=2, sort_keys=True)
    )

    embedding_summary, metadata_rows = _export_dev_embeddings(
        output_root=output_root,
        model=model,
        rows=dev_rows,
        manifest_path=config.data.dev_manifest,
        project_root=project_root,
        audio_request=audio_request,
        feature_request=feature_request,
        chunking_request=config.project.chunking,
        device=device,
    )
    trials_path, trial_rows = _load_or_generate_trials(
        output_root=output_root,
        configured_trials_manifest=config.data.trials_manifest,
        metadata_rows=metadata_rows,
        project_root=project_root,
    )
    score_summary = _score_trials(
        output_root=output_root,
        trials_path=trials_path,
        metadata_rows=metadata_rows,
        trial_rows=trial_rows,
    )
    score_summary_path = output_root / SCORE_SUMMARY_FILE_NAME
    score_summary_path.write_text(json.dumps(score_summary.to_dict(), indent=2, sort_keys=True))

    reproducibility = build_reproducibility_snapshot(
        config=config.project,
        config_path=resolve_project_path(str(project_root), config.base_config_path),
    )
    reproducibility_path = output_root / REPRODUCIBILITY_FILE_NAME
    reproducibility_path.write_text(json.dumps(reproducibility, indent=2, sort_keys=True))

    report_path = output_root / REPORT_FILE_NAME
    report_path.write_text(
        _render_markdown_report(
            config=config,
            training_summary=training_summary,
            embedding_summary=embedding_summary,
            score_summary=score_summary,
            output_root=output_root,
        )
    )

    if tracker_run is not None:
        final_epoch = training_summary.epochs[-1]
        tracker_run.log_metrics(
            {
                "train_loss": final_epoch.mean_loss,
                "train_accuracy": final_epoch.accuracy,
                "score_gap": score_summary.score_gap or 0.0,
            },
            step=config.project.training.max_epochs,
        )
        for artifact_path in (
            checkpoint_path,
            training_summary_path,
            Path(embedding_summary.embeddings_path),
            Path(embedding_summary.metadata_jsonl_path),
            Path(embedding_summary.metadata_parquet_path),
            Path(trials_path),
            Path(score_summary.scores_path),
            score_summary_path,
            reproducibility_path,
            report_path,
        ):
            tracker_run.log_artifact(artifact_path)
        tracker_run.finish(
            summary={
                "checkpoint_path": str(checkpoint_path),
                "score_gap": score_summary.score_gap,
                "trial_count": score_summary.trial_count,
            }
        )

    return CAMPPlusRunArtifacts(
        output_root=str(output_root),
        checkpoint_path=str(checkpoint_path),
        training_summary_path=str(training_summary_path),
        embeddings_path=embedding_summary.embeddings_path,
        embedding_metadata_jsonl_path=embedding_summary.metadata_jsonl_path,
        embedding_metadata_parquet_path=embedding_summary.metadata_parquet_path,
        trials_path=str(trials_path),
        scores_path=score_summary.scores_path,
        score_summary_path=str(score_summary_path),
        reproducibility_path=str(reproducibility_path),
        report_path=str(report_path),
        training_summary=training_summary,
        embedding_summary=embedding_summary,
        score_summary=score_summary,
        tracking_run_dir=(None if tracker_run is None else str(tracker_run.run_dir)),
    )


def _prepare_demo_artifacts_if_needed(config: CAMPPlusBaselineConfig) -> None:
    if not config.data.generate_demo_artifacts_if_missing:
        return
    project_root = resolve_project_path(config.project.paths.project_root, ".")
    train_manifest = resolve_project_path(str(project_root), config.data.train_manifest)
    dev_manifest = resolve_project_path(str(project_root), config.data.dev_manifest)
    if train_manifest.exists() and dev_manifest.exists():
        return
    generate_demo_artifacts(config=config.project)


def _validate_training_precision(precision: str) -> None:
    normalized = precision.lower()
    if normalized not in {"fp32", "float32"}:
        raise ValueError(
            "CAM++ baseline currently supports fp32 only; mixed precision will be added separately."
        )


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(normalized)


def _build_fixed_train_chunking_request(config: CAMPPlusBaselineConfig) -> Any:
    chunking = config.project.chunking
    if chunking.train_num_crops != 1:
        raise ValueError("CAM++ baseline requires chunking.train_num_crops=1.")
    if not math.isclose(
        chunking.train_min_crop_seconds,
        chunking.train_max_crop_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "CAM++ baseline requires fixed-size train crops; set "
            "chunking.train_min_crop_seconds == chunking.train_max_crop_seconds."
        )
    return UtteranceChunkingRequest.from_config(config.project.chunking)


def _build_lr_lambda(config: CAMPPlusBaselineConfig):
    max_epochs = config.project.training.max_epochs
    warmup_epochs = config.optimization.warmup_epochs
    max_lr = config.optimization.learning_rate
    min_lr = config.optimization.min_learning_rate
    min_ratio = min_lr / max_lr

    def _lambda(epoch_index: int) -> float:
        if warmup_epochs == 0 and epoch_index == 0:
            return 1.0
        current_epoch = epoch_index + 1
        if warmup_epochs > 0 and current_epoch <= warmup_epochs:
            return max(min_ratio, current_epoch / warmup_epochs)
        cosine_steps = max(1, max_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (current_epoch - warmup_epochs) / cosine_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + ((1.0 - min_ratio) * cosine)

    return _lambda


def _train_epochs(
    *,
    model: CAMPPlusEncoder,
    classifier: CosineClassifier,
    criterion: ArcMarginLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader: Iterable[TrainingBatch],
    dataset: ManifestSpeakerDataset,
    device: torch.device,
    max_epochs: int,
    grad_clip_norm: float | None,
    tracker_run: Any | None,
) -> list[EpochSummary]:
    summaries: list[EpochSummary] = []
    for epoch in range(max_epochs):
        dataset.set_epoch(epoch)
        model.train()
        classifier.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch in loader:
            features = batch.features.to(device=device, dtype=torch.float32)
            labels = batch.labels.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(features)
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()),
                    max_norm=grad_clip_norm,
                )
            optimizer.step()

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += batch_size

        if total_examples == 0:
            raise ValueError("Training loader produced zero examples.")

        learning_rate = round(float(optimizer.param_groups[0]["lr"]), 8)
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
        scheduler.step()
    return summaries


def _write_checkpoint(
    *,
    checkpoint_path: Path,
    model: CAMPPlusEncoder,
    classifier: CosineClassifier,
    config: CAMPPlusBaselineConfig,
    speaker_to_index: dict[str, int],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "model_config": asdict(config.model),
        "baseline_config": config.to_dict(),
        "speaker_to_index": speaker_to_index,
    }
    torch.save(payload, checkpoint_path)


def _export_dev_embeddings(
    *,
    output_root: Path,
    model: CAMPPlusEncoder,
    rows: list[Any],
    manifest_path: str,
    project_root: Path,
    audio_request: AudioLoadRequest,
    feature_request: FbankExtractionRequest,
    chunking_request: Any,
    device: torch.device,
) -> tuple[EmbeddingExportSummary, list[dict[str, Any]]]:
    model.eval()
    extractor = FbankExtractor(request=feature_request)
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
                request=UtteranceChunkingRequest.from_config(chunking_request),
            )
            chunk_embeddings: list[torch.Tensor] = []
            for chunk in eval_chunks.chunks:
                features = extractor.extract(
                    chunk.waveform, sample_rate_hz=loaded.audio.sample_rate_hz
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
                    "embedding_source": "campp_baseline",
                }
            )

    embeddings_matrix = torch.stack(embeddings, dim=0).to(dtype=torch.float32).numpy()
    npz_path = output_root / EMBEDDINGS_FILE_NAME
    jsonl_path = output_root / EMBEDDING_METADATA_JSONL_NAME
    parquet_path = output_root / EMBEDDING_METADATA_PARQUET_NAME
    import numpy as np

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


def _load_or_generate_trials(
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


def _score_trials(
    *,
    output_root: Path,
    trials_path: Path,
    metadata_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
) -> ScoreSummary:
    import numpy as np

    embeddings_payload = np.load(output_root / EMBEDDINGS_FILE_NAME)
    embeddings = embeddings_payload["embeddings"]
    point_id_to_embedding = {
        row["trial_item_id"]: embeddings[index] for index, row in enumerate(metadata_rows)
    }
    point_id_to_embedding.update(
        {
            row["audio_path"]: embeddings[index]
            for index, row in enumerate(metadata_rows)
            if row.get("audio_path")
        }
    )
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
    missing_embedding_count = 0
    for row in trial_rows:
        left_id = str(row.get("left_id", row.get("left_audio", "")))
        right_id = str(row.get("right_id", row.get("right_audio", "")))
        left_embedding = point_id_to_embedding.get(left_id)
        right_embedding = point_id_to_embedding.get(right_id)
        if left_embedding is None or right_embedding is None:
            missing_embedding_count += 1
            continue
        score = float(np.dot(left_embedding, right_embedding))
        label = int(row["label"])
        if label == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)
        scored_rows.append(
            {
                **row,
                "left_id": left_id,
                "right_id": right_id,
                "score": round(score, 8),
            }
        )

    scores_path = output_root / SCORES_FILE_NAME
    scores_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    mean_positive = _mean_or_none(positive_scores)
    mean_negative = _mean_or_none(negative_scores)
    gap = None
    if mean_positive is not None and mean_negative is not None:
        gap = round(mean_positive - mean_negative, 6)
    return ScoreSummary(
        trials_path=str(trials_path),
        scores_path=str(scores_path),
        trial_count=len(scored_rows),
        positive_count=len(positive_scores),
        negative_count=len(negative_scores),
        missing_embedding_count=missing_embedding_count,
        mean_positive_score=mean_positive,
        mean_negative_score=mean_negative,
        score_gap=gap,
    )


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _render_markdown_report(
    *,
    config: CAMPPlusBaselineConfig,
    training_summary: TrainingSummary,
    embedding_summary: EmbeddingExportSummary,
    score_summary: ScoreSummary,
    output_root: Path,
) -> str:
    final_epoch = training_summary.epochs[-1]
    project_root = resolve_project_path(config.project.paths.project_root, ".")
    relative_output_root = _relative_to_project(output_root, project_root=project_root)
    relative_checkpoint = _relative_to_project(
        Path(training_summary.checkpoint_path),
        project_root=project_root,
    )
    relative_embeddings = _relative_to_project(
        Path(embedding_summary.embeddings_path),
        project_root=project_root,
    )
    lines = [
        "# CAM++ Baseline Report",
        "",
        f"- Output root: `{relative_output_root}`",
        f"- Device: `{training_summary.device}`",
        f"- Train manifest: `{training_summary.train_manifest}`",
        f"- Dev manifest: `{training_summary.dev_manifest}`",
        f"- Speakers: `{training_summary.speaker_count}`",
        f"- Train rows: `{training_summary.train_row_count}`",
        f"- Dev rows: `{training_summary.dev_row_count}`",
        "",
        "## Training",
        "",
        f"- Epochs: `{len(training_summary.epochs)}`",
        f"- Final loss: `{final_epoch.mean_loss}`",
        f"- Final accuracy: `{final_epoch.accuracy}`",
        f"- Final learning rate: `{final_epoch.learning_rate}`",
        f"- Checkpoint: `{relative_checkpoint}`",
        "",
        "## Embeddings",
        "",
        f"- Utterances exported: `{embedding_summary.utterance_count}`",
        f"- Embedding dim: `{embedding_summary.embedding_dim}`",
        f"- Speaker count: `{embedding_summary.speaker_count}`",
        f"- Embeddings: `{relative_embeddings}`",
        "",
        "## Scores",
        "",
        f"- Scored trials: `{score_summary.trial_count}`",
        f"- Positive trials: `{score_summary.positive_count}`",
        f"- Negative trials: `{score_summary.negative_count}`",
        f"- Missing trial embeddings: `{score_summary.missing_embedding_count}`",
        f"- Mean positive score: `{score_summary.mean_positive_score}`",
        f"- Mean negative score: `{score_summary.mean_negative_score}`",
        f"- Score gap: `{score_summary.score_gap}`",
    ]
    return "\n".join(lines) + "\n"


def _relative_to_project(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())
