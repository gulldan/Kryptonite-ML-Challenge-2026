"""Fine-tune CAM++ with hard real labels and soft confidence-weighted pseudo labels."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kryptonite.data import AudioLoadRequest, ManifestRow
from kryptonite.deployment import resolve_project_path
from kryptonite.features import FbankExtractionRequest, UtteranceChunkingRequest
from kryptonite.models import ArcMarginLoss, CosineClassifier
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint
from kryptonite.repro import build_reproducibility_snapshot, set_global_seed
from kryptonite.tracking import build_tracker, create_run_id
from kryptonite.training.augmentation_runtime import TrainingAugmentationRuntime
from kryptonite.training.campp import load_campp_baseline_config
from kryptonite.training.manifest_speaker_data import (
    ManifestSpeakerDataset,
    TrainingExample,
    TrainingSampleRequest,
    build_speaker_index,
)
from kryptonite.training.optimization_runtime import build_training_runtime
from kryptonite.training.production_dataloader import BalancedSpeakerBatchSampler
from kryptonite.training.speaker_baseline import write_checkpoint


@dataclass(frozen=True, slots=True)
class SoftTrainingExample:
    base: TrainingExample
    is_pseudo: bool
    sample_weight: float
    soft_indices: tuple[int, ...] = field(default_factory=tuple)
    soft_probs: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SoftTrainingBatch:
    features: torch.Tensor
    hard_labels: torch.Tensor
    is_pseudo: torch.Tensor
    sample_weights: torch.Tensor
    soft_indices: torch.Tensor
    soft_probs: torch.Tensor


class SoftManifestSpeakerDataset(Dataset[SoftTrainingExample]):
    def __init__(
        self,
        *,
        rows: list[ManifestRow],
        speaker_to_index: dict[str, int],
        project_root: Path,
        audio_request: AudioLoadRequest,
        feature_request: FbankExtractionRequest,
        chunking_request: UtteranceChunkingRequest,
        seed: int,
        augmentation_runtime: TrainingAugmentationRuntime | None,
        soft_indices: list[tuple[int, ...]],
        soft_probs: list[tuple[float, ...]],
        sample_weights: list[float],
        is_pseudo: list[bool],
    ) -> None:
        self._base = ManifestSpeakerDataset(
            rows=rows,
            speaker_to_index=speaker_to_index,
            project_root=project_root,
            audio_request=audio_request,
            feature_request=feature_request,
            chunking_request=chunking_request,
            seed=seed,
            augmentation_runtime=augmentation_runtime,
        )
        self._soft_indices = soft_indices
        self._soft_probs = soft_probs
        self._sample_weights = sample_weights
        self._is_pseudo = is_pseudo

    def __len__(self) -> int:
        return len(self._base)

    def set_epoch(self, epoch: int) -> None:
        self._base.set_epoch(epoch)

    def __getitem__(self, index: int | TrainingSampleRequest) -> SoftTrainingExample:
        row_index = index.row_index if isinstance(index, TrainingSampleRequest) else int(index)
        base = self._base[index]
        return SoftTrainingExample(
            base=base,
            is_pseudo=self._is_pseudo[row_index],
            sample_weight=self._sample_weights[row_index],
            soft_indices=self._soft_indices[row_index],
            soft_probs=self._soft_probs[row_index],
        )


def main() -> None:
    args = _parse_args()
    baseline = load_campp_baseline_config(
        config_path=args.config,
        env_file=args.env_file,
        project_overrides=args.project_override or [],
    )
    device = _resolve_device(args.device or baseline.project.runtime.device)
    set_global_seed(
        baseline.project.runtime.seed,
        deterministic=baseline.project.reproducibility.deterministic,
        pythonhashseed=baseline.project.reproducibility.pythonhashseed,
    )
    project_root = resolve_project_path(baseline.project.paths.project_root, ".")
    raw_rows, rows = _load_raw_manifest_rows(
        baseline.data.train_manifest,
        project_root=project_root,
        limit=baseline.data.max_train_rows,
    )
    speaker_to_index = build_speaker_index(rows)
    target_width = args.soft_target_width
    soft_indices, soft_probs, sample_weights, is_pseudo = _build_soft_metadata(
        raw_rows=raw_rows,
        speaker_to_index=speaker_to_index,
        target_width=target_width,
    )
    checkpoint_path, checkpoint_model_config, encoder = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=args.init_checkpoint,
    )
    encoder = encoder.to(device)
    classifier = CosineClassifier(
        checkpoint_model_config.embedding_size,
        num_classes=len(speaker_to_index),
        num_blocks=baseline.objective.classifier_blocks,
        hidden_dim=baseline.objective.classifier_hidden_dim,
    ).to(device)
    hard_criterion = ArcMarginLoss(
        scale=baseline.objective.scale,
        margin=baseline.objective.margin,
        easy_margin=baseline.objective.easy_margin,
    )
    runtime = build_training_runtime(
        parameters=[*encoder.parameters(), *classifier.parameters()],
        optimization_config=baseline.optimization,
        precision=baseline.project.training.precision,
        device=device,
        max_epochs=baseline.project.training.max_epochs,
    )
    dataset, sampler, loader = _build_loader(
        rows=rows,
        speaker_to_index=speaker_to_index,
        project_root=project_root,
        baseline=baseline,
        total_epochs=baseline.project.training.max_epochs,
        device=device,
        soft_indices=soft_indices,
        soft_probs=soft_probs,
        sample_weights=sample_weights,
        is_pseudo=is_pseudo,
    )

    tracker_run = None
    if baseline.project.tracking.enabled:
        tracker_run = build_tracker(config=baseline.project).start_run(
            kind="campp-soft-pseudo-finetune",
            config={
                **baseline.to_dict(),
                "soft_pseudo": {
                    "init_checkpoint": str(checkpoint_path),
                    "soft_loss_weight": args.soft_loss_weight,
                    "hard_loss_weight": args.hard_loss_weight,
                    "pseudo_hard_loss_weight": args.pseudo_hard_loss_weight,
                    "soft_target_width": target_width,
                },
            },
        )
        run_id = tracker_run.run_id
    else:
        run_id = create_run_id()
    output_root = resolve_project_path(str(project_root), baseline.data.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    epoch_summaries = _train_epochs(
        model=encoder,
        classifier=classifier,
        hard_criterion=hard_criterion,
        training_runtime=runtime,
        loader=loader,
        dataset=dataset,
        sampler=sampler,
        device=device,
        max_epochs=baseline.project.training.max_epochs,
        scale=baseline.objective.scale,
        hard_loss_weight=args.hard_loss_weight,
        soft_loss_weight=args.soft_loss_weight,
        pseudo_hard_loss_weight=args.pseudo_hard_loss_weight,
        tracker_run=tracker_run,
    )
    checkpoint_out = output_root / baseline.data.checkpoint_name
    write_checkpoint(
        checkpoint_path=checkpoint_out,
        model=encoder,
        classifier=classifier,
        model_config=asdict(checkpoint_model_config),
        baseline_config={
            **baseline.to_dict(),
            "soft_pseudo": {
                "init_checkpoint": str(checkpoint_path),
                "soft_loss_weight": args.soft_loss_weight,
                "hard_loss_weight": args.hard_loss_weight,
                "pseudo_hard_loss_weight": args.pseudo_hard_loss_weight,
                "soft_target_width": target_width,
            },
        },
        speaker_to_index=speaker_to_index,
    )
    summary = {
        "run_id": run_id,
        "device": str(device),
        "config_path": str(args.config),
        "init_checkpoint": str(checkpoint_path),
        "train_manifest": baseline.data.train_manifest,
        "train_row_count": len(rows),
        "speaker_count": len(speaker_to_index),
        "pseudo_row_count": int(sum(is_pseudo)),
        "checkpoint_path": str(checkpoint_out),
        "epochs": epoch_summaries,
    }
    summary_path = output_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    repro_path = output_root / "reproducibility_snapshot.json"
    repro_path.write_text(
        json.dumps(
            build_reproducibility_snapshot(config=baseline.project, config_path=args.config),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if tracker_run is not None:
        tracker_run.log_artifact(summary_path)
        tracker_run.log_artifact(checkpoint_out)
        tracker_run.finish(summary=summary)
    payload = {
        "output_root": str(output_root),
        "checkpoint_path": str(checkpoint_out),
        "training_summary_path": str(summary_path),
        **summary,
    }
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output == "json":
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    else:
        final = epoch_summaries[-1]
        print(
            "\n".join(
                [
                    "CAM++ soft pseudo fine-tune complete",
                    f"Output root: {output_root}",
                    f"Checkpoint: {checkpoint_out}",
                    f"Final loss: {final['mean_loss']}",
                    f"Final accuracy: {final['accuracy']}",
                    f"Final soft loss: {final['soft_loss']}",
                ]
            ),
            flush=True,
        )


def _train_epochs(
    *,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    hard_criterion: torch.nn.Module,
    training_runtime: Any,
    loader: DataLoader[SoftTrainingBatch],
    dataset: SoftManifestSpeakerDataset,
    sampler: BalancedSpeakerBatchSampler,
    device: torch.device,
    max_epochs: int,
    scale: float,
    hard_loss_weight: float,
    soft_loss_weight: float,
    pseudo_hard_loss_weight: float,
    tracker_run: Any | None,
) -> list[dict[str, float | int]]:
    summaries: list[dict[str, float | int]] = []
    for epoch in range(max_epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        summary = _run_epoch(
            model=model,
            classifier=classifier,
            hard_criterion=hard_criterion,
            training_runtime=training_runtime,
            loader=loader,
            device=device,
            epoch=epoch + 1,
            max_epochs=max_epochs,
            scale=scale,
            hard_loss_weight=hard_loss_weight,
            soft_loss_weight=soft_loss_weight,
            pseudo_hard_loss_weight=pseudo_hard_loss_weight,
        )
        summaries.append(summary)
        if tracker_run is not None:
            tracker_run.log_metrics(
                {
                    "train_loss": float(summary["mean_loss"]),
                    "train_accuracy": float(summary["accuracy"]),
                    "learning_rate": float(summary["learning_rate"]),
                    "hard_loss": float(summary["hard_loss"]),
                    "soft_loss": float(summary["soft_loss"]),
                },
                step=int(summary["epoch"]),
            )
        training_runtime.step_scheduler(mean_loss=float(summary["mean_loss"]))
    return summaries


def _run_epoch(
    *,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    hard_criterion: torch.nn.Module,
    training_runtime: Any,
    loader: DataLoader[SoftTrainingBatch],
    device: torch.device,
    epoch: int,
    max_epochs: int,
    scale: float,
    hard_loss_weight: float,
    soft_loss_weight: float,
    pseudo_hard_loss_weight: float,
) -> dict[str, float | int]:
    model.train()
    classifier.train()
    total_loss = 0.0
    total_hard_loss = 0.0
    total_soft_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_batches = len(loader)
    accumulation_steps = training_runtime.gradient_accumulation_steps
    pending_step = False
    started_at = time.monotonic()
    log_every = max(1, total_batches // 20)
    print(
        f"[soft-train] epoch={epoch}/{max_epochs} start batches={total_batches} "
        f"accumulation_steps={accumulation_steps}",
        flush=True,
    )
    training_runtime.zero_grad()
    for batch_index, batch in enumerate(loader, start=1):
        features = batch.features.to(device=device, dtype=torch.float32)
        hard_labels = batch.hard_labels.to(device=device)
        is_pseudo = batch.is_pseudo.to(device=device)
        sample_weights = batch.sample_weights.to(device=device)
        soft_indices = batch.soft_indices.to(device=device)
        soft_probs = batch.soft_probs.to(device=device)
        with training_runtime.precision.autocast_context(device=device):
            embeddings = model(features)
            logits = classifier(embeddings)
            hard_mask = ~is_pseudo
            pseudo_mask = is_pseudo
            loss = logits.new_tensor(0.0)
            hard_loss = logits.new_tensor(0.0)
            soft_loss = logits.new_tensor(0.0)
            if hard_mask.any():
                hard_loss = hard_criterion(logits[hard_mask], hard_labels[hard_mask])
                loss = loss + (hard_loss_weight * hard_loss)
            if pseudo_mask.any():
                pseudo_logits = logits[pseudo_mask] * scale
                pseudo_log_probs = F.log_softmax(pseudo_logits, dim=1)
                gathered = pseudo_log_probs.gather(1, soft_indices[pseudo_mask].clamp_min(0))
                per_sample = -(soft_probs[pseudo_mask] * gathered).sum(dim=1)
                weights = sample_weights[pseudo_mask]
                soft_loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
                loss = loss + (soft_loss_weight * soft_loss)
                if pseudo_hard_loss_weight > 0.0:
                    pseudo_hard_loss = hard_criterion(
                        logits[pseudo_mask],
                        hard_labels[pseudo_mask],
                    )
                    loss = loss + (pseudo_hard_loss_weight * pseudo_hard_loss)
            scaled_loss = loss / accumulation_steps
        training_runtime.backward(scaled_loss)
        pending_step = True
        if batch_index % accumulation_steps == 0:
            training_runtime.step_optimizer()
            training_runtime.zero_grad()
            pending_step = False

        batch_size = int(hard_labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_hard_loss += float(hard_loss.detach().item()) * batch_size
        total_soft_loss += float(soft_loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == hard_labels).sum().item())
        total_examples += batch_size
        if batch_index == 1 or batch_index % log_every == 0 or batch_index == total_batches:
            elapsed = max(time.monotonic() - started_at, 1e-9)
            print(
                f"[soft-train] epoch={epoch}/{max_epochs} batch={batch_index}/{total_batches} "
                f"pct={100.0 * batch_index / total_batches:.1f} "
                f"loss={total_loss / total_examples:.6f} "
                f"acc={total_correct / total_examples:.6f} "
                f"ex_per_s={total_examples / elapsed:.1f}",
                flush=True,
            )
    if pending_step:
        training_runtime.step_optimizer()
        training_runtime.zero_grad()
    return {
        "epoch": epoch,
        "mean_loss": round(total_loss / total_examples, 6),
        "hard_loss": round(total_hard_loss / total_examples, 6),
        "soft_loss": round(total_soft_loss / total_examples, 6),
        "accuracy": round(total_correct / total_examples, 6),
        "learning_rate": round(training_runtime.current_learning_rate(), 8),
    }


def _build_loader(
    *,
    rows: list[ManifestRow],
    speaker_to_index: dict[str, int],
    project_root: Path,
    baseline: Any,
    total_epochs: int,
    device: torch.device,
    soft_indices: list[tuple[int, ...]],
    soft_probs: list[tuple[float, ...]],
    sample_weights: list[float],
    is_pseudo: list[bool],
) -> tuple[SoftManifestSpeakerDataset, BalancedSpeakerBatchSampler, DataLoader[SoftTrainingBatch]]:
    audio_request = AudioLoadRequest.from_config(
        baseline.project.normalization,
        vad=baseline.project.vad,
    )
    feature_request = FbankExtractionRequest.from_config(baseline.project.features)
    chunking_request = UtteranceChunkingRequest.from_config(baseline.project.chunking)
    augmentation_runtime = TrainingAugmentationRuntime.from_project_config(
        project_root=project_root,
        scheduler_config=baseline.project.augmentation_scheduler,
        silence_config=baseline.project.silence_augmentation,
        total_epochs=total_epochs,
    )
    if (
        not baseline.project.augmentation_scheduler.enabled
        or not augmentation_runtime.has_effective_augmentation
    ):
        augmentation_runtime = None
    dataset = SoftManifestSpeakerDataset(
        rows=rows,
        speaker_to_index=speaker_to_index,
        project_root=project_root,
        audio_request=audio_request,
        feature_request=feature_request,
        chunking_request=chunking_request,
        seed=baseline.project.runtime.seed,
        augmentation_runtime=augmentation_runtime,
        soft_indices=soft_indices,
        soft_probs=soft_probs,
        sample_weights=sample_weights,
        is_pseudo=is_pseudo,
    )
    sampler = BalancedSpeakerBatchSampler(
        rows=rows,
        batch_size=baseline.project.training.batch_size,
        seed=baseline.project.runtime.seed,
        chunking_request=chunking_request,
        augmentation_runtime=augmentation_runtime,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": baseline.project.runtime.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": _collate_soft_examples,
        "persistent_workers": baseline.project.runtime.num_workers > 0,
    }
    if baseline.project.runtime.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = cast(DataLoader[SoftTrainingBatch], DataLoader(**loader_kwargs))
    return dataset, sampler, loader


def _collate_soft_examples(batch: list[SoftTrainingExample]) -> SoftTrainingBatch:
    if not batch:
        raise ValueError("Training batch must not be empty.")
    first_shape = tuple(batch[0].base.features.shape)
    for example in batch[1:]:
        if tuple(example.base.features.shape) != first_shape:
            raise ValueError("Soft pseudo training requires fixed-size crops.")
    return SoftTrainingBatch(
        features=torch.stack([example.base.features for example in batch], dim=0),
        hard_labels=torch.tensor([example.base.label for example in batch], dtype=torch.long),
        is_pseudo=torch.tensor([example.is_pseudo for example in batch], dtype=torch.bool),
        sample_weights=torch.tensor(
            [example.sample_weight for example in batch], dtype=torch.float32
        ),
        soft_indices=torch.tensor([example.soft_indices for example in batch], dtype=torch.long),
        soft_probs=torch.tensor([example.soft_probs for example in batch], dtype=torch.float32),
    )


def _load_raw_manifest_rows(
    manifest_path: str,
    *,
    project_root: Path,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[ManifestRow]]:
    manifest_file = resolve_project_path(str(project_root), manifest_path)
    raw_rows: list[dict[str, Any]] = []
    rows: list[ManifestRow] = []
    for line_number, raw_line in enumerate(manifest_file.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object row in {manifest_file}:{line_number}.")
        raw_rows.append(payload)
        rows.append(
            ManifestRow.from_mapping(
                payload,
                manifest_path=str(manifest_file),
                line_number=line_number,
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No rows found in {manifest_file}.")
    return raw_rows, rows


def _build_soft_metadata(
    *,
    raw_rows: list[dict[str, Any]],
    speaker_to_index: dict[str, int],
    target_width: int,
) -> tuple[list[tuple[int, ...]], list[tuple[float, ...]], list[float], list[bool]]:
    soft_indices: list[tuple[int, ...]] = []
    soft_probs: list[tuple[float, ...]] = []
    sample_weights: list[float] = []
    is_pseudo: list[bool] = []
    for row in raw_rows:
        speaker_id = str(row["speaker_id"])
        primary_index = speaker_to_index[speaker_id]
        ids = row.get("soft_speaker_ids")
        probs = row.get("soft_probs")
        if isinstance(ids, list) and isinstance(probs, list) and ids and probs:
            pairs = []
            for item, prob in zip(ids, probs, strict=False):
                prob_value = float(cast(Any, prob))
                if str(item) in speaker_to_index and prob_value > 0.0:
                    pairs.append((speaker_to_index[str(item)], prob_value))
            if not pairs:
                pairs = [(primary_index, 1.0)]
            total = sum(prob for _, prob in pairs)
            pairs = [(index, prob / total) for index, prob in pairs[:target_width]]
            pairs.extend([(primary_index, 0.0)] * (target_width - len(pairs)))
            soft_indices.append(tuple(index for index, _ in pairs))
            soft_probs.append(tuple(prob for _, prob in pairs))
            sample_weights.append(float(row.get("pseudo_weight", 1.0)))
            is_pseudo.append(True)
        else:
            soft_indices.append(tuple([primary_index] * target_width))
            soft_probs.append(tuple([1.0] + [0.0] * (target_width - 1)))
            sample_weights.append(1.0)
            is_pseudo.append(False)
    return soft_indices, soft_probs, sample_weights, is_pseudo


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--project-override", action="append", default=[])
    parser.add_argument("--device", default=None)
    parser.add_argument("--soft-target-width", type=int, default=3)
    parser.add_argument("--hard-loss-weight", type=float, default=1.0)
    parser.add_argument("--soft-loss-weight", type=float, default=0.55)
    parser.add_argument("--pseudo-hard-loss-weight", type=float, default=0.0)
    parser.add_argument("--result-json", default="")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser.parse_args()


if __name__ == "__main__":
    main()
