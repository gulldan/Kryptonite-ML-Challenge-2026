"""Fine-tuning runtime for Hugging Face AudioXVector speaker models."""

from __future__ import annotations

import json
import math
import os
import random
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from kryptonite.data import AudioLoadRequest, ManifestRow, load_manifest_audio
from kryptonite.deployment import resolve_project_path
from kryptonite.eda.hf_xvector import extract_xvector_embeddings
from kryptonite.features import UtteranceChunkingRequest, chunk_utterance
from kryptonite.models import ArcMarginLoss, CosineClassifier
from kryptonite.repro import set_global_seed
from kryptonite.training.manifest_speaker_data import (
    TrainingSampleRequest,
    build_speaker_index,
    load_manifest_rows,
)
from kryptonite.training.production_dataloader import BalancedSpeakerBatchSampler
from kryptonite.training.speaker_baseline import resolve_device


@dataclass(frozen=True, slots=True)
class HfXVectorFineTuneConfig:
    model_id: str
    train_manifest: str
    output_root: str
    run_id: str = ""
    revision: str = ""
    hf_token_env: str = "HUGGINGFACE_HUB_TOKEN"
    project_root: str = "."
    max_train_rows: int | None = None
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 4
    max_epochs: int = 1
    steps_per_epoch: int = 2000
    crop_seconds: float = 4.0
    train_min_crop_seconds: float | None = None
    train_max_crop_seconds: float | None = None
    learning_rate: float = 2e-5
    classifier_learning_rate: float = 1e-3
    min_learning_rate_ratio: float = 0.05
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip_norm: float = 1.0
    scale: float = 32.0
    margin: float = 0.2
    precision: str = "bf16"
    device: str = "cuda"
    freeze_feature_encoder: bool = True
    gradient_checkpointing: bool = True
    log_every_steps: int = 50
    augmentation_enabled: bool = False
    bandlimit_probability: float = 0.0
    edge_silence_probability: float = 0.0
    max_leading_silence_seconds: float = 0.0
    max_trailing_silence_seconds: float = 0.0
    peak_limit_probability: float = 0.0
    gain_probability: float = 0.0
    gain_min_db: float = -3.0
    gain_max_db: float = 3.0
    noise_probability: float = 0.0
    noise_min_snr_db: float = 18.0
    noise_max_snr_db: float = 35.0

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not self.train_manifest.strip():
            raise ValueError("train_manifest must not be empty")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if self.crop_seconds <= 0.0:
            raise ValueError("crop_seconds must be positive")
        min_crop = self.train_min_crop_seconds or self.crop_seconds
        max_crop = self.train_max_crop_seconds or self.crop_seconds
        if min_crop <= 0.0:
            raise ValueError("train_min_crop_seconds must be positive")
        if max_crop < min_crop:
            raise ValueError("train_max_crop_seconds must be >= train_min_crop_seconds")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.classifier_learning_rate <= 0.0:
            raise ValueError("classifier_learning_rate must be positive")
        if not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ValueError("min_learning_rate_ratio must be within [0, 1]")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be one of: fp32, bf16")
        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive")
        for field_name in (
            "bandlimit_probability",
            "edge_silence_probability",
            "peak_limit_probability",
            "gain_probability",
            "noise_probability",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1]")
        if self.max_leading_silence_seconds < 0.0:
            raise ValueError("max_leading_silence_seconds must be non-negative")
        if self.max_trailing_silence_seconds < 0.0:
            raise ValueError("max_trailing_silence_seconds must be non-negative")
        if self.gain_max_db < self.gain_min_db:
            raise ValueError("gain_max_db must be >= gain_min_db")
        if self.noise_max_snr_db < self.noise_min_snr_db:
            raise ValueError("noise_max_snr_db must be >= noise_min_snr_db")


@dataclass(frozen=True, slots=True)
class HfXVectorEpochSummary:
    epoch: int
    train_loss: float
    train_accuracy: float
    learning_rate: float
    examples: int
    steps: int
    seconds: float


@dataclass(frozen=True, slots=True)
class HfXVectorFineTuneArtifacts:
    run_id: str
    output_root: str
    model_dir: str
    checkpoint_path: str
    training_summary_path: str
    metrics_path: str
    config_snapshot_path: str
    embedding_size: int
    speaker_count: int
    train_row_count: int
    epochs: tuple[HfXVectorEpochSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "output_root": self.output_root,
            "model_dir": self.model_dir,
            "checkpoint_path": self.checkpoint_path,
            "training_summary_path": self.training_summary_path,
            "metrics_path": self.metrics_path,
            "config_snapshot_path": self.config_snapshot_path,
            "embedding_size": self.embedding_size,
            "speaker_count": self.speaker_count,
            "train_row_count": self.train_row_count,
            "epochs": [asdict(epoch) for epoch in self.epochs],
        }


@dataclass(frozen=True, slots=True)
class RawWaveformExample:
    waveform: np.ndarray
    label: int
    speaker_id: str
    utterance_id: str | None


class RawWaveformSpeakerDataset(Dataset[RawWaveformExample]):
    def __init__(
        self,
        *,
        rows: list[ManifestRow],
        speaker_to_index: dict[str, int],
        project_root: Path | str,
        audio_request: AudioLoadRequest,
        chunking_request: UtteranceChunkingRequest,
        augmentation_config: HfXVectorFineTuneConfig,
        seed: int,
    ) -> None:
        self._rows = list(rows)
        self._speaker_to_index = dict(speaker_to_index)
        self._project_root = resolve_project_path(str(project_root), ".")
        self._audio_request = audio_request
        self._chunking_request = chunking_request
        self._augmentation_config = augmentation_config
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self._rows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __getitem__(self, index: int | TrainingSampleRequest) -> RawWaveformExample:
        if isinstance(index, TrainingSampleRequest):
            request = index
            row_index = request.row_index
        else:
            request = None
            row_index = index
        row = self._rows[row_index]
        loaded = load_manifest_audio(
            row,
            project_root=self._project_root,
            request=self._audio_request,
        )
        rng = random.Random(
            request.request_seed
            if request is not None
            else self._seed + (self._epoch * len(self._rows)) + row_index
        )
        chunk_batch = chunk_utterance(
            loaded.audio.waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
            stage="train",
            request=self._chunking_request,
            rng=rng,
        )
        if len(chunk_batch.chunks) != 1:
            raise ValueError("Hugging Face xvector fine-tune expects one waveform crop.")
        waveform = chunk_batch.chunks[0].waveform.squeeze(0).to(torch.float32).numpy()
        waveform = _apply_training_augmentations(
            waveform,
            rng=rng,
            config=self._augmentation_config,
        )
        return RawWaveformExample(
            waveform=waveform.astype(np.float32, copy=False),
            label=self._speaker_to_index[row.speaker_id],
            speaker_id=row.speaker_id,
            utterance_id=row.utterance_id,
        )


class HfXVectorCollator:
    def __init__(self, feature_extractor: Any) -> None:
        self._feature_extractor = feature_extractor

    def __call__(self, batch: list[RawWaveformExample]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Training batch must not be empty")
        inputs = self._feature_extractor(
            [example.waveform for example in batch],
            sampling_rate=16_000,
            padding=True,
            return_tensors="pt",
        )
        inputs["labels"] = torch.tensor([example.label for example in batch], dtype=torch.long)
        return dict(inputs)


def load_hf_xvector_finetune_config(config_path: Path | str) -> HfXVectorFineTuneConfig:
    payload = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    section = payload.get("run", payload)
    if not isinstance(section, dict):
        raise ValueError("Hugging Face xvector config must contain a [run] table.")
    return HfXVectorFineTuneConfig(**section)


def run_hf_xvector_finetune(
    config: HfXVectorFineTuneConfig,
    *,
    config_path: Path | str,
    device_override: str | None = None,
    run_id_override: str | None = None,
) -> HfXVectorFineTuneArtifacts:
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    set_global_seed(config.seed, deterministic=False, pythonhashseed=config.seed)
    device = resolve_device(device_override or config.device)
    project_root = resolve_project_path(config.project_root, ".")
    run_id = run_id_override or config.run_id or _default_run_id(config.model_id)
    output_root = resolve_project_path(str(project_root), config.output_root) / run_id
    model_dir = output_root / "hf_model"
    tracking_root = resolve_project_path(str(project_root), "artifacts/tracking") / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    tracking_root.mkdir(parents=True, exist_ok=True)

    token = os.environ.get(config.hf_token_env) or None
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        config.model_id,
        revision=config.revision or None,
        token=token,
    )
    model = AutoModelForAudioXVector.from_pretrained(
        config.model_id,
        revision=config.revision or None,
        token=token,
    ).to(device)
    _maybe_enable_gradient_checkpointing(model, enabled=config.gradient_checkpointing)
    _maybe_freeze_feature_encoder(model, enabled=config.freeze_feature_encoder)

    train_rows = load_manifest_rows(
        config.train_manifest,
        project_root=project_root,
        limit=config.max_train_rows,
    )
    speaker_to_index = build_speaker_index(train_rows)
    embedding_size = _infer_embedding_size(
        model=model,
        feature_extractor=feature_extractor,
        device=device,
    )
    classifier = CosineClassifier(
        embedding_size,
        num_classes=len(speaker_to_index),
    ).to(device)
    criterion = ArcMarginLoss(scale=config.scale, margin=config.margin)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": config.learning_rate},
            {"params": classifier.parameters(), "lr": config.classifier_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    total_steps = config.max_epochs * config.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=config.warmup_steps,
            min_ratio=config.min_learning_rate_ratio,
        ),
    )
    dataset, sampler, loader = _build_loader(
        rows=train_rows,
        speaker_to_index=speaker_to_index,
        project_root=project_root,
        feature_extractor=feature_extractor,
        config=config,
        pin_memory=device.type == "cuda",
    )

    metrics_path = tracking_root / "metrics.jsonl"
    training_summary_path = output_root / "training_summary.json"
    config_snapshot_path = output_root / "config_snapshot.json"
    config_snapshot_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "config_path": str(config_path),
                "run_id": run_id,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    epoch_summaries: list[HfXVectorEpochSummary] = []
    for epoch in range(config.max_epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        epoch_summary = _train_one_epoch(
            model=model,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=loader,
            device=device,
            config=config,
            epoch=epoch,
        )
        epoch_summaries.append(epoch_summary)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(epoch_summary), sort_keys=True) + "\n")
        _write_training_summary(
            path=training_summary_path,
            config=config,
            run_id=run_id,
            output_root=output_root,
            model_dir=model_dir,
            checkpoint_path=output_root / "hf_xvector_finetune.pt",
            metrics_path=metrics_path,
            embedding_size=embedding_size,
            speaker_count=len(speaker_to_index),
            train_row_count=len(train_rows),
            epoch_summaries=epoch_summaries,
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    feature_extractor.save_pretrained(model_dir)
    checkpoint_path = output_root / "hf_xvector_finetune.pt"
    torch.save(
        {
            "model_id": config.model_id,
            "revision": config.revision,
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "speaker_to_index": speaker_to_index,
            "embedding_size": embedding_size,
            "config": asdict(config),
            "epochs": [asdict(epoch) for epoch in epoch_summaries],
        },
        checkpoint_path,
    )
    _write_training_summary(
        path=training_summary_path,
        config=config,
        run_id=run_id,
        output_root=output_root,
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        embedding_size=embedding_size,
        speaker_count=len(speaker_to_index),
        train_row_count=len(train_rows),
        epoch_summaries=epoch_summaries,
    )

    return HfXVectorFineTuneArtifacts(
        run_id=run_id,
        output_root=str(output_root),
        model_dir=str(model_dir),
        checkpoint_path=str(checkpoint_path),
        training_summary_path=str(training_summary_path),
        metrics_path=str(metrics_path),
        config_snapshot_path=str(config_snapshot_path),
        embedding_size=embedding_size,
        speaker_count=len(speaker_to_index),
        train_row_count=len(train_rows),
        epochs=tuple(epoch_summaries),
    )


def _build_loader(
    *,
    rows: list[ManifestRow],
    speaker_to_index: dict[str, int],
    project_root: Path,
    feature_extractor: Any,
    config: HfXVectorFineTuneConfig,
    pin_memory: bool,
) -> tuple[RawWaveformSpeakerDataset, BalancedSpeakerBatchSampler, DataLoader[dict[str, Any]]]:
    audio_request = AudioLoadRequest(
        target_sample_rate_hz=16_000,
        target_channels=1,
        vad_mode="none",
        loudness_mode="none",
    )
    train_min_crop_seconds = config.train_min_crop_seconds or config.crop_seconds
    train_max_crop_seconds = config.train_max_crop_seconds or config.crop_seconds
    chunking_request = UtteranceChunkingRequest(
        train_min_crop_seconds=train_min_crop_seconds,
        train_max_crop_seconds=train_max_crop_seconds,
        train_num_crops=1,
        train_short_utterance_policy="repeat_pad",
        eval_max_full_utterance_seconds=config.crop_seconds,
        eval_chunk_seconds=config.crop_seconds,
        eval_chunk_overlap_seconds=0.0,
        eval_pooling="mean",
        demo_max_full_utterance_seconds=config.crop_seconds,
        demo_chunk_seconds=config.crop_seconds,
        demo_chunk_overlap_seconds=0.0,
        demo_pooling="mean",
    )
    dataset = RawWaveformSpeakerDataset(
        rows=rows,
        speaker_to_index=speaker_to_index,
        project_root=project_root,
        audio_request=audio_request,
        chunking_request=chunking_request,
        augmentation_config=config,
        seed=config.seed,
    )
    sampler = BalancedSpeakerBatchSampler(
        rows=rows,
        batch_size=config.batch_size,
        seed=config.seed,
        chunking_request=chunking_request,
        batches_per_epoch=config.steps_per_epoch,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": HfXVectorCollator(feature_extractor),
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = cast(DataLoader[dict[str, Any]], DataLoader(**loader_kwargs))
    return dataset, sampler, loader


def _apply_training_augmentations(
    waveform: np.ndarray,
    *,
    rng: random.Random,
    config: HfXVectorFineTuneConfig,
) -> np.ndarray:
    if not config.augmentation_enabled:
        return waveform.astype(np.float32, copy=False)

    augmented = waveform.astype(np.float32, copy=True)
    if rng.random() < config.bandlimit_probability:
        augmented = _apply_random_bandlimit(augmented, rng=rng)
    if rng.random() < config.peak_limit_probability:
        augmented = _apply_mild_peak_limit(augmented, rng=rng)
    if rng.random() < config.gain_probability:
        augmented = _apply_random_gain(augmented, rng=rng, config=config)
    if rng.random() < config.noise_probability:
        augmented = _add_gaussian_noise(augmented, rng=rng, config=config)
    if rng.random() < config.edge_silence_probability:
        augmented = _add_edge_silence(augmented, rng=rng, config=config)
    return np.clip(augmented, -1.0, 1.0).astype(np.float32, copy=False)


def _apply_random_bandlimit(waveform: np.ndarray, *, rng: random.Random) -> np.ndarray:
    if waveform.size < 16:
        return waveform
    effective_rate = rng.uniform(7_000.0, 10_000.0)
    downsampled_size = max(16, int(round(float(waveform.size) * effective_rate / 16_000.0)))
    if downsampled_size >= waveform.size:
        return waveform
    source_x = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False, dtype=np.float32)
    down_x = np.linspace(0.0, 1.0, num=downsampled_size, endpoint=False, dtype=np.float32)
    downsampled = np.interp(down_x, source_x, waveform).astype(np.float32, copy=False)
    return np.interp(source_x, down_x, downsampled).astype(np.float32, copy=False)


def _apply_mild_peak_limit(waveform: np.ndarray, *, rng: random.Random) -> np.ndarray:
    drive = rng.uniform(1.35, 2.75)
    normalizer = math.tanh(drive)
    if normalizer <= 0.0:
        return waveform
    return (np.tanh(waveform * drive) / normalizer).astype(np.float32, copy=False)


def _apply_random_gain(
    waveform: np.ndarray,
    *,
    rng: random.Random,
    config: HfXVectorFineTuneConfig,
) -> np.ndarray:
    gain_db = rng.uniform(config.gain_min_db, config.gain_max_db)
    return (waveform * (10.0 ** (gain_db / 20.0))).astype(np.float32, copy=False)


def _add_gaussian_noise(
    waveform: np.ndarray,
    *,
    rng: random.Random,
    config: HfXVectorFineTuneConfig,
) -> np.ndarray:
    signal_rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32))))
    if signal_rms <= 1e-6:
        return waveform
    snr_db = rng.uniform(config.noise_min_snr_db, config.noise_max_snr_db)
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    generator = np.random.default_rng(rng.randrange(0, 2**32))
    noise = generator.standard_normal(waveform.shape, dtype=np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float32))))
    if noise_rms <= 1e-9:
        return waveform
    return (waveform + (noise * (target_noise_rms / noise_rms))).astype(
        np.float32,
        copy=False,
    )


def _add_edge_silence(
    waveform: np.ndarray,
    *,
    rng: random.Random,
    config: HfXVectorFineTuneConfig,
) -> np.ndarray:
    leading_samples = int(round(rng.uniform(0.0, config.max_leading_silence_seconds) * 16_000))
    trailing_samples = int(round(rng.uniform(0.0, config.max_trailing_silence_seconds) * 16_000))
    if leading_samples <= 0 and trailing_samples <= 0:
        return waveform
    parts: list[np.ndarray] = []
    if leading_samples > 0:
        parts.append(np.zeros(leading_samples, dtype=np.float32))
    parts.append(waveform.astype(np.float32, copy=False))
    if trailing_samples > 0:
        parts.append(np.zeros(trailing_samples, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: HfXVectorFineTuneConfig,
    epoch: int,
) -> HfXVectorEpochSummary:
    model.train()
    classifier.train()
    started = time.perf_counter()
    loss_sum = 0.0
    correct = 0
    examples = 0
    print(
        f"[hf-xvector-ft] epoch={epoch + 1}/{config.max_epochs} "
        f"start steps={config.steps_per_epoch} batch_size={config.batch_size}",
        flush=True,
    )
    for step, batch in enumerate(loader, start=1):
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and config.precision == "bf16",
        ):
            outputs = model(**inputs)
            embeddings = extract_xvector_embeddings(outputs)
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*model.parameters(), *classifier.parameters()],
            max_norm=config.grad_clip_norm,
        )
        optimizer.step()
        scheduler.step()

        batch_size = int(labels.numel())
        examples += batch_size
        loss_sum += float(loss.detach().item()) * batch_size
        correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        if step == 1 or step % config.log_every_steps == 0 or step == config.steps_per_epoch:
            elapsed_s = max(time.perf_counter() - started, 1e-9)
            print(
                f"[hf-xvector-ft] epoch={epoch + 1}/{config.max_epochs} "
                f"step={step}/{config.steps_per_epoch} "
                f"loss={loss_sum / max(examples, 1):.6f} "
                f"acc={correct / max(examples, 1):.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f} "
                f"ex_per_s={examples / elapsed_s:.1f} elapsed_s={elapsed_s:.1f}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    return HfXVectorEpochSummary(
        epoch=epoch + 1,
        train_loss=loss_sum / max(examples, 1),
        train_accuracy=correct / max(examples, 1),
        learning_rate=float(optimizer.param_groups[0]["lr"]),
        examples=examples,
        steps=config.steps_per_epoch,
        seconds=round(seconds, 6),
    )


def _infer_embedding_size(
    *, model: torch.nn.Module, feature_extractor: Any, device: torch.device
) -> int:
    model.eval()
    dummy = np.zeros(16_000, dtype=np.float32)
    inputs = feature_extractor(
        [dummy],
        sampling_rate=16_000,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = extract_xvector_embeddings(outputs)
    return int(embeddings.shape[-1])


def _maybe_enable_gradient_checkpointing(model: torch.nn.Module, *, enabled: bool) -> None:
    method = getattr(model, "gradient_checkpointing_enable", None)
    if enabled and callable(method):
        method()


def _maybe_freeze_feature_encoder(model: torch.nn.Module, *, enabled: bool) -> None:
    if not enabled:
        return
    for method_name in ("freeze_feature_encoder", "freeze_feature_extractor"):
        method = getattr(model, method_name, None)
        if callable(method):
            method()
            return


def _warmup_cosine_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, float(step + 1) / float(warmup_steps))
    progress_denominator = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(progress_denominator)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + ((1.0 - min_ratio) * cosine)


def _write_training_summary(
    *,
    path: Path,
    config: HfXVectorFineTuneConfig,
    run_id: str,
    output_root: Path,
    model_dir: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    embedding_size: int,
    speaker_count: int,
    train_row_count: int,
    epoch_summaries: list[HfXVectorEpochSummary],
) -> None:
    payload = {
        "run_id": run_id,
        "config": asdict(config),
        "output_root": str(output_root),
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "embedding_size": embedding_size,
        "speaker_count": speaker_count,
        "train_row_count": train_row_count,
        "epochs": [asdict(epoch) for epoch in epoch_summaries],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_run_id(model_id: str) -> str:
    model_slug = "".join(char if char.isalnum() else "-" for char in model_id.lower()).strip("-")
    return f"{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}-{model_slug}"


__all__ = [
    "HfXVectorEpochSummary",
    "HfXVectorFineTuneArtifacts",
    "HfXVectorFineTuneConfig",
    "load_hf_xvector_finetune_config",
    "run_hf_xvector_finetune",
]
