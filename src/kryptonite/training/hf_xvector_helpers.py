"""Helper routines for Hugging Face xvector fine-tuning."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from kryptonite.eda.hf_xvector import extract_xvector_embeddings

if TYPE_CHECKING:
    from .hf_xvector import HfXVectorEpochSummary, HfXVectorFineTuneConfig


def apply_training_augmentations(
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


def train_one_epoch(
    *,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader: torch.utils.data.DataLoader[dict[str, Any]],
    device: torch.device,
    config: HfXVectorFineTuneConfig,
    epoch: int,
) -> HfXVectorEpochSummary:
    from .hf_xvector import HfXVectorEpochSummary

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


def infer_embedding_size(
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


def maybe_enable_gradient_checkpointing(model: torch.nn.Module, *, enabled: bool) -> None:
    method = getattr(model, "gradient_checkpointing_enable", None)
    if enabled and callable(method):
        method()


def maybe_freeze_feature_encoder(model: torch.nn.Module, *, enabled: bool) -> None:
    if not enabled:
        return
    for method_name in ("freeze_feature_encoder", "freeze_feature_extractor"):
        method = getattr(model, method_name, None)
        if callable(method):
            method()
            return


def warmup_cosine_multiplier(
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


def write_training_summary(
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


def default_run_id(model_id: str) -> str:
    model_slug = "".join(char if char.isalnum() else "-" for char in model_id.lower()).strip("-")
    return f"{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}-{model_slug}"
