"""Audio/trial helpers for ONNX Runtime parity reports."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kryptonite.data import AudioLoadRequest, load_audio
from kryptonite.features import (
    FbankExtractor,
    UtteranceChunkingRequest,
    chunk_utterance,
    pool_chunk_tensors,
)
from kryptonite.models import cosine_score_pairs

from .onnx_parity_config import ONNXParityVariantConfig
from .onnx_parity_models import ONNXParityAudioRecord, ONNXParityTrialRecord
from .verification_data import resolve_trial_side_identifier


@dataclass(frozen=True, slots=True)
class ResolvedAudioItem:
    item_id: str
    speaker_id: str | None
    role: str | None
    source_audio_path: str
    resolved_audio_path: str


@dataclass(frozen=True, slots=True)
class ResolvedTrial:
    label: int
    left_id: str
    right_id: str


def resolve_trials_and_audio_items(
    *,
    trial_rows: list[dict[str, Any]],
    metadata_index: dict[str, dict[str, Any]],
    prefer_demo_subset: bool,
) -> tuple[list[ResolvedTrial], dict[str, ResolvedAudioItem]]:
    resolved_trials: list[ResolvedTrial] = []
    audio_items: dict[str, ResolvedAudioItem] = {}
    for index, row in enumerate(trial_rows, start=1):
        left_id = resolve_trial_side_identifier(row, "left")
        right_id = resolve_trial_side_identifier(row, "right")
        if left_id is None or right_id is None:
            raise ValueError(f"Trial row {index} is missing left/right identifiers: {row!r}.")
        try:
            label = int(row["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Trial row {index} must define integer `label`.") from exc
        if label not in {0, 1}:
            raise ValueError(f"Trial row {index} has invalid label {label!r}; expected 0 or 1.")
        resolved_trials.append(ResolvedTrial(label=label, left_id=left_id, right_id=right_id))
        for item_id, side in ((left_id, "left"), (right_id, "right")):
            if item_id in audio_items:
                continue
            metadata_row = metadata_index.get(item_id)
            if metadata_row is None:
                raise ValueError(f"Could not resolve metadata row for trial item {item_id!r}.")
            audio_items[item_id] = resolve_audio_item(
                item_id=item_id,
                metadata_row=metadata_row,
                prefer_demo_subset=prefer_demo_subset,
                side=side,
            )
    if not resolved_trials:
        raise ValueError("ONNX parity report requires at least one verification trial.")
    return resolved_trials, audio_items


def resolve_audio_item(
    *,
    item_id: str,
    metadata_row: dict[str, Any],
    prefer_demo_subset: bool,
    side: str,
) -> ResolvedAudioItem:
    source_audio_path = require_string(metadata_row.get("audio_path"), field_name="audio_path")
    resolved_audio_path = source_audio_path
    demo_subset_path = coerce_optional_string(metadata_row.get("demo_subset_path"))
    if prefer_demo_subset and demo_subset_path is not None:
        resolved_audio_path = demo_subset_path
    role = coerce_optional_string(metadata_row.get("role"))
    if role is None:
        role = "enrollment" if side == "left" else "test"
    return ResolvedAudioItem(
        item_id=item_id,
        speaker_id=coerce_optional_string(metadata_row.get("speaker_id")),
        role=role,
        source_audio_path=source_audio_path,
        resolved_audio_path=resolved_audio_path,
    )


def build_variant_audio_records(
    *,
    variant: ONNXParityVariantConfig,
    audio_items: dict[str, ResolvedAudioItem],
    project_root: Path,
    audio_request: AudioLoadRequest,
    extractor: FbankExtractor,
    chunking_request: UtteranceChunkingRequest,
    embedding_stage: str,
    input_name: str,
    output_name: str,
    torch: Any,
    torch_model: Any,
    onnx_session: Any,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[ONNXParityAudioRecord]]:
    embeddings_by_item: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    records: list[ONNXParityAudioRecord] = []
    for item_id, item in sorted(audio_items.items()):
        record, torch_embedding, onnx_embedding = build_audio_record(
            variant=variant,
            item=item,
            project_root=project_root,
            audio_request=audio_request,
            extractor=extractor,
            chunking_request=chunking_request,
            embedding_stage=embedding_stage,
            input_name=input_name,
            output_name=output_name,
            torch=torch,
            torch_model=torch_model,
            onnx_session=onnx_session,
            seed=seed,
        )
        embeddings_by_item[item_id] = (torch_embedding, onnx_embedding)
        records.append(record)
    return embeddings_by_item, records


def build_audio_record(
    *,
    variant: ONNXParityVariantConfig,
    item: ResolvedAudioItem,
    project_root: Path,
    audio_request: AudioLoadRequest,
    extractor: FbankExtractor,
    chunking_request: UtteranceChunkingRequest,
    embedding_stage: str,
    input_name: str,
    output_name: str,
    torch: Any,
    torch_model: Any,
    onnx_session: Any,
    seed: int,
) -> tuple[ONNXParityAudioRecord, np.ndarray, np.ndarray]:
    loaded = load_audio(item.resolved_audio_path, project_root=project_root, request=audio_request)
    waveform = torch.as_tensor(loaded.waveform, dtype=torch.float32).clone()
    corruption_applied = variant.kind != "identity" and item.role in set(variant.apply_to_roles)
    if corruption_applied:
        waveform = apply_variant_to_waveform(
            waveform=waveform,
            variant=variant,
            seed=stable_seed(seed, variant.variant_id, item.item_id),
        )

    chunk_batch = chunk_utterance(
        waveform,
        sample_rate_hz=loaded.sample_rate_hz,
        stage=embedding_stage,
        request=chunking_request,
    )
    torch_chunk_embeddings: list[Any] = []
    onnx_chunk_embeddings: list[Any] = []
    chunk_max_abs_diffs: list[float] = []
    chunk_mean_abs_diffs: list[float] = []
    for chunk in chunk_batch.chunks:
        features = extractor.extract(chunk.waveform, sample_rate_hz=loaded.sample_rate_hz)
        model_input = features.unsqueeze(0).to(dtype=torch.float32)
        with torch.inference_mode():
            torch_output = torch_model(model_input).detach().cpu()
        onnx_output = np.asarray(
            onnx_session.run(
                [output_name],
                {input_name: model_input.detach().cpu().numpy().astype(np.float32, copy=False)},
            )[0],
            dtype=np.float32,
        )
        diff = np.abs(onnx_output - torch_output.numpy())
        chunk_max_abs_diffs.append(float(diff.max()) if diff.size else 0.0)
        chunk_mean_abs_diffs.append(float(diff.mean()) if diff.size else 0.0)
        torch_chunk_embeddings.append(torch_output[0].to(dtype=torch.float32))
        onnx_chunk_embeddings.append(torch.from_numpy(onnx_output[0]).to(dtype=torch.float32))

    pooled_torch = (
        pool_chunk_tensors(
            torch_chunk_embeddings,
            pooling_mode=chunk_batch.pooling_mode,
        )
        .detach()
        .cpu()
        .numpy()
    )
    pooled_onnx = (
        pool_chunk_tensors(
            onnx_chunk_embeddings,
            pooling_mode=chunk_batch.pooling_mode,
        )
        .detach()
        .cpu()
        .numpy()
    )
    pooled_diff = np.abs(pooled_torch - pooled_onnx)
    pooled_cosine_distance = 1.0 - float(
        cosine_score_pairs(
            pooled_torch.reshape(1, -1),
            pooled_onnx.reshape(1, -1),
            normalize=True,
        )[0]
    )
    record = ONNXParityAudioRecord(
        variant_id=variant.variant_id,
        variant_kind=variant.kind,
        item_id=item.item_id,
        speaker_id=item.speaker_id,
        role=item.role,
        source_audio_path=item.source_audio_path,
        audio_path=item.resolved_audio_path,
        corruption_applied=corruption_applied,
        sample_rate_hz=loaded.sample_rate_hz,
        duration_seconds=round(float(loaded.duration_seconds), 6),
        chunk_count=len(chunk_batch.chunks),
        max_chunk_max_abs_diff=round(max(chunk_max_abs_diffs, default=0.0), 8),
        mean_chunk_mean_abs_diff=round(mean(chunk_mean_abs_diffs), 8),
        pooled_max_abs_diff=round(float(pooled_diff.max()) if pooled_diff.size else 0.0, 8),
        pooled_mean_abs_diff=round(float(pooled_diff.mean()) if pooled_diff.size else 0.0, 8),
        pooled_cosine_distance=round(pooled_cosine_distance, 8),
        torch_embedding_norm=round(float(np.linalg.norm(pooled_torch)), 8),
        onnx_embedding_norm=round(float(np.linalg.norm(pooled_onnx)), 8),
    )
    return record, pooled_torch.astype(np.float32), pooled_onnx.astype(np.float32)


def build_variant_trial_records(
    *,
    variant: ONNXParityVariantConfig,
    trials: list[ResolvedTrial],
    audio_items: dict[str, ResolvedAudioItem],
    embeddings_by_item: dict[str, tuple[np.ndarray, np.ndarray]],
    normalize_scores: bool,
) -> list[ONNXParityTrialRecord]:
    records: list[ONNXParityTrialRecord] = []
    for trial in trials:
        left_torch, left_onnx = embeddings_by_item[trial.left_id]
        right_torch, right_onnx = embeddings_by_item[trial.right_id]
        torch_score = float(
            cosine_score_pairs(
                left_torch.reshape(1, -1),
                right_torch.reshape(1, -1),
                normalize=normalize_scores,
            )[0]
        )
        onnx_score = float(
            cosine_score_pairs(
                left_onnx.reshape(1, -1),
                right_onnx.reshape(1, -1),
                normalize=normalize_scores,
            )[0]
        )
        left_item = audio_items[trial.left_id]
        right_item = audio_items[trial.right_id]
        records.append(
            ONNXParityTrialRecord(
                variant_id=variant.variant_id,
                variant_kind=variant.kind,
                label=trial.label,
                left_id=trial.left_id,
                right_id=trial.right_id,
                left_audio_path=left_item.resolved_audio_path,
                right_audio_path=right_item.resolved_audio_path,
                torch_score=round(torch_score, 8),
                onnx_score=round(onnx_score, 8),
                score_abs_diff=round(abs(torch_score - onnx_score), 8),
            )
        )
    return records


def apply_variant_to_waveform(
    *,
    waveform: Any,
    variant: ONNXParityVariantConfig,
    seed: int,
) -> Any:
    if variant.kind == "identity":
        return waveform
    rng = np.random.default_rng(seed)
    if variant.kind == "gaussian_noise":
        signal_power = float(waveform.square().mean().item())
        snr_linear = 10.0 ** (float(variant.snr_db or 0.0) / 10.0)
        noise_power = max(signal_power / max(snr_linear, 1e-12), 1e-8)
        noise = rng.normal(loc=0.0, scale=math.sqrt(noise_power), size=tuple(waveform.shape))
        return (waveform + waveform.new_tensor(noise)).clamp_(-1.0, 1.0)
    if variant.kind == "clip":
        gain = 10.0 ** (float(variant.pre_gain_db or 0.0) / 20.0)
        return (waveform * gain).clamp_(-variant.clip_amplitude, variant.clip_amplitude)

    total_samples = int(waveform.shape[-1])
    pause_length = max(1, int(round(total_samples * float(variant.pause_ratio or 0.0))))
    pause_length = min(pause_length, max(total_samples - 1, 1))
    max_start = max(total_samples - pause_length, 1)
    start_index = int(rng.integers(0, max_start))
    augmented = waveform.clone()
    augmented[..., start_index : start_index + pause_length] = 0.0
    return augmented


def stable_seed(base_seed: int, variant_id: str, item_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{variant_id}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def mean(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(float(item) for item in items) / len(items))


def require_string(raw: object, *, field_name: str) -> str:
    value = coerce_optional_string(raw)
    if value is None:
        raise ValueError(f"Metadata rows must define non-empty `{field_name}`.")
    return value


def coerce_optional_string(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    return stripped or None


__all__ = [
    "build_variant_audio_records",
    "build_variant_trial_records",
    "coerce_optional_string",
    "mean",
    "resolve_trials_and_audio_items",
]
