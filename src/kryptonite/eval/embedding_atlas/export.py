"""Manifest-backed baseline embedding export for atlas workflows."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.data import AudioLoadRequest, load_manifest_audio
from kryptonite.data.manifest_artifacts import write_tabular_artifact
from kryptonite.deployment import resolve_project_path
from kryptonite.features import (
    SUPPORTED_CHUNKING_STAGES,
    FbankExtractionRequest,
    FbankExtractor,
    UtteranceChunkingRequest,
    chunk_utterance,
    pool_chunk_tensors,
)

SUPPORTED_BASELINE_EMBEDDING_MODES = frozenset({"mean", "mean_std"})
SUPPORTED_METADATA_SOURCE_FORMATS = frozenset({"jsonl", "csv", "parquet"})

EMBEDDINGS_NPZ_NAME = "manifest_embeddings.npz"
METADATA_JSONL_NAME = "manifest_embedding_metadata.jsonl"
METADATA_PARQUET_NAME = "manifest_embedding_metadata.parquet"
EXPORT_REPORT_JSON_NAME = "manifest_embedding_export.json"


@dataclass(frozen=True, slots=True)
class ManifestEmbeddingExportSummary:
    point_count: int
    speaker_count: int
    embedding_dim: int
    total_audio_seconds: float
    mean_audio_seconds: float
    stage: str
    embedding_mode: str
    device: str
    max_rows: int | None
    max_per_speaker: int | None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "point_count": self.point_count,
            "speaker_count": self.speaker_count,
            "embedding_dim": self.embedding_dim,
            "total_audio_seconds": self.total_audio_seconds,
            "mean_audio_seconds": self.mean_audio_seconds,
            "stage": self.stage,
            "embedding_mode": self.embedding_mode,
            "device": self.device,
        }
        if self.max_rows is not None:
            payload["max_rows"] = self.max_rows
        if self.max_per_speaker is not None:
            payload["max_per_speaker"] = self.max_per_speaker
        return payload


@dataclass(frozen=True, slots=True)
class ManifestEmbeddingExportArtifacts:
    manifest_path: str
    embeddings_path: str
    metadata_jsonl_path: str
    metadata_csv_path: str
    metadata_parquet_path: str
    report_path: str
    summary: ManifestEmbeddingExportSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "embeddings_path": self.embeddings_path,
            "metadata_jsonl_path": self.metadata_jsonl_path,
            "metadata_csv_path": self.metadata_csv_path,
            "metadata_parquet_path": self.metadata_parquet_path,
            "report_path": self.report_path,
            "summary": self.summary.to_dict(),
        }


def export_manifest_fbank_embeddings(
    *,
    project_root: Path | str,
    manifest_path: Path | str,
    output_root: Path | str,
    audio_request: AudioLoadRequest,
    fbank_request: FbankExtractionRequest,
    chunking_request: UtteranceChunkingRequest,
    stage: str = "eval",
    embedding_mode: str = "mean_std",
    device: str = "auto",
    max_rows: int | None = None,
    max_per_speaker: int | None = None,
) -> ManifestEmbeddingExportArtifacts:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")
    if max_per_speaker is not None and max_per_speaker <= 0:
        raise ValueError("max_per_speaker must be positive when provided")

    normalized_stage = _normalize_stage(stage)
    normalized_embedding_mode = _normalize_embedding_mode(embedding_mode)
    project_root_path = resolve_project_path(str(project_root), ".")
    manifest_file = resolve_project_path(str(project_root_path), str(manifest_path))
    output_root_path = resolve_project_path(str(project_root_path), str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    torch = _import_torch()
    resolved_device = _resolve_torch_device(torch=torch, preference=device)
    extractor = FbankExtractor(request=fbank_request)
    manifest_location = _relative_to_project(manifest_file, project_root_path)

    embeddings: list[np.ndarray] = []
    point_ids: list[str] = []
    metadata_rows: list[dict[str, object]] = []
    speaker_counts: Counter[str] = Counter()
    total_audio_seconds = 0.0

    with manifest_file.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSONL rows in {manifest_location}:{line_number}")

            speaker_id = _coerce_string(payload.get("speaker_id")) or "__missing_speaker__"
            if max_per_speaker is not None and speaker_counts[speaker_id] >= max_per_speaker:
                continue

            loaded = load_manifest_audio(
                payload,
                project_root=project_root_path,
                request=audio_request,
                manifest_path=manifest_location,
                line_number=line_number,
            )
            embedding_vector, chunk_count = _compute_embedding_vector(
                torch=torch,
                waveform=loaded.audio.waveform,
                sample_rate_hz=loaded.audio.sample_rate_hz,
                extractor=extractor,
                chunking_request=chunking_request,
                stage=normalized_stage,
                embedding_mode=normalized_embedding_mode,
                device=resolved_device,
            )
            point_id = _build_point_id(
                payload=payload, manifest_location=manifest_location, line_number=line_number
            )

            speaker_counts[speaker_id] += 1
            total_audio_seconds += loaded.audio.duration_seconds
            embeddings.append(embedding_vector)
            point_ids.append(point_id)
            metadata_rows.append(
                {
                    **payload,
                    "atlas_point_id": point_id,
                    "manifest_path": manifest_location,
                    "manifest_line_number": line_number,
                    "resolved_audio_path": _relative_to_project(
                        Path(loaded.audio.resolved_path),
                        project_root_path,
                    ),
                    "loaded_duration_seconds": loaded.audio.duration_seconds,
                    "loaded_sample_rate_hz": loaded.audio.sample_rate_hz,
                    "loaded_num_channels": loaded.audio.num_channels,
                    "chunk_count": chunk_count,
                    "embedding_stage": normalized_stage,
                    "embedding_mode": normalized_embedding_mode,
                    "embedding_device": str(resolved_device),
                }
            )
            if max_rows is not None and len(embeddings) >= max_rows:
                break

    if not embeddings:
        raise ValueError(f"No manifest rows were exported from {manifest_location}.")

    embeddings_matrix = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    npz_path = output_root_path / EMBEDDINGS_NPZ_NAME
    np.savez_compressed(
        npz_path,
        embeddings=embeddings_matrix,
        point_ids=np.asarray(point_ids, dtype=np.str_),
    )

    metadata_jsonl_path = output_root_path / METADATA_JSONL_NAME
    metadata_artifact = write_tabular_artifact(
        name="manifest_embedding_metadata",
        kind="metadata",
        rows=metadata_rows,
        jsonl_path=metadata_jsonl_path,
        project_root=str(project_root_path),
    )
    metadata_parquet_path = output_root_path / METADATA_PARQUET_NAME
    pl.DataFrame(metadata_rows).write_parquet(metadata_parquet_path)

    summary = ManifestEmbeddingExportSummary(
        point_count=int(embeddings_matrix.shape[0]),
        speaker_count=len(speaker_counts),
        embedding_dim=int(embeddings_matrix.shape[1]),
        total_audio_seconds=round(total_audio_seconds, 6),
        mean_audio_seconds=round(total_audio_seconds / float(len(embeddings)), 6),
        stage=normalized_stage,
        embedding_mode=normalized_embedding_mode,
        device=str(resolved_device),
        max_rows=max_rows,
        max_per_speaker=max_per_speaker,
    )
    artifacts = ManifestEmbeddingExportArtifacts(
        manifest_path=manifest_location,
        embeddings_path=str(npz_path),
        metadata_jsonl_path=str(metadata_jsonl_path),
        metadata_csv_path=str(
            resolve_project_path(str(project_root_path), metadata_artifact.csv_path)
        ),
        metadata_parquet_path=str(metadata_parquet_path),
        report_path=str(output_root_path / EXPORT_REPORT_JSON_NAME),
        summary=summary,
    )
    Path(artifacts.report_path).write_text(
        json.dumps(artifacts.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return artifacts


def _compute_embedding_vector(
    *,
    torch: Any,
    waveform: Any,
    sample_rate_hz: int,
    extractor: FbankExtractor,
    chunking_request: UtteranceChunkingRequest,
    stage: str,
    embedding_mode: str,
    device: Any,
) -> tuple[np.ndarray, int]:
    waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32, device=device)
    chunk_batch = chunk_utterance(
        waveform_tensor,
        sample_rate_hz=sample_rate_hz,
        stage=stage,
        request=chunking_request,
    )

    with torch.inference_mode():
        chunk_embeddings = [
            _pool_feature_frames(
                torch=torch,
                features=extractor.extract(
                    chunk.waveform,
                    sample_rate_hz=sample_rate_hz,
                ),
                embedding_mode=embedding_mode,
            )
            for chunk in chunk_batch.chunks
        ]
        pooled = pool_chunk_tensors(
            chunk_embeddings,
            pooling_mode=chunk_batch.pooling_mode,
        )
    return pooled.to(device="cpu", dtype=torch.float32).numpy(), len(chunk_embeddings)


def _pool_feature_frames(
    *,
    torch: Any,
    features: Any,
    embedding_mode: str,
) -> Any:
    if features.ndim != 2 or int(features.shape[0]) == 0:
        raise ValueError("Expected non-empty [frames, features] tensors from the Fbank extractor.")

    feature_matrix = features.to(dtype=torch.float32)
    mean = feature_matrix.mean(dim=0)
    if embedding_mode == "mean":
        return mean
    std = feature_matrix.std(dim=0, correction=0)
    return torch.cat((mean, std), dim=0)


def _build_point_id(
    *,
    payload: dict[str, object],
    manifest_location: str,
    line_number: int,
) -> str:
    utterance_id = _coerce_string(payload.get("utterance_id"))
    if utterance_id is not None:
        return utterance_id
    return f"{manifest_location}:{line_number}"


def _normalize_stage(stage: str) -> str:
    normalized = stage.lower()
    if normalized not in SUPPORTED_CHUNKING_STAGES:
        raise ValueError(f"stage must be one of {sorted(SUPPORTED_CHUNKING_STAGES)}")
    return normalized


def _normalize_embedding_mode(embedding_mode: str) -> str:
    normalized = embedding_mode.lower()
    if normalized not in SUPPORTED_BASELINE_EMBEDDING_MODES:
        raise ValueError(
            f"embedding_mode must be one of {sorted(SUPPORTED_BASELINE_EMBEDDING_MODES)}"
        )
    return normalized


def _resolve_torch_device(*, torch: Any, preference: str) -> Any:
    normalized = preference.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"device={preference!r} requested, but CUDA is not available")
    return torch.device(preference)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Manifest embedding export requires torch. Sync the repository environment with "
            "`uv sync --dev --group train`."
        ) from exc
    return torch


def _coerce_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


__all__ = [
    "EMBEDDINGS_NPZ_NAME",
    "EXPORT_REPORT_JSON_NAME",
    "METADATA_JSONL_NAME",
    "METADATA_PARQUET_NAME",
    "ManifestEmbeddingExportArtifacts",
    "ManifestEmbeddingExportSummary",
    "SUPPORTED_BASELINE_EMBEDDING_MODES",
    "SUPPORTED_METADATA_SOURCE_FORMATS",
    "export_manifest_fbank_embeddings",
]
