"""Embedding export helpers for manifest-backed speaker baselines."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as torch_functional

from kryptonite.config import ChunkingConfig
from kryptonite.data import AudioLoadRequest, ManifestRow, load_manifest_audio
from kryptonite.deployment import resolve_project_path
from kryptonite.features import (
    FbankExtractionRequest,
    FbankExtractor,
    UtteranceChunkingRequest,
    chunk_utterance,
    pool_chunk_tensors,
)

from .speaker_baseline_types import (
    EMBEDDING_METADATA_JSONL_NAME,
    EMBEDDING_METADATA_PARQUET_NAME,
    EMBEDDINGS_FILE_NAME,
    EmbeddingExportSummary,
)


def export_dev_embeddings(
    *,
    output_root: Path,
    model: torch.nn.Module,
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
    eval_chunking_request = UtteranceChunkingRequest.from_config(chunking)
    manifest_metadata_lookup = _load_manifest_metadata_lookup(
        manifest_path=manifest_path,
        project_root=project_root,
    )
    metadata_rows: list[dict[str, Any]] = []
    embeddings: list[torch.Tensor] = []
    point_ids: list[str] = []

    with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        for index, row in enumerate(rows):
            loaded = load_manifest_audio(row, project_root=project_root, request=audio_request)
            eval_chunks = chunk_utterance(
                loaded.audio.waveform,
                sample_rate_hz=loaded.audio.sample_rate_hz,
                stage="eval",
                request=eval_chunking_request,
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
                chunk_embeddings.append(embedding)
            pooled = pool_chunk_tensors(
                [e.detach().cpu().float() for e in chunk_embeddings],
                pooling_mode=eval_chunks.pooling_mode,
            )
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

    embeddings_matrix = torch.stack(embeddings, dim=0).float().numpy()
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
