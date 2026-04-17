"""Offline enrollment-embedding cache artifacts for thin runtime adapters."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from kryptonite.deployment import resolve_project_path
from kryptonite.models import average_normalized_embeddings
from kryptonite.serve.inference_backend import (
    FeatureStatisticsEmbeddingBackend,
    normalize_inference_stage,
    normalize_runtime_embedding_mode,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kryptonite.data import AudioLoadRequest
    from kryptonite.features import FbankExtractionRequest, UtteranceChunkingRequest

ENROLLMENT_CACHE_FORMAT_VERSION = "kryptonite.serve.enrollment_cache.v1"
ENROLLMENT_EMBEDDINGS_NPZ_NAME = "enrollment_embeddings.npz"
ENROLLMENT_METADATA_JSONL_NAME = "enrollment_metadata.jsonl"
ENROLLMENT_METADATA_PARQUET_NAME = "enrollment_metadata.parquet"
ENROLLMENT_SUMMARY_JSON_NAME = "enrollment_summary.json"
MODEL_BUNDLE_METADATA_NAME = "metadata.json"
ENROLLMENT_IDS_KEY = "enrollment_ids"


@dataclass(frozen=True, slots=True)
class EnrollmentCacheSummary:
    format_version: str
    source_manifest_path: str
    source_manifest_sha256: str
    model_metadata_path: str
    model_metadata_sha256: str
    compatibility_id: str
    output_root: str
    enrollment_count: int
    source_row_count: int
    source_speaker_count: int
    embedding_dim: int
    total_sample_count: int
    mean_samples_per_enrollment: float
    stage: str
    embedding_mode: str
    device: str
    counts_by_dataset: dict[str, int]
    counts_by_role: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WrittenEnrollmentEmbeddingCache:
    output_root: str
    embeddings_path: str
    metadata_jsonl_path: str
    metadata_parquet_path: str
    summary_path: str
    summary: EnrollmentCacheSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "embeddings_path": self.embeddings_path,
            "metadata_jsonl_path": self.metadata_jsonl_path,
            "metadata_parquet_path": self.metadata_parquet_path,
            "summary_path": self.summary_path,
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LoadedEnrollmentEmbeddingCache:
    embeddings: np.ndarray
    metadata_rows: list[dict[str, Any]]
    summary: EnrollmentCacheSummary


def build_enrollment_embedding_cache(
    *,
    project_root: Path | str,
    manifest_path: Path | str,
    output_root: Path | str,
    model_metadata_path: Path | str,
    audio_request: AudioLoadRequest,
    fbank_request: FbankExtractionRequest,
    chunking_request: UtteranceChunkingRequest,
    stage: str = "demo",
    embedding_mode: str = "mean_std",
    device: str = "auto",
) -> WrittenEnrollmentEmbeddingCache:
    from kryptonite.data import load_manifest_audio

    normalized_stage = normalize_inference_stage(stage)
    normalized_embedding_mode = normalize_runtime_embedding_mode(embedding_mode)
    resolved_project_root = resolve_project_path(str(project_root), ".")
    resolved_manifest_path = resolve_project_path(str(resolved_project_root), str(manifest_path))
    resolved_output_root = resolve_project_path(str(resolved_project_root), str(output_root))
    resolved_model_metadata_path = resolve_project_path(
        str(resolved_project_root),
        str(model_metadata_path),
    )
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    model_metadata = load_model_bundle_metadata(resolved_model_metadata_path)
    compatibility_id = resolve_enrollment_cache_compatibility_id(model_metadata)
    manifest_location = _relative_to_project(resolved_manifest_path, resolved_project_root)
    model_metadata_location = _relative_to_project(
        resolved_model_metadata_path,
        resolved_project_root,
    )

    embedding_backend = FeatureStatisticsEmbeddingBackend(
        feature_request=fbank_request,
        chunking_request=chunking_request,
        embedding_mode=normalized_embedding_mode,
        default_stage=normalized_stage,
        device=device,
    )
    resolved_device = embedding_backend.device

    grouped_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts_by_dataset: Counter[str] = Counter()
    counts_by_role: Counter[str] = Counter()

    with resolved_manifest_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSONL rows in {manifest_location}:{line_number}")

            role = _coerce_string(payload.get("role"))
            if role != "enrollment":
                continue

            enrollment_id = _resolve_enrollment_id(payload)
            loaded = load_manifest_audio(
                payload,
                project_root=resolved_project_root,
                request=audio_request,
                manifest_path=manifest_location,
                line_number=line_number,
            )
            embedding = embedding_backend.embed_waveform(
                waveform=loaded.audio.waveform,
                sample_rate_hz=loaded.audio.sample_rate_hz,
                stage=normalized_stage,
            ).embedding

            grouped_embeddings[enrollment_id].append(embedding)
            grouped_rows[enrollment_id].append(
                {
                    **payload,
                    "manifest_path": manifest_location,
                    "manifest_line_number": line_number,
                    "resolved_audio_path": _relative_to_project(
                        Path(loaded.audio.resolved_path),
                        resolved_project_root,
                    ),
                    "loaded_duration_seconds": loaded.audio.duration_seconds,
                    "loaded_sample_rate_hz": loaded.audio.sample_rate_hz,
                    "loaded_num_channels": loaded.audio.num_channels,
                }
            )
            counts_by_dataset[_summary_label(payload.get("dataset"))] += 1
            counts_by_role[_summary_label(role)] += 1

    if not grouped_embeddings:
        raise ValueError(f"No enrollment rows were found in {manifest_location}.")

    enrollment_ids: list[str] = []
    enrollment_embeddings: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    source_speakers: set[str] = set()
    total_sample_count = 0

    for enrollment_id in sorted(grouped_embeddings):
        source_rows = grouped_rows[enrollment_id]
        speaker_ids = sorted(
            {
                speaker_id
                for speaker_id in (_coerce_string(row.get("speaker_id")) for row in source_rows)
                if speaker_id is not None
            }
        )
        if len(speaker_ids) > 1:
            joined_speakers = ", ".join(speaker_ids)
            raise ValueError(
                f"Enrollment {enrollment_id!r} mixes multiple speaker ids: {joined_speakers}"
            )
        source_speakers.update(speaker_ids or [enrollment_id])

        embeddings = np.stack(grouped_embeddings[enrollment_id], axis=0)
        pooled_embedding = average_normalized_embeddings(
            embeddings,
            field_name=f"enrollment[{enrollment_id}]",
        )
        enrollment_ids.append(enrollment_id)
        enrollment_embeddings.append(np.asarray(pooled_embedding, dtype=np.float32))
        total_sample_count += len(source_rows)

        metadata_rows.append(
            {
                "enrollment_id": enrollment_id,
                "speaker_id": speaker_ids[0] if speaker_ids else enrollment_id,
                "sample_count": len(source_rows),
                "embedding_dim": int(pooled_embedding.shape[0]),
                "compatibility_id": compatibility_id,
                "embedding_stage": normalized_stage,
                "embedding_mode": normalized_embedding_mode,
                "embedding_device": str(resolved_device),
                "source_manifest_path": manifest_location,
                "source_manifest_line_numbers": [
                    int(row["manifest_line_number"]) for row in source_rows
                ],
                "source_utterance_ids": _collect_non_empty_strings(source_rows, "utterance_id"),
                "source_audio_paths": _collect_non_empty_strings(source_rows, "audio_path"),
                "source_demo_subset_paths": _collect_non_empty_strings(
                    source_rows,
                    "demo_subset_path",
                ),
                "source_resolved_audio_paths": _collect_non_empty_strings(
                    source_rows,
                    "resolved_audio_path",
                ),
                "source_datasets": _collect_non_empty_strings(source_rows, "dataset"),
                "source_roles": _collect_non_empty_strings(source_rows, "role"),
                "source_splits": _collect_non_empty_strings(source_rows, "split"),
            }
        )

    embeddings_matrix = np.stack(enrollment_embeddings, axis=0).astype(np.float32, copy=False)

    embeddings_path = resolved_output_root / ENROLLMENT_EMBEDDINGS_NPZ_NAME
    metadata_jsonl_path = resolved_output_root / ENROLLMENT_METADATA_JSONL_NAME
    metadata_parquet_path = resolved_output_root / ENROLLMENT_METADATA_PARQUET_NAME
    summary_path = resolved_output_root / ENROLLMENT_SUMMARY_JSON_NAME

    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings_matrix,
        enrollment_ids=np.asarray(enrollment_ids, dtype=np.str_),
    )
    metadata_jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )
    pl.DataFrame(metadata_rows).write_parquet(metadata_parquet_path)

    summary = EnrollmentCacheSummary(
        format_version=ENROLLMENT_CACHE_FORMAT_VERSION,
        source_manifest_path=manifest_location,
        source_manifest_sha256=_sha256_file(resolved_manifest_path),
        model_metadata_path=model_metadata_location,
        model_metadata_sha256=_sha256_file(resolved_model_metadata_path),
        compatibility_id=compatibility_id,
        output_root=str(resolved_output_root),
        enrollment_count=len(enrollment_ids),
        source_row_count=total_sample_count,
        source_speaker_count=len(source_speakers),
        embedding_dim=int(embeddings_matrix.shape[1]),
        total_sample_count=total_sample_count,
        mean_samples_per_enrollment=round(total_sample_count / float(len(enrollment_ids)), 6),
        stage=normalized_stage,
        embedding_mode=normalized_embedding_mode,
        device=str(resolved_device),
        counts_by_dataset={key: counts_by_dataset[key] for key in sorted(counts_by_dataset)},
        counts_by_role={key: counts_by_role[key] for key in sorted(counts_by_role)},
    )
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return WrittenEnrollmentEmbeddingCache(
        output_root=str(resolved_output_root),
        embeddings_path=str(embeddings_path),
        metadata_jsonl_path=str(metadata_jsonl_path),
        metadata_parquet_path=str(metadata_parquet_path),
        summary_path=str(summary_path),
        summary=summary,
    )


def load_enrollment_embedding_cache(output_root: Path | str) -> LoadedEnrollmentEmbeddingCache:
    resolved_output_root = Path(output_root)
    summary_payload = json.loads(
        (resolved_output_root / ENROLLMENT_SUMMARY_JSON_NAME).read_text(encoding="utf-8")
    )
    if not isinstance(summary_payload, dict):
        raise ValueError("Enrollment cache summary JSON must contain an object payload.")
    summary = EnrollmentCacheSummary(**summary_payload)
    if summary.format_version != ENROLLMENT_CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported enrollment cache format version: {summary.format_version!r}")

    embeddings_matrix, enrollment_ids = _load_embedding_matrix(
        resolved_output_root / ENROLLMENT_EMBEDDINGS_NPZ_NAME,
        embeddings_key="embeddings",
        ids_key=ENROLLMENT_IDS_KEY,
    )
    metadata_rows = _align_metadata_rows(
        metadata_rows=_load_metadata_rows(resolved_output_root / ENROLLMENT_METADATA_PARQUET_NAME),
        point_id_field="enrollment_id",
        point_ids=enrollment_ids,
        expected_count=int(embeddings_matrix.shape[0]),
    )
    return LoadedEnrollmentEmbeddingCache(
        embeddings=embeddings_matrix,
        metadata_rows=[dict(row) for row in metadata_rows],
        summary=summary,
    )


def load_model_bundle_metadata(path: Path | str) -> dict[str, Any]:
    resolved_path = Path(path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model bundle metadata at {resolved_path} must contain an object.")
    return dict(payload)


def resolve_enrollment_cache_compatibility_id(model_metadata: Mapping[str, Any]) -> str:
    explicit = _coerce_string(model_metadata.get("enrollment_cache_compatibility_id"))
    if explicit is not None:
        return explicit
    canonical = json.dumps(
        dict(model_metadata),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_enrollment_cache_compatibility(
    *,
    summary: EnrollmentCacheSummary,
    model_metadata: Mapping[str, Any],
) -> None:
    expected = resolve_enrollment_cache_compatibility_id(model_metadata)
    if summary.compatibility_id != expected:
        raise ValueError(
            "Enrollment cache compatibility mismatch: "
            f"cache={summary.compatibility_id!r}, model_bundle={expected!r}."
        )


def _resolve_enrollment_id(payload: dict[str, object]) -> str:
    enrollment_id = _coerce_string(payload.get("enrollment_id"))
    if enrollment_id is not None:
        return enrollment_id
    speaker_id = _coerce_string(payload.get("speaker_id"))
    if speaker_id is not None:
        return speaker_id
    raise ValueError("Enrollment manifest rows must define enrollment_id or speaker_id.")


def _collect_non_empty_strings(
    rows: list[dict[str, Any]],
    field_name: str,
) -> list[str]:
    values = {
        value
        for value in (_coerce_string(row.get(field_name)) for row in rows)
        if value is not None
    }
    return sorted(values)


def _summary_label(value: object) -> str:
    normalized = _coerce_string(value)
    return normalized if normalized is not None else "(missing)"


def _coerce_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _load_embedding_matrix(
    path: Path,
    *,
    embeddings_key: str,
    ids_key: str,
) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        if embeddings_key not in payload:
            raise ValueError(f"Embedding archive {path} does not contain {embeddings_key!r}.")
        if ids_key not in payload:
            raise ValueError(f"Embedding archive {path} does not contain {ids_key!r}.")
        embeddings = np.asarray(payload[embeddings_key], dtype=np.float32)
        raw_ids = np.asarray(payload[ids_key]).reshape(-1)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix in {path}.")
    point_ids = [str(value) for value in raw_ids.tolist()]
    if len(point_ids) != int(embeddings.shape[0]):
        raise ValueError(
            f"Embedding id count {len(point_ids)} does not match row count {embeddings.shape[0]}."
        )
    return embeddings, point_ids


def _load_metadata_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return [dict(row) for row in pl.read_parquet(path).to_dicts()]
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSONL row in {path}:{line_number}.")
            rows.append(dict(payload))
    return rows


def _align_metadata_rows(
    *,
    metadata_rows: list[dict[str, Any]],
    point_id_field: str,
    point_ids: list[str],
    expected_count: int,
) -> list[dict[str, Any]]:
    rows_by_id = {_coerce_string(row.get(point_id_field)): row for row in metadata_rows}
    aligned_rows: list[dict[str, Any]] = []
    for point_id in point_ids:
        row = rows_by_id.get(point_id)
        if row is None:
            raise ValueError(f"Missing metadata row for {point_id_field}={point_id!r}.")
        aligned_rows.append(row)
    if len(aligned_rows) != expected_count:
        raise ValueError(
            f"Aligned metadata count {len(aligned_rows)} does not match expected {expected_count}."
        )
    return aligned_rows


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ENROLLMENT_CACHE_FORMAT_VERSION",
    "ENROLLMENT_EMBEDDINGS_NPZ_NAME",
    "ENROLLMENT_IDS_KEY",
    "ENROLLMENT_METADATA_JSONL_NAME",
    "ENROLLMENT_METADATA_PARQUET_NAME",
    "ENROLLMENT_SUMMARY_JSON_NAME",
    "MODEL_BUNDLE_METADATA_NAME",
    "EnrollmentCacheSummary",
    "LoadedEnrollmentEmbeddingCache",
    "WrittenEnrollmentEmbeddingCache",
    "build_enrollment_embedding_cache",
    "load_enrollment_embedding_cache",
    "load_model_bundle_metadata",
    "resolve_enrollment_cache_compatibility_id",
    "validate_enrollment_cache_compatibility",
]
