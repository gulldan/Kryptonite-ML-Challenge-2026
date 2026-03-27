"""Unified local/service inference wrapper for raw-audio embedding flows."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

import numpy as np

from kryptonite.config import ProjectConfig
from kryptonite.data import AudioLoadRequest, load_audio
from kryptonite.deployment import render_artifact_report, resolve_project_path

from .deployment import build_infer_artifact_report
from .enrollment_cache import (
    ENROLLMENT_SUMMARY_JSON_NAME,
    MODEL_BUNDLE_METADATA_NAME,
    load_enrollment_embedding_cache,
    load_model_bundle_metadata,
    validate_enrollment_cache_compatibility,
)
from .inference_backend import build_runtime_embedding_backend
from .runtime import (
    ServeRuntimeReport,
    build_serve_runtime_report,
    build_service_metadata,
    render_serve_runtime_report,
)
from .scoring_service import EnrollmentRecord, ScoringService


@dataclass(frozen=True, slots=True)
class EmbeddedAudio:
    audio_path: str
    resolved_audio_path: str
    duration_seconds: float
    sample_rate_hz: int
    chunk_count: int
    trim_applied: bool
    trim_reason: str
    vad_speech_detected: bool
    loudness_applied: bool
    loudness_gain_db: float
    embedding: np.ndarray

    def to_dict(self, *, include_embedding: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "audio_path": self.audio_path,
            "resolved_audio_path": self.resolved_audio_path,
            "duration_seconds": self.duration_seconds,
            "sample_rate_hz": self.sample_rate_hz,
            "chunk_count": self.chunk_count,
            "trim_applied": self.trim_applied,
            "trim_reason": self.trim_reason,
            "vad_speech_detected": self.vad_speech_detected,
            "loudness_applied": self.loudness_applied,
            "loudness_gain_db": round(float(self.loudness_gain_db), 6),
        }
        if include_embedding:
            payload["embedding"] = [round(float(value), 8) for value in self.embedding]
        return payload


@dataclass(frozen=True, slots=True)
class AudioEmbeddingBatch:
    stage: str
    items: tuple[EmbeddedAudio, ...]
    backend: dict[str, object]

    @property
    def embedding_dim(self) -> int:
        return 0 if not self.items else int(self.items[0].embedding.shape[0])

    @property
    def total_chunk_count(self) -> int:
        return sum(item.chunk_count for item in self.items)

    def to_dict(self, *, include_embeddings: bool) -> dict[str, Any]:
        return {
            "mode": "embed",
            "stage": self.stage,
            "item_count": len(self.items),
            "embedding_dim": self.embedding_dim,
            "total_chunk_count": self.total_chunk_count,
            "backend": dict(self.backend),
            "items": [item.to_dict(include_embedding=include_embeddings) for item in self.items],
        }


class Inferencer:
    """Keep one raw-audio inference contract for local code and HTTP adapters."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        runtime_report: ServeRuntimeReport,
        artifact_report: Any,
        model_metadata: Mapping[str, Any] | None,
        enrollment_cache_payload: Mapping[str, Any],
        scoring_service: ScoringService,
        embedding_backend: Any,
    ) -> None:
        self._config = config
        self._runtime_report = runtime_report
        self._artifact_report = artifact_report
        self._model_metadata = dict(model_metadata or {})
        self._enrollment_cache_payload = dict(enrollment_cache_payload)
        self._scoring_service = scoring_service
        self._embedding_backend = embedding_backend
        self._audio_request = AudioLoadRequest.from_config(
            config.normalization,
            vad=config.vad,
        )

    @classmethod
    def from_config(
        cls,
        *,
        config: ProjectConfig,
        require_artifacts: bool = False,
    ) -> Inferencer:
        runtime_report = build_serve_runtime_report(config=config)
        if not runtime_report.passed:
            raise RuntimeError(render_serve_runtime_report(runtime_report))

        artifact_report = build_infer_artifact_report(config=config, strict=require_artifacts)
        if not artifact_report.passed:
            raise RuntimeError(render_artifact_report(artifact_report))

        model_metadata_path = (
            resolve_project_path(config.paths.project_root, config.deployment.model_bundle_root)
            / MODEL_BUNDLE_METADATA_NAME
        )
        model_metadata = (
            load_model_bundle_metadata(model_metadata_path)
            if model_metadata_path.exists()
            else None
        )
        scoring_service, enrollment_cache_payload = _build_scoring_service(
            config=config,
            model_metadata=model_metadata,
        )
        embedding_backend = build_runtime_embedding_backend(
            config=config,
            model_metadata=None if model_metadata is None else dict(model_metadata),
        )
        return cls(
            config=config,
            runtime_report=runtime_report,
            artifact_report=artifact_report,
            model_metadata=model_metadata,
            enrollment_cache_payload=enrollment_cache_payload,
            scoring_service=scoring_service,
            embedding_backend=embedding_backend,
        )

    def health_payload(self) -> dict[str, Any]:
        enrollment_cache_payload = {
            **self._enrollment_cache_payload,
            "process_enrollment_count": self.list_enrollments()["enrollment_count"],
        }
        payload = build_service_metadata(
            config=self._config,
            report=self._runtime_report,
            artifact_report=self._artifact_report,
            enrollment_cache=enrollment_cache_payload,
        )
        payload["inferencer"] = self._embedding_backend.describe()
        if self._model_metadata:
            payload["model_bundle"] = {
                "loaded": True,
                "input_name": self._model_metadata.get("input_name"),
                "output_name": self._model_metadata.get("output_name"),
            }
        else:
            payload["model_bundle"] = {"loaded": False}
        return payload

    def list_enrollments(self) -> dict[str, Any]:
        return self._scoring_service.list_enrollments()

    def score_pairwise(
        self,
        *,
        left: Any,
        right: Any,
        normalize: bool = True,
    ) -> dict[str, Any]:
        return self._scoring_service.score_pairwise(left=left, right=right, normalize=normalize)

    def score_one_to_many(
        self,
        *,
        queries: Any,
        references: Any,
        normalize: bool = True,
        top_k: int | None = None,
        query_ids: list[str] | None = None,
        reference_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._scoring_service.score_one_to_many(
            queries=queries,
            references=references,
            normalize=normalize,
            top_k=top_k,
            query_ids=query_ids,
            reference_ids=reference_ids,
        )

    def enroll_embeddings(
        self,
        *,
        enrollment_id: str,
        embeddings: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._scoring_service.enroll(
            enrollment_id=enrollment_id,
            embeddings=embeddings,
            metadata=metadata,
        )

    def verify_embeddings(
        self,
        *,
        enrollment_id: str,
        probes: Any,
        normalize: bool = True,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        return self._scoring_service.verify(
            enrollment_id=enrollment_id,
            probes=probes,
            normalize=normalize,
            threshold=threshold,
        )

    def embed_audio_paths(
        self,
        *,
        audio_paths: Sequence[str],
        stage: str | None = None,
        include_embeddings: bool = True,
    ) -> dict[str, Any]:
        batch = self._embed_audio_batch(audio_paths=audio_paths, stage=stage)
        return batch.to_dict(include_embeddings=include_embeddings)

    def enroll_audio_paths(
        self,
        *,
        enrollment_id: str,
        audio_paths: Sequence[str],
        stage: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        batch = self._embed_audio_batch(audio_paths=audio_paths, stage=stage)
        response = self._scoring_service.enroll(
            enrollment_id=enrollment_id,
            embeddings=_batch_embeddings(batch),
            metadata=metadata,
        )
        response["stage"] = batch.stage
        response["backend"] = dict(batch.backend)
        response["audio_items"] = [item.to_dict(include_embedding=False) for item in batch.items]
        return response

    def verify_audio_paths(
        self,
        *,
        enrollment_id: str,
        audio_paths: Sequence[str],
        stage: str | None = None,
        normalize: bool = True,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        batch = self._embed_audio_batch(audio_paths=audio_paths, stage=stage)
        response = self._scoring_service.verify(
            enrollment_id=enrollment_id,
            probes=_batch_embeddings(batch),
            normalize=normalize,
            threshold=threshold,
        )
        response["stage"] = batch.stage
        response["backend"] = dict(batch.backend)
        response["probe_items"] = [item.to_dict(include_embedding=False) for item in batch.items]
        return response

    def benchmark_audio_paths(
        self,
        *,
        audio_paths: Sequence[str],
        stage: str | None = None,
        iterations: int = 3,
        warmup_iterations: int = 1,
    ) -> dict[str, Any]:
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative.")

        normalized_paths = _coerce_audio_paths(audio_paths)
        for _ in range(warmup_iterations):
            self._embed_audio_batch(audio_paths=normalized_paths, stage=stage)

        durations: list[float] = []
        chunk_counts: list[int] = []
        effective_stage: str | None = None
        for _ in range(iterations):
            started = time.perf_counter()
            batch = self._embed_audio_batch(audio_paths=normalized_paths, stage=stage)
            durations.append(time.perf_counter() - started)
            chunk_counts.append(batch.total_chunk_count)
            effective_stage = batch.stage

        mean_iteration_seconds = fmean(durations)
        return {
            "mode": "benchmark",
            "stage": effective_stage,
            "audio_count": len(normalized_paths),
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "mean_iteration_seconds": round(mean_iteration_seconds, 8),
            "min_iteration_seconds": round(min(durations), 8),
            "max_iteration_seconds": round(max(durations), 8),
            "mean_ms_per_audio": round(
                (mean_iteration_seconds * 1000.0) / len(normalized_paths), 8
            ),
            "mean_chunk_count": round(float(fmean(chunk_counts)), 6),
            "backend": dict(self._embedding_backend.describe()),
        }

    def _embed_audio_batch(
        self,
        *,
        audio_paths: Sequence[str],
        stage: str | None,
    ) -> AudioEmbeddingBatch:
        normalized_paths = _coerce_audio_paths(audio_paths)
        items = tuple(
            self._embed_audio_path(audio_path=audio_path, stage=stage)
            for audio_path in normalized_paths
        )
        return AudioEmbeddingBatch(
            stage=self._embedding_backend.default_stage if stage is None else stage.lower(),
            items=items,
            backend=dict(self._embedding_backend.describe()),
        )

    def _embed_audio_path(
        self,
        *,
        audio_path: str,
        stage: str | None,
    ) -> EmbeddedAudio:
        loaded = load_audio(
            audio_path,
            project_root=self._config.paths.project_root,
            request=self._audio_request,
        )
        result = self._embedding_backend.embed_waveform(
            loaded.waveform,
            sample_rate_hz=loaded.sample_rate_hz,
            stage=stage,
        )
        return EmbeddedAudio(
            audio_path=loaded.configured_path,
            resolved_audio_path=loaded.resolved_path,
            duration_seconds=loaded.duration_seconds,
            sample_rate_hz=loaded.sample_rate_hz,
            chunk_count=result.chunk_count,
            trim_applied=loaded.trim_applied,
            trim_reason=loaded.trim_reason,
            vad_speech_detected=loaded.vad_speech_detected,
            loudness_applied=loaded.loudness_applied,
            loudness_gain_db=loaded.loudness_gain_db,
            embedding=np.asarray(result.embedding, dtype=np.float32),
        )


def _batch_embeddings(batch: AudioEmbeddingBatch) -> np.ndarray:
    if not batch.items:
        raise ValueError("embed batch must not be empty")
    return np.stack([item.embedding for item in batch.items], axis=0)


def _coerce_audio_paths(audio_paths: Sequence[str]) -> list[str]:
    if not audio_paths:
        raise ValueError("audio_paths must not be empty.")
    normalized: list[str] = []
    for item in audio_paths:
        if not isinstance(item, str):
            raise ValueError("audio_paths must contain only strings.")
        stripped = item.strip()
        if not stripped:
            raise ValueError("audio_paths must not contain empty strings.")
        normalized.append(stripped)
    return normalized


def _build_scoring_service(
    *,
    config: ProjectConfig,
    model_metadata: Mapping[str, object] | None,
) -> tuple[ScoringService, dict[str, object]]:
    cache_root = resolve_project_path(
        config.paths.project_root,
        config.deployment.enrollment_cache_root,
    )
    summary_path = cache_root / ENROLLMENT_SUMMARY_JSON_NAME
    if not summary_path.exists():
        return (
            ScoringService(),
            {
                "loaded": False,
                "cache_root": str(cache_root),
                "enrollment_count": 0,
            },
        )

    if model_metadata is None:
        raise ValueError(
            "Enrollment cache exists, but model bundle metadata is missing; "
            "cannot validate compatibility."
        )

    loaded_cache = load_enrollment_embedding_cache(cache_root)
    validate_enrollment_cache_compatibility(
        summary=loaded_cache.summary,
        model_metadata=model_metadata,
    )
    initial_enrollments = {
        enrollment_id: EnrollmentRecord(
            enrollment_id=enrollment_id,
            sample_count=int(metadata_row["sample_count"]),
            embedding_dim=int(embedding.shape[0]),
            embedding=embedding,
            metadata=dict(metadata_row),
        )
        for enrollment_id, embedding, metadata_row in zip(
            _resolve_enrollment_ids(loaded_cache.metadata_rows),
            loaded_cache.embeddings,
            loaded_cache.metadata_rows,
            strict=True,
        )
    }
    return (
        ScoringService(initial_enrollments=initial_enrollments),
        {
            "loaded": True,
            "cache_root": str(cache_root),
            "enrollment_count": loaded_cache.summary.enrollment_count,
            "compatibility_id": loaded_cache.summary.compatibility_id,
            "format_version": loaded_cache.summary.format_version,
            "source_manifest_path": loaded_cache.summary.source_manifest_path,
        },
    )


def _resolve_enrollment_ids(metadata_rows: list[dict[str, object]]) -> list[str]:
    enrollment_ids: list[str] = []
    for row in metadata_rows:
        enrollment_id = row.get("enrollment_id")
        if not isinstance(enrollment_id, str) or not enrollment_id.strip():
            raise ValueError("Enrollment cache metadata rows must define non-empty enrollment_id.")
        enrollment_ids.append(enrollment_id)
    return enrollment_ids


__all__ = ["Inferencer"]
