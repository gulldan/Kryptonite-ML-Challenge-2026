"""Shared raw-audio embedding backends for local and HTTP inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kryptonite.config import ProjectConfig
from kryptonite.features import (
    FbankExtractionRequest,
    FbankExtractor,
    UtteranceChunkingRequest,
    chunk_utterance,
    pool_chunk_tensors,
)
from kryptonite.features.chunking import SUPPORTED_CHUNKING_STAGES

SUPPORTED_INFERENCER_BACKENDS = frozenset({"feature_statistics"})
SUPPORTED_RUNTIME_EMBEDDING_MODES = frozenset({"mean", "mean_std"})


@dataclass(frozen=True, slots=True)
class RuntimeEmbeddingResult:
    embedding: np.ndarray
    chunk_count: int


class FeatureStatisticsEmbeddingBackend:
    """Embed waveform chunks via shared Fbank extraction and pooled feature statistics."""

    def __init__(
        self,
        *,
        feature_request: FbankExtractionRequest,
        chunking_request: UtteranceChunkingRequest,
        embedding_mode: str = "mean_std",
        default_stage: str = "demo",
        device: str = "auto",
    ) -> None:
        normalized_mode = normalize_runtime_embedding_mode(embedding_mode)
        normalized_stage = normalize_inference_stage(default_stage)

        self._torch = _import_torch()
        self._feature_request = feature_request
        self._chunking_request = chunking_request
        self._extractor = FbankExtractor(request=feature_request)
        self._embedding_mode = normalized_mode
        self._default_stage = normalized_stage
        self._device = _resolve_torch_device(torch=self._torch, preference=device)

    @property
    def implementation(self) -> str:
        return "feature_statistics"

    @property
    def default_stage(self) -> str:
        return self._default_stage

    @property
    def embedding_mode(self) -> str:
        return self._embedding_mode

    @property
    def embedding_dim(self) -> int:
        base_dim = self._feature_request.num_mel_bins
        return base_dim if self._embedding_mode == "mean" else base_dim * 2

    @property
    def device(self) -> str:
        return str(self._device)

    def describe(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "embedding_mode": self.embedding_mode,
            "default_stage": self.default_stage,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
        }

    def embed_waveform(
        self,
        waveform: Any,
        *,
        sample_rate_hz: int,
        stage: str | None = None,
    ) -> RuntimeEmbeddingResult:
        normalized_stage = (
            self._default_stage if stage is None else normalize_inference_stage(stage)
        )
        torch = self._torch
        waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32, device=self._device)
        chunk_batch = chunk_utterance(
            waveform_tensor,
            sample_rate_hz=sample_rate_hz,
            stage=normalized_stage,
            request=self._chunking_request,
        )
        with torch.inference_mode():
            chunk_embeddings = [
                pool_feature_frames(
                    torch=torch,
                    features=self._extractor.extract(
                        chunk.waveform,
                        sample_rate_hz=sample_rate_hz,
                    ),
                    embedding_mode=self._embedding_mode,
                )
                for chunk in chunk_batch.chunks
            ]
            pooled = pool_chunk_tensors(
                chunk_embeddings,
                pooling_mode=chunk_batch.pooling_mode,
            )
        return RuntimeEmbeddingResult(
            embedding=pooled.to(device="cpu", dtype=torch.float32).numpy(),
            chunk_count=len(chunk_batch.chunks),
        )


def build_runtime_embedding_backend(
    *,
    config: ProjectConfig,
    model_metadata: dict[str, Any] | None = None,
) -> FeatureStatisticsEmbeddingBackend:
    metadata = dict(model_metadata or {})
    implementation = _coerce_string(metadata.get("inferencer_backend")) or "feature_statistics"
    if implementation not in SUPPORTED_INFERENCER_BACKENDS:
        raise ValueError(
            "Unsupported inferencer backend "
            f"{implementation!r}; expected one of {sorted(SUPPORTED_INFERENCER_BACKENDS)}."
        )
    return FeatureStatisticsEmbeddingBackend(
        feature_request=FbankExtractionRequest.from_config(config.features),
        chunking_request=UtteranceChunkingRequest.from_config(config.chunking),
        embedding_mode=_coerce_string(metadata.get("embedding_mode")) or "mean_std",
        default_stage=_coerce_string(metadata.get("embedding_stage")) or "demo",
        device=config.runtime.device,
    )


def normalize_inference_stage(stage: str) -> str:
    normalized = stage.lower()
    if normalized not in SUPPORTED_CHUNKING_STAGES:
        raise ValueError(f"stage must be one of {sorted(SUPPORTED_CHUNKING_STAGES)}")
    return normalized


def normalize_runtime_embedding_mode(embedding_mode: str) -> str:
    normalized = embedding_mode.lower()
    if normalized not in SUPPORTED_RUNTIME_EMBEDDING_MODES:
        raise ValueError(
            f"embedding_mode must be one of {sorted(SUPPORTED_RUNTIME_EMBEDDING_MODES)}"
        )
    return normalized


def pool_feature_frames(
    *,
    torch: Any,
    features: Any,
    embedding_mode: str,
) -> Any:
    normalized_mode = normalize_runtime_embedding_mode(embedding_mode)
    if features.ndim != 2 or int(features.shape[0]) == 0:
        raise ValueError("Expected non-empty [frames, features] tensors from the Fbank extractor.")

    feature_matrix = features.to(dtype=torch.float32)
    mean = feature_matrix.mean(dim=0)
    if normalized_mode == "mean":
        return mean
    std = feature_matrix.std(dim=0, correction=0)
    return torch.cat((mean, std), dim=0)


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
    except ImportError as exc:  # pragma: no cover - depends on environment setup
        raise RuntimeError(
            "The unified inferencer requires torch for the current "
            "'feature_statistics' backend. Sync the repository environment with "
            "`uv sync --dev --group train`."
        ) from exc
    return torch


def _coerce_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "FeatureStatisticsEmbeddingBackend",
    "RuntimeEmbeddingResult",
    "SUPPORTED_INFERENCER_BACKENDS",
    "SUPPORTED_RUNTIME_EMBEDDING_MODES",
    "build_runtime_embedding_backend",
    "normalize_inference_stage",
    "normalize_runtime_embedding_mode",
    "pool_feature_frames",
]
