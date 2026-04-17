"""Pydantic request models for the FastAPI serving adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type EmbeddingVector = list[float]
type EmbeddingMatrix = list[EmbeddingVector]
type EmbeddingPayload = EmbeddingVector | EmbeddingMatrix


class APIRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbedAudioRequest(APIRequestModel):
    audio_path: str | None = None
    audio_paths: list[str] | None = None
    stage: str | None = None

    @field_validator("audio_path", "stage")
    @classmethod
    def _validate_optional_string(cls, value: str | None) -> str | None:
        return _normalize_optional_string(value)

    @field_validator("audio_paths")
    @classmethod
    def _validate_audio_paths(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_optional_string_list(value)

    @model_validator(mode="after")
    def _validate_payload(self) -> EmbedAudioRequest:
        _ensure_one_of(
            self.audio_path, self.audio_paths, singular_key="audio_path", plural_key="audio_paths"
        )
        return self

    def resolve_audio_paths(self) -> list[str]:
        return _resolve_string_paths(self.audio_path, self.audio_paths)


class BenchmarkAudioRequest(EmbedAudioRequest):
    iterations: int = Field(default=3, ge=1)
    warmup_iterations: int = Field(default=1, ge=0)


class PairwiseScoringRequest(APIRequestModel):
    left: EmbeddingPayload
    right: EmbeddingPayload
    normalize: bool = True


class OneToManyScoringRequest(APIRequestModel):
    queries: EmbeddingPayload
    references: EmbeddingPayload
    normalize: bool = True
    top_k: int | None = Field(default=None, ge=1)
    query_ids: list[str] | None = None
    reference_ids: list[str] | None = None

    @field_validator("query_ids", "reference_ids")
    @classmethod
    def _validate_identifier_list(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_optional_string_list(value)


class EnrollmentRequest(APIRequestModel):
    enrollment_id: str
    embedding: EmbeddingPayload | None = None
    embeddings: EmbeddingPayload | None = None
    audio_path: str | None = None
    audio_paths: list[str] | None = None
    stage: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("enrollment_id", "audio_path", "stage")
    @classmethod
    def _validate_optional_string(cls, value: str | None) -> str | None:
        return _normalize_optional_string(value)

    @field_validator("audio_paths")
    @classmethod
    def _validate_audio_paths(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_optional_string_list(value)

    @model_validator(mode="after")
    def _validate_payload(self) -> EnrollmentRequest:
        has_embeddings = self.embedding is not None or self.embeddings is not None
        has_audio_paths = self.audio_path is not None or self.audio_paths is not None
        if has_embeddings == has_audio_paths:
            raise ValueError(
                "Provide either embedding/embeddings or audio_path/audio_paths for enrollment."
            )
        return self

    @property
    def uses_audio_paths(self) -> bool:
        return self.audio_path is not None or self.audio_paths is not None

    def resolve_audio_paths(self) -> list[str]:
        return _resolve_string_paths(self.audio_path, self.audio_paths)

    def resolve_embeddings(self) -> EmbeddingPayload:
        return _resolve_single_or_many(
            singular=self.embedding,
            plural=self.embeddings,
            singular_key="embedding",
            plural_key="embeddings",
        )


class VerifyRequest(APIRequestModel):
    enrollment_id: str
    probe: EmbeddingPayload | None = None
    probes: EmbeddingPayload | None = None
    audio_path: str | None = None
    audio_paths: list[str] | None = None
    stage: str | None = None
    normalize: bool = True
    threshold: float | None = None

    @field_validator("enrollment_id", "audio_path", "stage")
    @classmethod
    def _validate_optional_string(cls, value: str | None) -> str | None:
        return _normalize_optional_string(value)

    @field_validator("audio_paths")
    @classmethod
    def _validate_audio_paths(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_optional_string_list(value)

    @model_validator(mode="after")
    def _validate_payload(self) -> VerifyRequest:
        has_probes = self.probe is not None or self.probes is not None
        has_audio_paths = self.audio_path is not None or self.audio_paths is not None
        if has_probes == has_audio_paths:
            raise ValueError(
                "Provide either probe/probes or audio_path/audio_paths for verification."
            )
        return self

    @property
    def uses_audio_paths(self) -> bool:
        return self.audio_path is not None or self.audio_paths is not None

    def resolve_audio_paths(self) -> list[str]:
        return _resolve_string_paths(self.audio_path, self.audio_paths)

    def resolve_probes(self) -> EmbeddingPayload:
        return _resolve_single_or_many(
            singular=self.probe,
            plural=self.probes,
            singular_key="probe",
            plural_key="probes",
        )


class DemoAudioUpload(APIRequestModel):
    filename: str
    content_base64: str

    @field_validator("filename", "content_base64")
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        return _normalize_required_string(value)

    @field_validator("filename")
    @classmethod
    def _validate_audio_filename(cls, value: str) -> str:
        suffix = value.rsplit(".", 1)
        if len(suffix) != 2 or f".{suffix[1].lower()}" not in {".wav", ".flac", ".mp3"}:
            raise ValueError("Demo uploads must use one of: .wav, .flac, .mp3.")
        return value


class DemoCompareRequest(APIRequestModel):
    left_audio: DemoAudioUpload
    right_audio: DemoAudioUpload
    stage: str | None = "demo"
    normalize: bool = True
    threshold: float | None = None

    @field_validator("stage")
    @classmethod
    def _validate_optional_string(cls, value: str | None) -> str | None:
        return _normalize_optional_string(value)


class DemoEnrollmentRequest(APIRequestModel):
    enrollment_id: str
    audio_files: list[DemoAudioUpload] = Field(min_length=1)
    stage: str | None = "demo"
    metadata: dict[str, Any] | None = None

    @field_validator("enrollment_id", "stage")
    @classmethod
    def _validate_optional_string(cls, value: str | None) -> str | None:
        return _normalize_optional_string(value)


class DemoVerifyRequest(APIRequestModel):
    enrollment_id: str
    audio_file: DemoAudioUpload
    stage: str | None = "demo"
    normalize: bool = True
    threshold: float | None = None

    @field_validator("enrollment_id", "stage")
    @classmethod
    def _validate_optional_string(cls, value: str | None) -> str | None:
        return _normalize_optional_string(value)


def _ensure_one_of(
    singular: object | None,
    plural: object | None,
    *,
    singular_key: str,
    plural_key: str,
) -> None:
    if singular is None and plural is None:
        raise ValueError(f"Either {singular_key} or {plural_key} is required.")
    if singular is not None and plural is not None:
        raise ValueError(f"Provide only one of {singular_key} or {plural_key}.")


def _resolve_single_or_many(
    *,
    singular: Any,
    plural: Any,
    singular_key: str,
    plural_key: str,
) -> Any:
    _ensure_one_of(singular, plural, singular_key=singular_key, plural_key=plural_key)
    return singular if singular is not None else plural


def _resolve_string_paths(single: str | None, multiple: list[str] | None) -> list[str]:
    resolved = _resolve_single_or_many(
        singular=single,
        plural=multiple,
        singular_key="audio_path",
        plural_key="audio_paths",
    )
    if isinstance(resolved, str):
        return [resolved]
    return resolved


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    return _normalize_required_string(value)


def _normalize_required_string(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("String fields must not be empty.")
    return stripped


def _normalize_optional_string_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    normalized: list[str] = []
    for value in values:
        stripped = _normalize_optional_string(value)
        if stripped is None:
            continue
        normalized.append(stripped)
    if not normalized:
        raise ValueError("List fields must not be empty.")
    return normalized


__all__ = [
    "BenchmarkAudioRequest",
    "DemoAudioUpload",
    "DemoCompareRequest",
    "DemoEnrollmentRequest",
    "DemoVerifyRequest",
    "EmbedAudioRequest",
    "EmbeddingPayload",
    "EnrollmentRequest",
    "OneToManyScoringRequest",
    "PairwiseScoringRequest",
    "VerifyRequest",
]
