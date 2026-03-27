"""Persistent runtime enrollment storage for demo/service enroll calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .enrollment_cache import resolve_enrollment_cache_compatibility_id

if TYPE_CHECKING:
    from collections.abc import Mapping

RUNTIME_ENROLLMENT_STORE_FORMAT_VERSION = "kryptonite.serve.runtime_enrollment_store.v1"
RUNTIME_ENROLLMENT_STORE_DB_NAME = "runtime_enrollments.sqlite3"


@dataclass(frozen=True, slots=True)
class RuntimeEnrollmentStoreRecord:
    enrollment_id: str
    sample_count: int
    embedding_dim: int
    embedding: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeEnrollmentStoreSummary:
    format_version: str
    store_path: str
    compatibility_id: str
    model_metadata_path: str
    model_metadata_sha256: str
    enrollment_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RuntimeEnrollmentStore:
    def __init__(
        self,
        *,
        store_root: Path | str,
        model_metadata_path: Path | str,
        model_metadata_location: str,
        model_metadata: Mapping[str, Any],
    ) -> None:
        self._store_root = Path(store_root)
        self._store_root.mkdir(parents=True, exist_ok=True)
        self._path = self._store_root / RUNTIME_ENROLLMENT_STORE_DB_NAME
        self._compatibility_id = resolve_enrollment_cache_compatibility_id(model_metadata)
        self._model_metadata_path = model_metadata_location
        self._model_metadata_sha256 = _sha256_file(Path(model_metadata_path))
        self._initialize()

    @property
    def path(self) -> Path:
        return self._path

    def load_records(self) -> dict[str, RuntimeEnrollmentStoreRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT enrollment_id, sample_count, embedding_dim, embedding, metadata_json
                FROM enrollments
                ORDER BY enrollment_id
                """
            ).fetchall()

        records: dict[str, RuntimeEnrollmentStoreRecord] = {}
        for row in rows:
            metadata_payload = json.loads(str(row["metadata_json"]))
            if not isinstance(metadata_payload, dict):
                raise ValueError("Runtime enrollment metadata must be stored as a JSON object.")

            raw_embedding = row["embedding"]
            if not isinstance(raw_embedding, bytes):
                raise ValueError("Runtime enrollment embedding payload must be stored as bytes.")
            embedding = np.frombuffer(raw_embedding, dtype=np.float32).copy()
            embedding_dim = int(row["embedding_dim"])
            if int(embedding.shape[0]) != embedding_dim:
                raise ValueError(
                    "Runtime enrollment embedding payload size does not match embedding_dim."
                )

            enrollment_id = str(row["enrollment_id"])
            records[enrollment_id] = RuntimeEnrollmentStoreRecord(
                enrollment_id=enrollment_id,
                sample_count=int(row["sample_count"]),
                embedding_dim=embedding_dim,
                embedding=embedding,
                metadata=dict(metadata_payload),
            )
        return records

    def upsert(
        self,
        *,
        enrollment_id: str,
        sample_count: int,
        embedding_dim: int,
        embedding: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> None:
        normalized_id = enrollment_id.strip()
        if not normalized_id:
            raise ValueError("enrollment_id must not be empty.")
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Runtime enrollment embeddings must be stored as 1-D vectors.")
        if int(vector.shape[0]) != int(embedding_dim):
            raise ValueError("embedding_dim does not match the provided runtime enrollment vector.")

        try:
            metadata_json = json.dumps(dict(metadata), sort_keys=True)
        except TypeError as exc:
            raise ValueError("metadata must be JSON-serializable.") from exc

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO enrollments (
                    enrollment_id,
                    sample_count,
                    embedding_dim,
                    embedding,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(enrollment_id) DO UPDATE SET
                    sample_count = excluded.sample_count,
                    embedding_dim = excluded.embedding_dim,
                    embedding = excluded.embedding,
                    metadata_json = excluded.metadata_json
                """,
                (
                    normalized_id,
                    int(sample_count),
                    int(embedding_dim),
                    sqlite3.Binary(np.ascontiguousarray(vector).tobytes()),
                    metadata_json,
                ),
            )
            connection.commit()

    def summary(self) -> RuntimeEnrollmentStoreSummary:
        with self._connect() as connection:
            enrollment_count = int(
                connection.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0]
            )
        return RuntimeEnrollmentStoreSummary(
            format_version=RUNTIME_ENROLLMENT_STORE_FORMAT_VERSION,
            store_path=str(self._path),
            compatibility_id=self._compatibility_id,
            model_metadata_path=self._model_metadata_path,
            model_metadata_sha256=self._model_metadata_sha256,
            enrollment_count=enrollment_count,
        )

    def describe(self) -> dict[str, object]:
        summary = self.summary()
        return {
            "enabled": True,
            "loaded": summary.enrollment_count > 0,
            **summary.to_dict(),
        }

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS enrollments (
                    enrollment_id TEXT PRIMARY KEY,
                    sample_count INTEGER NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

            existing = _load_store_metadata(connection)
            if existing:
                _validate_store_metadata(
                    existing,
                    expected_format_version=RUNTIME_ENROLLMENT_STORE_FORMAT_VERSION,
                    expected_compatibility_id=self._compatibility_id,
                    expected_model_metadata_sha256=self._model_metadata_sha256,
                )
            else:
                for key, value in (
                    ("format_version", RUNTIME_ENROLLMENT_STORE_FORMAT_VERSION),
                    ("compatibility_id", self._compatibility_id),
                    ("model_metadata_path", self._model_metadata_path),
                    ("model_metadata_sha256", self._model_metadata_sha256),
                ):
                    connection.execute(
                        "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
                        (key, value),
                    )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection


def _load_store_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT key, value FROM store_metadata").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def _validate_store_metadata(
    metadata: Mapping[str, str],
    *,
    expected_format_version: str,
    expected_compatibility_id: str,
    expected_model_metadata_sha256: str,
) -> None:
    format_version = metadata.get("format_version")
    if format_version != expected_format_version:
        raise ValueError(
            "Runtime enrollment store format mismatch: "
            f"store={format_version!r}, expected={expected_format_version!r}."
        )

    compatibility_id = metadata.get("compatibility_id")
    if compatibility_id != expected_compatibility_id:
        raise ValueError(
            "Runtime enrollment store compatibility mismatch: "
            f"store={compatibility_id!r}, model_bundle={expected_compatibility_id!r}."
        )

    metadata_sha256 = metadata.get("model_metadata_sha256")
    if metadata_sha256 != expected_model_metadata_sha256:
        raise ValueError(
            "Runtime enrollment store metadata hash mismatch: "
            f"store={metadata_sha256!r}, model_bundle={expected_model_metadata_sha256!r}."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "RUNTIME_ENROLLMENT_STORE_DB_NAME",
    "RUNTIME_ENROLLMENT_STORE_FORMAT_VERSION",
    "RuntimeEnrollmentStore",
    "RuntimeEnrollmentStoreRecord",
    "RuntimeEnrollmentStoreSummary",
]
