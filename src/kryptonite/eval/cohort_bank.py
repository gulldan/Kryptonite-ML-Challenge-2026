"""Reproducible cohort-embedding bank assembly from exported embedding artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.data import normalize_manifest_entry
from kryptonite.deployment import resolve_project_path
from kryptonite.models import l2_normalize_embeddings

from .verification_data import load_verification_trial_rows, resolve_trial_side_identifier


def _load_embedding_matrix(
    path: Path | str,
    *,
    embeddings_key: str = "embeddings",
    ids_key: str | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".npy":
        embeddings = np.load(source_path)
        point_ids = None
    elif suffix == ".npz":
        payload = np.load(source_path)
        if embeddings_key not in payload:
            available = ", ".join(sorted(payload.files))
            raise ValueError(
                f"Embeddings key {embeddings_key!r} missing from {source_path}; "
                f"available: {available}"
            )
        embeddings = payload[embeddings_key]
        point_ids = (
            [str(v) for v in payload[ids_key].tolist()]
            if ids_key is not None and ids_key in payload
            else None
        )
    else:
        raise ValueError(f"Unsupported embeddings format: {source_path}")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D matrix in {source_path}, got {matrix.shape}")
    if point_ids is not None and len(point_ids) != matrix.shape[0]:
        raise ValueError(f"IDs/embeddings length mismatch in {source_path}")
    return matrix, point_ids


def _load_metadata_rows(path: Path | str) -> list[dict[str, object]]:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, object]] = []
        for line in source_path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(dict(payload))
        return rows
    if suffix == ".parquet":
        return [dict(row) for row in pl.read_parquet(source_path).iter_rows(named=True)]
    raise ValueError(f"Unsupported metadata format: {source_path}")


def _align_metadata_rows(
    *,
    metadata_rows: list[dict[str, object]],
    point_id_field: str,
    point_ids: list[str] | None,
    expected_count: int,
) -> list[dict[str, object]]:
    if point_ids is None:
        if len(metadata_rows) != expected_count:
            raise ValueError(
                f"Metadata count mismatch: expected {expected_count}, got {len(metadata_rows)}"
            )
        return metadata_rows
    indexed: dict[str, dict[str, object]] = {}
    for row in metadata_rows:
        pid = _coerce_string(row.get(point_id_field))
        if pid is None:
            raise ValueError(f"Metadata rows must define non-empty {point_id_field!r}")
        indexed[pid] = row
    missing = [pid for pid in point_ids if pid not in indexed]
    if missing:
        raise ValueError(f"Metadata missing {len(missing)} ids: {', '.join(missing[:5])}")
    return [indexed[pid] for pid in point_ids]


COHORT_BANK_FORMAT_VERSION = "kryptonite.eval.cohort_bank.v1"
COHORT_EMBEDDINGS_NPZ_NAME = "cohort_embeddings.npz"
COHORT_METADATA_JSONL_NAME = "cohort_metadata.jsonl"
COHORT_METADATA_PARQUET_NAME = "cohort_metadata.parquet"
COHORT_SUMMARY_JSON_NAME = "cohort_summary.json"


@dataclass(frozen=True, slots=True)
class CohortEmbeddingBankSelection:
    include_roles: tuple[str, ...] = ()
    include_splits: tuple[str, ...] = ()
    include_datasets: tuple[str, ...] = ()
    min_embeddings_per_speaker: int = 1
    max_embeddings_per_speaker: int | None = None
    max_embeddings: int | None = None
    trial_paths: tuple[str, ...] = ()
    validation_manifest_paths: tuple[str, ...] = ()
    strict_speaker_disjointness: bool = True
    allow_trial_overlap_fallback: bool = True
    point_id_field: str = "atlas_point_id"
    embeddings_key: str = "embeddings"
    ids_key: str | None = "point_ids"

    def __post_init__(self) -> None:
        if not self.point_id_field.strip():
            raise ValueError("point_id_field must not be empty")
        if not self.embeddings_key.strip():
            raise ValueError("embeddings_key must not be empty")
        if self.ids_key is not None and not self.ids_key.strip():
            raise ValueError("ids_key must not be empty when provided")
        if self.min_embeddings_per_speaker <= 0:
            raise ValueError("min_embeddings_per_speaker must be positive")
        if (
            self.max_embeddings_per_speaker is not None
            and self.max_embeddings_per_speaker < self.min_embeddings_per_speaker
        ):
            raise ValueError(
                "max_embeddings_per_speaker must be at least min_embeddings_per_speaker"
            )
        if self.max_embeddings is not None and self.max_embeddings <= 0:
            raise ValueError("max_embeddings must be positive when provided")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CohortEmbeddingBankSummary:
    format_version: str
    source_embeddings_path: str
    source_metadata_path: str
    output_root: str
    embedding_dim: int
    source_row_count: int
    after_metadata_filters_row_count: int
    after_trial_exclusion_row_count: int
    selected_row_count: int
    selected_speaker_count: int
    excluded_by_metadata_filters: int
    excluded_by_trial_overlap: int
    excluded_by_speaker_minimum: int
    excluded_by_per_speaker_cap: int
    excluded_by_global_cap: int
    trial_overlap_fallback_used: bool
    validation_speaker_count: int
    overlapping_validation_speakers: tuple[str, ...]
    counts_by_dataset: dict[str, int]
    counts_by_split: dict[str, int]
    counts_by_role: dict[str, int]
    source_embeddings_sha256: str
    source_metadata_sha256: str
    trial_sha256: dict[str, str]
    validation_manifest_sha256: dict[str, str]
    selection: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WrittenCohortEmbeddingBank:
    output_root: str
    embeddings_path: str
    metadata_jsonl_path: str
    metadata_parquet_path: str
    summary_path: str
    summary: CohortEmbeddingBankSummary

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
class LoadedCohortEmbeddingBank:
    embeddings: np.ndarray
    metadata_rows: list[dict[str, Any]]
    summary: CohortEmbeddingBankSummary


@dataclass(frozen=True, slots=True)
class _CandidateRow:
    point_id: str
    speaker_id: str
    dataset: str | None
    split: str | None
    role: str | None
    metadata_row: dict[str, Any]
    embedding: np.ndarray
    match_keys: frozenset[str]


def build_cohort_embedding_bank(
    *,
    project_root: Path | str,
    output_root: Path | str,
    embeddings_path: Path | str,
    metadata_path: Path | str,
    selection: CohortEmbeddingBankSelection | None = None,
) -> WrittenCohortEmbeddingBank:
    resolved_project_root = resolve_project_path(str(project_root), ".")
    resolved_output_root = resolve_project_path(str(resolved_project_root), str(output_root))
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    resolved_embeddings_path = resolve_project_path(
        str(resolved_project_root),
        str(embeddings_path),
    )
    resolved_metadata_path = resolve_project_path(
        str(resolved_project_root),
        str(metadata_path),
    )
    resolved_selection = selection or CohortEmbeddingBankSelection()

    embeddings_matrix, point_ids = _load_embedding_matrix(
        resolved_embeddings_path,
        embeddings_key=resolved_selection.embeddings_key,
        ids_key=resolved_selection.ids_key,
    )
    aligned_metadata_rows = _align_metadata_rows(
        metadata_rows=_load_metadata_rows(resolved_metadata_path),
        point_id_field=resolved_selection.point_id_field,
        point_ids=point_ids,
        expected_count=int(embeddings_matrix.shape[0]),
    )
    source_candidates = _build_source_candidates(
        metadata_rows=aligned_metadata_rows,
        embeddings_matrix=embeddings_matrix,
        point_ids=point_ids,
        point_id_field=resolved_selection.point_id_field,
    )
    filtered_candidates = _apply_metadata_filters(
        source_candidates,
        include_roles=resolved_selection.include_roles,
        include_splits=resolved_selection.include_splits,
        include_datasets=resolved_selection.include_datasets,
    )

    excluded_trial_ids, trial_sha256 = _load_trial_ids(
        project_root=resolved_project_root,
        configured_paths=resolved_selection.trial_paths,
    )
    after_trial_candidates, excluded_by_trial_overlap, fallback_used = _exclude_trial_overlap(
        filtered_candidates,
        excluded_trial_ids=excluded_trial_ids,
        allow_fallback=resolved_selection.allow_trial_overlap_fallback,
    )
    speaker_filtered_candidates, excluded_by_speaker_minimum, excluded_by_per_speaker_cap = (
        _apply_speaker_limits(
            after_trial_candidates,
            min_embeddings_per_speaker=resolved_selection.min_embeddings_per_speaker,
            max_embeddings_per_speaker=resolved_selection.max_embeddings_per_speaker,
        )
    )
    final_candidates, excluded_by_global_cap = _apply_global_cap(
        speaker_filtered_candidates,
        max_embeddings=resolved_selection.max_embeddings,
    )

    if not final_candidates:
        raise ValueError("No cohort rows remained after applying the configured selection rules.")

    validation_speakers, validation_manifest_sha256 = _load_validation_speakers(
        project_root=resolved_project_root,
        configured_paths=resolved_selection.validation_manifest_paths,
    )
    overlapping_validation_speakers = tuple(
        sorted({candidate.speaker_id for candidate in final_candidates} & validation_speakers)
    )
    if overlapping_validation_speakers and resolved_selection.strict_speaker_disjointness:
        overlap_preview = ", ".join(overlapping_validation_speakers[:10])
        raise ValueError(
            f"Cohort bank overlaps speakers from the validation manifests: {overlap_preview}"
        )

    normalized_embeddings = l2_normalize_embeddings(
        np.stack([candidate.embedding for candidate in final_candidates], axis=0),
        field_name="cohort_embeddings",
    )
    written_metadata_rows = _build_written_metadata_rows(final_candidates)

    embeddings_out = resolved_output_root / COHORT_EMBEDDINGS_NPZ_NAME
    metadata_jsonl_out = resolved_output_root / COHORT_METADATA_JSONL_NAME
    metadata_parquet_out = resolved_output_root / COHORT_METADATA_PARQUET_NAME
    summary_out = resolved_output_root / COHORT_SUMMARY_JSON_NAME

    point_id_array = np.asarray([candidate.point_id for candidate in final_candidates], dtype=str)
    speaker_id_array = np.asarray(
        [candidate.speaker_id for candidate in final_candidates],
        dtype=str,
    )
    np.savez(
        embeddings_out,
        embeddings=normalized_embeddings.astype(np.float32),
        point_ids=point_id_array,
        speaker_ids=speaker_id_array,
    )
    metadata_jsonl_out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in written_metadata_rows),
        encoding="utf-8",
    )
    pl.DataFrame(written_metadata_rows).write_parquet(metadata_parquet_out)

    summary = CohortEmbeddingBankSummary(
        format_version=COHORT_BANK_FORMAT_VERSION,
        source_embeddings_path=str(resolved_embeddings_path),
        source_metadata_path=str(resolved_metadata_path),
        output_root=str(resolved_output_root),
        embedding_dim=int(normalized_embeddings.shape[1]),
        source_row_count=len(source_candidates),
        after_metadata_filters_row_count=len(filtered_candidates),
        after_trial_exclusion_row_count=len(after_trial_candidates),
        selected_row_count=len(final_candidates),
        selected_speaker_count=len({candidate.speaker_id for candidate in final_candidates}),
        excluded_by_metadata_filters=len(source_candidates) - len(filtered_candidates),
        excluded_by_trial_overlap=excluded_by_trial_overlap,
        excluded_by_speaker_minimum=excluded_by_speaker_minimum,
        excluded_by_per_speaker_cap=excluded_by_per_speaker_cap,
        excluded_by_global_cap=excluded_by_global_cap,
        trial_overlap_fallback_used=fallback_used,
        validation_speaker_count=len(validation_speakers),
        overlapping_validation_speakers=overlapping_validation_speakers,
        counts_by_dataset=_count_metadata_values(final_candidates, field_name="dataset"),
        counts_by_split=_count_metadata_values(final_candidates, field_name="split"),
        counts_by_role=_count_metadata_values(final_candidates, field_name="role"),
        source_embeddings_sha256=_sha256_file(resolved_embeddings_path),
        source_metadata_sha256=_sha256_file(resolved_metadata_path),
        trial_sha256=trial_sha256,
        validation_manifest_sha256=validation_manifest_sha256,
        selection=resolved_selection.to_dict(),
    )
    summary_out.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return WrittenCohortEmbeddingBank(
        output_root=str(resolved_output_root),
        embeddings_path=str(embeddings_out),
        metadata_jsonl_path=str(metadata_jsonl_out),
        metadata_parquet_path=str(metadata_parquet_out),
        summary_path=str(summary_out),
        summary=summary,
    )


def load_cohort_embedding_bank(output_root: Path | str) -> LoadedCohortEmbeddingBank:
    resolved_output_root = Path(output_root)
    embeddings_matrix, _ = _load_embedding_matrix(
        resolved_output_root / COHORT_EMBEDDINGS_NPZ_NAME,
        embeddings_key="embeddings",
        ids_key="point_ids",
    )
    metadata_rows = _load_metadata_rows(resolved_output_root / COHORT_METADATA_PARQUET_NAME)
    summary_payload = json.loads((resolved_output_root / COHORT_SUMMARY_JSON_NAME).read_text())
    if not isinstance(summary_payload, dict):
        raise ValueError("Cohort summary JSON must contain an object payload.")
    return LoadedCohortEmbeddingBank(
        embeddings=embeddings_matrix,
        metadata_rows=[dict(row) for row in metadata_rows],
        summary=CohortEmbeddingBankSummary(**summary_payload),
    )


def _build_source_candidates(
    *,
    metadata_rows: list[dict[str, Any]],
    embeddings_matrix: np.ndarray,
    point_ids: list[str] | None,
    point_id_field: str,
) -> list[_CandidateRow]:
    candidates: list[_CandidateRow] = []
    for index, row in enumerate(metadata_rows):
        speaker_id = _required_string(row.get("speaker_id"), field_name="speaker_id")
        if point_ids is None:
            point_id = _required_string(row.get(point_id_field), field_name=point_id_field)
        else:
            point_id = point_ids[index]
        candidates.append(
            _CandidateRow(
                point_id=point_id,
                speaker_id=speaker_id,
                dataset=_coerce_string(row.get("dataset")),
                split=_coerce_string(row.get("split")),
                role=_coerce_string(row.get("role")),
                metadata_row=dict(row),
                embedding=np.asarray(embeddings_matrix[index], dtype=np.float64),
                match_keys=frozenset(_metadata_match_keys(row)),
            )
        )
    candidates.sort(key=_candidate_sort_key)
    return candidates


def _apply_metadata_filters(
    candidates: Sequence[_CandidateRow],
    *,
    include_roles: Sequence[str],
    include_splits: Sequence[str],
    include_datasets: Sequence[str],
) -> list[_CandidateRow]:
    normalized_roles = {value.strip().lower() for value in include_roles if value.strip()}
    normalized_splits = {value.strip().lower() for value in include_splits if value.strip()}
    normalized_datasets = {value.strip().lower() for value in include_datasets if value.strip()}

    filtered: list[_CandidateRow] = []
    for candidate in candidates:
        if normalized_roles and (candidate.role or "").lower() not in normalized_roles:
            continue
        if normalized_splits and (candidate.split or "").lower() not in normalized_splits:
            continue
        if normalized_datasets and (candidate.dataset or "").lower() not in normalized_datasets:
            continue
        filtered.append(candidate)
    return filtered


def _exclude_trial_overlap(
    candidates: Sequence[_CandidateRow],
    *,
    excluded_trial_ids: set[str],
    allow_fallback: bool,
) -> tuple[list[_CandidateRow], int, bool]:
    if not excluded_trial_ids:
        return list(candidates), 0, False

    filtered = [
        candidate for candidate in candidates if candidate.match_keys.isdisjoint(excluded_trial_ids)
    ]
    if filtered:
        return filtered, len(candidates) - len(filtered), False
    if allow_fallback:
        return list(candidates), 0, True
    raise ValueError("Trial-overlap exclusions removed every available cohort row.")


def _apply_speaker_limits(
    candidates: Sequence[_CandidateRow],
    *,
    min_embeddings_per_speaker: int,
    max_embeddings_per_speaker: int | None,
) -> tuple[list[_CandidateRow], int, int]:
    grouped: dict[str, list[_CandidateRow]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.speaker_id].append(candidate)

    kept: list[_CandidateRow] = []
    excluded_by_minimum = 0
    excluded_by_cap = 0
    for speaker_id in sorted(grouped):
        rows = grouped[speaker_id]
        if len(rows) < min_embeddings_per_speaker:
            excluded_by_minimum += len(rows)
            continue
        if max_embeddings_per_speaker is not None and len(rows) > max_embeddings_per_speaker:
            excluded_by_cap += len(rows) - max_embeddings_per_speaker
            rows = rows[:max_embeddings_per_speaker]
        kept.extend(rows)
    return kept, excluded_by_minimum, excluded_by_cap


def _apply_global_cap(
    candidates: Sequence[_CandidateRow],
    *,
    max_embeddings: int | None,
) -> tuple[list[_CandidateRow], int]:
    if max_embeddings is None or len(candidates) <= max_embeddings:
        return list(candidates), 0

    grouped: dict[str, deque[_CandidateRow]] = defaultdict(deque)
    for candidate in candidates:
        grouped[candidate.speaker_id].append(candidate)

    selected: list[_CandidateRow] = []
    speaker_order = deque(sorted(grouped))
    while speaker_order and len(selected) < max_embeddings:
        speaker_id = speaker_order.popleft()
        rows = grouped[speaker_id]
        if not rows:
            continue
        selected.append(rows.popleft())
        if rows and len(selected) < max_embeddings:
            speaker_order.append(speaker_id)
    return selected, len(candidates) - len(selected)


def _load_trial_ids(
    *,
    project_root: Path,
    configured_paths: Sequence[str],
) -> tuple[set[str], dict[str, str]]:
    excluded_ids: set[str] = set()
    sha256: dict[str, str] = {}
    for configured_path in configured_paths:
        resolved_path = resolve_project_path(str(project_root), configured_path)
        sha256[str(resolved_path)] = _sha256_file(resolved_path)
        for row in load_verification_trial_rows(resolved_path):
            for side in ("left", "right"):
                identifier = resolve_trial_side_identifier(row, side)
                if identifier is not None:
                    excluded_ids.add(identifier)
    return excluded_ids, sha256


def _load_validation_speakers(
    *,
    project_root: Path,
    configured_paths: Sequence[str],
) -> tuple[set[str], dict[str, str]]:
    speakers: set[str] = set()
    sha256: dict[str, str] = {}
    for configured_path in configured_paths:
        resolved_path = resolve_project_path(str(project_root), configured_path)
        sha256[str(resolved_path)] = _sha256_file(resolved_path)
        for row in _load_jsonl_objects(resolved_path):
            normalized = normalize_manifest_entry(row)
            speaker_id = _coerce_string(normalized.get("speaker_id"))
            if speaker_id is not None:
                speakers.add(speaker_id)
    return speakers, sha256


def _build_written_metadata_rows(candidates: Sequence[_CandidateRow]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        row = dict(candidate.metadata_row)
        row["cohort_point_id"] = candidate.point_id
        row["cohort_row_index"] = index
        rows.append(row)
    return rows


def _count_metadata_values(
    candidates: Sequence[_CandidateRow],
    *,
    field_name: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        counts[_summary_label(candidate.metadata_row.get(field_name))] += 1
    return {key: counts[key] for key in sorted(counts)}


def _metadata_match_keys(row: dict[str, Any]) -> Iterable[str]:
    for field_name in ("trial_item_id", "utterance_id", "audio_path"):
        value = _coerce_string(row.get(field_name))
        if value is None:
            continue
        yield value
        if field_name == "audio_path":
            yield Path(value).name


def _candidate_sort_key(candidate: _CandidateRow) -> tuple[str, str, str]:
    trial_item_id = _coerce_string(candidate.metadata_row.get("trial_item_id")) or ""
    utterance_id = _coerce_string(candidate.metadata_row.get("utterance_id")) or ""
    audio_path = _coerce_string(candidate.metadata_row.get("audio_path")) or ""
    return candidate.speaker_id, trial_item_id or utterance_id or audio_path, candidate.point_id


def _summary_label(value: object) -> str:
    normalized = _coerce_string(value)
    return normalized if normalized is not None else "(missing)"


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object JSONL rows in {path}:{line_number}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _required_string(value: object, *, field_name: str) -> str:
    normalized = _coerce_string(value)
    if normalized is None:
        raise ValueError(f"Metadata rows must define a non-empty {field_name!r}.")
    return normalized


def _coerce_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "COHORT_BANK_FORMAT_VERSION",
    "COHORT_EMBEDDINGS_NPZ_NAME",
    "COHORT_METADATA_JSONL_NAME",
    "COHORT_METADATA_PARQUET_NAME",
    "COHORT_SUMMARY_JSON_NAME",
    "CohortEmbeddingBankSelection",
    "CohortEmbeddingBankSummary",
    "LoadedCohortEmbeddingBank",
    "WrittenCohortEmbeddingBank",
    "build_cohort_embedding_bank",
    "load_cohort_embedding_bank",
]
