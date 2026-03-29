"""Manifest-backed catalog loading for augmentation scheduling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

from kryptonite.config import SilenceAugmentationConfig
from kryptonite.data.silence_policy import (
    build_scaled_silence_profile as _scaled_silence_profile,
    has_effective_silence_profile as _has_effective_silence_profile,
)

from .models import (
    AugmentationCandidate,
    AugmentationCatalog,
    AugmentationFamily,
    BankManifestPaths,
)

_SEVERITY_ORDER = ("light", "medium", "heavy")


def resolve_bank_manifest_paths(
    *,
    project_root: Path | str,
    noise_manifest_path: Path | str | None = None,
    room_config_manifest_path: Path | str | None = None,
    distance_manifest_path: Path | str | None = None,
    codec_manifest_path: Path | str | None = None,
) -> BankManifestPaths:
    project_root_path = Path(project_root)
    return BankManifestPaths(
        noise_manifest_path=_resolve_explicit_or_first_existing(
            explicit=noise_manifest_path,
            candidates=(
                project_root_path
                / "artifacts/corruptions/noise-bank/manifests/noise_bank_manifest.jsonl",
            ),
        ),
        room_config_manifest_path=_resolve_explicit_or_first_existing(
            explicit=room_config_manifest_path,
            candidates=_unique_paths(
                (
                    project_root_path
                    / "artifacts/corruptions/rir-bank/manifests/room_simulation_configs.jsonl",
                    *sorted(
                        project_root_path.glob(
                            "artifacts/corruptions/rir-bank*/manifests/room_simulation_configs.jsonl"
                        )
                    ),
                )
            ),
        ),
        distance_manifest_path=_resolve_explicit_or_first_existing(
            explicit=distance_manifest_path,
            candidates=(
                project_root_path
                / "artifacts/corruptions/far-field-bank/manifests/far_field_bank_manifest.jsonl",
            ),
        ),
        codec_manifest_path=_resolve_explicit_or_first_existing(
            explicit=codec_manifest_path,
            candidates=(
                project_root_path
                / "artifacts/corruptions/codec-bank/manifests/codec_bank_manifest.jsonl",
            ),
        ),
    )


def load_augmentation_catalog(
    *,
    manifest_paths: BankManifestPaths,
    silence_config: SilenceAugmentationConfig,
) -> AugmentationCatalog:
    candidates_by_family: dict[AugmentationFamily, tuple[AugmentationCandidate, ...]] = {
        "noise": tuple(_load_noise_candidates(manifest_paths.noise_manifest_path)),
        "reverb": tuple(_load_reverb_candidates(manifest_paths.room_config_manifest_path)),
        "distance": tuple(_load_distance_candidates(manifest_paths.distance_manifest_path)),
        "codec": tuple(_load_codec_candidates(manifest_paths.codec_manifest_path)),
        "silence": tuple(_load_silence_candidates(silence_config)),
    }
    return AugmentationCatalog(candidates_by_family=candidates_by_family)


def _load_noise_candidates(path: str | None) -> list[AugmentationCandidate]:
    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl(path):
        severity = _coerce_severity(record.get("severity"))
        if severity is None:
            continue
        noise_id = _coerce_str(record.get("noise_id"))
        if noise_id is None:
            continue
        category = _coerce_str(record.get("category")) or "unknown"
        label = f"noise/{category}/{severity}"
        candidates.append(
            AugmentationCandidate(
                family="noise",
                candidate_id=noise_id,
                label=label,
                severity=severity,
                weight=_coerce_weight(record.get("sampling_weight")),
                tags=_coerce_tags(record.get("tags")),
                metadata={
                    "category": category,
                    "mix_mode": _coerce_str(record.get("mix_mode")),
                    "recommended_snr_db_min": record.get("recommended_snr_db_min"),
                    "recommended_snr_db_max": record.get("recommended_snr_db_max"),
                },
            )
        )
    return candidates


def _load_reverb_candidates(path: str | None) -> list[AugmentationCandidate]:
    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl(path):
        config_id = _coerce_str(record.get("config_id"))
        direct_condition = _coerce_str(record.get("direct_condition"))
        if config_id is None or direct_condition is None:
            continue
        severity = _severity_from_direct_condition(direct_condition)
        room_size = _coerce_str(record.get("room_size")) or "unknown"
        field = _coerce_str(record.get("field")) or "unknown"
        rt60_bucket = _coerce_str(record.get("rt60_bucket")) or "unknown"
        rir_count = _coerce_int(record.get("rir_count"), default=1)
        label = f"reverb/{room_size}/{field}/{direct_condition}"
        candidates.append(
            AugmentationCandidate(
                family="reverb",
                candidate_id=config_id,
                label=label,
                severity=severity,
                weight=max(1.0, float(rir_count)),
                metadata={
                    "room_size": room_size,
                    "field": field,
                    "rt60_bucket": rt60_bucket,
                    "direct_condition": direct_condition,
                    "rir_count": rir_count,
                },
            )
        )
    return candidates


def _load_distance_candidates(path: str | None) -> list[AugmentationCandidate]:
    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl(path):
        preset_id = _coerce_str(record.get("preset_id"))
        field = _coerce_str(record.get("field"))
        if preset_id is None or field is None:
            continue
        severity = _severity_from_distance_field(field)
        label = f"distance/{field}/{preset_id}"
        candidates.append(
            AugmentationCandidate(
                family="distance",
                candidate_id=preset_id,
                label=label,
                severity=severity,
                weight=_coerce_weight(record.get("sampling_weight")),
                tags=_coerce_tags(record.get("tags")),
                metadata={
                    "field": field,
                    "distance_meters": record.get("distance_meters"),
                    "target_drr_db": record.get("target_drr_db"),
                },
            )
        )
    return candidates


def _load_codec_candidates(path: str | None) -> list[AugmentationCandidate]:
    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl(path):
        preset_id = _coerce_str(record.get("preset_id"))
        severity = _coerce_severity(record.get("severity"))
        if preset_id is None or severity is None:
            continue
        family = _coerce_str(record.get("family")) or "unknown"
        label = f"codec/{family}/{severity}"
        candidates.append(
            AugmentationCandidate(
                family="codec",
                candidate_id=preset_id,
                label=label,
                severity=severity,
                weight=_coerce_weight(record.get("sampling_weight")),
                tags=_coerce_tags(record.get("tags")),
                metadata={
                    "codec_family": family,
                    "ffmpeg_encode_codec": _coerce_str(record.get("ffmpeg_encode_codec")),
                },
            )
        )
    return candidates


def _load_silence_candidates(
    silence_config: SilenceAugmentationConfig,
) -> list[AugmentationCandidate]:
    if not _has_effective_silence_profile(silence_config):
        return []

    candidates: list[AugmentationCandidate] = []
    for severity, scale in (("light", 0.5), ("medium", 0.8), ("heavy", 1.0)):
        candidate_id = f"silence-{severity}"
        label = f"silence/{severity}"
        scaled = _scaled_silence_profile(silence_config, scale=scale)
        candidates.append(
            AugmentationCandidate(
                family="silence",
                candidate_id=candidate_id,
                label=label,
                severity=severity,
                weight=1.0,
                metadata=scaled,
            )
        )
    return candidates


def _read_jsonl(path: str | None) -> list[dict[str, object]]:
    if path is None:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, object]] = []
    for raw_line in file_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _resolve_explicit_or_first_existing(
    *,
    explicit: Path | str | None,
    candidates: tuple[Path, ...],
) -> str | None:
    if explicit is not None:
        return str(Path(explicit))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return tuple(unique)


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _coerce_tags(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _coerce_weight(value: object) -> float:
    if isinstance(value, int | float) and value > 0:
        return float(value)
    return 1.0


def _coerce_int(value: object, *, default: int) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0:
        return int(value)
    return default


def _coerce_severity(value: object) -> Literal["light", "medium", "heavy"] | None:
    if isinstance(value, str) and value in _SEVERITY_ORDER:
        return cast(Literal["light", "medium", "heavy"], value)
    return None


def _severity_from_direct_condition(value: str) -> Literal["light", "medium", "heavy"]:
    return cast(
        Literal["light", "medium", "heavy"],
        {
            "high": "light",
            "medium": "medium",
            "low": "heavy",
        }.get(value, "medium"),
    )


def _severity_from_distance_field(value: str) -> Literal["light", "medium", "heavy"]:
    return cast(
        Literal["light", "medium", "heavy"],
        {
            "near": "light",
            "mid": "medium",
            "far": "heavy",
        }.get(value, "medium"),
    )
