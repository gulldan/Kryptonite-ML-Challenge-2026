"""Datamodels for reproducible noise-bank assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from kryptonite.data.normalization import AudioNormalizationPolicy

NoiseCategory = Literal["stationary", "babble", "music", "impulsive", "low_snr"]
NoiseSeverity = Literal["light", "medium", "heavy"]

ALLOWED_NOISE_CATEGORIES: tuple[NoiseCategory, ...] = (
    "stationary",
    "babble",
    "music",
    "impulsive",
    "low_snr",
)
ALLOWED_NOISE_SEVERITIES: tuple[NoiseSeverity, ...] = ("light", "medium", "heavy")
SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3"})
REPORT_JSON_NAME = "noise_bank_report.json"
REPORT_MARKDOWN_NAME = "noise_bank_report.md"
MANIFEST_JSONL_NAME = "noise_bank_manifest.jsonl"
QUARANTINE_JSONL_NAME = "noise_bank_quarantine.jsonl"
MIX_MODE_BY_CATEGORY = {
    "stationary": "additive",
    "babble": "babble_overlay",
    "music": "music_overlay",
    "impulsive": "event_overlay",
    "low_snr": "additive",
}


@dataclass(frozen=True, slots=True)
class NoiseSeverityProfile:
    snr_db_min: float
    snr_db_max: float
    weight_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.snr_db_max < self.snr_db_min:
            raise ValueError("snr_db_max must be greater than or equal to snr_db_min")
        if self.weight_multiplier <= 0.0:
            raise ValueError("weight_multiplier must be positive")

    def to_dict(self) -> dict[str, float]:
        return {
            "snr_db_min": self.snr_db_min,
            "snr_db_max": self.snr_db_max,
            "weight_multiplier": self.weight_multiplier,
        }


@dataclass(frozen=True, slots=True)
class NoiseClassificationRule:
    match_any: tuple[str, ...]
    category: NoiseCategory | None = None
    severity: NoiseSeverity | None = None
    tags: tuple[str, ...] = ()

    def matches(self, normalized_path: str) -> bool:
        return any(token in normalized_path for token in self.match_any)

    def to_dict(self) -> dict[str, object]:
        return {
            "match_any": list(self.match_any),
            "category": self.category,
            "severity": self.severity,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class NoiseClassification:
    category: NoiseCategory
    severity: NoiseSeverity
    tags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "severity": self.severity,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class NoiseSourcePlan:
    id: str
    name: str
    inventory_source_id: str
    root_candidates: tuple[str, ...]
    default_category: NoiseCategory
    default_severity: NoiseSeverity
    base_weight: float = 1.0
    tags: tuple[str, ...] = ()
    classification_rules: tuple[NoiseClassificationRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.root_candidates:
            raise ValueError(f"Noise source '{self.id}' must define at least one root candidate.")
        if self.base_weight <= 0.0:
            raise ValueError(f"Noise source '{self.id}' must define a positive base_weight.")

    def classify(self, relative_path: str) -> NoiseClassification:
        normalized_path = relative_path.lower().replace("-", " ").replace("_", " ")
        category = self.default_category
        severity = self.default_severity
        tags = list(self.tags)
        for rule in self.classification_rules:
            if not rule.matches(normalized_path):
                continue
            if rule.category is not None:
                category = rule.category
            if rule.severity is not None:
                severity = rule.severity
            tags.extend(rule.tags)
        return NoiseClassification(
            category=category,
            severity=severity,
            tags=tuple(sorted(set(tags))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "inventory_source_id": self.inventory_source_id,
            "root_candidates": list(self.root_candidates),
            "default_category": self.default_category,
            "default_severity": self.default_severity,
            "base_weight": self.base_weight,
            "tags": list(self.tags),
            "classification_rules": [rule.to_dict() for rule in self.classification_rules],
        }


@dataclass(frozen=True, slots=True)
class NoiseBankPlan:
    notes: tuple[str, ...]
    severity_profiles: dict[NoiseSeverity, NoiseSeverityProfile]
    sources: tuple[NoiseSourcePlan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "notes": list(self.notes),
            "severity_profiles": {
                name: profile.to_dict()
                for name, profile in sorted(self.severity_profiles.items(), key=_severity_sort_key)
            },
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass(frozen=True, slots=True)
class NoiseSourceStatus:
    source_id: str
    name: str
    inventory_source_id: str
    configured_roots: tuple[str, ...]
    resolved_root: str | None
    status: str
    discovered_audio_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "inventory_source_id": self.inventory_source_id,
            "configured_roots": list(self.configured_roots),
            "resolved_root": self.resolved_root,
            "status": self.status,
            "discovered_audio_count": self.discovered_audio_count,
        }


@dataclass(frozen=True, slots=True)
class NoiseBankEntry:
    noise_id: str
    source_id: str
    source_name: str
    inventory_source_id: str
    source_audio_path: str
    normalized_audio_path: str
    relative_path: str
    category: NoiseCategory
    severity: NoiseSeverity
    mix_mode: str
    tags: tuple[str, ...]
    sampling_weight: float
    recommended_snr_db_min: float
    recommended_snr_db_max: float
    source_sample_rate_hz: int
    source_num_channels: int
    source_duration_seconds: float
    normalized_duration_seconds: float
    normalization_profile: str
    normalization_resampled: bool
    normalization_downmixed: bool
    normalization_peak_scaled: bool
    normalization_loudness_applied: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "noise_id": self.noise_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "inventory_source_id": self.inventory_source_id,
            "source_audio_path": self.source_audio_path,
            "normalized_audio_path": self.normalized_audio_path,
            "relative_path": self.relative_path,
            "category": self.category,
            "severity": self.severity,
            "mix_mode": self.mix_mode,
            "tags": list(self.tags),
            "sampling_weight": self.sampling_weight,
            "recommended_snr_db_min": self.recommended_snr_db_min,
            "recommended_snr_db_max": self.recommended_snr_db_max,
            "source_sample_rate_hz": self.source_sample_rate_hz,
            "source_num_channels": self.source_num_channels,
            "source_duration_seconds": self.source_duration_seconds,
            "normalized_duration_seconds": self.normalized_duration_seconds,
            "normalization_profile": self.normalization_profile,
            "normalization_resampled": self.normalization_resampled,
            "normalization_downmixed": self.normalization_downmixed,
            "normalization_peak_scaled": self.normalization_peak_scaled,
            "normalization_loudness_applied": self.normalization_loudness_applied,
        }


@dataclass(frozen=True, slots=True)
class NoiseBankQuarantineRecord:
    source_id: str
    source_name: str
    inventory_source_id: str
    source_audio_path: str
    category: NoiseCategory
    severity: NoiseSeverity
    issue_code: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "inventory_source_id": self.inventory_source_id,
            "source_audio_path": self.source_audio_path,
            "category": self.category,
            "severity": self.severity,
            "issue_code": self.issue_code,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class NoiseBankSummary:
    source_count: int
    present_source_count: int
    missing_source_count: int
    entry_count: int
    quarantine_count: int
    total_duration_seconds: float
    category_counts: dict[str, int]
    severity_counts: dict[str, int]
    source_entry_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_count": self.source_count,
            "present_source_count": self.present_source_count,
            "missing_source_count": self.missing_source_count,
            "entry_count": self.entry_count,
            "quarantine_count": self.quarantine_count,
            "total_duration_seconds": self.total_duration_seconds,
            "category_counts": dict(self.category_counts),
            "severity_counts": dict(self.severity_counts),
            "source_entry_counts": dict(self.source_entry_counts),
        }


@dataclass(frozen=True, slots=True)
class NoiseBankReport:
    generated_at: str
    project_root: str
    dataset_root: str
    output_root: str
    plan_path: str | None
    policy: AudioNormalizationPolicy
    notes: tuple[str, ...]
    severity_profiles: dict[NoiseSeverity, NoiseSeverityProfile]
    sources: tuple[NoiseSourceStatus, ...]
    entries: tuple[NoiseBankEntry, ...]
    quarantined: tuple[NoiseBankQuarantineRecord, ...]
    summary: NoiseBankSummary

    def to_dict(
        self,
        *,
        include_entries: bool = False,
        include_quarantine: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "dataset_root": self.dataset_root,
            "output_root": self.output_root,
            "plan_path": self.plan_path,
            "policy": self.policy.to_dict(),
            "notes": list(self.notes),
            "severity_profiles": {
                name: profile.to_dict()
                for name, profile in sorted(self.severity_profiles.items(), key=_severity_sort_key)
            },
            "sources": [source.to_dict() for source in self.sources],
            "summary": self.summary.to_dict(),
        }
        if include_entries:
            payload["entries"] = [entry.to_dict() for entry in self.entries]
        if include_quarantine:
            payload["quarantined"] = [record.to_dict() for record in self.quarantined]
        return payload


@dataclass(frozen=True, slots=True)
class WrittenNoiseBankArtifacts:
    output_root: str
    manifest_path: str
    quarantine_path: str
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "manifest_path": self.manifest_path,
            "quarantine_path": self.quarantine_path,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


def _severity_sort_key(item: tuple[str, NoiseSeverityProfile]) -> int:
    return ALLOWED_NOISE_SEVERITIES.index(cast(NoiseSeverity, item[0]))


__all__ = [
    "ALLOWED_NOISE_CATEGORIES",
    "ALLOWED_NOISE_SEVERITIES",
    "MANIFEST_JSONL_NAME",
    "MIX_MODE_BY_CATEGORY",
    "NoiseBankEntry",
    "NoiseBankPlan",
    "NoiseBankQuarantineRecord",
    "NoiseBankReport",
    "NoiseBankSummary",
    "NoiseCategory",
    "NoiseClassification",
    "NoiseClassificationRule",
    "NoiseSeverity",
    "NoiseSeverityProfile",
    "NoiseSourcePlan",
    "NoiseSourceStatus",
    "QUARANTINE_JSONL_NAME",
    "REPORT_JSON_NAME",
    "REPORT_MARKDOWN_NAME",
    "SUPPORTED_AUDIO_SUFFIXES",
    "WrittenNoiseBankArtifacts",
]
