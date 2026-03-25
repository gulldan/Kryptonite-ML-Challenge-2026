"""Runtime augmentation lookup and waveform transforms for production training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from kryptonite.config import AugmentationSchedulerConfig, SilenceAugmentationConfig
from kryptonite.data import load_codec_bank_plan
from kryptonite.data.far_field_bank import load_far_field_bank_plan
from kryptonite.deployment import resolve_project_path
from kryptonite.eval.corrupted_dev_suites.audio import (
    CodecSuiteCandidate,
    DistanceSuiteCandidate,
    NoiseSuiteCandidate,
    ReverbSuiteCandidate,
    SilenceSuiteCandidate,
    TransformOutcome,
    apply_codec_transform,
    apply_distance_transform,
    apply_noise_transform,
    apply_reverb_transform,
    apply_silence_transform,
)
from kryptonite.eval.corrupted_dev_suites.models import SuiteSeverity

from .augmentation_scheduler import (
    AugmentationCandidate,
    AugmentationCatalog,
    AugmentationScheduler,
    ScheduledAugmentation,
    ScheduledSampleRecipe,
)


@dataclass(slots=True)
class TrainingAugmentationRuntime:
    project_root: Path
    scheduler: AugmentationScheduler | None
    noise_candidates: dict[str, NoiseSuiteCandidate]
    reverb_candidates: dict[str, ReverbSuiteCandidate]
    distance_candidates: dict[str, DistanceSuiteCandidate]
    codec_candidates: dict[str, CodecSuiteCandidate]
    silence_candidates: dict[str, SilenceSuiteCandidate]
    ffmpeg_path: str = "ffmpeg"

    @classmethod
    def from_project_config(
        cls,
        *,
        project_root: Path | str,
        scheduler_config: AugmentationSchedulerConfig,
        silence_config: SilenceAugmentationConfig,
        total_epochs: int,
        ffmpeg_path: str = "ffmpeg",
    ) -> TrainingAugmentationRuntime:
        project_root_path = resolve_project_path(str(project_root), ".")
        noise_candidates = _load_noise_candidates(project_root_path=project_root_path)
        reverb_candidates = _load_reverb_candidates(project_root_path=project_root_path)
        distance_candidates = _load_distance_candidates(project_root_path=project_root_path)
        codec_candidates = _load_codec_candidates(project_root_path=project_root_path)
        silence_candidates = _load_silence_candidates(silence_config)

        catalog = _build_augmentation_catalog(
            noise_candidates=noise_candidates,
            reverb_candidates=reverb_candidates,
            distance_candidates=distance_candidates,
            codec_candidates=codec_candidates,
            silence_candidates=silence_candidates,
        )
        scheduler = None
        if scheduler_config.enabled and catalog.available_families:
            scheduler = AugmentationScheduler(
                config=scheduler_config,
                catalog=catalog,
                total_epochs=total_epochs,
            )

        return cls(
            project_root=project_root_path,
            scheduler=scheduler,
            noise_candidates=noise_candidates,
            reverb_candidates=reverb_candidates,
            distance_candidates=distance_candidates,
            codec_candidates=codec_candidates,
            silence_candidates=silence_candidates,
            ffmpeg_path=ffmpeg_path,
        )

    @property
    def has_effective_augmentation(self) -> bool:
        return any(
            (
                self.noise_candidates,
                self.reverb_candidates,
                self.distance_candidates,
                self.codec_candidates,
                self.silence_candidates,
            )
        )

    def sample_recipe(
        self,
        *,
        epoch_index: int,
        rng: random.Random,
    ) -> ScheduledSampleRecipe | None:
        if self.scheduler is None:
            return None
        return self.scheduler.sample_recipe(epoch_index=epoch_index, rng=rng)

    def apply_augmentations(
        self,
        *,
        waveform: Any,
        sample_rate_hz: int,
        augmentations: tuple[ScheduledAugmentation, ...],
        rng: random.Random,
    ) -> tuple[np.ndarray, int, tuple[dict[str, object], ...]]:
        current_waveform = np.asarray(waveform, dtype=np.float32)
        current_sample_rate_hz = sample_rate_hz
        trace: list[dict[str, object]] = []

        for augmentation in augmentations:
            outcome = self._apply_scheduled_augmentation(
                waveform=current_waveform,
                sample_rate_hz=current_sample_rate_hz,
                augmentation=augmentation,
                rng=rng,
            )
            current_waveform = np.asarray(outcome.waveform, dtype=np.float32)
            current_sample_rate_hz = int(outcome.sample_rate_hz)
            trace.append(
                {
                    "family": augmentation.family,
                    "candidate_id": outcome.candidate_id,
                    "label": augmentation.label,
                    "severity": outcome.severity,
                    "metadata": dict(outcome.metadata),
                }
            )

        return current_waveform, current_sample_rate_hz, tuple(trace)

    def _apply_scheduled_augmentation(
        self,
        *,
        waveform: np.ndarray,
        sample_rate_hz: int,
        augmentation: ScheduledAugmentation,
        rng: random.Random,
    ) -> TransformOutcome:
        if augmentation.family == "noise":
            candidate = self.noise_candidates.get(augmentation.candidate_id)
            if candidate is None:
                raise KeyError(f"Unknown noise candidate {augmentation.candidate_id!r}.")
            return apply_noise_transform(
                waveform=waveform,
                sample_rate_hz=sample_rate_hz,
                candidate=candidate,
                rng=rng,
                project_root=self.project_root,
            )
        if augmentation.family == "reverb":
            candidate = self.reverb_candidates.get(augmentation.candidate_id)
            if candidate is None:
                raise KeyError(f"Unknown reverb candidate {augmentation.candidate_id!r}.")
            return apply_reverb_transform(
                waveform=waveform,
                sample_rate_hz=sample_rate_hz,
                candidate=candidate,
                rng=rng,
                project_root=self.project_root,
            )
        if augmentation.family == "distance":
            candidate = self.distance_candidates.get(augmentation.candidate_id)
            if candidate is None:
                raise KeyError(f"Unknown distance candidate {augmentation.candidate_id!r}.")
            return apply_distance_transform(
                waveform=waveform,
                sample_rate_hz=sample_rate_hz,
                candidate=candidate,
            )
        if augmentation.family == "codec":
            candidate = self.codec_candidates.get(augmentation.candidate_id)
            if candidate is None:
                raise KeyError(f"Unknown codec candidate {augmentation.candidate_id!r}.")
            return apply_codec_transform(
                waveform=waveform,
                sample_rate_hz=sample_rate_hz,
                candidate=candidate,
                ffmpeg_path=self.ffmpeg_path,
            )
        if augmentation.family == "silence":
            candidate = self.silence_candidates.get(augmentation.candidate_id)
            if candidate is None:
                raise KeyError(f"Unknown silence candidate {augmentation.candidate_id!r}.")
            return apply_silence_transform(
                waveform=waveform,
                sample_rate_hz=sample_rate_hz,
                candidate=candidate,
                rng=rng,
            )
        raise ValueError(f"Unsupported augmentation family {augmentation.family!r}.")


def _build_augmentation_catalog(
    *,
    noise_candidates: dict[str, NoiseSuiteCandidate],
    reverb_candidates: dict[str, ReverbSuiteCandidate],
    distance_candidates: dict[str, DistanceSuiteCandidate],
    codec_candidates: dict[str, CodecSuiteCandidate],
    silence_candidates: dict[str, SilenceSuiteCandidate],
) -> AugmentationCatalog:
    return AugmentationCatalog(
        candidates_by_family={
            "noise": tuple(
                _noise_catalog_candidate(candidate)
                for _, candidate in sorted(noise_candidates.items())
            ),
            "reverb": tuple(
                _reverb_catalog_candidate(candidate)
                for _, candidate in sorted(reverb_candidates.items())
            ),
            "distance": tuple(
                _distance_catalog_candidate(candidate)
                for _, candidate in sorted(distance_candidates.items())
            ),
            "codec": tuple(
                _codec_catalog_candidate(candidate)
                for _, candidate in sorted(codec_candidates.items())
            ),
            "silence": tuple(
                _silence_catalog_candidate(candidate)
                for _, candidate in sorted(silence_candidates.items())
            ),
        }
    )


def _load_noise_candidates(*, project_root_path: Path) -> dict[str, NoiseSuiteCandidate]:
    manifest_path = (
        project_root_path / "artifacts/corruptions/noise-bank/manifests/noise_bank_manifest.jsonl"
    )
    candidates: dict[str, NoiseSuiteCandidate] = {}
    for record in _read_jsonl_records(manifest_path):
        severity = _coerce_severity(record.get("severity"))
        normalized_audio_path = _coerce_str(record.get("normalized_audio_path"))
        if severity is None or normalized_audio_path is None:
            continue
        candidate_id = _coerce_str(record.get("noise_id")) or normalized_audio_path
        candidates[candidate_id] = NoiseSuiteCandidate(
            candidate_id=candidate_id,
            severity=severity,
            weight=_coerce_float(record.get("sampling_weight"), default=1.0),
            audio_path=normalized_audio_path,
            category=_coerce_str(record.get("category")) or "unknown",
            mix_mode=_coerce_str(record.get("mix_mode")) or "additive",
            snr_db_min=_coerce_float(record.get("recommended_snr_db_min"), default=0.0),
            snr_db_max=_coerce_float(record.get("recommended_snr_db_max"), default=0.0),
            tags=tuple(_coerce_string_list(record.get("tags"))),
        )
    return candidates


def _load_reverb_candidates(*, project_root_path: Path) -> dict[str, ReverbSuiteCandidate]:
    room_config_manifest = (
        project_root_path / "artifacts/corruptions/rir-bank/manifests/room_simulation_configs.jsonl"
    )
    rir_manifest = (
        project_root_path / "artifacts/corruptions/rir-bank/manifests/rir_bank_manifest.jsonl"
    )
    if not room_config_manifest.exists() or not rir_manifest.exists():
        return {}

    rir_lookup = {
        rir_id: record
        for record in _read_jsonl_records(rir_manifest)
        if (rir_id := _coerce_str(record.get("rir_id"))) is not None
    }
    candidates: dict[str, ReverbSuiteCandidate] = {}
    for record in _read_jsonl_records(room_config_manifest):
        config_id = _coerce_str(record.get("config_id"))
        direct_condition = _coerce_str(record.get("direct_condition"))
        if config_id is None or direct_condition is None:
            continue
        rir_ids = tuple(
            rir_id
            for rir_id in _coerce_string_list(record.get("sample_rir_ids"))
            if rir_id in rir_lookup
        )
        if not rir_ids:
            continue
        rir_audio_paths = tuple(
            _coerce_str(rir_lookup[rir_id].get("normalized_audio_path")) or ""
            for rir_id in rir_ids
            if _coerce_str(rir_lookup[rir_id].get("normalized_audio_path"))
        )
        if not rir_audio_paths:
            continue
        candidates[config_id] = ReverbSuiteCandidate(
            candidate_id=config_id,
            severity=_severity_from_direct_condition(direct_condition),
            weight=max(1.0, float(len(rir_audio_paths))),
            room_size=_coerce_str(record.get("room_size")) or "unknown",
            field=_coerce_str(record.get("field")) or "unknown",
            rt60_bucket=_coerce_str(record.get("rt60_bucket")) or "unknown",
            direct_condition=direct_condition,
            rir_ids=rir_ids,
            rir_audio_paths=rir_audio_paths,
        )
    return candidates


def _load_distance_candidates(*, project_root_path: Path) -> dict[str, DistanceSuiteCandidate]:
    plan_path = resolve_project_path(
        str(project_root_path),
        "configs/corruption/far-field-bank.toml",
    )
    if not plan_path.exists():
        return {}
    plan = load_far_field_bank_plan(plan_path)
    candidates: dict[str, DistanceSuiteCandidate] = {}
    for preset in plan.presets:
        candidates[preset.id] = DistanceSuiteCandidate(
            candidate_id=preset.id,
            severity=_severity_from_distance_field(preset.field),
            weight=preset.sampling_weight,
            field=preset.field,
            preset=preset,
            render_settings=plan.render,
        )
    return candidates


def _load_codec_candidates(*, project_root_path: Path) -> dict[str, CodecSuiteCandidate]:
    plan_path = resolve_project_path(str(project_root_path), "configs/corruption/codec-bank.toml")
    if not plan_path.exists():
        return {}
    plan = load_codec_bank_plan(plan_path)
    candidates: dict[str, CodecSuiteCandidate] = {}
    for preset in plan.presets:
        candidates[preset.id] = CodecSuiteCandidate(
            candidate_id=preset.id,
            severity=preset.severity,
            weight=preset.sampling_weight(plan.severity_profiles),
            family=preset.family,
            preset=preset,
        )
    return candidates


def _load_silence_candidates(
    silence_config: SilenceAugmentationConfig,
) -> dict[str, SilenceSuiteCandidate]:
    if not _has_effective_silence_profile(silence_config):
        return {}
    candidates: dict[str, SilenceSuiteCandidate] = {}
    for severity, scale in (("light", 0.5), ("medium", 0.8), ("heavy", 1.0)):
        candidate_id = f"silence-{severity}"
        candidates[candidate_id] = SilenceSuiteCandidate(
            candidate_id=candidate_id,
            severity=cast(Any, severity),
            weight=1.0,
            config=_scaled_silence_config(silence_config, scale=scale),
        )
    return candidates


def _noise_catalog_candidate(candidate: NoiseSuiteCandidate) -> AugmentationCandidate:
    return AugmentationCandidate(
        family="noise",
        candidate_id=candidate.candidate_id,
        label=f"noise/{candidate.category}/{candidate.severity}",
        severity=candidate.severity,
        weight=candidate.weight,
        tags=candidate.tags,
        metadata={
            "category": candidate.category,
            "mix_mode": candidate.mix_mode,
            "recommended_snr_db_min": candidate.snr_db_min,
            "recommended_snr_db_max": candidate.snr_db_max,
        },
    )


def _reverb_catalog_candidate(candidate: ReverbSuiteCandidate) -> AugmentationCandidate:
    return AugmentationCandidate(
        family="reverb",
        candidate_id=candidate.candidate_id,
        label=f"reverb/{candidate.room_size}/{candidate.field}/{candidate.direct_condition}",
        severity=candidate.severity,
        weight=candidate.weight,
        metadata={
            "room_size": candidate.room_size,
            "field": candidate.field,
            "rt60_bucket": candidate.rt60_bucket,
            "direct_condition": candidate.direct_condition,
            "rir_count": len(candidate.rir_audio_paths),
        },
    )


def _distance_catalog_candidate(candidate: DistanceSuiteCandidate) -> AugmentationCandidate:
    return AugmentationCandidate(
        family="distance",
        candidate_id=candidate.candidate_id,
        label=f"distance/{candidate.field}/{candidate.candidate_id}",
        severity=candidate.severity,
        weight=candidate.weight,
        tags=candidate.preset.tags,
        metadata={
            "field": candidate.field,
            "distance_meters": candidate.preset.distance_meters,
            "target_drr_db": candidate.preset.target_drr_db,
        },
    )


def _codec_catalog_candidate(candidate: CodecSuiteCandidate) -> AugmentationCandidate:
    return AugmentationCandidate(
        family="codec",
        candidate_id=candidate.candidate_id,
        label=f"codec/{candidate.family}/{candidate.severity}",
        severity=candidate.severity,
        weight=candidate.weight,
        tags=candidate.preset.tags,
        metadata={
            "codec_family": candidate.family,
            "codec_name": candidate.preset.codec_name,
        },
    )


def _silence_catalog_candidate(candidate: SilenceSuiteCandidate) -> AugmentationCandidate:
    return AugmentationCandidate(
        family="silence",
        candidate_id=candidate.candidate_id,
        label=f"silence/{candidate.severity}",
        severity=candidate.severity,
        weight=candidate.weight,
        metadata={
            "max_leading_padding_seconds": candidate.config.max_leading_padding_seconds,
            "max_trailing_padding_seconds": candidate.config.max_trailing_padding_seconds,
            "max_inserted_pauses": candidate.config.max_inserted_pauses,
            "pause_ratio_min": candidate.config.pause_ratio_min,
            "pause_ratio_max": candidate.config.pause_ratio_max,
        },
    )


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        payload
        for line in path.read_text().splitlines()
        if line.strip()
        for payload in [json.loads(line)]
        if isinstance(payload, dict)
    ]


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _coerce_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"Expected numeric value, got {type(value)!r}")
    return float(value)


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _coerce_severity(value: object) -> SuiteSeverity | None:
    normalized = _coerce_str(value)
    if normalized in {"light", "medium", "heavy"}:
        return cast(SuiteSeverity, normalized)
    return None


def _severity_from_direct_condition(value: str) -> SuiteSeverity:
    return cast(
        SuiteSeverity,
        {
            "high": "light",
            "medium": "medium",
            "low": "heavy",
        }.get(value, "medium"),
    )


def _severity_from_distance_field(value: str) -> SuiteSeverity:
    return cast(
        SuiteSeverity,
        {
            "near": "light",
            "mid": "medium",
            "far": "heavy",
        }.get(value, "medium"),
    )


def _has_effective_silence_profile(config: SilenceAugmentationConfig) -> bool:
    return any(
        (
            config.max_leading_padding_seconds > 0.0,
            config.max_trailing_padding_seconds > 0.0,
            config.max_inserted_pauses > 0 and config.max_inserted_pause_seconds > 0.0,
            config.pause_ratio_min < 1.0,
            config.pause_ratio_max > 1.0,
        )
    )


def _scaled_silence_config(
    config: SilenceAugmentationConfig,
    *,
    scale: float,
) -> SilenceAugmentationConfig:
    max_inserted_pauses = 0
    if config.max_inserted_pauses > 0:
        max_inserted_pauses = max(1, round(config.max_inserted_pauses * scale))
    return SilenceAugmentationConfig(
        enabled=True,
        max_leading_padding_seconds=round(config.max_leading_padding_seconds * scale, 6),
        max_trailing_padding_seconds=round(config.max_trailing_padding_seconds * scale, 6),
        max_inserted_pauses=max_inserted_pauses,
        min_inserted_pause_seconds=round(config.min_inserted_pause_seconds, 6),
        max_inserted_pause_seconds=round(config.max_inserted_pause_seconds * scale, 6),
        pause_ratio_min=round(_scale_pause_ratio(config.pause_ratio_min, scale), 6),
        pause_ratio_max=round(_scale_pause_ratio(config.pause_ratio_max, scale), 6),
        min_detected_pause_seconds=round(config.min_detected_pause_seconds, 6),
        max_perturbed_pause_seconds=round(
            config.min_detected_pause_seconds
            + (config.max_perturbed_pause_seconds - config.min_detected_pause_seconds) * scale,
            6,
        ),
        analysis_frame_ms=round(config.analysis_frame_ms, 6),
        silence_threshold_dbfs=round(config.silence_threshold_dbfs, 6),
    )


def _scale_pause_ratio(value: float, scale: float) -> float:
    if value >= 1.0:
        return 1.0 + ((value - 1.0) * scale)
    return 1.0 - ((1.0 - value) * scale)


__all__ = ["TrainingAugmentationRuntime"]
