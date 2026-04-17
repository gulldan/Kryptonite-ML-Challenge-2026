"""Runtime waveform transforms driven by the training augmentation scheduler."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kryptonite.config import AugmentationSchedulerConfig, SilenceAugmentationConfig

from .augmentation_runtime_catalog import (
    _candidate_map,
    build_augmentation_catalog_from_manifests,
)
from .augmentation_runtime_transforms import (
    _apply_codec,
    _apply_distance,
    _apply_noise,
    _apply_reverb,
    _apply_silence,
    _apply_speed,
    _coerce_audio,
)
from .augmentation_scheduler import (
    AugmentationCandidate,
    AugmentationCatalog,
    AugmentationScheduler,
    ScheduledAugmentation,
    ScheduledSampleRecipe,
    resolve_bank_manifest_paths,
)

_TRANSFORM_PRIORITY = {
    "speed": 0,
    "reverb": 1,
    "distance": 2,
    "codec": 3,
    "noise": 4,
    "silence": 5,
}


@dataclass(slots=True)
class TrainingAugmentationRuntime:
    scheduler: AugmentationScheduler
    catalog: AugmentationCatalog
    project_root: Path
    silence_config: SilenceAugmentationConfig
    noise_candidates: dict[str, AugmentationCandidate]
    reverb_candidates: dict[str, AugmentationCandidate]
    distance_candidates: dict[str, AugmentationCandidate]
    codec_candidates: dict[str, AugmentationCandidate]
    silence_candidates: dict[str, AugmentationCandidate]
    speed_candidates: dict[str, AugmentationCandidate]

    @classmethod
    def from_project_config(
        cls,
        *,
        project_root: Path | str,
        scheduler_config: AugmentationSchedulerConfig,
        silence_config: SilenceAugmentationConfig,
        total_epochs: int,
    ) -> TrainingAugmentationRuntime:
        resolved = resolve_bank_manifest_paths(project_root=project_root)
        catalog = build_augmentation_catalog_from_manifests(
            project_root=project_root,
            scheduler_config=scheduler_config,
            silence_config=silence_config,
            noise_manifest_path=resolved.noise_manifest_path,
            room_config_manifest_path=resolved.room_config_manifest_path,
            rir_manifest_path=resolved.rir_manifest_path,
            distance_manifest_path=resolved.distance_manifest_path,
            codec_manifest_path=resolved.codec_manifest_path,
            require_audio_files=True,
        )
        return cls(
            scheduler=AugmentationScheduler(
                config=scheduler_config,
                catalog=catalog,
                total_epochs=total_epochs,
            ),
            catalog=catalog,
            project_root=Path(project_root),
            silence_config=silence_config,
            noise_candidates=_candidate_map(catalog, "noise"),
            reverb_candidates=_candidate_map(catalog, "reverb"),
            distance_candidates=_candidate_map(catalog, "distance"),
            codec_candidates=_candidate_map(catalog, "codec"),
            silence_candidates=_candidate_map(catalog, "silence"),
            speed_candidates=_candidate_map(catalog, "speed"),
        )

    @property
    def has_effective_augmentation(self) -> bool:
        return bool(self.catalog.available_families)

    def sample_recipe(self, *, epoch: int, rng: random.Random) -> ScheduledSampleRecipe:
        return self.scheduler.sample_recipe(epoch=epoch, rng=rng)

    def apply_augmentations(
        self,
        waveform: Any,
        *,
        sample_rate_hz: int,
        augmentations: tuple[ScheduledAugmentation, ...],
        rng: random.Random,
    ) -> tuple[NDArray[np.float32], tuple[dict[str, object], ...]]:
        augmented = _coerce_audio(waveform)
        traces: list[dict[str, object]] = []
        ordered = sorted(
            augmentations,
            key=lambda augmentation: _TRANSFORM_PRIORITY.get(augmentation.family, 99),
        )
        for augmentation in ordered:
            before_samples = int(augmented.shape[-1])
            try:
                augmented, metadata = self._apply_scheduled_augmentation(
                    augmented,
                    sample_rate_hz=sample_rate_hz,
                    augmentation=augmentation,
                    rng=rng,
                )
            except FileNotFoundError as exc:
                metadata = {"skip_reason": f"missing_audio:{exc.filename}"}
            except ValueError as exc:
                metadata = {"skip_reason": str(exc)}
            traces.append(
                {
                    "family": augmentation.family,
                    "candidate_id": augmentation.candidate_id,
                    "label": augmentation.label,
                    "severity": augmentation.severity,
                    "input_samples": before_samples,
                    "output_samples": int(augmented.shape[-1]),
                    "metadata": metadata,
                }
            )
        return np.clip(augmented, -1.0, 1.0).astype(np.float32, copy=False), tuple(traces)

    def _apply_scheduled_augmentation(
        self,
        waveform: NDArray[np.float32],
        *,
        sample_rate_hz: int,
        augmentation: ScheduledAugmentation,
        rng: random.Random,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        candidate = self._candidate_for(augmentation)
        metadata = dict(candidate.metadata or {}) if candidate is not None else {}
        metadata.update(dict(augmentation.metadata or {}))
        metadata.setdefault("severity", augmentation.severity)
        if augmentation.family == "speed":
            return _apply_speed(waveform, metadata=metadata)
        if augmentation.family == "noise":
            return _apply_noise(
                waveform,
                project_root=self.project_root,
                metadata=metadata,
                sample_rate_hz=sample_rate_hz,
                rng=rng,
            )
        if augmentation.family == "reverb":
            return _apply_reverb(
                waveform,
                project_root=self.project_root,
                metadata=metadata,
                sample_rate_hz=sample_rate_hz,
                rng=rng,
            )
        if augmentation.family == "distance":
            return _apply_distance(
                waveform,
                project_root=self.project_root,
                metadata=metadata,
                sample_rate_hz=sample_rate_hz,
                rng=rng,
            )
        if augmentation.family == "codec":
            return _apply_codec(waveform, metadata=metadata, sample_rate_hz=sample_rate_hz, rng=rng)
        if augmentation.family == "silence":
            return _apply_silence(
                waveform,
                base_config=self.silence_config,
                metadata=metadata,
                sample_rate_hz=sample_rate_hz,
                rng=rng,
            )
        raise ValueError(f"Unsupported augmentation family: {augmentation.family!r}")

    def _candidate_for(
        self,
        augmentation: ScheduledAugmentation,
    ) -> AugmentationCandidate | None:
        candidates = {
            "noise": self.noise_candidates,
            "reverb": self.reverb_candidates,
            "distance": self.distance_candidates,
            "codec": self.codec_candidates,
            "silence": self.silence_candidates,
            "speed": self.speed_candidates,
        }.get(augmentation.family)
        if candidates is None:
            return None
        return candidates.get(augmentation.candidate_id)


__all__ = [
    "TrainingAugmentationRuntime",
    "build_augmentation_catalog_from_manifests",
]
