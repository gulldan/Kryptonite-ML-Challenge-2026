"""Runtime waveform transforms driven by the training augmentation scheduler."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from kryptonite.config import AugmentationSchedulerConfig, SilenceAugmentationConfig
from kryptonite.data.audio_io import read_audio_file
from kryptonite.data.convolution import fft_convolve_1d
from kryptonite.deployment import resolve_project_path

from .augmentation_scheduler import (
    AugmentationCandidate,
    AugmentationCatalog,
    AugmentationFamily,
    AugmentationScheduler,
    AugmentationSeverity,
    ScheduledAugmentation,
    ScheduledSampleRecipe,
    resolve_bank_manifest_paths,
)

_SEVERITY_SCALE = {"light": 0.45, "medium": 0.70, "heavy": 1.0}
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


def build_augmentation_catalog_from_manifests(
    *,
    project_root: Path | str,
    scheduler_config: AugmentationSchedulerConfig,
    silence_config: SilenceAugmentationConfig,
    noise_manifest_path: Path | str,
    room_config_manifest_path: Path | str,
    rir_manifest_path: Path | str,
    distance_manifest_path: Path | str,
    codec_manifest_path: Path | str,
    require_audio_files: bool = False,
) -> AugmentationCatalog:
    project_root_path = Path(project_root)
    candidates_by_family: dict[AugmentationFamily, tuple[AugmentationCandidate, ...]] = {
        "noise": (
            _load_noise_candidates(
                project_root_path,
                Path(noise_manifest_path),
                require_audio_files=require_audio_files,
            )
            + _load_musan_candidates(
                project_root_path,
                require_audio_files=require_audio_files,
            )
        ),
        "reverb": (
            _load_reverb_candidates(
                project_root_path,
                room_config_manifest=Path(room_config_manifest_path),
                rir_manifest=Path(rir_manifest_path),
                require_audio_files=require_audio_files,
            )
            + _load_direct_rir_candidates(
                project_root_path,
                require_audio_files=require_audio_files,
            )
        ),
        "distance": _load_distance_candidates(project_root_path, Path(distance_manifest_path)),
        "codec": _load_codec_candidates(project_root_path, Path(codec_manifest_path)),
        "silence": _load_silence_candidates(silence_config),
        "speed": (
            _load_speed_candidates()
            if float(getattr(scheduler_config.family_weights, "speed", 0.0)) > 0.0
            else ()
        ),
    }
    return AugmentationCatalog(candidates_by_family=candidates_by_family)


def _candidate_map(
    catalog: AugmentationCatalog,
    family: AugmentationFamily,
) -> dict[str, AugmentationCandidate]:
    return {
        candidate.candidate_id: candidate
        for candidate in catalog.candidates_by_family.get(family, ())
    }


def _load_noise_candidates(
    project_root: Path,
    manifest_path: Path,
    *,
    require_audio_files: bool,
) -> tuple[AugmentationCandidate, ...]:
    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl_records(manifest_path):
        audio_path = _coerce_str(record.get("normalized_audio_path"))
        if not audio_path:
            audio_path = ""
        resolved_audio_path = (
            resolve_project_path(str(project_root), audio_path) if audio_path else None
        )
        if require_audio_files and (
            resolved_audio_path is None or not resolved_audio_path.exists()
        ):
            continue
        severity = _coerce_severity(record.get("severity"))
        category = _coerce_str(record.get("category")) or "noise"
        candidate_id = _coerce_str(record.get("noise_id")) or f"noise-{len(candidates)}"
        candidates.append(
            AugmentationCandidate(
                family="noise",
                candidate_id=candidate_id,
                label=f"noise/{category}/{severity}",
                severity=severity,
                weight=_coerce_float(record.get("sampling_weight"), 1.0),
                metadata={
                    "normalized_audio_path": audio_path,
                    "category": category,
                    "mix_mode": _coerce_str(record.get("mix_mode")) or "additive",
                    "snr_db_min": _coerce_float(record.get("recommended_snr_db_min"), 8.0),
                    "snr_db_max": _coerce_float(record.get("recommended_snr_db_max"), 18.0),
                },
            )
        )
    return tuple(candidates)


def _load_musan_candidates(
    project_root: Path,
    *,
    require_audio_files: bool,
) -> tuple[AugmentationCandidate, ...]:
    musan_root = project_root / "datasets/musan"
    if require_audio_files and not musan_root.exists():
        return ()
    category_specs = (
        ("noise", "musan-noise", "noise", "medium", 1.2, 4.0, 18.0),
        ("music", "musan-music", "music", "medium", 1.0, 8.0, 20.0),
        ("speech", "musan-speech", "babble", "medium", 1.1, 5.0, 15.0),
    )
    candidates: list[AugmentationCandidate] = []
    for directory_name, source_id, category, severity, weight, snr_min, snr_max in category_specs:
        directory = musan_root / directory_name
        if not directory.exists():
            continue
        audio_paths = sorted(
            path
            for pattern in ("*.wav", "*.flac")
            for path in directory.rglob(pattern)
            if path.is_file()
        )
        for index, audio_path in enumerate(audio_paths):
            relative_path = audio_path.relative_to(project_root).as_posix()
            candidates.append(
                AugmentationCandidate(
                    family="noise",
                    candidate_id=f"{source_id}-{index:05d}",
                    label=f"noise/{category}/{severity}",
                    severity=cast(AugmentationSeverity, severity),
                    weight=weight,
                    metadata={
                        "normalized_audio_path": relative_path,
                        "category": category,
                        "mix_mode": "additive",
                        "snr_db_min": snr_min,
                        "snr_db_max": snr_max,
                        "source": "musan",
                    },
                )
            )
    return tuple(candidates)


def _load_reverb_candidates(
    project_root: Path,
    *,
    room_config_manifest: Path,
    rir_manifest: Path,
    require_audio_files: bool,
) -> tuple[AugmentationCandidate, ...]:
    rir_paths: dict[str, str] = {}
    for record in _read_jsonl_records(rir_manifest):
        rir_id = _coerce_str(record.get("rir_id"))
        audio_path = _coerce_str(record.get("normalized_audio_path"))
        if rir_id and audio_path and resolve_project_path(str(project_root), audio_path).exists():
            rir_paths[rir_id] = audio_path

    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl_records(room_config_manifest):
        sample_ids = _coerce_string_list(record.get("sample_rir_ids"))
        audio_paths = tuple(rir_paths[rir_id] for rir_id in sample_ids if rir_id in rir_paths)
        if require_audio_files and not audio_paths:
            continue
        config_id = _coerce_str(record.get("config_id")) or f"reverb-{len(candidates)}"
        direct_condition = _coerce_str(record.get("direct_condition")) or "medium"
        severity = _severity_from_direct_condition(direct_condition)
        candidates.append(
            AugmentationCandidate(
                family="reverb",
                candidate_id=config_id,
                label=(
                    "reverb/"
                    f"{_coerce_str(record.get('room_size')) or 'room'}/"
                    f"{_coerce_str(record.get('rt60_bucket')) or 'rt60'}/"
                    f"{severity}"
                ),
                severity=severity,
                weight=max(1.0, _coerce_float(record.get("rir_count"), 1.0) ** 0.25),
                metadata={
                    "rir_audio_paths": audio_paths,
                    "direct_condition": direct_condition,
                    "field": _coerce_str(record.get("field")) or "",
                    "rt60_bucket": _coerce_str(record.get("rt60_bucket")) or "",
                },
            )
        )
    return tuple(candidates)


def _load_direct_rir_candidates(
    project_root: Path,
    *,
    require_audio_files: bool,
) -> tuple[AugmentationCandidate, ...]:
    candidate_roots = (
        project_root / "datasets/rirs_noises/RIRS_NOISES",
        project_root / "datasets/RIRS_NOISES",
    )
    roots = tuple(root for root in candidate_roots if root.exists())
    if require_audio_files and not roots:
        return ()
    audio_paths = sorted(
        path
        for root in roots
        for directory in (
            root / "simulated_rirs",
            root / "real_rirs_isotropic_noises",
        )
        if directory.exists()
        for path in directory.rglob("*.wav")
        if path.is_file()
    )
    candidates: list[AugmentationCandidate] = []
    for index, audio_path in enumerate(audio_paths):
        relative_path = audio_path.relative_to(project_root).as_posix()
        severity, field, direct_condition = _direct_rir_profile(audio_path)
        candidates.append(
            AugmentationCandidate(
                family="reverb",
                candidate_id=f"direct-rir-{index:06d}",
                label=f"reverb/raw-rirs/{field}/{severity}",
                severity=severity,
                weight=1.0,
                metadata={
                    "rir_audio_paths": (relative_path,),
                    "direct_condition": direct_condition,
                    "field": field,
                    "rt60_bucket": "raw",
                    "source": "rirs_noises_raw",
                },
            )
        )
    return tuple(candidates)


def _direct_rir_profile(path: Path) -> tuple[AugmentationSeverity, str, str]:
    parts = {part.lower() for part in path.parts}
    if "largeroom" in parts:
        return "heavy", "far", "low"
    if "mediumroom" in parts:
        return "medium", "mid", "medium"
    if "smallroom" in parts:
        return "light", "near", "high"
    return "medium", "mid", "medium"


def _load_distance_candidates(
    project_root: Path,
    manifest_path: Path,
) -> tuple[AugmentationCandidate, ...]:
    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl_records(manifest_path):
        kernel_path = _coerce_str(record.get("kernel_audio_path"))
        if kernel_path and not resolve_project_path(str(project_root), kernel_path).exists():
            kernel_path = ""
        preset_id = _coerce_str(record.get("preset_id")) or f"distance-{len(candidates)}"
        field = _coerce_str(record.get("field")) or "mid"
        severity = _severity_from_distance_field(field)
        candidates.append(
            AugmentationCandidate(
                family="distance",
                candidate_id=preset_id,
                label=f"distance/{field}/{severity}",
                severity=severity,
                weight=_coerce_float(record.get("sampling_weight"), 1.0),
                metadata={
                    "kernel_audio_path": kernel_path,
                    "field": field,
                    "lowpass_hz": _coerce_float(record.get("lowpass_hz"), 5600.0),
                    "attenuation_db": _coerce_float(record.get("attenuation_db"), 3.0),
                    "target_drr_db": _coerce_float(record.get("target_drr_db"), 0.0),
                },
            )
        )
    return tuple(candidates)


def _load_codec_candidates(
    project_root: Path,
    manifest_path: Path,
) -> tuple[AugmentationCandidate, ...]:
    del project_root
    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl_records(manifest_path):
        preset_id = _coerce_str(record.get("preset_id")) or f"codec-{len(candidates)}"
        family = _coerce_str(record.get("family")) or "codec"
        severity = _coerce_severity(record.get("severity"))
        candidates.append(
            AugmentationCandidate(
                family="codec",
                candidate_id=preset_id,
                label=f"codec/{family}/{severity}",
                severity=severity,
                weight=_coerce_float(record.get("sampling_weight"), 1.0),
                metadata={
                    "codec_family": family,
                    "pre_filter_graph": _coerce_str(record.get("ffmpeg_pre_filter_graph")) or "",
                    "post_filter_graph": _coerce_str(record.get("ffmpeg_post_filter_graph")) or "",
                    "codec_name": _coerce_str(record.get("ffmpeg_encode_codec")) or "",
                },
            )
        )
    return tuple(candidates)


def _load_silence_candidates(
    silence_config: SilenceAugmentationConfig,
) -> tuple[AugmentationCandidate, ...]:
    if not _has_effective_silence_config(silence_config):
        return ()
    return (
        AugmentationCandidate(
            family="silence",
            candidate_id="silence-light",
            label="silence/edge-pause/light",
            severity="light",
            weight=1.0,
            metadata={"scale": 0.45},
        ),
        AugmentationCandidate(
            family="silence",
            candidate_id="silence-medium",
            label="silence/edge-pause/medium",
            severity="medium",
            weight=1.0,
            metadata={"scale": 0.70},
        ),
        AugmentationCandidate(
            family="silence",
            candidate_id="silence-heavy",
            label="silence/edge-pause/heavy",
            severity="heavy",
            weight=1.0,
            metadata={"scale": 1.0},
        ),
    )


def _load_speed_candidates() -> tuple[AugmentationCandidate, ...]:
    return (
        AugmentationCandidate(
            family="speed",
            candidate_id="speed-0.95",
            label="speed/0.95/light",
            severity="light",
            weight=1.0,
            metadata={"speed_factor": 0.95},
        ),
        AugmentationCandidate(
            family="speed",
            candidate_id="speed-1.05",
            label="speed/1.05/light",
            severity="light",
            weight=1.0,
            metadata={"speed_factor": 1.05},
        ),
        AugmentationCandidate(
            family="speed",
            candidate_id="speed-0.90",
            label="speed/0.90/medium",
            severity="medium",
            weight=0.8,
            metadata={"speed_factor": 0.90},
        ),
        AugmentationCandidate(
            family="speed",
            candidate_id="speed-1.10",
            label="speed/1.10/medium",
            severity="medium",
            weight=0.8,
            metadata={"speed_factor": 1.10},
        ),
    )


def _apply_noise(
    waveform: NDArray[np.float32],
    *,
    project_root: Path,
    metadata: dict[str, object],
    sample_rate_hz: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    noise_path = _metadata_str(metadata, "normalized_audio_path")
    if not noise_path:
        raise ValueError("missing_noise_audio_path")
    noise = _read_mono_audio(project_root / noise_path)
    noise_segment = _sample_or_tile(noise, target_samples=int(waveform.shape[-1]), rng=rng)
    signal_rms = _rms(waveform)
    noise_rms = _rms(noise_segment)
    if signal_rms <= 1e-8 or noise_rms <= 1e-8:
        return waveform, {"skip_reason": "zero_rms"}
    snr_min = _metadata_float(metadata, "snr_db_min", 8.0)
    snr_max = _metadata_float(metadata, "snr_db_max", 18.0)
    target_snr_db = rng.uniform(min(snr_min, snr_max), max(snr_min, snr_max))
    target_noise_rms = signal_rms / (10.0 ** (target_snr_db / 20.0))
    mixed = waveform + (noise_segment.reshape(1, -1) * (target_noise_rms / noise_rms))
    return mixed.astype(np.float32, copy=False), {
        "target_snr_db": round(float(target_snr_db), 4),
        "noise_path": noise_path,
        "category": metadata.get("category", ""),
    }


def _apply_reverb(
    waveform: NDArray[np.float32],
    *,
    project_root: Path,
    metadata: dict[str, object],
    sample_rate_hz: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    del sample_rate_hz
    raw_paths = metadata.get("rir_audio_paths", ())
    paths = (
        tuple(str(path) for path in raw_paths if str(path))
        if isinstance(raw_paths, tuple | list)
        else ()
    )
    if not paths:
        return waveform, {"skip_reason": "empty_rir_paths"}
    rir_path = rng.choice(paths)
    rir = _read_mono_audio(project_root / rir_path)
    if rir.size == 0:
        return waveform, {"skip_reason": "empty_rir"}
    peak = max(float(np.max(np.abs(rir))), 1e-8)
    kernel = rir / peak
    convolved = _convolve_channels(waveform, kernel, output_samples=int(waveform.shape[-1]))
    convolved = _match_rms(convolved, reference=waveform)
    return convolved, {
        "rir_path": rir_path,
        "direct_condition": metadata.get("direct_condition", ""),
        "rt60_bucket": metadata.get("rt60_bucket", ""),
    }


def _apply_distance(
    waveform: NDArray[np.float32],
    *,
    project_root: Path,
    metadata: dict[str, object],
    sample_rate_hz: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    del rng
    transformed = waveform
    kernel_path = _metadata_str(metadata, "kernel_audio_path")
    if kernel_path:
        kernel = _read_mono_audio(project_root / kernel_path)
        if kernel.size:
            peak = max(float(np.max(np.abs(kernel))), 1e-8)
            transformed = _convolve_channels(
                transformed,
                kernel / peak,
                output_samples=int(waveform.shape[-1]),
            )
    lowpass_hz = min(_metadata_float(metadata, "lowpass_hz", 5600.0), sample_rate_hz / 2.0 - 50.0)
    transformed = _bandpass_fft(
        transformed,
        sample_rate_hz=sample_rate_hz,
        low_hz=80.0,
        high_hz=lowpass_hz,
    )
    attenuation_db = _metadata_float(metadata, "attenuation_db", 3.0)
    transformed = transformed * (10.0 ** (-attenuation_db / 20.0))
    return _match_rms(transformed, reference=waveform, max_gain_db=3.0), {
        "field": metadata.get("field", ""),
        "lowpass_hz": round(float(lowpass_hz), 3),
        "attenuation_db": round(float(attenuation_db), 3),
        "kernel_audio_path": kernel_path,
    }


def _apply_codec(
    waveform: NDArray[np.float32],
    *,
    metadata: dict[str, object],
    sample_rate_hz: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    family = _metadata_str(metadata, "codec_family") or "codec"
    severity = _metadata_str(metadata, "severity") or ""
    low_hz, high_hz = _codec_band(family=family, severity=severity)
    filtered = _bandpass_fft(
        waveform,
        sample_rate_hz=sample_rate_hz,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    filtered = _random_eq(filtered, sample_rate_hz=sample_rate_hz, rng=rng, family=family)
    if family in {"telephony", "compression"} or severity == "heavy":
        bits = 8 if severity == "heavy" else 10
        filtered = _bit_crush(filtered, bits=bits)
    if severity == "heavy":
        filtered = np.tanh(filtered * 1.25).astype(np.float32, copy=False)
        filtered = _apply_packet_loss(filtered, sample_rate_hz=sample_rate_hz, rng=rng)
    return filtered, {
        "codec_family": family,
        "codec_name": metadata.get("codec_name", ""),
        "low_hz": low_hz,
        "high_hz": high_hz,
        "simulation": "fft_filter_eq_quantize_packet_loss",
    }


def _apply_silence(
    waveform: NDArray[np.float32],
    *,
    base_config: SilenceAugmentationConfig,
    metadata: dict[str, object],
    sample_rate_hz: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    scale = _metadata_float(metadata, "scale", 1.0)
    leading = _sample_seconds(base_config.max_leading_padding_seconds * scale, rng=rng)
    trailing = _sample_seconds(base_config.max_trailing_padding_seconds * scale, rng=rng)
    leading_frames = int(round(leading * sample_rate_hz))
    trailing_frames = int(round(trailing * sample_rate_hz))
    pieces: list[NDArray[np.float32]] = []
    if leading_frames > 0:
        pieces.append(np.zeros((waveform.shape[0], leading_frames), dtype=np.float32))
    body = waveform
    inserted = 0
    dropped = 0
    max_pauses = max(0, int(round(base_config.max_inserted_pauses * scale)))
    if max_pauses > 0 and body.shape[-1] > sample_rate_hz:
        body, inserted = _insert_random_pauses(
            body,
            sample_rate_hz=sample_rate_hz,
            max_pauses=max_pauses,
            min_seconds=base_config.min_inserted_pause_seconds,
            max_seconds=base_config.max_inserted_pause_seconds * max(scale, 0.25),
            rng=rng,
        )
    if scale >= 0.7:
        body, dropped = _vad_drop(
            body,
            sample_rate_hz=sample_rate_hz,
            max_drops=1 if scale < 1.0 else 2,
            rng=rng,
        )
    pieces.append(body)
    if trailing_frames > 0:
        pieces.append(np.zeros((waveform.shape[0], trailing_frames), dtype=np.float32))
    return np.concatenate(pieces, axis=-1), {
        "leading_padding_seconds": round(float(leading), 4),
        "trailing_padding_seconds": round(float(trailing), 4),
        "inserted_pause_count": inserted,
        "vad_drop_count": dropped,
    }


def _apply_speed(
    waveform: NDArray[np.float32],
    *,
    metadata: dict[str, object],
) -> tuple[NDArray[np.float32], dict[str, object]]:
    factor = _metadata_float(metadata, "speed_factor", 1.0)
    if math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1e-6):
        return waveform, {"speed_factor": 1.0}
    source_samples = int(waveform.shape[-1])
    target_samples = max(1, int(round(source_samples / factor)))
    source_positions = np.linspace(0.0, 1.0, num=source_samples, dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_samples, dtype=np.float32)
    channels = [
        np.interp(target_positions, source_positions, channel).astype(np.float32, copy=False)
        for channel in waveform
    ]
    return np.stack(channels, axis=0), {
        "speed_factor": round(float(factor), 4),
        "source_samples": source_samples,
        "target_samples": target_samples,
    }


def _coerce_audio(waveform: Any) -> NDArray[np.float32]:
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[-1] == 0:
        raise ValueError("augmentation waveform must be shaped [channels, samples]")
    return array


def _read_mono_audio(path: Path) -> NDArray[np.float32]:
    waveform, _ = read_audio_file(path)
    array = _coerce_audio(waveform)
    return array.mean(axis=0, dtype=np.float32)


def _sample_or_tile(
    waveform: NDArray[np.float32],
    *,
    target_samples: int,
    rng: random.Random,
) -> NDArray[np.float32]:
    if waveform.size >= target_samples:
        max_start = waveform.size - target_samples
        start = 0 if max_start <= 0 else rng.randint(0, max_start)
        return waveform[start : start + target_samples]
    repeats = int(math.ceil(target_samples / max(waveform.size, 1)))
    return np.tile(waveform, repeats)[:target_samples].astype(np.float32, copy=False)


def _convolve_channels(
    waveform: NDArray[np.float32],
    kernel: NDArray[np.float32],
    *,
    output_samples: int,
) -> NDArray[np.float32]:
    convolved = [fft_convolve_1d(channel, kernel)[:output_samples] for channel in waveform]
    return np.stack(convolved, axis=0).astype(np.float32, copy=False)


def _match_rms(
    waveform: NDArray[np.float32],
    *,
    reference: NDArray[np.float32],
    max_gain_db: float = 6.0,
) -> NDArray[np.float32]:
    source = _rms(waveform)
    target = _rms(reference)
    if source <= 1e-8 or target <= 1e-8:
        return waveform.astype(np.float32, copy=False)
    gain = min(target / source, 10.0 ** (max_gain_db / 20.0))
    return (waveform * gain).astype(np.float32, copy=False)


def _bandpass_fft(
    waveform: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    low_hz: float,
    high_hz: float,
) -> NDArray[np.float32]:
    sample_count = int(waveform.shape[-1])
    freqs = np.fft.rfftfreq(sample_count, d=1.0 / float(sample_rate_hz))
    mask = (freqs >= max(0.0, low_hz)) & (freqs <= min(high_hz, sample_rate_hz / 2.0))
    spectrum = np.fft.rfft(waveform.astype(np.float64), axis=-1)
    spectrum *= mask.reshape(1, -1)
    return np.fft.irfft(spectrum, n=sample_count, axis=-1).astype(np.float32, copy=False)


def _random_eq(
    waveform: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    rng: random.Random,
    family: str,
) -> NDArray[np.float32]:
    sample_count = int(waveform.shape[-1])
    freqs = np.fft.rfftfreq(sample_count, d=1.0 / float(sample_rate_hz))
    max_gain = 5.0 if family in {"channel", "telephony"} else 2.5
    anchor_freqs = np.asarray([0.0, 300.0, 1000.0, 2500.0, 5000.0, sample_rate_hz / 2.0])
    anchor_db = np.asarray([rng.uniform(-max_gain, max_gain) for _ in anchor_freqs])
    gains = 10.0 ** (np.interp(freqs, anchor_freqs, anchor_db) / 20.0)
    spectrum = np.fft.rfft(waveform.astype(np.float64), axis=-1)
    spectrum *= gains.reshape(1, -1)
    return np.fft.irfft(spectrum, n=sample_count, axis=-1).astype(np.float32, copy=False)


def _bit_crush(waveform: NDArray[np.float32], *, bits: int) -> NDArray[np.float32]:
    levels = float(2 ** max(2, bits - 1))
    return (np.round(np.clip(waveform, -1.0, 1.0) * levels) / levels).astype(
        np.float32,
        copy=False,
    )


def _apply_packet_loss(
    waveform: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    rng: random.Random,
) -> NDArray[np.float32]:
    output = waveform.copy()
    drop_count = rng.randint(1, 3)
    total = int(output.shape[-1])
    for _ in range(drop_count):
        span = rng.randint(int(0.02 * sample_rate_hz), int(0.08 * sample_rate_hz))
        if total <= span:
            continue
        start = rng.randint(0, total - span)
        output[:, start : start + span] *= rng.uniform(0.0, 0.15)
    return output


def _insert_random_pauses(
    waveform: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    max_pauses: int,
    min_seconds: float,
    max_seconds: float,
    rng: random.Random,
) -> tuple[NDArray[np.float32], int]:
    total = int(waveform.shape[-1])
    pause_count = rng.randint(0, max_pauses)
    if pause_count == 0:
        return waveform, 0
    points = sorted(rng.randint(0, total) for _ in range(pause_count))
    pieces: list[NDArray[np.float32]] = []
    cursor = 0
    inserted = 0
    for point in points:
        pieces.append(waveform[:, cursor:point])
        pause_seconds = rng.uniform(min_seconds, max(max_seconds, min_seconds))
        pause_frames = max(1, int(round(pause_seconds * sample_rate_hz)))
        pieces.append(np.zeros((waveform.shape[0], pause_frames), dtype=np.float32))
        cursor = point
        inserted += 1
    pieces.append(waveform[:, cursor:])
    return np.concatenate(pieces, axis=-1), inserted


def _vad_drop(
    waveform: NDArray[np.float32],
    *,
    sample_rate_hz: int,
    max_drops: int,
    rng: random.Random,
) -> tuple[NDArray[np.float32], int]:
    output = waveform.copy()
    total = int(output.shape[-1])
    drop_count = rng.randint(0, max_drops)
    for _ in range(drop_count):
        span = rng.randint(int(0.06 * sample_rate_hz), int(0.18 * sample_rate_hz))
        if total <= span:
            continue
        start = rng.randint(0, total - span)
        output[:, start : start + span] = 0.0
    return output, drop_count


def _codec_band(*, family: str, severity: str) -> tuple[float, float]:
    if family == "telephony":
        return 300.0, 3400.0 if severity != "heavy" else 3200.0
    if family == "compression":
        return 180.0, 5400.0
    if family == "channel":
        return 90.0, 7600.0 if severity != "heavy" else 5000.0
    if family == "voip":
        return 140.0, 6800.0
    return 120.0, 7200.0


def _rms(waveform: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32), dtype=np.float32)))


def _sample_seconds(max_seconds: float, *, rng: random.Random) -> float:
    if max_seconds <= 0.0:
        return 0.0
    return rng.uniform(0.0, max_seconds)


def _read_jsonl_records(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return tuple(rows)


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _coerce_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _coerce_severity(value: object) -> AugmentationSeverity:
    normalized = _coerce_str(value).lower()
    if normalized in {"light", "medium", "heavy"}:
        return cast(AugmentationSeverity, normalized)
    return "medium"


def _severity_from_direct_condition(value: str) -> AugmentationSeverity:
    return cast(
        AugmentationSeverity,
        {"high": "light", "medium": "medium", "low": "heavy"}.get(value.lower(), "medium"),
    )


def _severity_from_distance_field(value: str) -> AugmentationSeverity:
    return cast(
        AugmentationSeverity,
        {"near": "light", "mid": "medium", "far": "heavy"}.get(value.lower(), "medium"),
    )


def _has_effective_silence_config(config: SilenceAugmentationConfig) -> bool:
    return any(
        (
            config.max_leading_padding_seconds > 0.0,
            config.max_trailing_padding_seconds > 0.0,
            config.max_inserted_pauses > 0,
            not math.isclose(config.pause_ratio_min, 1.0),
            not math.isclose(config.pause_ratio_max, 1.0),
        )
    )


def _metadata_str(metadata: dict[str, object], key: str) -> str:
    return _coerce_str(metadata.get(key))


def _metadata_float(metadata: dict[str, object], key: str, default: float) -> float:
    return _coerce_float(metadata.get(key), default)


__all__ = [
    "TrainingAugmentationRuntime",
    "build_augmentation_catalog_from_manifests",
]
