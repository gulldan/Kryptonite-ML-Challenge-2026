"""Audio stats, frozen sampling, and synthetic robustness distortions."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from kryptonite.data.audio_io import read_audio_file, resample_waveform, write_audio_file
from kryptonite.data.convolution import fft_convolve_1d

SILENCE_THRESHOLD = 1e-3
CLIPPING_THRESHOLD = 0.98
TARGET_SAMPLE_RATE_HZ = 16_000


@dataclass(frozen=True, slots=True)
class DistortionCondition:
    name: str
    family: str
    severity: str
    parameters: dict[str, float | int | str]


def default_distortion_conditions() -> tuple[DistortionCondition, ...]:
    return (
        DistortionCondition(
            name="clean",
            family="clean",
            severity="clean",
            parameters={},
        ),
        DistortionCondition(
            name="noise_light",
            family="additive_noise",
            severity="light",
            parameters={"snr_db": 22.0, "color": "white"},
        ),
        DistortionCondition(
            name="noise_medium",
            family="additive_noise",
            severity="medium",
            parameters={"snr_db": 14.0, "color": "pink"},
        ),
        DistortionCondition(
            name="noise_heavy",
            family="additive_noise",
            severity="heavy",
            parameters={"snr_db": 7.0, "color": "brown"},
        ),
        DistortionCondition(
            name="reverb_light",
            family="reverb",
            severity="light",
            parameters={"rt60_s": 0.18, "direct_gain": 0.75},
        ),
        DistortionCondition(
            name="reverb_medium",
            family="reverb",
            severity="medium",
            parameters={"rt60_s": 0.42, "direct_gain": 0.45},
        ),
        DistortionCondition(
            name="reverb_heavy",
            family="reverb",
            severity="heavy",
            parameters={"rt60_s": 0.78, "direct_gain": 0.22},
        ),
        DistortionCondition(
            name="codec_light",
            family="codec_bandwidth",
            severity="light",
            parameters={"low_hz": 80.0, "high_hz": 6800.0, "bits": 16},
        ),
        DistortionCondition(
            name="codec_medium",
            family="codec_bandwidth",
            severity="medium",
            parameters={"low_hz": 140.0, "high_hz": 4200.0, "bits": 10},
        ),
        DistortionCondition(
            name="codec_heavy",
            family="codec_bandwidth",
            severity="heavy",
            parameters={"low_hz": 220.0, "high_hz": 3200.0, "bits": 8},
        ),
        DistortionCondition(
            name="level_light",
            family="level_clipping",
            severity="light",
            parameters={"gain_db": 2.5, "clip_threshold": 0.98},
        ),
        DistortionCondition(
            name="level_medium",
            family="level_clipping",
            severity="medium",
            parameters={"gain_db": 6.0, "clip_threshold": 0.88},
        ),
        DistortionCondition(
            name="level_heavy",
            family="level_clipping",
            severity="heavy",
            parameters={"gain_db": 12.0, "clip_threshold": 0.72},
        ),
    )


def resolve_audio_path(filepath: str, *, data_root: Path) -> Path:
    candidate = Path(filepath)
    if candidate.is_absolute():
        return candidate
    normalized = filepath.replace("\\", "/")
    for prefix in ("datasets/Для участников/", "./datasets/Для участников/"):
        if normalized.startswith(prefix):
            return data_root / normalized.removeprefix(prefix)
    return data_root / candidate


def collect_audio_stats(
    *,
    manifest_path: Path,
    data_root: Path,
    output_path: Path,
    workers: int,
) -> pl.DataFrame:
    if output_path.is_file():
        return pl.read_csv(output_path)

    manifest = pl.read_csv(manifest_path)
    rows = manifest.to_dicts()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        stats_rows = list(
            executor.map(
                lambda row: _audio_stats_for_row(row, data_root=data_root),
                rows,
            )
        )
    stats = pl.DataFrame(stats_rows).sort(["speaker_id", "utterance_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.write_csv(output_path)
    return stats


def build_frozen_clean_subset(
    *,
    stats: pl.DataFrame,
    target_size: int,
    seed: int,
) -> pl.DataFrame:
    if target_size <= 0:
        raise ValueError("target_size must be positive.")
    counts = stats.group_by("speaker_id").len().sort("speaker_id")
    speaker_count = counts.height
    minimum_per_speaker = min(12, target_size // max(1, speaker_count))
    quotas = _speaker_quotas(
        counts=counts,
        target_size=target_size,
        minimum_per_speaker=minimum_per_speaker,
    )
    duration_bins = _bin_series(stats["duration_seconds"].to_numpy(), quantiles=(0.25, 0.5, 0.75))
    rms_bins = _bin_series(stats["rms_dbfs"].to_numpy(), quantiles=(0.3333, 0.6667))
    silence_bins = _bin_series(stats["silence_ratio"].to_numpy(), quantiles=(0.5,))
    clipping_bins = _bin_series(stats["clipping_ratio"].to_numpy(), quantiles=(0.5,))
    enriched = stats.with_columns(
        pl.Series("duration_bin", duration_bins),
        pl.Series("rms_bin", rms_bins),
        pl.Series("silence_bin", silence_bins),
        pl.Series("clipping_bin", clipping_bins),
    )
    selected_rows: list[dict[str, Any]] = []
    for speaker_id in sorted(quotas):
        speaker_rows = enriched.filter(pl.col("speaker_id") == speaker_id).to_dicts()
        quota = quotas[speaker_id]
        selected_rows.extend(
            _select_rows_for_speaker(
                speaker_rows=speaker_rows,
                quota=quota,
                seed=seed,
            )
        )
    if len(selected_rows) != target_size:
        raise RuntimeError(
            f"Frozen clean subset has {len(selected_rows)} rows, expected {target_size}."
        )
    frozen = pl.DataFrame(selected_rows).with_row_index(name="clean_index").sort("clean_index")
    return frozen


def build_distorted_plan(
    *,
    clean_manifest: pl.DataFrame,
    runtime_root: Path,
    conditions: tuple[DistortionCondition, ...],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in clean_manifest.to_dicts():
        item_id = str(row["item_id"])
        for condition in conditions:
            if condition.name == "clean":
                continue
            output_path = (
                runtime_root
                / "cache"
                / "distortions"
                / condition.family
                / condition.severity
                / f"{item_id}.wav"
            )
            rows.append(
                {
                    "condition": condition.name,
                    "family": condition.family,
                    "severity": condition.severity,
                    "source_item_id": item_id,
                    "speaker_id": row["speaker_id"],
                    "source_filepath": row["filepath"],
                    "filepath": str(output_path),
                    "resolved_path": str(output_path),
                    "condition_parameters": json.dumps(condition.parameters, sort_keys=True),
                }
            )
    return pl.DataFrame(rows).sort(["condition", "source_item_id"])


def materialize_condition_audio(
    *,
    clean_manifest: pl.DataFrame,
    condition: DistortionCondition,
    runtime_root: Path,
    source_data_root: Path,
    workers: int,
    seed: int,
    manifest_path: Path,
) -> pl.DataFrame:
    if manifest_path.is_file():
        manifest = pl.read_csv(manifest_path)
        pending = [row for row in manifest.to_dicts() if not Path(str(row["filepath"])).is_file()]
        if not pending:
            return manifest
    else:
        rows: list[dict[str, Any]] = []
        for row in clean_manifest.to_dicts():
            item_id = str(row["item_id"])
            output_path = (
                runtime_root
                / "cache"
                / "distortions"
                / condition.family
                / condition.severity
                / f"{item_id}.wav"
            )
            rows.append(
                {
                    "clean_index": int(row["clean_index"]),
                    "item_id": item_id,
                    "speaker_id": row["speaker_id"],
                    "filepath": str(output_path),
                    "resolved_path": str(output_path),
                    "family": condition.family,
                    "severity": condition.severity,
                    "condition": condition.name,
                    "condition_parameters": json.dumps(condition.parameters, sort_keys=True),
                }
            )
        manifest = pl.DataFrame(rows).sort("clean_index")
        pending = manifest.to_dicts()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(
            executor.map(
                lambda row: _materialize_audio_row(
                    row=row,
                    clean_manifest=clean_manifest,
                    condition=condition,
                    source_data_root=source_data_root,
                    seed=seed,
                ),
                pending,
            )
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(manifest_path)
    return manifest


def _audio_stats_for_row(row: dict[str, Any], *, data_root: Path) -> dict[str, Any]:
    audio_path = resolve_audio_path(str(row["audio_path"]), data_root=data_root)
    waveform, info = read_audio_file(audio_path)
    mono = waveform.mean(axis=0, dtype=np.float32)
    peak_abs = float(np.max(np.abs(mono), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64), dtype=np.float64)))
    silence_ratio = float(np.mean(np.abs(mono) <= SILENCE_THRESHOLD))
    clipping_ratio = float(np.mean(np.abs(mono) >= CLIPPING_THRESHOLD))
    item_id = f"{row['speaker_id']}::{row['utterance_id']}"
    return {
        **row,
        "item_id": item_id,
        "filepath": _normalized_manifest_filepath(str(row["audio_path"])),
        "resolved_path": str(audio_path),
        "sample_rate_hz": info.sample_rate_hz,
        "duration_seconds": info.duration_seconds,
        "peak_abs": round(peak_abs, 8),
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-8)), 6),
        "silence_ratio": round(silence_ratio, 8),
        "clipping_ratio": round(clipping_ratio, 8),
    }


def _speaker_quotas(
    *,
    counts: pl.DataFrame,
    target_size: int,
    minimum_per_speaker: int,
) -> dict[str, int]:
    quotas: dict[str, int] = {}
    capacities: dict[str, int] = {}
    for row in counts.to_dicts():
        speaker_id = str(row["speaker_id"])
        capacity = int(row["len"])
        capacities[speaker_id] = capacity
        quotas[speaker_id] = min(capacity, minimum_per_speaker)
    allocated = sum(quotas.values())
    remaining = target_size - allocated
    if remaining < 0:
        raise ValueError("minimum_per_speaker is too large for target_size.")
    if remaining == 0:
        return quotas

    extra_capacity = {
        speaker_id: max(0, capacities[speaker_id] - quotas[speaker_id]) for speaker_id in quotas
    }
    weights = {speaker_id: math.sqrt(capacity) for speaker_id, capacity in extra_capacity.items()}
    weight_sum = sum(weights.values())
    fractional: list[tuple[float, str]] = []
    for speaker_id in sorted(quotas):
        if extra_capacity[speaker_id] <= 0 or weight_sum <= 0.0:
            continue
        raw = remaining * (weights[speaker_id] / weight_sum)
        whole = min(extra_capacity[speaker_id], int(math.floor(raw)))
        quotas[speaker_id] += whole
        fractional.append((raw - whole, speaker_id))
    still_needed = target_size - sum(quotas.values())
    for _, speaker_id in sorted(fractional, reverse=True):
        if still_needed <= 0:
            break
        if quotas[speaker_id] >= capacities[speaker_id]:
            continue
        quotas[speaker_id] += 1
        still_needed -= 1
    if sum(quotas.values()) != target_size:
        raise RuntimeError("Failed to resolve speaker quotas to the requested target size.")
    return quotas


def _select_rows_for_speaker(
    *,
    speaker_rows: list[dict[str, Any]],
    quota: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row in speaker_rows:
        group_key = (
            int(row["duration_bin"]),
            int(row["rms_bin"]),
            int(row["silence_bin"]),
            int(row["clipping_bin"]),
        )
        stable_rank = _stable_rank(seed=seed, text=str(row["item_id"]))
        grouped[group_key].append((stable_rank, row))
    ordered_groups: list[tuple[tuple[int, int, int, int], list[dict[str, Any]]]] = []
    for group_key, entries in grouped.items():
        ordered_rows = [row for _, row in sorted(entries, key=lambda item: item[0])]
        ordered_groups.append((group_key, ordered_rows))
    ordered_groups.sort(key=lambda item: _stable_rank(seed=seed, text=json.dumps(item[0])))

    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < quota:
        made_progress = False
        for _, rows in ordered_groups:
            if cursor < len(rows):
                selected.append(rows[cursor])
                if len(selected) == quota:
                    break
                made_progress = True
        if not made_progress:
            break
        cursor += 1
    if len(selected) != quota:
        raise RuntimeError(
            f"Failed to select {quota} rows for speaker={speaker_rows[0]['speaker_id']}."
        )
    return selected


def _bin_series(values: np.ndarray, *, quantiles: tuple[float, ...]) -> np.ndarray:
    edges = np.quantile(values, quantiles).astype(np.float64)
    unique_edges = np.unique(edges)
    if unique_edges.size == 0:
        return np.zeros(values.shape[0], dtype=np.int16)
    return np.searchsorted(unique_edges, values, side="right").astype(np.int16)


def _stable_rank(*, seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}|{text}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _normalized_manifest_filepath(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/")
    for prefix in ("datasets/Для участников/", "./datasets/Для участников/"):
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix)
    return normalized


def _materialize_audio_row(
    *,
    row: dict[str, Any],
    clean_manifest: pl.DataFrame,
    condition: DistortionCondition,
    source_data_root: Path,
    seed: int,
) -> None:
    output_path = Path(str(row["filepath"]))
    if output_path.is_file():
        return
    source_row = clean_manifest.filter(pl.col("item_id") == str(row["item_id"])).to_dicts()
    if len(source_row) != 1:
        raise RuntimeError(f"Expected exactly one source row for {row['item_id']}.")
    source = source_row[0]
    input_path = resolve_audio_path(str(source["filepath"]), data_root=source_data_root)
    waveform, info = read_audio_file(input_path)
    mono = waveform.mean(axis=0, dtype=np.float32).reshape(1, -1)
    sample_rate_hz = info.sample_rate_hz
    if sample_rate_hz != TARGET_SAMPLE_RATE_HZ:
        mono = cast(
            np.ndarray,
            resample_waveform(mono, orig_freq=sample_rate_hz, new_freq=TARGET_SAMPLE_RATE_HZ),
        )
        sample_rate_hz = TARGET_SAMPLE_RATE_HZ
    seed_value = _stable_rank(seed=seed, text=f"{condition.name}|{source['item_id']}")
    distorted = apply_distortion(
        waveform=mono.astype(np.float32, copy=False),
        sample_rate_hz=sample_rate_hz,
        condition=condition,
        seed_value=seed_value,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_audio_file(
        path=output_path,
        waveform=np.clip(distorted, -1.0, 1.0),
        sample_rate_hz=sample_rate_hz,
        output_format="wav",
        pcm_bits_per_sample=16,
    )


def apply_distortion(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
    condition: DistortionCondition,
    seed_value: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed_value)
    if condition.family == "additive_noise":
        return _apply_additive_noise(
            waveform=waveform,
            snr_db=float(condition.parameters["snr_db"]),
            color=str(condition.parameters["color"]),
            rng=rng,
        )
    if condition.family == "reverb":
        return _apply_reverb(
            waveform=waveform,
            sample_rate_hz=sample_rate_hz,
            rt60_s=float(condition.parameters["rt60_s"]),
            direct_gain=float(condition.parameters["direct_gain"]),
            rng=rng,
        )
    if condition.family == "codec_bandwidth":
        return _apply_codec_bandwidth(
            waveform=waveform,
            sample_rate_hz=sample_rate_hz,
            low_hz=float(condition.parameters["low_hz"]),
            high_hz=float(condition.parameters["high_hz"]),
            bits=int(condition.parameters["bits"]),
            heavy=condition.severity == "heavy",
            rng=rng,
        )
    if condition.family == "level_clipping":
        return _apply_level_clipping(
            waveform=waveform,
            gain_db=float(condition.parameters["gain_db"]),
            clip_threshold=float(condition.parameters["clip_threshold"]),
        )
    raise ValueError(f"Unsupported distortion family: {condition.family!r}")


def _apply_additive_noise(
    *,
    waveform: np.ndarray,
    snr_db: float,
    color: str,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = _colored_noise(waveform.shape[-1], color=color, rng=rng)
    signal_rms = _rms(waveform)
    noise_rms = _rms(noise.reshape(1, -1))
    if signal_rms <= 1e-8 or noise_rms <= 1e-8:
        return waveform.astype(np.float32, copy=False)
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    scaled = noise * (target_noise_rms / noise_rms)
    return (waveform + scaled.reshape(1, -1)).astype(np.float32, copy=False)


def _apply_reverb(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
    rt60_s: float,
    direct_gain: float,
    rng: np.random.Generator,
) -> np.ndarray:
    rir = _synthetic_rir(
        sample_rate_hz=sample_rate_hz,
        rt60_s=rt60_s,
        direct_gain=direct_gain,
        rng=rng,
    )
    convolved = np.stack(
        [
            fft_convolve_1d(channel.astype(np.float32, copy=False), rir)[: waveform.shape[-1]]
            for channel in waveform
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return _match_rms(convolved, reference=waveform)


def _apply_codec_bandwidth(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
    low_hz: float,
    high_hz: float,
    bits: int,
    heavy: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    filtered = _bandpass_fft(
        waveform,
        sample_rate_hz=sample_rate_hz,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    filtered = _bit_crush(filtered, bits=bits)
    if heavy:
        filtered = np.tanh(filtered * 1.18).astype(np.float32, copy=False)
        filtered = _apply_packet_drop(filtered, sample_rate_hz=sample_rate_hz, rng=rng)
    return _match_rms(filtered, reference=waveform)


def _apply_level_clipping(
    *,
    waveform: np.ndarray,
    gain_db: float,
    clip_threshold: float,
) -> np.ndarray:
    amplified = waveform * (10.0 ** (gain_db / 20.0))
    clipped = np.clip(amplified, -clip_threshold, clip_threshold)
    return (clipped / max(clip_threshold, 1e-6)).astype(np.float32, copy=False)


def _colored_noise(sample_count: int, *, color: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(sample_count).astype(np.float32, copy=False)
    if color == "white":
        return white
    spectrum = np.fft.rfft(white.astype(np.float64))
    freqs = np.fft.rfftfreq(sample_count, d=1.0)
    safe_freqs = np.maximum(freqs, 1.0 / max(1, sample_count))
    if color == "pink":
        scale = np.ones_like(safe_freqs)
        scale[1:] = 1.0 / np.sqrt(safe_freqs[1:])
    elif color == "brown":
        scale = np.ones_like(safe_freqs)
        scale[1:] = 1.0 / safe_freqs[1:]
    else:
        raise ValueError(f"Unsupported noise color: {color!r}")
    colored = np.fft.irfft(spectrum * scale, n=sample_count)
    return colored.astype(np.float32, copy=False)


def _synthetic_rir(
    *,
    sample_rate_hz: int,
    rt60_s: float,
    direct_gain: float,
    rng: np.random.Generator,
) -> np.ndarray:
    length = max(int(round(sample_rate_hz * min(max(rt60_s * 1.25, 0.12), 1.4))), 128)
    times = np.arange(length, dtype=np.float64) / float(sample_rate_hz)
    decay = np.exp(-6.90775527898 * times / max(rt60_s, 1e-3))
    noise = rng.standard_normal(length).astype(np.float64, copy=False)
    rir = decay * noise
    rir[0] += direct_gain
    for offset_ms, scale in ((7.5, 0.22), (15.0, 0.15), (27.5, 0.1)):
        offset = int(round(sample_rate_hz * offset_ms / 1000.0))
        if 0 <= offset < length:
            rir[offset] += scale
    peak = max(float(np.max(np.abs(rir), initial=0.0)), 1e-8)
    return (rir / peak).astype(np.float32, copy=False)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64), dtype=np.float64)))


def _match_rms(waveform: np.ndarray, *, reference: np.ndarray) -> np.ndarray:
    reference_rms = _rms(reference)
    waveform_rms = _rms(waveform)
    if reference_rms <= 1e-8 or waveform_rms <= 1e-8:
        return waveform.astype(np.float32, copy=False)
    gain = reference_rms / waveform_rms
    return (waveform * gain).astype(np.float32, copy=False)


def _bandpass_fft(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    sample_count = int(waveform.shape[-1])
    freqs = np.fft.rfftfreq(sample_count, d=1.0 / float(sample_rate_hz))
    mask = (freqs >= max(0.0, low_hz)) & (freqs <= min(high_hz, sample_rate_hz / 2.0))
    spectrum = np.fft.rfft(waveform.astype(np.float64), axis=-1)
    spectrum *= mask.reshape(1, -1)
    return np.fft.irfft(spectrum, n=sample_count, axis=-1).astype(np.float32, copy=False)


def _bit_crush(waveform: np.ndarray, *, bits: int) -> np.ndarray:
    levels = float(2 ** max(2, bits - 1))
    return (np.round(np.clip(waveform, -1.0, 1.0) * levels) / levels).astype(
        np.float32,
        copy=False,
    )


def _apply_packet_drop(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = waveform.copy()
    total = int(output.shape[-1])
    for _ in range(2):
        width = int(rng.integers(int(0.02 * sample_rate_hz), int(0.05 * sample_rate_hz)))
        if total <= width:
            continue
        start = int(rng.integers(0, total - width))
        output[:, start : start + width] *= float(rng.uniform(0.0, 0.15))
    return output.astype(np.float32, copy=False)


__all__ = [
    "DistortionCondition",
    "TARGET_SAMPLE_RATE_HZ",
    "apply_distortion",
    "build_distorted_plan",
    "build_frozen_clean_subset",
    "collect_audio_stats",
    "default_distortion_conditions",
    "materialize_condition_audio",
    "resolve_audio_path",
]
