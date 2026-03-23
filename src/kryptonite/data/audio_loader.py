"""Audio loading helpers for direct files and manifests-backed corpora."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kryptonite.config import NormalizationConfig
from kryptonite.deployment import resolve_project_path

from .audio_io import AudioFileInfo, inspect_audio_file, read_audio_file, resample_waveform
from .schema import ManifestRow


@dataclass(frozen=True, slots=True)
class AudioLoadRequest:
    target_sample_rate_hz: int | None = None
    target_channels: int | None = None
    start_seconds: float = 0.0
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.target_sample_rate_hz is not None and self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive when provided")
        if self.target_channels is not None and self.target_channels <= 0:
            raise ValueError("target_channels must be positive when provided")
        if self.start_seconds < 0.0:
            raise ValueError("start_seconds must be non-negative")
        if self.duration_seconds is not None and self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive when provided")

    @classmethod
    def from_config(
        cls,
        config: NormalizationConfig,
        *,
        start_seconds: float = 0.0,
        duration_seconds: float | None = None,
    ) -> AudioLoadRequest:
        return cls(
            target_sample_rate_hz=config.target_sample_rate_hz,
            target_channels=config.target_channels,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )


@dataclass(frozen=True, slots=True)
class LoadedAudio:
    configured_path: str
    resolved_path: str
    waveform: Any
    sample_rate_hz: int
    num_channels: int
    frame_count: int
    duration_seconds: float
    source_format: str
    source_subtype: str | None
    source_sample_rate_hz: int
    source_num_channels: int
    source_frame_count: int
    source_duration_seconds: float
    start_seconds: float
    requested_duration_seconds: float | None
    resampled: bool
    downmixed: bool


@dataclass(frozen=True, slots=True)
class LoadedManifestAudio:
    manifest_path: str | None
    line_number: int | None
    row: ManifestRow
    audio: LoadedAudio


def load_audio(
    path: Path | str,
    *,
    project_root: Path | str = ".",
    request: AudioLoadRequest | None = None,
) -> LoadedAudio:
    active_request = request or AudioLoadRequest()
    project_root_path = resolve_project_path(str(project_root), ".")
    configured_path = str(path)
    resolved_path = resolve_project_path(str(project_root_path), configured_path)

    source_info = _inspect_for_window(resolved_path)
    source_frame_offset = _seconds_to_frame_offset(
        active_request.start_seconds,
        sample_rate_hz=source_info.sample_rate_hz,
    )
    requested_source_frames = _seconds_to_frame_count(
        active_request.duration_seconds,
        sample_rate_hz=source_info.sample_rate_hz,
    )

    waveform, read_info = read_audio_file(
        resolved_path,
        frame_offset=source_frame_offset,
        frame_count=requested_source_frames,
    )
    if waveform.ndim != 2 or int(waveform.shape[-1]) == 0:
        raise ValueError(f"Decoded audio window is empty for {configured_path!r}")

    downmixed = False
    if active_request.target_channels is not None:
        waveform, downmixed = _apply_channel_policy(
            waveform,
            target_channels=active_request.target_channels,
        )

    sample_rate_hz = read_info.sample_rate_hz
    resampled = False
    if (
        active_request.target_sample_rate_hz is not None
        and active_request.target_sample_rate_hz != sample_rate_hz
    ):
        waveform = resample_waveform(
            waveform,
            orig_freq=sample_rate_hz,
            new_freq=active_request.target_sample_rate_hz,
        )
        sample_rate_hz = active_request.target_sample_rate_hz
        resampled = True

    frame_count = int(waveform.shape[-1])
    num_channels = int(waveform.shape[0])
    return LoadedAudio(
        configured_path=configured_path,
        resolved_path=str(resolved_path),
        waveform=waveform,
        sample_rate_hz=sample_rate_hz,
        num_channels=num_channels,
        frame_count=frame_count,
        duration_seconds=round(float(frame_count) / float(sample_rate_hz), 6),
        source_format=read_info.format,
        source_subtype=read_info.subtype,
        source_sample_rate_hz=read_info.sample_rate_hz,
        source_num_channels=read_info.num_channels,
        source_frame_count=read_info.frame_count,
        source_duration_seconds=read_info.duration_seconds,
        start_seconds=active_request.start_seconds,
        requested_duration_seconds=active_request.duration_seconds,
        resampled=resampled,
        downmixed=downmixed,
    )


def load_manifest_audio(
    row: ManifestRow | Mapping[str, object],
    *,
    project_root: Path | str = ".",
    request: AudioLoadRequest | None = None,
    manifest_path: Path | str | None = None,
    line_number: int | None = None,
) -> LoadedManifestAudio:
    manifest_row = row if isinstance(row, ManifestRow) else ManifestRow.from_mapping(row)
    audio = load_audio(
        manifest_row.audio_path,
        project_root=project_root,
        request=request,
    )
    project_root_path = resolve_project_path(str(project_root), ".")
    return LoadedManifestAudio(
        manifest_path=(
            None
            if manifest_path is None
            else _relative_to_project(
                resolve_project_path(str(project_root_path), str(manifest_path)),
                project_root=project_root_path,
            )
        ),
        line_number=line_number,
        row=manifest_row,
        audio=audio,
    )


def iter_manifest_audio(
    manifest_path: Path | str,
    *,
    project_root: Path | str = ".",
    request: AudioLoadRequest | None = None,
) -> Iterator[LoadedManifestAudio]:
    project_root_path = resolve_project_path(str(project_root), ".")
    manifest_file = resolve_project_path(str(project_root_path), str(manifest_path))
    manifest_location = _relative_to_project(manifest_file, project_root=project_root_path)

    for line_number, raw_line in enumerate(manifest_file.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object JSONL rows in {manifest_location}:{line_number}")
        yield load_manifest_audio(
            payload,
            project_root=project_root_path,
            request=request,
            manifest_path=manifest_location,
            line_number=line_number,
        )


def _inspect_for_window(path: Path) -> AudioFileInfo:
    return inspect_audio_file(path)


def _apply_channel_policy(waveform: Any, *, target_channels: int) -> tuple[Any, bool]:
    current_channels = int(waveform.shape[0])
    if current_channels == target_channels:
        return waveform, False
    if target_channels == 1:
        return waveform.mean(axis=0, keepdims=True, dtype="float32"), True
    raise ValueError(
        f"Unsupported channel conversion: source has {current_channels} channels, "
        f"target requested {target_channels}"
    )


def _seconds_to_frame_offset(start_seconds: float, *, sample_rate_hz: int) -> int:
    return int(start_seconds * float(sample_rate_hz))


def _seconds_to_frame_count(duration_seconds: float | None, *, sample_rate_hz: int) -> int | None:
    if duration_seconds is None:
        return None
    return max(1, math.ceil(duration_seconds * float(sample_rate_hz)))


def _relative_to_project(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())
