"""FFmpeg helpers for deterministic codec/channel simulation previews."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .models import CodecSimulationPreset, FFmpegToolMetadata


@dataclass(frozen=True, slots=True)
class CodecCommandTrace:
    encode_command: str | None
    decode_command: str


class CodecSimulationError(RuntimeError):
    """Raised when an FFmpeg codec-simulation command fails."""


def inspect_ffmpeg_tools(*, ffmpeg_path: str, ffprobe_path: str) -> FFmpegToolMetadata:
    ffmpeg_completed, ffmpeg_error = _run_metadata_command([ffmpeg_path, "-version"])
    ffprobe_completed, ffprobe_error = _run_metadata_command([ffprobe_path, "-version"])

    ffmpeg_lines = (
        ffmpeg_completed.stdout.splitlines()
        if ffmpeg_completed is not None and ffmpeg_completed.stdout
        else []
    )
    ffprobe_lines = (
        ffprobe_completed.stdout.splitlines()
        if ffprobe_completed is not None and ffprobe_completed.stdout
        else []
    )
    configuration = next(
        (
            line.removeprefix("configuration: ").strip()
            for line in ffmpeg_lines
            if line.startswith("configuration: ")
        ),
        None,
    )
    return FFmpegToolMetadata(
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        ffmpeg_available=ffmpeg_completed is not None,
        ffprobe_available=ffprobe_completed is not None,
        version_line=ffmpeg_lines[0] if ffmpeg_lines else None,
        configuration=configuration,
        ffprobe_version_line=ffprobe_lines[0] if ffprobe_lines else None,
        ffmpeg_error=ffmpeg_error,
        ffprobe_error=ffprobe_error,
    )


def apply_codec_preset(
    *,
    input_path: Path,
    output_path: Path,
    preset: CodecSimulationPreset,
    final_sample_rate_hz: int,
    ffmpeg_path: str,
) -> CodecCommandTrace:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pre_filter_graph = ",".join(preset.filters)
    post_filter_graph = ",".join(preset.post_filters)
    if not preset.uses_codec_stage:
        command = _build_direct_command(
            ffmpeg_path=ffmpeg_path,
            input_path=input_path,
            output_path=output_path,
            filter_graph=_merge_filter_graphs(pre_filter_graph, post_filter_graph),
            final_sample_rate_hz=final_sample_rate_hz,
        )
        _run_command(command)
        return CodecCommandTrace(
            encode_command=None,
            decode_command=shlex.join(command),
        )

    with TemporaryDirectory() as tmpdir:
        encoded_path = Path(tmpdir) / f"{preset.id}.{preset.container_extension.lstrip('.')}"
        encode_command = _build_encode_command(
            ffmpeg_path=ffmpeg_path,
            input_path=input_path,
            encoded_path=encoded_path,
            preset=preset,
            filter_graph=pre_filter_graph,
        )
        _run_command(encode_command)

        decode_command = _build_decode_command(
            ffmpeg_path=ffmpeg_path,
            encoded_path=encoded_path,
            output_path=output_path,
            filter_graph=post_filter_graph,
            final_sample_rate_hz=final_sample_rate_hz,
        )
        _run_command(decode_command)
        return CodecCommandTrace(
            encode_command=shlex.join(encode_command),
            decode_command=shlex.join(decode_command),
        )


def _build_direct_command(
    *,
    ffmpeg_path: str,
    input_path: Path,
    output_path: Path,
    filter_graph: str,
    final_sample_rate_hz: int,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
    ]
    if filter_graph:
        command.extend(["-af", filter_graph])
    command.extend(
        [
            "-ar",
            str(final_sample_rate_hz),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return command


def _build_encode_command(
    *,
    ffmpeg_path: str,
    input_path: Path,
    encoded_path: Path,
    preset: CodecSimulationPreset,
    filter_graph: str,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
    ]
    if filter_graph:
        command.extend(["-af", filter_graph])
    if preset.encode_sample_rate_hz is not None:
        command.extend(["-ar", str(preset.encode_sample_rate_hz)])
    command.extend(["-ac", "1", "-c:a", str(preset.codec_name)])
    if preset.encode_bitrate is not None:
        command.extend(["-b:a", preset.encode_bitrate])
    for option in preset.ffmpeg_options:
        key, value = option.split("=", 1)
        command.extend([f"-{key}", value])
    command.append(str(encoded_path))
    return command


def _build_decode_command(
    *,
    ffmpeg_path: str,
    encoded_path: Path,
    output_path: Path,
    filter_graph: str,
    final_sample_rate_hz: int,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(encoded_path),
    ]
    if filter_graph:
        command.extend(["-af", filter_graph])
    command.extend(
        [
            "-ar",
            str(final_sample_rate_hz),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return command


def _merge_filter_graphs(first: str, second: str) -> str:
    if first and second:
        return f"{first},{second}"
    return first or second


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode == 0:
        return
    details = _format_process_error(completed)
    raise CodecSimulationError(
        details or f"FFmpeg command failed with exit code {completed.returncode}."
    )


def _run_metadata_command(
    command: list[str],
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return None, _format_process_error(completed)
    return completed, None


def _format_process_error(completed: subprocess.CompletedProcess[str]) -> str | None:
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    if stderr:
        return stderr
    if stdout:
        return stdout
    return None


__all__ = [
    "CodecCommandTrace",
    "CodecSimulationError",
    "apply_codec_preset",
    "inspect_ffmpeg_tools",
]
