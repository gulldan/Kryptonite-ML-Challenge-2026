"""Silero VAD v6 ONNX-backed speech trimming for manifest-driven audio loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

import numpy as np

VADMode = Literal["none", "light", "aggressive"]
VADBackend = Literal["silero_vad_v6_onnx"]
VADProvider = Literal["auto", "cpu", "cuda"]

SUPPORTED_VAD_MODES: tuple[VADMode, ...] = ("none", "light", "aggressive")
SUPPORTED_VAD_BACKENDS: tuple[VADBackend, ...] = ("silero_vad_v6_onnx",)
SUPPORTED_VAD_PROVIDERS: tuple[VADProvider, ...] = ("auto", "cpu", "cuda")


@dataclass(frozen=True, slots=True)
class VADSettings:
    mode: VADMode
    backend: VADBackend
    provider: VADProvider
    threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int
    max_speech_duration_s: float

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_VAD_MODES:
            raise ValueError(
                f"Unsupported VAD mode {self.mode!r}; expected one of {SUPPORTED_VAD_MODES}"
            )
        if self.backend not in SUPPORTED_VAD_BACKENDS:
            raise ValueError(
                f"Unsupported VAD backend {self.backend!r}; expected one of "
                f"{SUPPORTED_VAD_BACKENDS}"
            )
        if self.provider not in SUPPORTED_VAD_PROVIDERS:
            raise ValueError(
                f"Unsupported VAD provider {self.provider!r}; expected one of "
                f"{SUPPORTED_VAD_PROVIDERS}"
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be within [0.0, 1.0]")
        for name in ("min_speech_duration_ms", "min_silence_duration_ms", "speech_pad_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_speech_duration_s <= 0.0:
            raise ValueError("max_speech_duration_s must be positive")


@dataclass(frozen=True, slots=True)
class TrimDecision:
    mode: VADMode
    applied: bool
    speech_detected: bool
    reason: str
    original_frame_count: int
    output_frame_count: int
    start_frame: int
    end_frame: int
    leading_trim_frames: int
    trailing_trim_frames: int


class SileroVADOnnxModel:
    """Thin ONNX Runtime wrapper around the official Silero VAD v6 model asset."""

    def __init__(self, *, model_path: str, providers: tuple[str, ...]) -> None:
        onnxruntime = _import_onnxruntime()

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self._torch = _import_torch()
        self.providers = providers
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=list(providers),
            sess_options=opts,
        )
        self.sample_rates = [8000, 16000]
        self.reset_states()

    def _validate_input(self, audio: Any, sampling_rate: int) -> tuple[Any, int]:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {audio.dim()}")

        if sampling_rate != 16000 and (sampling_rate % 16000 == 0):
            step = sampling_rate // 16000
            audio = audio[:, ::step]
            sampling_rate = 16000

        if sampling_rate not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or a multiple of 16000)"
            )
        if sampling_rate / audio.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")
        return audio, sampling_rate

    def reset_states(self, batch_size: int = 1) -> None:
        torch = self._torch
        self._state = torch.zeros((2, batch_size, 128), dtype=torch.float32)
        self._context = torch.zeros((0,), dtype=torch.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, audio: Any, sampling_rate: int) -> Any:
        torch = self._torch

        audio, sampling_rate = self._validate_input(audio, sampling_rate)
        num_samples = 512 if sampling_rate == 16000 else 256
        if audio.shape[-1] != num_samples:
            raise ValueError(
                "Provided number of samples is "
                f"{audio.shape[-1]} (supported values: 256 for 8000 Hz, 512 for 16000 Hz)"
            )

        batch_size = audio.shape[0]
        context_size = 64 if sampling_rate == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and self._last_sr != sampling_rate:
            self.reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = torch.zeros((batch_size, context_size), dtype=torch.float32)

        audio = torch.cat([self._context, audio], dim=1)
        ort_inputs = {
            "input": audio.cpu().numpy(),
            "state": self._state.cpu().numpy(),
            "sr": np.asarray(sampling_rate, dtype=np.int64),
        }
        output, state = self.session.run(None, ort_inputs)

        self._state = torch.from_numpy(state).to(dtype=torch.float32)
        self._context = audio[..., -context_size:]
        self._last_sr = sampling_rate
        self._last_batch_size = batch_size
        return torch.from_numpy(output)


def resolve_vad_settings(
    mode: str,
    *,
    backend: str = "silero_vad_v6_onnx",
    provider: str = "auto",
) -> VADSettings:
    normalized_mode = mode.lower()
    normalized_backend = backend.lower()
    normalized_provider = provider.lower()
    if normalized_mode == "none":
        return VADSettings(
            mode="none",
            backend=normalized_backend,
            provider=normalized_provider,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            speech_pad_ms=30,
            max_speech_duration_s=60.0,
        )
    if normalized_mode == "light":
        return VADSettings(
            mode="light",
            backend=normalized_backend,
            provider=normalized_provider,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=150,
            speech_pad_ms=120,
            max_speech_duration_s=60.0,
        )
    if normalized_mode == "aggressive":
        return VADSettings(
            mode="aggressive",
            backend=normalized_backend,
            provider=normalized_provider,
            threshold=0.55,
            min_speech_duration_ms=150,
            min_silence_duration_ms=80,
            speech_pad_ms=45,
            max_speech_duration_s=60.0,
        )
    raise ValueError(f"Unsupported VAD mode {mode!r}; expected one of {SUPPORTED_VAD_MODES}")


def apply_vad_policy(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    mode: str,
    backend: str = "silero_vad_v6_onnx",
    provider: str = "auto",
) -> tuple[np.ndarray, TrimDecision]:
    settings = resolve_vad_settings(mode, backend=backend, provider=provider)
    if waveform.ndim != 2:
        raise ValueError("waveform must be channel-first with shape (channels, frames)")

    total_frames = int(waveform.shape[-1])
    if total_frames == 0:
        decision = TrimDecision(
            mode=settings.mode,
            applied=False,
            speech_detected=False,
            reason="empty_waveform",
            original_frame_count=0,
            output_frame_count=0,
            start_frame=0,
            end_frame=0,
            leading_trim_frames=0,
            trailing_trim_frames=0,
        )
        return waveform, decision

    if settings.mode == "none":
        decision = TrimDecision(
            mode="none",
            applied=False,
            speech_detected=True,
            reason="disabled",
            original_frame_count=total_frames,
            output_frame_count=total_frames,
            start_frame=0,
            end_frame=total_frames,
            leading_trim_frames=0,
            trailing_trim_frames=0,
        )
        return waveform, decision

    speech_segments = _detect_speech_segments(
        waveform,
        sample_rate_hz=sample_rate_hz,
        settings=settings,
    )
    if not speech_segments:
        decision = TrimDecision(
            mode=settings.mode,
            applied=False,
            speech_detected=False,
            reason="no_speech_detected",
            original_frame_count=total_frames,
            output_frame_count=total_frames,
            start_frame=0,
            end_frame=total_frames,
            leading_trim_frames=0,
            trailing_trim_frames=0,
        )
        return waveform, decision

    start_frame = max(0, min(int(speech_segments[0]["start"]), total_frames))
    end_frame = max(start_frame, min(int(speech_segments[-1]["end"]), total_frames))
    if start_frame == 0 and end_frame == total_frames:
        decision = TrimDecision(
            mode=settings.mode,
            applied=False,
            speech_detected=True,
            reason="no_boundary_change",
            original_frame_count=total_frames,
            output_frame_count=total_frames,
            start_frame=0,
            end_frame=total_frames,
            leading_trim_frames=0,
            trailing_trim_frames=0,
        )
        return waveform, decision

    trimmed = waveform[:, start_frame:end_frame]
    decision = TrimDecision(
        mode=settings.mode,
        applied=True,
        speech_detected=True,
        reason="trimmed",
        original_frame_count=total_frames,
        output_frame_count=int(trimmed.shape[-1]),
        start_frame=start_frame,
        end_frame=end_frame,
        leading_trim_frames=start_frame,
        trailing_trim_frames=total_frames - end_frame,
    )
    return trimmed, decision


def _detect_speech_segments(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    settings: VADSettings,
) -> list[dict[str, int]]:
    torch = _import_torch()
    get_speech_timestamps = _import_get_speech_timestamps()

    mono = waveform.astype(np.float32, copy=False).mean(axis=0)
    audio = torch.as_tensor(mono, dtype=torch.float32)
    model = _load_silero_model(settings.backend, settings.provider)
    segments = get_speech_timestamps(
        audio,
        model,
        threshold=settings.threshold,
        sampling_rate=sample_rate_hz,
        min_speech_duration_ms=settings.min_speech_duration_ms,
        max_speech_duration_s=settings.max_speech_duration_s,
        min_silence_duration_ms=settings.min_silence_duration_ms,
        speech_pad_ms=settings.speech_pad_ms,
        return_seconds=False,
    )
    return [
        {
            "start": int(segment["start"]),
            "end": int(segment["end"]),
        }
        for segment in segments
    ]


@lru_cache(maxsize=3)
def _load_silero_model(backend: str, provider: str) -> SileroVADOnnxModel:
    if backend not in SUPPORTED_VAD_BACKENDS:
        raise ValueError(f"Unsupported VAD backend {backend!r}; expected {SUPPORTED_VAD_BACKENDS}")
    providers = _resolve_execution_providers(provider)
    return SileroVADOnnxModel(
        model_path=_resolve_silero_model_path(),
        providers=providers,
    )


def _resolve_silero_model_path() -> str:
    _import_silero_vad_package()
    package_root = resources.files("silero_vad.data")
    return str(package_root.joinpath("silero_vad.onnx"))


def _resolve_execution_providers(provider: str) -> tuple[str, ...]:
    onnxruntime = _import_onnxruntime()
    available_providers = set(onnxruntime.get_available_providers())
    normalized_provider = provider.lower()
    if normalized_provider == "cpu":
        if "CPUExecutionProvider" not in available_providers:
            raise RuntimeError("onnxruntime CPUExecutionProvider is not available")
        return ("CPUExecutionProvider",)
    if normalized_provider == "cuda":
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "VAD provider is set to 'cuda' but CUDAExecutionProvider is not available"
            )
        providers = ["CUDAExecutionProvider"]
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")
        return tuple(providers)
    if normalized_provider == "auto":
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider"]
            if "CPUExecutionProvider" in available_providers:
                providers.append("CPUExecutionProvider")
            return tuple(providers)
        if "CPUExecutionProvider" in available_providers:
            return ("CPUExecutionProvider",)
        if available_providers:
            return tuple(sorted(available_providers))
        raise RuntimeError("onnxruntime reported no available execution providers")
    raise ValueError(
        f"Unsupported VAD provider {provider!r}; expected one of {SUPPORTED_VAD_PROVIDERS}"
    )


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Silero VAD requires torch. Sync the repository environment with "
            "`uv sync --dev --group train --group tracking`."
        ) from exc
    return torch


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Silero VAD ONNX backend requires onnxruntime. Sync the repository environment with "
            "`uv sync --dev --group train --group tracking`."
        ) from exc
    return onnxruntime


def _import_silero_vad_package() -> Any:
    try:
        import silero_vad
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Silero VAD package is not installed. Sync the repository environment with "
            "`uv sync --dev --group train --group tracking`."
        ) from exc
    return silero_vad


def _import_get_speech_timestamps() -> Any:
    try:
        from silero_vad import get_speech_timestamps
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Silero VAD package is not installed. Sync the repository environment with "
            "`uv sync --dev --group train --group tracking`."
        ) from exc
    return get_speech_timestamps
