"""Shared 80-dim log-Mel/Fbank extraction for offline and streaming paths."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as torch_functional
import torchaudio.functional as torchaudio_functional

from kryptonite.config import FeaturesConfig

SUPPORTED_FBANK_CMVN_MODES = frozenset({"none", "sliding"})
SUPPORTED_FBANK_OUTPUT_DTYPES = frozenset({"float16", "float32", "bfloat16"})
SUPPORTED_FBANK_WINDOW_TYPES = frozenset({"hann", "hamming"})
_CMVN_EPSILON = 1e-5


@dataclass(frozen=True, slots=True)
class FbankExtractionRequest:
    sample_rate_hz: int = 16_000
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    fft_size: int = 512
    window_type: str = "hann"
    f_min_hz: float = 20.0
    f_max_hz: float | None = None
    power: float = 2.0
    log_offset: float = 1e-6
    pad_end: bool = True
    cmvn_mode: str = "none"
    cmvn_window_frames: int = 300
    output_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.num_mel_bins <= 0:
            raise ValueError("num_mel_bins must be positive")
        if self.frame_length_ms <= 0.0:
            raise ValueError("frame_length_ms must be positive")
        if self.frame_shift_ms <= 0.0:
            raise ValueError("frame_shift_ms must be positive")
        if self.frame_length_samples <= 0:
            raise ValueError("frame_length_samples must be positive")
        if self.frame_shift_samples <= 0:
            raise ValueError("frame_shift_samples must be positive")
        if self.fft_size < self.frame_length_samples:
            raise ValueError("fft_size must be >= frame_length_samples")
        if self.window_type.lower() not in SUPPORTED_FBANK_WINDOW_TYPES:
            raise ValueError(f"window_type must be one of {sorted(SUPPORTED_FBANK_WINDOW_TYPES)}")
        if self.f_min_hz < 0.0:
            raise ValueError("f_min_hz must be non-negative")
        if self.f_max_hz is not None and self.f_max_hz <= self.f_min_hz:
            raise ValueError("f_max_hz must be greater than f_min_hz when provided")
        if self.f_max_hz is not None and self.f_max_hz > self.nyquist_hz:
            raise ValueError("f_max_hz must not exceed the Nyquist frequency")
        if self.power <= 0.0:
            raise ValueError("power must be positive")
        if self.log_offset <= 0.0:
            raise ValueError("log_offset must be positive")
        if self.cmvn_mode.lower() not in SUPPORTED_FBANK_CMVN_MODES:
            raise ValueError(f"cmvn_mode must be one of {sorted(SUPPORTED_FBANK_CMVN_MODES)}")
        if self.cmvn_window_frames <= 0:
            raise ValueError("cmvn_window_frames must be positive")
        if self.output_dtype.lower() not in SUPPORTED_FBANK_OUTPUT_DTYPES:
            raise ValueError(f"output_dtype must be one of {sorted(SUPPORTED_FBANK_OUTPUT_DTYPES)}")

    @property
    def frame_length_samples(self) -> int:
        return max(1, round(self.sample_rate_hz * self.frame_length_ms / 1000.0))

    @property
    def frame_shift_samples(self) -> int:
        return max(1, round(self.sample_rate_hz * self.frame_shift_ms / 1000.0))

    @property
    def nyquist_hz(self) -> float:
        return float(self.sample_rate_hz) / 2.0

    @property
    def normalized_window_type(self) -> str:
        return self.window_type.lower()

    @property
    def normalized_cmvn_mode(self) -> str:
        return self.cmvn_mode.lower()

    @property
    def torch_output_dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[self.output_dtype.lower()]

    @classmethod
    def from_config(cls, config: FeaturesConfig) -> FbankExtractionRequest:
        return cls(
            sample_rate_hz=config.sample_rate_hz,
            num_mel_bins=config.num_mel_bins,
            frame_length_ms=config.frame_length_ms,
            frame_shift_ms=config.frame_shift_ms,
            fft_size=config.fft_size,
            window_type=config.window_type,
            f_min_hz=config.f_min_hz,
            f_max_hz=config.f_max_hz,
            power=config.power,
            log_offset=config.log_offset,
            pad_end=config.pad_end,
            cmvn_mode=config.cmvn_mode,
            cmvn_window_frames=config.cmvn_window_frames,
            output_dtype=config.output_dtype,
        )


@dataclass(frozen=True, slots=True)
class _RuntimeBuffers:
    window: torch.Tensor
    mel_filterbank: torch.Tensor


class _SlidingCMVNState:
    def __init__(self, *, feature_dim: int, window_frames: int, device: torch.device) -> None:
        self._window_frames = window_frames
        self._frames: deque[torch.Tensor] = deque()
        self._running_sum = torch.zeros(feature_dim, dtype=torch.float32, device=device)
        self._running_sumsq = torch.zeros(feature_dim, dtype=torch.float32, device=device)

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features

        normalized_frames: list[torch.Tensor] = []
        for frame in features:
            current_frame = frame.to(dtype=torch.float32)
            self._frames.append(current_frame)
            self._running_sum += current_frame
            self._running_sumsq += current_frame.square()

            if len(self._frames) > self._window_frames:
                removed = self._frames.popleft()
                self._running_sum -= removed
                self._running_sumsq -= removed.square()

            frame_count = float(len(self._frames))
            mean = self._running_sum / frame_count
            variance = self._running_sumsq / frame_count - mean.square()
            std = torch.sqrt(variance.clamp_min(_CMVN_EPSILON))
            normalized_frames.append((current_frame - mean) / std)

        return torch.stack(normalized_frames, dim=0)


class FbankExtractor:
    def __init__(self, request: FbankExtractionRequest | None = None) -> None:
        self.request = request or FbankExtractionRequest()
        self._runtime_by_device: dict[str, _RuntimeBuffers] = {}

    def extract(self, waveform: Any, *, sample_rate_hz: int) -> torch.Tensor:
        self._validate_sample_rate(sample_rate_hz)
        mono_waveform = _coerce_mono_waveform(waveform)
        frames, _ = self._frame_waveform(mono_waveform, final=True)
        raw_features = self._compute_raw_features(frames)
        return self._finalize_features(raw_features)

    def create_online_extractor(self) -> OnlineFbankExtractor:
        return OnlineFbankExtractor(self)

    def _validate_sample_rate(self, sample_rate_hz: int) -> None:
        if sample_rate_hz != self.request.sample_rate_hz:
            raise ValueError(
                f"Feature extractor expects sample_rate_hz={self.request.sample_rate_hz}, "
                f"got {sample_rate_hz}"
            )

    def _frame_waveform(
        self,
        waveform: torch.Tensor,
        *,
        final: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_length = self.request.frame_length_samples
        frame_shift = self.request.frame_shift_samples
        current = waveform.to(dtype=torch.float32)

        if final and self.request.pad_end:
            current = _pad_waveform_for_final_frame(
                current,
                frame_length=frame_length,
                frame_shift=frame_shift,
            )

        if current.numel() < frame_length:
            empty_frames = current.new_empty((0, frame_length))
            return empty_frames, current

        frames = current.unfold(0, frame_length, frame_shift)
        if final:
            remainder = current.new_empty((0,))
        else:
            processed_samples = frames.shape[0] * frame_shift
            remainder = current[processed_samples:]
        return frames, remainder

    def _compute_raw_features(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.numel() == 0:
            return frames.new_empty((0, self.request.num_mel_bins))

        runtime = self._get_runtime(frames.device)
        windowed_frames = frames * runtime.window
        spectrum = torch.fft.rfft(windowed_frames, n=self.request.fft_size, dim=-1)
        magnitudes = spectrum.abs()
        power_spectrum = magnitudes.pow(self.request.power)
        mel_energies = power_spectrum @ runtime.mel_filterbank
        return torch.log(mel_energies.clamp_min(self.request.log_offset))

    def _finalize_features(
        self,
        raw_features: torch.Tensor,
        *,
        cmvn_state: _SlidingCMVNState | None = None,
    ) -> torch.Tensor:
        if raw_features.numel() == 0:
            return raw_features.to(dtype=self.request.torch_output_dtype)

        if self.request.normalized_cmvn_mode == "sliding":
            state = cmvn_state or _SlidingCMVNState(
                feature_dim=self.request.num_mel_bins,
                window_frames=self.request.cmvn_window_frames,
                device=raw_features.device,
            )
            normalized = state.normalize(raw_features)
        else:
            normalized = raw_features.to(dtype=torch.float32)

        return normalized.to(dtype=self.request.torch_output_dtype)

    def _get_runtime(self, device: torch.device) -> _RuntimeBuffers:
        cache_key = str(device)
        cached = self._runtime_by_device.get(cache_key)
        if cached is not None:
            return cached

        window = _build_window(
            self.request.normalized_window_type,
            frame_length=self.request.frame_length_samples,
            device=device,
        )
        mel_filterbank = torchaudio_functional.melscale_fbanks(
            n_freqs=self.request.fft_size // 2 + 1,
            f_min=self.request.f_min_hz,
            f_max=self.request.f_max_hz or self.request.nyquist_hz,
            n_mels=self.request.num_mel_bins,
            sample_rate=self.request.sample_rate_hz,
            norm=None,
            mel_scale="htk",
        ).to(device=device, dtype=torch.float32)
        runtime = _RuntimeBuffers(window=window, mel_filterbank=mel_filterbank)
        self._runtime_by_device[cache_key] = runtime
        return runtime


class OnlineFbankExtractor:
    def __init__(self, extractor: FbankExtractor) -> None:
        self._extractor = extractor
        self._buffer = torch.empty(0, dtype=torch.float32)
        self._device = torch.device("cpu")
        self._cmvn_state: _SlidingCMVNState | None = None
        self._initialized = False

    def push(self, waveform_chunk: Any, *, sample_rate_hz: int) -> torch.Tensor:
        self._extractor._validate_sample_rate(sample_rate_hz)
        chunk = _coerce_mono_waveform(waveform_chunk)
        if chunk.numel() == 0:
            return self._empty_features(device=chunk.device)

        if not self._initialized:
            self._device = chunk.device
            self._buffer = torch.empty(0, dtype=torch.float32, device=self._device)
            if self._extractor.request.normalized_cmvn_mode == "sliding":
                self._cmvn_state = _SlidingCMVNState(
                    feature_dim=self._extractor.request.num_mel_bins,
                    window_frames=self._extractor.request.cmvn_window_frames,
                    device=self._device,
                )
            self._initialized = True

        chunk = chunk.to(device=self._device, dtype=torch.float32)
        self._buffer = torch.cat((self._buffer, chunk), dim=0)
        frames, remainder = self._extractor._frame_waveform(self._buffer, final=False)
        self._buffer = remainder
        raw_features = self._extractor._compute_raw_features(frames)
        return self._extractor._finalize_features(raw_features, cmvn_state=self._cmvn_state)

    def flush(self) -> torch.Tensor:
        if not self._initialized:
            return self._empty_features(device=self._device)

        frames, _ = self._extractor._frame_waveform(self._buffer, final=True)
        raw_features = self._extractor._compute_raw_features(frames)
        features = self._extractor._finalize_features(raw_features, cmvn_state=self._cmvn_state)
        self.reset()
        return features

    def reset(self) -> None:
        self._buffer = torch.empty(0, dtype=torch.float32, device=self._device)
        self._cmvn_state = None
        self._initialized = False

    def _empty_features(self, *, device: torch.device) -> torch.Tensor:
        return torch.empty(
            (0, self._extractor.request.num_mel_bins),
            dtype=self._extractor.request.torch_output_dtype,
            device=device,
        )


def extract_fbank(
    waveform: Any,
    *,
    sample_rate_hz: int,
    request: FbankExtractionRequest | None = None,
) -> torch.Tensor:
    extractor = FbankExtractor(request=request)
    return extractor.extract(waveform, sample_rate_hz=sample_rate_hz)


def _coerce_mono_waveform(waveform: Any) -> torch.Tensor:
    tensor = torch.as_tensor(waveform)
    if tensor.ndim == 2:
        if int(tensor.shape[0]) != 1:
            raise ValueError("Expected a mono waveform shaped as [samples] or [1, samples]")
        tensor = tensor.squeeze(0)
    elif tensor.ndim != 1:
        raise ValueError("Expected a mono waveform shaped as [samples] or [1, samples]")

    if tensor.numel() == 0:
        raise ValueError("waveform must not be empty")
    return tensor.to(dtype=torch.float32)


def _pad_waveform_for_final_frame(
    waveform: torch.Tensor,
    *,
    frame_length: int,
    frame_shift: int,
) -> torch.Tensor:
    if waveform.numel() < frame_length:
        return torch_functional.pad(waveform, (0, frame_length - waveform.numel()))

    remainder = (waveform.numel() - frame_length) % frame_shift
    if remainder == 0:
        return waveform

    pad_amount = frame_shift - remainder
    return torch_functional.pad(waveform, (0, pad_amount))


def _build_window(
    window_type: str,
    *,
    frame_length: int,
    device: torch.device,
) -> torch.Tensor:
    if window_type == "hann":
        return torch.hann_window(
            frame_length,
            periodic=False,
            dtype=torch.float32,
            device=device,
        )
    if window_type == "hamming":
        return torch.hamming_window(
            frame_length,
            periodic=False,
            dtype=torch.float32,
            device=device,
        )
    raise ValueError(f"Unsupported window_type={window_type!r}")


__all__ = [
    "FbankExtractionRequest",
    "FbankExtractor",
    "OnlineFbankExtractor",
    "SUPPORTED_FBANK_CMVN_MODES",
    "SUPPORTED_FBANK_OUTPUT_DTYPES",
    "SUPPORTED_FBANK_WINDOW_TYPES",
    "extract_fbank",
]
