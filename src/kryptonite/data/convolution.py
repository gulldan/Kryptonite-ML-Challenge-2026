"""Small convolution helpers for audio corruption transforms."""

from __future__ import annotations

import numpy as np


def fft_convolve_1d(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return the full one-dimensional convolution using an FFT backend."""
    signal_array = np.asarray(signal, dtype=np.float64)
    kernel_array = np.asarray(kernel, dtype=np.float64)
    if signal_array.ndim != 1 or kernel_array.ndim != 1:
        raise ValueError("fft_convolve_1d expects one-dimensional inputs")
    if signal_array.size == 0 or kernel_array.size == 0:
        raise ValueError("fft_convolve_1d inputs must be non-empty")

    output_size = signal_array.size + kernel_array.size - 1
    fft_size = 1 << (output_size - 1).bit_length()
    spectrum = np.fft.rfft(signal_array, n=fft_size) * np.fft.rfft(kernel_array, n=fft_size)
    return np.fft.irfft(spectrum, n=fft_size)[:output_size]


__all__ = ["fft_convolve_1d"]
