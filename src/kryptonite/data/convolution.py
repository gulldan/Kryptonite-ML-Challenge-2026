"""Reusable 1-D convolution helpers for audio rendering."""

from __future__ import annotations

import numpy as np


def fft_convolve_1d(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve 1-D signals via FFT to avoid quadratic runtime on long kernels."""

    signal_array = np.asarray(signal, dtype=np.float64)
    kernel_array = np.asarray(kernel, dtype=np.float64)
    if signal_array.ndim != 1 or kernel_array.ndim != 1:
        raise ValueError("fft_convolve_1d expects 1-D inputs.")
    if signal_array.size == 0 or kernel_array.size == 0:
        raise ValueError("fft_convolve_1d requires non-empty inputs.")

    output_size = signal_array.size + kernel_array.size - 1
    fft_size = 1 << (output_size - 1).bit_length()
    spectrum = np.fft.rfft(signal_array, n=fft_size) * np.fft.rfft(kernel_array, n=fft_size)
    return np.fft.irfft(spectrum, n=fft_size)[:output_size].astype(np.float64, copy=False)


__all__ = ["fft_convolve_1d"]
