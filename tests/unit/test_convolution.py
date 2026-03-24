from __future__ import annotations

import numpy as np
import pytest

from kryptonite.data.convolution import fft_convolve_1d


def test_fft_convolve_matches_numpy_convolve() -> None:
    rng = np.random.default_rng(20260324)
    signal = rng.normal(size=257)
    kernel = rng.normal(size=63)

    expected = np.convolve(signal, kernel, mode="full")
    actual = fft_convolve_1d(signal, kernel)

    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_fft_convolve_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fft_convolve_1d(np.array([], dtype=np.float64), np.array([1.0], dtype=np.float64))
