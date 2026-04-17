"""Shared review-package bucket definitions."""

from __future__ import annotations

from typing import Any

BUCKET_DEFS: dict[str, Any] = {
    "duration_bucket": {
        "very_short": [0.0, 2.0],
        "short": [2.0, 4.0],
        "medium": [4.0, 6.0],
        "normal": [6.0, 10.0],
        "long": [10.0, 20.0],
        "very_long": [20.0, 40.0],
        "extra_long": [40.0, None],
    },
    "silence_heavy": {"threshold": 0.5, "metric": "silence_ratio_40db"},
    "peak_limited": {"peak_dbfs_threshold": -0.1},
    "hard_clipped": {"clipping_frac_threshold": 0.01},
    "narrowband_like": {"narrowband_proxy_min": 0.5, "rolloff95_hz_max": 3800.0},
    "low_rms": {"rms_dbfs_max": -40.0},
    "rms_bucket": {
        "very_low": [-100.0, -40.0],
        "low": [-40.0, -30.0],
        "mid": [-30.0, -20.0],
        "high": [-20.0, 0.0],
    },
    "leading_silence_bucket": {"none": [0.0, 0.2], "short": [0.2, 1.0], "long": [1.0, None]},
    "trailing_silence_bucket": {"none": [0.0, 0.2], "short": [0.2, 1.0], "long": [1.0, None]},
}
