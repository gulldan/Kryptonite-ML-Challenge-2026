"""Constants for audio-quality EDA."""

from __future__ import annotations

KNOWN_DATA_SPLITS: tuple[str, ...] = ("train", "dev", "demo")
DURATION_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("0-1s", 0.0, 1.0),
    ("1-2s", 1.0, 2.0),
    ("2-5s", 2.0, 5.0),
    ("5-10s", 5.0, 10.0),
    ("10-30s", 10.0, 30.0),
    ("30-60s", 30.0, 60.0),
    ("60s+", 60.0, None),
)
LOUDNESS_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<-35", None, -35.0),
    ("-35:-30", -35.0, -30.0),
    ("-30:-25", -30.0, -25.0),
    ("-25:-20", -25.0, -20.0),
    ("-20:-15", -20.0, -15.0),
    ("-15:-10", -15.0, -10.0),
    (">=-10", -10.0, None),
)
SILENCE_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("0-5%", 0.0, 0.05),
    ("5-20%", 0.05, 0.20),
    ("20-50%", 0.20, 0.50),
    ("50-80%", 0.50, 0.80),
    ("80%+", 0.80, None),
)
SILENCE_CHUNK_MS = 100
SILENCE_THRESHOLD_DBFS = -45.0
TARGET_SAMPLE_RATE_HZ = 16_000
TARGET_CHANNELS = 1
SHORT_DURATION_SECONDS = 1.0
LONG_DURATION_SECONDS = 15.0
LOW_LOUDNESS_DBFS = -30.0
VERY_LOW_LOUDNESS_DBFS = -35.0
CLIPPING_PEAK_DBFS = -0.1
HIGH_SILENCE_RATIO = 0.50
MODERATE_SILENCE_RATIO = 0.20
HIGH_DC_OFFSET_RATIO = 0.01
MAX_EXAMPLES = 12
