"""Flagged-example selection for audio-quality reports."""

from __future__ import annotations

from .constants import MAX_EXAMPLES
from .models import FlaggedExample


def select_examples(records) -> list[FlaggedExample]:
    flagged_records = [record for record in records if record.quality_flags]
    flagged_records.sort(
        key=lambda record: (-example_score(record), record.audio_path or record.identity_key)
    )
    return [
        FlaggedExample(
            audio_path=record.audio_path or record.identity_key,
            split_name=record.split_name,
            dataset_name=record.dataset_name,
            source_label=record.source_label,
            condition_label=record.condition_label,
            duration_seconds=record.duration_seconds,
            rms_dbfs=record.rms_dbfs,
            peak_dbfs=record.peak_dbfs,
            silence_ratio=record.silence_ratio,
            flags=record.quality_flags,
        )
        for record in flagged_records[:MAX_EXAMPLES]
    ]


def example_score(record) -> int:
    priority = {
        "missing_audio_file": 100,
        "audio_read_error": 90,
        "zero_signal": 85,
        "clipping_risk": 80,
        "high_silence_ratio": 75,
        "very_low_loudness": 70,
        "non_16k_sample_rate": 60,
        "non_mono_audio": 55,
        "dc_offset_risk": 50,
        "moderate_silence_ratio": 40,
        "low_loudness": 35,
        "long_duration": 20,
        "short_duration": 15,
        "missing_audio_path": 5,
    }
    return sum(priority.get(flag, 1) for flag in record.quality_flags)
