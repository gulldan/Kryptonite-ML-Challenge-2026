from __future__ import annotations

from kryptonite.eval.verification_error_analysis.slices import (
    derive_slice_value as derive_error_slice_value,
)
from kryptonite.eval.verification_slices import derive_slice_value


def test_derive_slice_value_supports_codec_rt60_and_silence_ratio_fields() -> None:
    codec_metadata = {
        "corruption_family": "codec",
        "corruption_suite": "dev_codec",
        "corruption_severity": "medium",
        "silence_ratio": 0.35,
        "corruption_metadata": {
            "codec_family": "telephony",
            "codec_name": "pcm_mulaw",
        },
    }
    reverb_metadata = {
        "corruption_family": "reverb",
        "corruption_severity": "heavy",
        "corruption_metadata": {
            "field": "far",
            "direct_condition": "low",
            "rt60_bucket": "long",
        },
    }
    channel_metadata = {
        "corruption_family": "codec",
        "corruption_suite": "dev_channel",
        "corruption_severity": "light",
        "corruption_metadata": {
            "codec_family": "channel",
            "codec_name": "device_color",
        },
    }

    assert (
        derive_slice_value(
            "codec_slice",
            left_metadata=codec_metadata,
            right_metadata=codec_metadata,
        )
        == "telephony/pcm_mulaw/medium"
    )
    assert (
        derive_slice_value(
            "channel_slice",
            left_metadata=channel_metadata,
            right_metadata=channel_metadata,
        )
        == "channel/device_color/light"
    )
    assert (
        derive_slice_value(
            "rt60_slice",
            left_metadata=reverb_metadata,
            right_metadata=reverb_metadata,
        )
        == "long"
    )
    assert (
        derive_slice_value(
            "silence_ratio_bucket",
            left_metadata=codec_metadata,
            right_metadata=codec_metadata,
        )
        == "20_to_50pct"
    )


def test_error_analysis_slices_reuse_shared_derivation_logic() -> None:
    metadata = {
        "corruption_family": "codec",
        "corruption_suite": "dev_codec",
        "corruption_severity": "heavy",
        "corruption_metadata": {
            "codec_family": "compression",
            "codec_name": "aac",
        },
    }

    assert (
        derive_error_slice_value(
            "codec_slice",
            left_metadata=metadata,
            right_metadata=metadata,
        )
        == "compression/aac/heavy"
    )
