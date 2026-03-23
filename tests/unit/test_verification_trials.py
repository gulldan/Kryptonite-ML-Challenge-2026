from __future__ import annotations

from kryptonite.data.verification_trials import (
    VerificationUtterance,
    build_balanced_verification_trials,
)


def test_build_balanced_verification_trials_is_deterministic_and_bucket_balanced() -> None:
    utterances: list[VerificationUtterance] = []
    for speaker_id in ("speaker_alpha", "speaker_bravo"):
        for duration_bucket, duration_seconds in (("short", 1.0), ("medium", 3.0)):
            for domain in ("F", "S"):
                for channel in ("I", "PAD"):
                    for replica_index in range(2):
                        utterances.append(
                            VerificationUtterance(
                                audio_basename=(
                                    f"{speaker_id}-{duration_bucket}-{domain}-{channel}-{replica_index}.wav"
                                ),
                                speaker_id=speaker_id,
                                duration_seconds=duration_seconds,
                                domain=domain,
                                channel=channel,
                            )
                        )

    rows_a, summary_a = build_balanced_verification_trials(
        utterances=utterances,
        seed=17,
        target_trials_per_bucket=1,
    )
    rows_b, summary_b = build_balanced_verification_trials(
        utterances=utterances,
        seed=17,
        target_trials_per_bucket=1,
    )

    assert rows_a == rows_b
    assert summary_a.to_dict() == summary_b.to_dict()
    assert summary_a.is_valid is True
    assert summary_a.trial_count == 16
    assert summary_a.label_counts == {"negative": 8, "positive": 8}
    assert all(count == 1 for count in summary_a.bucket_counts.values())
    assert len(summary_a.bucket_counts) == 16
    assert len(summary_a.unavailable_buckets) == 8
    assert all("long" in key for key in summary_a.unavailable_buckets)
    assert summary_a.missing_speakers == ()

    positive_rows = [row for row in rows_a if row["label"] == 1]
    negative_rows = [row for row in rows_a if row["label"] == 0]
    assert all(row["left_speaker_id"] == row["right_speaker_id"] for row in positive_rows)
    assert all(row["left_speaker_id"] != row["right_speaker_id"] for row in negative_rows)
    assert all(
        row["channel_mismatch"] == (row["channel_relation"] == "cross_channel") for row in rows_a
    )


def test_build_balanced_verification_trials_requires_multiple_speakers() -> None:
    utterances = [
        VerificationUtterance(
            audio_basename="speaker_only-short-F-I-0.wav",
            speaker_id="speaker_only",
            duration_seconds=1.0,
            domain="F",
            channel="I",
        ),
        VerificationUtterance(
            audio_basename="speaker_only-short-S-PAD-0.wav",
            speaker_id="speaker_only",
            duration_seconds=1.0,
            domain="S",
            channel="PAD",
        ),
    ]

    try:
        build_balanced_verification_trials(
            utterances=utterances,
            seed=17,
            target_trials_per_bucket=1,
        )
    except ValueError as exc:
        assert "at least two held-out speakers" in str(exc)
    else:
        raise AssertionError("Expected verification-trial generation to reject one-speaker input.")
