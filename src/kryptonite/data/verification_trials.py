"""Deterministic balanced verification-trial generation from held-out utterances."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class VerificationUtterance:
    audio_basename: str
    speaker_id: str
    duration_seconds: float
    domain: str
    channel: str


@dataclass(frozen=True, slots=True)
class TrialStratum:
    label_name: str
    duration_bucket: str
    domain_relation: str
    channel_relation: str

    def key(self) -> str:
        return "|".join(
            (
                self.label_name,
                self.duration_bucket,
                self.domain_relation,
                self.channel_relation,
            )
        )


@dataclass(frozen=True, slots=True)
class PairOption:
    left_speaker_id: str
    right_speaker_id: str
    left_group: tuple[str, str, str]
    right_group: tuple[str, str, str]
    same_speaker: bool


@dataclass(slots=True)
class PairCursor:
    left_index: int = 0
    right_offset: int = 0


@dataclass(slots=True)
class VerificationTrialSummary:
    seed: int
    target_trials_per_bucket: int
    length_threshold_seconds: tuple[float, float]
    dev_utterance_count: int
    dev_speaker_count: int
    trial_count: int
    label_counts: dict[str, int]
    bucket_counts: dict[str, int]
    bucket_shortfalls: dict[str, int]
    unavailable_buckets: tuple[str, ...]
    covered_speaker_count: int
    missing_speakers: tuple[str, ...]
    covered_audio_count: int
    uncovered_audio_count: int
    duplicate_pair_count: int
    self_pair_count: int
    label_balance_gap: int

    @property
    def is_valid(self) -> bool:
        return (
            self.trial_count > 0
            and self.label_counts.get("positive", 0) > 0
            and self.label_counts.get("negative", 0) > 0
            and not self.bucket_shortfalls
            and not self.missing_speakers
            and self.duplicate_pair_count == 0
            and self.self_pair_count == 0
            and self.label_balance_gap <= 1
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "target_trials_per_bucket": self.target_trials_per_bucket,
            "length_threshold_seconds": list(self.length_threshold_seconds),
            "dev_utterance_count": self.dev_utterance_count,
            "dev_speaker_count": self.dev_speaker_count,
            "trial_count": self.trial_count,
            "label_counts": dict(self.label_counts),
            "bucket_counts": dict(self.bucket_counts),
            "bucket_shortfalls": dict(self.bucket_shortfalls),
            "unavailable_buckets": list(self.unavailable_buckets),
            "covered_speaker_count": self.covered_speaker_count,
            "missing_speakers": list(self.missing_speakers),
            "covered_audio_count": self.covered_audio_count,
            "uncovered_audio_count": self.uncovered_audio_count,
            "duplicate_pair_count": self.duplicate_pair_count,
            "self_pair_count": self.self_pair_count,
            "label_balance_gap": self.label_balance_gap,
            "is_valid": self.is_valid,
        }


LABEL_NAME_BY_VALUE: Final[dict[int, str]] = {1: "positive", 0: "negative"}
DOMAIN_RELATIONS: Final[tuple[str, str]] = ("same_domain", "cross_domain")
CHANNEL_RELATIONS: Final[tuple[str, str]] = ("same_channel", "cross_channel")
LENGTH_BUCKETS: Final[tuple[str, str, str]] = ("short", "medium", "long")


def build_balanced_verification_trials(
    *,
    utterances: list[VerificationUtterance],
    seed: int,
    target_trials_per_bucket: int = 128,
    short_duration_threshold_seconds: float = 2.0,
    medium_duration_threshold_seconds: float = 5.0,
) -> tuple[list[dict[str, object]], VerificationTrialSummary]:
    if target_trials_per_bucket <= 0:
        raise ValueError(
            f"target_trials_per_bucket must be positive, got {target_trials_per_bucket}"
        )
    if short_duration_threshold_seconds <= 0:
        raise ValueError("short_duration_threshold_seconds must be positive.")
    if medium_duration_threshold_seconds <= short_duration_threshold_seconds:
        raise ValueError(
            "medium_duration_threshold_seconds must be greater than "
            "short_duration_threshold_seconds."
        )

    normalized = sorted(
        utterances,
        key=lambda utterance: (
            utterance.speaker_id,
            utterance.audio_basename,
            utterance.domain,
            utterance.channel,
        ),
    )
    speaker_ids = sorted({utterance.speaker_id for utterance in normalized})
    if len(speaker_ids) < 2:
        raise ValueError("Verification trials require at least two held-out speakers.")

    grouped_utterances = _group_utterances(
        utterances=normalized,
        short_duration_threshold_seconds=short_duration_threshold_seconds,
        medium_duration_threshold_seconds=medium_duration_threshold_seconds,
    )
    options_by_stratum, unavailable_buckets = _build_options_by_stratum(
        grouped_utterances=grouped_utterances,
        speaker_ids=speaker_ids,
    )

    rows: list[dict[str, object]] = []
    used_pairs: set[tuple[str, str]] = set()
    bucket_counts: Counter[str] = Counter()
    bucket_shortfalls: dict[str, int] = {}

    for label_value, label_name in sorted(LABEL_NAME_BY_VALUE.items(), reverse=True):
        for duration_bucket in LENGTH_BUCKETS:
            for domain_relation in DOMAIN_RELATIONS:
                for channel_relation in CHANNEL_RELATIONS:
                    stratum = TrialStratum(
                        label_name=label_name,
                        duration_bucket=duration_bucket,
                        domain_relation=domain_relation,
                        channel_relation=channel_relation,
                    )
                    options = options_by_stratum.get(stratum.key(), ())
                    if not options:
                        continue
                    selected_rows = _collect_trials_for_stratum(
                        label_value=label_value,
                        stratum=stratum,
                        options=list(options),
                        grouped_utterances=grouped_utterances,
                        seed=seed,
                        target_count=target_trials_per_bucket,
                        used_pairs=used_pairs,
                    )
                    rows.extend(selected_rows)
                    bucket_counts[stratum.key()] += len(selected_rows)
                    shortfall = target_trials_per_bucket - len(selected_rows)
                    if shortfall:
                        bucket_shortfalls[stratum.key()] = shortfall

    rows.sort(
        key=lambda row: (
            int(row["label"]),
            str(row["duration_bucket"]),
            str(row["domain_relation"]),
            str(row["channel_relation"]),
            str(row["left_audio"]),
            str(row["right_audio"]),
        ),
        reverse=True,
    )
    summary = _build_summary(
        rows=rows,
        utterances=normalized,
        seed=seed,
        target_trials_per_bucket=target_trials_per_bucket,
        short_duration_threshold_seconds=short_duration_threshold_seconds,
        medium_duration_threshold_seconds=medium_duration_threshold_seconds,
        unavailable_buckets=tuple(unavailable_buckets),
        bucket_counts=dict(bucket_counts),
        bucket_shortfalls=bucket_shortfalls,
    )
    return rows, summary


def _group_utterances(
    *,
    utterances: list[VerificationUtterance],
    short_duration_threshold_seconds: float,
    medium_duration_threshold_seconds: float,
) -> dict[str, dict[tuple[str, str, str], tuple[VerificationUtterance, ...]]]:
    grouped: dict[str, defaultdict[tuple[str, str, str], list[VerificationUtterance]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for utterance in utterances:
        bucket = _duration_bucket(
            duration_seconds=utterance.duration_seconds,
            short_duration_threshold_seconds=short_duration_threshold_seconds,
            medium_duration_threshold_seconds=medium_duration_threshold_seconds,
        )
        grouped[utterance.speaker_id][(bucket, utterance.domain, utterance.channel)].append(
            utterance
        )

    return {
        speaker_id: {
            key: tuple(
                sorted(
                    items,
                    key=lambda utterance: (
                        utterance.audio_basename,
                        utterance.domain,
                        utterance.channel,
                    ),
                )
            )
            for key, items in groups.items()
        }
        for speaker_id, groups in grouped.items()
    }


def _build_options_by_stratum(
    *,
    grouped_utterances: dict[str, dict[tuple[str, str, str], tuple[VerificationUtterance, ...]]],
    speaker_ids: list[str],
) -> tuple[dict[str, tuple[PairOption, ...]], tuple[str, ...]]:
    options_by_stratum: defaultdict[str, list[PairOption]] = defaultdict(list)

    for speaker_id in speaker_ids:
        groups = sorted(grouped_utterances[speaker_id])
        for left_index, left_group in enumerate(groups):
            for right_group in groups[left_index:]:
                if left_group[0] != right_group[0]:
                    continue
                left_items = grouped_utterances[speaker_id][left_group]
                if left_group == right_group and len(left_items) < 2:
                    continue
                stratum_key = _stratum_key(
                    label_name="positive",
                    duration_bucket=left_group[0],
                    left_domain=left_group[1],
                    right_domain=right_group[1],
                    left_channel=left_group[2],
                    right_channel=right_group[2],
                )
                options_by_stratum[stratum_key].append(
                    PairOption(
                        left_speaker_id=speaker_id,
                        right_speaker_id=speaker_id,
                        left_group=left_group,
                        right_group=right_group,
                        same_speaker=True,
                    )
                )

    for left_index, left_speaker_id in enumerate(speaker_ids):
        for right_speaker_id in speaker_ids[left_index + 1 :]:
            left_groups = sorted(grouped_utterances[left_speaker_id])
            right_groups = sorted(grouped_utterances[right_speaker_id])
            for left_group in left_groups:
                for right_group in right_groups:
                    if left_group[0] != right_group[0]:
                        continue
                    stratum_key = _stratum_key(
                        label_name="negative",
                        duration_bucket=left_group[0],
                        left_domain=left_group[1],
                        right_domain=right_group[1],
                        left_channel=left_group[2],
                        right_channel=right_group[2],
                    )
                    options_by_stratum[stratum_key].append(
                        PairOption(
                            left_speaker_id=left_speaker_id,
                            right_speaker_id=right_speaker_id,
                            left_group=left_group,
                            right_group=right_group,
                            same_speaker=False,
                        )
                    )

    unavailable_buckets: list[str] = []
    materialized = {
        stratum_key: tuple(options) for stratum_key, options in sorted(options_by_stratum.items())
    }
    for label_name in LABEL_NAME_BY_VALUE.values():
        for duration_bucket in LENGTH_BUCKETS:
            for domain_relation in DOMAIN_RELATIONS:
                for channel_relation in CHANNEL_RELATIONS:
                    stratum_key = TrialStratum(
                        label_name=label_name,
                        duration_bucket=duration_bucket,
                        domain_relation=domain_relation,
                        channel_relation=channel_relation,
                    ).key()
                    if stratum_key not in materialized:
                        unavailable_buckets.append(stratum_key)
    return materialized, tuple(unavailable_buckets)


def _collect_trials_for_stratum(
    *,
    label_value: int,
    stratum: TrialStratum,
    options: list[PairOption],
    grouped_utterances: dict[str, dict[tuple[str, str, str], tuple[VerificationUtterance, ...]]],
    seed: int,
    target_count: int,
    used_pairs: set[tuple[str, str]],
) -> list[dict[str, object]]:
    rng = random.Random(f"{seed}|{stratum.key()}")
    shuffled_options = list(options)
    rng.shuffle(shuffled_options)
    cursors = {id(option): PairCursor() for option in shuffled_options}
    rows: list[dict[str, object]] = []
    active_options = shuffled_options

    while active_options and len(rows) < target_count:
        next_round: list[PairOption] = []
        progress_made = False
        for option in active_options:
            candidate = _next_candidate(
                option=option,
                cursor=cursors[id(option)],
                grouped_utterances=grouped_utterances,
                used_pairs=used_pairs,
            )
            if candidate is None:
                continue
            next_round.append(option)
            rows.append(
                _build_trial_row(
                    label_value=label_value,
                    stratum=stratum,
                    left_utterance=candidate[0],
                    right_utterance=candidate[1],
                )
            )
            progress_made = True
            if len(rows) >= target_count:
                break
        if not progress_made:
            break
        active_options = next_round
    return rows


def _next_candidate(
    *,
    option: PairOption,
    cursor: PairCursor,
    grouped_utterances: dict[str, dict[tuple[str, str, str], tuple[VerificationUtterance, ...]]],
    used_pairs: set[tuple[str, str]],
) -> tuple[VerificationUtterance, VerificationUtterance] | None:
    left_items = grouped_utterances[option.left_speaker_id][option.left_group]
    right_items = grouped_utterances[option.right_speaker_id][option.right_group]

    while True:
        candidate = _advance_cursor(
            option=option,
            cursor=cursor,
            left_items=left_items,
            right_items=right_items,
        )
        if candidate is None:
            return None
        pair_key = _pair_key(candidate[0].audio_basename, candidate[1].audio_basename)
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        return candidate


def _advance_cursor(
    *,
    option: PairOption,
    cursor: PairCursor,
    left_items: tuple[VerificationUtterance, ...],
    right_items: tuple[VerificationUtterance, ...],
) -> tuple[VerificationUtterance, VerificationUtterance] | None:
    if option.same_speaker and option.left_group == option.right_group:
        item_count = len(left_items)
        while cursor.right_offset < item_count - 1:
            if cursor.left_index >= item_count:
                cursor.left_index = 0
                cursor.right_offset += 1
                continue
            left_utterance = left_items[cursor.left_index]
            right_utterance = left_items[(cursor.left_index + cursor.right_offset + 1) % item_count]
            cursor.left_index += 1
            if left_utterance.audio_basename == right_utterance.audio_basename:
                continue
            return _canonicalize_pair(left_utterance, right_utterance)
        return None

    left_count = len(left_items)
    right_count = len(right_items)
    while cursor.right_offset < right_count:
        left_utterance = left_items[cursor.left_index % left_count]
        right_utterance = right_items[(cursor.left_index + cursor.right_offset) % right_count]
        cursor.left_index += 1
        if cursor.left_index >= left_count:
            cursor.left_index = 0
            cursor.right_offset += 1
        if left_utterance.audio_basename == right_utterance.audio_basename:
            continue
        return _canonicalize_pair(left_utterance, right_utterance)
    return None


def _build_trial_row(
    *,
    label_value: int,
    stratum: TrialStratum,
    left_utterance: VerificationUtterance,
    right_utterance: VerificationUtterance,
) -> dict[str, object]:
    return {
        "label": label_value,
        "left_audio": left_utterance.audio_basename,
        "right_audio": right_utterance.audio_basename,
        "duration_bucket": stratum.duration_bucket,
        "domain_relation": stratum.domain_relation,
        "channel_relation": stratum.channel_relation,
        "channel_mismatch": stratum.channel_relation == "cross_channel",
        "left_speaker_id": left_utterance.speaker_id,
        "right_speaker_id": right_utterance.speaker_id,
        "left_domain": left_utterance.domain,
        "right_domain": right_utterance.domain,
        "left_channel": left_utterance.channel,
        "right_channel": right_utterance.channel,
    }


def _build_summary(
    *,
    rows: list[dict[str, object]],
    utterances: list[VerificationUtterance],
    seed: int,
    target_trials_per_bucket: int,
    short_duration_threshold_seconds: float,
    medium_duration_threshold_seconds: float,
    unavailable_buckets: tuple[str, ...],
    bucket_counts: dict[str, int],
    bucket_shortfalls: dict[str, int],
) -> VerificationTrialSummary:
    label_counts = Counter()
    covered_speakers: set[str] = set()
    covered_audio: set[str] = set()
    pair_counts: Counter[tuple[str, str]] = Counter()
    self_pair_count = 0

    for row in rows:
        label_counts[LABEL_NAME_BY_VALUE[_coerce_int(row.get("label"))]] += 1
        left_audio = str(row["left_audio"])
        right_audio = str(row["right_audio"])
        pair_counts[_pair_key(left_audio, right_audio)] += 1
        if left_audio == right_audio:
            self_pair_count += 1
        covered_speakers.update((str(row["left_speaker_id"]), str(row["right_speaker_id"])))
        covered_audio.update((left_audio, right_audio))

    speaker_ids = sorted({utterance.speaker_id for utterance in utterances})
    missing_speakers = tuple(sorted(set(speaker_ids) - covered_speakers))
    duplicate_pair_count = sum(count - 1 for count in pair_counts.values() if count > 1)
    label_balance_gap = abs(label_counts.get("positive", 0) - label_counts.get("negative", 0))

    return VerificationTrialSummary(
        seed=seed,
        target_trials_per_bucket=target_trials_per_bucket,
        length_threshold_seconds=(
            short_duration_threshold_seconds,
            medium_duration_threshold_seconds,
        ),
        dev_utterance_count=len(utterances),
        dev_speaker_count=len(speaker_ids),
        trial_count=len(rows),
        label_counts=dict(label_counts),
        bucket_counts=dict(sorted(bucket_counts.items())),
        bucket_shortfalls=dict(sorted(bucket_shortfalls.items())),
        unavailable_buckets=tuple(sorted(unavailable_buckets)),
        covered_speaker_count=len(covered_speakers),
        missing_speakers=missing_speakers,
        covered_audio_count=len(covered_audio),
        uncovered_audio_count=len(utterances) - len(covered_audio),
        duplicate_pair_count=duplicate_pair_count,
        self_pair_count=self_pair_count,
        label_balance_gap=label_balance_gap,
    )


def _duration_bucket(
    *,
    duration_seconds: float,
    short_duration_threshold_seconds: float,
    medium_duration_threshold_seconds: float,
) -> str:
    if duration_seconds < short_duration_threshold_seconds:
        return "short"
    if duration_seconds < medium_duration_threshold_seconds:
        return "medium"
    return "long"


def _stratum_key(
    *,
    label_name: str,
    duration_bucket: str,
    left_domain: str,
    right_domain: str,
    left_channel: str,
    right_channel: str,
) -> str:
    return TrialStratum(
        label_name=label_name,
        duration_bucket=duration_bucket,
        domain_relation="same_domain" if left_domain == right_domain else "cross_domain",
        channel_relation="same_channel" if left_channel == right_channel else "cross_channel",
    ).key()


def _canonicalize_pair(
    left_utterance: VerificationUtterance,
    right_utterance: VerificationUtterance,
) -> tuple[VerificationUtterance, VerificationUtterance]:
    if right_utterance.audio_basename < left_utterance.audio_basename:
        return right_utterance, left_utterance
    return left_utterance, right_utterance


def _pair_key(left_audio: str, right_audio: str) -> tuple[str, str]:
    if left_audio <= right_audio:
        return (left_audio, right_audio)
    return (right_audio, left_audio)


def _coerce_int(value: object | None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Unexpected int-like value: {type(value)!r}")
