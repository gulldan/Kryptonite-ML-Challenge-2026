"""Hard-negative-aware batch sampler for stage-2 multi-condition training."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict

from torch.utils.data import Sampler

from kryptonite.data import ManifestRow
from kryptonite.features import UtteranceChunkingRequest

from ..augmentation_runtime import TrainingAugmentationRuntime
from ..manifest_speaker_data import TrainingSampleRequest


class Stage2BatchSampler(Sampler[list[TrainingSampleRequest]]):
    """Balanced speaker batch sampler with hard-negative speaker oversampling.

    Functionally identical to ``BalancedSpeakerBatchSampler`` when
    ``speaker_weights`` are all equal (the default).  After each hard-negative
    mining pass, call :meth:`update_speaker_weights` to bias sampling toward
    confusable speakers.

    Hard-negative oversampling is implemented by expanding the round-robin
    speaker pool proportionally to each speaker's weight, so speakers with
    higher weight appear more often in the rotation.
    """

    def __init__(
        self,
        *,
        rows: list[ManifestRow],
        batch_size: int,
        seed: int,
        chunking_request: UtteranceChunkingRequest,
        hard_negative_fraction: float = 0.5,
        batches_per_epoch: int | None = None,
        augmentation_runtime: TrainingAugmentationRuntime | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if chunking_request.train_num_crops != 1:
            raise ValueError("Stage2BatchSampler currently supports train_num_crops=1 only.")
        if not (0.0 <= hard_negative_fraction <= 1.0):
            raise ValueError("hard_negative_fraction must be in [0, 1]")

        self._rows = rows
        self._batch_size = batch_size
        self._seed = seed
        self._chunking_request = chunking_request
        self._hard_negative_fraction = hard_negative_fraction
        self._augmentation_runtime = augmentation_runtime
        self._speaker_to_row_indices = _group_rows_by_speaker(rows)
        self._speaker_ids = sorted(self._speaker_to_row_indices)
        self._batches_per_epoch = batches_per_epoch or max(1, math.ceil(len(rows) / batch_size))
        self._epoch = 0
        self._next_batch_index = 0
        self._speaker_weights: dict[str, float] = {
            speaker_id: 1.0 for speaker_id in self._speaker_ids
        }

    def __len__(self) -> int:
        return max(0, self._batches_per_epoch - self._next_batch_index)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = epoch
        self._next_batch_index = 0

    def update_speaker_weights(self, speaker_weights: dict[str, float]) -> None:
        """Replace sampling weights.  Unknown speakers are silently ignored."""
        for speaker_id in self._speaker_ids:
            weight = speaker_weights.get(speaker_id)
            if weight is None:
                self._speaker_weights[speaker_id] = 1.0
                continue
            if weight <= 0.0:
                raise ValueError(
                    f"Speaker weight must be positive, got {weight!r} for speaker {speaker_id!r}"
                )
            self._speaker_weights[speaker_id] = weight

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self._epoch,
            "next_batch_index": self._next_batch_index,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._epoch = int(state["epoch"])
        self._next_batch_index = int(state["next_batch_index"])

    def __iter__(self):
        batches = self._build_epoch_batches(self._epoch)
        start_index = self._next_batch_index
        for batch_index in range(start_index, len(batches)):
            self._next_batch_index = batch_index + 1
            yield batches[batch_index]

    def _build_epoch_batches(self, epoch: int) -> list[list[TrainingSampleRequest]]:
        epoch_number = epoch + 1
        epoch_rng = random.Random(_stable_seed(self._seed, "epoch", str(epoch_number)))

        speaker_order = list(self._speaker_ids)
        epoch_rng.shuffle(speaker_order)
        hard_negative_order = _expand_speaker_order_by_weights(
            speaker_ids=self._speaker_ids,
            weights=self._speaker_weights,
            rng=epoch_rng,
        )
        speaker_cursor = 0
        hard_negative_cursor = 0
        row_orders = {
            speaker_id: _shuffled_row_indices(
                self._speaker_to_row_indices[speaker_id],
                seed=_stable_seed(self._seed, "rows", str(epoch_number), speaker_id, "0"),
            )
            for speaker_id in self._speaker_ids
        }
        row_cursors = {speaker_id: 0 for speaker_id in self._speaker_ids}
        row_refills = {speaker_id: 0 for speaker_id in self._speaker_ids}

        batches: list[list[TrainingSampleRequest]] = []
        for batch_index in range(self._batches_per_epoch):
            crop_seconds = _sample_crop_seconds(epoch_rng, self._chunking_request)
            unique_slots = min(self._batch_size, len(self._speaker_ids))
            target_hard_slots = min(
                self._batch_size,
                round(self._batch_size * self._hard_negative_fraction),
            )

            batch_speakers: list[str] = []
            selected_speakers: set[str] = set()
            hard_slots_filled = 0

            hard_negative_cursor, hard_slots_filled = _append_unique_speakers(
                batch_speakers=batch_speakers,
                selected_speakers=selected_speakers,
                target_slots=min(unique_slots, target_hard_slots),
                speaker_order=hard_negative_order,
                speaker_cursor=hard_negative_cursor,
                epoch_rng=epoch_rng,
            )
            speaker_cursor, _ = _append_unique_speakers(
                batch_speakers=batch_speakers,
                selected_speakers=selected_speakers,
                target_slots=unique_slots - len(batch_speakers),
                speaker_order=speaker_order,
                speaker_cursor=speaker_cursor,
                epoch_rng=epoch_rng,
            )

            remaining_hard_slots = max(0, target_hard_slots - hard_slots_filled)
            while len(batch_speakers) < self._batch_size and remaining_hard_slots > 0:
                speaker_id, hard_negative_cursor = _next_speaker_id(
                    speaker_order=hard_negative_order,
                    speaker_cursor=hard_negative_cursor,
                    epoch_rng=epoch_rng,
                )
                batch_speakers.append(speaker_id)
                remaining_hard_slots -= 1
            while len(batch_speakers) < self._batch_size:
                speaker_id, speaker_cursor = _next_speaker_id(
                    speaker_order=speaker_order,
                    speaker_cursor=speaker_cursor,
                    epoch_rng=epoch_rng,
                )
                batch_speakers.append(speaker_id)

            batch_requests: list[TrainingSampleRequest] = []
            for position_in_batch, speaker_id in enumerate(batch_speakers):
                row_index = _next_row_index(
                    speaker_id=speaker_id,
                    row_orders=row_orders,
                    row_cursors=row_cursors,
                    row_refills=row_refills,
                    batch_seed=self._seed,
                    epoch_number=epoch_number,
                )
                recipe_rng = random.Random(
                    _stable_seed(
                        self._seed,
                        "recipe",
                        str(epoch_number),
                        str(batch_index),
                        str(position_in_batch),
                        str(row_index),
                    )
                )
                recipe = (
                    None
                    if self._augmentation_runtime is None
                    else self._augmentation_runtime.sample_recipe(
                        epoch_index=epoch_number,
                        rng=recipe_rng,
                    )
                )
                batch_requests.append(
                    TrainingSampleRequest(
                        row_index=row_index,
                        request_seed=_stable_seed(
                            self._seed,
                            "request",
                            str(epoch_number),
                            str(batch_index),
                            str(position_in_batch),
                            str(row_index),
                        ),
                        crop_seconds=crop_seconds,
                        clean_sample=recipe is None or recipe.clean_sample,
                        recipe_stage="steady" if recipe is None else recipe.stage,
                        recipe_intensity="clean" if recipe is None else recipe.intensity,
                        augmentations=() if recipe is None else recipe.augmentations,
                    )
                )
            batches.append(batch_requests)
        return batches


def _expand_speaker_order_by_weights(
    *,
    speaker_ids: list[str],
    weights: dict[str, float],
    rng: random.Random,
) -> list[str]:
    """Return an expanded speaker list where each speaker appears proportionally to weight.

    Weights are normalised and then scaled to produce integer repeat counts.
    The minimum repeat count is 1 (every speaker appears at least once).
    The maximum scale factor is capped at 8× to avoid extreme list sizes.
    """
    if not speaker_ids:
        return []
    raw_weights = [weights.get(s, 1.0) for s in speaker_ids]
    min_w = min(raw_weights)
    max_w = max(raw_weights)
    if max_w <= min_w or max_w <= 0.0:
        expanded = list(speaker_ids)
        rng.shuffle(expanded)
        return expanded

    scale = min(8.0, max_w / min_w)
    expanded: list[str] = []
    for speaker_id, w in zip(speaker_ids, raw_weights, strict=True):
        repeats = max(1, round((w / max_w) * scale))
        expanded.extend([speaker_id] * repeats)
    rng.shuffle(expanded)
    return expanded


def _append_unique_speakers(
    *,
    batch_speakers: list[str],
    selected_speakers: set[str],
    target_slots: int,
    speaker_order: list[str],
    speaker_cursor: int,
    epoch_rng: random.Random,
) -> tuple[int, int]:
    added = 0
    while added < target_slots:
        speaker_id, speaker_cursor = _next_speaker_id(
            speaker_order=speaker_order,
            speaker_cursor=speaker_cursor,
            epoch_rng=epoch_rng,
        )
        if speaker_id in selected_speakers:
            continue
        selected_speakers.add(speaker_id)
        batch_speakers.append(speaker_id)
        added += 1
    return speaker_cursor, added


def _group_rows_by_speaker(rows: list[ManifestRow]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row.speaker_id].append(index)
    if len(grouped) < 2:
        raise ValueError("Stage2BatchSampler requires at least two speakers.")
    return dict(grouped)


def _next_speaker_id(
    *,
    speaker_order: list[str],
    speaker_cursor: int,
    epoch_rng: random.Random,
) -> tuple[str, int]:
    if speaker_cursor >= len(speaker_order):
        epoch_rng.shuffle(speaker_order)
        speaker_cursor = 0
    speaker_id = speaker_order[speaker_cursor]
    return speaker_id, speaker_cursor + 1


def _next_row_index(
    *,
    speaker_id: str,
    row_orders: dict[str, list[int]],
    row_cursors: dict[str, int],
    row_refills: dict[str, int],
    batch_seed: int,
    epoch_number: int,
) -> int:
    cursor = row_cursors[speaker_id]
    current_order = row_orders[speaker_id]
    if cursor >= len(current_order):
        row_refills[speaker_id] += 1
        current_order = _shuffled_row_indices(
            current_order,
            seed=_stable_seed(
                batch_seed,
                "rows",
                str(epoch_number),
                speaker_id,
                str(row_refills[speaker_id]),
            ),
        )
        row_orders[speaker_id] = current_order
        cursor = 0
    row_cursors[speaker_id] = cursor + 1
    return current_order[cursor]


def _shuffled_row_indices(row_indices: list[int], *, seed: int) -> list[int]:
    shuffled = list(row_indices)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _sample_crop_seconds(
    rng: random.Random,
    chunking_request: UtteranceChunkingRequest,
) -> float:
    minimum = chunking_request.train_min_crop_seconds
    maximum = chunking_request.train_max_crop_seconds
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-9):
        return minimum
    return round(rng.uniform(minimum, maximum), 6)


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


__all__ = ["Stage2BatchSampler"]
