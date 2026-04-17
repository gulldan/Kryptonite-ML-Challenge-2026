"""Production-grade training dataloader with balanced speaker sampling and resume state."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, cast

from torch.utils.data import DataLoader, Sampler

from kryptonite.config import ProjectConfig
from kryptonite.data import AudioLoadRequest, ManifestRow
from kryptonite.deployment import resolve_project_path
from kryptonite.features import FbankExtractionRequest, UtteranceChunkingRequest

from .augmentation_runtime import TrainingAugmentationRuntime
from .manifest_speaker_data import (
    ManifestSpeakerDataset,
    TrainingBatch,
    TrainingSampleRequest,
    collate_training_examples,
)


class BalancedSpeakerBatchSampler(Sampler[list[TrainingSampleRequest]]):
    def __init__(
        self,
        *,
        rows: list[ManifestRow],
        batch_size: int,
        seed: int,
        chunking_request: UtteranceChunkingRequest,
        augmentation_runtime: TrainingAugmentationRuntime | None = None,
        batches_per_epoch: int | None = None,
        domain_balance_external_share: float | None = None,
        domain_balance_external_source_prefixes: tuple[str, ...] = ("cnceleb", "ffsvc"),
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if chunking_request.train_num_crops != 1:
            raise ValueError("Production dataloader currently supports train_num_crops=1 only.")
        if domain_balance_external_share is not None and not (
            0.0 <= domain_balance_external_share <= 1.0
        ):
            raise ValueError("domain_balance_external_share must be within [0.0, 1.0]")

        self._rows = rows
        self._batch_size = batch_size
        self._seed = seed
        self._chunking_request = chunking_request
        self._augmentation_runtime = augmentation_runtime
        self._speaker_to_row_indices = _group_rows_by_speaker(rows)
        self._speaker_ids = sorted(self._speaker_to_row_indices)
        self._domain_balance_external_share = domain_balance_external_share
        self._external_source_prefixes = tuple(
            prefix.strip().lower() for prefix in domain_balance_external_source_prefixes
        )
        self._base_speaker_ids, self._external_speaker_ids = _split_speakers_by_domain(
            rows=rows,
            speaker_ids=self._speaker_ids,
            external_source_prefixes=self._external_source_prefixes,
        )
        if domain_balance_external_share is not None and (
            not self._base_speaker_ids or not self._external_speaker_ids
        ):
            raise ValueError(
                "Domain-balanced sampling requires both base and external-domain speakers."
            )
        self._batches_per_epoch = batches_per_epoch or max(1, math.ceil(len(rows) / batch_size))
        self._epoch = 0
        self._next_batch_index = 0

    def __len__(self) -> int:
        return max(0, self._batches_per_epoch - self._next_batch_index)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = epoch
        self._next_batch_index = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self._epoch,
            "next_batch_index": self._next_batch_index,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._epoch = int(state["epoch"])
        self._next_batch_index = int(state["next_batch_index"])

    def __iter__(self):
        start_index = self._next_batch_index
        for batch_index, batch in enumerate(self._iter_epoch_batches(self._epoch)):
            if batch_index < start_index:
                continue
            self._next_batch_index = batch_index + 1
            yield batch

    def _iter_epoch_batches(self, epoch: int) -> Iterator[list[TrainingSampleRequest]]:
        epoch_number = epoch + 1
        epoch_rng = random.Random(_stable_seed(self._seed, "epoch", str(epoch_number)))
        speaker_order = list(self._speaker_ids)
        epoch_rng.shuffle(speaker_order)
        speaker_cursor = 0
        row_orders = {
            speaker_id: _shuffled_row_indices(
                self._speaker_to_row_indices[speaker_id],
                seed=_stable_seed(self._seed, "rows", str(epoch_number), speaker_id, "0"),
            )
            for speaker_id in speaker_order
        }
        row_cursors = {speaker_id: 0 for speaker_id in speaker_order}
        row_refills = {speaker_id: 0 for speaker_id in speaker_order}
        base_speaker_order = list(self._base_speaker_ids)
        external_speaker_order = list(self._external_speaker_ids)
        epoch_rng.shuffle(base_speaker_order)
        epoch_rng.shuffle(external_speaker_order)
        domain_cursors = {"base": 0, "external": 0}

        for batch_index in range(self._batches_per_epoch):
            crop_seconds = _sample_crop_seconds(epoch_rng, self._chunking_request)
            if self._domain_balance_external_share is None:
                batch_speakers: list[str] = []
                selected_speakers: set[str] = set()
                unique_slots = min(self._batch_size, len(speaker_order))
                while len(batch_speakers) < unique_slots:
                    speaker_id, speaker_cursor = _next_speaker_id(
                        speaker_order=speaker_order,
                        speaker_cursor=speaker_cursor,
                        epoch_rng=epoch_rng,
                    )
                    if speaker_id in selected_speakers:
                        continue
                    selected_speakers.add(speaker_id)
                    batch_speakers.append(speaker_id)
                while len(batch_speakers) < self._batch_size:
                    speaker_id, speaker_cursor = _next_speaker_id(
                        speaker_order=speaker_order,
                        speaker_cursor=speaker_cursor,
                        epoch_rng=epoch_rng,
                    )
                    batch_speakers.append(speaker_id)
            else:
                batch_speakers = _domain_balanced_batch_speakers(
                    batch_size=self._batch_size,
                    external_share=self._domain_balance_external_share,
                    base_speaker_order=base_speaker_order,
                    external_speaker_order=external_speaker_order,
                    domain_cursors=domain_cursors,
                    epoch_rng=epoch_rng,
                )

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
                request_seed = _stable_seed(
                    self._seed,
                    "request",
                    str(epoch_number),
                    str(batch_index),
                    str(position_in_batch),
                    str(row_index),
                )
                recipe = (
                    self._augmentation_runtime.sample_recipe(
                        epoch=epoch_number,
                        rng=random.Random(request_seed),
                    )
                    if self._augmentation_runtime is not None
                    else None
                )
                batch_requests.append(
                    TrainingSampleRequest(
                        row_index=row_index,
                        request_seed=request_seed,
                        crop_seconds=crop_seconds,
                        clean_sample=True if recipe is None else recipe.clean_sample,
                        recipe_stage="steady" if recipe is None else recipe.stage,
                        recipe_intensity="clean" if recipe is None else recipe.intensity,
                        augmentations=() if recipe is None else recipe.augmentations,
                    )
                )
            yield batch_requests


def build_production_train_dataloader(
    *,
    rows: list[ManifestRow],
    speaker_to_index: dict[str, int],
    project: ProjectConfig,
    total_epochs: int,
    pin_memory: bool,
    batches_per_epoch: int | None = None,
) -> tuple[ManifestSpeakerDataset, BalancedSpeakerBatchSampler, DataLoader[TrainingBatch]]:
    project_root = resolve_project_path(project.paths.project_root, ".")
    audio_request = AudioLoadRequest.from_config(
        project.normalization,
        vad=project.vad,
    )
    feature_request = FbankExtractionRequest.from_config(project.features)
    chunking_request = UtteranceChunkingRequest.from_config(project.chunking)
    if chunking_request.train_num_crops != 1:
        raise ValueError("Production dataloader currently supports train_num_crops=1 only.")

    augmentation_runtime = TrainingAugmentationRuntime.from_project_config(
        project_root=project_root,
        scheduler_config=project.augmentation_scheduler,
        silence_config=project.silence_augmentation,
        total_epochs=total_epochs,
    )
    if (
        not project.augmentation_scheduler.enabled
        or not augmentation_runtime.has_effective_augmentation
    ):
        augmentation_runtime = None

    dataset = ManifestSpeakerDataset(
        rows=rows,
        speaker_to_index=speaker_to_index,
        project_root=project_root,
        audio_request=audio_request,
        feature_request=feature_request,
        chunking_request=chunking_request,
        seed=project.runtime.seed,
        augmentation_runtime=augmentation_runtime,
    )
    sampler = BalancedSpeakerBatchSampler(
        rows=rows,
        batch_size=project.training.batch_size,
        seed=project.runtime.seed,
        chunking_request=chunking_request,
        augmentation_runtime=augmentation_runtime,
        batches_per_epoch=batches_per_epoch,
        domain_balance_external_share=(
            project.training.domain_balance_external_share
            if project.training.domain_balance_enabled
            else None
        ),
        domain_balance_external_source_prefixes=tuple(
            project.training.domain_balance_external_source_prefixes
        ),
    )
    num_workers = project.runtime.num_workers
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_training_examples,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = cast(DataLoader[TrainingBatch], DataLoader(**loader_kwargs))
    return dataset, sampler, loader


def _group_rows_by_speaker(rows: list[ManifestRow]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row.speaker_id].append(index)
    if len(grouped) < 2:
        raise ValueError("Production dataloader requires at least two speakers.")
    return dict(grouped)


def _split_speakers_by_domain(
    *,
    rows: list[ManifestRow],
    speaker_ids: list[str],
    external_source_prefixes: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    speaker_is_external: dict[str, bool] = {}
    for row in rows:
        source = row.source_dataset.lower()
        is_external = source.startswith(external_source_prefixes)
        speaker_is_external[row.speaker_id] = (
            speaker_is_external.get(row.speaker_id, False) or is_external
        )
    base_speakers = [
        speaker_id for speaker_id in speaker_ids if not speaker_is_external.get(speaker_id, False)
    ]
    external_speakers = [
        speaker_id for speaker_id in speaker_ids if speaker_is_external.get(speaker_id, False)
    ]
    return base_speakers, external_speakers


def _domain_balanced_batch_speakers(
    *,
    batch_size: int,
    external_share: float,
    base_speaker_order: list[str],
    external_speaker_order: list[str],
    domain_cursors: dict[str, int],
    epoch_rng: random.Random,
) -> list[str]:
    external_slots = int(round(batch_size * external_share))
    if external_share > 0.0:
        external_slots = max(1, external_slots)
    if external_share < 1.0:
        external_slots = min(batch_size - 1, external_slots)
    base_slots = batch_size - external_slots
    batch_speakers = [
        *_take_domain_speakers(
            speaker_order=base_speaker_order,
            count=base_slots,
            cursor_key="base",
            domain_cursors=domain_cursors,
            epoch_rng=epoch_rng,
        ),
        *_take_domain_speakers(
            speaker_order=external_speaker_order,
            count=external_slots,
            cursor_key="external",
            domain_cursors=domain_cursors,
            epoch_rng=epoch_rng,
        ),
    ]
    epoch_rng.shuffle(batch_speakers)
    return batch_speakers


def _take_domain_speakers(
    *,
    speaker_order: list[str],
    count: int,
    cursor_key: str,
    domain_cursors: dict[str, int],
    epoch_rng: random.Random,
) -> list[str]:
    speakers: list[str] = []
    selected: set[str] = set()
    unique_slots = min(count, len(speaker_order))
    cursor = domain_cursors[cursor_key]
    while len(speakers) < unique_slots:
        speaker_id, cursor = _next_speaker_id(
            speaker_order=speaker_order,
            speaker_cursor=cursor,
            epoch_rng=epoch_rng,
        )
        if speaker_id in selected:
            continue
        selected.add(speaker_id)
        speakers.append(speaker_id)
    while len(speakers) < count:
        speaker_id, cursor = _next_speaker_id(
            speaker_order=speaker_order,
            speaker_cursor=cursor,
            epoch_rng=epoch_rng,
        )
        speakers.append(speaker_id)
    domain_cursors[cursor_key] = cursor
    return speakers


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


__all__ = [
    "BalancedSpeakerBatchSampler",
    "build_production_train_dataloader",
]
