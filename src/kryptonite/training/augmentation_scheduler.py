"""Epoch-aware waveform augmentation scheduler for speaker training."""

from __future__ import annotations

import json
import random
from bisect import bisect_left
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from kryptonite.config import AugmentationSchedulerConfig, SilenceAugmentationConfig

AugmentationFamily = Literal["noise", "reverb", "distance", "codec", "silence", "speed"]
AugmentationSeverity = Literal["light", "medium", "heavy"]
RecipeIntensity = Literal["clean", "light", "medium", "heavy"]
RecipeStage = Literal["warmup", "ramp", "steady"]

_INTENSITIES: tuple[RecipeIntensity, ...] = ("clean", "light", "medium", "heavy")
_FAMILIES: tuple[AugmentationFamily, ...] = (
    "noise",
    "reverb",
    "distance",
    "codec",
    "silence",
    "speed",
)
_SEVERITY_ORDER: dict[AugmentationSeverity, int] = {"light": 0, "medium": 1, "heavy": 2}


@dataclass(frozen=True, slots=True)
class AugmentationCandidate:
    family: AugmentationFamily
    candidate_id: str
    label: str
    severity: AugmentationSeverity
    weight: float = 1.0
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError(f"Unsupported augmentation family: {self.family!r}")
        if self.severity not in _SEVERITY_ORDER:
            raise ValueError(f"Unsupported augmentation severity: {self.severity!r}")
        if self.weight <= 0.0:
            raise ValueError("augmentation candidate weight must be positive")


_CandidatePool = tuple[tuple[AugmentationCandidate, ...], tuple[float, ...], float]


@dataclass(frozen=True, slots=True)
class ScheduledAugmentation:
    family: AugmentationFamily
    candidate_id: str
    label: str
    severity: AugmentationSeverity
    metadata: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "candidate_id": self.candidate_id,
            "label": self.label,
            "severity": self.severity,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class ScheduledSampleRecipe:
    stage: RecipeStage
    intensity: RecipeIntensity
    clean_sample: bool
    augmentations: tuple[ScheduledAugmentation, ...]


@dataclass(frozen=True, slots=True)
class EpochAugmentationPlan:
    epoch: int
    stage: RecipeStage
    intensity_probabilities: dict[RecipeIntensity, float]
    family_probabilities: dict[AugmentationFamily, float]
    max_augmentations_per_sample: int


@dataclass(frozen=True, slots=True)
class AugmentationCatalog:
    candidates_by_family: Mapping[AugmentationFamily, tuple[AugmentationCandidate, ...]]

    def __post_init__(self) -> None:
        normalized: dict[AugmentationFamily, tuple[AugmentationCandidate, ...]] = {}
        for family, candidates in self.candidates_by_family.items():
            if family not in _FAMILIES:
                raise ValueError(f"Unsupported augmentation family: {family!r}")
            normalized[family] = tuple(candidates)
        object.__setattr__(self, "candidates_by_family", normalized)

    @property
    def candidate_counts_by_family(self) -> dict[str, int]:
        return {
            family: len(self.candidates_by_family.get(family, ()))
            for family in _FAMILIES
            if self.candidates_by_family.get(family, ())
        }

    @property
    def available_families(self) -> tuple[AugmentationFamily, ...]:
        return tuple(family for family in _FAMILIES if self.candidates_by_family.get(family, ()))


class AugmentationScheduler:
    def __init__(
        self,
        *,
        config: AugmentationSchedulerConfig,
        catalog: AugmentationCatalog,
        total_epochs: int,
    ) -> None:
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        self._config = config
        self._catalog = catalog
        self._total_epochs = total_epochs
        self._candidate_pool_cache: dict[tuple[AugmentationFamily, int], _CandidatePool] = {}

    def plan_for_epoch(self, epoch: int) -> EpochAugmentationPlan:
        if epoch <= 0:
            raise ValueError("epoch must be one-based and positive")
        stage = _stage_for_epoch(
            epoch,
            warmup_epochs=self._config.warmup_epochs,
            ramp_epochs=self._config.ramp_epochs,
        )
        progress = _ramp_progress(
            epoch,
            total_epochs=self._total_epochs,
            warmup_epochs=self._config.warmup_epochs,
            ramp_epochs=self._config.ramp_epochs,
        )
        intensity_probabilities = _normalize_probabilities(
            {
                "clean": _lerp(
                    self._config.clean_probability_start,
                    self._config.clean_probability_end,
                    progress,
                ),
                "light": _lerp(
                    self._config.light_probability_start,
                    self._config.light_probability_end,
                    progress,
                ),
                "medium": _lerp(
                    self._config.medium_probability_start,
                    self._config.medium_probability_end,
                    progress,
                ),
                "heavy": _lerp(
                    self._config.heavy_probability_start,
                    self._config.heavy_probability_end,
                    progress,
                ),
            }
        )
        family_probabilities = _normalize_probabilities(
            {
                family: _family_weight(self._config, family)
                * _stage_family_multiplier(stage, family)
                for family in self._catalog.available_families
            }
        )
        return EpochAugmentationPlan(
            epoch=epoch,
            stage=stage,
            intensity_probabilities={
                intensity: intensity_probabilities.get(intensity, 0.0) for intensity in _INTENSITIES
            },
            family_probabilities=family_probabilities,
            max_augmentations_per_sample=(
                1 if stage == "warmup" else self._config.max_augmentations_per_sample
            ),
        )

    def sample_recipe(self, *, epoch: int, rng: random.Random) -> ScheduledSampleRecipe:
        if not self._config.enabled or not self._catalog.available_families:
            return ScheduledSampleRecipe(
                stage="steady",
                intensity="clean",
                clean_sample=True,
                augmentations=(),
            )
        plan = self.plan_for_epoch(epoch)
        intensity = _weighted_choice(plan.intensity_probabilities, rng=rng)
        if intensity == "clean" or not plan.family_probabilities:
            return ScheduledSampleRecipe(
                stage=plan.stage,
                intensity="clean",
                clean_sample=True,
                augmentations=(),
            )

        augmentation_count = _augmentation_count_for_intensity(
            intensity,
            max_augmentations_per_sample=plan.max_augmentations_per_sample,
        )
        selected_families = _sample_distinct_families(
            plan.family_probabilities,
            count=augmentation_count,
            rng=rng,
        )
        scheduled = tuple(
            self._schedule_family_augmentation(
                family,
                intensity=intensity,
                rng=rng,
            )
            for family in selected_families
        )
        return ScheduledSampleRecipe(
            stage=plan.stage,
            intensity=intensity,
            clean_sample=False,
            augmentations=scheduled,
        )

    def _schedule_family_augmentation(
        self,
        family: AugmentationFamily,
        *,
        intensity: RecipeIntensity,
        rng: random.Random,
    ) -> ScheduledAugmentation:
        candidates = self._catalog.candidates_by_family.get(family, ())
        if not candidates:
            raise ValueError(f"No candidates available for augmentation family {family!r}")
        if intensity == "clean":
            raise ValueError("clean recipes do not schedule augmentations")
        max_severity = _SEVERITY_ORDER[intensity]
        pool, cumulative_weights, total_weight = self._candidate_pool(
            family=family,
            max_severity=max_severity,
            candidates=candidates,
        )
        chosen = _weighted_candidate_choice_from_cumulative(
            pool,
            cumulative_weights=cumulative_weights,
            total_weight=total_weight,
            rng=rng,
        )
        return ScheduledAugmentation(
            family=chosen.family,
            candidate_id=chosen.candidate_id,
            label=chosen.label,
            severity=chosen.severity,
            metadata=dict(chosen.metadata or {}),
        )

    def _candidate_pool(
        self,
        *,
        family: AugmentationFamily,
        max_severity: int,
        candidates: tuple[AugmentationCandidate, ...],
    ) -> _CandidatePool:
        cache_key = (family, max_severity)
        cached = self._candidate_pool_cache.get(cache_key)
        if cached is not None:
            return cached

        allowed = tuple(
            candidate
            for candidate in candidates
            if _SEVERITY_ORDER[candidate.severity] <= max_severity
        )
        pool = allowed or candidates
        cumulative: list[float] = []
        total = 0.0
        for candidate in pool:
            total += candidate.weight
            cumulative.append(total)
        if total <= 0.0:
            raise ValueError(f"No positive weights available for {family!r} candidates")
        resolved = (pool, tuple(cumulative), total)
        self._candidate_pool_cache[cache_key] = resolved
        return resolved


@dataclass(frozen=True, slots=True)
class ResolvedBankManifestPaths:
    noise_manifest_path: str
    room_config_manifest_path: str
    rir_manifest_path: str
    distance_manifest_path: str
    codec_manifest_path: str


@dataclass(frozen=True, slots=True)
class AugmentationEpochReport:
    epoch: int
    stage: RecipeStage
    target_intensity_probabilities: dict[str, float]
    empirical_intensity_ratios: dict[str, float]
    family_coverage: dict[str, int]
    severity_coverage: dict[str, int]
    top_labels: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class AugmentationSchedulerReportSummary:
    missing_families: tuple[str, ...]
    overall_family_counts: dict[str, int]
    overall_severity_counts: dict[str, int]
    overall_intensity_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class AugmentationSchedulerReport:
    catalog: AugmentationCatalog
    epochs: tuple[AugmentationEpochReport, ...]
    summary: AugmentationSchedulerReportSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog": {
                "candidate_counts_by_family": self.catalog.candidate_counts_by_family,
            },
            "epochs": [asdict(epoch) for epoch in self.epochs],
            "summary": asdict(self.summary),
        }


@dataclass(frozen=True, slots=True)
class WrittenAugmentationSchedulerReport:
    json_path: str
    markdown_path: str
    epochs_path: str


def resolve_bank_manifest_paths(*, project_root: Path | str = ".") -> ResolvedBankManifestPaths:
    root = Path(project_root)
    noise = root / "artifacts/corruptions/noise-bank/manifests/noise_bank_manifest.jsonl"
    room = root / "artifacts/corruptions/rir-bank/manifests/room_simulation_configs.jsonl"
    if not room.exists():
        candidates = sorted(
            (root / "artifacts/corruptions").glob(
                "rir-bank*/manifests/room_simulation_configs.jsonl"
            )
        )
        if candidates:
            room = candidates[0]
    rir = room.parent / "rir_bank_manifest.jsonl"
    distance = root / "artifacts/corruptions/far-field-bank/manifests/far_field_bank_manifest.jsonl"
    codec = root / "artifacts/corruptions/codec-bank/manifests/codec_bank_manifest.jsonl"
    return ResolvedBankManifestPaths(
        noise_manifest_path=str(noise),
        room_config_manifest_path=str(room),
        rir_manifest_path=str(rir),
        distance_manifest_path=str(distance),
        codec_manifest_path=str(codec),
    )


def build_augmentation_scheduler_report(
    *,
    project_root: Path | str,
    scheduler_config: AugmentationSchedulerConfig,
    silence_config: SilenceAugmentationConfig,
    total_epochs: int,
    samples_per_epoch: int,
    seed: int = 42,
    noise_manifest_path: Path | str | None = None,
    room_config_manifest_path: Path | str | None = None,
    rir_manifest_path: Path | str | None = None,
    distance_manifest_path: Path | str | None = None,
    codec_manifest_path: Path | str | None = None,
) -> AugmentationSchedulerReport:
    from .augmentation_runtime import build_augmentation_catalog_from_manifests

    resolved = resolve_bank_manifest_paths(project_root=project_root)
    catalog = build_augmentation_catalog_from_manifests(
        project_root=project_root,
        scheduler_config=scheduler_config,
        silence_config=silence_config,
        noise_manifest_path=noise_manifest_path or resolved.noise_manifest_path,
        room_config_manifest_path=room_config_manifest_path or resolved.room_config_manifest_path,
        rir_manifest_path=rir_manifest_path or resolved.rir_manifest_path,
        distance_manifest_path=distance_manifest_path or resolved.distance_manifest_path,
        codec_manifest_path=codec_manifest_path or resolved.codec_manifest_path,
    )
    scheduler = AugmentationScheduler(
        config=scheduler_config,
        catalog=catalog,
        total_epochs=total_epochs,
    )
    epoch_reports: list[AugmentationEpochReport] = []
    overall_family_counts: Counter[str] = Counter()
    overall_severity_counts: Counter[str] = Counter()
    overall_intensity_counts: Counter[str] = Counter()
    for epoch in range(1, total_epochs + 1):
        plan = scheduler.plan_for_epoch(epoch)
        intensity_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        for sample_index in range(samples_per_epoch):
            rng = random.Random((seed * 1_000_003) + (epoch * 10_007) + sample_index)
            recipe = scheduler.sample_recipe(epoch=epoch, rng=rng)
            intensity_counts[recipe.intensity] += 1
            for augmentation in recipe.augmentations:
                family_counts[augmentation.family] += 1
                severity_counts[augmentation.severity] += 1
                label_counts[augmentation.label] += 1
        overall_family_counts.update(family_counts)
        overall_severity_counts.update(severity_counts)
        overall_intensity_counts.update(intensity_counts)
        epoch_reports.append(
            AugmentationEpochReport(
                epoch=epoch,
                stage=plan.stage,
                target_intensity_probabilities={
                    key: round(value, 6) for key, value in plan.intensity_probabilities.items()
                },
                empirical_intensity_ratios={
                    key: round(float(intensity_counts.get(key, 0)) / samples_per_epoch, 6)
                    for key in _INTENSITIES
                },
                family_coverage=dict(sorted(family_counts.items())),
                severity_coverage=dict(sorted(severity_counts.items())),
                top_labels=tuple(label_counts.most_common(10)),
            )
        )
    required_families = tuple(
        family
        for family in _FAMILIES
        if family != "speed" or float(getattr(scheduler_config.family_weights, "speed", 0.0)) > 0.0
    )
    missing = tuple(
        family for family in required_families if family not in catalog.available_families
    )
    return AugmentationSchedulerReport(
        catalog=catalog,
        epochs=tuple(epoch_reports),
        summary=AugmentationSchedulerReportSummary(
            missing_families=missing,
            overall_family_counts=dict(sorted(overall_family_counts.items())),
            overall_severity_counts=dict(sorted(overall_severity_counts.items())),
            overall_intensity_counts=dict(sorted(overall_intensity_counts.items())),
        ),
    )


def write_augmentation_scheduler_report(
    *,
    report: AugmentationSchedulerReport,
    output_root: Path | str,
) -> WrittenAugmentationSchedulerReport:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "augmentation_scheduler_report.json"
    markdown_path = output_path / "augmentation_scheduler_report.md"
    epochs_path = output_path / "augmentation_scheduler_epochs.jsonl"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    epochs_path.write_text(
        "".join(json.dumps(asdict(epoch), sort_keys=True) + "\n" for epoch in report.epochs),
        encoding="utf-8",
    )
    markdown_path.write_text(_render_report_markdown(report), encoding="utf-8")
    return WrittenAugmentationSchedulerReport(
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        epochs_path=str(epochs_path),
    )


def _render_report_markdown(report: AugmentationSchedulerReport) -> str:
    lines = [
        "# Augmentation Scheduler Report",
        "",
        "## Catalog",
        "",
    ]
    for family, count in sorted(report.catalog.candidate_counts_by_family.items()):
        lines.append(f"- `{family}`: `{count}` candidates")
    lines.extend(["", "## Epochs", ""])
    for epoch in report.epochs:
        lines.append(f"### Epoch {epoch.epoch}")
        lines.append("")
        lines.append(f"- Stage: `{epoch.stage}`")
        lines.append(f"- Empirical intensity ratios: `{epoch.empirical_intensity_ratios}`")
        lines.append(f"- Family coverage: `{epoch.family_coverage}`")
        lines.append("")
    return "\n".join(lines)


def _stage_for_epoch(
    epoch: int,
    *,
    warmup_epochs: int,
    ramp_epochs: int,
) -> RecipeStage:
    if epoch <= warmup_epochs:
        return "warmup"
    if epoch <= warmup_epochs + ramp_epochs:
        return "ramp"
    return "steady"


def _ramp_progress(
    epoch: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    if epoch <= warmup_epochs + ramp_epochs:
        return min(1.0, max(0.0, (epoch - warmup_epochs) / ramp_epochs))
    if epoch >= total_epochs:
        return 1.0
    return 1.0


def _stage_family_multiplier(stage: RecipeStage, family: AugmentationFamily) -> float:
    if stage != "warmup":
        return 1.0
    return {
        "codec": 0.45,
        "distance": 0.65,
        "reverb": 0.75,
        "silence": 0.80,
        "speed": 0.85,
    }.get(family, 1.0)


def _family_weight(config: AugmentationSchedulerConfig, family: AugmentationFamily) -> float:
    return float(getattr(config.family_weights, family, 1.0))


def _lerp(start: float, end: float, progress: float) -> float:
    return start + ((end - start) * progress)


def _normalize_probabilities[T: str](values: Mapping[T, float]) -> dict[T, float]:
    positive = {key: max(0.0, float(value)) for key, value in values.items() if value > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in positive.items()}


def _weighted_choice[T: str](weights: Mapping[T, float], *, rng: random.Random) -> T:
    if not weights:
        raise ValueError("cannot sample from empty weights")
    threshold = rng.random() * sum(weights.values())
    cumulative = 0.0
    last_key: T | None = None
    for key, value in weights.items():
        last_key = key
        cumulative += value
        if threshold <= cumulative:
            return key
    assert last_key is not None
    return last_key


def _weighted_candidate_choice(
    candidates: tuple[AugmentationCandidate, ...],
    *,
    rng: random.Random,
) -> AugmentationCandidate:
    weights = {str(index): candidate.weight for index, candidate in enumerate(candidates)}
    selected_index = int(_weighted_choice(weights, rng=rng))
    return candidates[selected_index]


def _weighted_candidate_choice_from_cumulative(
    candidates: tuple[AugmentationCandidate, ...],
    *,
    cumulative_weights: tuple[float, ...],
    total_weight: float,
    rng: random.Random,
) -> AugmentationCandidate:
    threshold = rng.random() * total_weight
    selected_index = bisect_left(cumulative_weights, threshold)
    if selected_index >= len(candidates):
        selected_index = len(candidates) - 1
    return candidates[selected_index]


def _augmentation_count_for_intensity(
    intensity: RecipeIntensity,
    *,
    max_augmentations_per_sample: int,
) -> int:
    if intensity == "light":
        return 1
    if intensity == "medium":
        return min(2, max_augmentations_per_sample)
    return max_augmentations_per_sample


def _sample_distinct_families(
    weights: Mapping[AugmentationFamily, float],
    *,
    count: int,
    rng: random.Random,
) -> tuple[AugmentationFamily, ...]:
    remaining = dict(weights)
    selected: list[AugmentationFamily] = []
    for _ in range(min(count, len(remaining))):
        family = _weighted_choice(remaining, rng=rng)
        selected.append(family)
        remaining.pop(family, None)
    return tuple(selected)


__all__ = [
    "AugmentationCandidate",
    "AugmentationCatalog",
    "AugmentationEpochReport",
    "AugmentationFamily",
    "AugmentationScheduler",
    "AugmentationSchedulerReport",
    "AugmentationSchedulerReportSummary",
    "AugmentationSeverity",
    "EpochAugmentationPlan",
    "RecipeIntensity",
    "RecipeStage",
    "ResolvedBankManifestPaths",
    "ScheduledAugmentation",
    "ScheduledSampleRecipe",
    "WrittenAugmentationSchedulerReport",
    "build_augmentation_scheduler_report",
    "resolve_bank_manifest_paths",
    "write_augmentation_scheduler_report",
]
