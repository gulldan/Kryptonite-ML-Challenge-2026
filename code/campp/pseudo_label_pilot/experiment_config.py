from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_EXPERIMENT_LOG_RELATIVE = Path("docs/experiment_log.md")
_MAKE_DURATION_MATCHED_SAMPLE_RELATIVE = Path("code/campp/make_duration_matched_sample.py")
_EXPECTED_BACKBONE_FAMILY = "CAM++"
_EXPECTED_PRIMARY_METRIC = "public_MS32"
_EXPECTED_SELECTOR_GATE_METRIC = "simulated_pairwise_purity"
_EXPECTED_LOCAL_PROBE_METRIC = "validation_precision@10"
_EXPECTED_STAGE0_COVERAGE_METRIC = "accepted_pseudo_rows"
_SELECTOR_STAGE = "selector"
_PROBE_STAGE = "probe"
_ROUND2_STAGE = "round2"
_VALID_STAGES = {_SELECTOR_STAGE, _PROBE_STAGE, _ROUND2_STAGE}
_REQUIRED_SEEDS = (0, 1, 2)
_MAX_CONDITION_COUNT = 8


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(payload)!r}")
    return payload


def _extract_literal_assignment(module_path: Path, variable_name: str) -> Any:
    source = module_path.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(module_path))
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == variable_name and node.value is not None:
                return ast.literal_eval(node.value)
    raise KeyError(f"Could not find literal assignment for {variable_name!r} in {module_path}")


@dataclass(slots=True)
class PathsConfig:
    repo_root: str = "."
    campp_base_config: str = "code/campp/configs/campp_en_ft.base.yaml"
    full_train_manifest: str = "data/campp_runs/campp_en_ft/prepared/train_manifest.csv"
    val_unlabeled_manifest: str = "data/campp_runs/campp_en_ft/prepared/val_manifest.csv"
    heldout_test_manifest: str = "data/campp_runs/campp_en_ft/prepared/test_manifest.csv"
    sample_v1_manifest: str = (
        "data/train_duration_matched_sample_v1/train_manifest_duration_matched.csv"
    )
    sample_v2_manifest: str = (
        "data/train_duration_matched_sample_v2/train_manifest_duration_matched.csv"
    )
    acoustic_parquet: str = "data/eda/full/acoustic_all/cache/acoustic_sample.parquet"
    test_public_csv: str = "data/Для участников/test_public.csv"
    experiment_root: str = "data/campp_runs/pseudo_label_pilot"
    results_json: str = "data/campp_runs/pseudo_label_pilot/results.json"
    cache_dir: str = "data/campp_runs/pseudo_label_pilot/cache"

    def _resolve_value(self, value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path(self.repo_root).resolve() / candidate).resolve()

    def resolve_all(self) -> dict[str, Path]:
        repo_root = Path(self.repo_root).resolve()
        resolved: dict[str, Path] = {"repo_root": repo_root}
        for field_name in (
            "campp_base_config",
            "full_train_manifest",
            "val_unlabeled_manifest",
            "heldout_test_manifest",
            "sample_v1_manifest",
            "sample_v2_manifest",
            "acoustic_parquet",
            "test_public_csv",
            "experiment_root",
            "results_json",
            "cache_dir",
        ):
            resolved[field_name] = self._resolve_value(getattr(self, field_name))
        resolved["experiment_log"] = (repo_root / _EXPERIMENT_LOG_RELATIVE).resolve()
        resolved["make_duration_matched_sample"] = (
            repo_root / _MAKE_DURATION_MATCHED_SAMPLE_RELATIVE
        ).resolve()
        return resolved

    def validate_exists(self) -> None:
        resolved = self.resolve_all()
        required_keys = (
            "campp_base_config",
            "full_train_manifest",
            "val_unlabeled_manifest",
            "heldout_test_manifest",
            "sample_v1_manifest",
            "sample_v2_manifest",
            "acoustic_parquet",
            "test_public_csv",
            "experiment_log",
            "make_duration_matched_sample",
        )
        missing_keys = [key for key in required_keys if not resolved[key].exists()]
        if missing_keys:
            missing_key = missing_keys[0]
            raise FileNotFoundError(f"Required repository artifact is missing: {missing_key}")

        repo_root = resolved["repo_root"]
        code_root = (repo_root / "code").resolve()
        data_root = (repo_root / "data").resolve()
        experiment_root = resolved["experiment_root"]
        if _is_relative_to(experiment_root, code_root):
            raise ValueError("experiment_root must live under data/, not inside code/")
        if not _is_relative_to(experiment_root, data_root):
            raise ValueError("experiment_root must be rooted under repo data/")


@dataclass(slots=True)
class ConditionSpec:
    name: str
    class_name: str
    stage: str
    enabled_by_default: bool
    depends_on: list[str]
    seed_count: int
    max_seconds: int

    def __post_init__(self) -> None:
        if self.stage not in _VALID_STAGES:
            raise ValueError(f"Unsupported condition stage: {self.stage!r}")
        if self.seed_count <= 0:
            raise ValueError(f"seed_count must be positive for {self.name}")
        if self.max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive for {self.name}")
        if not self.name.strip():
            raise ValueError("Condition name must not be empty")
        if not self.class_name.strip():
            raise ValueError(f"class_name must not be empty for {self.name}")

    def is_runnable(self, context: dict[str, object]) -> bool:
        stage_enabled = self.enabled_by_default
        enabled_stages = context.get("enabled_stages")
        if isinstance(enabled_stages, Mapping):
            stage_enabled = stage_enabled and bool(enabled_stages.get(self.stage, True))
        else:
            explicit_flag_key = {
                _SELECTOR_STAGE: "allow_selectors",
                _PROBE_STAGE: "allow_stage1",
                _ROUND2_STAGE: "allow_stage2",
            }[self.stage]
            if explicit_flag_key in context:
                stage_enabled = stage_enabled and bool(context[explicit_flag_key])

        if not stage_enabled:
            return False

        available_conditions_obj = context.get("available_conditions")
        if not isinstance(available_conditions_obj, Mapping):
            available_conditions_obj = context.get("completed_conditions")
        if not isinstance(available_conditions_obj, Mapping):
            available_conditions_obj = context.get("passed_conditions", context)
        if not isinstance(available_conditions_obj, Mapping):
            return False

        for dependency_name in self.depends_on:
            dependency_payload = available_conditions_obj.get(dependency_name)
            if dependency_payload is None:
                return False
        return True


@dataclass(slots=True)
class ExperimentConfig:
    paths: PathsConfig
    v2_features: list[str] = field(default_factory=list)
    v2_weights: dict[str, float] = field(default_factory=dict)
    knn_k: int = 30
    top1_quantile_by_regime: float = 0.75
    min_top1_cosine: float = 0.55
    min_margin: float = 0.015
    min_component_size: int = 3
    strict_max_component_size: int | None = 24
    strict_min_top1_score: float | None = 0.58
    strict_min_top1_margin: float | None = 0.02
    strict_indegree_top_k: int | None = 10
    hubness_quantile: float = 0.9
    prior_distance_quantile: float = 0.9
    diversity_floor_quantile: float = 0.1
    strict_max_rows_per_component: int | None = 12
    selector_pass_purity: float = 0.975
    selector_soft_min_rows: int = 3000
    selector_target_rows: int = 5000
    stage0_budget_seconds: int = 180
    stage1_default_enabled: bool = False
    stage2_default_enabled: bool = False
    final_seed_count: int = 10
    fallback_seed_count: int = 5
    selector_seed: int = 42
    bootstrap_samples: int = 10000
    official_mode: str = "segment_mean"
    official_segment_count: int = 3
    official_topk: int = 10
    one_epoch_probe_epochs: int = 1
    round2_overlap_gate: float = 0.95
    round2_margin_gain_gate: float = 0.01
    round2_retention_gate: float = 0.4
    round2_max_drop_p10: float = 0.002

    def __post_init__(self) -> None:
        if not self.v2_features:
            raise ValueError("v2_features must not be empty")
        if set(self.v2_features) != set(self.v2_weights):
            raise ValueError("v2_weights keys must match v2_features exactly")
        required_seed_count = len(_REQUIRED_SEEDS)
        if self.final_seed_count != required_seed_count:
            raise ValueError("final_seed_count must remain fixed at 3 to enforce seeds [0, 1, 2]")
        if self.fallback_seed_count != required_seed_count:
            raise ValueError(
                "fallback_seed_count must remain fixed at 3 to enforce seeds [0, 1, 2]"
            )
        if self.strict_max_component_size is not None:
            if self.strict_max_component_size < self.min_component_size:
                raise ValueError("strict_max_component_size must be >= min_component_size")
        if self.strict_min_top1_score is not None and not (
            0.0 <= self.strict_min_top1_score <= 1.0
        ):
            raise ValueError("strict_min_top1_score must satisfy 0 <= value <= 1")
        if self.strict_min_top1_margin is not None and self.strict_min_top1_margin < 0.0:
            raise ValueError("strict_min_top1_margin must be non-negative")
        if self.strict_indegree_top_k is not None and self.strict_indegree_top_k <= 0:
            raise ValueError("strict_indegree_top_k must be positive when provided")
        if self.strict_max_rows_per_component is not None:
            if self.strict_max_rows_per_component <= 0:
                raise ValueError("strict_max_rows_per_component must be positive when provided")
        if self.official_topk != 10:
            raise ValueError("official_topk must remain fixed at 10")
        if self.official_mode != "segment_mean":
            raise ValueError("official_mode must remain fixed at segment_mean")
        if self.official_segment_count != 3:
            raise ValueError("official_segment_count must remain fixed at 3")

    def seed_values(self, count: int | None = None) -> list[int]:
        required_seed_count = len(_REQUIRED_SEEDS)
        if count is not None and int(count) != required_seed_count:
            raise ValueError("seed count must remain fixed at 3 to enforce seeds [0, 1, 2]")
        return list(_REQUIRED_SEEDS)

    @classmethod
    def from_repo_defaults(
        cls,
        repo_root: str,
        allow_stage1: bool = False,
        allow_stage2: bool = False,
        seed_count: int | None = None,
    ) -> ExperimentConfig:
        del allow_stage1, allow_stage2
        paths = PathsConfig(repo_root=repo_root)
        resolved = paths.resolve_all()
        features = _extract_literal_assignment(
            resolved["make_duration_matched_sample"], "V2_FEATURES"
        )
        weights = _extract_literal_assignment(
            resolved["make_duration_matched_sample"], "V2_WEIGHTS"
        )
        if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
            raise TypeError("V2_FEATURES must be a list[str]")
        if not isinstance(weights, dict) or not all(isinstance(key, str) for key in weights):
            raise TypeError("V2_WEIGHTS must be a dict[str, float]")

        required_seed_count = len(_REQUIRED_SEEDS)
        if seed_count is not None and int(seed_count) != required_seed_count:
            raise ValueError("seed_count must remain fixed at 3 to enforce seeds [0, 1, 2]")

        return cls(
            paths=paths,
            v2_features=list(features),
            v2_weights={key: float(value) for key, value in weights.items()},
            stage1_default_enabled=True,
            stage2_default_enabled=True,
            final_seed_count=required_seed_count,
            fallback_seed_count=required_seed_count,
        )

    def validate_repo_constraints(self) -> None:
        self.paths.validate_exists()
        resolved = self.paths.resolve_all()
        repo_root = resolved["repo_root"]
        data_root = (repo_root / "data").resolve()
        experiment_root = resolved["experiment_root"]
        if not _is_relative_to(experiment_root, data_root):
            raise ValueError("experiment_root must stay inside repo data/")

        base_config = _load_yaml_mapping(resolved["campp_base_config"])
        training = base_config.get("training", {})
        evaluation = base_config.get("evaluation", {})
        pretrained = base_config.get("pretrained", {})
        if float(training.get("scale", -1.0)) != 32.0:
            raise ValueError("CAM++ training.scale must remain fixed at 32.0")
        if str(evaluation.get("primary_mode", "")) != self.official_mode:
            raise ValueError("CAM++ evaluation.primary_mode must remain segment_mean")
        if int(evaluation.get("segment_count", -1)) != self.official_segment_count:
            raise ValueError("CAM++ evaluation.segment_count must remain 3")
        compare_modes = evaluation.get("compare_modes", [])
        if self.official_mode not in compare_modes:
            raise ValueError("CAM++ compare_modes must include segment_mean")
        model_id = str(pretrained.get("model_id", ""))
        weight_filename = str(pretrained.get("weight_filename", ""))
        if "campplus" not in model_id.lower() and "campplus" not in weight_filename.lower():
            raise ValueError(f"Backbone family must remain {_EXPECTED_BACKBONE_FAMILY}")
        if self.official_topk != 10:
            raise ValueError("Official retrieval top-k must remain fixed at 10")
        if not _is_relative_to(resolved["results_json"], data_root):
            raise ValueError("results_json must stay inside repo data/")
        if not _is_relative_to(resolved["cache_dir"], data_root):
            raise ValueError("cache_dir must stay inside repo data/")

    def build_condition_specs(self) -> list[ConditionSpec]:
        required_seed_count = len(_REQUIRED_SEEDS)
        probe_seed_count = required_seed_count
        round2_seed_count = required_seed_count
        selector_time = max(30, self.stage0_budget_seconds)
        probe_time = max(300, probe_seed_count * 120)
        round2_time = max(300, round2_seed_count * 120)
        specs = [
            ConditionSpec(
                name="AdaptiveConfidenceOnlyCosineGate",
                class_name="AdaptiveConfidenceOnlyCosineGate",
                stage=_SELECTOR_STAGE,
                enabled_by_default=True,
                depends_on=[],
                seed_count=required_seed_count,
                max_seconds=selector_time,
            ),
            ConditionSpec(
                name="MutualKnnMarginGraphGate",
                class_name="MutualKnnMarginGraphGate",
                stage=_SELECTOR_STAGE,
                enabled_by_default=True,
                depends_on=[],
                seed_count=required_seed_count,
                max_seconds=selector_time,
            ),
            ConditionSpec(
                name="AcousticPriorLowHubDiverseMutualGraphPseudoLabels",
                class_name="AcousticPriorLowHubDiverseMutualGraphPseudoLabels",
                stage=_SELECTOR_STAGE,
                enabled_by_default=True,
                depends_on=[],
                seed_count=required_seed_count,
                max_seconds=selector_time,
            ),
            ConditionSpec(
                name="NoHubnessAndDiversityGateInAcousticPriorGraph",
                class_name="NoHubnessAndDiversityGateInAcousticPriorGraph",
                stage=_SELECTOR_STAGE,
                enabled_by_default=True,
                depends_on=["AcousticPriorLowHubDiverseMutualGraphPseudoLabels"],
                seed_count=required_seed_count,
                max_seconds=selector_time,
            ),
            ConditionSpec(
                name="SampleV2AcousticPriorWithoutFullTrainBaseSwap",
                class_name="SampleV2AcousticPriorWithoutFullTrainBaseSwap",
                stage=_PROBE_STAGE,
                enabled_by_default=self.stage1_default_enabled,
                depends_on=["AcousticPriorLowHubDiverseMutualGraphPseudoLabels"],
                seed_count=probe_seed_count,
                max_seconds=probe_time,
            ),
            ConditionSpec(
                name="FullTrainBaseWithAcousticPriorStrictPseudoLabels",
                class_name="FullTrainBaseWithAcousticPriorStrictPseudoLabels",
                stage=_PROBE_STAGE,
                enabled_by_default=self.stage1_default_enabled,
                depends_on=["AcousticPriorLowHubDiverseMutualGraphPseudoLabels"],
                seed_count=probe_seed_count,
                max_seconds=probe_time,
            ),
            ConditionSpec(
                name="StableCoreMarginGainRound2Contraction",
                class_name="StableCoreMarginGainRound2Contraction",
                stage=_ROUND2_STAGE,
                enabled_by_default=True,
                depends_on=["AcousticPriorLowHubDiverseMutualGraphPseudoLabels"],
                seed_count=round2_seed_count,
                max_seconds=round2_time,
            ),
            ConditionSpec(
                name="Round2CarryAllStableAssignmentsWithoutContraction",
                class_name="Round2CarryAllStableAssignmentsWithoutContraction",
                stage=_ROUND2_STAGE,
                enabled_by_default=True,
                depends_on=["StableCoreMarginGainRound2Contraction"],
                seed_count=round2_seed_count,
                max_seconds=round2_time,
            ),
        ]
        if len(specs) > _MAX_CONDITION_COUNT:
            raise ValueError(f"Condition registry exceeds hard limit of {_MAX_CONDITION_COUNT}")
        return specs

    def metric_definition(self) -> dict[str, str]:
        return {
            "primary_metric": _EXPECTED_PRIMARY_METRIC,
            "selector_gate_metric": _EXPECTED_SELECTOR_GATE_METRIC,
            "local_probe_metric": _EXPECTED_LOCAL_PROBE_METRIC,
            "stage0_coverage_metric": _EXPECTED_STAGE0_COVERAGE_METRIC,
        }
