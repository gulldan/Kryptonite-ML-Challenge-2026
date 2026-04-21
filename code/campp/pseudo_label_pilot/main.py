from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiment_harness import ExperimentHarness
except ImportError:  # pragma: no cover - local fallback when harness is unavailable

    class ExperimentHarness:  # type: ignore[no-redef]
        def __init__(self, time_budget: int) -> None:
            self.time_budget = int(time_budget)
            self.start_time = time.perf_counter()
            self.metrics: dict[str, float] = {}

        def should_stop(self) -> bool:
            elapsed = time.perf_counter() - self.start_time
            return elapsed >= 0.8 * float(self.time_budget)

        def check_value(self, value: float, metric_name: str) -> bool:
            del metric_name
            return bool(np.isfinite(value))

        def report_metric(self, metric_name: str, value: float) -> None:
            self.metrics[str(metric_name)] = float(value)

        def finalize(self) -> None:
            Path("results.json").write_text(
                json.dumps({"metrics": self.metrics}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


_THIS_DIR = Path(__file__).resolve().parent
_CAMP_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_THIS_DIR), str(_CAMP_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from experiment_config import ExperimentConfig
except ImportError:  # pragma: no cover - package-style import fallback
    from .experiment_config import ExperimentConfig

try:
    from data import EmbeddingCacheDataset, ManifestRepository
except ImportError:  # pragma: no cover - package-style import fallback
    from .data import EmbeddingCacheDataset, ManifestRepository

try:
    from evaluate import SelectorEvaluator, TrainingProbeRunner
except ImportError:  # pragma: no cover - package-style import fallback
    from .evaluate import SelectorEvaluator, TrainingProbeRunner

try:
    from methods import (
        AcousticPriorLowHubDiverseMutualGraphPseudoLabels,
        AdaptiveConfidenceOnlyCosineGate,
        BaseConditionStrategy,
        CamppEmbeddingModel,
        ConditionContext,
        FullTrainBaseWithAcousticPriorStrictPseudoLabels,
        MutualKnnMarginGraphGate,
        NoHubnessAndDiversityGateInAcousticPriorGraph,
        Round2CarryAllStableAssignmentsWithoutContraction,
        SampleV2AcousticPriorWithoutFullTrainBaseSwap,
        StableCoreMarginGainRound2Contraction,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .methods import (
        AcousticPriorLowHubDiverseMutualGraphPseudoLabels,
        AdaptiveConfidenceOnlyCosineGate,
        BaseConditionStrategy,
        CamppEmbeddingModel,
        ConditionContext,
        FullTrainBaseWithAcousticPriorStrictPseudoLabels,
        MutualKnnMarginGraphGate,
        NoHubnessAndDiversityGateInAcousticPriorGraph,
        Round2CarryAllStableAssignmentsWithoutContraction,
        SampleV2AcousticPriorWithoutFullTrainBaseSwap,
        StableCoreMarginGainRound2Contraction,
    )

STRICT_SELECTOR_NAME = "AcousticPriorLowHubDiverseMutualGraphPseudoLabels"
PROBE_CONDITION_NAMES = (
    "SampleV2AcousticPriorWithoutFullTrainBaseSwap",
    "FullTrainBaseWithAcousticPriorStrictPseudoLabels",
)
ROUND2_CONDITION_NAMES = (
    "StableCoreMarginGainRound2Contraction",
    "Round2CarryAllStableAssignmentsWithoutContraction",
)
HYPERPARAMETERS: dict[str, object] = {}


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _skip_result(condition: BaseConditionStrategy, reason: str) -> dict[str, object]:
    return {
        "condition": condition.spec.name,
        "stage": condition.spec.stage,
        "status": "skipped",
        "passed": False,
        "reason": reason,
        "runtime_seconds": 0.0,
        "artifacts": {},
    }


def _select_device(device_arg: str) -> torch.device:
    requested = device_arg.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _config_seed_values(
    config: ExperimentConfig,
    count: int | None = None,
) -> list[int]:
    seed_values = getattr(config, "seed_values", None)
    if callable(seed_values):
        return [int(seed) for seed in seed_values(count)]
    effective_count = int(config.final_seed_count if count is None else count)
    if effective_count <= 0:
        raise ValueError("seed count must be positive")
    return list(range(effective_count))


def _probe_metric_key(config: ExperimentConfig) -> str:
    metric_definition = getattr(config, "metric_definition", None)
    if callable(metric_definition):
        metrics = metric_definition()
        if isinstance(metrics, dict):
            metric_key = metrics.get("local_probe_metric")
            if isinstance(metric_key, str) and metric_key:
                return metric_key
    return "heldout_test_precision@10"


def _update_hyperparameters(
    config: ExperimentConfig,
    repo_root: Path,
    allow_stage1: bool,
    allow_stage2: bool,
    device: torch.device,
) -> None:
    del allow_stage1, allow_stage2
    HYPERPARAMETERS.clear()
    HYPERPARAMETERS.update(
        {
            "repo_root": _relative_to_repo(repo_root, repo_root),
            "allow_stage1": bool(config.stage1_default_enabled),
            "allow_stage2": bool(config.stage2_default_enabled),
            "device": str(device),
            "seeds": _config_seed_values(config),
            "knn_k": int(config.knn_k),
            "top1_quantile_by_regime": float(config.top1_quantile_by_regime),
            "min_top1_cosine": float(config.min_top1_cosine),
            "min_margin": float(config.min_margin),
            "min_component_size": int(config.min_component_size),
            "strict_max_component_size": (
                int(config.strict_max_component_size)
                if getattr(config, "strict_max_component_size", None) is not None
                else None
            ),
            "strict_min_top1_score": getattr(config, "strict_min_top1_score", None),
            "strict_min_top1_margin": getattr(
                config,
                "strict_min_top1_margin",
                None,
            ),
            "strict_indegree_top_k": getattr(config, "strict_indegree_top_k", None),
            "hubness_quantile": float(config.hubness_quantile),
            "prior_distance_quantile": float(config.prior_distance_quantile),
            "diversity_floor_quantile": float(config.diversity_floor_quantile),
            "strict_max_rows_per_component": getattr(
                config,
                "strict_max_rows_per_component",
                None,
            ),
            "selector_pass_purity": float(config.selector_pass_purity),
            "selector_soft_min_rows": int(config.selector_soft_min_rows),
            "selector_target_rows": int(config.selector_target_rows),
            "stage0_budget_seconds": int(config.stage0_budget_seconds),
            "final_seed_count": int(config.final_seed_count),
            "fallback_seed_count": int(config.fallback_seed_count),
            "selector_seed": int(config.selector_seed),
            "bootstrap_samples": int(config.bootstrap_samples),
            "official_mode": str(config.official_mode),
            "official_segment_count": int(config.official_segment_count),
            "official_topk": int(config.official_topk),
            "one_epoch_probe_epochs": int(config.one_epoch_probe_epochs),
            "round2_overlap_gate": float(config.round2_overlap_gate),
            "round2_margin_gain_gate": float(config.round2_margin_gain_gate),
            "round2_retention_gate": float(config.round2_retention_gate),
            "round2_max_drop_p10": float(config.round2_max_drop_p10),
        }
    )


def _compute_harness_time_budget(
    registry: list[BaseConditionStrategy],
    config: ExperimentConfig,
) -> int:
    enabled_budget = sum(
        int(condition.spec.max_seconds)
        for condition in registry
        if condition.spec.enabled_by_default
    )
    return int(max(config.stage0_budget_seconds, enabled_budget))


def _print_seed_lines(result: dict[str, object], metric_key: str) -> None:
    del metric_key
    seed_results = result.get("seed_results")
    if not isinstance(seed_results, list):
        return
    for seed_row in seed_results:
        if not isinstance(seed_row, dict):
            continue
        seed = seed_row.get("seed")
        value = seed_row.get("primary_metric")
        if seed is None:
            continue
        print(f"condition={result['condition']} seed={seed} primary_metric: {value}")


def _condition_metric_for_output(result: dict[str, object]) -> float:
    primary_metric = result.get("primary_metric")
    if primary_metric is not None:
        metric_value = float(primary_metric)
        if np.isfinite(metric_value):
            return metric_value

    aggregate_metrics = result.get("aggregate_metrics", {})
    if isinstance(aggregate_metrics, dict):
        payload = aggregate_metrics.get("primary_metric")
        if isinstance(payload, dict) and "mean" in payload:
            metric_value = float(payload["mean"])
            if np.isfinite(metric_value):
                return metric_value

    metrics = result.get("metrics", {})
    if isinstance(metrics, dict):
        for metric_name in (
            "primary_metric",
            "simulated_pairwise_purity",
            "validation_precision@10",
        ):
            if metric_name not in metrics:
                continue
            metric_value = float(metrics[metric_name])
            if np.isfinite(metric_value):
                return metric_value

    return 0.0


def _print_aggregate_lines(result: dict[str, object]) -> None:
    metric_mean = _condition_metric_for_output(result)
    print(f"condition={result['condition']} metric={metric_mean}")


def _report_metrics_to_harness(
    harness: ExperimentHarness,
    result: dict[str, object],
) -> None:
    condition_name = str(result["condition"])
    if result.get("stage") == "selector":
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            return
        selector_metrics = {
            "simulated_pairwise_purity": metrics.get("simulated_pairwise_purity"),
            "accepted_pseudo_rows": metrics.get("accepted_pseudo_rows"),
        }
        for metric_name, value in selector_metrics.items():
            if value is None:
                continue
            metric_key = f"{condition_name}.{metric_name}"
            if harness.check_value(float(value), metric_key):
                harness.report_metric(metric_key, float(value))
            else:
                print("SKIP: NaN/Inf detected")
        return

    aggregate_metrics = result.get("aggregate_metrics", {})
    if not isinstance(aggregate_metrics, dict):
        return
    for metric_name, payload in aggregate_metrics.items():
        if not isinstance(payload, dict) or "mean" not in payload:
            continue
        metric_value = float(payload["mean"])
        metric_key = f"{condition_name}.{metric_name}"
        if harness.check_value(metric_value, metric_key):
            harness.report_metric(metric_key, metric_value)
        else:
            print("SKIP: NaN/Inf detected")


def _extract_seed_metric_vector(
    result: dict[str, object],
    metric_key: str,
) -> dict[int, float]:
    seed_results = result.get("seed_results")
    if not isinstance(seed_results, list):
        raise ValueError(f"{result['condition']} is missing seed_results")
    seed_to_value: dict[int, float] = {}
    for seed_row in seed_results:
        if not isinstance(seed_row, dict):
            continue
        if "seed" not in seed_row or metric_key not in seed_row:
            continue
        seed_to_value[int(seed_row["seed"])] = float(seed_row[metric_key])
    if not seed_to_value:
        raise ValueError(f"{result['condition']} has no seed metric {metric_key}")
    return seed_to_value


def _intersect_seed_metric_vectors(
    left_result: dict[str, object],
    right_result: dict[str, object],
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    left_map = _extract_seed_metric_vector(left_result, metric_key)
    right_map = _extract_seed_metric_vector(right_result, metric_key)
    shared_seeds = sorted(set(left_map) & set(right_map))
    left_only = sorted(set(left_map) - set(right_map))
    right_only = sorted(set(right_map) - set(left_map))
    shared_seed_array = np.asarray(shared_seeds, dtype=np.int64)
    left_values = np.asarray(
        [left_map[seed] for seed in shared_seeds],
        dtype=np.float64,
    )
    right_values = np.asarray(
        [right_map[seed] for seed in shared_seeds],
        dtype=np.float64,
    )
    return shared_seed_array, left_values, right_values, left_only, right_only


def _load_results_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_results_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonify(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _condition_metric_snapshot(result: dict[str, object]) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "stage": str(result.get("stage", "")),
        "status": str(result.get("status", "")),
        "primary_metric_source": result.get("primary_metric_source"),
        "primary_metric": result.get("primary_metric"),
    }
    if result.get("stage") == "selector":
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            return snapshot
        snapshot.update(
            {
                "simulated_pairwise_purity": float(metrics.get("simulated_pairwise_purity", 0.0)),
                "accepted_pseudo_rows": int(metrics.get("accepted_pseudo_rows", 0)),
            }
        )
        return snapshot
    aggregate_metrics = result.get("aggregate_metrics", {})
    if not isinstance(aggregate_metrics, dict):
        return snapshot
    for metric_name, payload in aggregate_metrics.items():
        if not isinstance(payload, dict):
            continue
        if "mean" in payload:
            snapshot[f"{metric_name}_mean"] = float(payload["mean"])
        if "std" in payload:
            snapshot[f"{metric_name}_std"] = float(payload["std"])
    return snapshot


def build_condition_registry(config: ExperimentConfig) -> list[BaseConditionStrategy]:
    specs = config.build_condition_specs()
    return [
        AdaptiveConfidenceOnlyCosineGate(specs[0], config),
        MutualKnnMarginGraphGate(specs[1], config),
        AcousticPriorLowHubDiverseMutualGraphPseudoLabels(specs[2], config),
        NoHubnessAndDiversityGateInAcousticPriorGraph(specs[3], config),
        SampleV2AcousticPriorWithoutFullTrainBaseSwap(specs[4], config),
        FullTrainBaseWithAcousticPriorStrictPseudoLabels(specs[5], config),
        StableCoreMarginGainRound2Contraction(specs[6], config),
        Round2CarryAllStableAssignmentsWithoutContraction(specs[7], config),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label pilot runner for CAM++.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--allow-stage1",
        action="store_true",
        help="Compatibility flag; Stage-1 probes are attempted by default.",
    )
    parser.add_argument(
        "--allow-stage2",
        action="store_true",
        help="Compatibility flag; Stage-2 probes are attempted by default.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help="Compatibility override; when provided it must remain fixed at 3 for seeds [0, 1, 2].",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Target torch device. Use auto, cpu, or cuda.",
    )
    args = parser.parse_args()

    if args.seed_count is not None and int(args.seed_count) != 3:
        raise ValueError("--seed-count must remain fixed at 3 for seeds [0, 1, 2]")

    repo_root = Path(args.repo_root).resolve()
    device = _select_device(args.device)
    config = ExperimentConfig.from_repo_defaults(
        repo_root=str(repo_root),
        allow_stage1=bool(args.allow_stage1),
        allow_stage2=bool(args.allow_stage2),
        seed_count=args.seed_count,
    )
    config.validate_repo_constraints()
    probe_metric_key = _probe_metric_key(config)
    _update_hyperparameters(
        config=config,
        repo_root=repo_root,
        allow_stage1=bool(args.allow_stage1),
        allow_stage2=bool(args.allow_stage2),
        device=device,
    )

    print(f"METRIC_DEF {json.dumps(config.metric_definition(), sort_keys=True)}")

    repo = ManifestRepository(config)
    prior = repo.build_sample_v2_prior()
    model = CamppEmbeddingModel(config).to(device)
    bundle = EmbeddingCacheDataset(config).compute_or_load_validation_bundle(
        model,
        repo,
        device,
    )
    evaluator = SelectorEvaluator(config)
    runner = TrainingProbeRunner(config)
    context = ConditionContext(
        config=config,
        prior=prior,
        validation_bundle=bundle,
        results=[],
        completed_conditions={},
        passed_conditions={},
        strict_selector_result=None,
        stage1_winner=None,
        wall_clock_start=time.perf_counter(),
    )
    registry = build_condition_registry(config)
    print(f"REGISTERED_CONDITIONS {json.dumps([condition.spec.name for condition in registry])}")

    harness = ExperimentHarness(time_budget=_compute_harness_time_budget(registry, config))
    results_path = config.paths.resolve_all()["results_json"]
    collected_metrics: dict[str, object] = {}
    comparisons: dict[str, object] = {}
    summary_payload: dict[str, object] = {}
    infrastructure_failed = False

    try:
        time_estimate = float(sum(item.spec.max_seconds for item in registry))
        print(f"TIME_ESTIMATE: {time_estimate:.1f}s")

        for condition in registry:
            result = condition.run(context, repo, evaluator, runner, device)

            context.record_result(result)
            runner.append_results_json(result)
            _report_metrics_to_harness(harness, result)

            if result.get("stage") == "selector":
                _print_seed_lines(result, "simulated_pairwise_purity")
            else:
                _print_seed_lines(result, probe_metric_key)
            _print_aggregate_lines(result)
            collected_metrics[str(result["condition"])] = _condition_metric_snapshot(result)

        result_by_name = {str(result["condition"]): result for result in context.results}

        probe_results = [
            result_by_name[name]
            for name in PROBE_CONDITION_NAMES
            if name in result_by_name and result_by_name[name].get("status") == "passed"
        ]
        if len(probe_results) == 2:
            shared_seeds, values_a, values_b, seeds_only_a, seeds_only_b = (
                _intersect_seed_metric_vectors(
                    probe_results[0],
                    probe_results[1],
                    probe_metric_key,
                )
            )
            paired_payload: dict[str, object] = {
                "metric": probe_metric_key,
                "paired_seed_count": int(shared_seeds.size),
                "paired_seeds": shared_seeds.tolist(),
                "only_condition_a_seeds": seeds_only_a,
                "only_condition_b_seeds": seeds_only_b,
                "condition_a": str(probe_results[0]["condition"]),
                "condition_b": str(probe_results[1]["condition"]),
            }
            if shared_seeds.size == 0:
                paired_payload["p_value"] = "no_overlapping_successful_seeds"
            else:
                paired_payload.update(
                    evaluator.compare_seed_vectors(
                        values_a,
                        values_b,
                    )
                )
            comparisons["stage1_probe_wilcoxon"] = paired_payload

        round2_comparisons: dict[str, object] = {}
        if context.stage1_winner is not None:
            stage1_mean = float(
                context.stage1_winner.get("aggregate_metrics", {})
                .get(probe_metric_key, {})
                .get("mean", float("nan"))
            )
            for condition_name in ROUND2_CONDITION_NAMES:
                result = result_by_name.get(condition_name)
                if result is None or result.get("stage") != "round2":
                    continue
                round2_mean = float(
                    result.get("aggregate_metrics", {})
                    .get(probe_metric_key, {})
                    .get("mean", float("nan"))
                )
                if np.isfinite(stage1_mean) and np.isfinite(round2_mean):
                    round2_comparisons[condition_name] = {
                        "stage1_winner": str(context.stage1_winner["condition"]),
                        f"delta_{probe_metric_key}": round2_mean - stage1_mean,
                    }
            if round2_comparisons:
                comparisons["round2_vs_stage1"] = round2_comparisons

        selector_results = [
            result
            for result in context.results
            if result.get("stage") == "selector" and isinstance(result.get("metrics"), dict)
        ]
        best_selector = None
        if selector_results:
            best_selector = max(
                selector_results,
                key=lambda result: (
                    float(result["metrics"].get("simulated_pairwise_purity", 0.0)),
                    int(result["metrics"].get("accepted_pseudo_rows", 0)),
                ),
            )

        probe_passed_results = [
            result
            for result in context.results
            if result.get("stage") == "probe" and result.get("status") == "passed"
        ]
        best_probe = None
        if probe_passed_results:
            best_probe = max(
                probe_passed_results,
                key=lambda result: float(
                    result.get("aggregate_metrics", {})
                    .get(probe_metric_key, {})
                    .get("mean", float("-inf"))
                ),
            )

        round2_results = [result for result in context.results if result.get("stage") == "round2"]
        round2_helped = any(
            float(result.get(f"delta_{probe_metric_key}", float("-inf"))) > 0.0
            for result in round2_results
            if result.get("status") == "passed"
        )
        round2_justified = any(
            isinstance(result.get("round2_gate_metrics"), dict)
            and bool(result["round2_gate_metrics"].get("pass_round2", False))
            for result in round2_results
        )

        condition_comparison = []
        for result in context.results:
            primary_metric = result.get("primary_metric")
            comparison_row = {
                "condition": str(result["condition"]),
                "stage": str(result.get("stage", "")),
                "status": str(result.get("status", "")),
                "primary_metric_source": result.get("primary_metric_source"),
                "primary_metric_mean": (
                    float(primary_metric) if primary_metric is not None else None
                ),
            }
            condition_comparison.append(comparison_row)

        summary_payload = {
            "condition_comparison": condition_comparison,
            "best_selector": (
                {
                    "condition": str(best_selector["condition"]),
                    "purity": float(best_selector["metrics"].get("simulated_pairwise_purity", 0.0)),
                    "accepted_pseudo_rows": int(
                        best_selector["metrics"].get("accepted_pseudo_rows", 0)
                    ),
                }
                if best_selector is not None
                else None
            ),
            "strict_selector_passed": bool(
                result_by_name.get(STRICT_SELECTOR_NAME, {}).get("status") == "passed"
            ),
            "best_probe": (
                {
                    "condition": str(best_probe["condition"]),
                    f"{probe_metric_key}_mean": float(
                        best_probe.get("aggregate_metrics", {})
                        .get(probe_metric_key, {})
                        .get("mean", float("nan"))
                    ),
                }
                if best_probe is not None
                else None
            ),
            "round2_justified": bool(round2_justified),
            "round2_helped": bool(round2_helped),
        }
        print(f"SUMMARY {json.dumps(_jsonify(summary_payload), ensure_ascii=False)}")

    except Exception:
        infrastructure_failed = True
        raise
    finally:
        harness.finalize()
        results_payload = _load_results_json(results_path)
        results_payload["hyperparameters"] = _jsonify(HYPERPARAMETERS)
        results_payload["metrics"] = _jsonify(collected_metrics)
        results_payload["comparisons"] = _jsonify(comparisons)
        results_payload["summary"] = _jsonify(summary_payload)
        results_payload["infrastructure_failed"] = bool(infrastructure_failed)
        harness_results_path = Path("results.json")
        if harness_results_path.exists():
            try:
                harness_results = json.loads(harness_results_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                harness_results = {}
            results_payload["harness_results"] = _jsonify(harness_results)
        _write_results_json(results_path, results_payload)


if __name__ == "__main__":
    main()
