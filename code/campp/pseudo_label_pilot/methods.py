from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

_THIS_DIR = Path(__file__).resolve().parent
_CAMP_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_THIS_DIR), str(_CAMP_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from experiment_config import ConditionSpec, ExperimentConfig
except ImportError:  # pragma: no cover - package-style import fallback
    from .experiment_config import ConditionSpec, ExperimentConfig

try:
    from data import ManifestRepository
except ImportError:  # pragma: no cover - package-style import fallback
    from .data import ManifestRepository

try:
    from selector import PseudoLabelSelectionConfig, select_pseudo_label_rows
except ImportError:  # pragma: no cover - package-style import fallback
    try:
        from .selector import PseudoLabelSelectionConfig, select_pseudo_label_rows
    except ImportError:  # pragma: no cover - sandbox bundle may omit selector.py
        _FALLBACK_SELECTOR_COLUMNS = [
            "row_index",
            "component_id",
            "component_size",
            "pseudo_spk",
            "anchor_index",
            "top1_score",
            "top1_cosine",
            "top1_margin",
            "path",
            "dur",
            "start",
            "stop",
            "orig_filepath",
            "manifest_split",
            "duration_bucket",
            "profile_bucket",
            "prior_distance",
        ]
        _FALLBACK_SELECTOR_EPSILON = 1e-6

        def _fallback_require_columns(
            df: pd.DataFrame,
            columns: list[str],
            context: str,
        ) -> None:
            missing = [column for column in columns if column not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns for {context}: {missing}")

        def _fallback_weighted_prior_distance(
            features: np.ndarray,
            mu: np.ndarray,
            scale: np.ndarray,
            weight: np.ndarray,
        ) -> np.ndarray:
            centered = (features - mu[None, :]) / scale[None, :]
            return np.sum(
                weight[None, :] * np.square(centered),
                axis=1,
                dtype=np.float32,
            )

        def _fallback_append_stage(
            stage_counts: list[dict[str, int]],
            stage: str,
            row_count: int,
            cluster_count: int = 0,
        ) -> None:
            stage_counts.append(
                {
                    "stage": str(stage),
                    "row_count": int(row_count),
                    "cluster_count": int(cluster_count),
                }
            )

        def _fallback_build_mutual_margin_edges(
            *,
            topk_idx: np.ndarray,
            top1_margin: np.ndarray,
            min_margin: float | None,
        ) -> np.ndarray:
            required_margin = 0.0 if min_margin is None else float(min_margin)
            neighbor_sets = [set(row.tolist()) for row in topk_idx]
            edges: list[tuple[int, int]] = []
            for src_index in range(topk_idx.shape[0]):
                if float(top1_margin[src_index]) < required_margin:
                    continue
                for dst_index in topk_idx[src_index]:
                    dst = int(dst_index)
                    if src_index == dst:
                        continue
                    if src_index not in neighbor_sets[dst]:
                        continue
                    if float(top1_margin[dst]) < required_margin:
                        continue
                    left, right = sorted((src_index, dst))
                    edges.append((left, right))
            if not edges:
                return np.zeros((0, 2), dtype=np.int64)
            return np.unique(np.asarray(edges, dtype=np.int64), axis=0)

        def _fallback_component_frame(
            *,
            selector_meta: pd.DataFrame,
            topk_idx: np.ndarray,
            topk_sim: np.ndarray,
            top1_margin: np.ndarray,
            row_indices: np.ndarray,
            component_id: int,
            prefix: str,
            extra_columns: dict[str, Any] | None = None,
        ) -> pd.DataFrame:
            component_rows = selector_meta.iloc[row_indices].copy().reset_index(drop=True)
            component_rows.insert(0, "row_index", row_indices.astype(np.int64, copy=False))
            component_rows["component_id"] = int(component_id)
            component_rows["component_size"] = int(len(row_indices))
            component_rows["pseudo_spk"] = f"{prefix}_{component_id}"
            component_rows["anchor_index"] = topk_idx[row_indices, 0].astype(
                np.int64,
                copy=False,
            )
            component_rows["top1_score"] = topk_sim[row_indices, 0].astype(
                np.float32,
                copy=False,
            )
            component_rows["top1_cosine"] = component_rows["top1_score"].to_numpy(copy=False)
            component_rows["top1_margin"] = top1_margin[row_indices].astype(
                np.float32,
                copy=False,
            )
            if extra_columns:
                for name, value in extra_columns.items():
                    if isinstance(value, np.ndarray):
                        component_rows[name] = value[row_indices]
                    else:
                        component_rows[name] = value
            extra_tail = [
                column
                for column in component_rows.columns
                if column not in _FALLBACK_SELECTOR_COLUMNS
            ]
            return component_rows.loc[:, _FALLBACK_SELECTOR_COLUMNS + extra_tail]

        def _fallback_limit_rows_per_cluster(
            accepted_df: pd.DataFrame,
            *,
            max_rows_per_cluster: int,
        ) -> pd.DataFrame:
            if accepted_df.empty:
                return accepted_df.copy()
            sort_by: list[str] = []
            ascending: list[bool] = []
            for name, is_ascending in (
                ("top1_score", False),
                ("top1_margin", False),
                ("prior_distance", True),
                ("inbound_degree", True),
                ("row_index", True),
            ):
                if name in accepted_df.columns:
                    sort_by.append(name)
                    ascending.append(is_ascending)
            ranked = accepted_df.sort_values(
                by=sort_by,
                ascending=ascending,
                kind="mergesort",
            ).copy()
            ranked["_cluster_rank"] = ranked.groupby("component_id", sort=False).cumcount() + 1
            ranked = ranked.loc[ranked["_cluster_rank"] <= int(max_rows_per_cluster)].copy()
            return ranked.drop(columns=["_cluster_rank"]).reset_index(drop=True)

        @dataclass(slots=True)
        class PseudoLabelSelectionConfig:
            min_cluster_size: int
            max_cluster_size: int | None = None
            min_top1_score: float | None = None
            min_top1_margin: float | None = None
            max_indegree_quantile: float | None = None
            indegree_top_k: int | None = None
            max_rows_per_cluster: int | None = None

            def __post_init__(self) -> None:
                if self.min_cluster_size <= 0:
                    raise ValueError("min_cluster_size must be positive")
                if self.max_cluster_size is not None:
                    if self.max_cluster_size < self.min_cluster_size:
                        raise ValueError("max_cluster_size must be >= min_cluster_size")
                if self.min_top1_score is not None and not (0.0 <= self.min_top1_score <= 1.0):
                    raise ValueError("min_top1_score must satisfy 0 <= value <= 1")
                if self.min_top1_margin is not None and self.min_top1_margin < 0.0:
                    raise ValueError("min_top1_margin must be non-negative")
                if self.max_indegree_quantile is not None and not (
                    0.0 < self.max_indegree_quantile <= 1.0
                ):
                    raise ValueError("max_indegree_quantile must satisfy 0 < value <= 1")
                if self.indegree_top_k is not None and self.indegree_top_k <= 0:
                    raise ValueError("indegree_top_k must be positive when provided")
                if self.max_rows_per_cluster is not None and self.max_rows_per_cluster <= 0:
                    raise ValueError("max_rows_per_cluster must be positive when provided")

        def select_pseudo_label_rows(
            *,
            bundle: dict[str, object],
            prior: dict[str, object],
            selection: PseudoLabelSelectionConfig,
            feature_names: list[str],
            label_prefix: str,
        ) -> tuple[pd.DataFrame, dict[str, object]]:
            selector_meta = _selector_meta_df(bundle)
            if selector_meta.empty:
                summary = {
                    "selection_stage_counts": [
                        {"stage": "initial", "row_count": 0, "cluster_count": 0}
                    ],
                    "selection_thresholds": {},
                    "final_row_count": 0,
                    "final_cluster_count": 0,
                }
                return pd.DataFrame(columns=_FALLBACK_SELECTOR_COLUMNS), summary

            _fallback_require_columns(
                selector_meta,
                feature_names,
                "selector_meta_df",
            )
            topk_idx, topk_sim, top1_margin = _bundle_arrays(bundle)
            if selector_meta.shape[0] != topk_idx.shape[0]:
                raise ValueError("selector_meta_df must align with topk arrays")

            allowed_nodes = np.ones(topk_idx.shape[0], dtype=bool)
            stage_counts: list[dict[str, int]] = []
            thresholds: dict[str, float] = {}
            _fallback_append_stage(
                stage_counts,
                "initial",
                int(allowed_nodes.sum()),
            )

            top1_score = topk_sim[:, 0].astype(np.float32, copy=False)
            if selection.min_top1_score is not None:
                allowed_nodes &= top1_score >= float(selection.min_top1_score)
                thresholds["top1_score_min"] = float(selection.min_top1_score)
                _fallback_append_stage(
                    stage_counts,
                    "top1_score",
                    int(allowed_nodes.sum()),
                )

            if selection.min_top1_margin is not None:
                allowed_nodes &= top1_margin >= float(selection.min_top1_margin)
                thresholds["top1_margin_min"] = float(selection.min_top1_margin)
                _fallback_append_stage(
                    stage_counts,
                    "top1_margin",
                    int(allowed_nodes.sum()),
                )

            effective_top_k = topk_idx.shape[1]
            if selection.indegree_top_k is not None:
                effective_top_k = min(
                    int(selection.indegree_top_k),
                    topk_idx.shape[1],
                )
            inbound_degree = np.bincount(
                topk_idx[:, :effective_top_k].reshape(-1),
                minlength=topk_idx.shape[0],
            ).astype(np.int64, copy=False)
            if selection.max_indegree_quantile is not None:
                indegree_threshold = float(
                    np.quantile(
                        inbound_degree.astype(np.float32, copy=False),
                        float(selection.max_indegree_quantile),
                    )
                )
                allowed_nodes &= inbound_degree <= indegree_threshold
                thresholds[f"indegree_at_{effective_top_k}_max"] = indegree_threshold
                _fallback_append_stage(
                    stage_counts,
                    f"indegree_at_{effective_top_k}",
                    int(allowed_nodes.sum()),
                )

            base_edges = _fallback_build_mutual_margin_edges(
                topk_idx=topk_idx,
                top1_margin=top1_margin,
                min_margin=selection.min_top1_margin,
            )
            if base_edges.size == 0:
                summary = {
                    "selection_stage_counts": stage_counts,
                    "selection_thresholds": thresholds,
                    "final_row_count": 0,
                    "final_cluster_count": 0,
                }
                return (
                    pd.DataFrame(
                        columns=_FALLBACK_SELECTOR_COLUMNS + ["inbound_degree", "diversity_score"]
                    ),
                    summary,
                )

            pruned_edges = base_edges[
                allowed_nodes[base_edges[:, 0]] & allowed_nodes[base_edges[:, 1]]
            ]
            labels, counts = _connected_components_for_edges(
                len(selector_meta),
                pruned_edges,
            )

            mu = np.asarray(prior["mu"], dtype=np.float32)
            scale = np.clip(
                np.asarray(prior["scale"], dtype=np.float32),
                _FALLBACK_SELECTOR_EPSILON,
                None,
            )
            weight = np.asarray(prior["weight"], dtype=np.float32)

            frames: list[pd.DataFrame] = []
            next_component_id = 0
            for raw_component_id, component_size in enumerate(counts.tolist()):
                if component_size < selection.min_cluster_size:
                    continue
                if (
                    selection.max_cluster_size is not None
                    and component_size > selection.max_cluster_size
                ):
                    continue
                component_rows = np.flatnonzero(labels == raw_component_id)
                if component_rows.size == 0:
                    continue
                x_comp = (
                    selector_meta.iloc[component_rows]
                    .loc[:, feature_names]
                    .to_numpy(
                        dtype=np.float32,
                        copy=True,
                    )
                )
                centroid = np.median(x_comp, axis=0).astype(
                    np.float32,
                    copy=False,
                )
                prior_distance = float(
                    _fallback_weighted_prior_distance(
                        centroid.reshape(1, -1),
                        mu=mu,
                        scale=scale,
                        weight=weight,
                    )[0]
                )
                feature_std = np.std(x_comp, axis=0, ddof=0).astype(
                    np.float32,
                    copy=False,
                )
                diversity_score = float(np.sum(weight * (feature_std / scale), dtype=np.float32))
                if prior_distance > float(prior["prior_distance_threshold"]):
                    continue
                if diversity_score < float(prior["diversity_floor"]):
                    continue
                frames.append(
                    _fallback_component_frame(
                        selector_meta=selector_meta,
                        topk_idx=topk_idx,
                        topk_sim=topk_sim,
                        top1_margin=top1_margin,
                        row_indices=component_rows,
                        component_id=next_component_id,
                        prefix=label_prefix,
                        extra_columns={
                            "inbound_degree": inbound_degree,
                            "prior_distance": prior_distance,
                            "diversity_score": diversity_score,
                        },
                    )
                )
                next_component_id += 1

            accepted_df = (
                pd.concat(frames, axis=0, ignore_index=True)
                if frames
                else pd.DataFrame(
                    columns=_FALLBACK_SELECTOR_COLUMNS + ["inbound_degree", "diversity_score"]
                )
            )
            _fallback_append_stage(
                stage_counts,
                "prior_distance",
                int(len(accepted_df)),
                int(accepted_df["component_id"].nunique()) if not accepted_df.empty else 0,
            )

            if selection.max_rows_per_cluster is not None:
                accepted_df = _fallback_limit_rows_per_cluster(
                    accepted_df,
                    max_rows_per_cluster=selection.max_rows_per_cluster,
                )
                thresholds["max_rows_per_cluster"] = float(selection.max_rows_per_cluster)
                _fallback_append_stage(
                    stage_counts,
                    "cluster_cap",
                    int(len(accepted_df)),
                    int(accepted_df["component_id"].nunique()) if not accepted_df.empty else 0,
                )

            accepted_df = accepted_df.sort_values(
                by=["component_id", "row_index"],
                ascending=[True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            summary = {
                "selection_stage_counts": stage_counts,
                "selection_thresholds": thresholds,
                "final_row_count": int(len(accepted_df)),
                "final_cluster_count": int(accepted_df["component_id"].nunique())
                if not accepted_df.empty
                else 0,
                "prior_distance_max": float(prior["prior_distance_threshold"]),
                "diversity_floor_min": float(prior["diversity_floor"]),
                "indegree_top_k": int(effective_top_k),
            }
            if "duration_bucket" in accepted_df.columns:
                summary["final_duration_bucket_count"] = int(
                    accepted_df["duration_bucket"].astype(str).nunique()
                )
            if "profile_bucket" in accepted_df.columns:
                summary["final_profile_bucket_count"] = int(
                    accepted_df["profile_bucket"].astype(str).nunique()
                )
            return accepted_df, summary


from common import (  # noqa: E402
    build_campp_embedding_model,
    load_config,
    load_pretrained_embedding,
)
from retrieval import extract_embeddings  # noqa: E402

if TYPE_CHECKING:
    try:
        from evaluate import SelectorEvaluator, TrainingProbeRunner
    except ImportError:  # pragma: no cover - package-style import fallback
        from .evaluate import SelectorEvaluator, TrainingProbeRunner


_SELECTOR_COLUMNS = [
    "row_index",
    "component_id",
    "component_size",
    "pseudo_spk",
    "anchor_index",
    "top1_cosine",
    "top1_margin",
    "path",
    "dur",
    "start",
    "stop",
    "orig_filepath",
    "manifest_split",
    "duration_bucket",
    "profile_bucket",
    "prior_distance",
]
_MANIFEST_COLUMNS = ["ID", "dur", "path", "start", "stop", "spk", "orig_filepath"]
_STRICT_SELECTOR_NAME = "AcousticPriorLowHubDiverseMutualGraphPseudoLabels"


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _metric_mean_from_result(result: dict[str, object], metric_key: str) -> float:
    aggregate = result.get("aggregate_metrics", {})
    if isinstance(aggregate, dict):
        metric_payload = aggregate.get(metric_key)
        if isinstance(metric_payload, dict) and "mean" in metric_payload:
            return float(metric_payload["mean"])
    metrics = result.get("metrics", {})
    if isinstance(metrics, dict) and metric_key in metrics:
        return float(metrics[metric_key])
    return float("-inf")


def _selector_meta_df(bundle: dict[str, object]) -> pd.DataFrame:
    meta = bundle.get("selector_meta_df")
    if not isinstance(meta, pd.DataFrame):
        raise TypeError("validation bundle must contain selector_meta_df as a DataFrame")
    return meta.drop(columns=["spk", "oracle_spk"], errors="ignore").copy()


def _bundle_arrays(
    bundle: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topk_idx = np.asarray(bundle["topk_idx"], dtype=np.int64)
    topk_sim = np.asarray(bundle["topk_sim"], dtype=np.float32)
    top1_margin = np.asarray(bundle["top1_margin"], dtype=np.float32)
    if topk_idx.shape != topk_sim.shape:
        raise ValueError("topk_idx and topk_sim must have the same shape")
    if top1_margin.shape != (topk_idx.shape[0],):
        raise ValueError("top1_margin must be a 1D array aligned with topk_idx rows")
    return topk_idx, topk_sim, top1_margin


def _connected_components_for_edges(
    num_nodes: int,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if num_nodes <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if edges.size == 0:
        labels = np.arange(num_nodes, dtype=np.int64)
        counts = np.ones(num_nodes, dtype=np.int64)
        return labels, counts
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    component_count, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    counts = np.bincount(labels, minlength=component_count).astype(np.int64, copy=False)
    return labels.astype(np.int64, copy=False), counts


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_from_repo(path_like: str | Path, repo_root: Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def _seed_values_for_spec(config: ExperimentConfig, spec: ConditionSpec) -> list[int]:
    return config.seed_values(spec.seed_count)


class CamppEmbeddingModel(torch.nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        resolved_paths = self.config.paths.resolve_all()
        self.repo_config = load_config(resolved_paths["campp_base_config"])
        self.backbone = build_campp_embedding_model(self.repo_config)
        self.pretrained_weight_path = load_pretrained_embedding(
            self.repo_config,
            self.backbone,
        )

    def forward(self, fbank_batch: torch.Tensor) -> torch.Tensor:
        fbank_batch = fbank_batch.to(dtype=torch.float32)
        embeddings = self.backbone(fbank_batch)
        return embeddings.to(dtype=torch.float32)

    def encode_manifest_batches(
        self,
        manifest_df: pd.DataFrame,
        device: torch.device,
        batch_size: int = 32,
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            embeddings, _ = extract_embeddings(
                manifest=manifest_df,
                model=self,
                data_root=self.repo_config["paths"]["data_root"],
                sample_rate=int(self.repo_config["model"]["sample_rate"]),
                n_mels=int(self.repo_config["model"]["n_mels"]),
                mode=self.config.official_mode,
                eval_chunk_sec=float(self.repo_config["training"]["eval_chunk_sec"]),
                segment_count=int(self.config.official_segment_count),
                long_file_threshold_sec=float(
                    self.repo_config["evaluation"]["long_file_threshold_sec"]
                ),
                batch_size=int(batch_size),
                device=device,
                pad_mode=str(self.repo_config["training"]["short_clip_pad_mode"]),
            )
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"Expected embeddings with shape [N, 512], got {emb.shape}")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return (emb / norms).astype(np.float32, copy=False)


@dataclass(slots=True)
class ConditionContext:
    config: ExperimentConfig
    prior: dict[str, object]
    validation_bundle: dict[str, object]
    results: list[dict[str, object]]
    completed_conditions: dict[str, dict[str, object]]
    passed_conditions: dict[str, dict[str, object]]
    strict_selector_result: dict[str, object] | None
    stage1_winner: dict[str, object] | None
    wall_clock_start: float

    def record_result(self, result: dict[str, object]) -> None:
        self.results.append(result)
        self.completed_conditions[str(result["condition"])] = result
        if result.get("status") == "passed":
            self.passed_conditions[str(result["condition"])] = result
        if result.get("condition") == _STRICT_SELECTOR_NAME and isinstance(
            result.get("accepted_df"), pd.DataFrame
        ):
            self.strict_selector_result = result
        if result.get("stage") == "probe":
            candidate_score = _metric_mean_from_result(
                result,
                "validation_precision@10",
            )
            if not np.isfinite(candidate_score):
                return
            incumbent_score = float("-inf")
            if self.stage1_winner is not None:
                incumbent_score = _metric_mean_from_result(
                    self.stage1_winner,
                    "validation_precision@10",
                )
            if candidate_score > incumbent_score:
                self.stage1_winner = result


class BaseConditionStrategy:
    def __init__(self, spec: ConditionSpec, config: ExperimentConfig) -> None:
        self.spec = spec
        self.config = config
        if self.__class__.__name__ != self.spec.class_name:
            raise ValueError(
                f"Registry mismatch: class={self.__class__.__name__} spec={self.spec.class_name}"
            )
        self._resolved_paths = self.config.paths.resolve_all()

    def should_run(self, context: ConditionContext) -> bool:
        runnable = self.spec.is_runnable(
            {
                "completed_conditions": context.completed_conditions,
                "available_conditions": context.completed_conditions,
                "passed_conditions": context.passed_conditions,
                "enabled_stages": {
                    "selector": True,
                    "probe": self.config.stage1_default_enabled,
                    "round2": self.config.stage2_default_enabled,
                },
            }
        )
        if self.spec.stage == "selector":
            return runnable
        if self.spec.stage == "probe":
            return runnable and context.strict_selector_result is not None
        if self.spec.stage == "round2":
            return runnable and context.stage1_winner is not None
        return False

    def run(
        self,
        context: ConditionContext,
        repo: ManifestRepository,
        evaluator: SelectorEvaluator,
        runner: TrainingProbeRunner,
        device: torch.device,
    ) -> dict[str, object]:
        stage_start = time.perf_counter()
        if not self.should_run(context):
            result = self._skip_result(reason="condition_gate_not_met")
        elif self.spec.stage == "selector":
            result = self.run_selector(context, repo, evaluator)
        elif self.spec.stage == "probe":
            result = self.run_probe(context, repo, runner, evaluator, device)
        elif self.spec.stage == "round2":
            result = self.run_round2(context, repo, runner, evaluator, device)
        else:
            result = self._skip_result(reason=f"unsupported_stage_{self.spec.stage}")
        result = self._finalize_result(result)
        result.setdefault("condition", self.spec.name)
        result.setdefault("stage", self.spec.stage)
        result["runtime_seconds"] = float(time.perf_counter() - stage_start)
        result.setdefault("status", "skipped")
        result.setdefault("artifacts", {})
        return result

    def run_selector(
        self,
        context: ConditionContext,
        repo: ManifestRepository,
        evaluator: SelectorEvaluator,
    ) -> dict[str, object]:
        del repo
        accepted_df = self.select_pseudo_labels(context.validation_bundle, context.prior)
        selector_root = self._stage_root("selector")
        selector_root.mkdir(parents=True, exist_ok=True)
        repo_root = self._resolved_paths["repo_root"]
        accepted_rows_path = selector_root / "accepted_rows.parquet"
        selected_pool_path = selector_root / "selected_pool.csv"
        component_summary_path = selector_root / "component_summary.csv"
        metrics_path = selector_root / "selector_metrics.json"
        selection_summary_path = selector_root / "selection_summary.json"

        accepted_df.to_parquet(accepted_rows_path, index=False)
        accepted_df.to_csv(selected_pool_path, index=False)
        component_summary = self._component_summary(accepted_df)
        component_summary.to_csv(component_summary_path, index=False)
        selection_summary = dict(accepted_df.attrs.get("selection_summary", {}))
        if not selection_summary:
            selection_summary = {
                "selection_stage_counts": [
                    {
                        "stage": "selector_output",
                        "row_count": int(len(accepted_df)),
                        "cluster_count": int(accepted_df["component_id"].nunique())
                        if not accepted_df.empty and "component_id" in accepted_df.columns
                        else 0,
                    }
                ]
            }
        selection_summary_path.write_text(
            json.dumps(_jsonify(selection_summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        metrics = evaluator.evaluate_selector(accepted_df, context.validation_bundle)
        metrics_json = _jsonify(metrics)
        metrics_path.write_text(
            json.dumps(metrics_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        status = "passed" if evaluator.passes_selector(metrics) else "failed"
        return {
            "condition": self.spec.name,
            "stage": self.spec.stage,
            "status": status,
            "passed": status == "passed",
            "metrics": metrics,
            "accepted_df": accepted_df,
            "selector_meta_df": _selector_meta_df(context.validation_bundle),
            "artifacts": {
                "accepted_rows_path": _relative_to_repo(accepted_rows_path, repo_root),
                "selected_pool_path": _relative_to_repo(selected_pool_path, repo_root),
                "component_summary_path": _relative_to_repo(
                    component_summary_path,
                    repo_root,
                ),
                "selection_summary_path": _relative_to_repo(
                    selection_summary_path,
                    repo_root,
                ),
                "selector_metrics_path": _relative_to_repo(metrics_path, repo_root),
            },
        }

    def run_probe(
        self,
        context: ConditionContext,
        repo: ManifestRepository,
        runner: TrainingProbeRunner,
        evaluator: SelectorEvaluator,
        device: torch.device,
    ) -> dict[str, object]:
        del device
        if context.strict_selector_result is None:
            return self._skip_result(reason="strict_selector_missing")

        base_manifest_df = self.compose_supervised_base_manifest(
            repo,
            context.strict_selector_result,
        )
        accepted_df = context.strict_selector_result["accepted_df"].copy()
        probe_train_df = self.compose_probe_training_manifest(base_manifest_df, accepted_df)
        validation_df = repo.load_manifest("val_unlabeled")
        heldout_test_df = repo.load_manifest("heldout_test")
        prepared_paths = repo.materialize_prepared_directory(
            train_df=probe_train_df,
            eval_df=validation_df,
            test_df=heldout_test_df,
            output_dir=self._stage_root("probe"),
        )

        seed_results: list[dict[str, object]] = []
        successful_eval_results: list[dict[str, object]] = []
        seed_values = _seed_values_for_spec(self.config, self.spec)
        repo_root = self._resolved_paths["repo_root"]
        best_checkpoint_path: str | None = None
        best_seed_score = float("-inf")
        for seed in seed_values:
            train_result = runner.launch_one_epoch_probe(
                condition_name=self.spec.name,
                prepared_paths=prepared_paths,
                seed=seed,
            )
            seed_payload: dict[str, object] = {"seed": seed, **_jsonify(train_result)}
            if not bool(train_result.get("success", False)):
                seed_results.append(seed_payload)
                continue

            override_config = Path(str(train_result["override_config"]))
            checkpoint_path = Path(str(train_result["checkpoint_path"]))
            eval_result = runner.evaluate_checkpoint(
                condition_name=self.spec.name,
                override_config=override_config,
                checkpoint_path=checkpoint_path,
                seed=seed,
                split_name="validation",
            )
            seed_payload.update(_jsonify(eval_result))
            seed_results.append(seed_payload)
            successful_eval_results.append(
                {
                    "seed": seed,
                    "checkpoint_path": _relative_to_repo(checkpoint_path, repo_root),
                    **{key: float(value) for key, value in eval_result.items()},
                }
            )
            score = float(eval_result.get("validation_precision@10", float("-inf")))
            if score > best_seed_score:
                best_seed_score = score
                best_checkpoint_path = _relative_to_repo(checkpoint_path, repo_root)

        aggregate_metrics = self._aggregate_probe_metrics(
            successful_eval_results,
            evaluator=evaluator,
        )
        success_rate = float(len(successful_eval_results) / max(len(seed_values), 1))
        status = "passed" if successful_eval_results else "failed"
        base_manifest_kind = str(base_manifest_df.attrs.get("base_manifest_kind", "full_train"))
        seed_warning = None
        if self.spec.seed_count < self.config.fallback_seed_count:
            seed_warning = (
                "SEED_WARNING: probe ran below documented fallback minimum "
                f"({self.spec.seed_count} < {self.config.fallback_seed_count})"
            )
        return {
            "condition": self.spec.name,
            "stage": self.spec.stage,
            "status": status,
            "passed": status == "passed",
            "seed_results": seed_results,
            "aggregate_metrics": aggregate_metrics,
            "success_rate": success_rate,
            "best_checkpoint_path": best_checkpoint_path,
            "base_manifest_kind": base_manifest_kind,
            "seed_count": int(self.spec.seed_count),
            "seed_warning": seed_warning,
            "artifacts": {
                "prepared_dir": _relative_to_repo(prepared_paths["prepared_dir"], repo_root),
                "train_manifest": _relative_to_repo(
                    prepared_paths["train_manifest"],
                    repo_root,
                ),
                "validation_manifest": _relative_to_repo(
                    prepared_paths["validation_manifest"],
                    repo_root,
                ),
                "test_manifest": _relative_to_repo(
                    prepared_paths["test_manifest"],
                    repo_root,
                ),
                "split_summary": _relative_to_repo(
                    prepared_paths["split_summary"],
                    repo_root,
                ),
                "speaker_to_index": _relative_to_repo(
                    prepared_paths["speaker_to_index"],
                    repo_root,
                ),
            },
        }

    def run_round2(
        self,
        context: ConditionContext,
        repo: ManifestRepository,
        runner: TrainingProbeRunner,
        evaluator: SelectorEvaluator,
        device: torch.device,
    ) -> dict[str, object]:
        if context.strict_selector_result is None:
            return self._skip_result(reason="strict_selector_missing")
        if context.stage1_winner is None:
            return self._skip_result(reason="stage1_winner_missing")

        best_checkpoint_path = context.stage1_winner.get("best_checkpoint_path")
        if not best_checkpoint_path:
            return self._skip_result(reason="stage1_checkpoint_missing")
        repo_root = self._resolved_paths["repo_root"]

        student_bundle = runner.build_student_bundle(
            checkpoint_path=_resolve_from_repo(str(best_checkpoint_path), repo_root),
            repo=repo,
            device=device,
        )
        round2_gate = runner.round2_gate_metrics(
            teacher_bundle=context.validation_bundle,
            student_bundle=student_bundle,
        )
        if not bool(round2_gate.get("pass_round2", False)):
            return {
                **self._skip_result(reason="round2_gate_not_met"),
                "round2_gate_metrics": round2_gate,
            }
        round2_bundle = {**context.validation_bundle, **student_bundle}

        strict_spec = next(
            spec
            for spec in self.config.build_condition_specs()
            if spec.class_name == _STRICT_SELECTOR_NAME
        )
        strict_selector = AcousticPriorLowHubDiverseMutualGraphPseudoLabels(
            strict_spec,
            self.config,
        )
        round2_candidate_df = strict_selector.select_pseudo_labels(
            round2_bundle,
            context.prior,
        )
        refined_df = self.refine_round2_assignments(
            round1_df=context.strict_selector_result["accepted_df"],
            round2_df=round2_candidate_df,
            bundle=context.validation_bundle,
            student_bundle=student_bundle,
        )
        retention_rate = float(
            len(refined_df) / max(len(context.strict_selector_result["accepted_df"]), 1)
        )
        if retention_rate < self.config.round2_retention_gate:
            return {
                **self._skip_result(reason="round2_retention_below_gate"),
                "retention_rate": retention_rate,
                "round2_gate_metrics": round2_gate,
            }

        base_manifest_kind = str(context.stage1_winner.get("base_manifest_kind", "full_train"))
        base_manifest_df = repo.load_manifest(base_manifest_kind)
        base_manifest_df.attrs["base_manifest_kind"] = base_manifest_kind
        round2_train_df = self.compose_probe_training_manifest(base_manifest_df, refined_df)
        validation_df = repo.load_manifest("val_unlabeled")
        heldout_test_df = repo.load_manifest("heldout_test")
        prepared_paths = repo.materialize_prepared_directory(
            train_df=round2_train_df,
            eval_df=validation_df,
            test_df=heldout_test_df,
            output_dir=self._stage_root("round2"),
        )

        seed_results: list[dict[str, object]] = []
        successful_eval_results: list[dict[str, object]] = []
        seed_values = _seed_values_for_spec(self.config, self.spec)
        best_round2_checkpoint: str | None = None
        best_round2_score = float("-inf")
        for seed in seed_values:
            train_result = runner.launch_one_epoch_probe(
                condition_name=self.spec.name,
                prepared_paths=prepared_paths,
                seed=seed,
            )
            seed_payload: dict[str, object] = {"seed": seed, **_jsonify(train_result)}
            if not bool(train_result.get("success", False)):
                seed_results.append(seed_payload)
                continue

            override_config = Path(str(train_result["override_config"]))
            checkpoint_path = Path(str(train_result["checkpoint_path"]))
            eval_result = runner.evaluate_checkpoint(
                condition_name=self.spec.name,
                override_config=override_config,
                checkpoint_path=checkpoint_path,
                seed=seed,
                split_name="validation",
            )
            seed_payload.update(_jsonify(eval_result))
            seed_results.append(seed_payload)
            successful_eval_results.append(
                {
                    "seed": seed,
                    "checkpoint_path": _relative_to_repo(checkpoint_path, repo_root),
                    **{key: float(value) for key, value in eval_result.items()},
                }
            )
            score = float(eval_result.get("validation_precision@10", float("-inf")))
            if score > best_round2_score:
                best_round2_score = score
                best_round2_checkpoint = _relative_to_repo(checkpoint_path, repo_root)

        aggregate_metrics = self._aggregate_probe_metrics(
            successful_eval_results,
            evaluator=evaluator,
        )
        stage1_mean = _metric_mean_from_result(
            context.stage1_winner,
            "validation_precision@10",
        )
        round2_mean = float(
            aggregate_metrics.get("validation_precision@10", {}).get(
                "mean",
                float("-inf"),
            )
        )
        delta_p10 = round2_mean - stage1_mean
        status = (
            "passed"
            if successful_eval_results and delta_p10 >= -self.config.round2_max_drop_p10
            else "failed"
        )
        seed_warning = None
        if self.spec.seed_count < self.config.fallback_seed_count:
            seed_warning = (
                "SEED_WARNING: round2 ran below documented fallback minimum "
                f"({self.spec.seed_count} < {self.config.fallback_seed_count})"
            )
        return {
            "condition": self.spec.name,
            "stage": self.spec.stage,
            "status": status,
            "passed": status == "passed",
            "seed_results": seed_results,
            "aggregate_metrics": aggregate_metrics,
            "success_rate": float(len(successful_eval_results) / max(len(seed_values), 1)),
            "best_checkpoint_path": best_round2_checkpoint,
            "delta_validation_precision@10": delta_p10,
            "retention_rate": retention_rate,
            "round2_candidate_rows": int(len(round2_candidate_df)),
            "round2_refined_rows": int(len(refined_df)),
            "round2_gate_metrics": round2_gate,
            "base_manifest_kind": base_manifest_kind,
            "seed_count": int(self.spec.seed_count),
            "seed_warning": seed_warning,
            "artifacts": {
                "prepared_dir": _relative_to_repo(prepared_paths["prepared_dir"], repo_root),
                "train_manifest": _relative_to_repo(
                    prepared_paths["train_manifest"],
                    repo_root,
                ),
                "validation_manifest": _relative_to_repo(
                    prepared_paths["validation_manifest"],
                    repo_root,
                ),
                "test_manifest": _relative_to_repo(
                    prepared_paths["test_manifest"],
                    repo_root,
                ),
                "split_summary": _relative_to_repo(
                    prepared_paths["split_summary"],
                    repo_root,
                ),
                "speaker_to_index": _relative_to_repo(
                    prepared_paths["speaker_to_index"],
                    repo_root,
                ),
            },
        }

    def select_pseudo_labels(
        self,
        bundle: dict[str, object],
        prior: dict[str, object],
    ) -> pd.DataFrame:
        del bundle, prior
        return pd.DataFrame(columns=_SELECTOR_COLUMNS)

    def compose_supervised_base_manifest(
        self,
        repo: ManifestRepository,
        strict_selector_result: dict[str, object],
    ) -> pd.DataFrame:
        del strict_selector_result
        base_df = repo.load_manifest("full_train").copy()
        base_df.attrs["base_manifest_kind"] = "full_train"
        return base_df

    def compose_probe_training_manifest(
        self,
        base_manifest_df: pd.DataFrame,
        accepted_df: pd.DataFrame,
    ) -> pd.DataFrame:
        base_manifest = base_manifest_df.loc[:, _MANIFEST_COLUMNS].copy()
        base_manifest["spk"] = base_manifest["spk"].astype(str)
        if accepted_df.empty:
            return base_manifest.reset_index(drop=True)

        val_manifest_path = self._resolved_paths["val_unlabeled_manifest"]
        val_manifest = pd.read_csv(val_manifest_path, usecols=_MANIFEST_COLUMNS)
        row_indices = accepted_df["row_index"].astype(np.int64).to_numpy(copy=False)
        pseudo_rows = val_manifest.iloc[row_indices].copy().reset_index(drop=True)
        if len(pseudo_rows) != len(accepted_df):
            raise ValueError("Pseudo manifest extraction misaligned with accepted_df")
        pseudo_rows["spk"] = accepted_df["pseudo_spk"].astype(str).to_numpy(copy=False)
        pseudo_rows = pseudo_rows.loc[:, _MANIFEST_COLUMNS]
        pseudo_rows = pseudo_rows.sort_values(
            by=["spk", "path"],
            kind="mergesort",
        ).reset_index(drop=True)
        combined = pd.concat([base_manifest, pseudo_rows], axis=0, ignore_index=True)
        combined = combined.drop_duplicates(subset=["ID"], keep="first")
        return combined.reset_index(drop=True)

    def refine_round2_assignments(
        self,
        round1_df: pd.DataFrame,
        round2_df: pd.DataFrame,
        bundle: dict[str, object],
        student_bundle: dict[str, object],
    ) -> pd.DataFrame:
        del round1_df, round2_df, bundle, student_bundle
        return pd.DataFrame(columns=_SELECTOR_COLUMNS + ["margin_gain", "top10_overlap"])

    def _skip_result(self, reason: str) -> dict[str, object]:
        nan_value = float("nan")
        seed_results = [
            {
                "seed": seed,
                "primary_metric": nan_value,
                "primary_metric_source": self._primary_metric_source(),
            }
            for seed in _seed_values_for_spec(self.config, self.spec)
        ]
        return {
            "condition": self.spec.name,
            "stage": self.spec.stage,
            "status": "skipped",
            "passed": False,
            "reason": reason,
            "primary_metric": nan_value,
            "primary_metric_source": self._primary_metric_source(),
            "seed_results": seed_results,
            "aggregate_metrics": {
                "primary_metric": {
                    "mean": nan_value,
                    "std": nan_value,
                    "values": [nan_value for _ in seed_results],
                }
            },
            "artifacts": {},
        }

    def _primary_metric_source(self) -> str:
        if self.spec.stage == "selector":
            return "simulated_pairwise_purity"
        return "validation_precision@10"

    def _finalize_result(self, result: dict[str, object]) -> dict[str, object]:
        primary_metric_source = self._primary_metric_source()
        result["primary_metric_source"] = primary_metric_source
        expected_seeds = _seed_values_for_spec(self.config, self.spec)
        nan_value = float("nan")

        if self.spec.stage == "selector":
            metrics = result.get("metrics")
            metric_value = nan_value
            accepted_rows = 0
            if isinstance(metrics, dict):
                metric_value = float(metrics.get(primary_metric_source, nan_value))
                accepted_rows = int(metrics.get("accepted_pseudo_rows", 0))
                metrics["primary_metric"] = metric_value
            result["primary_metric"] = metric_value
            result["seed_results"] = [
                {
                    "seed": seed,
                    primary_metric_source: metric_value,
                    "accepted_pseudo_rows": accepted_rows,
                    "primary_metric": metric_value,
                    "primary_metric_source": primary_metric_source,
                }
                for seed in expected_seeds
            ]
            result["aggregate_metrics"] = {
                **(
                    result.get("aggregate_metrics", {})
                    if isinstance(result.get("aggregate_metrics"), dict)
                    else {}
                ),
                "primary_metric": {
                    "mean": metric_value,
                    "std": 0.0,
                    "values": [metric_value for _ in expected_seeds],
                },
            }
            return result

        aggregate_metrics = result.get("aggregate_metrics")
        if not isinstance(aggregate_metrics, dict):
            aggregate_metrics = {}
        primary_payload = aggregate_metrics.get(primary_metric_source)
        if isinstance(primary_payload, dict):
            primary_metric = float(primary_payload.get("mean", nan_value))
            aggregate_metrics.setdefault(
                "primary_metric",
                {key: value for key, value in primary_payload.items()},
            )
        else:
            primary_metric = nan_value
            aggregate_metrics.setdefault(
                "primary_metric",
                {
                    "mean": nan_value,
                    "std": nan_value,
                    "values": [nan_value for _ in expected_seeds],
                },
            )
        result["aggregate_metrics"] = aggregate_metrics
        result["primary_metric"] = primary_metric

        raw_seed_results = result.get("seed_results")
        seed_rows: dict[int, dict[str, object]] = {}
        if isinstance(raw_seed_results, list):
            for row in raw_seed_results:
                if not isinstance(row, dict) or "seed" not in row:
                    continue
                seed = int(row["seed"])
                normalized_row = dict(row)
                normalized_row["primary_metric_source"] = primary_metric_source
                normalized_row["primary_metric"] = float(
                    normalized_row.get(primary_metric_source, nan_value)
                )
                seed_rows[seed] = normalized_row
        for seed in expected_seeds:
            seed_rows.setdefault(
                seed,
                {
                    "seed": seed,
                    "primary_metric": nan_value,
                    "primary_metric_source": primary_metric_source,
                },
            )
        result["seed_results"] = [seed_rows[seed] for seed in expected_seeds]
        return result

    def _stage_root(self, stage_name: str) -> Path:
        root = self._resolved_paths["experiment_root"] / self.spec.name / stage_name
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _component_summary(self, accepted_df: pd.DataFrame) -> pd.DataFrame:
        if accepted_df.empty:
            return pd.DataFrame(
                columns=[
                    "component_id",
                    "component_size",
                    "pseudo_spk",
                    "mean_top1_cosine",
                    "mean_top1_margin",
                ]
            )
        aggregations: dict[str, tuple[str, str]] = {
            "component_size": ("component_size", "max"),
            "pseudo_spk": ("pseudo_spk", "first"),
            "mean_top1_cosine": ("top1_cosine", "mean"),
            "mean_top1_margin": ("top1_margin", "mean"),
        }
        for optional_column in ("prior_distance", "diversity_score", "inbound_degree"):
            if optional_column in accepted_df.columns:
                aggregations[f"mean_{optional_column}"] = (optional_column, "mean")
        return accepted_df.groupby("component_id", sort=True).agg(**aggregations).reset_index()

    def _aggregate_probe_metrics(
        self,
        successful_eval_results: list[dict[str, object]],
        evaluator: SelectorEvaluator,
    ) -> dict[str, dict[str, object]]:
        if not successful_eval_results:
            return {}
        metric_names = [
            key
            for key, value in successful_eval_results[0].items()
            if key not in {"seed", "checkpoint_path"}
            and isinstance(value, (int, float, np.integer, np.floating))
        ]
        aggregate: dict[str, dict[str, object]] = {}
        for metric_name in metric_names:
            values = np.asarray(
                [float(item[metric_name]) for item in successful_eval_results],
                dtype=np.float32,
            )
            stats = evaluator.bootstrap_ci(values, rng_seed=self.config.selector_seed)
            aggregate[metric_name] = {
                **_jsonify(stats),
                "values": values.tolist(),
            }
        return aggregate

    def _build_mutual_margin_edges(
        self,
        bundle: dict[str, object],
    ) -> np.ndarray:
        topk_idx, _, top1_margin = _bundle_arrays(bundle)
        num_rows = topk_idx.shape[0]
        neighbor_sets = [set(row.tolist()) for row in topk_idx]
        edges: list[tuple[int, int]] = []
        for src_index in range(num_rows):
            if float(top1_margin[src_index]) < self.config.min_margin:
                continue
            for dst_index in topk_idx[src_index]:
                dst = int(dst_index)
                if src_index == dst:
                    continue
                if src_index not in neighbor_sets[dst]:
                    continue
                if float(top1_margin[dst]) < self.config.min_margin:
                    continue
                left, right = sorted((src_index, dst))
                edges.append((left, right))
        if not edges:
            return np.zeros((0, 2), dtype=np.int64)
        edge_array = np.asarray(edges, dtype=np.int64)
        edge_array = np.unique(edge_array, axis=0)
        return edge_array

    def _component_frame(
        self,
        bundle: dict[str, object],
        row_indices: np.ndarray,
        component_id: int,
        prefix: str,
        extra_columns: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        selector_meta = _selector_meta_df(bundle)
        topk_idx, topk_sim, top1_margin = _bundle_arrays(bundle)
        component_rows = selector_meta.iloc[row_indices].copy().reset_index(drop=True)
        component_rows.insert(0, "row_index", row_indices.astype(np.int64, copy=False))
        component_rows["component_id"] = int(component_id)
        component_rows["component_size"] = int(len(row_indices))
        component_rows["pseudo_spk"] = f"{prefix}_{component_id}"
        component_rows["anchor_index"] = topk_idx[row_indices, 0].astype(np.int64, copy=False)
        component_rows["top1_cosine"] = topk_sim[row_indices, 0].astype(np.float32, copy=False)
        component_rows["top1_margin"] = top1_margin[row_indices].astype(np.float32, copy=False)
        if extra_columns:
            for name, value in extra_columns.items():
                if isinstance(value, np.ndarray):
                    component_rows[name] = value[row_indices]
                else:
                    component_rows[name] = value
        preferred_columns = [
            "row_index",
            "component_id",
            "component_size",
            "pseudo_spk",
            "anchor_index",
            "top1_cosine",
            "top1_margin",
            "path",
            "dur",
            "start",
            "stop",
            "orig_filepath",
            "manifest_split",
            "duration_bucket",
            "profile_bucket",
            "prior_distance",
        ]
        extra_tail = [
            column for column in component_rows.columns if column not in preferred_columns
        ]
        return component_rows.loc[:, preferred_columns + extra_tail]


class AdaptiveConfidenceOnlyCosineGate(BaseConditionStrategy):
    def select_pseudo_labels(
        self,
        bundle: dict[str, object],
        prior: dict[str, object],
    ) -> pd.DataFrame:
        del prior
        selector_meta = _selector_meta_df(bundle)
        topk_idx, topk_sim, top1_margin = _bundle_arrays(bundle)
        if topk_idx.shape[0] == 0:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS)

        top1_idx = topk_idx[:, 0]
        top1_cosine = topk_sim[:, 0]
        regime_key = (
            selector_meta["duration_bucket"].astype(str)
            + "__"
            + selector_meta["profile_bucket"].astype(str)
        )
        tau_by_regime: dict[str, float] = {}
        for regime_value in regime_key.unique().tolist():
            mask = regime_key == regime_value
            regime_scores = top1_cosine[mask.to_numpy(dtype=bool, copy=False)]
            if regime_scores.size == 0:
                continue
            tau = max(
                self.config.min_top1_cosine,
                float(np.quantile(regime_scores, self.config.top1_quantile_by_regime)),
            )
            tau_by_regime[regime_value] = tau

        directed_edges: list[tuple[int, int]] = []
        for row_index in range(len(selector_meta)):
            tau = tau_by_regime[str(regime_key.iloc[row_index])]
            if float(top1_cosine[row_index]) < tau:
                continue
            dst = int(top1_idx[row_index])
            if row_index == dst:
                continue
            left, right = sorted((row_index, dst))
            directed_edges.append((left, right))

        if not directed_edges:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS)
        edge_array = np.unique(np.asarray(directed_edges, dtype=np.int64), axis=0)
        labels, counts = _connected_components_for_edges(len(selector_meta), edge_array)

        frames: list[pd.DataFrame] = []
        next_component_id = 0
        for raw_component_id, component_size in enumerate(counts.tolist()):
            if component_size < 2:
                continue
            component_rows = np.flatnonzero(labels == raw_component_id)
            frames.append(
                self._component_frame(
                    bundle=bundle,
                    row_indices=component_rows,
                    component_id=next_component_id,
                    prefix="adaptive_r1",
                )
            )
            next_component_id += 1
        if not frames:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS)
        accepted_df = pd.concat(frames, axis=0, ignore_index=True)
        return accepted_df.reset_index(drop=True)


class MutualKnnMarginGraphGate(BaseConditionStrategy):
    def select_pseudo_labels(
        self,
        bundle: dict[str, object],
        prior: dict[str, object],
    ) -> pd.DataFrame:
        del prior
        selector_meta = _selector_meta_df(bundle)
        if selector_meta.empty:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS)
        edge_array = self._build_mutual_margin_edges(bundle)
        labels, counts = _connected_components_for_edges(len(selector_meta), edge_array)

        frames: list[pd.DataFrame] = []
        next_component_id = 0
        for raw_component_id, component_size in enumerate(counts.tolist()):
            if component_size < self.config.min_component_size:
                continue
            component_rows = np.flatnonzero(labels == raw_component_id)
            frames.append(
                self._component_frame(
                    bundle=bundle,
                    row_indices=component_rows,
                    component_id=next_component_id,
                    prefix="mutual_r1",
                )
            )
            next_component_id += 1
        if not frames:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS)
        accepted_df = pd.concat(frames, axis=0, ignore_index=True)
        return accepted_df.reset_index(drop=True)


class AcousticPriorLowHubDiverseMutualGraphPseudoLabels(BaseConditionStrategy):
    def select_pseudo_labels(
        self,
        bundle: dict[str, object],
        prior: dict[str, object],
    ) -> pd.DataFrame:
        selected_df, selection_summary = select_pseudo_label_rows(
            bundle=bundle,
            prior=prior,
            selection=PseudoLabelSelectionConfig(
                min_cluster_size=self.config.min_component_size,
                max_cluster_size=self.config.strict_max_component_size,
                min_top1_score=self.config.strict_min_top1_score,
                min_top1_margin=(
                    self.config.strict_min_top1_margin
                    if self.config.strict_min_top1_margin is not None
                    else self.config.min_margin
                ),
                max_indegree_quantile=self.config.hubness_quantile,
                indegree_top_k=self.config.strict_indegree_top_k,
                max_rows_per_cluster=self.config.strict_max_rows_per_component,
            ),
            feature_names=list(self.config.v2_features),
            label_prefix="strict_r1",
        )
        selected_df.attrs["selection_summary"] = selection_summary
        return selected_df


class NoHubnessAndDiversityGateInAcousticPriorGraph(BaseConditionStrategy):
    def select_pseudo_labels(
        self,
        bundle: dict[str, object],
        prior: dict[str, object],
    ) -> pd.DataFrame:
        selector_meta = _selector_meta_df(bundle)
        if selector_meta.empty:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS)
        base_edges = self._build_mutual_margin_edges(bundle)
        labels, counts = _connected_components_for_edges(len(selector_meta), base_edges)

        mu = np.asarray(prior["mu"], dtype=np.float32)
        scale = np.asarray(prior["scale"], dtype=np.float32)
        weight = np.asarray(prior["weight"], dtype=np.float32)
        feature_names = list(self.config.v2_features)

        frames: list[pd.DataFrame] = []
        next_component_id = 0
        for raw_component_id, component_size in enumerate(counts.tolist()):
            if component_size < self.config.min_component_size:
                continue
            component_rows = np.flatnonzero(labels == raw_component_id)
            x_comp = (
                selector_meta.iloc[component_rows]
                .loc[:, feature_names]
                .to_numpy(
                    dtype=np.float32,
                    copy=True,
                )
            )
            centroid = np.median(x_comp, axis=0).astype(np.float32, copy=False)
            prior_distance = float(
                np.sum(weight * np.square((centroid - mu) / scale), dtype=np.float32)
            )
            if prior_distance > float(prior["prior_distance_threshold"]):
                continue
            frames.append(
                self._component_frame(
                    bundle=bundle,
                    row_indices=component_rows,
                    component_id=next_component_id,
                    prefix="nohubs_nodiv_r1",
                    extra_columns={"prior_distance": prior_distance},
                )
            )
            next_component_id += 1
        if not frames:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS)
        accepted_df = pd.concat(frames, axis=0, ignore_index=True)
        return accepted_df.reset_index(drop=True)


class SampleV2AcousticPriorWithoutFullTrainBaseSwap(BaseConditionStrategy):
    def compose_supervised_base_manifest(
        self,
        repo: ManifestRepository,
        strict_selector_result: dict[str, object],
    ) -> pd.DataFrame:
        del strict_selector_result
        base_df = repo.load_manifest("sample_v2").copy()
        base_df = base_df.sort_values(by=["spk", "path"], kind="mergesort").reset_index(drop=True)
        base_df.attrs["base_manifest_kind"] = "sample_v2"
        return base_df


class FullTrainBaseWithAcousticPriorStrictPseudoLabels(BaseConditionStrategy):
    def compose_supervised_base_manifest(
        self,
        repo: ManifestRepository,
        strict_selector_result: dict[str, object],
    ) -> pd.DataFrame:
        del strict_selector_result
        base_df = repo.load_manifest("full_train").copy()
        base_df = base_df.sort_values(by=["spk", "path"], kind="mergesort").reset_index(drop=True)
        base_df.attrs["base_manifest_kind"] = "full_train"
        return base_df


class StableCoreMarginGainRound2Contraction(BaseConditionStrategy):
    def refine_round2_assignments(
        self,
        round1_df: pd.DataFrame,
        round2_df: pd.DataFrame,
        bundle: dict[str, object],
        student_bundle: dict[str, object],
    ) -> pd.DataFrame:
        teacher_topk = np.asarray(bundle["topk_idx"], dtype=np.int64)
        student_topk = np.asarray(student_bundle["topk_idx"], dtype=np.int64)
        merged = round1_df.merge(
            round2_df,
            on="row_index",
            suffixes=("_r1", "_r2"),
            how="inner",
        )
        if merged.empty:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS + ["margin_gain", "top10_overlap"])

        margin_gain = merged["top1_margin_r2"].to_numpy(dtype=np.float32) - merged[
            "top1_margin_r1"
        ].to_numpy(dtype=np.float32)
        overlap_values = []
        for row_index in merged["row_index"].astype(np.int64).tolist():
            teacher_set = set(teacher_topk[row_index, :10].tolist())
            student_set = set(student_topk[row_index, :10].tolist())
            overlap_values.append(len(teacher_set & student_set) / 10.0)
        merged["margin_gain"] = margin_gain
        merged["top10_overlap"] = np.asarray(overlap_values, dtype=np.float32)
        merged = merged.loc[merged["margin_gain"] >= self.config.round2_margin_gain_gate].copy()
        if merged.empty:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS + ["margin_gain", "top10_overlap"])

        merged["stable_group_key"] = list(
            zip(
                merged["component_id_r1"].astype(int),
                merged["component_id_r2"].astype(int),
                strict=False,
            )
        )
        keep_group_keys: list[tuple[int, int]] = []
        for _, round2_component_df in merged.groupby("component_id_r2", sort=False):
            stable_counts = (
                round2_component_df.groupby("stable_group_key", sort=False)
                .size()
                .reset_index(name="stable_size")
                .sort_values(
                    by=["stable_size"],
                    ascending=[False],
                    kind="mergesort",
                )
            )
            if stable_counts.empty:
                continue
            keep_group_keys.append(stable_counts.iloc[0]["stable_group_key"])

        refined = merged.loc[merged["stable_group_key"].isin(keep_group_keys)].copy()
        if refined.empty:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS + ["margin_gain", "top10_overlap"])

        component_size_by_key = refined.groupby("stable_group_key", sort=False).size().to_dict()
        next_component_id = 0
        frames: list[pd.DataFrame] = []
        for stable_key, stable_df in refined.groupby("stable_group_key", sort=False):
            stable_size = int(component_size_by_key[stable_key])
            if stable_size < self.config.min_component_size:
                continue
            component_frame = pd.DataFrame(
                {
                    "row_index": stable_df["row_index"].astype(np.int64).to_numpy(),
                    "component_id": next_component_id,
                    "component_size": stable_size,
                    "pseudo_spk": f"strict_r2_contract_{next_component_id}",
                    "anchor_index": stable_df["anchor_index_r2"].astype(np.int64).to_numpy(),
                    "top1_cosine": stable_df["top1_cosine_r2"].astype(np.float32).to_numpy(),
                    "top1_margin": stable_df["top1_margin_r2"].astype(np.float32).to_numpy(),
                    "path": stable_df["path_r2"].astype(str).to_numpy(),
                    "dur": stable_df["dur_r2"].astype(np.float32).to_numpy(),
                    "start": stable_df["start_r2"].to_numpy(),
                    "stop": stable_df["stop_r2"].to_numpy(),
                    "orig_filepath": stable_df["orig_filepath_r2"].astype(str).to_numpy(),
                    "manifest_split": stable_df["manifest_split_r2"].astype(str).to_numpy(),
                    "duration_bucket": stable_df["duration_bucket_r2"].to_numpy(),
                    "profile_bucket": stable_df["profile_bucket_r2"].to_numpy(),
                    "prior_distance": stable_df["prior_distance_r2"].astype(np.float32).to_numpy(),
                    "margin_gain": stable_df["margin_gain"].astype(np.float32).to_numpy(),
                    "top10_overlap": stable_df["top10_overlap"].astype(np.float32).to_numpy(),
                }
            )
            frames.append(component_frame)
            next_component_id += 1
        if not frames:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS + ["margin_gain", "top10_overlap"])
        return pd.concat(frames, axis=0, ignore_index=True)


class Round2CarryAllStableAssignmentsWithoutContraction(BaseConditionStrategy):
    def refine_round2_assignments(
        self,
        round1_df: pd.DataFrame,
        round2_df: pd.DataFrame,
        bundle: dict[str, object],
        student_bundle: dict[str, object],
    ) -> pd.DataFrame:
        teacher_topk = np.asarray(bundle["topk_idx"], dtype=np.int64)
        student_topk = np.asarray(student_bundle["topk_idx"], dtype=np.int64)
        merged = round1_df.merge(
            round2_df,
            on="row_index",
            suffixes=("_r1", "_r2"),
            how="inner",
        )
        if merged.empty:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS + ["margin_gain", "top10_overlap"])

        margin_gain = merged["top1_margin_r2"].to_numpy(dtype=np.float32) - merged[
            "top1_margin_r1"
        ].to_numpy(dtype=np.float32)
        overlap_values = []
        for row_index in merged["row_index"].astype(np.int64).tolist():
            teacher_set = set(teacher_topk[row_index, :10].tolist())
            student_set = set(student_topk[row_index, :10].tolist())
            overlap_values.append(len(teacher_set & student_set) / 10.0)
        merged["margin_gain"] = margin_gain
        merged["top10_overlap"] = np.asarray(overlap_values, dtype=np.float32)
        merged = merged.loc[merged["margin_gain"] >= self.config.round2_margin_gain_gate].copy()
        if merged.empty:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS + ["margin_gain", "top10_overlap"])

        component_sizes = merged.groupby("component_id_r2", sort=False).size().to_dict()
        merged["component_id"] = merged["component_id_r2"].astype(np.int64)
        merged["component_size"] = merged["component_id_r2"].map(component_sizes).astype(np.int64)
        merged = merged.loc[merged["component_size"] >= self.config.min_component_size].copy()
        if merged.empty:
            return pd.DataFrame(columns=_SELECTOR_COLUMNS + ["margin_gain", "top10_overlap"])

        return pd.DataFrame(
            {
                "row_index": merged["row_index"].astype(np.int64).to_numpy(),
                "component_id": merged["component_id"].astype(np.int64).to_numpy(),
                "component_size": merged["component_size"].astype(np.int64).to_numpy(),
                "pseudo_spk": merged["component_id"].map(
                    lambda value: f"strict_r2_allstable_{int(value)}"
                ),
                "anchor_index": merged["anchor_index_r2"].astype(np.int64).to_numpy(),
                "top1_cosine": merged["top1_cosine_r2"].astype(np.float32).to_numpy(),
                "top1_margin": merged["top1_margin_r2"].astype(np.float32).to_numpy(),
                "path": merged["path_r2"].astype(str).to_numpy(),
                "dur": merged["dur_r2"].astype(np.float32).to_numpy(),
                "start": merged["start_r2"].to_numpy(),
                "stop": merged["stop_r2"].to_numpy(),
                "orig_filepath": merged["orig_filepath_r2"].astype(str).to_numpy(),
                "manifest_split": merged["manifest_split_r2"].astype(str).to_numpy(),
                "duration_bucket": merged["duration_bucket_r2"].to_numpy(),
                "profile_bucket": merged["profile_bucket_r2"].to_numpy(),
                "prior_distance": merged["prior_distance_r2"].astype(np.float32).to_numpy(),
                "margin_gain": merged["margin_gain"].astype(np.float32).to_numpy(),
                "top10_overlap": merged["top10_overlap"].astype(np.float32).to_numpy(),
            }
        )
