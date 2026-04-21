from __future__ import annotations

import contextlib
import importlib
import io
import json
import multiprocessing as mp
import os
import shlex
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import wilcoxon

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
    from methods import CamppEmbeddingModel
except ImportError:  # pragma: no cover - package-style import fallback
    from .methods import CamppEmbeddingModel

from common import load_embedding_checkpoint  # noqa: E402

_DURATION_BUCKETS = ("short_lt_3s", "medium_3_to_6s", "long_gt_6s")
_PROFILE_BUCKETS = ("near_v2_profile", "far_v2_profile")
_EVAL_METRIC_MAP = {
    "precision@1": "precision@1",
    "precision@5": "precision@5",
    "precision@10": "precision@10",
    "ndcg@10": "ndcg@10",
    "mrr@10": "mrr@10",
}


@dataclass(slots=True)
class _InProcessCliResult:
    returncode: int
    stdout: str
    stderr: str


def _metric_prefix_for_split(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    if normalized == "validation":
        return "validation"
    if normalized == "test":
        return "heldout_test"
    raise ValueError(f"Unsupported evaluation split for metric prefix: {split_name!r}")


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


def _pathish_string(value: str, repo_root: Path) -> str:
    try:
        candidate = Path(value)
    except (TypeError, ValueError):
        return value
    if not candidate.is_absolute():
        return value
    try:
        return str(candidate.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return value


def _jsonify(value: Any, repo_root: Path) -> Any:
    if isinstance(value, Path):
        return _relative_to_repo(value, repo_root)
    if isinstance(value, pd.DataFrame):
        return [_jsonify(row, repo_root) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _jsonify(value.to_dict(), repo_root)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item, repo_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item, repo_root) for item in value]
    if isinstance(value, str):
        return _pathish_string(value, repo_root)
    return value


def _require_bundle_df(bundle: dict[str, object], key: str) -> pd.DataFrame:
    value = bundle.get(key)
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"bundle[{key!r}] must be a pandas DataFrame")
    return value


def _ensure_required_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={emb.shape}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return emb / norms


@contextlib.contextmanager
def _temporary_argv(argv: list[str]):
    original_argv = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original_argv


@contextlib.contextmanager
def _temporary_cwd(path: Path):
    original_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _run_module_main(
    module_name: str,
    argv: list[str],
    cwd: Path,
) -> _InProcessCliResult:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_run_module_main_worker,
        args=(module_name, list(argv), str(cwd), child_conn),
    )
    process.start()
    child_conn.close()
    process.join()

    payload: dict[str, object] = {}
    if parent_conn.poll():
        payload = parent_conn.recv()
    parent_conn.close()

    returncode = int(
        payload.get("returncode", process.exitcode if process.exitcode is not None else 1)
    )
    stdout = str(payload.get("stdout", ""))
    stderr = str(payload.get("stderr", ""))
    if process.exitcode not in (None, 0) and not stderr:
        stderr = f"{module_name} worker exited with code {process.exitcode}"

    return _InProcessCliResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _run_module_main_in_current_process(
    module_name: str,
    argv: list[str],
    cwd: Path,
) -> _InProcessCliResult:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    returncode = 0

    with (
        contextlib.redirect_stdout(stdout_buffer),
        contextlib.redirect_stderr(stderr_buffer),
        _temporary_argv(argv),
        _temporary_cwd(cwd),
    ):
        try:
            module = importlib.import_module(module_name)
            entrypoint = getattr(module, "main", None)
            if not callable(entrypoint):
                raise AttributeError(f"{module_name}.main is not callable")
            entrypoint()
        except SystemExit as exc:
            if exc.code is None:
                returncode = 0
            elif isinstance(exc.code, int):
                returncode = int(exc.code)
            else:
                print(str(exc.code), file=sys.stderr)
                returncode = 1
        except Exception:
            traceback.print_exc(file=sys.stderr)
            returncode = 1

    return _InProcessCliResult(
        returncode=returncode,
        stdout=stdout_buffer.getvalue(),
        stderr=stderr_buffer.getvalue(),
    )


def _run_module_main_worker(
    module_name: str,
    argv: list[str],
    cwd: str,
    conn,
) -> None:
    try:
        result = _run_module_main_in_current_process(
            module_name=module_name,
            argv=argv,
            cwd=Path(cwd),
        )
        conn.send(
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    finally:
        conn.close()


@dataclass(slots=True)
class SelectorEvaluator:
    config: ExperimentConfig

    def evaluate_selector(
        self,
        accepted_df: pd.DataFrame,
        bundle: dict[str, object],
    ) -> dict[str, object]:
        selector_meta_df = _require_bundle_df(bundle, "selector_meta_df")
        oracle_labels = np.asarray(bundle["oracle_labels"], dtype=object)
        if oracle_labels.shape != (len(selector_meta_df),):
            raise ValueError("oracle_labels must align with selector_meta_df rows")

        accepted_eval_df = accepted_df.copy().reset_index(drop=True)
        if accepted_eval_df.empty:
            purity = 0.0
            accepted_rows = 0
            accepted_speakers = 0
            component_count = 0
            mean_component_size = 0.0
        else:
            _ensure_required_columns(
                accepted_eval_df,
                ["row_index", "pseudo_spk", "top1_margin", "top1_cosine"],
                "evaluate_selector accepted_df",
            )
            row_index = accepted_eval_df["row_index"].astype(np.int64).to_numpy(copy=False)
            if np.any(row_index < 0) or np.any(row_index >= len(selector_meta_df)):
                raise IndexError("accepted_df row_index is out of bounds for selector_meta_df")
            meta_subset = selector_meta_df.iloc[row_index].reset_index(drop=True)
            for column in (
                "duration_bucket",
                "profile_bucket",
                "path",
                "dur",
                "start",
                "stop",
                "orig_filepath",
                "manifest_split",
                "prior_distance",
            ):
                if column not in accepted_eval_df.columns and column in meta_subset.columns:
                    accepted_eval_df[column] = meta_subset[column].to_numpy(copy=False)
            accepted_eval_df["oracle_spk"] = oracle_labels[row_index]
            purity = self.pairwise_purity(accepted_eval_df)
            accepted_rows = int(len(accepted_eval_df))
            accepted_speakers = int(accepted_eval_df["pseudo_spk"].nunique())
            if "component_id" in accepted_eval_df.columns:
                component_count = int(accepted_eval_df["component_id"].nunique())
                mean_component_size = float(
                    accepted_eval_df.groupby("component_id", sort=False).size().mean()
                )
            else:
                component_count = int(accepted_eval_df["pseudo_spk"].nunique())
                mean_component_size = float(
                    accepted_eval_df.groupby("pseudo_spk", sort=False).size().mean()
                )

        metrics = {
            "simulated_pairwise_purity": float(purity),
            "accepted_pseudo_rows": int(accepted_rows),
            "accepted_pseudo_speakers": int(accepted_speakers),
            "component_count": int(component_count),
            "mean_component_size": float(mean_component_size),
            "regime_breakdown": self.regime_breakdown(accepted_eval_df, bundle),
        }
        metrics["passes_selector"] = self.passes_selector(metrics)
        metrics["purity_gate_passed"] = bool(
            metrics["simulated_pairwise_purity"] >= self.config.selector_pass_purity
        )
        metrics["soft_min_rows_gate_passed"] = bool(
            metrics["accepted_pseudo_rows"] >= self.config.selector_soft_min_rows
        )
        metrics["target_rows_gate_passed"] = bool(
            metrics["accepted_pseudo_rows"] >= self.config.selector_target_rows
        )
        return metrics

    def pairwise_purity(self, accepted_df: pd.DataFrame) -> float:
        _ensure_required_columns(
            accepted_df,
            ["pseudo_spk", "oracle_spk"],
            "pairwise_purity",
        )
        total_pred_pairs = 0.0
        total_true_positive_pairs = 0.0
        for _, component_df in accepted_df.groupby("pseudo_spk", sort=False):
            component_size = int(len(component_df))
            pred_pairs = component_size * (component_size - 1) / 2.0
            total_pred_pairs += pred_pairs
            if pred_pairs <= 0.0:
                continue
            oracle_counts = component_df["oracle_spk"].value_counts(dropna=False)
            total_true_positive_pairs += float(
                np.sum(
                    oracle_counts.to_numpy(dtype=np.float64)
                    * (oracle_counts.to_numpy(dtype=np.float64) - 1.0)
                    / 2.0
                )
            )
        return float(total_true_positive_pairs / max(total_pred_pairs, 1.0))

    def regime_breakdown(
        self,
        accepted_df: pd.DataFrame,
        bundle: dict[str, object],
    ) -> list[dict[str, object]]:
        selector_meta_df = _require_bundle_df(bundle, "selector_meta_df")
        oracle_labels = np.asarray(bundle["oracle_labels"], dtype=object)
        breakdown_df = accepted_df.copy().reset_index(drop=True)
        if not breakdown_df.empty:
            _ensure_required_columns(
                breakdown_df,
                ["row_index", "top1_margin", "top1_cosine"],
                "regime_breakdown accepted_df",
            )
            row_index = breakdown_df["row_index"].astype(np.int64).to_numpy(copy=False)
            meta_subset = selector_meta_df.iloc[row_index].reset_index(drop=True)
            for column in ("duration_bucket", "profile_bucket"):
                if column not in breakdown_df.columns:
                    breakdown_df[column] = meta_subset[column].to_numpy(copy=False)
            if "oracle_spk" not in breakdown_df.columns:
                breakdown_df["oracle_spk"] = oracle_labels[row_index]

        rows: list[dict[str, object]] = []
        for duration_bucket in _DURATION_BUCKETS:
            for profile_bucket in _PROFILE_BUCKETS:
                if breakdown_df.empty:
                    cell_df = breakdown_df
                else:
                    cell_df = breakdown_df.loc[
                        (breakdown_df["duration_bucket"].astype(str) == duration_bucket)
                        & (breakdown_df["profile_bucket"].astype(str) == profile_bucket)
                    ]
                row_payload: dict[str, object] = {
                    "duration_bucket": duration_bucket,
                    "profile_bucket": profile_bucket,
                    "rows": int(len(cell_df)),
                    "purity": float(self.pairwise_purity(cell_df)) if len(cell_df) > 0 else 0.0,
                    "median_top1_margin": (
                        float(np.median(cell_df["top1_margin"].to_numpy(dtype=np.float32)))
                        if len(cell_df) > 0
                        else None
                    ),
                    "median_top1_cosine": (
                        float(np.median(cell_df["top1_cosine"].to_numpy(dtype=np.float32)))
                        if len(cell_df) > 0
                        else None
                    ),
                }
                rows.append(row_payload)
        return rows

    def bootstrap_ci(self, values: np.ndarray, rng_seed: int) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("bootstrap_ci expects a non-empty 1D array")
        rng = np.random.default_rng(int(rng_seed))
        sample_indices = rng.integers(
            0,
            array.size,
            size=(int(self.config.bootstrap_samples), array.size),
            endpoint=False,
        )
        bootstrap_means = array[sample_indices].mean(axis=1)
        percentiles = np.percentile(bootstrap_means, [2.5, 97.5])
        return {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "ci95_low": float(percentiles[0]),
            "ci95_high": float(percentiles[1]),
        }

    def passes_selector(self, metrics: dict[str, object]) -> bool:
        accepted_rows = int(metrics.get("accepted_pseudo_rows", 0))
        purity = float(metrics.get("simulated_pairwise_purity", 0.0))
        if accepted_rows < self.config.selector_soft_min_rows:
            return False
        if purity < self.config.selector_pass_purity:
            return False
        if accepted_rows < self.config.selector_target_rows:
            return False
        return True

    def compare_seed_vectors(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> dict[str, float | str]:
        a_array = np.asarray(a, dtype=np.float64)
        b_array = np.asarray(b, dtype=np.float64)
        if a_array.shape != b_array.shape:
            raise ValueError("Seed vectors must have the same shape")
        if a_array.ndim != 1 or a_array.size == 0:
            raise ValueError("Seed vectors must be non-empty 1D arrays")
        delta = a_array - b_array
        if np.allclose(delta, 0.0):
            return {
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "p_value": "all_zero_deltas",
            }
        if a_array.size < 2:
            return {
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "p_value": "insufficient_overlap_for_wilcoxon",
            }
        statistic = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        return {
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "p_value": float(statistic.pvalue),
        }


@dataclass(slots=True)
class TrainingProbeRunner:
    config: ExperimentConfig
    resolved_paths: dict[str, Path] = field(init=False, repr=False)
    repo_root: Path = field(init=False)
    base_config_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.resolved_paths = self.config.paths.resolve_all()
        self.repo_root = self.resolved_paths["repo_root"]
        self.base_config_path = self.resolved_paths["campp_base_config"]

    def write_override_config(
        self,
        prepared_dir: Path,
        experiment_dir: Path,
        seed: int,
    ) -> Path:
        prepared_dir = prepared_dir.resolve()
        experiment_dir = experiment_dir.resolve()
        experiment_dir.mkdir(parents=True, exist_ok=True)

        payload = yaml.safe_load(self.base_config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Expected mapping in {self.base_config_path}")

        override_path = experiment_dir / f"override_seed_{int(seed):03d}.yaml"
        project_root_relative = os.path.relpath(self.repo_root, override_path.parent)
        payload["project_root"] = project_root_relative

        paths = payload.setdefault("paths", {})
        paths["experiment_root"] = _relative_to_repo(experiment_dir, self.repo_root)
        paths["train_manifest"] = _relative_to_repo(
            prepared_dir / "train_manifest.csv",
            self.repo_root,
        )
        paths["validation_manifest"] = _relative_to_repo(
            prepared_dir / "val_manifest.csv",
            self.repo_root,
        )
        paths["test_manifest"] = _relative_to_repo(
            prepared_dir / "test_manifest.csv",
            self.repo_root,
        )

        training = payload.setdefault("training", {})
        training["epochs"] = int(self.config.one_epoch_probe_epochs)

        evaluation = payload.setdefault("evaluation", {})
        evaluation["primary_mode"] = str(self.config.official_mode)
        evaluation["segment_count"] = int(self.config.official_segment_count)
        compare_modes = list(evaluation.get("compare_modes") or [])
        if self.config.official_mode not in compare_modes:
            compare_modes.append(self.config.official_mode)
        evaluation["compare_modes"] = compare_modes

        data_prep = payload.setdefault("data_prep", {})
        data_prep["seed"] = int(seed)
        data_prep["write_absolute_paths"] = False

        override_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return override_path

    def launch_one_epoch_probe(
        self,
        condition_name: str,
        prepared_paths: dict[str, Path],
        seed: int,
    ) -> dict[str, object]:
        experiment_dir = (
            self.resolved_paths["experiment_root"] / condition_name / f"seed_{int(seed):03d}"
        ).resolve()
        experiment_dir.mkdir(parents=True, exist_ok=True)

        prepared_dir = Path(prepared_paths["prepared_dir"]).resolve()
        override_config = self.write_override_config(
            prepared_dir=prepared_dir,
            experiment_dir=experiment_dir,
            seed=seed,
        )
        run_name = f"{condition_name}_seed{int(seed)}"
        run_root = experiment_dir / "runs" / run_name
        launcher_log_path = experiment_dir / "launcher_train.log"
        command = [
            "finetune_campp.py",
            "--config",
            str(override_config),
            "--run-name",
            run_name,
            "--max-epochs",
            str(int(self.config.one_epoch_probe_epochs)),
        ]
        completed = _run_module_main(
            module_name="finetune_campp",
            argv=command,
            cwd=self.repo_root,
        )
        launcher_log_path.write_text(
            "\n".join(
                [
                    f"command: {shlex.join(command)}",
                    f"returncode: {completed.returncode}",
                    "--- STDOUT ---",
                    completed.stdout,
                    "--- STDERR ---",
                    completed.stderr,
                ]
            ),
            encoding="utf-8",
        )

        summary_path = run_root / "run_summary.json"
        history_path = run_root / "history.csv"
        success = completed.returncode == 0 and summary_path.exists() and history_path.exists()

        summary_payload: dict[str, object] = {}
        history_row: dict[str, object] = {}
        checkpoint_path = run_root / "checkpoints" / "last.pt"
        epoch_seconds = float("nan")
        if summary_path.exists():
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            best_checkpoint = str(summary_payload.get("best_checkpoint") or "").strip()
            if best_checkpoint:
                checkpoint_path = _resolve_from_repo(best_checkpoint, self.repo_root)
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            if not history_df.empty:
                if "epoch" in history_df.columns and (history_df["epoch"] == 1).any():
                    epoch_row = history_df.loc[history_df["epoch"] == 1].iloc[0]
                else:
                    epoch_row = history_df.iloc[-1]
                history_row = {
                    str(key): _jsonify(value, self.repo_root)
                    for key, value in epoch_row.to_dict().items()
                }
                epoch_seconds = float(epoch_row.get("epoch_seconds", np.nan))
        if not np.isfinite(epoch_seconds):
            total_seconds = summary_payload.get("total_seconds")
            if total_seconds is not None:
                epoch_seconds = float(total_seconds)

        return {
            "success": bool(success and checkpoint_path.exists()),
            "returncode": int(completed.returncode),
            "override_config": _relative_to_repo(override_config, self.repo_root),
            "run_root": _relative_to_repo(run_root, self.repo_root),
            "launcher_log_path": _relative_to_repo(launcher_log_path, self.repo_root),
            "summary_path": _relative_to_repo(summary_path, self.repo_root),
            "history_path": _relative_to_repo(history_path, self.repo_root),
            "checkpoint_path": _relative_to_repo(checkpoint_path, self.repo_root),
            "epoch_seconds": float(epoch_seconds),
            "history_row": history_row,
            "run_summary": _jsonify(summary_payload, self.repo_root),
        }

    def evaluate_checkpoint(
        self,
        condition_name: str,
        override_config: Path,
        checkpoint_path: Path,
        seed: int,
        split_name: str = "validation",
    ) -> dict[str, float]:
        override_config = _resolve_from_repo(override_config, self.repo_root)
        checkpoint_path = _resolve_from_repo(checkpoint_path, self.repo_root)
        experiment_dir = override_config.parent.resolve()
        run_name = f"{condition_name}_eval_seed{int(seed)}"
        run_root = experiment_dir / "runs" / run_name
        launcher_log_path = experiment_dir / "launcher_eval.log"

        command = [
            "eval_campp.py",
            "--config",
            str(override_config),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            str(split_name),
            "--modes",
            str(self.config.official_mode),
            "--run-name",
            run_name,
        ]
        completed = _run_module_main(
            module_name="eval_campp",
            argv=command,
            cwd=self.repo_root,
        )
        launcher_log_path.write_text(
            "\n".join(
                [
                    f"command: {shlex.join(command)}",
                    f"returncode: {completed.returncode}",
                    "--- STDOUT ---",
                    completed.stdout,
                    "--- STDERR ---",
                    completed.stderr,
                ]
            ),
            encoding="utf-8",
        )

        metrics_path = run_root / "metrics.csv"
        summary_path = run_root / "run_summary.json"
        if completed.returncode != 0 or not metrics_path.exists() or not summary_path.exists():
            raise RuntimeError(
                f"Evaluation failed for {condition_name} seed={seed}; see {launcher_log_path}"
            )

        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            raise ValueError(f"metrics.csv is empty for {condition_name} seed={seed}")
        if "mode" not in metrics_df.columns:
            raise ValueError(f"metrics.csv is missing mode column: {metrics_path}")
        mode_rows = metrics_df.loc[metrics_df["mode"].astype(str) == str(self.config.official_mode)]
        metrics_row = mode_rows.iloc[0] if not mode_rows.empty else metrics_df.iloc[0]

        result: dict[str, float] = {}
        metric_prefix = _metric_prefix_for_split(split_name)
        for metric_suffix, source_key in _EVAL_METRIC_MAP.items():
            if source_key not in metrics_row:
                raise KeyError(f"{source_key} is missing in {metrics_path}")
            result[f"{metric_prefix}_{metric_suffix}"] = float(metrics_row[source_key])
        return result

    def build_student_bundle(
        self,
        checkpoint_path: Path,
        repo: ManifestRepository,
        device: torch.device,
    ) -> dict[str, object]:
        checkpoint_path = _resolve_from_repo(checkpoint_path, self.repo_root)
        model = CamppEmbeddingModel(self.config).to(device)
        load_embedding_checkpoint(checkpoint_path, model.backbone)
        prior = repo.build_sample_v2_prior()
        val_manifest_df = repo.load_manifest("val_unlabeled")
        enriched_df = repo.merge_manifest_with_acoustics(
            val_manifest_df,
            split_name="val_unlabeled",
        )
        annotated_df = repo.annotate_regime_buckets(enriched_df, prior=prior)
        selector_meta_df = annotated_df.drop(columns=["oracle_spk"]).reset_index(drop=True)
        oracle_labels = annotated_df["oracle_spk"].astype(str).to_numpy(copy=True)

        embeddings = np.asarray(
            model.encode_manifest_batches(selector_meta_df, device),
            dtype=np.float32,
        )
        embeddings = _normalize_embeddings(embeddings)
        if len(selector_meta_df) < 2:
            raise ValueError("Need at least two validation rows to build student bundle")

        cache_dataset = EmbeddingCacheDataset(self.config)
        effective_k = min(self.config.knn_k, len(selector_meta_df) - 1)
        topk_idx, topk_sim = cache_dataset.compute_topk_cosine(
            embeddings=embeddings,
            k=effective_k,
            chunk_size=1024,
        )
        if topk_sim.shape[1] >= 2:
            top1_margin = (topk_sim[:, 0] - topk_sim[:, 1]).astype(
                np.float32,
                copy=False,
            )
        else:
            top1_margin = np.zeros(len(selector_meta_df), dtype=np.float32)

        return {
            "selector_meta_df": selector_meta_df,
            "oracle_labels": oracle_labels,
            "embeddings": embeddings,
            "topk_idx": topk_idx.astype(np.int64, copy=False),
            "topk_sim": topk_sim.astype(np.float32, copy=False),
            "top1_margin": top1_margin,
        }

    def round2_gate_metrics(
        self,
        teacher_bundle: dict[str, object],
        student_bundle: dict[str, object],
    ) -> dict[str, float | bool]:
        teacher_topk = np.asarray(teacher_bundle["topk_idx"], dtype=np.int64)
        student_topk = np.asarray(student_bundle["topk_idx"], dtype=np.int64)
        teacher_margin = np.asarray(teacher_bundle["top1_margin"], dtype=np.float32)
        student_margin = np.asarray(student_bundle["top1_margin"], dtype=np.float32)
        if teacher_topk.shape[0] != student_topk.shape[0]:
            raise ValueError("Teacher and student bundles must have the same row count")
        if teacher_margin.shape != student_margin.shape:
            raise ValueError("Teacher and student top1_margin arrays must align")

        top_limit = min(10, teacher_topk.shape[1], student_topk.shape[1])
        if top_limit <= 0:
            raise ValueError("Bundles must contain at least one neighbor for round2 gating")

        overlap = np.zeros(teacher_topk.shape[0], dtype=np.float32)
        for row_index in range(teacher_topk.shape[0]):
            teacher_set = set(teacher_topk[row_index, :top_limit].tolist())
            student_set = set(student_topk[row_index, :top_limit].tolist())
            overlap[row_index] = len(teacher_set & student_set) / float(top_limit)

        margin_gain = student_margin - teacher_margin
        teacher_student_top10_overlap = float(np.mean(overlap))
        median_top1_top2_margin_gain = float(np.median(margin_gain))
        pass_round2 = bool(
            (teacher_student_top10_overlap < self.config.round2_overlap_gate)
            or (median_top1_top2_margin_gain >= self.config.round2_margin_gain_gate)
        )
        return {
            "teacher_student_top10_overlap": teacher_student_top10_overlap,
            "median_top1_top2_margin_gain": median_top1_top2_margin_gain,
            "pass_round2": pass_round2,
        }

    def append_results_json(self, payload: dict[str, object]) -> None:
        results_path = self.resolved_paths["results_json"]
        results_path.parent.mkdir(parents=True, exist_ok=True)
        if results_path.exists():
            current = json.loads(results_path.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                current = {}
        else:
            current = {}

        conditions = current.get("conditions")
        if not isinstance(conditions, dict):
            conditions = {}
        sanitized_payload = _jsonify(payload, self.repo_root)
        condition_name = str(sanitized_payload["condition"])
        conditions[condition_name] = sanitized_payload
        current["conditions"] = conditions
        results_path.write_text(
            json.dumps(current, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
