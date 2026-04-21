from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

try:
    from experiment_config import ExperimentConfig
except ImportError:  # pragma: no cover - package-style import fallback
    from .experiment_config import ExperimentConfig

if TYPE_CHECKING:
    try:
        from methods import CamppEmbeddingModel
    except ImportError:  # pragma: no cover - package-style import fallback
        from .methods import CamppEmbeddingModel


_PSEUDO_PREFIXES = (
    "adaptive_",
    "mutual_",
    "strict_",
    "nohubs_",
    "pseudo_",
)
_DURATION_BUCKETS = ("short_lt_3s", "medium_3_to_6s", "long_gt_6s")
_PROFILE_BUCKETS = ("near_v2_profile", "far_v2_profile")


def _required_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
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


def _weighted_prior_distance(
    features: np.ndarray,
    mu: np.ndarray,
    scale: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    centered = (features - mu[None, :]) / scale[None, :]
    return np.sum(weight[None, :] * np.square(centered), axis=1, dtype=np.float32)


def _stable_frame_hash(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True)
    digest = hashlib.sha256()
    digest.update(hashed.to_numpy(dtype=np.uint64, copy=False).tobytes())
    digest.update("|".join(df.columns).encode("utf-8"))
    digest.update(str(df.shape).encode("utf-8"))
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value type: {type(value)!r}")


def _as_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _is_pseudo_speaker(speaker_id: str) -> bool:
    return speaker_id.startswith(_PSEUDO_PREFIXES)


def _copy_prior_payload(prior: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in prior.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.copy()
        elif isinstance(value, dict):
            payload[key] = dict(value)
        else:
            payload[key] = value
    return payload


@dataclass(slots=True)
class ManifestRepository:
    config: ExperimentConfig
    acoustic_columns: list[str] = field(
        default_factory=lambda: [
            "split",
            "speaker_id",
            "filepath",
            "duration_sec",
            "rms_dbfs",
            "non_silent_ratio",
            "leading_silence_sec",
            "trailing_silence_sec",
            "spectral_bandwidth_hz",
            "band_energy_ratio_3_8k",
        ]
    )
    manifest_columns: list[str] = field(
        default_factory=lambda: [
            "ID",
            "dur",
            "path",
            "start",
            "stop",
            "spk",
            "orig_filepath",
        ]
    )
    resolved_paths: dict[str, Path] = field(init=False, repr=False)
    _acoustic_table_cache: pd.DataFrame | None = field(init=False, default=None, repr=False)
    _sample_v2_prior_cache: dict[str, object] | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.resolved_paths = self.config.paths.resolve_all()

    def load_manifest(self, kind: str) -> pd.DataFrame:
        key_map = {
            "full_train_manifest": "full_train_manifest",
            "full_train": "full_train_manifest",
            "sample_v1_manifest": "sample_v1_manifest",
            "sample_v1": "sample_v1_manifest",
            "sample_v2_manifest": "sample_v2_manifest",
            "sample_v2": "sample_v2_manifest",
            "val_unlabeled_manifest": "val_unlabeled_manifest",
            "val_unlabeled": "val_unlabeled_manifest",
            "heldout_test_manifest": "heldout_test_manifest",
            "heldout_test": "heldout_test_manifest",
        }
        if kind not in key_map:
            valid = ", ".join(sorted(key_map))
            raise KeyError(f"Unknown manifest kind={kind!r}. Expected one of: {valid}")
        manifest_path = self.resolved_paths[key_map[kind]]
        manifest_df = pd.read_csv(manifest_path)
        _required_columns(manifest_df, self.manifest_columns, f"manifest {kind}")
        manifest_df = manifest_df.loc[:, self.manifest_columns].copy()
        manifest_df["ID"] = manifest_df["ID"].astype(str)
        manifest_df["path"] = manifest_df["path"].astype(str)
        manifest_df["spk"] = manifest_df["spk"].astype(str)
        manifest_df["orig_filepath"] = manifest_df["orig_filepath"].astype(str)
        manifest_df["dur"] = manifest_df["dur"].astype(np.float32)
        manifest_df["start"] = pd.to_numeric(manifest_df["start"], errors="raise")
        manifest_df["stop"] = pd.to_numeric(manifest_df["stop"], errors="coerce")
        return manifest_df.reset_index(drop=True)

    def load_acoustic_table(self) -> pd.DataFrame:
        if self._acoustic_table_cache is not None:
            return self._acoustic_table_cache.copy()

        parquet_columns = list(self.acoustic_columns) + ["error"]
        acoustic_df = pd.read_parquet(
            self.resolved_paths["acoustic_parquet"], columns=parquet_columns
        )
        _required_columns(acoustic_df, parquet_columns, "acoustic parquet")
        acoustic_df = acoustic_df.copy()
        acoustic_df["_error_rank"] = acoustic_df["error"].notna().astype(np.int64)
        acoustic_df = acoustic_df.sort_values(
            by=["filepath", "_error_rank"],
            kind="mergesort",
        )
        acoustic_df = acoustic_df.drop_duplicates(subset=["filepath"], keep="first")
        acoustic_df = acoustic_df.drop(columns=["_error_rank", "error"])
        acoustic_df = acoustic_df.rename(
            columns={
                "filepath": "path",
                "duration_sec": "dur_acoustic",
            }
        )
        acoustic_df["path"] = acoustic_df["path"].astype(str)
        acoustic_df["speaker_id"] = acoustic_df["speaker_id"].astype(str)
        self._acoustic_table_cache = acoustic_df.reset_index(drop=True)
        return self._acoustic_table_cache.copy()

    def merge_manifest_with_acoustics(
        self,
        manifest_df: pd.DataFrame,
        split_name: str,
    ) -> pd.DataFrame:
        _required_columns(manifest_df, self.manifest_columns, f"manifest merge {split_name}")
        acoustic_df = self.load_acoustic_table()
        merged = manifest_df.copy().merge(
            acoustic_df,
            on="path",
            how="left",
            sort=False,
            validate="one_to_one",
        )
        merged["dur"] = merged["dur_acoustic"].fillna(merged["dur"]).astype(np.float32)
        missing_mask = merged[self.config.v2_features].isna().any(axis=1)
        missing_count = int(missing_mask.sum())
        if missing_count > 0:
            missing_examples = merged.loc[missing_mask, "path"].head(5).tolist()
            raise ValueError(
                f"Acoustic feature merge left {missing_count} rows incomplete for {split_name}; "
                f"examples={missing_examples}"
            )
        merged["manifest_split"] = str(split_name)
        merged["oracle_spk"] = merged["spk"].astype(str)
        return merged.reset_index(drop=True)

    def build_sample_v2_prior(self) -> dict[str, object]:
        if self._sample_v2_prior_cache is not None:
            return _copy_prior_payload(self._sample_v2_prior_cache)

        full_train_df = self.merge_manifest_with_acoustics(
            self.load_manifest("full_train"),
            split_name="full_train",
        )
        sample_v2_df = self.merge_manifest_with_acoustics(
            self.load_manifest("sample_v2"),
            split_name="sample_v2",
        )
        val_unlabeled_df = self.merge_manifest_with_acoustics(
            self.load_manifest("val_unlabeled"),
            split_name="val_unlabeled",
        )

        feature_names = list(self.config.v2_features)
        weight = np.asarray(
            [self.config.v2_weights[feature] for feature in feature_names],
            dtype=np.float32,
        )
        x_sample_v2 = sample_v2_df.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True)
        x_reference = pd.concat(
            [
                full_train_df.loc[:, feature_names],
                val_unlabeled_df.loc[:, feature_names],
            ],
            axis=0,
            ignore_index=True,
        ).to_numpy(dtype=np.float32, copy=True)
        if x_sample_v2.size == 0:
            raise ValueError("sample_v2 manifest is empty; cannot build acoustic prior")
        if x_reference.size == 0:
            raise ValueError("reference pool is empty; cannot build acoustic prior")

        mu = np.median(x_sample_v2, axis=0).astype(np.float32, copy=False)
        scale = np.std(x_reference, axis=0, ddof=0).astype(np.float32, copy=False)
        scale = np.clip(scale, 1e-6, None)

        sample_v2_distances = _weighted_prior_distance(
            features=x_sample_v2,
            mu=mu,
            scale=scale,
            weight=weight,
        )
        prior_distance_threshold = float(
            np.quantile(sample_v2_distances, self.config.prior_distance_quantile)
        )

        diversity_scores: list[float] = []
        for _, speaker_df in sample_v2_df.groupby("spk", sort=False):
            if len(speaker_df) < self.config.min_component_size:
                continue
            x_speaker = speaker_df.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True)
            feature_std = np.std(x_speaker, axis=0, ddof=0).astype(np.float32, copy=False)
            diversity_score = float(np.sum(weight * (feature_std / scale), dtype=np.float32))
            diversity_scores.append(diversity_score)
        if not diversity_scores:
            raise ValueError(
                "No sample_v2 speakers met min_component_size; cannot estimate diversity floor"
            )
        diversity_floor = float(
            np.quantile(
                np.asarray(diversity_scores, dtype=np.float32),
                self.config.diversity_floor_quantile,
            )
        )

        prior = {
            "feature_names": feature_names,
            "mu": mu.copy(),
            "scale": scale.copy(),
            "weight": weight.copy(),
            "prior_distance_threshold": prior_distance_threshold,
            "diversity_floor": diversity_floor,
            "sample_v2_rows": int(len(sample_v2_df)),
            "reference_rows": int(len(x_reference)),
            "diversity_speaker_count": int(len(diversity_scores)),
        }
        self._sample_v2_prior_cache = _copy_prior_payload(prior)
        return _copy_prior_payload(prior)

    def annotate_regime_buckets(
        self,
        df: pd.DataFrame,
        prior: dict[str, object],
    ) -> pd.DataFrame:
        feature_names = list(self.config.v2_features)
        _required_columns(df, feature_names + ["dur"], "regime annotation")
        mu = np.asarray(prior["mu"], dtype=np.float32)
        scale = np.asarray(prior["scale"], dtype=np.float32)
        weight = np.asarray(prior["weight"], dtype=np.float32)

        annotated = df.copy()
        x = annotated.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True)
        prior_distance = _weighted_prior_distance(x, mu=mu, scale=scale, weight=weight)
        duration = annotated["dur"].to_numpy(dtype=np.float32, copy=False)

        duration_bucket = np.where(
            duration < 3.0,
            _DURATION_BUCKETS[0],
            np.where(duration <= 6.0, _DURATION_BUCKETS[1], _DURATION_BUCKETS[2]),
        )
        profile_bucket = np.where(
            prior_distance <= float(prior["prior_distance_threshold"]),
            _PROFILE_BUCKETS[0],
            _PROFILE_BUCKETS[1],
        )
        annotated["prior_distance"] = prior_distance.astype(np.float32, copy=False)
        annotated["duration_bucket"] = pd.Categorical(
            duration_bucket,
            categories=list(_DURATION_BUCKETS),
            ordered=True,
        )
        annotated["profile_bucket"] = pd.Categorical(
            profile_bucket,
            categories=list(_PROFILE_BUCKETS),
            ordered=True,
        )
        return annotated.reset_index(drop=True)

    def materialize_prepared_directory(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
    ) -> dict[str, Path]:
        prepared_dir = output_dir.resolve() / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        repo_root = self.resolved_paths["repo_root"]

        train_out = self._prepare_manifest_output(train_df, name="train")
        eval_out = self._prepare_manifest_output(eval_df, name="validation")
        test_out = self._prepare_manifest_output(test_df, name="test")

        pseudo_mask = self._pseudo_mask(train_df)
        supervised_speakers = sorted(
            train_df.loc[~pseudo_mask, "spk"].astype(str).unique().tolist()
        )
        pseudo_speakers = sorted(train_df.loc[pseudo_mask, "spk"].astype(str).unique().tolist())
        speaker_order = supervised_speakers + pseudo_speakers
        speaker_to_index = {speaker_id: index for index, speaker_id in enumerate(speaker_order)}

        train_manifest_path = prepared_dir / "train_manifest.csv"
        val_manifest_path = prepared_dir / "val_manifest.csv"
        test_manifest_path = prepared_dir / "test_manifest.csv"
        split_summary_path = prepared_dir / "split_summary.json"
        speaker_index_path = prepared_dir / "speaker_to_index.json"

        train_out.to_csv(train_manifest_path, index=False)
        eval_out.to_csv(val_manifest_path, index=False)
        test_out.to_csv(test_manifest_path, index=False)
        with speaker_index_path.open("w", encoding="utf-8") as handle:
            json.dump(speaker_to_index, handle, indent=2, ensure_ascii=False, sort_keys=True)

        split_summary = {
            "train_rows": int(len(train_out)),
            "validation_rows": int(len(eval_out)),
            "test_rows": int(len(test_out)),
            "train_unique_speakers": int(train_out["spk"].nunique()),
            "validation_unique_speakers": int(eval_out["spk"].nunique()),
            "test_unique_speakers": int(test_out["spk"].nunique()),
            "pseudo_rows": int(pseudo_mask.sum()),
            "pseudo_speakers": int(len(pseudo_speakers)),
            "speaker_to_index_size": int(len(speaker_to_index)),
            "paths": {
                "train_manifest": _as_repo_relative(train_manifest_path, repo_root),
                "validation_manifest": _as_repo_relative(val_manifest_path, repo_root),
                "test_manifest": _as_repo_relative(test_manifest_path, repo_root),
                "speaker_to_index": _as_repo_relative(speaker_index_path, repo_root),
            },
        }
        with split_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                split_summary,
                handle,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )

        return {
            "prepared_dir": prepared_dir,
            "train_manifest": train_manifest_path,
            "validation_manifest": val_manifest_path,
            "test_manifest": test_manifest_path,
            "split_summary": split_summary_path,
            "speaker_to_index": speaker_index_path,
        }

    def _prepare_manifest_output(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        _required_columns(df, self.manifest_columns, f"prepared {name}")
        manifest = df.loc[:, self.manifest_columns].copy()
        manifest["ID"] = manifest["ID"].astype(str)
        manifest["path"] = manifest["path"].astype(str)
        manifest["spk"] = manifest["spk"].astype(str)
        manifest["orig_filepath"] = manifest["orig_filepath"].astype(str)
        manifest["dur"] = manifest["dur"].astype(np.float32)
        manifest["start"] = pd.to_numeric(manifest["start"], errors="raise")
        manifest["stop"] = pd.to_numeric(manifest["stop"], errors="coerce")

        absolute_path_mask = manifest["path"].map(lambda value: Path(value).is_absolute())
        absolute_orig_mask = manifest["orig_filepath"].map(lambda value: Path(value).is_absolute())
        if bool(absolute_path_mask.any()) or bool(absolute_orig_mask.any()):
            raise ValueError(f"Prepared {name} manifest must not contain absolute audio paths")
        return manifest.reset_index(drop=True)

    def _pseudo_mask(self, train_df: pd.DataFrame) -> pd.Series:
        spk_series = train_df["spk"].astype(str)
        if "oracle_spk" in train_df.columns:
            oracle_series = train_df["oracle_spk"].astype(str)
            pseudo_mask = spk_series != oracle_series
        else:
            pseudo_mask = spk_series.map(_is_pseudo_speaker)
        return pseudo_mask.astype(bool)


@dataclass(slots=True)
class EmbeddingCacheDataset:
    config: ExperimentConfig
    cache_dir: Path = field(init=False)
    resolved_paths: dict[str, Path] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.resolved_paths = self.config.paths.resolve_all()
        self.cache_dir = self.resolved_paths["cache_dir"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_or_load_validation_bundle(
        self,
        model: CamppEmbeddingModel,
        repo: ManifestRepository,
        device: torch.device,
    ) -> dict[str, object]:
        prior = repo.build_sample_v2_prior()
        val_df = repo.load_manifest("val_unlabeled")
        merged_df = repo.merge_manifest_with_acoustics(val_df, split_name="val_unlabeled")
        annotated_df = repo.annotate_regime_buckets(merged_df, prior=prior)
        selector_meta_df = annotated_df.drop(columns=["oracle_spk"]).reset_index(drop=True)
        oracle_labels = annotated_df["oracle_spk"].astype(str).to_numpy(copy=True)

        fingerprint = self._build_validation_fingerprint(selector_meta_df)
        cache_npz_path = self.cache_dir / "validation_bundle.npz"
        cache_fingerprint_path = self.cache_dir / "validation_bundle_fingerprint.json"
        effective_k = min(self.config.knn_k, max(len(selector_meta_df) - 1, 1))

        if cache_npz_path.exists() and cache_fingerprint_path.exists():
            cached_fingerprint = json.loads(cache_fingerprint_path.read_text(encoding="utf-8"))
            if cached_fingerprint == fingerprint:
                with np.load(cache_npz_path, allow_pickle=False) as payload:
                    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
                    topk_idx = np.asarray(payload["topk_idx"], dtype=np.int64)
                    topk_sim = np.asarray(payload["topk_sim"], dtype=np.float32)
                    top1_margin = np.asarray(payload["top1_margin"], dtype=np.float32)
                self._validate_cached_shapes(
                    selector_meta_df=selector_meta_df,
                    embeddings=embeddings,
                    topk_idx=topk_idx,
                    topk_sim=topk_sim,
                    top1_margin=top1_margin,
                    expected_k=effective_k,
                )
                return {
                    "selector_meta_df": selector_meta_df,
                    "oracle_labels": oracle_labels,
                    "embeddings": embeddings,
                    "topk_idx": topk_idx,
                    "topk_sim": topk_sim,
                    "top1_margin": top1_margin,
                    "knn_k": int(topk_idx.shape[1]),
                    "cache_path": cache_npz_path,
                    "fingerprint_path": cache_fingerprint_path,
                }

        embeddings = np.asarray(
            model.encode_manifest_batches(selector_meta_df, device),
            dtype=np.float32,
        )
        embeddings = _normalize_embeddings(embeddings)
        if not np.isfinite(embeddings).all():
            raise ValueError("Validation embeddings contain non-finite values")
        topk_idx, topk_sim = self.compute_topk_cosine(
            embeddings=embeddings,
            k=effective_k,
            chunk_size=1024,
        )
        if topk_sim.shape[1] >= 2:
            top1_margin = (topk_sim[:, 0] - topk_sim[:, 1]).astype(np.float32, copy=False)
        else:
            top1_margin = np.zeros(topk_sim.shape[0], dtype=np.float32)

        np.savez_compressed(
            cache_npz_path,
            embeddings=embeddings,
            topk_idx=topk_idx,
            topk_sim=topk_sim,
            top1_margin=top1_margin,
        )
        cache_fingerprint_path.write_text(
            json.dumps(fingerprint, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        return {
            "selector_meta_df": selector_meta_df,
            "oracle_labels": oracle_labels,
            "embeddings": embeddings,
            "topk_idx": topk_idx,
            "topk_sim": topk_sim,
            "top1_margin": top1_margin,
            "knn_k": int(topk_idx.shape[1]),
            "cache_path": cache_npz_path,
            "fingerprint_path": cache_fingerprint_path,
        }

    def compute_topk_cosine(
        self,
        embeddings: np.ndarray,
        k: int,
        chunk_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        emb = _normalize_embeddings(embeddings)
        if emb.shape[0] < 2:
            raise ValueError("Need at least two rows to compute nearest neighbours")
        if k <= 0:
            raise ValueError("k must be positive")
        if k >= emb.shape[0]:
            raise ValueError(f"k={k} must be smaller than number of rows={emb.shape[0]}")

        num_rows = emb.shape[0]
        topk_indices: list[np.ndarray] = []
        topk_scores: list[np.ndarray] = []
        for start in range(0, num_rows, chunk_size):
            stop = min(num_rows, start + chunk_size)
            sim_chunk = emb[start:stop] @ emb.T
            local_rows = np.arange(stop - start, dtype=np.int64)
            global_rows = np.arange(start, stop, dtype=np.int64)
            sim_chunk[local_rows, global_rows] = -np.inf

            kth = sim_chunk.shape[1] - k
            partition_idx = np.argpartition(sim_chunk, kth=kth, axis=1)[:, -k:]
            partition_scores = np.take_along_axis(sim_chunk, partition_idx, axis=1)
            order = np.argsort(-partition_scores, axis=1)
            sorted_idx = np.take_along_axis(partition_idx, order, axis=1)
            sorted_scores = np.take_along_axis(partition_scores, order, axis=1)
            topk_indices.append(sorted_idx.astype(np.int64, copy=False))
            topk_scores.append(sorted_scores.astype(np.float32, copy=False))

        return np.concatenate(topk_indices, axis=0), np.concatenate(topk_scores, axis=0)

    def build_directed_neighbor_frame(self, bundle: dict[str, object]) -> pd.DataFrame:
        topk_idx = np.asarray(bundle["topk_idx"], dtype=np.int64)
        topk_sim = np.asarray(bundle["topk_sim"], dtype=np.float32)
        top1_margin = np.asarray(bundle["top1_margin"], dtype=np.float32)
        if topk_idx.shape != topk_sim.shape:
            raise ValueError("topk_idx and topk_sim must have the same shape")
        num_rows, k = topk_idx.shape
        src_index = np.repeat(np.arange(num_rows, dtype=np.int64), k)
        rank = np.tile(np.arange(k, dtype=np.int64), num_rows)
        dst_index = topk_idx.reshape(-1)
        cosine = topk_sim.reshape(-1)
        src_margin = np.repeat(top1_margin, k)
        return pd.DataFrame(
            {
                "src_index": src_index,
                "rank": rank,
                "dst_index": dst_index,
                "cosine": cosine,
                "src_margin": src_margin,
            }
        )

    def _build_validation_fingerprint(
        self,
        selector_meta_df: pd.DataFrame,
    ) -> dict[str, object]:
        repo_root = self.resolved_paths["repo_root"]
        fingerprint_columns = [
            "ID",
            "dur",
            "path",
            "spk",
            "rms_dbfs",
            "non_silent_ratio",
            "leading_silence_sec",
            "trailing_silence_sec",
            "spectral_bandwidth_hz",
            "band_energy_ratio_3_8k",
            "duration_bucket",
            "profile_bucket",
            "prior_distance",
        ]
        _required_columns(selector_meta_df, fingerprint_columns, "validation fingerprint")
        stable_df = selector_meta_df.loc[:, fingerprint_columns].copy()
        stable_df["duration_bucket"] = stable_df["duration_bucket"].astype(str)
        stable_df["profile_bucket"] = stable_df["profile_bucket"].astype(str)
        return {
            "frame_hash": _stable_frame_hash(stable_df),
            "row_count": int(len(selector_meta_df)),
            "knn_k": int(self.config.knn_k),
            "official_mode": self.config.official_mode,
            "official_segment_count": int(self.config.official_segment_count),
            "prior_distance_quantile": float(self.config.prior_distance_quantile),
            "diversity_floor_quantile": float(self.config.diversity_floor_quantile),
            "cache_dir": _as_repo_relative(self.cache_dir, repo_root),
            "source_paths": {
                "val_unlabeled_manifest": _as_repo_relative(
                    self.resolved_paths["val_unlabeled_manifest"],
                    repo_root,
                ),
                "acoustic_parquet": _as_repo_relative(
                    self.resolved_paths["acoustic_parquet"],
                    repo_root,
                ),
            },
        }

    def _validate_cached_shapes(
        self,
        selector_meta_df: pd.DataFrame,
        embeddings: np.ndarray,
        topk_idx: np.ndarray,
        topk_sim: np.ndarray,
        top1_margin: np.ndarray,
        expected_k: int,
    ) -> None:
        num_rows = len(selector_meta_df)
        if embeddings.shape[0] != num_rows:
            raise ValueError("Cached embeddings row count does not match current manifest")
        if embeddings.ndim != 2:
            raise ValueError("Cached embeddings must be 2D")
        if topk_idx.shape != (num_rows, expected_k):
            raise ValueError("Cached topk_idx shape does not match expected validation shape")
        if topk_sim.shape != (num_rows, expected_k):
            raise ValueError("Cached topk_sim shape does not match expected validation shape")
        if top1_margin.shape != (num_rows,):
            raise ValueError("Cached top1_margin shape does not match expected validation shape")
        if not np.isfinite(embeddings).all():
            raise ValueError("Cached embeddings contain non-finite values")
        if not np.isfinite(topk_sim).all():
            raise ValueError("Cached topk similarities contain non-finite values")
