from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "microsoft/wavlm-base-plus-sv"
DEFAULT_MODEL_REVISION = "main"
MANIFEST_FILENAMES = {
    "train": "train_manifest.csv",
    "validation": "val_manifest.csv",
    "val": "val_manifest.csv",
    "test": "test_manifest.csv",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a mapping, got {type(payload)!r}")
    return payload


def merge_nested_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_nested_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_raw_config(config_path: Path) -> dict[str, Any]:
    payload = read_yaml(config_path)
    base_ref = payload.pop("_base_", None)
    if not base_ref:
        return payload

    refs = base_ref if isinstance(base_ref, list) else [base_ref]
    merged: dict[str, Any] = {}
    for ref in refs:
        base_path = (config_path.parent / ref).resolve()
        merged = merge_nested_config(merged, load_raw_config(base_path))
    return merge_nested_config(merged, payload)


def resolve_path(project_root: Path, value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [serialize_value(item) for item in value]
    return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_raw_config(config_path)
    project_root = resolve_path(config_path.parent, config.get("project_root")) or PROJECT_ROOT
    config["project_root"] = project_root

    paths = config.setdefault("paths", {})
    default_paths = {
        "data_root": "data/Для участников",
        "train_csv": "data/Для участников/train.csv",
        "test_csv": "data/Для участников/test_public.csv",
        "audio_header_cache": "data/eda/full/acoustic_all/cache/audio_header.parquet",
        "experiment_root": "data/wavlm_runs/wavlm_base_plus_sv",
        "pretrained_root": "data/pretrained_models/speaker_verification/wavlm_base_plus_sv",
        "mlflow_tracking_uri": "",
        "mlflow_experiment": "kriptio_tembr_wavlm_base_plus_sv",
        "train_manifest": None,
        "validation_manifest": None,
        "test_manifest": None,
    }
    for key, default in default_paths.items():
        value = paths.get(key, default)
        if key.startswith("mlflow_"):
            paths[key] = value or ""
        else:
            paths[key] = resolve_path(project_root, value)

    pretrained = config.setdefault("pretrained", {})
    pretrained.setdefault("model_id", DEFAULT_MODEL_ID)
    pretrained.setdefault("revision", DEFAULT_MODEL_REVISION)

    model = config.setdefault("model", {})
    model.setdefault("sample_rate", 16000)
    model.setdefault("variant", "wavlm_base_plus_sv")

    data_prep = config.setdefault("data_prep", {})
    data_prep.setdefault("seed", 42)
    data_prep.setdefault("validation_speaker_fraction", 0.1)
    data_prep.setdefault("test_speaker_fraction", 0.1)
    data_prep.setdefault("min_eval_utterances", 11)
    data_prep.setdefault("write_absolute_paths", False)
    data_prep.setdefault("speaker_count_bins", [11, 21, 51, 1000000])

    evaluation = config.setdefault("evaluation", {})
    evaluation.setdefault("ks", [1, 5, 10])
    evaluation.setdefault("compare_modes", ["xvector_full_file", "xvector_chunk_mean"])
    evaluation.setdefault("primary_mode", "xvector_full_file")
    evaluation.setdefault("batch_size", 8)
    evaluation.setdefault("chunk_sec", 10.0)
    evaluation.setdefault("max_load_len_sec", 90.0)
    evaluation.setdefault("retrieval_chunk_size", 1024)
    evaluation.setdefault("progress_every_rows", 1000)
    evaluation.setdefault("pooling", "xvector_head")

    mlflow_cfg = config.setdefault("mlflow", {})
    mlflow_cfg.setdefault("enabled", True)
    mlflow_cfg.setdefault("run_name", None)
    mlflow_cfg.setdefault("tracking_uri", paths["mlflow_tracking_uri"])
    mlflow_cfg.setdefault("experiment", paths["mlflow_experiment"])
    mlflow_cfg.setdefault(
        "tags",
        {
            "project": "kriptio_tembr",
            "model": "wavlm_base_plus_sv",
        },
    )
    return config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepared_root(config: dict[str, Any]) -> Path:
    return ensure_dir(config["paths"]["experiment_root"] / "prepared")


def runs_root(config: dict[str, Any]) -> Path:
    return ensure_dir(config["paths"]["experiment_root"] / "runs")


def submissions_root(config: dict[str, Any]) -> Path:
    return ensure_dir(config["paths"]["experiment_root"] / "submissions")


def mlruns_root(config: dict[str, Any]) -> Path:
    return config["paths"]["experiment_root"] / "mlruns"


def manifest_path_for_split(config: dict[str, Any], split: str) -> Path:
    normalized = split.strip().lower()
    if normalized not in MANIFEST_FILENAMES:
        raise KeyError(f"Unknown split: {split}")
    override_key = {
        "train": "train_manifest",
        "validation": "validation_manifest",
        "val": "validation_manifest",
        "test": "test_manifest",
    }[normalized]
    override_path = config["paths"].get(override_key)
    if override_path:
        return Path(override_path)
    return config["paths"]["experiment_root"] / "prepared" / MANIFEST_FILENAMES[normalized]


def resolve_mlflow_tracking_uri(config: dict[str, Any]) -> str:
    tracking_uri = str(config.get("mlflow", {}).get("tracking_uri") or "").strip()
    if tracking_uri:
        return tracking_uri
    legacy_uri = str(config["paths"].get("mlflow_tracking_uri") or "").strip()
    if legacy_uri:
        return legacy_uri
    return mlruns_root(config).resolve().as_uri()


def resolve_mlflow_experiment(config: dict[str, Any]) -> str:
    experiment = str(config.get("mlflow", {}).get("experiment") or "").strip()
    if experiment:
        return experiment
    return str(config["paths"].get("mlflow_experiment") or "kriptio_tembr_wavlm_base_plus_sv")


def write_resolved_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_value(config)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def get_git_sha(project_root: Path) -> str:
    head_path = project_root / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head.split(" ", 1)[1]
        ref_path = project_root / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()[:12]
    return head[:12]


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sanitize_mlflow_key(name: str) -> str:
    return name.replace("@", "_at_")


def compute_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def load_duration_lookup(config: dict[str, Any]) -> dict[str, float]:
    cache_path = config["paths"]["audio_header_cache"]
    if cache_path and cache_path.exists():
        frame = pd.read_parquet(cache_path, columns=["filepath", "duration_sec"])
        return dict(zip(frame["filepath"], frame["duration_sec"], strict=False))
    train_csv = pd.read_csv(config["paths"]["train_csv"])
    data_root = config["paths"]["data_root"]
    result: dict[str, float] = {}
    for filepath in train_csv["filepath"].tolist():
        result[filepath] = compute_duration_sec(data_root / filepath)
    return result


def speaker_bin(count: int, bins: list[int]) -> str:
    if count < bins[0]:
        return f"<{bins[0]}"
    for left, right in zip(bins[:-1], bins[1:], strict=False):
        if left <= count < right:
            return f"{left}-{right - 1}"
    return f"{bins[-1]}+"


def _split_counts_for_bucket(
    bucket_size: int, validation_fraction: float, test_fraction: float
) -> tuple[int, int]:
    if bucket_size <= 1:
        return 0, 0
    if bucket_size == 2:
        return 1, 1
    val_take = (
        max(1, int(round(bucket_size * validation_fraction))) if validation_fraction > 0 else 0
    )
    test_take = max(1, int(round(bucket_size * test_fraction))) if test_fraction > 0 else 0
    while val_take + test_take > bucket_size - 1:
        if val_take >= test_take and val_take > 1:
            val_take -= 1
        elif test_take > 1:
            test_take -= 1
        else:
            break
    return val_take, test_take


def stratified_speaker_train_val_test_split(
    train_df: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    min_eval_utterances: int,
    bins: list[int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    speaker_counts = train_df.groupby("speaker_id").size().rename("utterance_count").reset_index()
    speaker_counts["speaker_bin"] = speaker_counts["utterance_count"].map(
        lambda value: speaker_bin(int(value), bins)
    )
    speaker_counts["eligible_for_eval"] = speaker_counts["utterance_count"] >= min_eval_utterances

    eligible = speaker_counts[speaker_counts["eligible_for_eval"]].copy()
    rng = np.random.default_rng(seed)
    validation_speakers: list[str] = []
    test_speakers: list[str] = []

    for _, group in eligible.groupby("speaker_bin", sort=True):
        speakers = group["speaker_id"].tolist()
        rng.shuffle(speakers)
        val_take, test_take = _split_counts_for_bucket(
            len(speakers), validation_fraction, test_fraction
        )
        validation_speakers.extend(speakers[:val_take])
        test_speakers.extend(speakers[val_take : val_take + test_take])

    validation_set = set(validation_speakers)
    test_set = set(test_speakers)
    validation_df = train_df[train_df["speaker_id"].isin(validation_set)].copy()
    test_df = train_df[train_df["speaker_id"].isin(test_set)].copy()
    train_split_df = train_df[~train_df["speaker_id"].isin(validation_set | test_set)].copy()

    speaker_counts["split_role"] = "train"
    speaker_counts.loc[~speaker_counts["eligible_for_eval"], "split_role"] = "train_only_lt11"
    speaker_counts.loc[speaker_counts["speaker_id"].isin(validation_set), "split_role"] = (
        "validation"
    )
    speaker_counts.loc[speaker_counts["speaker_id"].isin(test_set), "split_role"] = "test"
    return train_split_df, validation_df, test_df, speaker_counts


def maybe_log_mlflow(
    config: dict[str, Any],
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path],
    tags: dict[str, str] | None = None,
) -> None:
    if not config.get("mlflow", {}).get("enabled", True):
        return
    local_mlruns = mlruns_root(config)
    if (
        not str(config.get("mlflow", {}).get("tracking_uri") or "").strip()
        and not str(config["paths"].get("mlflow_tracking_uri") or "").strip()
    ):
        ensure_dir(local_mlruns)
    mlflow.set_tracking_uri(resolve_mlflow_tracking_uri(config))
    mlflow.set_experiment(resolve_mlflow_experiment(config))
    all_tags = dict(config.get("mlflow", {}).get("tags", {}))
    if tags:
        all_tags.update(tags)
    with mlflow.start_run(run_name=run_name):
        if all_tags:
            mlflow.set_tags(all_tags)
        flat_params = {sanitize_mlflow_key(key): str(value) for key, value in params.items()}
        if flat_params:
            mlflow.log_params(flat_params)
        if metrics:
            mlflow.log_metrics({sanitize_mlflow_key(key): value for key, value in metrics.items()})
        for artifact in artifacts:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))


def load_pretrained_components(config: dict[str, Any], device: Any) -> tuple[Any, Any, str]:
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    pretrained_root = ensure_dir(config["paths"]["pretrained_root"])
    model_id = str(config["pretrained"]["model_id"])
    revision = str(config["pretrained"]["revision"])
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(pretrained_root),
    )
    model = WavLMForXVector.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(pretrained_root),
    ).to(device)
    return feature_extractor, model, f"{model_id}@{revision}"
