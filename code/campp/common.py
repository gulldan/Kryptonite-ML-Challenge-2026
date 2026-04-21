from __future__ import annotations

import csv
import io
import json
import random
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import soundfile as sf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THREED_SPEAKER_COMMIT = "065629c313eaf1a01c65c640c46d77e61e9607b4"
DEFAULT_MODEL_ID = "iic/speech_campplus_sv_en_voxceleb_16k"
DEFAULT_MODEL_REVISION = "v1.0.2"
DEFAULT_MODEL_FILE = "campplus_voxceleb.bin"
CANONICAL_PRETRAINED_ROOT = "data/pretrained_models/speaker_verification/campplus"
LEGACY_PRETRAINED_ROOT = "data/campp_assets/pretrained"
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
        "experiment_root": "data/campp_runs/campp_en_ft",
        "external_root": "data/campp_assets/external",
        "pretrained_root": CANONICAL_PRETRAINED_ROOT,
        "mlflow_tracking_uri": "",
        "mlflow_experiment": "kriptio_tembr_campp_en_ft",
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
    pretrained.setdefault("weight_filename", DEFAULT_MODEL_FILE)
    pretrained.setdefault("3dspeaker_commit", THREED_SPEAKER_COMMIT)
    default_weight_path = paths["pretrained_root"] / pretrained["weight_filename"]
    pretrained["weight_path"] = (
        resolve_path(project_root, pretrained.get("weight_path")) or default_weight_path
    )

    model = config.setdefault("model", {})
    model.setdefault("sample_rate", 16000)
    model.setdefault("n_mels", 80)
    model.setdefault("embedding_size", 512)

    data_prep = config.setdefault("data_prep", {})
    data_prep.setdefault("seed", 42)
    if "validation_speaker_fraction" not in data_prep:
        data_prep["validation_speaker_fraction"] = float(data_prep.get("val_speaker_fraction", 0.1))
    if "test_speaker_fraction" not in data_prep:
        data_prep["test_speaker_fraction"] = 0.1
    if "min_eval_utterances" not in data_prep:
        data_prep["min_eval_utterances"] = int(data_prep.get("min_validation_utterances", 11))
    data_prep.setdefault("write_absolute_paths", False)
    data_prep.setdefault("speaker_count_bins", [11, 21, 51, 1000000])

    training = config.setdefault("training", {})
    training.setdefault("train_chunk_sec", 3.0)
    training.setdefault("eval_chunk_sec", 6.0)
    training.setdefault("short_clip_pad_mode", "repeat")
    training.setdefault("batch_size", 32)
    training.setdefault("num_workers", 8)
    training.setdefault("prefetch_factor", 4)
    training.setdefault("persistent_workers", True)
    training.setdefault("epochs", 5)
    training.setdefault("backbone_lr", 1e-4)
    training.setdefault("classifier_lr", 1e-2)
    training.setdefault("freeze_backbone_epochs", 0)
    training.setdefault("weight_decay", 1e-4)
    training.setdefault("momentum", 0.9)
    training.setdefault("nesterov", True)
    training.setdefault("margin", 0.2)
    training.setdefault("scale", 32.0)
    training.setdefault("warmup_epochs", 1)
    training.setdefault("mixed_precision", True)
    training.setdefault("eval_every_epochs", 1)
    training.setdefault("save_every_epochs", 1)
    training.setdefault("max_grad_norm", 5.0)
    training.setdefault("benchmark_steps", 0)

    evaluation = config.setdefault("evaluation", {})
    evaluation.setdefault("ks", [1, 5, 10])
    evaluation.setdefault("compare_modes", ["single_crop", "segment_mean"])
    evaluation.setdefault("primary_mode", "segment_mean")
    evaluation.setdefault("segment_count", 3)
    evaluation.setdefault("long_file_threshold_sec", 6.0)
    evaluation.setdefault("retrieval_chunk_size", 1024)

    augmentations = config.setdefault("augmentations", {})
    augmentations.setdefault("enabled", False)
    augmentations.setdefault("max_augments_per_sample", 1)
    noise_cfg = augmentations.setdefault("noise", {})
    noise_cfg.setdefault("probability", 0.0)
    noise_cfg.setdefault("snr_db_min", 10.0)
    noise_cfg.setdefault("snr_db_max", 20.0)
    reverb_cfg = augmentations.setdefault("reverb", {})
    reverb_cfg.setdefault("probability", 0.0)
    reverb_cfg.setdefault("impulse_ms_min", 30.0)
    reverb_cfg.setdefault("impulse_ms_max", 120.0)
    reverb_cfg.setdefault("decay_power_min", 2.0)
    reverb_cfg.setdefault("decay_power_max", 6.0)
    reverb_cfg.setdefault("dry_wet_min", 0.15)
    reverb_cfg.setdefault("dry_wet_max", 0.35)
    band_cfg = augmentations.setdefault("band_limit", {})
    band_cfg.setdefault("probability", 0.0)
    band_cfg.setdefault("lowpass_hz_min", 2200.0)
    band_cfg.setdefault("lowpass_hz_max", 4200.0)
    band_cfg.setdefault("highpass_hz_min", 80.0)
    band_cfg.setdefault("highpass_hz_max", 300.0)
    silence_cfg = augmentations.setdefault("silence_shift", {})
    silence_cfg.setdefault("probability", 0.0)
    silence_cfg.setdefault("max_shift_sec", 0.35)

    mlflow_cfg = config.setdefault("mlflow", {})
    mlflow_cfg.setdefault("enabled", True)
    mlflow_cfg.setdefault("run_name", None)
    mlflow_cfg.setdefault("tracking_uri", paths["mlflow_tracking_uri"])
    mlflow_cfg.setdefault("experiment", paths["mlflow_experiment"])
    mlflow_cfg.setdefault(
        "tags",
        {
            "project": "kriptio_tembr",
            "model": "campplus_en_voxceleb",
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


def mlruns_root(config: dict[str, Any]) -> Path:
    return config["paths"]["experiment_root"] / "mlruns"


def submissions_root(config: dict[str, Any]) -> Path:
    return ensure_dir(config["paths"]["experiment_root"] / "submissions")


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
    return str(config["paths"].get("mlflow_experiment") or "kriptio_tembr_campp_en_ft")


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_mlflow_key(name: str) -> str:
    return name.replace("@", "_at_")


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


def compute_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def speaker_bin(count: int, bins: list[int]) -> str:
    if count < bins[0]:
        return f"<{bins[0]}"
    for left, right in zip(bins[:-1], bins[1:], strict=False):
        if left <= count < right:
            high = right - 1
            return f"{left}-{high}"
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
    overlap = validation_set & test_set
    if overlap:
        raise RuntimeError(f"Validation and test speaker overlap detected: {len(overlap)}")

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
    try:
        import mlflow
    except ImportError:
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


def ensure_3dspeaker_repo(config: dict[str, Any]) -> Path:
    external_root = ensure_dir(config["paths"]["external_root"])
    commit = config["pretrained"]["3dspeaker_commit"]
    repo_root = external_root / f"3D-Speaker-{commit}"
    if repo_root.exists():
        return repo_root

    archive_url = f"https://github.com/modelscope/3D-Speaker/archive/{commit}.zip"
    response = requests.get(archive_url, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(external_root)
    if not repo_root.exists():
        raise FileNotFoundError(f"Expected extracted 3D-Speaker repo at {repo_root}")
    return repo_root


def add_3dspeaker_to_syspath(config: dict[str, Any]) -> Path:
    repo_root = ensure_3dspeaker_repo(config)
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root


def ensure_pretrained_weight(config: dict[str, Any]) -> Path:
    weight_filename = str(config["pretrained"]["weight_filename"])
    weight_path = Path(config["pretrained"]["weight_path"])
    candidate_paths = [
        weight_path,
        (config["project_root"] / LEGACY_PRETRAINED_ROOT / weight_filename).resolve(),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise FileNotFoundError(
            "Pretrained CAM++ weights not found locally and modelscope is not installed. "
            f"Expected weight file: {weight_path}"
        ) from exc

    ensure_dir(weight_path.parent)
    cache_dir = snapshot_download(
        config["pretrained"]["model_id"],
        revision=config["pretrained"]["revision"],
    )
    source = Path(cache_dir) / config["pretrained"]["weight_filename"]
    if not source.exists():
        raise FileNotFoundError(f"Downloaded snapshot is missing {source.name}: {source}")
    weight_path.write_bytes(source.read_bytes())
    return weight_path


def build_campp_embedding_model(config: dict[str, Any]) -> Any:
    add_3dspeaker_to_syspath(config)
    from speakerlab.models.campplus.DTDNN import CAMPPlus

    return CAMPPlus(
        feat_dim=int(config["model"]["n_mels"]),
        embedding_size=int(config["model"]["embedding_size"]),
    )


def build_campp_classifier(config: dict[str, Any], num_classes: int) -> Any:
    add_3dspeaker_to_syspath(config)
    from speakerlab.models.campplus.classifier import CosineClassifier

    return CosineClassifier(
        input_dim=int(config["model"]["embedding_size"]),
        out_neurons=num_classes,
    )


def build_arc_margin_loss(config: dict[str, Any]) -> Any:
    add_3dspeaker_to_syspath(config)
    from speakerlab.loss.margin_loss import ArcMarginLoss

    return ArcMarginLoss(
        scale=float(config["training"]["scale"]),
        margin=float(config["training"]["margin"]),
        easy_margin=False,
    )


def load_pretrained_embedding(config: dict[str, Any], model: Any) -> Path:
    import torch

    weight_path = ensure_pretrained_weight(config)
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return weight_path


def save_checkpoint(
    path: Path,
    embedding_model: Any,
    classifier: Any,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    global_step: int | None = None,
    best_metrics: dict[str, float] | None = None,
    best_epoch: int | None = None,
) -> None:
    import torch

    payload = {
        "embedding_model": embedding_model.state_dict(),
        "classifier": classifier.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "pretrained_model_id": config["pretrained"]["model_id"],
        "git_sha": get_git_sha(config["project_root"]),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if global_step is not None:
        payload["global_step"] = int(global_step)
    if best_metrics is not None:
        payload["best_metrics"] = best_metrics
    if best_epoch is not None:
        payload["best_epoch"] = best_epoch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_embedding_checkpoint(path: Path, model: Any) -> dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu")
    if "embedding_model" in state:
        model.load_state_dict(state["embedding_model"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return state
