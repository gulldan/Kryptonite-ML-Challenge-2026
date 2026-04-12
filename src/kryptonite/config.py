"""Typed project configuration with override and dotenv support."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PathsConfig:
    project_root: str
    dataset_root: str
    artifacts_root: str
    cache_root: str
    manifests_root: str


@dataclass(slots=True)
class RuntimeConfig:
    seed: int
    device: str
    log_level: str
    num_workers: int


@dataclass(slots=True)
class ReproducibilityConfig:
    deterministic: bool
    pythonhashseed: int
    fingerprint_paths: list[str]


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    eval_batch_size: int
    max_epochs: int
    precision: str


@dataclass(slots=True)
class BackendsConfig:
    inference: str
    export: str
    allow_torch: bool
    allow_onnx: bool
    allow_tensorrt: bool

    def __post_init__(self) -> None:
        normalized_inference = self.inference.strip().lower()
        if normalized_inference not in {"auto", "torch", "onnx", "onnxruntime", "tensorrt"}:
            raise ValueError(
                "backends.inference must be one of "
                "['auto', 'onnx', 'onnxruntime', 'tensorrt', 'torch']."
            )
        normalized_export = self.export.strip().lower()
        if normalized_export not in {"onnx", "torch"}:
            raise ValueError("backends.export must be one of ['onnx', 'torch'].")


@dataclass(slots=True)
class ExportConfig:
    opset: int
    dynamic_axes: bool
    profile: str
    boundary: str = "encoder_only"
    input_name: str = "encoder_input"
    output_name: str = "embedding"


@dataclass(slots=True)
class TrackingConfig:
    enabled: bool
    backend: str
    experiment: str
    run_name_prefix: str
    output_root: str
    copy_artifacts: bool


@dataclass(slots=True)
class TelemetryConfig:
    enabled: bool = True
    structured_logs: bool = True
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"

    def __post_init__(self) -> None:
        if not self.metrics_path.startswith("/"):
            raise ValueError("metrics_path must start with '/'.")
        if self.metrics_path == "/":
            raise ValueError("metrics_path must not be '/'.")


@dataclass(slots=True)
class NormalizationConfig:
    target_sample_rate_hz: int
    target_channels: int
    output_format: str
    output_pcm_bits_per_sample: int
    peak_headroom_db: float
    dc_offset_threshold: float
    clipped_sample_threshold: float
    loudness_mode: str = "none"
    target_loudness_dbfs: float = -27.0
    max_loudness_gain_db: float = 20.0
    max_loudness_attenuation_db: float = 12.0


@dataclass(slots=True)
class VADConfig:
    mode: str
    backend: str = "silero_vad_v6_onnx"
    provider: str = "auto"
    min_output_duration_seconds: float | None = 1.0
    min_retained_ratio: float | None = 0.4


@dataclass(slots=True)
class SilenceAugmentationConfig:
    enabled: bool = False
    max_leading_padding_seconds: float = 0.0
    max_trailing_padding_seconds: float = 0.0
    max_inserted_pauses: int = 0
    min_inserted_pause_seconds: float = 0.08
    max_inserted_pause_seconds: float = 0.25
    pause_ratio_min: float = 1.0
    pause_ratio_max: float = 1.0
    min_detected_pause_seconds: float = 0.08
    max_perturbed_pause_seconds: float = 0.6
    analysis_frame_ms: float = 20.0
    silence_threshold_dbfs: float = -45.0

    def __post_init__(self) -> None:
        for name in (
            "max_leading_padding_seconds",
            "max_trailing_padding_seconds",
            "min_inserted_pause_seconds",
            "max_inserted_pause_seconds",
            "min_detected_pause_seconds",
            "max_perturbed_pause_seconds",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_inserted_pauses < 0:
            raise ValueError("max_inserted_pauses must be non-negative")
        if self.min_inserted_pause_seconds > self.max_inserted_pause_seconds:
            raise ValueError(
                "min_inserted_pause_seconds must be less than or equal to "
                "max_inserted_pause_seconds"
            )
        if self.pause_ratio_min <= 0.0 or self.pause_ratio_max <= 0.0:
            raise ValueError("pause_ratio_min and pause_ratio_max must be positive")
        if self.pause_ratio_min > self.pause_ratio_max:
            raise ValueError("pause_ratio_min must be less than or equal to pause_ratio_max")
        if self.max_perturbed_pause_seconds < self.min_detected_pause_seconds:
            raise ValueError(
                "max_perturbed_pause_seconds must be greater than or equal to "
                "min_detected_pause_seconds"
            )
        if self.analysis_frame_ms <= 0.0:
            raise ValueError("analysis_frame_ms must be positive")


@dataclass(slots=True)
class AugmentationFamilyWeightsConfig:
    noise: float = 1.0
    reverb: float = 1.0
    distance: float = 0.9
    codec: float = 0.8
    silence: float = 0.6
    speed: float = 0.0

    def __post_init__(self) -> None:
        for name in ("noise", "reverb", "distance", "codec", "silence"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} family weight must be positive")
        if self.speed < 0.0:
            raise ValueError("speed family weight must be non-negative")


@dataclass(slots=True)
class AugmentationSchedulerConfig:
    enabled: bool = False
    warmup_epochs: int = 2
    ramp_epochs: int = 3
    max_augmentations_per_sample: int = 2
    clean_probability_start: float = 0.7
    clean_probability_end: float = 0.25
    light_probability_start: float = 0.25
    light_probability_end: float = 0.3
    medium_probability_start: float = 0.05
    medium_probability_end: float = 0.25
    heavy_probability_start: float = 0.0
    heavy_probability_end: float = 0.2
    family_weights: AugmentationFamilyWeightsConfig = field(
        default_factory=AugmentationFamilyWeightsConfig
    )

    def __post_init__(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.ramp_epochs < 0:
            raise ValueError("ramp_epochs must be non-negative")
        if self.max_augmentations_per_sample <= 0:
            raise ValueError("max_augmentations_per_sample must be positive")

        start_sum = 0.0
        end_sum = 0.0
        for name in ("clean", "light", "medium", "heavy"):
            start_value = getattr(self, f"{name}_probability_start")
            end_value = getattr(self, f"{name}_probability_end")
            if not 0.0 <= start_value <= 1.0:
                raise ValueError(f"{name}_probability_start must be within [0.0, 1.0]")
            if not 0.0 <= end_value <= 1.0:
                raise ValueError(f"{name}_probability_end must be within [0.0, 1.0]")
            start_sum += start_value
            end_sum += end_value
        if abs(start_sum - 1.0) > 1e-6:
            raise ValueError("augmentation_scheduler *_probability_start values must sum to 1.0")
        if abs(end_sum - 1.0) > 1e-6:
            raise ValueError("augmentation_scheduler *_probability_end values must sum to 1.0")


@dataclass(slots=True)
class FeaturesConfig:
    sample_rate_hz: int
    num_mel_bins: int
    frame_length_ms: float
    frame_shift_ms: float
    fft_size: int
    window_type: str
    f_min_hz: float
    f_max_hz: float | None = None
    power: float = 2.0
    log_offset: float = 1e-6
    pad_end: bool = True
    cmvn_mode: str = "none"
    cmvn_window_frames: int = 300
    output_dtype: str = "float32"


@dataclass(slots=True)
class FeatureCacheConfig:
    namespace: str = "fbank-v1"
    train_policy: str = "precompute_cpu"
    dev_policy: str = "optional"
    infer_policy: str = "runtime"
    benchmark_device: str = "auto"
    benchmark_warmup_iterations: int = 1
    benchmark_iterations: int = 3


@dataclass(slots=True)
class ChunkingConfig:
    train_min_crop_seconds: float = 1.0
    train_max_crop_seconds: float = 4.0
    train_num_crops: int = 1
    train_short_utterance_policy: str = "repeat_pad"
    eval_max_full_utterance_seconds: float = 4.0
    eval_chunk_seconds: float = 4.0
    eval_chunk_overlap_seconds: float = 1.0
    eval_pooling: str = "mean"
    demo_max_full_utterance_seconds: float = 4.0
    demo_chunk_seconds: float = 4.0
    demo_chunk_overlap_seconds: float = 1.0
    demo_pooling: str = "mean"


@dataclass(slots=True)
class SecretRefs:
    wandb_api_key: str
    mlflow_tracking_token: str
    huggingface_hub_token: str


@dataclass(slots=True)
class DeploymentConfig:
    model_bundle_root: str
    demo_subset_root: str
    enrollment_cache_root: str
    require_artifacts: bool


@dataclass(slots=True)
class ProjectConfig:
    paths: PathsConfig
    runtime: RuntimeConfig
    reproducibility: ReproducibilityConfig
    training: TrainingConfig
    backends: BackendsConfig
    export: ExportConfig
    tracking: TrackingConfig
    telemetry: TelemetryConfig
    normalization: NormalizationConfig
    vad: VADConfig
    silence_augmentation: SilenceAugmentationConfig
    augmentation_scheduler: AugmentationSchedulerConfig
    features: FeaturesConfig
    feature_cache: FeatureCacheConfig
    chunking: ChunkingConfig
    secrets: SecretRefs
    deployment: DeploymentConfig
    resolved_secrets: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        payload = {
            "paths": asdict(self.paths),
            "runtime": asdict(self.runtime),
            "reproducibility": asdict(self.reproducibility),
            "training": asdict(self.training),
            "backends": asdict(self.backends),
            "export": asdict(self.export),
            "tracking": asdict(self.tracking),
            "telemetry": asdict(self.telemetry),
            "normalization": asdict(self.normalization),
            "vad": asdict(self.vad),
            "silence_augmentation": asdict(self.silence_augmentation),
            "augmentation_scheduler": asdict(self.augmentation_scheduler),
            "features": asdict(self.features),
            "feature_cache": asdict(self.feature_cache),
            "chunking": asdict(self.chunking),
            "secrets": asdict(self.secrets),
            "deployment": asdict(self.deployment),
            "resolved_secrets": dict(self.resolved_secrets),
        }
        if mask_secrets:
            payload["resolved_secrets"] = {
                key: mask_secret(value) for key, value in self.resolved_secrets.items()
            }
        return payload


def load_project_config(
    *,
    config_path: Path | str,
    overrides: list[str] | None = None,
    env_file: Path | str | None = None,
) -> ProjectConfig:
    config_file = Path(config_path)
    data = tomllib.loads(config_file.read_text())
    for override in overrides or []:
        apply_override(data, override)

    env = dict(os.environ)
    if env_file is not None:
        env.update(load_dotenv(Path(env_file)))

    config = ProjectConfig(
        paths=PathsConfig(**require_section(data, "paths")),
        runtime=RuntimeConfig(**require_section(data, "runtime")),
        reproducibility=ReproducibilityConfig(**require_section(data, "reproducibility")),
        training=TrainingConfig(**require_section(data, "training")),
        backends=BackendsConfig(**require_section(data, "backends")),
        export=ExportConfig(**require_section(data, "export")),
        tracking=TrackingConfig(**require_section(data, "tracking")),
        telemetry=TelemetryConfig(
            **optional_section(
                data,
                "telemetry",
                {
                    "enabled": True,
                    "structured_logs": True,
                    "metrics_enabled": True,
                    "metrics_path": "/metrics",
                },
            )
        ),
        normalization=NormalizationConfig(
            **optional_section(
                data,
                "normalization",
                {
                    "target_sample_rate_hz": 16000,
                    "target_channels": 1,
                    "output_format": "wav",
                    "output_pcm_bits_per_sample": 16,
                    "peak_headroom_db": 1.0,
                    "dc_offset_threshold": 0.01,
                    "clipped_sample_threshold": 0.999,
                    "loudness_mode": "none",
                    "target_loudness_dbfs": -27.0,
                    "max_loudness_gain_db": 20.0,
                    "max_loudness_attenuation_db": 12.0,
                },
            )
        ),
        vad=VADConfig(
            **optional_section(
                data,
                "vad",
                {
                    "mode": "none",
                    "backend": "silero_vad_v6_onnx",
                    "provider": "auto",
                    "min_output_duration_seconds": 1.0,
                    "min_retained_ratio": 0.4,
                },
            )
        ),
        silence_augmentation=SilenceAugmentationConfig(
            **optional_section(
                data,
                "silence_augmentation",
                {
                    "enabled": False,
                    "max_leading_padding_seconds": 0.0,
                    "max_trailing_padding_seconds": 0.0,
                    "max_inserted_pauses": 0,
                    "min_inserted_pause_seconds": 0.08,
                    "max_inserted_pause_seconds": 0.25,
                    "pause_ratio_min": 1.0,
                    "pause_ratio_max": 1.0,
                    "min_detected_pause_seconds": 0.08,
                    "max_perturbed_pause_seconds": 0.6,
                    "analysis_frame_ms": 20.0,
                    "silence_threshold_dbfs": -45.0,
                },
            )
        ),
        augmentation_scheduler=_load_augmentation_scheduler_config(data),
        features=FeaturesConfig(
            **optional_section(
                data,
                "features",
                {
                    "sample_rate_hz": 16000,
                    "num_mel_bins": 80,
                    "frame_length_ms": 25.0,
                    "frame_shift_ms": 10.0,
                    "fft_size": 512,
                    "window_type": "hann",
                    "f_min_hz": 20.0,
                    "f_max_hz": None,
                    "power": 2.0,
                    "log_offset": 1e-6,
                    "pad_end": True,
                    "cmvn_mode": "none",
                    "cmvn_window_frames": 300,
                    "output_dtype": "float32",
                },
            )
        ),
        feature_cache=FeatureCacheConfig(
            **optional_section(
                data,
                "feature_cache",
                {
                    "namespace": "fbank-v1",
                    "train_policy": "precompute_cpu",
                    "dev_policy": "optional",
                    "infer_policy": "runtime",
                    "benchmark_device": "auto",
                    "benchmark_warmup_iterations": 1,
                    "benchmark_iterations": 3,
                },
            )
        ),
        chunking=ChunkingConfig(
            **optional_section(
                data,
                "chunking",
                {
                    "train_min_crop_seconds": 1.0,
                    "train_max_crop_seconds": 4.0,
                    "train_num_crops": 1,
                    "train_short_utterance_policy": "repeat_pad",
                    "eval_max_full_utterance_seconds": 4.0,
                    "eval_chunk_seconds": 4.0,
                    "eval_chunk_overlap_seconds": 1.0,
                    "eval_pooling": "mean",
                    "demo_max_full_utterance_seconds": 4.0,
                    "demo_chunk_seconds": 4.0,
                    "demo_chunk_overlap_seconds": 1.0,
                    "demo_pooling": "mean",
                },
            )
        ),
        secrets=SecretRefs(**require_section(data, "secrets")),
        deployment=DeploymentConfig(
            **optional_section(
                data,
                "deployment",
                {
                    "model_bundle_root": "artifacts/model-bundle",
                    "demo_subset_root": "artifacts/demo-subset",
                    "enrollment_cache_root": "artifacts/enrollment-cache",
                    "require_artifacts": False,
                },
            )
        ),
    )
    config.resolved_secrets = {
        key: env.get(env_var_name) for key, env_var_name in asdict(config.secrets).items()
    }
    return config


def _load_augmentation_scheduler_config(data: dict[str, Any]) -> AugmentationSchedulerConfig:
    scheduler_section = optional_section(
        data,
        "augmentation_scheduler",
        {
            "enabled": False,
            "warmup_epochs": 2,
            "ramp_epochs": 3,
            "max_augmentations_per_sample": 2,
            "clean_probability_start": 0.7,
            "clean_probability_end": 0.25,
            "light_probability_start": 0.25,
            "light_probability_end": 0.3,
            "medium_probability_start": 0.05,
            "medium_probability_end": 0.25,
            "heavy_probability_start": 0.0,
            "heavy_probability_end": 0.2,
            "family_weights": {
                "noise": 1.0,
                "reverb": 1.0,
                "distance": 0.9,
                "codec": 0.8,
                "silence": 0.6,
            },
        },
    )
    family_weights_data = scheduler_section.get("family_weights", {})
    if not isinstance(family_weights_data, dict):
        raise ValueError(
            "Config section 'augmentation_scheduler.family_weights' is missing or invalid."
        )
    scheduler_values = dict(scheduler_section)
    scheduler_values["family_weights"] = AugmentationFamilyWeightsConfig(**family_weights_data)
    return AugmentationSchedulerConfig(**scheduler_values)


def require_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' is missing or invalid.")
    return value


def optional_section(data: dict[str, Any], name: str, default: dict[str, Any]) -> dict[str, Any]:
    value = data.get(name)
    if value is None:
        return dict(default)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' is missing or invalid.")
    return {**default, **value}


def apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected dotted.key=value.")

    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.split(".")
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if next_value is None:
            next_value = {}
            cursor[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot override nested key '{dotted_key}'.")
        cursor = next_value
    cursor[keys[-1]] = parse_override_value(raw_value)


def parse_override_value(raw_value: str) -> Any:
    value = raw_value.strip()
    try:
        return tomllib.loads(f"override = {value}")["override"]
    except tomllib.TOMLDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    result: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if not key or not _:
            raise ValueError(f"Invalid dotenv line: {raw_line}")
        result[key.strip()] = value.strip().strip("'").strip('"')
    return result


def mask_secret(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
