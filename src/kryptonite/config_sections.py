"""Typed configuration sections shared by the project config loader."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    domain_balance_enabled: bool = False
    domain_balance_external_share: float = 0.0
    domain_balance_external_source_prefixes: list[str] = field(
        default_factory=lambda: ["cnceleb", "ffsvc"]
    )

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.eval_batch_size <= 0:
            raise ValueError("training.eval_batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("training.max_epochs must be positive")
        if not 0.0 <= self.domain_balance_external_share <= 1.0:
            raise ValueError("training.domain_balance_external_share must be within [0.0, 1.0]")
        prefixes = [
            prefix.strip().lower() for prefix in self.domain_balance_external_source_prefixes
        ]
        if any(not prefix for prefix in prefixes):
            raise ValueError("training.domain_balance_external_source_prefixes cannot be empty")
        self.domain_balance_external_source_prefixes = prefixes


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
    frontend: str = "local"


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
