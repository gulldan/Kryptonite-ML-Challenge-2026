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


@dataclass(slots=True)
class ExportConfig:
    opset: int
    dynamic_axes: bool
    profile: str


@dataclass(slots=True)
class TrackingConfig:
    enabled: bool
    backend: str
    experiment: str
    run_name_prefix: str
    output_root: str
    copy_artifacts: bool


@dataclass(slots=True)
class NormalizationConfig:
    target_sample_rate_hz: int
    target_channels: int
    output_format: str
    output_pcm_bits_per_sample: int
    peak_headroom_db: float
    dc_offset_threshold: float
    clipped_sample_threshold: float


@dataclass(slots=True)
class VADConfig:
    mode: str
    backend: str = "silero_vad_v6_onnx"
    provider: str = "auto"
    min_output_duration_seconds: float | None = 1.0
    min_retained_ratio: float | None = 0.4


@dataclass(slots=True)
class SecretRefs:
    wandb_api_key: str
    mlflow_tracking_token: str
    huggingface_hub_token: str


@dataclass(slots=True)
class DeploymentConfig:
    model_bundle_root: str
    demo_subset_root: str
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
    normalization: NormalizationConfig
    vad: VADConfig
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
            "normalization": asdict(self.normalization),
            "vad": asdict(self.vad),
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
        secrets=SecretRefs(**require_section(data, "secrets")),
        deployment=DeploymentConfig(
            **optional_section(
                data,
                "deployment",
                {
                    "model_bundle_root": "artifacts/model-bundle",
                    "demo_subset_root": "artifacts/demo-subset",
                    "require_artifacts": False,
                },
            )
        ),
    )
    config.resolved_secrets = {
        key: env.get(env_var_name) for key, env_var_name in asdict(config.secrets).items()
    }
    return config


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
