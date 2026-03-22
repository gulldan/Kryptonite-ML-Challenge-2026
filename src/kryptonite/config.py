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
class SecretRefs:
    wandb_api_key: str
    mlflow_tracking_token: str
    huggingface_hub_token: str


@dataclass(slots=True)
class ProjectConfig:
    paths: PathsConfig
    runtime: RuntimeConfig
    training: TrainingConfig
    backends: BackendsConfig
    export: ExportConfig
    secrets: SecretRefs
    resolved_secrets: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        payload = {
            "paths": asdict(self.paths),
            "runtime": asdict(self.runtime),
            "training": asdict(self.training),
            "backends": asdict(self.backends),
            "export": asdict(self.export),
            "secrets": asdict(self.secrets),
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
        training=TrainingConfig(**require_section(data, "training")),
        backends=BackendsConfig(**require_section(data, "backends")),
        export=ExportConfig(**require_section(data, "export")),
        secrets=SecretRefs(**require_section(data, "secrets")),
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
