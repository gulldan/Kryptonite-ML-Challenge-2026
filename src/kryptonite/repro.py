"""Seed control, fingerprints, and lightweight reproducibility checks."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import ProjectConfig


def set_global_seed(
    seed: int,
    *,
    deterministic: bool,
    pythonhashseed: int,
) -> dict[str, Any]:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(pythonhashseed)

    numpy_seeded = False
    torch_seeded = False
    torch_deterministic = False

    try:
        numpy = importlib.import_module("numpy")
    except ImportError:
        numpy = None
    if numpy is not None:
        numpy.random.seed(seed)
        numpy_seeded = True

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(seed)
        torch_seeded = True
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            torch_deterministic = True

    return {
        "seed": seed,
        "deterministic": deterministic,
        "pythonhashseed": pythonhashseed,
        "numpy_seeded": numpy_seeded,
        "torch_seeded": torch_seeded,
        "torch_deterministic": torch_deterministic,
    }


def fingerprint_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "kind": "missing",
            "sha256": None,
            "file_count": 0,
        }

    if path.is_file():
        return {
            "path": str(path),
            "exists": True,
            "kind": "file",
            "sha256": sha256_bytes(path.read_bytes()),
            "file_count": 1,
        }

    hasher = hashlib.sha256()
    file_count = 0
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = child.relative_to(path).as_posix().encode()
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(child.read_bytes())
        file_count += 1

    return {
        "path": str(path),
        "exists": True,
        "kind": "directory",
        "sha256": hasher.hexdigest(),
        "file_count": file_count,
    }


def collect_fingerprints(project_root: Path, configured_paths: list[str]) -> list[dict[str, Any]]:
    return [fingerprint_path((project_root / raw_path).resolve()) for raw_path in configured_paths]


def build_reproducibility_snapshot(
    *,
    config: ProjectConfig,
    config_path: Path | str,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent.resolve()
    seed_state = set_global_seed(
        config.runtime.seed,
        deterministic=config.reproducibility.deterministic,
        pythonhashseed=config.reproducibility.pythonhashseed,
    )

    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "git_commit": get_git_commit(project_root),
        "config_path": str(config_file),
        "config_sha256": sha256_bytes(config_file.read_bytes()),
        "seed_state": seed_state,
        "fingerprints": collect_fingerprints(
            project_root, config.reproducibility.fingerprint_paths
        ),
    }
    probe = run_probe()
    return {"metadata": metadata, "probe": probe}


def run_probe(sample_size: int = 8) -> dict[str, Any]:
    values = [random.random() for _ in range(sample_size)]
    return {
        "sample_size": sample_size,
        "random_trace": values,
        "random_mean": sum(values) / sample_size,
        "string_hash": hash("kryptonite"),
    }


def run_reproducibility_self_check(
    *,
    script_path: Path,
    config_path: Path,
    env_file: Path,
    overrides: list[str],
    config: ProjectConfig,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(config.reproducibility.pythonhashseed)

    first = run_snapshot_subprocess(
        script_path=script_path,
        config_path=config_path,
        env_file=env_file,
        overrides=overrides,
        env=env,
    )
    second = run_snapshot_subprocess(
        script_path=script_path,
        config_path=config_path,
        env_file=env_file,
        overrides=overrides,
        env=env,
    )
    comparison = compare_snapshots(first, second)
    return {"first": first, "second": second, "comparison": comparison}


def run_snapshot_subprocess(
    *,
    script_path: Path,
    config_path: Path,
    env_file: Path,
    overrides: list[str],
    env: dict[str, str],
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--env-file",
        str(env_file),
        "--emit-json",
    ]
    for override in overrides:
        command.extend(["--override", override])

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


def compare_snapshots(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    metadata_equal = first["metadata"] == second["metadata"]
    random_mean_close = math.isclose(
        first["probe"]["random_mean"],
        second["probe"]["random_mean"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    trace_equal = first["probe"]["random_trace"] == second["probe"]["random_trace"]
    string_hash_equal = first["probe"]["string_hash"] == second["probe"]["string_hash"]

    return {
        "passed": metadata_equal and random_mean_close and trace_equal and string_hash_equal,
        "metadata_equal": metadata_equal,
        "random_mean_close": random_mean_close,
        "trace_equal": trace_equal,
        "string_hash_equal": string_hash_equal,
    }


def get_git_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
