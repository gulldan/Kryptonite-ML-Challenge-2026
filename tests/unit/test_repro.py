from __future__ import annotations

import random
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.repro import (
    build_reproducibility_snapshot,
    compare_snapshots,
    fingerprint_path,
    run_reproducibility_self_check,
    set_global_seed,
)


def test_set_global_seed_repeats_python_random_sequence() -> None:
    set_global_seed(17, deterministic=True, pythonhashseed=17)
    first = [random.random() for _ in range(4)]

    set_global_seed(17, deterministic=True, pythonhashseed=17)
    second = [random.random() for _ in range(4)]

    assert first == second


def test_fingerprint_path_is_stable_for_files(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("kryptonite\n")

    first = fingerprint_path(target)
    second = fingerprint_path(target)

    assert first == second
    assert first["kind"] == "file"
    assert first["sha256"] is not None


def test_build_reproducibility_snapshot_contains_expected_seed_metadata() -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))

    snapshot = build_reproducibility_snapshot(
        config=config,
        config_path=Path("configs/base.toml"),
    )

    assert snapshot["metadata"]["seed_state"]["seed"] == 42
    assert snapshot["metadata"]["seed_state"]["pythonhashseed"] == 42
    assert snapshot["probe"]["sample_size"] == 8


def test_reproducibility_self_check_passes() -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))

    result = run_reproducibility_self_check(
        script_path=Path("scripts/repro_check.py"),
        config_path=Path("configs/base.toml"),
        env_file=Path(".env"),
        overrides=[],
        config=config,
    )

    assert result["comparison"]["passed"] is True
    assert compare_snapshots(result["first"], result["second"])["passed"] is True
