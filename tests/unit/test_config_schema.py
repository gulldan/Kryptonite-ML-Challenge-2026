from __future__ import annotations

from pathlib import Path

from kryptonite.config_schema import validate_config_file


def test_base_config_matches_repository_schema() -> None:
    errors = validate_config_file(
        config_path=Path("configs/base.toml"),
        schema_path=Path("configs/schema.json"),
    )

    assert errors == []
