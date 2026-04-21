from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from kryptonite.models.campp.model import CAMPPlusConfig


def _load_weight_soup_module() -> Any:
    script_path = Path(__file__).parents[2] / "scripts" / "build_campp_weight_soup.py"
    spec = importlib.util.spec_from_file_location("build_campp_weight_soup", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_campp_weight_soup_averages_floating_tensors_and_copies_buffers(
    tmp_path: Path,
) -> None:
    module = _load_weight_soup_module()
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    _write_checkpoint(first, weight_value=1.0, buffer_value=3)
    _write_checkpoint(second, weight_value=5.0, buffer_value=9)

    output_checkpoint = tmp_path / "soup" / "campp_encoder.pt"
    result = module.build_campp_weight_soup(
        sources=(
            module.SoupSource(
                name="first",
                path=first,
                raw_weight=1.0,
                normalized_weight=0.25,
            ),
            module.SoupSource(
                name="second",
                path=second,
                raw_weight=3.0,
                normalized_weight=0.75,
            ),
        ),
        output_checkpoint=output_checkpoint,
        project_root=tmp_path,
        reference_source="first",
    )

    payload = torch.load(output_checkpoint, map_location="cpu", weights_only=True)
    assert result.floating_tensor_count == 1
    assert result.copied_tensor_count == 1
    torch.testing.assert_close(
        payload["model_state_dict"]["layer.weight"],
        torch.full((2, 2), 4.0),
    )
    torch.testing.assert_close(
        payload["model_state_dict"]["layer.num_batches_tracked"],
        torch.tensor(3, dtype=torch.int64),
    )
    assert payload["soup_metadata"]["reference_source"] == "first"
    assert Path(result.metadata_path).is_file()


def test_normalize_sources_rejects_unknown_weight_override() -> None:
    module = _load_weight_soup_module()

    try:
        module._normalize_sources(
            parsed_sources=(("first", Path("first.pt")),),
            weight_overrides={"second": 1.0},
            allow_negative_weights=False,
        )
    except ValueError as error:
        assert "unknown source" in str(error)
    else:
        raise AssertionError("Expected an unknown source weight override to fail.")


def _write_checkpoint(path: Path, *, weight_value: float, buffer_value: int) -> None:
    torch.save(
        {
            "model_config": asdict(CAMPPlusConfig()),
            "model_state_dict": {
                "layer.weight": torch.full((2, 2), weight_value),
                "layer.num_batches_tracked": torch.tensor(buffer_value, dtype=torch.int64),
            },
        },
        path,
    )
