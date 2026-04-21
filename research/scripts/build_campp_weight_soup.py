"""Build a CAM++ weight-space soup checkpoint from compatible encoder checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from kryptonite.models.campp.checkpoint import (
    load_campp_checkpoint_payload,
    load_campp_state_and_config,
    resolve_campp_checkpoint_path,
)
from kryptonite.models.campp.model import CAMPPlusConfig


@dataclass(frozen=True, slots=True)
class SoupSource:
    name: str
    path: Path
    raw_weight: float
    normalized_weight: float


@dataclass(frozen=True, slots=True)
class SoupBuildResult:
    output_checkpoint: Path
    metadata_path: Path
    source_count: int
    floating_tensor_count: int
    copied_tensor_count: int


def main() -> None:
    args = _parse_args()
    weight_overrides = _parse_weight_overrides(args.weight)
    parsed_sources = _parse_source_specs(args.source)
    sources = _normalize_sources(
        parsed_sources=parsed_sources,
        weight_overrides=weight_overrides,
        allow_negative_weights=args.allow_negative_weights,
    )
    result = build_campp_weight_soup(
        sources=sources,
        output_checkpoint=Path(args.output_checkpoint),
        project_root=Path(args.project_root),
        reference_source=args.reference_source,
        metadata_path=Path(args.metadata_path) if args.metadata_path else None,
    )
    print(
        json.dumps(
            {
                "output_checkpoint": str(result.output_checkpoint),
                "metadata_path": str(result.metadata_path),
                "source_count": result.source_count,
                "floating_tensor_count": result.floating_tensor_count,
                "copied_tensor_count": result.copied_tensor_count,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def build_campp_weight_soup(
    *,
    sources: tuple[SoupSource, ...],
    output_checkpoint: Path,
    project_root: Path = Path("."),
    reference_source: str = "",
    metadata_path: Path | None = None,
) -> SoupBuildResult:
    if not sources:
        raise ValueError("At least one source checkpoint is required.")
    reference_index = _resolve_reference_index(sources=sources, reference_source=reference_source)
    loaded_sources = [
        _load_source_checkpoint(source=source, project_root=project_root) for source in sources
    ]
    reference_source_data = loaded_sources[reference_index]
    model_config = _require_matching_configs(loaded_sources)
    state_keys = tuple(reference_source_data["state"].keys())
    _require_matching_state_dicts(loaded_sources, state_keys=state_keys)

    averaged_state: dict[str, torch.Tensor] = {}
    copied_tensor_names: list[str] = []
    floating_tensor_count = 0
    for key in state_keys:
        reference_tensor = reference_source_data["state"][key]
        if torch.is_floating_point(reference_tensor):
            averaged_state[key] = _weighted_tensor_average(
                key=key,
                loaded_sources=loaded_sources,
                reference_tensor=reference_tensor,
            )
            floating_tensor_count += 1
        else:
            averaged_state[key] = reference_tensor.detach().cpu().clone()
            copied_tensor_names.append(key)

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata = _build_metadata(
        sources=sources,
        resolved_paths=[source_data["resolved_path"] for source_data in loaded_sources],
        reference_source=sources[reference_index].name,
        floating_tensor_count=floating_tensor_count,
        copied_tensor_names=copied_tensor_names,
    )
    output_payload = {
        "model_state_dict": averaged_state,
        "model_config": asdict(model_config),
        "soup_metadata": metadata,
    }
    torch.save(output_payload, output_checkpoint)

    resolved_metadata_path = metadata_path or output_checkpoint.with_suffix(".metadata.json")
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SoupBuildResult(
        output_checkpoint=output_checkpoint,
        metadata_path=resolved_metadata_path,
        source_count=len(sources),
        floating_tensor_count=floating_tensor_count,
        copied_tensor_count=len(copied_tensor_names),
    )


def _load_source_checkpoint(*, source: SoupSource, project_root: Path) -> dict[str, Any]:
    resolved_path = resolve_campp_checkpoint_path(
        checkpoint_path=source.path,
        project_root=project_root,
    )
    payload = load_campp_checkpoint_payload(torch=torch, checkpoint_path=resolved_path)
    model_config, state = load_campp_state_and_config(payload)
    return {
        "source": source,
        "resolved_path": resolved_path,
        "model_config": model_config,
        "state": state,
    }


def _require_matching_configs(loaded_sources: list[dict[str, Any]]) -> CAMPPlusConfig:
    reference_config = loaded_sources[0]["model_config"]
    for source_data in loaded_sources[1:]:
        if source_data["model_config"] != reference_config:
            left_name = loaded_sources[0]["source"].name
            right_name = source_data["source"].name
            raise ValueError(
                f"CAM++ model_config mismatch between {left_name!r} and {right_name!r}."
            )
    return reference_config


def _require_matching_state_dicts(
    loaded_sources: list[dict[str, Any]],
    *,
    state_keys: tuple[str, ...],
) -> None:
    reference_key_set = set(state_keys)
    reference_state = loaded_sources[0]["state"]
    for source_data in loaded_sources[1:]:
        state = source_data["state"]
        name = source_data["source"].name
        if set(state.keys()) != reference_key_set:
            missing = sorted(reference_key_set - set(state.keys()))[:10]
            extra = sorted(set(state.keys()) - reference_key_set)[:10]
            raise ValueError(
                f"State dict keys differ for {name!r}; missing={missing}, extra={extra}."
            )
        for key in state_keys:
            if state[key].shape != reference_state[key].shape:
                raise ValueError(
                    f"Tensor shape mismatch for key={key!r} in source={name!r}: "
                    f"{tuple(state[key].shape)} != {tuple(reference_state[key].shape)}."
                )


def _weighted_tensor_average(
    *,
    key: str,
    loaded_sources: list[dict[str, Any]],
    reference_tensor: torch.Tensor,
) -> torch.Tensor:
    accumulator = torch.zeros_like(reference_tensor, dtype=torch.float32, device="cpu")
    for source_data in loaded_sources:
        source = source_data["source"]
        tensor = source_data["state"][key].detach().cpu()
        if not torch.is_floating_point(tensor):
            raise ValueError(f"Tensor key={key!r} is not floating in source={source.name!r}.")
        accumulator.add_(tensor.to(dtype=torch.float32), alpha=source.normalized_weight)
    return accumulator.to(dtype=reference_tensor.dtype)


def _build_metadata(
    *,
    sources: tuple[SoupSource, ...],
    resolved_paths: list[Path],
    reference_source: str,
    floating_tensor_count: int,
    copied_tensor_names: list[str],
) -> dict[str, Any]:
    return {
        "format_version": "kryptonite.campp.weight_soup.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reference_source": reference_source,
        "sources": [
            {
                "name": source.name,
                "path": str(source.path),
                "resolved_path": str(resolved_path),
                "raw_weight": source.raw_weight,
                "normalized_weight": source.normalized_weight,
            }
            for source, resolved_path in zip(sources, resolved_paths, strict=True)
        ],
        "floating_tensor_count": floating_tensor_count,
        "copied_non_floating_tensor_count": len(copied_tensor_names),
        "copied_non_floating_tensor_names": copied_tensor_names,
    }


def _parse_source_specs(specs: list[str]) -> tuple[tuple[str, Path], ...]:
    parsed: list[tuple[str, Path]] = []
    names: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --source {spec!r}; expected NAME=PATH.")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --source {spec!r}; source name is empty.")
        if name in names:
            raise ValueError(f"Duplicate source name {name!r}.")
        names.add(name)
        parsed.append((name, Path(raw_path)))
    return tuple(parsed)


def _parse_weight_overrides(specs: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --weight {spec!r}; expected NAME=FLOAT.")
        name, raw_weight = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --weight {spec!r}; source name is empty.")
        if name in weights:
            raise ValueError(f"Duplicate weight for source {name!r}.")
        weights[name] = float(raw_weight)
    return weights


def _normalize_sources(
    *,
    parsed_sources: tuple[tuple[str, Path], ...],
    weight_overrides: dict[str, float],
    allow_negative_weights: bool,
) -> tuple[SoupSource, ...]:
    names = {name for name, _ in parsed_sources}
    unknown_weights = sorted(set(weight_overrides) - names)
    if unknown_weights:
        raise ValueError(f"Weight override references unknown source(s): {unknown_weights}.")
    raw_weights = {name: weight_overrides.get(name, 1.0) for name, _ in parsed_sources}
    if not allow_negative_weights:
        negative = {name: weight for name, weight in raw_weights.items() if weight < 0.0}
        if negative:
            raise ValueError(
                f"Negative source weights require --allow-negative-weights: {negative}."
            )
    weight_sum = sum(raw_weights.values())
    if weight_sum == 0.0:
        raise ValueError("Source weights sum to zero.")
    return tuple(
        SoupSource(
            name=name,
            path=path,
            raw_weight=raw_weights[name],
            normalized_weight=raw_weights[name] / weight_sum,
        )
        for name, path in parsed_sources
    )


def _resolve_reference_index(
    *,
    sources: tuple[SoupSource, ...],
    reference_source: str,
) -> int:
    if not reference_source:
        return 0
    for index, source in enumerate(sources):
        if source.name == reference_source:
            return index
    raise ValueError(f"Unknown --reference-source {reference_source!r}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source checkpoint as NAME=PATH. Repeat for every checkpoint.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Optional source weight as NAME=FLOAT. Missing sources default to 1.",
    )
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--metadata-path", default="")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--reference-source",
        default="",
        help="Source name used for non-floating buffers. Defaults to the first source.",
    )
    parser.add_argument(
        "--allow-negative-weights",
        action="store_true",
        help="Allow non-convex interpolation weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
