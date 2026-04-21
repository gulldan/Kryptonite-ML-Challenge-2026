"""Apply NVIDIA ModelOpt ONNX AutoCast and write a patched model bundle."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    metadata_path = Path(args.metadata).resolve()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = _load_json_object(metadata_path)
    source_model_path = _resolve(project_root, _metadata_model_path(metadata))
    output_model_path = output_root / args.output_model_name
    _convert_onnx(
        source_model_path=source_model_path,
        output_model_path=output_model_path,
        low_precision_type=args.low_precision_type,
        keep_io_types=args.keep_io_types,
        data_max=args.data_max,
        init_max=args.init_max,
        providers=args.providers,
    )
    patched_metadata = _patched_metadata(
        metadata,
        project_root=project_root,
        output_model_path=output_model_path,
        source_model_path=source_model_path,
        low_precision_type=args.low_precision_type,
        keep_io_types=args.keep_io_types,
    )
    output_metadata_path = output_root / "metadata.json"
    output_metadata_path.write_text(
        json.dumps(patched_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_root / "modelopt_autocast_report.md"
    report_path.write_text(
        _render_report(
            source_model_path=source_model_path,
            output_model_path=output_model_path,
            output_metadata_path=output_metadata_path,
            low_precision_type=args.low_precision_type,
            keep_io_types=args.keep_io_types,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_model_path": str(source_model_path),
                "output_model_path": str(output_model_path),
                "output_metadata_path": str(output_metadata_path),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _convert_onnx(
    *,
    source_model_path: Path,
    output_model_path: Path,
    low_precision_type: str,
    keep_io_types: bool,
    data_max: str,
    init_max: str,
    providers: list[str],
) -> None:
    try:
        onnx = importlib.import_module("onnx")
        autocast = importlib.import_module("modelopt.onnx.autocast")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ModelOpt AutoCast requires `nvidia-modelopt` in the repo-local `.venv`. "
            "Install it with `uv pip install --python .venv/bin/python "
            "--extra-index-url https://pypi.nvidia.com nvidia-modelopt[onnx]`."
        ) from exc

    converted_model = autocast.convert_to_mixed_precision(
        onnx_path=str(source_model_path),
        low_precision_type=low_precision_type,
        keep_io_types=keep_io_types,
        data_max=float("inf") if data_max == "inf" else float(data_max),
        init_max=float("inf") if init_max == "inf" else float(init_max),
        providers=providers,
    )
    onnx.save_model(
        converted_model,
        str(output_model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_model_path.name}.data",
        size_threshold=1024,
    )


def _patched_metadata(
    metadata: dict[str, Any],
    *,
    project_root: Path,
    output_model_path: Path,
    source_model_path: Path,
    low_precision_type: str,
    keep_io_types: bool,
) -> dict[str, Any]:
    patched = json.loads(json.dumps(metadata))
    relative_model_path = _relative_to_project(output_model_path, project_root)
    patched["model_file"] = relative_model_path
    patched["model_path"] = relative_model_path
    inference_package = patched.get("inference_package")
    if isinstance(inference_package, dict):
        artifacts = inference_package.setdefault("artifacts", {})
        if isinstance(artifacts, dict):
            artifacts["onnx_model_file"] = relative_model_path
    export_validation = patched.setdefault("export_validation", {})
    if isinstance(export_validation, dict):
        export_validation["modelopt_autocast_applied"] = True
        export_validation["modelopt_autocast_low_precision_type"] = low_precision_type
        export_validation["modelopt_autocast_keep_io_types"] = keep_io_types
        export_validation["modelopt_autocast_source_model"] = _relative_to_project(
            source_model_path,
            project_root,
        )
        export_validation["modelopt_autocast_at_utc"] = datetime.now(tz=UTC).isoformat(
            timespec="seconds"
        )
    return patched


def _metadata_model_path(metadata: dict[str, Any]) -> str:
    inference_package = metadata.get("inference_package")
    if isinstance(inference_package, dict):
        artifacts = inference_package.get("artifacts")
        if isinstance(artifacts, dict):
            model_file = artifacts.get("onnx_model_file")
            if isinstance(model_file, str) and model_file.strip():
                return model_file
    model_file = metadata.get("model_file")
    if isinstance(model_file, str) and model_file.strip():
        return model_file
    model_path = metadata.get("model_path")
    if isinstance(model_path, str) and model_path.strip():
        return model_path
    raise ValueError("Metadata does not contain an ONNX model path.")


def _render_report(
    *,
    source_model_path: Path,
    output_model_path: Path,
    output_metadata_path: Path,
    low_precision_type: str,
    keep_io_types: bool,
) -> str:
    return "\n".join(
        [
            "# ModelOpt ONNX AutoCast",
            "",
            f"- Source ONNX: `{source_model_path}`",
            f"- Output ONNX: `{output_model_path}`",
            f"- Output metadata: `{output_metadata_path}`",
            f"- Low precision type: `{low_precision_type}`",
            f"- Keep IO types: `{str(keep_io_types).lower()}`",
            "",
        ]
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Source model-bundle metadata.json.")
    parser.add_argument("--output-root", required=True, help="Output model bundle directory.")
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--output-model-name", default="model_modelopt_fp16.onnx")
    parser.add_argument("--low-precision-type", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--keep-io-types", action="store_true")
    parser.add_argument("--data-max", default="inf")
    parser.add_argument("--init-max", default="inf")
    parser.add_argument("--providers", nargs="+", default=["cpu"])
    return parser.parse_args()


if __name__ == "__main__":
    main()
