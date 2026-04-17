"""Extract Teacher PEFT embeddings and apply the graph retrieval tail."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from kryptonite.eda.community import (
    LabelPropagationConfig,
    exact_topk,
    label_propagation_rerank,
    write_submission,
)
from kryptonite.eda.dense_audio import eval_crops, l2_normalize_rows, load_eval_waveform
from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission
from kryptonite.serve.tensorrt_generic import MultiInputTensorRTEngineRunner
from kryptonite.training.teacher_peft import (
    load_teacher_peft_encoder_from_checkpoint,
    resolve_teacher_peft_checkpoint_path,
)


@dataclass(frozen=True, slots=True)
class _EvalCropExample:
    row_index: int
    crops: list[np.ndarray]


@dataclass(frozen=True, slots=True)
class _EvalCropBatch:
    model_inputs: dict[str, Any]
    owners: Any
    row_count: int


class _TorchTeacherPeftEncoder:
    def __init__(self, encoder: Any) -> None:
        self._encoder = encoder

    def __call__(self, model_inputs: dict[str, Any]) -> np.ndarray:
        values = self._encoder(**model_inputs).detach().float().cpu().numpy()
        return values.astype(np.float32, copy=False)


class _TensorRTTeacherPeftEncoder:
    def __init__(
        self,
        *,
        engine_path: Path,
        output_name: str,
        profile_index: int,
    ) -> None:
        self._runner = MultiInputTensorRTEngineRunner(
            engine_path=engine_path,
            output_name=output_name,
        )
        self._profile_index = profile_index

    def __call__(self, model_inputs: dict[str, Any]) -> np.ndarray:
        output = self._runner.run(model_inputs, profile_index=self._profile_index)
        return output.detach().float().cpu().numpy().astype(np.float32, copy=False)


class _TeacherPeftEvalDataset:
    def __init__(
        self,
        *,
        paths: list[str],
        row_indices: np.ndarray,
        trim: bool,
        crop_samples: int,
        n_crops: int,
    ) -> None:
        self._paths = paths
        self._row_indices = row_indices.astype(np.int64, copy=False)
        self._trim = trim
        self._crop_samples = crop_samples
        self._n_crops = n_crops

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> _EvalCropExample:
        waveform = load_eval_waveform(Path(self._paths[index]), trim=self._trim)
        crops = eval_crops(
            waveform,
            crop_samples=self._crop_samples,
            n_crops=self._n_crops,
        )
        return _EvalCropExample(
            row_index=int(self._row_indices[index]),
            crops=crops,
        )


class _TeacherPeftEvalCollator:
    def __init__(self, *, feature_extractor: Any, sample_rate_hz: int) -> None:
        self._feature_extractor = feature_extractor
        self._sample_rate_hz = sample_rate_hz

    def __call__(self, batch: list[_EvalCropExample]) -> _EvalCropBatch:
        import torch

        if not batch:
            raise ValueError("Evaluation batch must not be empty.")
        waveforms: list[np.ndarray] = []
        owners: list[int] = []
        for example in batch:
            waveforms.extend(example.crops)
            owners.extend([example.row_index] * len(example.crops))
        encoded = self._feature_extractor(
            waveforms,
            sampling_rate=self._sample_rate_hz,
            padding=True,
            return_tensors="pt",
        )
        model_inputs: dict[str, Any] = {}
        for key, value in encoded.items():
            if key == "attention_mask":
                model_inputs[key] = value.to(dtype=torch.int32)
                continue
            model_inputs[key] = value.to(dtype=torch.float32)
        return _EvalCropBatch(
            model_inputs=model_inputs,
            owners=torch.tensor(owners, dtype=torch.long),
            row_count=len(batch),
        )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    if args.max_rows > 0:
        manifest = manifest.head(args.max_rows)
    print(
        f"[teacher-peft-tail] start experiment={args.experiment_id} "
        f"checkpoint={args.checkpoint_path} rows={manifest.height} output_dir={output_dir}",
        flush=True,
    )

    total_started = time.perf_counter()
    started = time.perf_counter()
    embeddings, embedding_path, embedding_source = _load_or_extract_embeddings(
        args,
        manifest,
        output_dir,
    )
    embedding_s = time.perf_counter() - started

    print("[teacher-peft-tail] exact_topk start", flush=True)
    started = time.perf_counter()
    indices, scores = exact_topk(
        embeddings,
        top_k=args.top_cache_k,
        batch_size=args.search_batch_size,
        device=args.search_device,
    )
    search_s = time.perf_counter() - started

    label_config = LabelPropagationConfig(
        experiment_id=args.experiment_id,
        edge_top=args.edge_top,
        reciprocal_top=args.reciprocal_top,
        rank_top=args.rank_top,
        iterations=args.iterations,
        label_min_size=args.label_min_size,
        label_max_size=args.label_max_size,
        label_min_candidates=args.label_min_candidates,
        shared_top=args.shared_top,
        shared_min_count=args.shared_min_count,
        reciprocal_bonus=args.reciprocal_bonus,
        density_penalty=args.density_penalty,
    )
    print("[teacher-peft-tail] label_propagation_rerank start", flush=True)
    started = time.perf_counter()
    top_indices, top_scores, label_meta = label_propagation_rerank(
        indices=indices,
        scores=scores,
        config=label_config,
        top_k=10,
    )
    rerank_s = time.perf_counter() - started

    indegree = np.bincount(top_indices.ravel(), minlength=manifest.height)
    rows: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "encoder_backend": args.encoder_backend,
        "checkpoint_path": args.checkpoint_path,
        "tensorrt_engine_path": args.tensorrt_engine_path,
        "tensorrt_profile_index": args.tensorrt_profile_index,
        "manifest_csv": args.manifest_csv,
        "embedding_s": round(embedding_s, 6),
        "search_s": round(search_s, 6),
        "rerank_s": round(rerank_s, 6),
        "top1_score_mean": float(top_scores[:, 0].mean()),
        "top10_mean_score_mean": float(top_scores.mean()),
        "indegree_gini_10": gini(indegree),
        "indegree_max_10": int(indegree.max()),
        "embedding_path": str(embedding_path),
        "embedding_source": embedding_source,
        "label_config": _jsonable_label_config(label_config),
        **label_meta,
    }
    if args.template_csv:
        submission_path = output_dir / f"submission_{args.experiment_id}.csv"
        started = time.perf_counter()
        write_submission(manifest=manifest, top_indices=top_indices, output_csv=submission_path)
        submit_write_s = time.perf_counter() - started
        started = time.perf_counter()
        validation = validate_submission(
            template_csv=Path(args.template_csv),
            submission_csv=submission_path,
        )
        validation_s = time.perf_counter() - started
        validation_path = output_dir / f"submission_{args.experiment_id}_validation.json"
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.update(
            {
                "submission_path": str(submission_path),
                "validator_passed": bool(validation["passed"]),
                "submit_write_s": round(submit_write_s, 6),
                "validation_s": round(validation_s, 6),
                "submit_generation_s": round(
                    embedding_s + search_s + rerank_s + submit_write_s,
                    6,
                ),
            }
        )
    rows["wall_total_s"] = round(time.perf_counter() - total_started, 6)
    (output_dir / f"{args.experiment_id}_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pl.DataFrame([{key: _csv_value(value) for key, value in rows.items()}]).write_csv(
        output_dir / f"{args.experiment_id}_summary.csv"
    )
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


def _load_or_extract_embeddings(
    args: argparse.Namespace,
    manifest: pl.DataFrame,
    output_dir: Path,
) -> tuple[np.ndarray, Path, str]:
    output_path = output_dir / f"embeddings_{args.experiment_id}.npy"
    if output_path.is_file() and not args.force_embeddings:
        print(f"[teacher-peft-tail] load cached embeddings path={output_path}", flush=True)
        return np.load(output_path), output_path, "output_cache"
    if args.embedding_cache_path and not args.force_embeddings:
        return _load_external_embedding_cache(
            cache_path=Path(args.embedding_cache_path),
            output_path=output_path,
            row_count=manifest.height,
            copy_cache=args.copy_embedding_cache,
        )

    import torch
    from torch.utils.data import DataLoader

    feature_extractor, encoder = _build_encoder_backend(args)

    crop_samples = int(round(args.crop_seconds * 16_000))
    paths = _resolve_manifest_paths(manifest, data_root=args.data_root)
    dataset = _TeacherPeftEvalDataset(
        paths=paths,
        row_indices=_manifest_row_indices(manifest),
        trim=args.trim,
        crop_samples=crop_samples,
        n_crops=args.n_crops,
    )
    loader = DataLoader(
        cast(Any, dataset),
        batch_size=_resolve_row_batch_size(args),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=_TeacherPeftEvalCollator(
            feature_extractor=feature_extractor,
            sample_rate_hz=16_000,
        ),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    sums: np.ndarray | None = None
    counts = np.zeros(manifest.height, dtype=np.int32)
    started_at = time.perf_counter()
    log_every_rows = max(1, manifest.height // 20)
    next_log_row = 1
    completed_rows = 0
    with (
        torch.inference_mode(),
        torch.amp.autocast(
            args.device,
            enabled=args.device == "cuda" and args.precision == "bf16",
        ),
    ):
        for batch in loader:
            sums = _flush_embeddings(
                encoder=encoder,
                model_inputs=batch.model_inputs,
                owners=batch.owners,
                sums=sums,
                counts=counts,
                row_count=manifest.height,
                args=args,
            )
            completed_rows += batch.row_count
            if (
                completed_rows == 1
                or completed_rows >= next_log_row
                or completed_rows == manifest.height
            ):
                elapsed_s = max(time.perf_counter() - started_at, 1e-9)
                print(
                    f"[teacher-peft-tail] extract rows={completed_rows}/{manifest.height} "
                    f"pct={100.0 * completed_rows / manifest.height:.1f} "
                    f"rows_per_s={completed_rows / elapsed_s:.2f} elapsed_s={elapsed_s:.1f}",
                    flush=True,
                )
                next_log_row = completed_rows + log_every_rows
    if sums is None:
        raise RuntimeError("No embeddings were extracted.")
    embeddings = l2_normalize_rows(sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    np.save(output_path, embeddings)
    return embeddings, output_path, "extracted"


def _load_external_embedding_cache(
    *,
    cache_path: Path,
    output_path: Path,
    row_count: int,
    copy_cache: bool,
) -> tuple[np.ndarray, Path, str]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"Embedding cache does not exist: {cache_path}")
    print(f"[teacher-peft-tail] load external embeddings path={cache_path}", flush=True)
    embeddings = np.load(cache_path)
    if embeddings.ndim != 2 or embeddings.shape[0] != row_count:
        raise ValueError(
            "Embedding cache shape is incompatible with manifest: "
            f"shape={embeddings.shape}, expected rows={row_count}."
        )
    embeddings = embeddings.astype(np.float32, copy=False)
    if copy_cache and cache_path.resolve() != output_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
        return embeddings, output_path, "external_cache_copy"
    return embeddings, cache_path, "external_cache"


def _flush_embeddings(
    *,
    encoder: Any,
    model_inputs: dict[str, Any],
    owners: Any,
    sums: np.ndarray | None,
    counts: np.ndarray,
    row_count: int,
    args: argparse.Namespace,
) -> np.ndarray:
    inputs = {
        key: value.to(args.device, non_blocking=args.pin_memory)
        for key, value in model_inputs.items()
    }
    owner_indices = owners.cpu().numpy().astype(np.int64, copy=False)
    values = encoder(inputs)
    values = l2_normalize_rows(values.astype(np.float32, copy=False))
    if sums is None:
        sums = np.zeros((row_count, values.shape[1]), dtype=np.float32)
    np.add.at(sums, owner_indices, values)
    np.add.at(counts, owner_indices, 1)
    return sums


def _build_encoder_backend(args: argparse.Namespace) -> tuple[Any, Any]:
    if args.encoder_backend == "torch":
        token = os.environ.get(args.hf_token_env) or None
        _, _, feature_extractor, encoder = load_teacher_peft_encoder_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            token=token,
            trainable=False,
        )
        encoder = encoder.to(args.device)
        encoder.eval()
        return feature_extractor, _TorchTeacherPeftEncoder(encoder)
    if args.encoder_backend == "tensorrt":
        if not args.tensorrt_engine_path:
            raise ValueError("--tensorrt-engine-path is required for --encoder-backend=tensorrt.")
        feature_extractor = _load_feature_extractor_only(args)
        encoder = _TensorRTTeacherPeftEncoder(
            engine_path=Path(args.tensorrt_engine_path),
            output_name=args.tensorrt_output_name,
            profile_index=args.tensorrt_profile_index,
        )
        return feature_extractor, encoder
    raise ValueError(f"Unsupported encoder backend: {args.encoder_backend!r}")


def _load_feature_extractor_only(args: argparse.Namespace) -> Any:
    from transformers import AutoFeatureExtractor

    token = os.environ.get(args.hf_token_env) or None
    checkpoint_dir = resolve_teacher_peft_checkpoint_path(
        checkpoint_path=args.checkpoint_path,
        project_root=".",
    )
    return AutoFeatureExtractor.from_pretrained(
        checkpoint_dir / "feature_extractor",
        token=token,
    )


def _manifest_row_indices(manifest: pl.DataFrame) -> np.ndarray:
    if "gallery_index" in manifest.columns:
        return np.asarray(manifest.get_column("gallery_index"), dtype=np.int64)
    if "row_index" in manifest.columns:
        return np.asarray(manifest.get_column("row_index"), dtype=np.int64)
    return np.arange(manifest.height, dtype=np.int64)


def _resolve_row_batch_size(args: argparse.Namespace) -> int:
    if args.row_batch_size > 0:
        return args.row_batch_size
    return max(1, args.batch_size // max(args.n_crops, 1))


def _resolve_manifest_paths(manifest: pl.DataFrame, *, data_root: str) -> list[str]:
    resolved = (
        manifest.get_column("resolved_path").to_list()
        if "resolved_path" in manifest.columns
        else []
    )
    filepaths = manifest.get_column("filepath").to_list() if "filepath" in manifest.columns else []
    root = Path(data_root).expanduser() if data_root else None
    paths: list[str] = []
    remapped = 0
    for index in range(manifest.height):
        candidate = str(resolved[index]) if index < len(resolved) else ""
        candidate_path = Path(candidate) if candidate else None
        if candidate_path is not None and candidate_path.is_file():
            paths.append(str(candidate_path))
            continue
        if root is not None and index < len(filepaths):
            filepath = str(filepaths[index])
            fallback = (root / filepath).resolve(strict=False)
            paths.append(str(fallback))
            remapped += 1
            continue
        if candidate_path is not None:
            paths.append(str(candidate_path))
            continue
        raise ValueError(
            "Manifest row is missing a usable path. Provide --data-root when resolved_path "
            "points to a different machine."
        )
    if remapped > 0:
        print(
            f"[teacher-peft-tail] remapped resolved_path via data_root "
            f"rows={remapped}/{manifest.height} "
            f"data_root={root}",
            flush=True,
        )
    return paths


def _jsonable_label_config(config: LabelPropagationConfig) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "edge_top": config.edge_top,
        "reciprocal_top": config.reciprocal_top,
        "rank_top": config.rank_top,
        "iterations": config.iterations,
        "label_min_size": config.label_min_size,
        "label_max_size": config.label_max_size,
        "label_min_candidates": config.label_min_candidates,
        "shared_top": config.shared_top,
        "shared_min_count": config.shared_min_count,
        "reciprocal_bonus": config.reciprocal_bonus,
        "density_penalty": config.density_penalty,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument(
        "--data-root",
        default="",
        help=(
            "Optional dataset root used to rebuild absolute audio paths from the manifest "
            "filepath column when resolved_path points to a different machine."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hf-token-env", default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--encoder-backend", choices=("torch", "tensorrt"), default="torch")
    parser.add_argument("--tensorrt-engine-path", default="")
    parser.add_argument("--tensorrt-output-name", default="embedding")
    parser.add_argument("--tensorrt-profile-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--row-batch-size",
        type=int,
        default=0,
        help=(
            "Optional manifest-row batch size override. Default derives from batch-size / n-crops."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--search-batch-size", type=int, default=4096)
    parser.add_argument("--top-cache-k", type=int, default=300)
    parser.add_argument("--crop-seconds", type=float, default=6.0)
    parser.add_argument("--n-crops", type=int, default=3)
    parser.add_argument("--trim", dest="trim", action="store_true", default=True)
    parser.add_argument("--no-trim", dest="trim", action="store_false")
    parser.add_argument("--edge-top", type=int, default=10)
    parser.add_argument("--reciprocal-top", type=int, default=20)
    parser.add_argument("--rank-top", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--label-min-size", type=int, default=5)
    parser.add_argument("--label-max-size", type=int, default=120)
    parser.add_argument("--label-min-candidates", type=int, default=3)
    parser.add_argument("--shared-top", type=int, default=20)
    parser.add_argument("--shared-min-count", type=int, default=0)
    parser.add_argument("--reciprocal-bonus", type=float, default=0.03)
    parser.add_argument("--density-penalty", type=float, default=0.02)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument(
        "--embedding-cache-path",
        default="",
        help=(
            "Optional precomputed embedding .npy cache. When provided without "
            "--force-embeddings, skips audio/encoder extraction and recomputes "
            "only search, graph rerank, and submission writing."
        ),
    )
    parser.add_argument(
        "--copy-embedding-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy external embedding cache into the output directory for reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
