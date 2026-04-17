"""Run official 3D-Speaker ERes2Net embeddings and retrieval tails."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import soundfile as sf

from kryptonite.eda.community import (
    LabelPropagationConfig,
    exact_topk,
    label_propagation_rerank,
    write_submission,
)
from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission
from kryptonite.serve.tensorrt_generic import MultiInputTensorRTEngineRunner


class _TorchFeatureEncoder:
    def __init__(self, *, model: Any, device: Any, precision: str) -> None:
        self._model = model
        self._device = device
        self._precision = precision

    def __call__(self, features: Any) -> np.ndarray:
        import torch

        with torch.amp.autocast(
            self._device.type,
            enabled=self._device.type == "cuda" and self._precision == "bf16",
            dtype=torch.bfloat16,
        ):
            values_tensor = self._model(features)
        return values_tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)


class _TensorRTFeatureEncoder:
    def __init__(
        self,
        *,
        engine_path: Path,
        input_name: str,
        output_name: str,
        profile_index: int,
    ) -> None:
        self._input_name = input_name
        self._profile_index = profile_index
        self._runner = MultiInputTensorRTEngineRunner(
            engine_path=engine_path,
            output_name=output_name,
        )

    def __call__(self, features: Any) -> np.ndarray:
        output = self._runner.run(
            {self._input_name: features},
            profile_index=self._profile_index,
        )
        return output.detach().float().cpu().numpy().astype(np.float32, copy=False)


class _OfficialFBank:
    def __init__(self, *, n_mels: int, sample_rate_hz: int, mean_nor: bool) -> None:
        self._n_mels = n_mels
        self._sample_rate_hz = sample_rate_hz
        self._mean_nor = mean_nor

    def __call__(self, waveform: Any) -> Any:
        import torchaudio.compliance.kaldi as kaldi

        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform[0, :].unsqueeze(0)
        if len(waveform.shape) != 2 or waveform.shape[0] != 1:
            raise ValueError(f"Expected mono waveform tensor, got shape={tuple(waveform.shape)}")
        features = kaldi.fbank(
            waveform,
            num_mel_bins=self._n_mels,
            sample_frequency=self._sample_rate_hz,
            dither=0,
        )
        if self._mean_nor:
            features = features - features.mean(0, keepdim=True)
        return features


class _EResEvalDataset:
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        data_root: Path,
        sample_rate_hz: int,
        max_load_seconds: float,
    ) -> None:
        self._rows = rows
        self._data_root = data_root
        self._sample_rate_hz = sample_rate_hz
        self._max_load_seconds = max_load_seconds

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[int, np.ndarray]:
        row = self._rows[index]
        row_index = int(row.get("gallery_index", row.get("row_index", index)))
        waveform = _load_waveform(
            row=row,
            data_root=self._data_root,
            sample_rate_hz=self._sample_rate_hz,
            max_load_seconds=self._max_load_seconds,
        )
        return row_index, waveform


class _EResEvalCollator:
    def __init__(
        self,
        *,
        sample_rate_hz: int,
        chunk_samples: int,
    ) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._chunk_samples = chunk_samples
        self._fbank = _OfficialFBank(n_mels=80, sample_rate_hz=sample_rate_hz, mean_nor=True)

    def __call__(self, items: list[tuple[int, np.ndarray]]) -> _EResFeatureBatch:
        import torch

        features: list[Any] = []
        owners: list[int] = []
        for row_index, waveform in items:
            for chunk in _chunk_waveform(waveform, chunk_samples=self._chunk_samples):
                features.append(self._fbank(torch.from_numpy(chunk)))
                owners.append(row_index)
        return _EResFeatureBatch(features=features, owners=owners, row_count=len(items))


class _EResFeatureBatch:
    def __init__(self, *, features: list[Any], owners: list[int], row_count: int) -> None:
        self.features = features
        self.owners = owners
        self.row_count = row_count


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest_csv)
    print(
        f"[official-3dspeaker-eres2net] start experiment={args.experiment_id} "
        f"rows={manifest.height} output_dir={output_dir}",
        flush=True,
    )
    total_started = time.perf_counter()
    started = time.perf_counter()
    embeddings = _load_or_extract_embeddings(args, manifest, output_dir)
    embedding_s = time.perf_counter() - started

    print("[official-3dspeaker-eres2net] exact_topk start", flush=True)
    started = time.perf_counter()
    indices, scores = exact_topk(
        embeddings,
        top_k=args.top_cache_k,
        batch_size=args.search_batch_size,
        device=args.search_device,
    )
    search_s = time.perf_counter() - started
    np.save(output_dir / f"indices_{args.experiment_id}_top{args.top_cache_k}.npy", indices)
    np.save(output_dir / f"scores_{args.experiment_id}_top{args.top_cache_k}.npy", scores)

    rows: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "encoder_backend": args.encoder_backend,
        "checkpoint_path": args.checkpoint_path,
        "speakerlab_root": args.speakerlab_root,
        "tensorrt_engine_path": args.tensorrt_engine_path,
        "tensorrt_profile_index": args.tensorrt_profile_index,
        "manifest_csv": args.manifest_csv,
        "data_root": args.data_root,
        "embedding_s": round(embedding_s, 6),
        "search_s": round(search_s, 6),
        "chunk_seconds": args.chunk_seconds,
        "max_load_seconds": args.max_load_seconds,
        "batch_size": args.batch_size,
        "exact_top1_score_mean": float(scores[:, 0].mean()),
        "exact_top10_mean_score_mean": float(scores[:, :10].mean()),
        "exact_indegree_gini_10": gini(
            np.bincount(indices[:, :10].ravel(), minlength=manifest.height)
        ),
        "exact_indegree_max_10": int(
            np.bincount(indices[:, :10].ravel(), minlength=manifest.height).max()
        ),
    }
    if args.template_csv:
        exact_submission_path = output_dir / f"submission_{args.experiment_id}_exact.csv"
        started = time.perf_counter()
        write_submission(
            manifest=manifest,
            top_indices=indices[:, :10],
            output_csv=exact_submission_path,
        )
        exact_submit_write_s = time.perf_counter() - started
        started = time.perf_counter()
        exact_validation = validate_submission(
            template_csv=Path(args.template_csv),
            submission_csv=exact_submission_path,
        )
        exact_validation_s = time.perf_counter() - started
        (output_dir / f"submission_{args.experiment_id}_exact_validation.json").write_text(
            json.dumps(exact_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.update(
            {
                "exact_submission_path": str(exact_submission_path),
                "exact_validator_passed": bool(exact_validation["passed"]),
                "exact_submit_write_s": round(exact_submit_write_s, 6),
                "exact_validation_s": round(exact_validation_s, 6),
                "exact_submit_generation_s": round(
                    embedding_s + search_s + exact_submit_write_s,
                    6,
                ),
            }
        )

    print("[official-3dspeaker-eres2net] label_propagation_rerank start", flush=True)
    config = LabelPropagationConfig(
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
    started = time.perf_counter()
    top_indices, top_scores, meta = label_propagation_rerank(
        indices=indices,
        scores=scores,
        config=config,
        top_k=10,
    )
    rerank_s = time.perf_counter() - started
    rows.update(
        {
            "c4_rerank_s": round(rerank_s, 6),
            "c4_top1_score_mean": float(top_scores[:, 0].mean()),
            "c4_top10_mean_score_mean": float(top_scores.mean()),
            "c4_indegree_gini_10": gini(
                np.bincount(top_indices.ravel(), minlength=manifest.height)
            ),
            "c4_indegree_max_10": int(
                np.bincount(top_indices.ravel(), minlength=manifest.height).max()
            ),
            **meta,
        }
    )
    if args.template_csv:
        c4_submission_path = output_dir / f"submission_{args.experiment_id}_c4.csv"
        started = time.perf_counter()
        write_submission(manifest=manifest, top_indices=top_indices, output_csv=c4_submission_path)
        c4_submit_write_s = time.perf_counter() - started
        started = time.perf_counter()
        c4_validation = validate_submission(
            template_csv=Path(args.template_csv),
            submission_csv=c4_submission_path,
        )
        c4_validation_s = time.perf_counter() - started
        (output_dir / f"submission_{args.experiment_id}_c4_validation.json").write_text(
            json.dumps(c4_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.update(
            {
                "c4_submission_path": str(c4_submission_path),
                "c4_validator_passed": bool(c4_validation["passed"]),
                "c4_submit_write_s": round(c4_submit_write_s, 6),
                "c4_validation_s": round(c4_validation_s, 6),
                "c4_submit_generation_s": round(
                    embedding_s + search_s + rerank_s + c4_submit_write_s,
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
) -> np.ndarray:
    output_path = output_dir / f"embeddings_{args.experiment_id}.npy"
    if output_path.is_file() and not args.force_embeddings:
        print(
            f"[official-3dspeaker-eres2net] load cached embeddings path={output_path}", flush=True
        )
        return np.load(output_path)

    import torch

    speakerlab_root = Path(args.speakerlab_root)
    if str(speakerlab_root) not in sys.path:
        sys.path.insert(0, str(speakerlab_root))
    eres_module = importlib.import_module("speakerlab.models.eres2net.ERes2Net")
    from torch.utils.data import DataLoader

    device = torch.device(args.device)
    encoder = _build_feature_encoder(
        args=args,
        torch=torch,
        eres_module=eres_module,
        device=device,
    )

    sums: dict[int, list[np.ndarray]] = {}
    batch_features: list[Any] = []
    batch_owners: list[int] = []
    dataset = _EResEvalDataset(
        rows=list(manifest.iter_rows(named=True)),
        data_root=Path(args.data_root),
        sample_rate_hz=args.sample_rate_hz,
        max_load_seconds=args.max_load_seconds,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.row_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=_EResEvalCollator(
            sample_rate_hz=args.sample_rate_hz,
            chunk_samples=int(round(args.chunk_seconds * args.sample_rate_hz)),
        ),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    started_at = time.perf_counter()
    log_every_rows = max(1, manifest.height // 20)
    next_log_row = 1
    completed_rows = 0
    with torch.no_grad():
        for batch in loader:
            batch_features.extend(batch.features)
            batch_owners.extend(batch.owners)
            while len(batch_features) >= args.batch_size:
                _flush_embeddings(
                    encoder=encoder,
                    batch_features=batch_features[: args.batch_size],
                    batch_owners=batch_owners[: args.batch_size],
                    sums=sums,
                    device=device,
                )
                del batch_features[: args.batch_size]
                del batch_owners[: args.batch_size]
            completed_rows += batch.row_count
            if (
                completed_rows == 1
                or completed_rows >= next_log_row
                or completed_rows == manifest.height
            ):
                elapsed_s = max(time.perf_counter() - started_at, 1e-9)
                print(
                    "[official-3dspeaker-eres2net] extract "
                    f"rows={completed_rows}/{manifest.height} "
                    f"pct={100.0 * completed_rows / manifest.height:.1f} "
                    f"rows_per_s={completed_rows / elapsed_s:.1f} elapsed_s={elapsed_s:.1f}",
                    flush=True,
                )
                next_log_row = completed_rows + log_every_rows
        if batch_features:
            _flush_embeddings(
                encoder=encoder,
                batch_features=batch_features,
                batch_owners=batch_owners,
                sums=sums,
                device=device,
            )

    embeddings = np.empty(
        (manifest.height, next(iter(sums.values()))[0].shape[0]), dtype=np.float32
    )
    for index in range(manifest.height):
        embeddings[index] = np.mean(np.stack(sums[index], axis=0), axis=0)
    np.save(output_path, embeddings)
    return embeddings


def _flush_embeddings(
    encoder: Any,
    batch_features: list[Any],
    batch_owners: list[int],
    sums: dict[int, list[np.ndarray]],
    device: Any,
) -> None:
    import torch

    features = torch.stack(batch_features).to(device)
    values = encoder(features)
    for owner, embedding in zip(batch_owners, values, strict=True):
        sums.setdefault(owner, []).append(embedding.astype(np.float32, copy=False))


def _build_feature_encoder(
    *,
    args: argparse.Namespace,
    torch: Any,
    eres_module: Any,
    device: Any,
) -> Any:
    if args.encoder_backend == "torch":
        model = eres_module.ERes2Net(feat_dim=80, embedding_size=512, m_channels=64)
        payload = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(_extract_model_state_dict(payload))
        model = model.to(device)
        model.eval()
        return _TorchFeatureEncoder(model=model, device=device, precision=args.precision)
    if args.encoder_backend == "tensorrt":
        if not args.tensorrt_engine_path:
            raise ValueError("--tensorrt-engine-path is required for --encoder-backend=tensorrt.")
        return _TensorRTFeatureEncoder(
            engine_path=Path(args.tensorrt_engine_path),
            input_name=args.tensorrt_input_name,
            output_name=args.tensorrt_output_name,
            profile_index=args.tensorrt_profile_index,
        )
    raise ValueError(f"Unsupported encoder backend: {args.encoder_backend!r}")


def _load_waveform(
    *,
    row: dict[str, Any],
    data_root: Path,
    sample_rate_hz: int,
    max_load_seconds: float,
) -> np.ndarray:
    path = Path(str(row.get("resolved_path") or ""))
    if not path.is_file():
        path = data_root / str(row["filepath"])
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = waveform.mean(axis=1, dtype=np.float32)
    if sample_rate != sample_rate_hz:
        import torch
        import torchaudio.functional as F

        tensor = torch.from_numpy(mono).unsqueeze(0)
        mono = F.resample(tensor, sample_rate, sample_rate_hz).squeeze(0).numpy()
    max_samples = int(round(max_load_seconds * sample_rate_hz))
    return mono[:max_samples].astype(np.float32, copy=False)


def _chunk_waveform(waveform: np.ndarray, *, chunk_samples: int) -> list[np.ndarray]:
    if waveform.size == 0:
        waveform = np.zeros(chunk_samples, dtype=np.float32)
    chunk_count = int(np.ceil(waveform.size / chunk_samples))
    target_samples = max(chunk_samples, chunk_count * chunk_samples)
    if waveform.size < target_samples:
        repeats = int(np.ceil(target_samples / max(1, waveform.size)))
        waveform = np.tile(waveform, repeats)[:target_samples]
    return [
        waveform[index * chunk_samples : (index + 1) * chunk_samples].astype(np.float32, copy=False)
        for index in range(target_samples // chunk_samples)
    ]


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


def _extract_model_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state dict payload, got {type(state_dict)!r}.")
    return state_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--speakerlab-root", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--data-root", default="datasets/Для участников")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--encoder-backend", choices=("torch", "tensorrt"), default="torch")
    parser.add_argument("--tensorrt-engine-path", default="")
    parser.add_argument("--tensorrt-input-name", default="features")
    parser.add_argument("--tensorrt-output-name", default="embedding")
    parser.add_argument("--tensorrt-profile-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--row-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--search-batch-size", type=int, default=2048)
    parser.add_argument("--top-cache-k", type=int, default=200)
    parser.add_argument("--sample-rate-hz", type=int, default=16_000)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--max-load-seconds", type=float, default=90.0)
    parser.add_argument("--force-embeddings", action="store_true")
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
