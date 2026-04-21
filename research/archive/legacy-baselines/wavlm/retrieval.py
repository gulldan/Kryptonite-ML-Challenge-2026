from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

SUPPORTED_EVAL_MODES = {"xvector_full_file", "xvector_chunk_mean"}


def repeat_or_pad(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    current = waveform.numel()
    if current >= target_len:
        return waveform[:target_len]
    if current == 0:
        return torch.zeros(target_len, dtype=torch.float32)
    repeats = int(np.ceil(target_len / current))
    return waveform.repeat(repeats)[:target_len]


def sequential_chunk_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_sec: float,
    max_load_len_sec: float,
) -> list[torch.Tensor]:
    waveform = waveform.flatten().float()
    if max_load_len_sec > 0:
        waveform = waveform[: int(round(sample_rate * max_load_len_sec))]
    chunk_len = int(round(sample_rate * chunk_sec))
    if chunk_len <= 0:
        raise ValueError("chunk_sec must be positive")
    if waveform.numel() <= chunk_len:
        return [waveform]
    chunk_count = max(1, int(np.ceil(waveform.numel() / chunk_len))) if waveform.numel() > 0 else 1
    padded = repeat_or_pad(waveform, chunk_count * chunk_len)
    return [padded[i * chunk_len : (i + 1) * chunk_len] for i in range(chunk_count)]


def load_audio_waveform(path_value: str, data_root: Path) -> tuple[torch.Tensor, int]:
    path = Path(path_value)
    if not path.is_absolute():
        path = data_root / path
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T).float()
    return waveform, int(sample_rate)


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, dim=-1)


def prepare_segments_for_mode(
    waveform: torch.Tensor,
    sample_rate: int,
    mode: str,
    chunk_sec: float,
    max_load_len_sec: float,
) -> list[torch.Tensor]:
    waveform = waveform.flatten().float()
    if max_load_len_sec > 0:
        waveform = waveform[: int(round(sample_rate * max_load_len_sec))]
    if mode == "xvector_full_file":
        return [waveform]
    if mode == "xvector_chunk_mean":
        return sequential_chunk_segments(
            waveform=waveform,
            sample_rate=sample_rate,
            chunk_sec=chunk_sec,
            max_load_len_sec=0.0,
        )
    raise ValueError(
        f"Unsupported evaluation mode: {mode}. Supported modes: {sorted(SUPPORTED_EVAL_MODES)}"
    )


@torch.no_grad()
def extract_embeddings(
    manifest,
    feature_extractor,
    model: torch.nn.Module,
    data_root: Path,
    sample_rate: int,
    mode: str,
    chunk_sec: float,
    max_load_len_sec: float,
    batch_size: int,
    device: torch.device,
    progress_every_rows: int = 0,
    progress_label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    if mode not in SUPPORTED_EVAL_MODES:
        raise ValueError(
            f"Unsupported evaluation mode: {mode}. Supported modes: {sorted(SUPPORTED_EVAL_MODES)}"
        )

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    model.eval()

    batched_segments: list[np.ndarray] = []
    batched_indices: list[int] = []
    pooled_embeddings: dict[int, list[np.ndarray]] = {}
    total_rows = int(len(manifest))
    started_at = time.perf_counter()
    segments_seen = 0

    def flush_batch() -> None:
        nonlocal batched_segments, batched_indices
        if not batched_segments:
            return
        inputs = feature_extractor(
            batched_segments,
            sampling_rate=sample_rate,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        batch_embeddings = model(**inputs).embeddings.detach().cpu().numpy()
        for idx, emb in zip(batched_indices, batch_embeddings, strict=True):
            pooled_embeddings.setdefault(idx, []).append(emb.astype(np.float32, copy=False))
        batched_segments = []
        batched_indices = []

    ordered_manifest = manifest
    if "dur" in manifest.columns:
        ordered_manifest = manifest.sort_values("dur", ascending=False, kind="stable")

    for position, (idx, row) in enumerate(ordered_manifest.iterrows(), start=1):
        waveform, sr = load_audio_waveform(str(row["path"]), data_root)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        waveform = waveform.mean(dim=0).float()
        segments = prepare_segments_for_mode(
            waveform,
            sample_rate=sample_rate,
            mode=mode,
            chunk_sec=chunk_sec,
            max_load_len_sec=max_load_len_sec,
        )
        segments_seen += len(segments)
        for segment in segments:
            batched_segments.append(segment.numpy())
            batched_indices.append(idx)
            if len(batched_segments) >= batch_size:
                flush_batch()
        if progress_every_rows > 0 and (
            position % progress_every_rows == 0 or position == total_rows
        ):
            elapsed = max(1e-6, time.perf_counter() - started_at)
            rows_per_sec = position / elapsed
            remaining_rows = max(0, total_rows - position)
            eta_sec = remaining_rows / rows_per_sec if rows_per_sec > 0 else 0.0
            label = progress_label or mode
            print(
                f"[extract_embeddings] {label} rows={position}/{total_rows} "
                f"segments={segments_seen} rate={rows_per_sec:.2f} rows/s "
                f"eta_min={eta_sec / 60.0:.1f}",
                flush=True,
            )
    flush_batch()

    for idx, row in manifest.iterrows():
        pooled = pooled_embeddings[idx]
        embeddings.append(np.mean(np.stack(pooled, axis=0), axis=0))
        labels.append(str(row["spk"]))

    return np.asarray(embeddings, dtype=np.float32), np.asarray(labels)


def topk_indices_from_embeddings(
    embeddings: np.ndarray,
    topk: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    if topk <= 0:
        raise ValueError("topk must be positive")
    emb = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    emb = normalize_embeddings(emb)
    n = int(emb.shape[0])
    if topk >= n:
        raise ValueError(f"topk={topk} must be smaller than number of rows={n}")
    rows: list[np.ndarray] = []
    neg_inf = torch.tensor(float("-inf"), device=device)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        chunk = emb[start:stop]
        scores = chunk @ emb.T
        local_diag = torch.arange(start, stop, device=device)
        scores[torch.arange(stop - start, device=device), local_diag] = neg_inf
        topk_idx = torch.topk(scores, k=topk, dim=1).indices
        rows.append(topk_idx.detach().cpu().numpy().astype(np.int64, copy=False))
    return np.concatenate(rows, axis=0)


def retrieval_metrics_from_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    ks: Iterable[int],
) -> dict[str, float]:
    ks = sorted(int(k) for k in ks)
    max_k = max(ks)
    _, encoded = np.unique(labels, return_inverse=True)
    encoded = np.asarray(encoded, dtype=np.int64)
    neighbour_labels = encoded[indices[:, :max_k]]
    matches = neighbour_labels == encoded[:, None]
    label_counts = np.bincount(encoded)
    positives_per_query = np.maximum(label_counts[encoded] - 1, 0)

    metrics: dict[str, float] = {}
    for k in ks:
        current = matches[:, :k]
        metrics[f"precision@{k}"] = float(current.mean())
        metrics[f"hit_rate@{k}"] = float(current.any(axis=1).mean())
        denom = np.minimum(positives_per_query, k)
        denom = np.where(denom > 0, denom, 1)
        metrics[f"recall@{k}"] = float((current.sum(axis=1) / denom).mean())

    reciprocal_ranks = np.where(matches, 1.0 / (np.arange(max_k, dtype=np.float32) + 1.0), 0.0)
    discounts = 1.0 / np.log2(np.arange(max_k, dtype=np.float32) + 2.0)
    dcg = (matches.astype(np.float32) * discounts).sum(axis=1)
    ideal_lengths = np.minimum(positives_per_query, max_k)
    idcg = np.zeros_like(dcg, dtype=np.float32)
    for length in np.unique(ideal_lengths):
        if length <= 0:
            continue
        idcg[ideal_lengths == length] = discounts[:length].sum()
    ndcg = np.divide(dcg, np.where(idcg > 0, idcg, 1.0), out=np.zeros_like(dcg), where=idcg > 0)
    metrics[f"ndcg@{max_k}"] = float(ndcg.mean())
    metrics[f"mrr@{max_k}"] = float(reciprocal_ranks.max(axis=1).mean())
    return metrics


def retrieval_metrics_from_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: Iterable[int],
    chunk_size: int,
    device: torch.device,
) -> dict[str, float]:
    n = int(len(labels))
    ks = sorted(int(k) for k in ks if int(k) < n)
    if not ks:
        raise ValueError(f"At least one K must be smaller than number of rows={n}")
    indices = topk_indices_from_embeddings(
        embeddings=embeddings,
        topk=max(ks),
        chunk_size=chunk_size,
        device=device,
    )
    return retrieval_metrics_from_indices(indices=indices, labels=labels, ks=ks)
