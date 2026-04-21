from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset


def repeat_or_pad(
    waveform: torch.Tensor, target_len: int, pad_mode: str = "repeat"
) -> torch.Tensor:
    current = waveform.numel()
    if current >= target_len:
        return waveform[:target_len]
    if current == 0:
        return torch.zeros(target_len, dtype=torch.float32)
    if pad_mode == "repeat":
        repeats = int(np.ceil(target_len / current))
        waveform = waveform.repeat(repeats)[:target_len]
        return waveform
    return torch.nn.functional.pad(waveform, (0, target_len - current))


def random_crop_or_pad(
    waveform: torch.Tensor, sample_rate: int, chunk_sec: float, pad_mode: str = "repeat"
) -> torch.Tensor:
    chunk_len = int(round(sample_rate * chunk_sec))
    waveform = waveform.flatten().float()
    if waveform.numel() >= chunk_len:
        max_start = waveform.numel() - chunk_len
        start = 0 if max_start == 0 else int(torch.randint(0, max_start + 1, (1,)).item())
        return waveform[start : start + chunk_len]
    return repeat_or_pad(waveform, chunk_len, pad_mode=pad_mode)


def even_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_sec: float,
    segment_count: int,
    pad_mode: str = "repeat",
) -> list[torch.Tensor]:
    waveform = waveform.flatten().float()
    chunk_len = int(round(sample_rate * chunk_sec))
    if waveform.numel() <= chunk_len:
        return [repeat_or_pad(waveform, chunk_len, pad_mode=pad_mode)]

    max_start = waveform.numel() - chunk_len
    if segment_count <= 1:
        offsets = [max_start // 2]
    else:
        offsets = np.linspace(0, max_start, num=segment_count, dtype=int).tolist()
    return [waveform[start : start + chunk_len] for start in offsets]


def waveform_to_fbank(waveform: torch.Tensor, sample_rate: int, n_mels: int) -> torch.Tensor:
    waveform = waveform.flatten().float().unsqueeze(0)
    features = kaldi.fbank(
        waveform,
        num_mel_bins=int(n_mels),
        sample_frequency=float(sample_rate),
        dither=0.0,
    )
    return features - features.mean(dim=0, keepdim=True)


def add_gaussian_noise(
    waveform: torch.Tensor,
    snr_db_min: float,
    snr_db_max: float,
) -> torch.Tensor:
    snr_db = random.uniform(float(snr_db_min), float(snr_db_max))
    signal_rms = waveform.pow(2).mean().sqrt().clamp(min=1e-6)
    noise = torch.randn_like(waveform)
    noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-6)
    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    return waveform + noise * (target_noise_rms / noise_rms)


def apply_random_reverb(
    waveform: torch.Tensor,
    sample_rate: int,
    impulse_ms_min: float,
    impulse_ms_max: float,
    decay_power_min: float,
    decay_power_max: float,
    dry_wet_min: float,
    dry_wet_max: float,
) -> torch.Tensor:
    impulse_len = max(
        8,
        int(
            round(
                sample_rate * random.uniform(float(impulse_ms_min), float(impulse_ms_max)) / 1000.0
            )
        ),
    )
    t = torch.arange(impulse_len, dtype=waveform.dtype, device=waveform.device)
    decay_power = random.uniform(float(decay_power_min), float(decay_power_max))
    decay = torch.exp(-t / max(1.0, impulse_len / decay_power))
    rir = torch.randn(impulse_len, dtype=waveform.dtype, device=waveform.device) * decay
    rir[0] = rir[0] + 1.0
    rir = rir / rir.abs().sum().clamp(min=1e-6)
    reverbed = F.conv1d(
        waveform.view(1, 1, -1),
        rir.flip(0).view(1, 1, -1),
        padding=impulse_len - 1,
    ).view(-1)[: waveform.numel()]
    dry_wet = random.uniform(float(dry_wet_min), float(dry_wet_max))
    return (1.0 - dry_wet) * waveform + dry_wet * reverbed


def apply_band_limit(
    waveform: torch.Tensor,
    sample_rate: int,
    lowpass_hz_min: float,
    lowpass_hz_max: float,
    highpass_hz_min: float,
    highpass_hz_max: float,
) -> torch.Tensor:
    lowpass_hz = random.uniform(float(lowpass_hz_min), float(lowpass_hz_max))
    highpass_hz = min(
        lowpass_hz - 50.0,
        random.uniform(float(highpass_hz_min), float(highpass_hz_max)),
    )
    filtered = torchaudio.functional.lowpass_biquad(
        waveform.unsqueeze(0),
        sample_rate=sample_rate,
        cutoff_freq=max(200.0, lowpass_hz),
    ).squeeze(0)
    if highpass_hz > 20.0:
        filtered = torchaudio.functional.highpass_biquad(
            filtered.unsqueeze(0),
            sample_rate=sample_rate,
            cutoff_freq=highpass_hz,
        ).squeeze(0)
    return filtered


def apply_silence_shift(
    waveform: torch.Tensor, sample_rate: int, max_shift_sec: float
) -> torch.Tensor:
    max_shift_samples = min(int(round(float(max_shift_sec) * sample_rate)), waveform.numel() // 3)
    if max_shift_samples <= 0:
        return waveform
    shift = random.randint(-max_shift_samples, max_shift_samples)
    if shift == 0:
        return waveform
    output = torch.zeros_like(waveform)
    if shift > 0:
        output[shift:] = waveform[:-shift]
    else:
        output[:shift] = waveform[-shift:]
    return output


class WaveformAugmenter:
    def __init__(self, config: dict | None, sample_rate: int) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.sample_rate = int(sample_rate)
        self.max_augments = max(1, int(cfg.get("max_augments_per_sample", 1)))
        self.noise_cfg = dict(cfg.get("noise", {}))
        self.reverb_cfg = dict(cfg.get("reverb", {}))
        self.band_cfg = dict(cfg.get("band_limit", {}))
        self.silence_cfg = dict(cfg.get("silence_shift", {}))

    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return waveform

        candidates: list[str] = []
        if random.random() < float(self.noise_cfg.get("probability", 0.0)):
            candidates.append("noise")
        if random.random() < float(self.reverb_cfg.get("probability", 0.0)):
            candidates.append("reverb")
        if random.random() < float(self.band_cfg.get("probability", 0.0)):
            candidates.append("band_limit")
        if random.random() < float(self.silence_cfg.get("probability", 0.0)):
            candidates.append("silence_shift")

        if not candidates:
            return waveform

        random.shuffle(candidates)
        augmented = waveform
        for name in candidates[: self.max_augments]:
            if name == "noise":
                augmented = add_gaussian_noise(
                    augmented,
                    snr_db_min=float(self.noise_cfg.get("snr_db_min", 10.0)),
                    snr_db_max=float(self.noise_cfg.get("snr_db_max", 20.0)),
                )
            elif name == "reverb":
                augmented = apply_random_reverb(
                    augmented,
                    sample_rate=self.sample_rate,
                    impulse_ms_min=float(self.reverb_cfg.get("impulse_ms_min", 30.0)),
                    impulse_ms_max=float(self.reverb_cfg.get("impulse_ms_max", 120.0)),
                    decay_power_min=float(self.reverb_cfg.get("decay_power_min", 2.0)),
                    decay_power_max=float(self.reverb_cfg.get("decay_power_max", 6.0)),
                    dry_wet_min=float(self.reverb_cfg.get("dry_wet_min", 0.15)),
                    dry_wet_max=float(self.reverb_cfg.get("dry_wet_max", 0.35)),
                )
            elif name == "band_limit":
                augmented = apply_band_limit(
                    augmented,
                    sample_rate=self.sample_rate,
                    lowpass_hz_min=float(self.band_cfg.get("lowpass_hz_min", 2200.0)),
                    lowpass_hz_max=float(self.band_cfg.get("lowpass_hz_max", 4200.0)),
                    highpass_hz_min=float(self.band_cfg.get("highpass_hz_min", 80.0)),
                    highpass_hz_max=float(self.band_cfg.get("highpass_hz_max", 300.0)),
                )
            elif name == "silence_shift":
                augmented = apply_silence_shift(
                    augmented,
                    sample_rate=self.sample_rate,
                    max_shift_sec=float(self.silence_cfg.get("max_shift_sec", 0.35)),
                )
        return augmented


def collate_features(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    feats, labels = zip(*batch, strict=False)
    max_frames = max(feat.shape[0] for feat in feats)
    padded = []
    for feat in feats:
        if feat.shape[0] < max_frames:
            feat = torch.nn.functional.pad(feat, (0, 0, 0, max_frames - feat.shape[0]))
        padded.append(feat)
    return torch.stack(padded, dim=0), torch.tensor(labels, dtype=torch.long)


class ManifestTrainDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        speaker_to_index: dict[str, int],
        data_root: Path,
        sample_rate: int,
        n_mels: int,
        chunk_sec: float,
        pad_mode: str = "repeat",
        augmentations: dict | None = None,
    ) -> None:
        import pandas as pd

        manifest = pd.read_csv(manifest_path, usecols=["path", "spk"])
        self.paths = [resolve_audio_path(path, data_root) for path in manifest["path"].tolist()]
        self.labels = [speaker_to_index[spk] for spk in manifest["spk"].tolist()]
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.chunk_sec = chunk_sec
        self.pad_mode = pad_mode
        self.augmenter = WaveformAugmenter(augmentations, sample_rate=sample_rate)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        waveform, sr = load_audio_waveform(self.paths[index], None)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)
        chunk = random_crop_or_pad(
            waveform, self.sample_rate, self.chunk_sec, pad_mode=self.pad_mode
        )
        chunk = self.augmenter.apply(chunk)
        peak = chunk.abs().max()
        if torch.isfinite(peak) and peak > 1.0:
            chunk = chunk / peak
        feat = waveform_to_fbank(chunk, self.sample_rate, self.n_mels)
        return feat, self.labels[index]


def resolve_audio_path(raw_path: str, data_root: Path | None) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    if data_root is None:
        raise ValueError("data_root must be provided for relative paths")
    return str((data_root / path).resolve())


def load_audio_waveform(raw_path: str, data_root: Path | None) -> tuple[torch.Tensor, int]:
    samples, sample_rate = sf.read(
        resolve_audio_path(raw_path, data_root), dtype="float32", always_2d=True
    )
    waveform = torch.from_numpy(samples.T.copy())
    return waveform, int(sample_rate)


@torch.no_grad()
def extract_embeddings(
    manifest,
    model: torch.nn.Module,
    data_root: Path,
    sample_rate: int,
    n_mels: int,
    mode: str,
    eval_chunk_sec: float,
    segment_count: int,
    long_file_threshold_sec: float,
    batch_size: int,
    device: torch.device,
    pad_mode: str = "repeat",
) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    model.eval()

    batched_features: list[torch.Tensor] = []
    batched_indices: list[int] = []
    pending_segments: list[list[int]] = []
    pooled_embeddings: dict[int, list[np.ndarray]] = {}
    direct_embeddings: dict[int, np.ndarray] = {}

    def flush_batch() -> None:
        nonlocal batched_features, batched_indices, pending_segments
        if not batched_features:
            return
        max_frames = max(feat.shape[0] for feat in batched_features)
        padded = []
        for feat in batched_features:
            if feat.shape[0] < max_frames:
                feat = torch.nn.functional.pad(feat, (0, 0, 0, max_frames - feat.shape[0]))
            padded.append(feat)
        batch = torch.stack(padded, dim=0).to(device)
        batch_emb = model(batch).detach().cpu().numpy()
        for idx, emb in zip(batched_indices, batch_emb, strict=False):
            pooled_embeddings.setdefault(idx, []).append(emb)
        batched_features = []
        batched_indices = []
        pending_segments = []

    for idx, row in manifest.iterrows():
        waveform, sr = load_audio_waveform(row["path"], data_root)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        waveform = waveform.mean(dim=0)
        duration = float(row["dur"])

        if mode == "full_file":
            flush_batch()
            feat = waveform_to_fbank(waveform, sample_rate, n_mels).unsqueeze(0).to(device)
            direct_embeddings[idx] = model(feat).detach().cpu().numpy()[0]
            continue

        if mode == "segment_mean" and duration > long_file_threshold_sec:
            segments = even_segments(
                waveform,
                sample_rate=sample_rate,
                chunk_sec=eval_chunk_sec,
                segment_count=segment_count,
                pad_mode=pad_mode,
            )
        else:
            segments = even_segments(
                waveform,
                sample_rate=sample_rate,
                chunk_sec=eval_chunk_sec,
                segment_count=1,
                pad_mode=pad_mode,
            )
        for segment in segments:
            batched_features.append(waveform_to_fbank(segment, sample_rate, n_mels))
            batched_indices.append(idx)
            if len(batched_features) >= batch_size:
                flush_batch()

    flush_batch()

    for idx, row in manifest.iterrows():
        if idx in direct_embeddings:
            embeddings.append(direct_embeddings[idx])
            labels.append(row["spk"])
            continue
        pooled = pooled_embeddings[idx]
        embeddings.append(np.mean(np.stack(pooled, axis=0), axis=0))
        labels.append(row["spk"])

    emb = np.asarray(embeddings, dtype=np.float32)
    lab = np.asarray(labels)
    return emb, lab


def precision_at_ks_from_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: Iterable[int],
    chunk_size: int,
    device: torch.device,
) -> dict[str, float]:
    metrics = retrieval_metrics_from_embeddings(
        embeddings=embeddings,
        labels=labels,
        ks=ks,
        chunk_size=chunk_size,
        device=device,
    )
    return {key: value for key, value in metrics.items() if key.startswith("precision@")}


def topk_indices_from_embeddings(
    embeddings: np.ndarray,
    topk: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    if topk <= 0:
        raise ValueError("topk must be positive")

    emb = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    emb = torch.nn.functional.normalize(emb, dim=1)
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
    if indices.ndim != 2:
        raise ValueError("indices must be 2D")
    if indices.shape[1] < max_k:
        raise ValueError(f"indices has only {indices.shape[1]} columns, but max_k={max_k}")

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
    ks = sorted(int(k) for k in ks)
    indices = topk_indices_from_embeddings(
        embeddings=embeddings,
        topk=max(ks),
        chunk_size=chunk_size,
        device=device,
    )
    return retrieval_metrics_from_indices(indices=indices, labels=labels, ks=ks)
