import argparse
import json
import os
import random
from collections.abc import Sequence
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from src.dataset import SpeakerDataset
from src.metrics import precision_at_k
from src.model import ECAPASpeakerId
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / max(1, total)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    desc: str = "train",
) -> tuple[float, float]:
    model.train(True)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    total = len(loader)
    pbar = tqdm(loader, total=total, desc=desc, dynamic_ncols=True, leave=False)
    for batch_idx, (wave, label) in enumerate(pbar, start=1):
        wave = wave.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(wave)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        acc = accuracy_from_logits(logits.detach(), label)
        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

        if batch_idx == 1 or batch_idx % 10 == 0 or (total is not None and batch_idx == total):
            avg_loss = total_loss / max(1, total_batches)
            avg_acc = total_acc / max(1, total_batches)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc:.4f}",
                avg_loss=f"{avg_loss:.4f}",
                avg_acc=f"{avg_acc:.4f}",
            )

    avg_loss = total_loss / max(1, total_batches)
    avg_acc = total_acc / max(1, total_batches)
    return avg_loss, avg_acc


@torch.no_grad()
def validate_retrieval(
    model: ECAPASpeakerId, loader, device: str, ks: Sequence[int] = (1, 10, 50)
) -> dict[str, float]:
    model.train(False)
    all_embeddings = []
    all_labels = []
    for wave, label in loader:
        wave = wave.to(device, non_blocking=True)
        emb = model.extract_embeddings(wave)
        all_embeddings.append(emb.cpu())
        all_labels.append(label)
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    sim_metrics = precision_at_k(embeddings.numpy(), labels.numpy(), ks=ks)
    return sim_metrics


def format_metrics(metrics: dict[str, float]) -> str:
    keys = sorted(
        metrics.keys(),
        key=lambda x: (x.split("@")[0], int(x.split("@")[1]) if "@" in x else 0),
    )
    return ", ".join([f"{k} {metrics[k]:.4f}" for k in keys])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def split_data(
    train_csv: str,
    exp_dir: str,
    train_ratio: float,
    seed: int = 42,
    min_val_utts: int = 11,
    speaker_id_col: str = "speaker_id",
):
    df = pd.read_csv(train_csv)
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if min_val_utts <= 0:
        raise ValueError("min_val_utts must be positive.")
    if speaker_id_col not in df.columns:
        raise ValueError(f"Column '{speaker_id_col}' not found in {train_csv}.")

    speaker_counts = df.groupby(speaker_id_col).size().sort_index()
    eligible_speakers = speaker_counts[speaker_counts >= min_val_utts]
    if eligible_speakers.empty:
        raise ValueError(
            f"No speakers have enough utterances for validation: min_val_utts={min_val_utts}."
        )

    rng = np.random.default_rng(seed)
    target_val_rows = max(1, int(round(len(df) * (1.0 - train_ratio))))
    shuffled_speakers = rng.permutation(eligible_speakers.index.to_numpy())
    val_speakers = []
    val_rows = 0
    for speaker in shuffled_speakers:
        val_speakers.append(speaker)
        val_rows += int(speaker_counts.loc[speaker])
        if val_rows >= target_val_rows:
            break

    val_mask = df[speaker_id_col].isin(val_speakers)
    train_df = df.loc[~val_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError(
            "Speaker-disjoint split produced an empty split. "
            f"train_rows={len(train_df)} val_rows={len(val_df)}"
        )

    train_split_path = os.path.join(exp_dir, "train_split.csv")
    val_split_path = os.path.join(exp_dir, "val_split.csv")

    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)

    print(f"[INFO] Train split saved: {len(train_df)} rows. Path: {train_split_path}.")
    print(f"[INFO] Validation split saved: {len(val_df)} rows. Path: {val_split_path}.")
    print(
        "[INFO] Speaker-disjoint split: "
        f"train_speakers={train_df[speaker_id_col].nunique()} "
        f"val_speakers={val_df[speaker_id_col].nunique()} "
        f"train_ratio={train_ratio:.3f} min_val_utts={min_val_utts} seed={seed}"
    )

    return train_split_path, val_split_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    os.makedirs(cfg.get("exp_dir"), exist_ok=True)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    train_split_path, val_split_path = split_data(
        cfg["train_csv"],
        cfg["exp_dir"],
        train_ratio=float(cfg.get("train_ratio", cfg.get("split_ratio", 0.8))),
        seed=seed,
        min_val_utts=int(cfg.get("min_val_utts", 11)),
        speaker_id_col=cfg.get("speaker_id_col", "speaker_id"),
    )

    train_ds = SpeakerDataset(
        csv_path=train_split_path,
        sample_rate=cfg["sample_rate"],
        chunk_seconds=cfg["train_chunk_seconds"],
        is_train=True,
        base_dir=cfg["data_base_dir"],
        filepath_col=cfg.get("filepath_col", "filepath"),
        speaker_id_col=cfg.get("speaker_id_col", "speaker_id"),
    )
    val_ds = SpeakerDataset(
        csv_path=val_split_path,
        sample_rate=cfg["sample_rate"],
        chunk_seconds=cfg["val_chunk_seconds"],
        is_train=False,
        base_dir=cfg["data_base_dir"],
        filepath_col=cfg.get("filepath_col", "filepath"),
        speaker_id_col=cfg.get("speaker_id_col", "speaker_id"),
    )

    print("[INFO] ===== Dataset summary =====")
    print(f"[INFO] train_csv: {cfg['train_csv']}")
    print(f"[INFO] train speakers: {train_ds.num_speakers} | train audio files: {len(train_ds)}")
    print(f"[INFO] val speakers:   {val_ds.num_speakers} | val audio files:   {len(val_ds)}")
    print(
        f"[INFO] sample_rate={cfg['sample_rate']} "
        f"train_chunk_seconds={cfg['train_chunk_seconds']} "
        f"val_chunk_seconds={cfg['val_chunk_seconds']}"
    )
    print(
        f"[INFO] batch_size={cfg['batch_size']} num_workers={cfg['num_workers']} "
        f"device={device} seed={seed}"
    )
    print("[INFO] ===========================")

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = ECAPASpeakerId(
        cfg["sample_rate"],
        cfg["n_fft"],
        cfg["hop_length"],
        cfg["n_mels"],
        embed_dim=cfg.get("embed_dim", 192),
        num_classes=train_ds.num_speakers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    best_metric = 0.0
    metric_key = "precision@10"
    model_save_path = cfg.get("save_path") or os.path.join(cfg.get("exp_dir"), "model.pt")
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)

    ks = tuple(cfg.get("val_ks", [1, 10, 50]))

    epochs = int(cfg["epochs"])
    print(f"[INFO] Starting training for {epochs} epochs.")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch:03d}/{epochs:03d} | start")
        t0 = perf_counter()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            desc=f"train {epoch:03d}/{epochs:03d}",
        )
        metrics = validate_retrieval(model, val_loader, device, ks=ks)
        dt = perf_counter() - t0
        if metrics.get(metric_key, 0.0) > best_metric:
            best_metric = metrics[metric_key]
            torch.save(model.state_dict(), model_save_path)
        print(
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val {format_metrics(metrics)} | "
            f"time {dt:.1f}s | best_{metric_key} {best_metric:.4f}"
        )


if __name__ == "__main__":
    main()
