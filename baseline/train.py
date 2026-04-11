import argparse
import json
import os
import torch
import numpy as np
import pandas as pd

from time import perf_counter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Dict, Sequence

from src.metrics import precision_at_k
from src.dataset import SpeakerDataset
from src.model import ECAPASpeakerId


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
) -> Tuple[float, float]:
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

        if (
            batch_idx == 1
            or batch_idx % 10 == 0
            or (total is not None and batch_idx == total)
        ):
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
) -> Dict[str, float]:
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


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = sorted(
        metrics.keys(),
        key=lambda x: (x.split("@")[0], int(x.split("@")[1]) if "@" in x else 0),
    )
    return ", ".join([f"{k} {metrics[k]:.4f}" for k in keys])


def split_data(train_csv: str, exp_dir: str, split_ratio: float):
    df = pd.read_csv(train_csv)
    n = len(df)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(n)
    split_idx = int(n * split_ratio)

    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    train_split_path = os.path.join(exp_dir, "train_split.csv")
    val_split_path = os.path.join(exp_dir, "val_split.csv")

    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)

    print(f"[INFO] Train split saved: {len(train_df)} rows. Path: {train_split_path}.")
    print(f"[INFO] Validation split saved: {len(val_df)} rows. Path: {val_split_path}.")

    return train_split_path, val_split_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    os.makedirs(cfg.get("exp_dir"), exist_ok=True)

    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    train_split_path, val_split_path = split_data(
        cfg["train_csv"], cfg["exp_dir"], cfg["split_ratio"]
    )

    train_ds = SpeakerDataset(
        csv_path=train_split_path,
        sample_rate=cfg["sample_rate"],
        chunk_seconds=cfg["train_chunk_seconds"],
        is_train=True,
        base_dir=cfg["data_base_dir"],
    )
    val_ds = SpeakerDataset(
        csv_path=val_split_path,
        sample_rate=cfg["sample_rate"],
        chunk_seconds=cfg["val_chunk_seconds"],
        is_train=False,
        base_dir=cfg["data_base_dir"],
    )

    print("[INFO] ===== Dataset summary =====")
    print(f"[INFO] train_csv: {cfg['train_csv']}")
    print(
        f"[INFO] train speakers: {train_ds.num_speakers} | train audio files: {len(train_ds)}"
    )
    print(
        f"[INFO] val speakers:   {val_ds.num_speakers} | val audio files:   {len(val_ds)}"
    )
    print(
        f"[INFO] sample_rate={cfg['sample_rate']} train_chunk_seconds={cfg['train_chunk_seconds']} val_chunk_seconds={cfg['val_chunk_seconds']}"
    )
    print(
        f"[INFO] batch_size={cfg['batch_size']} num_workers={cfg['num_workers']} device={device}"
    )
    print("[INFO] ===========================")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
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
    model_save_path = os.path.join(cfg.get("exp_dir"), "model.pt")

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
