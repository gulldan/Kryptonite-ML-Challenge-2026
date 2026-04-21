import argparse
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

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
    grad_clip_norm: float | None = None,
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
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def metric_improved(
    candidate: float,
    best: float | None,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if mode == "min":
        return candidate < best - min_delta
    if mode == "max":
        return candidate > best + min_delta
    raise ValueError("mode must be one of: min, max")


def append_jsonl(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


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

    metric_key = str(cfg.get("checkpoint_metric", "precision@10"))
    early_stopping_enabled = bool(cfg.get("early_stopping_enabled", False))
    early_metric_key = str(cfg.get("early_stopping_metric", metric_key))
    early_mode = str(cfg.get("early_stopping_mode", "max")).lower()
    early_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    early_patience = int(cfg.get("early_stopping_patience", 3))
    early_min_epochs = int(cfg.get("early_stopping_min_epochs", 1))
    early_restore_best = bool(cfg.get("early_stopping_restore_best", True))
    stop_train_accuracy = cfg.get("early_stopping_stop_train_accuracy")
    stop_train_accuracy = None if stop_train_accuracy is None else float(stop_train_accuracy)
    if early_mode not in {"min", "max"}:
        raise ValueError("early_stopping_mode must be one of: min, max")
    scheduler_mode = cast(Literal["min", "max"], early_mode)
    if early_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if early_min_epochs <= 0:
        raise ValueError("early_stopping_min_epochs must be positive")
    if early_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if stop_train_accuracy is not None and not 0.0 <= stop_train_accuracy <= 1.0:
        raise ValueError("early_stopping_stop_train_accuracy must be within [0.0, 1.0]")

    scheduler_name = str(cfg.get("scheduler", "none")).lower()
    if scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=float(cfg.get("scheduler_factor", 0.5)),
            patience=int(cfg.get("scheduler_patience", 1)),
            threshold=float(cfg.get("scheduler_threshold", early_min_delta)),
            min_lr=float(cfg.get("scheduler_min_lr", 1e-6)),
        )
    elif scheduler_name in {"", "none"}:
        scheduler = None
    else:
        raise ValueError("scheduler must be one of: none, reduce_on_plateau")

    best_metric: float | None = None
    best_epoch: int | None = None
    best_early_metric: float | None = None
    bad_epochs = 0
    stopped_early = False
    stop_reason: str | None = None
    epoch_records: list[dict[str, Any]] = []
    model_save_path = cfg.get("save_path") or os.path.join(cfg.get("exp_dir"), "model.pt")
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    metrics_path = cfg.get("metrics_path") or os.path.join(cfg.get("exp_dir"), "metrics.jsonl")
    training_summary_path = cfg.get("training_summary_path") or os.path.join(
        cfg.get("exp_dir"), "training_summary.json"
    )
    Path(metrics_path).write_text("", encoding="utf-8")

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
            grad_clip_norm=cfg.get("grad_clip_norm"),
            desc=f"train {epoch:03d}/{epochs:03d}",
        )
        metrics = validate_retrieval(model, val_loader, device, ks=ks)
        dt = perf_counter() - t0
        if metric_key not in metrics:
            raise ValueError(f"Checkpoint metric '{metric_key}' not found in validation metrics.")
        if early_metric_key not in metrics:
            raise ValueError(
                f"Early-stopping metric '{early_metric_key}' not found in validation metrics."
            )

        checkpoint_metric = float(metrics[metric_key])
        early_metric = float(metrics[early_metric_key])
        is_best = metric_improved(
            checkpoint_metric,
            best_metric,
            mode="max",
            min_delta=0.0,
        )
        if is_best:
            best_metric = checkpoint_metric
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)

        early_improved = metric_improved(
            early_metric,
            best_early_metric,
            mode=early_mode,
            min_delta=early_min_delta,
        )
        if early_improved:
            best_early_metric = early_metric
            bad_epochs = 0
        else:
            bad_epochs += 1

        if scheduler is not None:
            scheduler.step(early_metric)

        learning_rate = current_learning_rate(optimizer)
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "train_accuracy": round(float(train_acc), 6),
            "learning_rate": round(learning_rate, 10),
            "validation": {key: round(float(value), 6) for key, value in sorted(metrics.items())},
            "seconds": round(float(dt), 3),
            "is_best_checkpoint": is_best,
            "best_epoch": best_epoch,
            "best_metric": None if best_metric is None else round(float(best_metric), 6),
            "early_stopping_bad_epochs": bad_epochs,
        }
        epoch_records.append(epoch_record)
        append_jsonl(metrics_path, epoch_record)

        print(
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val {format_metrics(metrics)} | "
            f"lr {learning_rate:.6g} | time {dt:.1f}s | "
            f"best_{metric_key} {0.0 if best_metric is None else best_metric:.4f}"
        )

        if early_stopping_enabled and epoch >= early_min_epochs:
            if stop_train_accuracy is not None and train_acc >= stop_train_accuracy:
                stopped_early = True
                stop_reason = "train_accuracy_threshold"
            elif bad_epochs > 0 and bad_epochs >= early_patience:
                stopped_early = True
                stop_reason = "patience_exhausted"
            if stopped_early:
                print(
                    "[INFO] Early stopping triggered: "
                    f"epoch={epoch} reason={stop_reason} "
                    f"best_epoch={best_epoch} best_{metric_key}={best_metric:.4f}"
                )
                break

    if early_restore_best and os.path.isfile(model_save_path):
        state_dict = torch.load(model_save_path, map_location=device)
        model.load_state_dict(state_dict)

    write_json(
        training_summary_path,
        {
            "config_path": args.config,
            "exp_dir": cfg.get("exp_dir"),
            "model_save_path": model_save_path,
            "train_split_path": train_split_path,
            "val_split_path": val_split_path,
            "device": device,
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_completed": len(epoch_records),
            "checkpoint_metric": metric_key,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "metrics_path": metrics_path,
            "early_stopping": {
                "enabled": early_stopping_enabled,
                "metric": early_metric_key,
                "mode": early_mode,
                "min_delta": early_min_delta,
                "patience": early_patience,
                "min_epochs": early_min_epochs,
                "restore_best": early_restore_best,
                "stop_train_accuracy": stop_train_accuracy,
                "stopped_early": stopped_early,
                "reason": stop_reason,
                "best_early_metric": best_early_metric,
            },
            "scheduler": {
                "name": scheduler_name,
                "final_learning_rate": current_learning_rate(optimizer),
            },
            "epochs": epoch_records,
        },
    )
    print(f"[INFO] Metrics written to {metrics_path}")
    print(f"[INFO] Training summary written to {training_summary_path}")


if __name__ == "__main__":
    main()
