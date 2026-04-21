#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import csv
import json
import logging
import math
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from common import (
    build_arc_margin_loss,
    build_campp_classifier,
    build_campp_embedding_model,
    ensure_dir,
    get_git_sha,
    load_config,
    load_pretrained_embedding,
    manifest_path_for_split,
    prepared_root,
    resolve_mlflow_experiment,
    resolve_mlflow_tracking_uri,
    runs_root,
    sanitize_mlflow_key,
    save_checkpoint,
    seed_everything,
    write_json,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune a speaker embedding model from pretrained weights."
    )
    parser.add_argument("--config", required=True, help="Path to model YAML config.")
    parser.add_argument("--run-name", default="", help="Optional run name.")
    parser.add_argument(
        "--max-steps", type=int, default=0, help="Optional hard stop after N optimizer steps."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=0, help="Optional hard stop after N epochs."
    )
    parser.add_argument(
        "--resume-checkpoint", default="", help="Optional checkpoint path to resume training from."
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_path))
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def set_module_trainable(module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(trainable)


def build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(1e-6, float(current_step + 1) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate(config: dict, manifest: pd.DataFrame, model, device) -> dict[str, float]:
    from retrieval import extract_embeddings, retrieval_metrics_from_embeddings

    embeddings, labels = extract_embeddings(
        manifest=manifest,
        model=model,
        data_root=config["paths"]["data_root"],
        sample_rate=int(config["model"]["sample_rate"]),
        n_mels=int(config["model"]["n_mels"]),
        mode=str(config["evaluation"]["primary_mode"]),
        eval_chunk_sec=float(config["training"]["eval_chunk_sec"]),
        segment_count=int(config["evaluation"]["segment_count"]),
        long_file_threshold_sec=float(config["evaluation"]["long_file_threshold_sec"]),
        batch_size=int(config["training"]["batch_size"]),
        device=device,
        pad_mode=str(config["training"]["short_clip_pad_mode"]),
    )
    return retrieval_metrics_from_embeddings(
        embeddings=embeddings,
        labels=labels,
        ks=config["evaluation"]["ks"],
        chunk_size=int(config["evaluation"]["retrieval_chunk_size"]),
        device=device,
    )


def main() -> None:
    args = parse_args()

    import torch
    from retrieval import ManifestTrainDataset, collate_features
    from torch.utils.data import DataLoader

    config = load_config(args.config)
    seed_everything(int(config["data_prep"]["seed"]))
    ks = [int(k) for k in config["evaluation"]["ks"]]
    max_k = max(ks)
    tracked_metric_keys = [
        "backbone_frozen",
        *[f"precision@{k}" for k in ks],
        *[f"hit_rate@{k}" for k in ks],
        *[f"recall@{k}" for k in ks],
        f"ndcg@{max_k}",
        f"mrr@{max_k}",
    ]

    prepared = prepared_root(config)
    train_manifest_path = manifest_path_for_split(config, "train")
    val_manifest_path = manifest_path_for_split(config, "validation")
    speaker_index_path = prepared / "speaker_to_index.json"
    split_summary_path = prepared / "split_summary.json"
    if (
        not train_manifest_path.exists()
        or not val_manifest_path.exists()
        or not speaker_index_path.exists()
    ):
        raise FileNotFoundError("Prepared manifests are missing. Run prepare_data.py first.")

    with speaker_index_path.open("r", encoding="utf-8") as handle:
        speaker_to_index = json.load(handle)

    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"campp_en_ft_noaug_{run_stamp}"
    run_root = ensure_dir(runs_root(config) / run_name)
    ckpt_root = ensure_dir(run_root / "checkpoints")
    log_path = run_root / "train.log"
    logger = setup_logger(log_path)
    write_resolved_config(config, run_root / "config_resolved.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.info("Using device %s (%s)", device, gpu_name)

    train_dataset = ManifestTrainDataset(
        manifest_path=train_manifest_path,
        speaker_to_index=speaker_to_index,
        data_root=config["paths"]["data_root"],
        sample_rate=int(config["model"]["sample_rate"]),
        n_mels=int(config["model"]["n_mels"]),
        chunk_sec=float(config["training"]["train_chunk_sec"]),
        pad_mode=str(config["training"]["short_clip_pad_mode"]),
        augmentations=config.get("augmentations"),
    )
    train_loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "shuffle": True,
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": True,
        "drop_last": True,
        "collate_fn": collate_features,
    }
    if int(config["training"]["num_workers"]) > 0:
        train_loader_kwargs["persistent_workers"] = bool(
            config["training"].get("persistent_workers", True)
        )
        train_loader_kwargs["prefetch_factor"] = int(config["training"].get("prefetch_factor", 4))
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_manifest = pd.read_csv(val_manifest_path)

    embedding_model = build_campp_embedding_model(config)
    pretrained_path = load_pretrained_embedding(config, embedding_model)
    embedding_model = embedding_model.to(device)
    classifier = build_campp_classifier(config, num_classes=len(speaker_to_index)).to(device)
    criterion = build_arc_margin_loss(config).to(device)

    optimizer = torch.optim.SGD(
        [
            {
                "params": embedding_model.parameters(),
                "lr": float(config["training"]["backbone_lr"]),
            },
            {"params": classifier.parameters(), "lr": float(config["training"]["classifier_lr"])},
        ],
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"]),
        nesterov=bool(config["training"]["nesterov"]),
    )
    total_epochs = int(args.max_epochs or config["training"]["epochs"])
    benchmark_steps = int(args.max_steps or config["training"]["benchmark_steps"])
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    planned_total_steps = min(total_steps, benchmark_steps) if benchmark_steps else total_steps
    scheduler = build_scheduler(
        optimizer,
        total_steps=max(1, total_steps),
        warmup_steps=max(1, int(config["training"]["warmup_epochs"]) * steps_per_epoch),
    )

    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history_rows: list[dict[str, object]] = []
    best_p10 = float("-inf")
    best_epoch = None
    best_metrics: dict[str, float] = {}
    best_ckpt_path = ckpt_root / "best_p10.pt"
    global_step = 0
    start_epoch = 1
    wall_start = time.perf_counter()

    history_path = run_root / "history.csv"
    summary_path = run_root / "run_summary.json"

    def write_history_file() -> None:
        history_fields = [
            "epoch",
            "global_step",
            "train_loss",
            "epoch_seconds",
            *tracked_metric_keys,
        ]
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history_fields)
            writer.writeheader()
            writer.writerows(history_rows)

    def build_final_summary(
        total_seconds: float, benchmark_projection: dict | None
    ) -> dict[str, object]:
        return {
            "run_name": run_name,
            "run_root": str(run_root),
            "pretrained_weight_path": str(pretrained_path),
            "best_checkpoint": str(
                best_ckpt_path if best_ckpt_path.exists() else ckpt_root / "last.pt"
            ),
            "best_precision@10": best_p10 if best_p10 > float("-inf") else None,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "epochs_completed": len(history_rows),
            "global_steps": global_step,
            "total_seconds": total_seconds,
            "git_sha": params_for_mlflow["git_sha"],
            "benchmark_projection": benchmark_projection,
        }

    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint).resolve()
        state = torch.load(checkpoint_path, map_location="cpu")
        embedding_model.load_state_dict(state["embedding_model"], strict=True)
        classifier.load_state_dict(state["classifier"], strict=True)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state and use_amp:
            scaler.load_state_dict(state["scaler"])
        global_step = int(state.get("global_step", 0))
        best_metrics = dict(state.get("best_metrics") or {})
        best_epoch = state.get("best_epoch")
        if best_metrics and "precision@10" in best_metrics:
            best_p10 = float(best_metrics["precision@10"])
        elif (
            "metrics" in state
            and isinstance(state["metrics"], dict)
            and "precision@10" in state["metrics"]
        ):
            best_p10 = float(state["metrics"]["precision@10"])
            best_metrics = dict(state["metrics"])
            best_epoch = int(state.get("epoch", 0))
        start_epoch = int(state.get("epoch", 0)) + 1

    aug_cfg = dict(config.get("augmentations", {}))
    stage_name = (
        "benchmark"
        if benchmark_steps
        else ("finetune_aug" if bool(aug_cfg.get("enabled", False)) else "finetune_noaug")
    )
    params_for_mlflow = {
        "stage": stage_name,
        "pretrained_model_id": config["pretrained"]["model_id"],
        "pretrained_weight_path": str(pretrained_path),
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "train_rows": len(train_dataset),
        "val_rows": len(val_manifest),
        "train_speakers": len(speaker_to_index),
        "val_speakers": int(val_manifest["spk"].nunique()),
        "train_chunk_sec": config["training"]["train_chunk_sec"],
        "eval_chunk_sec": config["training"]["eval_chunk_sec"],
        "eval_mode": config["evaluation"]["primary_mode"],
        "ks": ",".join(str(k) for k in config["evaluation"]["ks"]),
        "batch_size": config["training"]["batch_size"],
        "epochs": total_epochs,
        "num_workers": config["training"]["num_workers"],
        "prefetch_factor": config["training"].get("prefetch_factor", 4),
        "persistent_workers": config["training"].get("persistent_workers", True),
        "backbone_lr": config["training"]["backbone_lr"],
        "classifier_lr": config["training"]["classifier_lr"],
        "freeze_backbone_epochs": config["training"].get("freeze_backbone_epochs", 0),
        "augmentations_enabled": bool(aug_cfg.get("enabled", False)),
        "aug_noise_probability": aug_cfg.get("noise", {}).get("probability", 0.0),
        "aug_reverb_probability": aug_cfg.get("reverb", {}).get("probability", 0.0),
        "aug_band_limit_probability": aug_cfg.get("band_limit", {}).get("probability", 0.0),
        "aug_silence_shift_probability": aug_cfg.get("silence_shift", {}).get("probability", 0.0),
        "git_sha": get_git_sha(config["project_root"]),
        "device": str(device),
        "gpu_name": gpu_name,
        "resume_checkpoint": str(Path(args.resume_checkpoint).resolve())
        if args.resume_checkpoint
        else "",
    }
    mlflow_module = None
    mlflow_enabled = bool(config.get("mlflow", {}).get("enabled", True))
    if mlflow_enabled:
        try:
            import mlflow as _mlflow

            _mlflow.set_tracking_uri(resolve_mlflow_tracking_uri(config))
            _mlflow.set_experiment(resolve_mlflow_experiment(config))
            all_tags = dict(config.get("mlflow", {}).get("tags", {}))
            all_tags["stage"] = stage_name
            _mlflow.start_run(run_name=run_name)
            if all_tags:
                _mlflow.set_tags(all_tags)
            flat_params = {
                sanitize_mlflow_key(key): str(value) for key, value in params_for_mlflow.items()
            }
            if flat_params:
                _mlflow.log_params(flat_params)
            mlflow_module = _mlflow
        except ImportError:
            mlflow_enabled = False

    run_final_status = {"status": "KILLED"}

    def close_mlflow_run(status: str | None = None) -> None:
        nonlocal mlflow_module
        if mlflow_module is None:
            return
        final_status = status or run_final_status["status"]
        try:
            mlflow_module.end_run(status=final_status)
        except TypeError:
            mlflow_module.end_run()
        finally:
            mlflow_module = None

    atexit.register(lambda: close_mlflow_run(run_final_status["status"]))

    def handle_signal(signum, _frame) -> None:
        signame = signal.Signals(signum).name
        logger.error("Received %s, terminating run.", signame)
        run_final_status["status"] = "KILLED"
        close_mlflow_run("KILLED")
        raise KeyboardInterrupt(signame)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)

    logger.info("Run root: %s", run_root)
    logger.info("Pretrained weights: %s", pretrained_path)
    logger.info("Train rows=%s, val rows=%s", len(train_dataset), len(val_manifest))
    if args.resume_checkpoint:
        logger.info(
            "Resuming from checkpoint %s starting at epoch=%s global_step=%s best_p10=%s",
            Path(args.resume_checkpoint).resolve(),
            start_epoch,
            global_step,
            f"{best_p10:.4f}" if best_p10 > float("-inf") else "n/a",
        )
    logger.info(
        "Train setup: batch_size=%s steps_per_epoch=%s planned_total_steps=%s log_every_steps=%s",
        config["training"]["batch_size"],
        steps_per_epoch,
        planned_total_steps,
        int(config["training"].get("log_every_steps", 200)),
    )

    stop_training = False
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.perf_counter()
        backbone_frozen = epoch <= int(config["training"].get("freeze_backbone_epochs", 0))
        set_module_trainable(embedding_model, not backbone_frozen)
        embedding_model.train(not backbone_frozen)
        classifier.train(True)
        if backbone_frozen:
            embedding_model.eval()
        running_loss = 0.0
        running_batches = 0
        log_every_steps = int(config["training"].get("log_every_steps", 200))
        logger.info(
            "epoch=%s/%s backbone_state=%s batch_size=%s",
            epoch,
            total_epochs,
            "frozen" if backbone_frozen else "trainable",
            config["training"]["batch_size"],
        )

        for epoch_step, (feats, labels) in enumerate(train_loader, start=1):
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                embeddings = embedding_model(feats)
                logits = classifier(embeddings)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if float(config["training"]["max_grad_norm"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(embedding_model.parameters()) + list(classifier.parameters()),
                    float(config["training"]["max_grad_norm"]),
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            running_batches += 1
            global_step += 1

            should_log_step = (
                epoch_step == 1
                or epoch_step % log_every_steps == 0
                or epoch_step == steps_per_epoch
                or (benchmark_steps and global_step >= benchmark_steps)
            )
            if should_log_step:
                epoch_elapsed = time.perf_counter() - epoch_start
                total_elapsed = time.perf_counter() - wall_start
                epoch_rate = epoch_step / max(epoch_elapsed, 1e-9)
                total_rate = global_step / max(total_elapsed, 1e-9)
                epoch_eta = (steps_per_epoch - epoch_step) / max(epoch_rate, 1e-9)
                total_eta = (planned_total_steps - global_step) / max(total_rate, 1e-9)
                logger.info(
                    "progress epoch=%s/%s batch=%s/%s global_step=%s/%s "
                    "loss=%.4f avg_loss=%.4f lr_backbone=%.6g "
                    "lr_classifier=%.6g eta_epoch=%s eta_total=%s",
                    epoch,
                    total_epochs,
                    epoch_step,
                    steps_per_epoch,
                    global_step,
                    planned_total_steps,
                    float(loss.item()),
                    running_loss / max(1, running_batches),
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                    format_eta(epoch_eta),
                    format_eta(total_eta),
                )

            if benchmark_steps and global_step >= benchmark_steps:
                stop_training = True
                break

        epoch_seconds = time.perf_counter() - epoch_start
        avg_train_loss = running_loss / max(1, running_batches)
        epoch_metrics: dict[str, float] = {
            "train_loss": avg_train_loss,
            "backbone_frozen": float(backbone_frozen),
        }

        if epoch % int(config["training"]["eval_every_epochs"]) == 0:
            val_metrics = evaluate(config, val_manifest, embedding_model, device)
            epoch_metrics.update(val_metrics)
            if val_metrics["precision@10"] > best_p10:
                best_p10 = val_metrics["precision@10"]
                best_epoch = epoch
                best_metrics = dict(val_metrics)
                save_checkpoint(
                    best_ckpt_path,
                    embedding_model,
                    classifier,
                    epoch,
                    val_metrics,
                    config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=global_step,
                    best_metrics=best_metrics,
                    best_epoch=best_epoch,
                )
        epoch_ckpt_path = ckpt_root / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            epoch_ckpt_path,
            embedding_model,
            classifier,
            epoch,
            epoch_metrics,
            config,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            best_metrics=best_metrics,
            best_epoch=best_epoch,
        )
        if epoch % int(config["training"]["save_every_epochs"]) == 0:
            save_checkpoint(
                ckpt_root / "last.pt",
                embedding_model,
                classifier,
                epoch,
                epoch_metrics,
                config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                best_metrics=best_metrics,
                best_epoch=best_epoch,
            )

        history_rows.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": avg_train_loss,
                "epoch_seconds": epoch_seconds,
                **{key: epoch_metrics.get(key) for key in tracked_metric_keys},
            }
        )
        if mlflow_module is not None:
            epoch_mlflow_metrics = {
                "epoch.train_loss": float(avg_train_loss),
                "epoch.epoch_seconds": float(epoch_seconds),
                "epoch.backbone_frozen": float(backbone_frozen),
            }
            for key in tracked_metric_keys:
                if epoch_metrics.get(key) is not None:
                    metric_name = key if key == "backbone_frozen" else f"epoch.{key}"
                    epoch_mlflow_metrics[metric_name] = float(epoch_metrics[key])
            mlflow_module.log_metrics(
                {sanitize_mlflow_key(key): value for key, value in epoch_mlflow_metrics.items()},
                step=epoch,
            )
        write_history_file()
        if mlflow_module is not None:
            runtime_epoch_dir = run_root / "mlflow_runtime" / f"epoch_{epoch:03d}"
            runtime_epoch_dir.mkdir(parents=True, exist_ok=True)
            runtime_log = runtime_epoch_dir / "train.log"
            runtime_history = runtime_epoch_dir / "history.csv"
            runtime_log.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
            runtime_history.write_text(history_path.read_text(encoding="utf-8"), encoding="utf-8")
            mlflow_module.log_artifact(str(runtime_log), artifact_path=f"runtime/epoch_{epoch:03d}")
            mlflow_module.log_artifact(
                str(runtime_history), artifact_path=f"runtime/epoch_{epoch:03d}"
            )
        logger.info(
            "epoch=%s step=%s backbone_state=%s train_loss=%.4f epoch_sec=%.2f "
            "p@1=%s p@5=%s p@10=%s hr@10=%s r@10=%s ndcg@10=%s mrr@10=%s",
            epoch,
            global_step,
            "frozen" if backbone_frozen else "trainable",
            avg_train_loss,
            epoch_seconds,
            f"{epoch_metrics.get('precision@1', float('nan')):.4f}"
            if "precision@1" in epoch_metrics
            else "n/a",
            f"{epoch_metrics.get('precision@5', float('nan')):.4f}"
            if "precision@5" in epoch_metrics
            else "n/a",
            f"{epoch_metrics.get('precision@10', float('nan')):.4f}"
            if "precision@10" in epoch_metrics
            else "n/a",
            f"{epoch_metrics.get('hit_rate@10', float('nan')):.4f}"
            if "hit_rate@10" in epoch_metrics
            else "n/a",
            f"{epoch_metrics.get('recall@10', float('nan')):.4f}"
            if "recall@10" in epoch_metrics
            else "n/a",
            f"{epoch_metrics.get('ndcg@10', float('nan')):.4f}"
            if "ndcg@10" in epoch_metrics
            else "n/a",
            f"{epoch_metrics.get('mrr@10', float('nan')):.4f}"
            if "mrr@10" in epoch_metrics
            else "n/a",
        )

        if stop_training:
            logger.info(
                "Stopping early because max benchmark steps=%s was reached.", benchmark_steps
            )
            break

    total_seconds = time.perf_counter() - wall_start
    write_history_file()

    benchmark_projection = None
    if history_rows and benchmark_steps:
        training_seconds_total = sum(float(row["epoch_seconds"]) for row in history_rows)
        eval_runs_observed = sum(1 for row in history_rows if row.get("precision@10") is not None)
        validation_seconds_total = max(0.0, total_seconds - training_seconds_total)
        avg_train_step_seconds = training_seconds_total / max(1, global_step)
        projected_epoch_train_seconds = avg_train_step_seconds * len(train_loader)
        projected_validation_seconds_per_eval = (
            validation_seconds_total / max(1, eval_runs_observed) if eval_runs_observed else 0.0
        )
        eval_every_epochs = max(1, int(config["training"]["eval_every_epochs"]))
        projected_epoch_total_seconds = projected_epoch_train_seconds + (
            projected_validation_seconds_per_eval / eval_every_epochs
        )
        projected_5_epoch_seconds = (
            projected_epoch_train_seconds * 5
            + projected_validation_seconds_per_eval * math.ceil(5 / eval_every_epochs)
        )
        benchmark_projection = {
            "avg_train_step_seconds": avg_train_step_seconds,
            "projected_epoch_train_seconds": projected_epoch_train_seconds,
            "projected_validation_seconds_per_eval": projected_validation_seconds_per_eval,
            "projected_epoch_total_seconds": projected_epoch_total_seconds,
            "projected_5_epoch_seconds": projected_5_epoch_seconds,
        }

    final_summary = build_final_summary(
        total_seconds=total_seconds, benchmark_projection=benchmark_projection
    )
    write_json(summary_path, final_summary)

    if mlflow_module is not None:
        if history_rows:
            last_row = history_rows[-1]
            final_mlflow_metrics = {
                "last.train_loss": float(last_row["train_loss"]),
                "last.epoch_seconds": float(last_row["epoch_seconds"]),
            }
            for key in tracked_metric_keys:
                if last_row.get(key) is not None:
                    final_mlflow_metrics[f"last.{key}"] = float(last_row[key])
            if best_metrics:
                for key, value in best_metrics.items():
                    final_mlflow_metrics[f"best.{key}"] = float(value)
            mlflow_module.log_metrics(
                {sanitize_mlflow_key(key): value for key, value in final_mlflow_metrics.items()}
            )
        for artifact in [
            val_manifest_path,
            split_summary_path,
            history_path,
            summary_path,
            log_path,
            run_root / "config_resolved.yaml",
        ]:
            if artifact.exists():
                mlflow_module.log_artifact(str(artifact))
        run_final_status["status"] = "FINISHED"
        close_mlflow_run("FINISHED")

    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    print(history_path)


if __name__ == "__main__":
    main()
