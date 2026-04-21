"""Evaluate a trained speaker encoder on CN-Celeb eval set.

Usage:
    uv run python research/scripts/eval_cnceleb.py \
        --model campp \
        --checkpoint artifacts/baselines/campp/ffsvc2022-surrogate/<run>/campp_encoder.pt \
        --device cuda
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
import typer

app = typer.Typer(add_completion=False, help=__doc__)

CNCELEB_ROOT = Path("datasets/CN-Celeb_flac")
EVAL_ROOT = CNCELEB_ROOT / "eval"
OUTPUT_ROOT = Path("artifacts/eval/cn-celeb")


def load_encoder(model_name: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load encoder from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt["model_config"]

    if model_name == "campp":
        from kryptonite.models.campp import CAMPPlusConfig, CAMPPlusEncoder

        config = CAMPPlusConfig(**model_config)
        encoder = CAMPPlusEncoder(config)
    elif model_name == "eres2netv2":
        from kryptonite.models.eres2netv2 import ERes2NetV2Config, ERes2NetV2Encoder

        config = ERes2NetV2Config(**model_config)
        encoder = ERes2NetV2Encoder(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.to(device)
    encoder.eval()
    return encoder


# Cache project config and extractor to avoid reloading per file
_CACHED_RESOURCES: dict[str, Any] = {}


def _get_resources():
    if "project" not in _CACHED_RESOURCES:
        from kryptonite.config import load_project_config
        from kryptonite.data import AudioLoadRequest
        from kryptonite.features import FbankExtractionRequest, FbankExtractor

        project = load_project_config(config_path="configs/base.toml")
        _CACHED_RESOURCES["project"] = project
        _CACHED_RESOURCES["audio_request"] = AudioLoadRequest.from_config(
            project.normalization, vad=project.vad
        )
        feature_request = FbankExtractionRequest.from_config(project.features)
        _CACHED_RESOURCES["extractor"] = FbankExtractor(request=feature_request)
    return _CACHED_RESOURCES


def extract_embedding_fast(
    encoder: torch.nn.Module,
    audio_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Extract L2-normalized embedding (uses cached resources)."""
    from kryptonite.data import load_audio

    res = _get_resources()
    loaded = load_audio(audio_path, request=res["audio_request"])
    features = res["extractor"].extract(loaded.waveform, sample_rate_hz=loaded.sample_rate_hz)

    with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        embedding = encoder(features.unsqueeze(0).to(device=device, dtype=torch.float32)).squeeze(0)

    normalized = torch_functional.normalize(embedding.detach().cpu().float(), dim=0)
    return normalized.numpy()


def load_trials(trials_path: Path) -> list[dict[str, Any]]:
    """Load CN-Celeb trial list."""
    trials = []
    for line in trials_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        enroll_id, test_path, label = parts
        trials.append(
            {
                "enroll_id": enroll_id,
                "test_path": test_path,
                "label": int(label),
            }
        )
    return trials


def _resolve_audio_path(rel_path: str) -> Path:
    """Resolve audio path, trying .flac if .wav doesn't exist."""
    full = EVAL_ROOT / rel_path
    if full.exists():
        return full
    flac = full.with_suffix(".flac")
    if flac.exists():
        return flac
    return full


def load_enroll_map(enroll_lst: Path) -> dict[str, Path]:
    """Map enrollment ID to audio path."""
    mapping = {}
    for line in enroll_lst.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        enroll_id, rel_path = parts
        mapping[enroll_id] = _resolve_audio_path(rel_path)
    return mapping


MODEL_OPTION = typer.Option(..., "--model", help="Model: campp or eres2netv2")
CHECKPOINT_OPTION = typer.Option(..., "--checkpoint", help="Path to encoder checkpoint")
DEVICE_OPTION = typer.Option("cuda", "--device", help="Device")
MAX_TEST_OPTION = typer.Option(None, "--max-test", help="Limit test utterances (debug)")
OUTPUT_DIR_OPTION = typer.Option(None, "--output-dir", help="Override output directory")


@app.command()
def main(
    model: str = MODEL_OPTION,
    checkpoint: Path = CHECKPOINT_OPTION,
    device: str = DEVICE_OPTION,
    max_test: int | None = MAX_TEST_OPTION,
    output_dir: Path | None = OUTPUT_DIR_OPTION,
) -> None:
    dev = torch.device(device)
    typer.echo(f"Loading {model} encoder from {checkpoint}")
    encoder = load_encoder(model, checkpoint, dev)

    out = output_dir or (OUTPUT_ROOT / model)
    out.mkdir(parents=True, exist_ok=True)

    # Load enrollment map and trial list
    enroll_map = load_enroll_map(EVAL_ROOT / "lists" / "enroll.lst")
    trials = load_trials(EVAL_ROOT / "lists" / "trials.lst")
    typer.echo(f"Enrollment speakers: {len(enroll_map)}")
    typer.echo(f"Total trials: {len(trials)}")

    # Extract enrollment embeddings
    typer.echo("Extracting enrollment embeddings...")
    enroll_embeddings: dict[str, np.ndarray] = {}
    for i, (enroll_id, audio_path) in enumerate(enroll_map.items()):
        enroll_embeddings[enroll_id] = extract_embedding_fast(encoder, audio_path, dev)
        if (i + 1) % 50 == 0:
            typer.echo(f"  Enrollment: {i + 1}/{len(enroll_map)}")
    typer.echo(f"  Enrollment done: {len(enroll_embeddings)} speakers")

    # Collect unique test paths
    test_paths_set: set[str] = set()
    for trial in trials:
        test_paths_set.add(trial["test_path"])
    test_paths_sorted = sorted(test_paths_set)
    if max_test:
        test_paths_sorted = test_paths_sorted[:max_test]
        trials = [t for t in trials if t["test_path"] in set(test_paths_sorted)]
    typer.echo(f"Test utterances to embed: {len(test_paths_sorted)}")

    # Extract test embeddings
    typer.echo("Extracting test embeddings...")
    test_embeddings: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, rel_path in enumerate(test_paths_sorted):
        audio_path = _resolve_audio_path(rel_path)
        if not audio_path.exists():
            continue
        test_embeddings[rel_path] = extract_embedding_fast(encoder, audio_path, dev)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(test_paths_sorted) - i - 1) / rate
            typer.echo(f"  Test: {i + 1}/{len(test_paths_sorted)} ({rate:.1f}/s, ETA {eta:.0f}s)")
    typer.echo(f"  Test done: {len(test_embeddings)} utterances")

    # Score trials
    typer.echo("Scoring trials...")
    from kryptonite.models.scoring import cosine_score_pairs

    scored_rows: list[dict[str, Any]] = []
    batch_left: list[np.ndarray] = []
    batch_right: list[np.ndarray] = []
    batch_trials: list[dict[str, Any]] = []

    for trial in trials:
        e_emb = enroll_embeddings.get(trial["enroll_id"])
        t_emb = test_embeddings.get(trial["test_path"])
        if e_emb is None or t_emb is None:
            continue
        batch_left.append(e_emb)
        batch_right.append(t_emb)
        batch_trials.append(trial)

    if batch_left:
        scores = cosine_score_pairs(np.stack(batch_left), np.stack(batch_right), normalize=True)
        for trial, score in zip(batch_trials, scores, strict=True):
            scored_rows.append(
                {
                    "label": trial["label"],
                    "score": round(float(score), 8),
                    "left_id": trial["enroll_id"],
                    "right_id": trial["test_path"],
                    "left_speaker_id": trial["enroll_id"].rsplit("-", 1)[0],
                    "right_speaker_id": trial["test_path"].split("/")[-1].split("-")[0],
                }
            )

    typer.echo(f"Scored {len(scored_rows)} trials")

    # Save scores
    scores_path = out / "cnceleb_scores.jsonl"
    scores_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in scored_rows),
        encoding="utf-8",
    )

    # Compute verification metrics
    from kryptonite.eval import (
        build_verification_evaluation_report,
        compute_verification_metrics,
        write_verification_evaluation_report,
    )

    metrics = compute_verification_metrics(scored_rows)
    typer.echo(f"\n{'=' * 50}")
    typer.echo(f"CN-Celeb Eval Results ({model})")
    typer.echo(f"{'=' * 50}")
    typer.echo(f"Trials:  {metrics.trial_count}")
    typer.echo(f"EER:     {metrics.eer:.4f} (threshold {metrics.eer_threshold:.4f})")
    typer.echo(f"MinDCF:  {metrics.min_dcf:.4f}")

    pos_scores = [r["score"] for r in scored_rows if r["label"] == 1]
    neg_scores = [r["score"] for r in scored_rows if r["label"] == 0]
    mean_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0
    mean_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0
    typer.echo(f"Mean positive score: {mean_pos:.4f}")
    typer.echo(f"Mean negative score: {mean_neg:.4f}")
    typer.echo(f"Score gap: {mean_pos - mean_neg:.4f}")

    # Write full verification report
    report = build_verification_evaluation_report(
        scored_rows,
        scores_path=str(scores_path),
        trials_path=str(EVAL_ROOT / "lists" / "trials.lst"),
    )
    write_verification_evaluation_report(report, output_root=out)

    # Save summary
    summary = {
        "model": model,
        "checkpoint": str(checkpoint),
        "dataset": "CN-Celeb",
        "eer": metrics.eer,
        "min_dcf": metrics.min_dcf,
        "trial_count": metrics.trial_count,
        "positive_count": metrics.positive_count,
        "negative_count": metrics.negative_count,
        "mean_positive_score": round(mean_pos, 6),
        "mean_negative_score": round(mean_neg, 6),
        "score_gap": round(mean_pos - mean_neg, 6),
    }
    summary_path = out / "cnceleb_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    typer.echo(f"\nReport: {out}")


if __name__ == "__main__":
    app()
