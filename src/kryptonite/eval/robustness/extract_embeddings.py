from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from kryptonite.eda.official_campp_tail.config import OfficialCamPPTailConfig
from kryptonite.eda.official_campp_tail.extraction import load_or_extract_embeddings as load_campp
from kryptonite.eval.robustness.benchmark import (
    ModelSpec,
    _extract_teacher_peft_embeddings,
)
from kryptonite.models.scoring import l2_normalize_embeddings


def main() -> None:
    args = _parse_args()
    manifest = pl.read_csv(args.manifest_csv)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.model_kind == "campp":
        config = OfficialCamPPTailConfig(
            checkpoint_path=args.checkpoint_path,
            manifest_csv=args.manifest_csv,
            output_dir=str(output_path.parent),
            experiment_id=args.condition_name,
            data_root=args.data_root,
            device=args.device,
            search_device=args.search_device,
            batch_size=args.campp_batch_size,
            frontend_workers=args.campp_frontend_workers,
            frontend_prefetch=args.campp_frontend_prefetch,
            frontend_cache_dir=args.campp_frontend_cache_dir,
            frontend_cache_mode="readwrite",
        )
        embeddings = load_campp(config, manifest, output_path.parent)
        embeddings = l2_normalize_embeddings(
            embeddings,
            field_name="campp_embeddings",
        ).astype(np.float32, copy=False)
        np.save(output_path, embeddings)
        return

    model = ModelSpec(
        key=args.model_key,
        label=args.model_key,
        kind=args.model_kind,
        checkpoint_path=Path(args.checkpoint_path),
        backbone_path=(None if not args.backbone_path else Path(args.backbone_path)),
    )
    _extract_teacher_peft_embeddings(
        model=model,
        manifest=manifest,
        output_path=output_path,
        device=args.device,
        batch_size=args.w2v_batch_size,
        num_workers=args.w2v_num_workers,
        prefetch_factor=args.w2v_prefetch_factor,
        crop_seconds=args.w2v_crop_seconds,
        n_crops=args.w2v_n_crops,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--backbone-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-device", default="cuda")
    parser.add_argument("--campp-batch-size", type=int, default=64)
    parser.add_argument("--campp-frontend-workers", type=int, default=8)
    parser.add_argument("--campp-frontend-prefetch", type=int, default=128)
    parser.add_argument("--campp-frontend-cache-dir", required=True)
    parser.add_argument("--w2v-batch-size", type=int, default=1024)
    parser.add_argument("--w2v-num-workers", type=int, default=4)
    parser.add_argument("--w2v-prefetch-factor", type=int, default=1)
    parser.add_argument("--w2v-crop-seconds", type=float, default=6.0)
    parser.add_argument("--w2v-n-crops", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
