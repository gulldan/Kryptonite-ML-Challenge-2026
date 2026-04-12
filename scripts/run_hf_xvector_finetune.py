"""Fine-tune a Hugging Face AudioXVector speaker model on challenge manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.training.hf_xvector import (
    load_hf_xvector_finetune_config,
    run_hf_xvector_finetune,
)


def main() -> None:
    args = _parse_args()
    config = load_hf_xvector_finetune_config(args.config)
    artifacts = run_hf_xvector_finetune(
        config,
        config_path=args.config,
        device_override=args.device or None,
        run_id_override=args.run_id or None,
    )
    payload = artifacts.to_dict()
    if args.output == "json":
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return
    print(
        "\n".join(
            [
                "Hugging Face AudioXVector fine-tune complete",
                f"Run id: {artifacts.run_id}",
                f"Output root: {artifacts.output_root}",
                f"Model dir: {artifacts.model_dir}",
                f"Checkpoint: {artifacts.checkpoint_path}",
                f"Metrics: {artifacts.metrics_path}",
                f"Embedding size: {artifacts.embedding_size}",
                f"Speaker count: {artifacts.speaker_count}",
                f"Train rows: {artifacts.train_row_count}",
            ]
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser.parse_args()


if __name__ == "__main__":
    main()
