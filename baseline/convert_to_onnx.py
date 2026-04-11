import argparse
import json
import os

import torch
import torch.nn as nn
from src.ecapa import ECAPA_TDNN
from src.mel_frontend import MelFrontend
from src.model import ECAPASpeakerId


class ExportECAPA(nn.Module):
    """ONNX wrapper that exports embeddings by default."""

    def __init__(
        self,
        ecapa: ECAPA_TDNN,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        classifier: nn.Linear | None = None,
    ):
        super().__init__()
        self.frontend = MelFrontend(sample_rate, n_fft, hop_length, n_mels)
        self.ecapa = ecapa
        self.classifier = classifier

    def forward(self, waveform: torch.Tensor):
        feats = self.frontend(waveform)
        emb = self.ecapa(feats)
        if self.classifier is None:
            return emb
        logits = self.classifier(emb)
        return emb, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to train JSON config (e.g. configs/baseline.json)",
    )
    parser.add_argument(
        "--pt",
        type=str,
        default="",
        help="Path to .pt checkpoint (default: config.save_path or exp_dir/model.pt)",
    )
    parser.add_argument("--out", type=str, default="", help="Output ONNX path")
    parser.add_argument("--chunk_seconds", type=float, default=0.0)
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset version for export. Default follows current PyTorch default opset.",
    )
    parser.add_argument(
        "--include_logits",
        action="store_true",
        help="Export classifier logits as a second output. Default exports embeddings only.",
    )
    args = parser.parse_args()

    device = "cpu"
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)

    sample_rate = int(cfg.get("sample_rate", 16000))
    n_fft = int(cfg.get("n_fft", 400))
    hop_length = int(cfg.get("hop_length", 160))
    n_mels = int(cfg.get("n_mels", 80))
    embed_dim = int(cfg.get("embed_dim", 192))

    pt_path = args.pt or str(cfg.get("save_path") or "")
    if not pt_path and cfg.get("exp_dir"):
        pt_path = os.path.join(str(cfg["exp_dir"]), "model.pt")
    if not pt_path:
        raise ValueError("Checkpoint path is empty. Provide --pt or set save_path in config.")

    out_path = args.out or pt_path.replace(".pt", ".onnx")
    chunk_seconds = float(
        args.chunk_seconds
        or cfg.get("val_chunk_seconds")
        or cfg.get("chunk_seconds")
        or cfg.get("train_chunk_seconds")
        or 6.0
    )

    sd = torch.load(pt_path, map_location=device, weights_only=True)

    num_classes = None
    if num_classes is None and isinstance(sd, dict) and "classifier.weight" in sd:
        num_classes = int(sd["classifier.weight"].shape[0])
    if args.include_logits and num_classes is None:
        raise ValueError("num_classes is not specified and not inferable from checkpoint")

    model = ECAPASpeakerId(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        embed_dim=embed_dim,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    wrapper = (
        ExportECAPA(
            ecapa=model.ecapa,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            classifier=model.classifier if args.include_logits else None,
        )
        .to(device)
        .eval()
    )

    T = int(sample_rate * float(chunk_seconds))
    dummy = torch.randn(1, T, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    output_names = ["embeddings", "logits"] if args.include_logits else ["embeddings"]
    dynamic_axes = {
        "waveform": {0: "batch", 1: "time"},
        "embeddings": {0: "batch"},
    }
    if args.include_logits:
        dynamic_axes["logits"] = {0: "batch"}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            input_names=["waveform"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=args.opset,
            dynamo=False,
        )
    print(f"Saved ONNX to {out_path} with opset={args.opset}")


if __name__ == "__main__":
    main()
