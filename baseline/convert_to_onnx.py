import argparse
import json
import os
import torch
import torch.nn as nn

from src.model import ECAPASpeakerId
from src.ecapa import ECAPA_TDNN
from src.mel_frontend import MelFrontend


class ExportECAPA(nn.Module):
    """Обёртка для экспорта модели в ONNX."""
    
    def __init__(self, ecapa: ECAPA_TDNN, classifier: nn.Linear, 
                 sample_rate: int, n_fft: int, hop_length: int, n_mels: int):
        super().__init__()
        self.frontend = MelFrontend(sample_rate, n_fft, hop_length, n_mels)
        self.ecapa = ecapa
        self.classifier = classifier

    def forward(self, waveform: torch.Tensor):
        feats = self.frontend(waveform)
        emb = self.ecapa(feats)
        logits = self.classifier(emb)
        return emb, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to train JSON config (e.g. configs/train_ecapa.json)")
    parser.add_argument("--pt", type=str, default="", help="Path to .pt checkpoint (default: config.save_path)")
    args = parser.parse_args()

    device = "cpu"
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)

    sample_rate = int(cfg.get("sample_rate", 16000))
    n_fft = int(cfg.get("n_fft", 400))
    hop_length = int(cfg.get("hop_length", 160))
    n_mels = int(cfg.get("n_mels", 80))
    embed_dim = int(cfg.get("embed_dim", 192))

    pt_path = args.pt or str(cfg.get("save_path", ""))
    if not pt_path:
        raise ValueError("Checkpoint path is empty. Provide --pt or set save_path in config.")

    out_path = pt_path.replace(".pt", ".onnx")

    chunk_seconds = 3

    sd = torch.load(pt_path, map_location=device, weights_only=False)

    num_classes = None
    if num_classes is None and isinstance(sd, dict) and "classifier.weight" in sd:
        num_classes = int(sd["classifier.weight"].shape[0])
    if num_classes is None:
        raise ValueError("num_classes is not specified and not inferable from checkpoint")

    # Загружаем оригинальную модель для получения весов
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

    # Создаём ONNX-совместимую обёртку
    wrapper = ExportECAPA(
        ecapa=model.ecapa,
        classifier=model.classifier,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    ).to(device).eval()

    T = int(sample_rate * float(chunk_seconds))
    dummy = torch.randn(1, T, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            input_names=["waveform"],
            output_names=["embeddings", "logits"],
            dynamic_axes={
                "waveform": {0: "batch", 1: "time"},
                "embeddings": {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
        )
    print(f"Saved ONNX to {out_path}")


if __name__ == "__main__":
    main()

