import argparse
import json

import torch
from src.dataset import SpeakerDataset
from src.metrics import precision_at_k
from src.model import ECAPASpeakerId
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(model: ECAPASpeakerId, loader, device: str):
    model.eval()
    all_emb = []
    all_lab = []
    for wave, lab in loader:
        wave = wave.to(device, non_blocking=True)
        emb = model.extract_embeddings(wave)
        all_emb.append(emb.cpu())
        all_lab.append(lab)
    return torch.cat(all_emb, dim=0), torch.cat(all_lab, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--chunk_seconds", type=float, default=6.0)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--ks", type=str, default="10")
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--data_base_dir", type=str, default="data")

    args = parser.parse_args()

    device = (
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    base_dir = args.data_base_dir
    ds = SpeakerDataset(
        args.csv,
        args.sample_rate,
        args.chunk_seconds,
        is_train=False,
        base_dir=base_dir,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = ECAPASpeakerId(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        embed_dim=args.embed_dim,
    ).to(device)

    sd = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)

    embeddings, labels = extract_embeddings(model, loader, device)

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    base_metrics: dict[str, float] = precision_at_k(embeddings.numpy(), labels.numpy(), ks=ks)

    print(json.dumps(base_metrics, indent=2))


if __name__ == "__main__":
    main()
