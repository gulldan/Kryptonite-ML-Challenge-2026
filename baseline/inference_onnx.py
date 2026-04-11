import torch
import argparse
import json
import os
import time
import csv
import faiss
import numpy as np
import onnxruntime as ort

from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SpeakerDataset
from src.metrics import _l2_normalize_rows


def get_onnx_providers(device: str = "auto") -> List[str]:
    """Получить список провайдеров для ONNX Runtime."""
    available = ort.get_available_providers()

    if device == "cpu":
        return ["CPUExecutionProvider"]

    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    return providers if providers else ["CPUExecutionProvider"]


def save_topk_indices_csv(
    embeddings: np.ndarray,
    filepaths: List[str],
    out_csv: str,
    topk: int = 10,
) -> None:
    """
    Сохраняет submission CSV с top-k ближайшими соседями для каждого файла.

    Формат:
    - колонка filepath: путь к файлу из входного CSV.
    - колонка neighbours: top-k индексов ближайших соседей через запятую.
    """
    k_search = topk + 1

    emb = np.asarray(embeddings, dtype=np.float32)
    emb = _l2_normalize_rows(emb).astype(np.float32, copy=False)

    index = faiss.IndexFlatIP(int(emb.shape[1]))
    index.add(emb)
    _, I = index.search(emb, k_search)
    I = np.asarray(I, dtype=np.int64)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "neighbours"])
        for row_idx, (filepath, row) in enumerate(zip(filepaths, I)):
            neighbours = [int(x) for x in row.tolist() if x >= 0 and x != row_idx][:topk]
            w.writerow([filepath, ",".join(str(x) for x in neighbours)])


def run_session(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    inputs = session.get_inputs()
    input_name = inputs[0].name
    outputs = session.run(None, {input_name: x.astype(np.float32)})
    return outputs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output_emb", type=str, default="emb.npy")
    parser.add_argument("--output_labels", type=str, default="labels.npy")
    parser.add_argument(
        "--output_indices",
        type=str,
        default="submission.csv",
        help="Путь для сохранения submission CSV (default: submission.csv)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Сколько соседей сохранять (default: 10)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--chunk_seconds", type=float, default=6.0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Устройство для инференса: auto, cuda или cpu",
    )
    parser.add_argument(
        "--filepath_col",
        type=str,
        default="filepath",
        help="Имя колонки с путями к файлам (default: filepath)",
    )
    parser.add_argument(
        "--speaker_id_col",
        type=str,
        default="speaker_id",
        help="Имя колонки со speaker_id (default: speaker_id)",
    )
    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="",
        help="Базовая директория для относительных путей (default: директория CSV)",
    )
    args = parser.parse_args()

    data_base_dir = (
        args.data_base_dir
        if args.data_base_dir
        else os.path.dirname(os.path.abspath(args.csv))
    )
    ds = SpeakerDataset(
        args.csv,
        args.sample_rate,
        args.chunk_seconds,
        is_train=False,
        base_dir=data_base_dir,
        filepath_col=args.filepath_col,
        speaker_id_col=args.speaker_id_col,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Получаем провайдеры для выбранного устройства
    providers = get_onnx_providers(args.device)
    print(f"[INFO] Доступные ONNX провайдеры: {ort.get_available_providers()}")
    print(f"[INFO] Используемые провайдеры: {providers}")

    sess = ort.InferenceSession(args.onnx_path, providers=providers)
    actual_provider = sess.get_providers()[0]
    print(f"[INFO] Активный провайдер: {actual_provider}")

    all_emb, all_lab = [], []
    t_wall_start = time.perf_counter()
    for wave, lab in tqdm(loader, total=len(loader)):
        x = wave.numpy()
        emb = run_session(sess, x)
        all_emb.append(emb)
        all_lab.append(np.asarray(lab))
    emb_np = np.concatenate(all_emb, axis=0)
    labels_np = np.concatenate(all_lab, axis=0, dtype=np.int64)

    out_dir = os.path.dirname(args.output_emb)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output_emb, emb_np)

    # Сохраняем индексы ближайших соседей
    indices_path = args.output_indices
    if not indices_path:
        indices_path = os.path.join(out_dir or ".", indices_path)
    filepaths = ds.df[args.filepath_col].astype(str).tolist()
    save_topk_indices_csv(emb_np, filepaths, indices_path, topk=args.topk)
    t_wall_end = time.perf_counter()
    print(f"[INFO] Inference + indexing wall time: {t_wall_end - t_wall_start:.3f}s")
    print(json.dumps({"saved_indices": indices_path, "topk": int(args.topk)}, indent=2))

    # Сохраняем labels только если они есть в CSV (не все -1)
    if ds.has_sid:
        np.save(args.output_labels, labels_np)
        print(
            json.dumps(
                {
                    "saved_emb": args.output_emb,
                    "saved_labels": args.output_labels,
                    "count": int(emb_np.shape[0]),
                    "dim": int(emb_np.shape[1]),
                }
            )
        )
    else:
        print(
            json.dumps(
                {
                    "saved_emb": args.output_emb,
                    "count": int(emb_np.shape[0]),
                    "dim": int(emb_np.shape[1]),
                }
            )
        )


if __name__ == "__main__":
    main()
