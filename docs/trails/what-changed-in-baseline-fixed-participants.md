# What Changed In `baseline_fixed_participants`

- Crop policy:
  - было: train получал фиксированный начальный кусок, val/test получали случайный кусок;
  - стало: train получает random crop, val/test получают deterministic center crop.
- Split:
  - было: random row split;
  - стало: speaker-disjoint split с контролем `min_val_utts >= 11`.
- Train/val ratio:
  - было: `split_ratio=0.2` фактически давал train 20%, val 80%;
  - стало: `train_ratio=0.98`, train `659804` rows, val `13473` rows.
- Reproducibility:
  - добавлены seed control и deterministic public inference.
- Export/inference:
  - ONNX теперь экспортирует только embeddings, без classifier logits;
  - ONNX opset поднят до `20`;
  - submission собран через cosine-equivalent L2-normalized embeddings и exact FAISS top-10.
