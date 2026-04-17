from collections.abc import Sequence

import faiss
import numpy as np


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = norms + eps
    return x / norms


def precision_at_k_from_indices(
    indices: np.ndarray, labels: np.ndarray, ks: int | Sequence[int] = (1, 10, 50)
) -> dict[str, float]:
    """
    Вычисляет Precision@K на основе заранее вычисленных индексов соседей.

    Parameters
    -----------
    indices : np.ndarray
        Массив индексов соседей размерности (N, M), где N — количество объектов,
        M — количество соседей для каждого объекта. Первый столбец может содержать
        сам объект (self).
    labels : np.ndarray
        Массив меток объектов размерности (N,). Используется для проверки,
        совпадает ли сосед по индексу с меткой текущего объекта.
    ks : int или Sequence[int], по умолчанию (1, 10, 50)
        Значение K или список значений K, для которых вычисляется Precision@K.

    Returns
    -----------
    Dict[str, float]
        Словарь, где ключи имеют вид "precision@K", а значения — вычисленные показатели Precision@K.
        Precision@K показывает долю соседей того же класса среди K ближайших соседей
        для каждого объекта, усреднённую по всем объектам. При наличии self-индекса
        он автоматически исключается из подсчёта.
    """
    neighbor_indices = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels)

    if neighbor_indices.ndim != 2:
        raise ValueError("indices must be a 2D array of shape (N, K+1)")
    n = int(neighbor_indices.shape[0])
    if y.shape[0] != n:
        raise ValueError(f"labels length {y.shape[0]} != indices rows {n}")

    if isinstance(ks, int):
        ks = (int(ks),)
    else:
        ks = tuple(int(k) for k in ks)

    if len(ks) == 0:
        raise ValueError("ks must contain at least one value")

    k_max = max(ks)
    row_ids = np.arange(n, dtype=neighbor_indices.dtype)[:, None]
    order = np.argsort(neighbor_indices == row_ids, axis=1, kind="stable")

    neighbor_indices = np.take_along_axis(neighbor_indices, order, axis=1)
    neighbor_indices = neighbor_indices[:, :k_max]

    matches = y[neighbor_indices] == y[:, None]
    pref = np.cumsum(matches, axis=1, dtype=np.int32)

    res = {}
    for k in ks:
        hits = pref[:, k - 1]
        res[f"precision@{k}"] = float((hits / k).mean())
    return res


def precision_at_k(
    embeddings: np.ndarray, labels: np.ndarray, ks: Sequence[int] = (1, 10, 50)
) -> dict[str, float]:
    """
    Вычисляет Precision@K для заданных эмбеддингов и меток.

    Parameters:
    -----------
    embeddings : np.ndarray
        Массив эмбеддингов объектов размерности (N, D), где N — количество объектов,
        D — размерность эмбеддингов.
    labels : np.ndarray
        Массив меток объектов размерности (N,). Используется для определения,
        совпадает ли ближайший сосед по эмбеддингам с тем же классом.
    ks : Sequence[int], по умолчанию (1, 10, 50)
        Список значений K, для которых вычисляется Precision@K.

    Returns
    -----------
    Dict[str, float]
        Словарь, где ключи имеют вид "precision@K", а значения — вычисленные показатели Precision@K.
        Precision@K показывает долю правильных соседей того же класса среди K ближайших
        объектов для каждой точки, усреднённую по всем объектам.
    """
    emb_np = np.asarray(embeddings, dtype=np.float32)
    labels_np = np.asarray(labels)

    n = int(emb_np.shape[0])
    if n <= 1:
        return {f"precision@{k}": 0.0 for k in ks}

    emb_norm = _l2_normalize_rows(emb_np).astype(np.float32, copy=False)

    index = faiss.IndexFlatIP(int(emb_norm.shape[1]))
    index.add(emb_norm)

    k_search = max(ks) + 1
    _, neighbor_indices = index.search(emb_norm, k_search)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    return precision_at_k_from_indices(neighbor_indices, labels_np, ks=ks)
