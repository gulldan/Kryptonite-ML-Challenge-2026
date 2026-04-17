from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.official_campp_tail import _try_extract_embeddings_from_contiguous_frontend_pack


def test_contiguous_frontend_pack_fast_path_averages_segments_by_manifest_owner() -> None:
    pack = SimpleNamespace(
        features=np.array(
            [
                [[1.0], [1.0]],
                [[3.0], [3.0]],
                [[5.0], [5.0]],
            ],
            dtype=np.float32,
        ),
        row_offsets=np.array([0, 2], dtype=np.int64),
        row_counts=np.array([2, 1], dtype=np.int32),
    )
    manifest = pl.DataFrame({"gallery_index": [0, 1]})
    sums: dict[int, list[np.ndarray]] = {}

    def encoder(batch: Any) -> np.ndarray:
        return batch.numpy().mean(axis=(1, 2), keepdims=False)[:, None]

    used_fast_path = _try_extract_embeddings_from_contiguous_frontend_pack(
        pack=pack,
        manifest=manifest,
        encoder=encoder,
        sums=sums,
        batch_size=2,
        started_at=0.0,
        log_every_rows=100,
    )

    assert used_fast_path
    np.testing.assert_array_equal(sums[0][0], np.array([2.0], dtype=np.float32))
    np.testing.assert_array_equal(sums[1][0], np.array([5.0], dtype=np.float32))
