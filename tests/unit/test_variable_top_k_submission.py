from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from kryptonite.eda.community import LabelPropagationConfig, label_propagation_rerank

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_run_sh_dry_run_forwards_top_k_to_w2v_runner() -> None:
    env = os.environ.copy()
    env["USE_UV_RUN"] = "0"

    result = subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "run.sh"),
            "--container-only",
            "--model",
            "w2v-trt",
            "--test-csv",
            "/tmp/missing-test.csv",
            "--data-root",
            "/tmp/missing-data-root",
            "--top-k",
            "42",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--output-top-k 42" in result.stderr


def test_run_sh_rejects_top_k_outside_platform_range() -> None:
    result = subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "run.sh"),
            "--container-only",
            "--test-csv",
            "/tmp/missing-test.csv",
            "--data-root",
            "/tmp/missing-data-root",
            "--top-k",
            "9",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "10 <= K < 1000" in result.stderr


def test_label_propagation_rerank_expands_rank_top_to_output_top_k() -> None:
    indices = np.array(
        [
            [1, 2, 3],
            [0, 2, 3],
            [3, 0, 1],
            [2, 1, 0],
        ],
        dtype=np.int64,
    )
    scores = np.array(
        [
            [0.9, 0.8, 0.7],
            [0.9, 0.8, 0.7],
            [0.9, 0.8, 0.7],
            [0.9, 0.8, 0.7],
        ],
        dtype=np.float32,
    )

    top_indices, top_scores, meta = label_propagation_rerank(
        indices=indices,
        scores=scores,
        config=LabelPropagationConfig(
            "variable_k",
            edge_top=1,
            reciprocal_top=1,
            rank_top=1,
            label_min_size=99,
            label_max_size=100,
            label_min_candidates=99,
        ),
        top_k=3,
    )

    assert top_indices.shape == (4, 3)
    assert top_scores.shape == (4, 3)
    assert meta["rank_top"] == 3
