from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from kryptonite.eda.pseudo_labels import (
    PseudoLabelSelectionConfig,
    build_pseudo_label_manifests,
    select_pseudo_label_rows,
)


def test_build_pseudo_label_manifests_writes_default_outputs(tmp_path: Path) -> None:
    clusters_csv = tmp_path / "clusters.csv"
    public_manifest_csv = tmp_path / "public_manifest.csv"
    base_manifest = tmp_path / "train_manifest.jsonl"
    output_dir = tmp_path / "pseudo"

    pl.DataFrame(
        {
            "cluster_id": [1, 1, 1, 2, 2],
            "cluster_size": [3, 3, 3, 2, 2],
            "row_index": [0, 1, 2, 3, 4],
        }
    ).write_csv(clusters_csv)
    pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3, 4],
            "filepath": [f"test_public/{index:06d}.flac" for index in range(5)],
        }
    ).write_csv(public_manifest_csv)
    base_manifest.write_text('{"speaker_id":"real_000001","audio_path":"datasets/train/a.flac"}\n')

    summary = build_pseudo_label_manifests(
        clusters_csv=clusters_csv,
        public_manifest_csv=public_manifest_csv,
        output_dir=output_dir,
        selection=PseudoLabelSelectionConfig(
            experiment_id="control",
            min_cluster_size=2,
            max_cluster_size=3,
        ),
        base_train_manifest=base_manifest,
    )

    pseudo_manifest = output_dir / "control_pseudo_manifest.jsonl"
    mixed_manifest = output_dir / "control_mixed_train_manifest.jsonl"
    selected_pool = output_dir / "control_selected_pool.csv"

    assert pseudo_manifest.is_file()
    assert mixed_manifest.is_file()
    assert selected_pool.is_file()
    assert summary["pseudo_row_count"] == 5
    assert summary["pseudo_cluster_count"] == 2
    assert summary["mixed_row_count"] == 6
    assert summary["selection_stage_counts"] == [
        {"stage": "initial", "row_count": 5, "cluster_count": 2},
        {"stage": "cluster_size", "row_count": 5, "cluster_count": 2},
    ]

    first_row = json.loads(pseudo_manifest.read_text(encoding="utf-8").splitlines()[0])
    assert first_row["speaker_id"] == "pseudo_g6_000001"
    assert first_row["audio_path"] == "datasets/Для участников/test_public/000000.flac"


def test_pseudo_label_selector_applies_score_hubness_and_cluster_cap() -> None:
    clusters = pl.DataFrame(
        {
            "cluster_id": [10, 10, 10, 20, 20, 20],
            "cluster_size": [3, 3, 3, 3, 3, 3],
            "row_index": [0, 1, 2, 3, 4, 5],
        }
    )
    public_manifest = pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3, 4, 5],
            "filepath": [f"test_public/{index:06d}.flac" for index in range(6)],
        }
    )
    topk_scores = np.asarray(
        [
            [0.95, 0.60],
            [0.92, 0.70],
            [0.65, 0.40],
            [0.96, 0.50],
            [0.90, 0.75],
            [0.88, 0.87],
        ],
        dtype=np.float32,
    )
    topk_indices = np.asarray(
        [
            [1, 2],
            [0, 2],
            [0, 1],
            [4, 5],
            [3, 5],
            [3, 4],
        ],
        dtype=np.int64,
    )

    selected, summary = select_pseudo_label_rows(
        clusters=clusters,
        public_manifest=public_manifest,
        selection=PseudoLabelSelectionConfig(
            experiment_id="tight",
            min_cluster_size=3,
            max_cluster_size=3,
            min_top1_score=0.8,
            min_top1_margin=0.1,
            max_indegree_quantile=0.5,
            indegree_top_k=1,
            max_rows_per_cluster=1,
        ),
        topk_scores=topk_scores,
        topk_indices=topk_indices,
    )

    assert selected.get_column("row_index").to_list() == [1, 4]
    assert selected.get_column("cluster_id").to_list() == [10, 20]
    assert summary["selection_stage_counts"] == [
        {"stage": "initial", "row_count": 6, "cluster_count": 2},
        {"stage": "cluster_size", "row_count": 6, "cluster_count": 2},
        {"stage": "top1_score", "row_count": 5, "cluster_count": 2},
        {"stage": "top1_margin", "row_count": 4, "cluster_count": 2},
        {"stage": "indegree_at_1", "row_count": 2, "cluster_count": 2},
        {"stage": "cluster_cap", "row_count": 2, "cluster_count": 2},
    ]
    assert "top1_score" in selected.columns
    assert "top1_margin" in selected.columns
    assert "indegree_at_1" in selected.columns


def test_prior_distance_filter_uses_bucket_floor_to_keep_regime_coverage() -> None:
    clusters = pl.DataFrame(
        {
            "cluster_id": [10, 10, 10, 10],
            "cluster_size": [4, 4, 4, 4],
            "row_index": [0, 1, 2, 3],
        }
    )
    public_manifest = pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "filepath": [f"test_public/{index:06d}.flac" for index in range(4)],
        }
    )
    public_stats = pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "filepath": [f"test_public/{index:06d}.flac" for index in range(4)],
            "duration_s": [1.0, 1.1, 9.8, 10.2],
        }
    )
    reference_stats = pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "filepath": [f"ref/{index:06d}.flac" for index in range(4)],
            "duration_s": [1.0, 1.05, 1.1, 1.15],
        }
    )

    selected_without_floor, _ = select_pseudo_label_rows(
        clusters=clusters,
        public_manifest=public_manifest,
        public_stats=public_stats,
        prior_reference_stats=reference_stats,
        selection=PseudoLabelSelectionConfig(
            experiment_id="prior_no_floor",
            min_cluster_size=4,
            max_cluster_size=4,
            max_prior_distance_quantile=0.25,
            prior_feature_weights={"duration_s": 1.0},
        ),
    )
    selected_with_floor, _ = select_pseudo_label_rows(
        clusters=clusters,
        public_manifest=public_manifest,
        public_stats=public_stats,
        prior_reference_stats=reference_stats,
        selection=PseudoLabelSelectionConfig(
            experiment_id="prior_floor",
            min_cluster_size=4,
            max_cluster_size=4,
            max_prior_distance_quantile=0.25,
            diversity_floor_quantile=0.5,
            prior_feature_weights={"duration_s": 1.0},
        ),
    )

    assert selected_without_floor.height < selected_with_floor.height
    assert selected_without_floor.height == 1
    assert selected_without_floor.get_column("duration_s").max() < 2.0
    assert selected_with_floor.height >= 2
    assert selected_with_floor.get_column("duration_s").min() < 2.0
    assert selected_with_floor.get_column("duration_s").max() > 9.0
    assert "prior_distance" in selected_with_floor.columns
    assert "acoustic_bucket" in selected_with_floor.columns
