"""Generate public graph/community postprocess submissions from cached B4 embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from kryptonite.eda.community import (
    CommunityConfig,
    LabelPropagationConfig,
    run_public_community_package,
)

DEFAULT_CONFIGS = [
    LabelPropagationConfig("C4_b8_labelprop_mutual10", edge_top=10, shared_min_count=0),
    LabelPropagationConfig("C5_b8_labelprop_mutual10_shared2", edge_top=10, shared_min_count=2),
    LabelPropagationConfig("C6_b8_labelprop_mutual15", edge_top=15, shared_min_count=0),
    CommunityConfig(
        "C1_b8_mutual20_component",
        edge_top=20,
        reciprocal_top=20,
        rank_top=100,
        component_max_size=120,
        component_min_candidates=6,
    ),
    CommunityConfig(
        "C2_b8_mutual30_component",
        edge_top=30,
        reciprocal_top=20,
        rank_top=100,
        component_max_size=160,
        component_min_candidates=6,
    ),
    CommunityConfig(
        "C3_b8_mutual50_component",
        edge_top=50,
        reciprocal_top=20,
        rank_top=100,
        component_max_size=220,
        component_min_candidates=6,
    ),
]


def main() -> None:
    args = _parse_args()
    run_public_community_package(
        embeddings_path=Path(args.embeddings_path),
        manifest_csv=Path(args.manifest_csv),
        template_csv=Path(args.template_csv),
        output_dir=Path(args.output_dir),
        configs=DEFAULT_CONFIGS,
        top_cache_k=args.top_cache_k,
        search_batch_size=args.search_batch_size,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-path",
        default="artifacts/eda/public_ablation_cycle/embeddings_B4_trim_3crop.npy",
    )
    parser.add_argument(
        "--manifest-csv",
        default="artifacts/eda/participants_public_baseline/test_public_manifest.csv",
    )
    parser.add_argument("--template-csv", default="datasets/Для участников/test_public.csv")
    parser.add_argument("--output-dir", default="artifacts/eda/public_graph_community")
    parser.add_argument("--top-cache-k", type=int, default=100)
    parser.add_argument("--search-batch-size", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    main()
