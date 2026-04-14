"""Build pseudo-label manifests from public cluster assignments."""

from __future__ import annotations

import argparse
import json

from kryptonite.eda.pseudo_labels import (
    DEFAULT_PRIOR_FEATURE_WEIGHTS,
    PseudoLabelSelectionConfig,
    build_pseudo_label_manifests,
)


def main() -> None:
    args = _parse_args()
    summary = build_pseudo_label_manifests(
        clusters_csv=args.clusters_csv,
        public_manifest_csv=args.public_manifest_csv,
        output_dir=args.output_dir,
        selection=PseudoLabelSelectionConfig(
            experiment_id=args.experiment_id,
            dataset_name=args.dataset_name,
            label_prefix=args.label_prefix,
            public_audio_prefix=args.public_audio_prefix,
            min_cluster_size=args.min_cluster_size,
            max_cluster_size=args.max_cluster_size,
            min_top1_score=args.min_top1_score,
            min_top1_margin=args.min_top1_margin,
            max_indegree_quantile=args.max_indegree_quantile,
            indegree_top_k=args.indegree_top_k,
            max_prior_distance_quantile=args.max_prior_distance_quantile,
            diversity_floor_quantile=args.diversity_floor_quantile,
            max_rows_per_cluster=args.max_rows_per_cluster,
            prior_feature_weights=_parse_prior_feature_weights(args.prior_feature_weight),
        ),
        base_train_manifest=args.base_train_manifest or None,
        topk_scores_npy=args.topk_scores_npy or None,
        topk_indices_npy=args.topk_indices_npy or None,
        public_stats_path=args.public_stats or None,
        prior_reference_stats_path=args.prior_reference_stats or None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-csv", required=True)
    parser.add_argument("--public-manifest-csv", required=True)
    parser.add_argument("--base-train-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--max-cluster-size", type=int, default=80)
    parser.add_argument("--topk-scores-npy", default="")
    parser.add_argument("--topk-indices-npy", default="")
    parser.add_argument("--min-top1-score", type=float, default=None)
    parser.add_argument("--min-top1-margin", type=float, default=None)
    parser.add_argument("--max-indegree-quantile", type=float, default=None)
    parser.add_argument("--indegree-top-k", type=int, default=10)
    parser.add_argument("--public-stats", default="")
    parser.add_argument("--prior-reference-stats", default="")
    parser.add_argument("--max-prior-distance-quantile", type=float, default=None)
    parser.add_argument("--diversity-floor-quantile", type=float, default=None)
    parser.add_argument("--max-rows-per-cluster", type=int, default=None)
    parser.add_argument(
        "--prior-feature-weight",
        action="append",
        default=[],
        help=(
            "Repeatable feature=weight override for prior-distance gating. "
            "Defaults to duration_s, non_silent_ratio, leading_silence_s, trailing_silence_s, "
            "spectral_bandwidth_hz, band_energy_ratio_3_8k."
        ),
    )
    parser.add_argument("--label-prefix", default="pseudo_g6_")
    parser.add_argument("--dataset-name", default="participants_g6_pseudo")
    parser.add_argument("--public-audio-prefix", default="datasets/Для участников")
    return parser.parse_args()


def _parse_prior_feature_weights(entries: list[str]) -> dict[str, float]:
    if not entries:
        return dict(DEFAULT_PRIOR_FEATURE_WEIGHTS)
    weights: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"prior feature weight must look like feature=weight, got {entry!r}")
        name, raw_value = entry.split("=", 1)
        feature_name = name.strip()
        if not feature_name:
            raise ValueError(f"prior feature name must not be empty in {entry!r}")
        weights[feature_name] = float(raw_value)
    return weights


if __name__ == "__main__":
    main()
