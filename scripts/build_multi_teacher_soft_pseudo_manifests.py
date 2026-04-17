"""Build confidence-weighted soft pseudo labels from multiple public teacher graphs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.community import ClusterFirstConfig, cluster_first_rerank, mutual_mask
from kryptonite.eda.soft_pseudo_stability import stability_scores


@dataclass(frozen=True, slots=True)
class TeacherSpec:
    name: str
    indices_path: str
    scores_path: str
    weight: float = 1.0


@dataclass(slots=True)
class TeacherCache:
    spec: TeacherSpec
    indices: np.ndarray
    scores: np.ndarray
    zscores: np.ndarray
    mutual: np.ndarray
    shared: np.ndarray
    hub_z: np.ndarray


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.public_manifest_csv)
    teacher_specs = [_parse_teacher_spec(value) for value in args.teacher]
    if len(teacher_specs) < 2:
        raise ValueError("At least two --teacher specs are required.")

    teachers = _load_teachers(
        specs=teacher_specs,
        manifest_height=manifest.height,
        source_top_k=args.source_top_k,
        reciprocal_top=args.reciprocal_top,
        shared_edge_top=args.shared_edge_top,
        shared_top=args.shared_top,
    )
    print(
        f"[soft-pseudo] loaded teachers={','.join(t.spec.name for t in teachers)} "
        f"rows={manifest.height}",
        flush=True,
    )

    fused_indices, fused_scores, agreement, fusion_meta = _fuse_teachers(
        teachers=teachers,
        output_top_k=args.top_cache_k,
        score_z_weight=args.score_z_weight,
        raw_score_weight=args.raw_score_weight,
        rank_weight=args.rank_weight,
        reciprocal_bonus=args.reciprocal_bonus,
        agreement_bonus=args.agreement_bonus,
        shared_weight=args.shared_weight,
        hubness_penalty=args.hubness_penalty,
    )
    np.save(output_dir / f"indices_{args.experiment_id}_top{args.top_cache_k}.npy", fused_indices)
    np.save(output_dir / f"scores_{args.experiment_id}_top{args.top_cache_k}.npy", fused_scores)

    cluster_config = ClusterFirstConfig(
        experiment_id=args.experiment_id,
        edge_top=args.cluster_edge_top,
        reciprocal_top=args.cluster_reciprocal_top,
        rank_top=args.cluster_rank_top,
        iterations=args.cluster_iterations,
        cluster_min_size=args.cluster_min_size,
        cluster_max_size=args.cluster_max_size,
        cluster_min_candidates=args.cluster_min_candidates,
        shared_top=args.cluster_shared_top,
        shared_min_count=args.cluster_shared_min_count,
        reciprocal_bonus=args.cluster_reciprocal_bonus,
        density_penalty=args.cluster_density_penalty,
        edge_score_quantile=args.cluster_edge_score_quantile,
        edge_min_score=args.cluster_edge_min_score,
        shared_weight=args.cluster_shared_weight,
        rank_weight=args.cluster_rank_weight,
        self_weight=args.cluster_self_weight,
        label_size_penalty=args.cluster_label_size_penalty,
        split_oversized=not args.no_split_oversized,
        split_edge_top=args.split_edge_top,
    )
    _, _, labels, cluster_meta = cluster_first_rerank(
        indices=fused_indices,
        scores=fused_scores,
        config=cluster_config,
        top_k=10,
    )
    stability, stability_meta = stability_scores(
        args=args,
        teachers=teachers,
        main_indices=fused_indices,
        main_labels=labels,
        fuse_teachers=_fuse_teachers,
    )

    pseudo_path = output_dir / f"{args.experiment_id}_pseudo_manifest.jsonl"
    mixed_path = output_dir / f"{args.experiment_id}_mixed_train_manifest.jsonl"
    audit_path = output_dir / f"{args.experiment_id}_row_audit.csv"
    selected, pseudo_summary = _write_soft_manifests(
        manifest=manifest,
        labels=labels,
        fused_indices=fused_indices,
        fused_scores=fused_scores,
        agreement=agreement,
        stability=stability,
        output_pseudo_path=pseudo_path,
        output_mixed_path=mixed_path,
        output_audit_path=audit_path,
        base_train_manifest=Path(args.base_train_manifest),
        label_prefix=args.label_prefix,
        dataset_name=args.dataset_name,
        public_audio_prefix=args.public_audio_prefix,
        soft_top_clusters=args.soft_top_clusters,
        soft_rank_top=args.soft_rank_top,
        self_soft_weight=args.self_soft_weight,
        min_pseudo_cluster_size=args.min_pseudo_cluster_size,
        max_pseudo_cluster_size=args.max_pseudo_cluster_size,
        min_teacher_agreement=args.min_teacher_agreement,
        min_confidence=args.min_confidence,
        min_stability=args.min_stability,
    )

    summary: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "public_manifest_csv": args.public_manifest_csv,
        "base_train_manifest": args.base_train_manifest,
        "teacher_specs": [asdict(spec) for spec in teacher_specs],
        "fusion_meta": fusion_meta,
        "cluster_config": asdict(cluster_config),
        "cluster_meta": cluster_meta,
        "stability_meta": stability_meta,
        "pseudo_summary": pseudo_summary,
        "selected_row_count": int(selected.sum()),
        "pseudo_manifest": str(pseudo_path),
        "mixed_train_manifest": str(mixed_path),
        "row_audit_csv": str(audit_path),
    }
    summary_path = output_dir / f"{args.experiment_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _load_teachers(
    *,
    specs: list[TeacherSpec],
    manifest_height: int,
    source_top_k: int,
    reciprocal_top: int,
    shared_edge_top: int,
    shared_top: int,
) -> list[TeacherCache]:
    caches: list[TeacherCache] = []
    for spec in specs:
        indices = np.load(spec.indices_path)
        scores = np.load(spec.scores_path)
        if indices.shape != scores.shape:
            raise ValueError(f"{spec.name}: indices/scores shape mismatch.")
        if indices.ndim != 2 or indices.shape[0] != manifest_height:
            raise ValueError(f"{spec.name}: expected [{manifest_height}, k], got {indices.shape}.")
        width = min(source_top_k, indices.shape[1])
        if width <= 0:
            raise ValueError(f"{spec.name}: empty top-k cache.")
        indices = indices[:, :width].astype(np.int64, copy=False)
        scores = scores[:, :width].astype(np.float32, copy=False)
        row_mean = scores.mean(axis=1, keepdims=True)
        row_std = np.clip(scores.std(axis=1, keepdims=True), 1e-6, None)
        zscores = ((scores - row_mean) / row_std).astype(np.float32, copy=False)
        mutual = mutual_mask(
            indices,
            edge_top=width,
            reciprocal_top=min(reciprocal_top, indices.shape[1]),
        )
        shared = _shared_counts(
            indices=indices,
            edge_top=min(shared_edge_top, width),
            shared_top=min(shared_top, width),
        )
        indegree = np.bincount(indices.ravel(), minlength=manifest_height).astype(np.float32)
        hub_values = np.log1p(indegree)
        hub_z = ((hub_values - hub_values.mean()) / max(float(hub_values.std()), 1e-6)).astype(
            np.float32,
            copy=False,
        )
        caches.append(
            TeacherCache(
                spec=spec,
                indices=indices,
                scores=scores,
                zscores=zscores,
                mutual=mutual,
                shared=shared,
                hub_z=hub_z,
            )
        )
    return caches


def _shared_counts(*, indices: np.ndarray, edge_top: int, shared_top: int) -> np.ndarray:
    out = np.zeros((indices.shape[0], indices.shape[1]), dtype=np.int16)
    if edge_top <= 0 or shared_top <= 0:
        return out
    neighbor_sets = [set(row) for row in indices[:, :shared_top].tolist()]
    for row in range(indices.shape[0]):
        row_set = neighbor_sets[row]
        for position, candidate in enumerate(indices[row, :edge_top].tolist()):
            out[row, position] = len(row_set.intersection(neighbor_sets[int(candidate)]))
    return out


def _fuse_teachers(
    *,
    teachers: list[TeacherCache],
    output_top_k: int,
    score_z_weight: float,
    raw_score_weight: float,
    rank_weight: float,
    reciprocal_bonus: float,
    agreement_bonus: float,
    shared_weight: float,
    hubness_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    row_count = teachers[0].indices.shape[0]
    out_indices = np.empty((row_count, output_top_k), dtype=np.int64)
    out_scores = np.empty((row_count, output_top_k), dtype=np.float32)
    out_agreement = np.empty((row_count, output_top_k), dtype=np.int16)
    rank_bonuses = [
        rank_weight / np.sqrt(np.arange(teacher.indices.shape[1], dtype=np.float32) + 1.0)
        for teacher in teachers
    ]

    for row in range(row_count):
        scores_by_candidate: dict[int, float] = {}
        seen_by_candidate: dict[int, int] = {}
        for teacher_index, teacher in enumerate(teachers):
            weight = teacher.spec.weight
            candidates = teacher.indices[row]
            base = (
                (score_z_weight * teacher.zscores[row])
                + (raw_score_weight * teacher.scores[row])
                + rank_bonuses[teacher_index]
                + (reciprocal_bonus * teacher.mutual[row].astype(np.float32))
                + (shared_weight * teacher.shared[row].astype(np.float32))
                - (hubness_penalty * teacher.hub_z[candidates])
            )
            for candidate, score in zip(candidates.tolist(), base.tolist(), strict=True):
                candidate = int(candidate)
                scores_by_candidate[candidate] = scores_by_candidate.get(candidate, 0.0) + (
                    weight * float(score)
                )
                seen_by_candidate[candidate] = seen_by_candidate.get(candidate, 0) + 1
        ranked = sorted(
            (
                (
                    candidate,
                    score + (agreement_bonus * max(seen_by_candidate[candidate] - 1, 0)),
                    seen_by_candidate[candidate],
                )
                for candidate, score in scores_by_candidate.items()
                if candidate != row
            ),
            key=lambda item: (item[1], item[2], -item[0]),
            reverse=True,
        )
        if len(ranked) < output_top_k:
            raise ValueError(f"Row {row} has only {len(ranked)} fused candidates.")
        top = ranked[:output_top_k]
        out_indices[row] = np.asarray([item[0] for item in top], dtype=np.int64)
        out_scores[row] = np.asarray([item[1] for item in top], dtype=np.float32)
        out_agreement[row] = np.asarray([item[2] for item in top], dtype=np.int16)
        if row == 0 or (row + 1) % max(row_count // 20, 1) == 0 or row + 1 == row_count:
            print(f"[soft-pseudo] fused rows={row + 1}/{row_count}", flush=True)

    return (
        out_indices,
        out_scores,
        out_agreement,
        {
            "teacher_count": len(teachers),
            "output_top_k": output_top_k,
            "score_z_weight": score_z_weight,
            "raw_score_weight": raw_score_weight,
            "rank_weight": rank_weight,
            "reciprocal_bonus": reciprocal_bonus,
            "agreement_bonus": agreement_bonus,
            "shared_weight": shared_weight,
            "hubness_penalty": hubness_penalty,
            "agreement_top1_mean": float(out_agreement[:, 0].mean()),
            "agreement_top10_mean": float(out_agreement[:, :10].mean()),
        },
    )


def _write_soft_manifests(
    *,
    manifest: pl.DataFrame,
    labels: np.ndarray,
    fused_indices: np.ndarray,
    fused_scores: np.ndarray,
    agreement: np.ndarray,
    stability: np.ndarray,
    output_pseudo_path: Path,
    output_mixed_path: Path,
    output_audit_path: Path,
    base_train_manifest: Path,
    label_prefix: str,
    dataset_name: str,
    public_audio_prefix: str,
    soft_top_clusters: int,
    soft_rank_top: int,
    self_soft_weight: float,
    min_pseudo_cluster_size: int,
    max_pseudo_cluster_size: int,
    min_teacher_agreement: float,
    min_confidence: float,
    min_stability: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    cluster_sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    teacher_agreement = agreement[:, : min(20, agreement.shape[1])].max(axis=1) / max(
        float(agreement.max()),
        1.0,
    )
    density = fused_scores[:, : min(20, fused_scores.shape[1])].mean(axis=1)
    density_score = _percentile_score(density)
    soft_labels, soft_probs, margins = _soft_cluster_targets(
        labels=labels,
        cluster_sizes=cluster_sizes,
        fused_indices=fused_indices,
        fused_scores=fused_scores,
        soft_top_clusters=soft_top_clusters,
        soft_rank_top=soft_rank_top,
        self_soft_weight=self_soft_weight,
        min_pseudo_cluster_size=min_pseudo_cluster_size,
        max_pseudo_cluster_size=max_pseudo_cluster_size,
    )
    margin_score = np.clip(margins / max(float(np.quantile(margins, 0.90)), 1e-6), 0.0, 1.0)
    confidence = (
        (0.40 * teacher_agreement)
        + (0.25 * margin_score)
        + (0.20 * density_score)
        + (0.15 * stability)
    ).astype(np.float32, copy=False)
    selected = _selected_rows(
        labels=labels,
        cluster_sizes=cluster_sizes,
        confidence=confidence,
        stability=stability,
        teacher_agreement=teacher_agreement,
        min_pseudo_cluster_size=min_pseudo_cluster_size,
        max_pseudo_cluster_size=max_pseudo_cluster_size,
        min_teacher_agreement=min_teacher_agreement,
        min_confidence=min_confidence,
        min_stability=min_stability,
    )
    selected_clusters = np.zeros(cluster_sizes.shape[0], dtype=bool)
    selected_clusters[np.unique(labels[selected])] = True
    allowed_labels = set(np.flatnonzero(selected_clusters).tolist())
    filepaths = manifest["filepath"].cast(pl.Utf8).to_list()

    rows: list[dict[str, Any]] = []
    with output_pseudo_path.open("w", encoding="utf-8") as handle:
        for row_index in np.flatnonzero(selected).tolist():
            filtered = [
                (int(label), float(prob))
                for label, prob in zip(soft_labels[row_index], soft_probs[row_index], strict=True)
                if int(label) in allowed_labels and float(prob) > 0.0
            ]
            if not filtered:
                continue
            total_prob = sum(prob for _, prob in filtered)
            filtered = [(label, prob / total_prob) for label, prob in filtered]
            primary_label = int(filtered[0][0])
            speaker_ids = [f"{label_prefix}{label:06d}" for label, _ in filtered]
            payload = {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": dataset_name,
                "source_dataset": "test_public_multi_teacher_soft_pseudo",
                "speaker_id": f"{label_prefix}{primary_label:06d}",
                "utterance_id": f"{label_prefix}{primary_label:06d}:{row_index:06d}",
                "split": "pseudo_train",
                "audio_path": f"{public_audio_prefix.rstrip('/')}/{filepaths[row_index]}",
                "channel": "mono",
                "pseudo_weight": round(float(confidence[row_index]), 6),
                "soft_speaker_ids": speaker_ids,
                "soft_probs": [round(float(prob), 8) for _, prob in filtered],
            }
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            rows.append(
                {
                    "row_index": row_index,
                    "filepath": filepaths[row_index],
                    "cluster_id": int(labels[row_index]),
                    "cluster_size": int(cluster_sizes[labels[row_index]]),
                    "teacher_agreement": float(teacher_agreement[row_index]),
                    "margin": float(margins[row_index]),
                    "density_score": float(density_score[row_index]),
                    "stability": float(stability[row_index]),
                    "confidence": float(confidence[row_index]),
                    "soft_speaker_ids": ",".join(speaker_ids),
                    "soft_probs": ",".join(f"{prob:.8f}" for _, prob in filtered),
                }
            )
    _write_mixed_manifest(base_train_manifest, output_pseudo_path, output_mixed_path)
    pl.DataFrame(rows).write_csv(output_audit_path)
    selected_cluster_ids = np.unique(labels[selected])
    return selected, {
        "pseudo_row_count": len(rows),
        "pseudo_cluster_count": int(selected_cluster_ids.size),
        "mixed_row_count": _count_lines(output_mixed_path),
        "confidence_mean": float(confidence[selected].mean()) if selected.any() else 0.0,
        "confidence_p10": float(np.quantile(confidence[selected], 0.10)) if selected.any() else 0.0,
        "confidence_p50": float(np.quantile(confidence[selected], 0.50)) if selected.any() else 0.0,
    }


def _soft_cluster_targets(
    *,
    labels: np.ndarray,
    cluster_sizes: np.ndarray,
    fused_indices: np.ndarray,
    fused_scores: np.ndarray,
    soft_top_clusters: int,
    soft_rank_top: int,
    self_soft_weight: float,
    min_pseudo_cluster_size: int,
    max_pseudo_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count = labels.shape[0]
    out_labels = np.full((row_count, soft_top_clusters), -1, dtype=np.int64)
    out_probs = np.zeros((row_count, soft_top_clusters), dtype=np.float32)
    margins = np.zeros(row_count, dtype=np.float32)
    for row in range(row_count):
        scores_by_label: dict[int, float] = {int(labels[row]): float(self_soft_weight)}
        for candidate, score in zip(
            fused_indices[row, :soft_rank_top].tolist(),
            fused_scores[row, :soft_rank_top].tolist(),
            strict=True,
        ):
            label = int(labels[int(candidate)])
            size = int(cluster_sizes[label])
            if min_pseudo_cluster_size <= size <= max_pseudo_cluster_size:
                scores_by_label[label] = scores_by_label.get(label, 0.0) + float(score)
        ranked = sorted(scores_by_label.items(), key=lambda item: (item[1], -item[0]), reverse=True)
        top = ranked[:soft_top_clusters]
        values = np.asarray([score for _, score in top], dtype=np.float32)
        values = values - values.max()
        probs = np.exp(values)
        probs = probs / probs.sum()
        out_labels[row, : len(top)] = np.asarray([label for label, _ in top], dtype=np.int64)
        out_probs[row, : len(top)] = probs.astype(np.float32, copy=False)
        margins[row] = 0.0 if len(ranked) < 2 else float(ranked[0][1] - ranked[1][1])
    return out_labels, out_probs, margins


def _selected_rows(
    *,
    labels: np.ndarray,
    cluster_sizes: np.ndarray,
    confidence: np.ndarray,
    stability: np.ndarray,
    teacher_agreement: np.ndarray,
    min_pseudo_cluster_size: int,
    max_pseudo_cluster_size: int,
    min_teacher_agreement: float,
    min_confidence: float,
    min_stability: float,
) -> np.ndarray:
    row_selected = (
        (cluster_sizes[labels] >= min_pseudo_cluster_size)
        & (cluster_sizes[labels] <= max_pseudo_cluster_size)
        & (teacher_agreement >= min_teacher_agreement)
        & (confidence >= min_confidence)
        & (stability >= min_stability)
    )
    selected_counts = np.bincount(labels[row_selected], minlength=cluster_sizes.shape[0])
    return row_selected & (selected_counts[labels] >= min_pseudo_cluster_size)


def _percentile_score(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    return ranks


def _write_mixed_manifest(base_train_manifest: Path, pseudo_path: Path, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as mixed:
        with base_train_manifest.open(encoding="utf-8") as base:
            for line in base:
                if line.strip():
                    mixed.write(line.rstrip() + "\n")
        with pseudo_path.open(encoding="utf-8") as pseudo:
            for line in pseudo:
                if line.strip():
                    mixed.write(line.rstrip() + "\n")


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _parse_teacher_spec(value: str) -> TeacherSpec:
    parts: dict[str, str] = {}
    for item in value.split(","):
        key, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid --teacher item {item!r}; expected key=value.")
        parts[key.strip()] = raw.strip()
    required = {"name", "indices", "scores"}
    missing = sorted(required - set(parts))
    if missing:
        raise ValueError(f"--teacher is missing keys: {', '.join(missing)}")
    return TeacherSpec(
        name=parts["name"],
        indices_path=parts["indices"],
        scores_path=parts["scores"],
        weight=float(parts.get("weight", "1.0")),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--public-manifest-csv", required=True)
    parser.add_argument("--base-train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--label-prefix", default="pseudo_ms35_soft_")
    parser.add_argument("--dataset-name", default="participants_ms35_multiteacher_soft_pseudo")
    parser.add_argument("--public-audio-prefix", default="datasets/Для участников")
    parser.add_argument("--source-top-k", type=int, default=200)
    parser.add_argument("--top-cache-k", type=int, default=300)
    parser.add_argument("--score-z-weight", type=float, default=0.35)
    parser.add_argument("--raw-score-weight", type=float, default=0.20)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--reciprocal-top", type=int, default=50)
    parser.add_argument("--reciprocal-bonus", type=float, default=0.08)
    parser.add_argument("--agreement-bonus", type=float, default=0.12)
    parser.add_argument("--shared-edge-top", type=int, default=50)
    parser.add_argument("--shared-top", type=int, default=50)
    parser.add_argument("--shared-weight", type=float, default=0.015)
    parser.add_argument("--hubness-penalty", type=float, default=0.035)
    parser.add_argument("--cluster-edge-top", type=int, default=30)
    parser.add_argument("--cluster-reciprocal-top", type=int, default=80)
    parser.add_argument("--cluster-rank-top", type=int, default=300)
    parser.add_argument("--cluster-iterations", type=int, default=8)
    parser.add_argument("--cluster-min-size", type=int, default=5)
    parser.add_argument("--cluster-max-size", type=int, default=140)
    parser.add_argument("--cluster-min-candidates", type=int, default=4)
    parser.add_argument("--cluster-shared-top", type=int, default=50)
    parser.add_argument("--cluster-shared-min-count", type=int, default=3)
    parser.add_argument("--cluster-reciprocal-bonus", type=float, default=0.03)
    parser.add_argument("--cluster-density-penalty", type=float, default=0.02)
    parser.add_argument("--cluster-edge-score-quantile", type=float, default=None)
    parser.add_argument("--cluster-edge-min-score", type=float, default=None)
    parser.add_argument("--cluster-shared-weight", type=float, default=0.04)
    parser.add_argument("--cluster-rank-weight", type=float, default=0.02)
    parser.add_argument("--cluster-self-weight", type=float, default=0.0)
    parser.add_argument("--cluster-label-size-penalty", type=float, default=0.15)
    parser.add_argument("--no-split-oversized", action="store_true")
    parser.add_argument("--split-edge-top", type=int, default=10)
    parser.add_argument("--soft-top-clusters", type=int, default=3)
    parser.add_argument("--soft-rank-top", type=int, default=80)
    parser.add_argument("--self-soft-weight", type=float, default=0.50)
    parser.add_argument("--min-pseudo-cluster-size", type=int, default=8)
    parser.add_argument("--max-pseudo-cluster-size", type=int, default=90)
    parser.add_argument("--min-teacher-agreement", type=float, default=0.40)
    parser.add_argument("--min-confidence", type=float, default=0.48)
    parser.add_argument("--min-stability", type=float, default=0.35)
    parser.add_argument("--stability-drop-teacher", action="append", default=[])
    parser.add_argument("--stability-top", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    main()
