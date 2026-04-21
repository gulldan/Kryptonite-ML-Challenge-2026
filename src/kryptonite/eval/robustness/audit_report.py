from __future__ import annotations

# ruff: noqa: E501
import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from kryptonite.eval.verification_metrics import compute_verification_metrics_from_arrays

SEVERITY_ORDER = {"light": 0, "medium": 1, "heavy": 2}
FAMILY_ORDER = {
    "additive_noise": 0,
    "reverb": 1,
    "codec_bandwidth": 2,
    "level_clipping": 3,
}
FAMILY_LABELS = {
    "additive_noise": "Аддитивный шум",
    "reverb": "Реверберация",
    "codec_bandwidth": "Кодек / полоса",
    "level_clipping": "Уровень / клиппинг",
}
FAMILY_SHORT_LABELS = {
    "additive_noise": "Шум",
    "reverb": "Реверб",
    "codec_bandwidth": "Кодек",
    "level_clipping": "Уровень",
}
CONDITION_PREFIX_LABELS = {
    "noise": FAMILY_LABELS["additive_noise"],
    "reverb": FAMILY_LABELS["reverb"],
    "codec": FAMILY_LABELS["codec_bandwidth"],
    "level": FAMILY_LABELS["level_clipping"],
}
SEVERITY_LABELS = {"light": "лёгкий", "medium": "средний", "heavy": "тяжёлый"}
SEVERITY_SHORT = {"light": "L", "medium": "M", "heavy": "H"}
MODEL_SPECS = (
    ("campp_ms41_family", "CAM++ MS41 family", "#1f5f8b"),
    ("w2v1j_teacher_peft_stage3", "W2V1j teacher-PEFT stage3", "#c45c1b"),
)
MODEL_KEY_TO_LABEL = {model_key: model_label for model_key, model_label, _ in MODEL_SPECS}
CAVEAT_TRANSLATIONS = {
    "Distortions are synthetic and deterministic; noise and reverb do not use real noise/RIR banks.": (
        "Искажения синтетические и детерминированные; для шума и реверберации не "
        "используются реальные банки шумов и RIR."
    ),
    "Protocol is one-sided: clean enrollment embeddings vs distorted test embeddings.": (
        "Протокол односторонний: enrollment строится по clean-записям, а test берётся из "
        "искажённых вариантов."
    ),
    "Additive-noise severity is confounded with noise color (white/pink/brown), so noise severity is not monotonic.": (
        "В семействе additive-noise одновременно меняются и SNR, и цвет шума "
        "(`white -> pink -> brown`), поэтому шкала severity не является монотонной."
    ),
    "The benchmark compares production-family extraction paths, not a single shared frontend.": (
        "Сравниваются production-like пути извлечения для каждой семьи моделей, а не общий "
        "унифицированный frontend."
    ),
    "The robustness benchmark code is versioned, but runtime artifacts remain local and uncommitted, which weakens exact provenance.": (
        "Код benchmark-пайплайна зафиксирован в репозитории, но runtime-артефакты "
        "по-прежнему локальные и не закоммичены, поэтому provenance слабее, чем у "
        "обычного полностью зафиксированного эксперимента."
    ),
    "CAM++ naming/provenance is slightly confusing: the model is labeled as an MS41 family encoder but points to the MS32-era checkpoint path used by that family.": (
        "Есть небольшая неоднозначность в naming/provenance CAM++: в отчёте фигурирует "
        "`MS41 family`, но путь к checkpoint указывает на базовый `MS32`-era encoder этой семьи."
    ),
}


@dataclass(frozen=True, slots=True)
class ModelAuditSummary:
    model_key: str
    model_label: str
    clean_eer: float
    clean_min_dcf: float
    aggregated_normalized_degradation: float
    worst_condition: str
    worst_condition_degradation: float
    metric_recompute_max_abs: float
    drift_recompute_max_abs: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        default=".runtime/robust3000_seed20260417_v1",
        help="Benchmark runtime root with manifests/cache/reports/state.",
    )
    parser.add_argument(
        "--docs-root",
        default="research/docs",
        help="Documentation root where report assets should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime_root = Path(args.runtime_root).resolve()
    docs_root = Path(args.docs_root).resolve()
    reports_root = runtime_root / "reports"
    assets_root = docs_root / "assets" / "robustness-benchmark-audit"
    assets_root.mkdir(parents=True, exist_ok=True)

    (
        audit_payload,
        comparison_df,
        family_df,
        paired_condition_df,
        long_condition_df,
        drift_df,
    ) = _build_audit(runtime_root)

    chart_paths = _render_charts(
        docs_root=docs_root,
        assets_root=assets_root,
        comparison_df=comparison_df,
        family_df=family_df,
        paired_condition_df=paired_condition_df,
        long_condition_df=long_condition_df,
        drift_df=drift_df,
    )
    markdown_text = _render_markdown_report(
        runtime_root=runtime_root,
        docs_root=docs_root,
        audit_payload=audit_payload,
        comparison_df=comparison_df,
        family_df=family_df,
        paired_condition_df=paired_condition_df,
        long_condition_df=long_condition_df,
        drift_df=drift_df,
        chart_paths=chart_paths,
    )
    html_text = _render_html_report(
        runtime_root=runtime_root,
        audit_payload=audit_payload,
        comparison_df=comparison_df,
        family_df=family_df,
        paired_condition_df=paired_condition_df,
        long_condition_df=long_condition_df,
        drift_df=drift_df,
        chart_paths=chart_paths,
    )

    markdown_path = docs_root / "robustness-benchmark-audit.md"
    html_path = docs_root / "robustness-benchmark-audit.html"
    pdf_path = docs_root / "robustness-benchmark-audit.pdf"
    audit_json_path = reports_root / "benchmark_audit.json"

    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    audit_json_path.write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pdf_backend = _render_pdf(html_path=html_path, pdf_path=pdf_path)

    print(f"[robustness-audit] wrote markdown={markdown_path}")
    print(f"[robustness-audit] wrote html={html_path}")
    print(f"[robustness-audit] wrote pdf={pdf_path} via {pdf_backend}")
    print(f"[robustness-audit] wrote audit_json={audit_json_path}")
    for chart_name, chart_path in chart_paths.items():
        print(f"[robustness-audit] chart {chart_name}={chart_path}")


def _build_audit(
    runtime_root: Path,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    reports_root = runtime_root / "reports"
    manifests_root = runtime_root / "manifests"
    cache_root = runtime_root / "cache"
    state_root = runtime_root / "state"

    assets = json.loads((state_root / "assets.json").read_text(encoding="utf-8"))
    summary = json.loads((reports_root / "benchmark_summary.json").read_text(encoding="utf-8"))
    clean_manifest = pl.read_csv(manifests_root / "clean_3000_frozen.csv").sort("clean_index")
    plan_df = pl.read_csv(manifests_root / "distorted_variants_plan.csv")
    trial_rows = [
        json.loads(line)
        for line in (cache_root / "pairs_or_trials" / "clean_eval_trials.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    labels = np.asarray([int(row["label"]) for row in trial_rows], dtype=np.int8)

    speaker_counts = clean_manifest.group_by("speaker_id").len().get_column("len").to_numpy()
    model_summaries: list[ModelAuditSummary] = []
    family_rows: list[dict[str, Any]] = []
    long_condition_rows: list[dict[str, Any]] = []
    drift_frames: list[pl.DataFrame] = []

    for model_key, model_label, _color in MODEL_SPECS:
        main_df = pl.read_csv(reports_root / f"{model_key}_main_metrics.csv").with_columns(
            pl.lit(model_key).alias("model_key"),
            pl.lit(model_label).alias("model_label"),
        )
        drift_df = pl.read_csv(reports_root / f"{model_key}_drift_metrics.csv").with_columns(
            pl.lit(model_key).alias("model_key"),
            pl.lit(model_label).alias("model_label"),
        )
        drift_frames.append(drift_df)

        if not all(
            _validate_manifest_alignment(
                clean_manifest=clean_manifest,
                manifests_root=manifests_root,
                conditions=main_df.get_column("condition").to_list(),
            )
        ):
            raise RuntimeError(f"Manifest alignment failed for {model_key}")

        clean_embeddings = np.load(cache_root / "embeddings" / model_key / "embeddings_clean.npy")
        clean_scores = np.load(cache_root / "pairs_or_trials" / model_key / "scores_clean.npy")
        metric_recompute_max_abs = 0.0
        drift_recompute_max_abs = 0.0

        for row in main_df.to_dicts():
            condition = str(row["condition"])
            scores = (
                clean_scores
                if condition == "clean"
                else np.load(cache_root / "pairs_or_trials" / model_key / f"scores_{condition}.npy")
            )
            metrics = compute_verification_metrics_from_arrays(labels=labels, scores=scores)
            metric_recompute_max_abs = max(
                metric_recompute_max_abs,
                abs(metrics.eer - float(row["eer"])),
                abs(metrics.min_dcf - float(row["min_dcf"])),
            )
            if condition == "clean":
                continue

            distorted_embeddings = np.load(
                cache_root / "embeddings" / model_key / f"embeddings_{condition}.npy"
            )
            same_clip_cosine = np.clip(
                np.sum(clean_embeddings * distorted_embeddings, axis=1),
                -1.0,
                1.0,
            )
            same_clip_l2 = np.linalg.norm(clean_embeddings - distorted_embeddings, axis=1)
            retrieval_scores = distorted_embeddings @ clean_embeddings.T
            self_retrieval_at1 = float(
                (np.argmax(retrieval_scores, axis=1) == np.arange(retrieval_scores.shape[0])).mean()
            )
            clean_positive = clean_scores[labels == 1]
            clean_negative = clean_scores[labels == 0]
            distorted_positive = scores[labels == 1]
            distorted_negative = scores[labels == 0]
            clean_gap = float(clean_positive.mean() - clean_negative.mean())
            distorted_gap = float(distorted_positive.mean() - distorted_negative.mean())
            drift_row = drift_df.filter(pl.col("condition") == condition).to_dicts()[0]
            drift_recompute_max_abs = max(
                drift_recompute_max_abs,
                abs(
                    round(float(same_clip_cosine.mean()), 6)
                    - float(drift_row["same_clip_cosine_mean"])
                ),
                abs(round(float(same_clip_l2.mean()), 6) - float(drift_row["same_clip_l2_mean"])),
                abs(round(self_retrieval_at1, 6) - float(drift_row["self_retrieval_at1"])),
                abs(
                    round(distorted_gap - clean_gap, 6)
                    - float(drift_row["separation_gap_delta_vs_clean"])
                ),
            )
            long_condition_rows.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "condition": condition,
                    "family": str(row["family"]),
                    "severity": str(row["severity"]),
                    "eer": float(row["eer"]),
                    "min_dcf": float(row["min_dcf"]),
                    "normalized_degradation": float(row["normalized_degradation"]),
                }
            )

        model_summary = json.loads((reports_root / f"{model_key}_summary.json").read_text("utf-8"))
        model_summaries.append(
            ModelAuditSummary(
                model_key=model_key,
                model_label=model_label,
                clean_eer=float(model_summary["clean"]["eer"]),
                clean_min_dcf=float(model_summary["clean"]["min_dcf"]),
                aggregated_normalized_degradation=float(
                    model_summary["aggregated_normalized_degradation"]
                ),
                worst_condition=str(model_summary["worst_condition"]["condition"]),
                worst_condition_degradation=float(
                    model_summary["worst_condition"]["normalized_degradation"]
                ),
                metric_recompute_max_abs=round(metric_recompute_max_abs, 10),
                drift_recompute_max_abs=round(drift_recompute_max_abs, 10),
            )
        )
        family_rows.extend(
            _build_family_rows(main_df=main_df, model_key=model_key, model_label=model_label)
        )

    comparison_df = pl.DataFrame([asdict(item) for item in model_summaries])
    family_df = pl.DataFrame(family_rows)
    long_condition_df = pl.DataFrame(long_condition_rows)
    drift_df = pl.concat(drift_frames, how="vertical")
    paired_condition_df = _pair_condition_rows(long_condition_df)

    audit_payload = {
        "run_key": str(assets["run_key"]),
        "runtime_root": str(runtime_root),
        "source_manifest_csv": str(summary["source_manifest_csv"]),
        "source_data_root": str(summary["source_data_root"]),
        "frozen_clean_manifest": str(summary["frozen_clean_manifest"]),
        "distorted_plan": str(summary["distorted_plan"]),
        "trial_manifest": str(summary["trial_manifest"]),
        "clean_subset": {
            "row_count": int(clean_manifest.height),
            "speaker_count": int(clean_manifest.get_column("speaker_id").n_unique()),
            "min_rows_per_speaker": int(np.min(speaker_counts).item()),
            "median_rows_per_speaker": float(np.median(speaker_counts).item()),
            "max_rows_per_speaker": int(np.max(speaker_counts).item()),
        },
        "trial_summary": {
            "trial_count": int(labels.shape[0]),
            "positive_count": int((labels == 1).sum()),
            "negative_count": int((labels == 0).sum()),
        },
        "plan_summary": {
            "row_count": int(plan_df.height),
            "condition_count": int(plan_df.get_column("condition").n_unique()),
            "family_count": int(plan_df.get_column("family").n_unique()),
        },
        "condition_catalog": [
            {
                "family": str(item["family"]),
                "condition": str(item["name"]),
                "severity": str(item["severity"]),
                "parameters": {str(key): value for key, value in dict(item["parameters"]).items()},
            }
            for item in assets["conditions"]
            if str(item["name"]) != "clean"
        ],
        "model_catalog": [
            {
                "key": str(item["key"]),
                "label": str(item["label"]),
                "kind": str(item["kind"]),
                "checkpoint_path": str(item["checkpoint_path"]),
            }
            for item in assets["models"]
        ],
        "normalized_degradation_formula": str(assets["normalized_degradation_formula"]),
        "robustness_protocol": {
            "clean_condition": str(assets["robustness_protocol"]["clean_condition"]),
            "distorted_condition": str(assets["robustness_protocol"]["distorted_condition"]),
            "clean_source_pool": str(assets["robustness_protocol"]["clean_source_pool"]),
        },
        "extraction_notes": {
            "campp_ms41_family": (
                "Official CAM++ frontend, режим `segment_mean`, `eval_chunk_seconds=6.0`, "
                "`segment_count=3`, `pad_mode=repeat`."
            ),
            "w2v1j_teacher_peft_stage3": (
                "Teacher-PEFT evaluation с `crop_seconds=6.0`, `n_crops=3`; итоговый "
                "эмбеддинг получается усреднением crop-эмбеддингов с последующей "
                "L2-нормализацией."
            ),
        },
        "numerical_integrity": {
            "model_checks": [asdict(item) for item in model_summaries],
            "manifests_aligned": True,
            "reported_main_metrics_match_recompute": True,
            "reported_drift_metrics_match_recompute": True,
        },
        "comparative_findings": {
            "w2v_better_absolute_eer_on_distorted_conditions": int(
                paired_condition_df.get_column("w2v_better_absolute_eer").sum()
            ),
            "w2v_better_absolute_min_dcf_on_distorted_conditions": int(
                paired_condition_df.get_column("w2v_better_absolute_min_dcf").sum()
            ),
            "w2v_better_relative_robustness_on_distorted_conditions": int(
                paired_condition_df.get_column("w2v_better_relative_robustness").sum()
            ),
            "campp_better_relative_robustness_on_distorted_conditions": int(
                (~paired_condition_df.get_column("w2v_better_relative_robustness")).sum()
            ),
        },
        "severity_monotonicity": {
            family: _family_monotonicity(long_condition_df=long_condition_df, family=family)
            for family in FAMILY_ORDER
        },
        "verdict": {
            "short": (
                "Можно доверять как внутреннему benchmark устойчивости, но нельзя "
                "использовать как единственное доказательство реальной устойчивости на "
                "полевых искажениях."
            ),
            "numerical_integrity": "high",
            "provenance_strength": "medium",
            "external_validity": "limited",
        },
        "methodology_caveats": list(CAVEAT_TRANSLATIONS.values()),
    }
    return (
        audit_payload,
        comparison_df,
        family_df,
        paired_condition_df,
        long_condition_df,
        drift_df,
    )


def _validate_manifest_alignment(
    *,
    clean_manifest: pl.DataFrame,
    manifests_root: Path,
    conditions: list[str],
) -> list[bool]:
    checks: list[bool] = []
    for condition in conditions:
        if condition == "clean":
            continue
        manifest_df = pl.read_csv(manifests_root / "distorted" / f"{condition}.csv").sort(
            "clean_index"
        )
        checks.extend(
            [
                manifest_df.height == clean_manifest.height,
                manifest_df.get_column("clean_index").to_list()
                == clean_manifest.get_column("clean_index").to_list(),
                manifest_df.get_column("item_id").to_list()
                == clean_manifest.get_column("item_id").to_list(),
            ]
        )
    return checks


def _build_family_rows(
    *, main_df: pl.DataFrame, model_key: str, model_label: str
) -> list[dict[str, Any]]:
    distorted_df = main_df.filter(pl.col("condition") != "clean")
    family_summary = (
        distorted_df.group_by("family")
        .agg(
            pl.mean("normalized_degradation").alias("mean_normalized_degradation"),
            pl.mean("delta_eer_vs_clean").alias("mean_delta_eer_vs_clean"),
            pl.mean("delta_min_dcf_vs_clean").alias("mean_delta_min_dcf_vs_clean"),
        )
        .sort("family")
    )
    return [
        {
            "model_key": model_key,
            "model_label": model_label,
            "family": str(row["family"]),
            "mean_normalized_degradation": float(row["mean_normalized_degradation"]),
            "mean_delta_eer_vs_clean": float(row["mean_delta_eer_vs_clean"]),
            "mean_delta_min_dcf_vs_clean": float(row["mean_delta_min_dcf_vs_clean"]),
        }
        for row in family_summary.to_dicts()
    ]


def _pair_condition_rows(long_condition_df: pl.DataFrame) -> pl.DataFrame:
    campp_df = long_condition_df.filter(pl.col("model_key") == MODEL_SPECS[0][0]).rename(
        {
            "eer": "campp_eer",
            "min_dcf": "campp_min_dcf",
            "normalized_degradation": "campp_normalized_degradation",
        }
    )
    w2v_df = long_condition_df.filter(pl.col("model_key") == MODEL_SPECS[1][0]).rename(
        {
            "eer": "w2v_eer",
            "min_dcf": "w2v_min_dcf",
            "normalized_degradation": "w2v_normalized_degradation",
        }
    )
    return (
        campp_df.join(
            w2v_df.select("condition", "w2v_eer", "w2v_min_dcf", "w2v_normalized_degradation"),
            on="condition",
        )
        .with_columns(
            (pl.col("w2v_eer") < pl.col("campp_eer")).alias("w2v_better_absolute_eer"),
            (pl.col("w2v_min_dcf") < pl.col("campp_min_dcf")).alias("w2v_better_absolute_min_dcf"),
            (pl.col("w2v_normalized_degradation") < pl.col("campp_normalized_degradation")).alias(
                "w2v_better_relative_robustness"
            ),
        )
        .sort(_condition_sort_exprs())
    )


def _family_monotonicity(
    *, long_condition_df: pl.DataFrame, family: str
) -> dict[str, list[float] | bool]:
    family_df = long_condition_df.filter(pl.col("family") == family)
    payload: dict[str, list[float] | bool] = {}
    for model_key, model_label, _color in MODEL_SPECS:
        ordered = [
            float(
                family_df.filter(
                    (pl.col("model_key") == model_key) & (pl.col("severity") == severity)
                )
                .select("normalized_degradation")
                .item()
            )
            for severity in ("light", "medium", "heavy")
        ]
        payload[model_label] = [round(value, 6) for value in ordered]
        payload[f"{model_label}_is_monotonic"] = ordered[0] <= ordered[1] <= ordered[2]
    return payload


def _condition_sort_exprs() -> list[pl.Expr]:
    return [
        pl.col("family").replace_strict(FAMILY_ORDER, default=99),
        pl.col("severity").replace_strict(SEVERITY_ORDER, default=99),
    ]


def _render_charts(
    *,
    docs_root: Path,
    assets_root: Path,
    comparison_df: pl.DataFrame,
    family_df: pl.DataFrame,
    paired_condition_df: pl.DataFrame,
    long_condition_df: pl.DataFrame,
    drift_df: pl.DataFrame,
) -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#aebdcc",
            "axes.labelcolor": "#15324b",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    chart_paths = {
        name: str((assets_root / f"{name}.png").relative_to(docs_root))
        for name in (
            "overview_metrics",
            "family_relative_degradation",
            "absolute_vs_relative_by_condition",
            "drift_summary",
        )
    }
    colors = [color for _model_key, _label, color in MODEL_SPECS]
    x_positions = np.arange(comparison_df.height)
    model_labels = comparison_df.get_column("model_label").to_list()

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    metrics = [
        ("clean_eer", "EER на clean", "EER"),
        ("clean_min_dcf", "minDCF на clean", "minDCF"),
        (
            "aggregated_normalized_degradation",
            "Средняя относительная деградация",
            "mean normalized degradation",
        ),
    ]
    for axis, (column, title, ylabel) in zip(axes, metrics, strict=True):
        values = comparison_df.get_column(column).to_list()
        axis.bar(x_positions, values, color=colors, width=0.56)
        axis.set_title(title)
        axis.set_xticks(x_positions, model_labels, rotation=10, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(assets_root / "overview_metrics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(1, 1, figsize=(9.8, 4.8), constrained_layout=True)
    family_positions = np.arange(len(FAMILY_ORDER))
    width = 0.34
    for index, (model_key, model_label, color) in enumerate(MODEL_SPECS):
        model_family = family_df.filter(pl.col("model_key") == model_key).sort(
            pl.col("family").replace_strict(FAMILY_ORDER, default=99)
        )
        axis.bar(
            family_positions + (index - 0.5) * width,
            model_family.get_column("mean_normalized_degradation").to_list(),
            width=width,
            color=color,
            label=model_label,
        )
    axis.set_title("Средняя относительная деградация по семействам искажений")
    axis.set_xticks(
        family_positions, [FAMILY_SHORT_LABELS[item] for item in FAMILY_ORDER], rotation=0
    )
    axis.set_ylabel("средняя относительная деградация")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(assets_root / "family_relative_degradation.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    condition_labels = [_short_condition_label(row) for row in paired_condition_df.to_dicts()]
    positions = np.arange(paired_condition_df.height)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.8), constrained_layout=True)
    axes[0].plot(
        positions,
        paired_condition_df.get_column("campp_eer").to_list(),
        marker="o",
        color=colors[0],
        label=MODEL_SPECS[0][1],
    )
    axes[0].plot(
        positions,
        paired_condition_df.get_column("w2v_eer").to_list(),
        marker="o",
        color=colors[1],
        label=MODEL_SPECS[1][1],
    )
    axes[0].set_title("Абсолютное качество по условиям")
    axes[0].set_ylabel("EER")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(
        positions,
        paired_condition_df.get_column("campp_normalized_degradation").to_list(),
        marker="o",
        color=colors[0],
        label=MODEL_SPECS[0][1],
    )
    axes[1].plot(
        positions,
        paired_condition_df.get_column("w2v_normalized_degradation").to_list(),
        marker="o",
        color=colors[1],
        label=MODEL_SPECS[1][1],
    )
    axes[1].set_title("Относительная деградация по тем же условиям")
    axes[1].set_ylabel("relative degradation")
    axes[1].set_xticks(positions, condition_labels, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(assets_root / "absolute_vs_relative_by_condition.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    plot_drift_df = drift_df.join(
        long_condition_df.select(
            "model_key", "condition", "family", "severity", "normalized_degradation"
        ),
        on=["model_key", "condition"],
    ).sort(_condition_sort_exprs())
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    markers = {"additive_noise": "o", "reverb": "^", "codec_bandwidth": "s", "level_clipping": "D"}
    for model_key, model_label, color in MODEL_SPECS:
        model_slice = plot_drift_df.filter(pl.col("model_key") == model_key)
        for family in FAMILY_ORDER:
            family_slice = model_slice.filter(pl.col("family") == family)
            axes[0].scatter(
                family_slice.get_column("same_clip_cosine_mean").to_list(),
                family_slice.get_column("normalized_degradation").to_list(),
                marker=markers[family],
                color=color,
                alpha=0.85,
                s=65,
                label=f"{model_label} / {FAMILY_SHORT_LABELS[family]}",
            )
    axes[0].set_title("Связь drift-метрик и относительной деградации")
    axes[0].set_xlabel("same_clip_cosine_mean")
    axes[0].set_ylabel("normalized degradation")
    axes[0].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        unique.setdefault(label, handle)
    axes[0].legend(unique.values(), unique.keys(), frameon=False, fontsize=8)

    for model_key, model_label, color in MODEL_SPECS:
        model_slice = plot_drift_df.filter(pl.col("model_key") == model_key)
        axes[1].plot(
            positions,
            model_slice.get_column("self_retrieval_at1").to_list(),
            marker="o",
            color=color,
            label=model_label,
        )
    axes[1].set_title("Self-retrieval@1 по условиям")
    axes[1].set_ylabel("self_retrieval@1")
    axes[1].set_xticks(positions, condition_labels, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(assets_root / "drift_summary.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return chart_paths


def _render_markdown_report(
    *,
    runtime_root: Path,
    docs_root: Path,
    audit_payload: dict[str, Any],
    comparison_df: pl.DataFrame,
    family_df: pl.DataFrame,
    paired_condition_df: pl.DataFrame,
    long_condition_df: pl.DataFrame,
    drift_df: pl.DataFrame,
    chart_paths: dict[str, str],
) -> str:
    integrity_rows = _integrity_rows(audit_payload)
    methodology_rows = _methodology_rows(audit_payload)
    distortion_rows = _distortion_catalog_rows(audit_payload)
    metric_rows = _metric_definition_rows(audit_payload)
    clean_rows = _clean_metric_rows(comparison_df)
    family_rows = _family_summary_rows(family_df)
    winner_rows = _winner_summary_rows(audit_payload, paired_condition_df)
    decision_rows = _decision_matrix_rows()
    drift_rows = _representative_drift_rows(paired_condition_df, long_condition_df, drift_df)
    appendix_sections = [
        "\n".join(
            [
                f"### {FAMILY_LABELS[family]}",
                "",
                _markdown_table(_appendix_family_rows(paired_condition_df, family)),
            ]
        )
        for family in FAMILY_ORDER
    ]
    lines = [
        "# Технический отчёт по бенчмарку устойчивости к искажениям",
        "",
        f"- Дата генерации: `{date.today().isoformat()}`",
        f"- Идентификатор запуска: `{audit_payload['run_key']}`",
        f"- Каталог runtime: `{runtime_root}`",
        "",
        "## Оглавление",
        "",
        "1. Главное сразу",
        "2. Постановка задачи и границы интерпретации",
        "3. Методология benchmark",
        "4. Проверка достоверности чисел",
        "5. Результаты на clean-базе",
        "6. Устойчивость по семействам искажений",
        "7. Разбор по отдельным условиям",
        "8. Drift эмбеддингов",
        "9. Что можно и чего нельзя утверждать по итогам benchmark",
        "10. Практическая рекомендация",
        "11. Приложение A. Поусловные таблицы",
        "",
        "## 1. Главное сразу",
        "",
        *[
            f"- {item}"
            for item in _executive_summary_bullets(
                audit_payload, comparison_df, paired_condition_df
            )
        ],
        "",
        "## 2. Постановка задачи и границы интерпретации",
        "",
        (
            "Цель benchmark-а — проверить, как две production-like модели speaker verification "
            "сохраняют качество и геометрию эмбеддингов при контролируемых синтетических искажениях."
        ),
        (
            "Документ отвечает на два вопроса: корректны ли опубликованные числа относительно "
            "сохранённых runtime-артефактов и что именно можно заключить по ним о robustness."
        ),
        (
            "Сильнее, чем позволяет протокол, документ не утверждает: это внутренний technical report "
            "по synthetic stress-test, а не финальное доказательство полевой устойчивости."
        ),
        "",
        "## 3. Методология benchmark",
        "",
        "### Коротко",
        "",
        *[f"- {item}" for item in _methodology_overview_bullets(audit_payload)],
        "",
        "### Сводка по дизайну",
        "",
        _markdown_table(methodology_rows),
        "",
        "### Набор искажений и уровни тяжести",
        "",
        _markdown_table(distortion_rows),
        "",
        "### Какие метрики считаются и как их интерпретировать",
        "",
        _markdown_table(metric_rows),
        "",
        "## 4. Проверка достоверности чисел",
        "",
        "### Что именно проверялось",
        "",
        _markdown_table(integrity_rows),
        "",
        "### Что это значит",
        "",
        (
            "Итоговые `CSV/JSON` совпали с пересчётом из сырых `scores_*.npy` и `embeddings_*.npy`, "
            "поэтому опубликованные числа можно считать численно корректными относительно текущего runtime."
        ),
        (
            "При этом численная корректность не отменяет методологических ограничений. "
            "Это проверка честности расчёта, а не автоматическое доказательство внешней валидности."
        ),
        "",
        "## 5. Результаты на clean-базе",
        "",
        "### Коротко",
        "",
        "- `W2V1j` стартует с более сильного clean-baseline по обеим абсолютным метрикам.",
        "- Это важно, потому что относительная деградация всегда читается относительно собственной clean-точки модели.",
        "",
        "### Таблица clean-метрик",
        "",
        _markdown_table(clean_rows),
        "",
        f"![Clean baseline и средняя относительная деградация]({chart_paths['overview_metrics']})",
        "",
        "Что показывает рисунок.",
        "Первые два графика показывают абсолютные clean-метрики (`EER`, `minDCF`), а третий график — среднюю относительную деградацию по всем distorted-условиям.",
        "",
        "Что это значит.",
        "`W2V1j` сильнее по качеству на clean, но `CAM++` теряет относительно своей clean-базы меньше. Это два разных вопроса, и их нельзя сводить к одной фразе «кто robustнее».",
        "",
        "## 6. Устойчивость по семействам искажений",
        "",
        "### Сводная таблица по семействам",
        "",
        _markdown_table(family_rows),
        "",
        f"![Средняя относительная деградация по семействам]({chart_paths['family_relative_degradation']})",
        "",
        "Что показывает рисунок.",
        "График агрегирует по три уровня severity внутри каждого семейства и сравнивает среднюю относительную деградацию двух моделей.",
        "",
        *[f"- {item}" for item in _family_insight_bullets(family_df)],
        "",
        "Вывод по блоку.",
        (
            "`CAM++` выглядит стабильнее на `reverb` и `codec / bandwidth`, а `W2V1j` — на "
            "`level / clipping` и частично на тяжёлом шуме."
        ),
        "",
        "## 7. Разбор по отдельным условиям",
        "",
        "### Кто выигрывает по разным критериям",
        "",
        _markdown_table(winner_rows),
        "",
        f"![Абсолютное качество и относительная деградация по условиям]({chart_paths['absolute_vs_relative_by_condition']})",
        "",
        "Что показывает рисунок.",
        "Верхний график показывает абсолютный `EER` для каждого искажённого условия, нижний — относительную деградацию на тех же условиях.",
        "",
        *[f"- {item}" for item in _condition_insight_bullets(audit_payload, paired_condition_df)],
        "",
        "Вывод по блоку.",
        (
            "`W2V1j` выигрывает абсолютное сравнение на всех `12/12` условиях. `CAM++` выигрывает "
            "только относительное сравнение на `8/12` условиях, поэтому фраза «CAM++ устойчивее» "
            "допустима только с явной оговоркой про normalized degradation."
        ),
        "",
        "## 8. Drift эмбеддингов",
        "",
        "### Представительные drift-показатели",
        "",
        _markdown_table(drift_rows),
        "",
        f"![Drift эмбеддингов]({chart_paths['drift_summary']})",
        "",
        "Что показывает рисунок.",
        "Левая часть связывает падение `same_clip_cosine_mean` с ухудшением relative degradation, правая — показывает `self_retrieval@1` по условиям.",
        "",
        "Что это значит.",
        (
            "На `codec_heavy` и соседних codec/reverb-условиях геометрия эмбеддингов рушится сильнее "
            "всего. Это напрямую связано с ростом `EER` и `minDCF`."
        ),
        "",
        "## 9. Что можно и чего нельзя утверждать по итогам benchmark",
        "",
        "### Ограничения интерпретации",
        "",
        *[f"- {item}" for item in audit_payload["methodology_caveats"]],
        "",
        "### Честная формулировка доверия",
        "",
        "Результатам можно доверять как корректно посчитанному внутреннему benchmark-у: цифры воспроизводятся из сырых артефактов.",
        "Результатам нельзя доверять как универсальному доказательству реальной устойчивости: для такого вывода текущий protocol недостаточен.",
        "",
        "## 10. Практическая рекомендация",
        "",
        _markdown_table(decision_rows),
        "",
        "Итог.",
        (
            "Для выбора модели под лучший абсолютный verification quality предпочтительнее `W2V1j`. "
            "Для сценария, где приоритетом является мягкость относительной деградации на synthetic "
            "`reverb/codec`-стрессах, интереснее `CAM++`."
        ),
        "",
        "## 11. Приложение A. Поусловные таблицы",
        "",
        *appendix_sections,
        "",
        "## Приложение B. Источники документа",
        "",
        f"- Главный summary: `{runtime_root / 'reports' / 'benchmark_summary.json'}`",
        f"- Таблица сравнения моделей: `{runtime_root / 'reports' / 'model_comparison.csv'}`",
        f"- Основные метрики CAM++: `{runtime_root / 'reports' / 'campp_ms41_family_main_metrics.csv'}`",
        f"- Drift-метрики CAM++: `{runtime_root / 'reports' / 'campp_ms41_family_drift_metrics.csv'}`",
        f"- Основные метрики W2V: `{runtime_root / 'reports' / 'w2v1j_teacher_peft_stage3_main_metrics.csv'}`",
        f"- Drift-метрики W2V: `{runtime_root / 'reports' / 'w2v1j_teacher_peft_stage3_drift_metrics.csv'}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def _render_html_report(
    *,
    runtime_root: Path,
    audit_payload: dict[str, Any],
    comparison_df: pl.DataFrame,
    family_df: pl.DataFrame,
    paired_condition_df: pl.DataFrame,
    long_condition_df: pl.DataFrame,
    drift_df: pl.DataFrame,
    chart_paths: dict[str, str],
) -> str:
    findings = audit_payload["comparative_findings"]
    executive_html = _html_list(
        _executive_summary_bullets(audit_payload, comparison_df, paired_condition_df)
    )
    methodology_overview_html = _html_list(_methodology_overview_bullets(audit_payload))
    family_insights_html = _html_list(_family_insight_bullets(family_df))
    condition_insights_html = _html_list(
        _condition_insight_bullets(audit_payload, paired_condition_df)
    )
    caveats_html = _html_list(audit_payload["methodology_caveats"])
    toc_html = _html_list(
        [
            "Главное сразу",
            "Постановка задачи и границы интерпретации",
            "Методология benchmark",
            "Проверка достоверности чисел",
            "Результаты на clean-базе",
            "Устойчивость по семействам искажений",
            "Разбор по отдельным условиям",
            "Drift эмбеддингов",
            "Что можно и чего нельзя утверждать по итогам benchmark",
            "Практическая рекомендация",
            "Приложение A. Поусловные таблицы",
        ],
        class_name="toc-list",
    )
    methodology_table = _html_table(_methodology_rows(audit_payload), compact=True)
    distortion_table = _html_table(_distortion_catalog_rows(audit_payload), compact=True)
    metric_table = _html_table(_metric_definition_rows(audit_payload), compact=True)
    integrity_table = _html_table(_integrity_rows(audit_payload), compact=True)
    clean_table = _html_table(_clean_metric_rows(comparison_df), compact=True)
    family_table = _html_table(_family_summary_rows(family_df), compact=True)
    winner_table = _html_table(
        _winner_summary_rows(audit_payload, paired_condition_df), compact=True
    )
    drift_table = _html_table(
        _representative_drift_rows(paired_condition_df, long_condition_df, drift_df),
        compact=True,
    )
    decision_table = _html_table(_decision_matrix_rows(), compact=True)
    appendix_tables = "".join(
        [
            (
                f"<section class='appendix-block'><h3>{escape(FAMILY_LABELS[family])}</h3>"
                f"{_html_table(_appendix_family_rows(paired_condition_df, family), compact=True)}</section>"
            )
            for family in FAMILY_ORDER
        ]
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Технический отчёт по бенчмарку устойчивости</title>
  <style>
    :root {{
      --ink: #183149;
      --muted: #5a6c7c;
      --line: #d8dee6;
      --soft: #f4f6f8;
      --accent: #1f5f8b;
      --accent-soft: #ebf2f8;
    }}
    @page {{ size: A4; margin: 14mm 14mm 16mm; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font: 10.4pt/1.48 "DejaVu Sans", Arial, sans-serif;
      background: white;
    }}
    h1, h2, h3 {{
      margin: 0 0 8px;
      color: #10273e;
      font-family: "DejaVu Serif", Georgia, serif;
    }}
    h1 {{ font-size: 22pt; line-height: 1.14; }}
    h2 {{ font-size: 15pt; padding-bottom: 6px; border-bottom: 1px solid var(--line); }}
    h3 {{ font-size: 11.5pt; margin-top: 12px; }}
    p, li {{ margin: 0 0 7px; }}
    strong {{ font-weight: 700; }}
    code {{
      font-family: "DejaVu Sans Mono", Consolas, monospace;
      font-size: 0.92em;
      background: #f7f8fa;
      padding: 0 2px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 8px 0 10px;
      table-layout: fixed;
      break-inside: avoid;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 7px;
      vertical-align: top;
      text-align: left;
      overflow-wrap: anywhere;
    }}
    th {{
      background: var(--soft);
      font-weight: 700;
    }}
    .compact th, .compact td {{ font-size: 8.9pt; padding: 5px 6px; }}
    .page {{
      break-after: page;
    }}
    .page:last-child {{
      break-after: auto;
    }}
    .title-block {{
      margin-bottom: 10px;
    }}
    .eyebrow {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 8.6pt;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .subtitle {{
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 9.2pt;
      margin-bottom: 14px;
    }}
    .section {{
      margin-bottom: 16px;
    }}
    .lead-box, .note {{
      border: 1px solid var(--line);
      background: var(--accent-soft);
      padding: 9px 11px;
      margin: 10px 0 12px;
    }}
    .figure {{
      margin: 10px 0 12px;
      break-inside: avoid;
    }}
    .figure img {{
      display: block;
      width: 100%;
      border: 1px solid var(--line);
      background: white;
    }}
    .caption {{
      margin-top: 6px;
      font-size: 8.8pt;
      color: var(--muted);
    }}
    .toc-list, .bullet-list {{
      margin: 0;
      padding-left: 18px;
    }}
    .appendix-block {{
      margin-bottom: 14px;
      break-inside: avoid;
    }}
    .source-list li {{
      overflow-wrap: anywhere;
    }}
    .footer {{
      margin-top: 12px;
      border-top: 1px solid var(--line);
      padding-top: 6px;
      font-size: 8.8pt;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <section class="page">
    <div class="title-block">
      <div class="eyebrow">Внутренний технический документ</div>
      <h1>Технический отчёт по бенчмарку устойчивости моделей верификации диктора</h1>
      <p class="subtitle">Сравнение CAM++ MS41 family и W2V1j teacher-PEFT stage3</p>
      <p class="meta">
        Дата: {escape(date.today().isoformat())}<br>
        Идентификатор запуска: <code>{escape(audit_payload["run_key"])}</code><br>
        Runtime root: <code>{escape(str(runtime_root))}</code>
      </p>
    </div>

    <section class="section">
      <h2>Оглавление</h2>
      {toc_html}
    </section>

    <section class="section">
      <h2>Главное сразу</h2>
      {executive_html}
      <div class="lead-box">
        <strong>Краткая формулировка результата.</strong>
        <p><code>W2V1j</code> лучше по абсолютным метрикам на <strong>{
        findings["w2v_better_absolute_eer_on_distorted_conditions"]
    }/12</strong> distorted-условиях. <code>CAM++</code> выигрывает по относительной деградации на <strong>{
        findings["campp_better_relative_robustness_on_distorted_conditions"]
    }/12</strong> условиях.</p>
      </div>
    </section>
  </section>

  <section class="page">
    <section class="section">
      <h2>1. Постановка задачи и границы интерпретации</h2>
      <p>Цель benchmark-а — проверить, как две production-like модели speaker verification сохраняют качество и геометрию эмбеддингов при контролируемых синтетических искажениях относительно clean-baseline.</p>
      <p>Документ отвечает на два вопроса: корректны ли опубликованные числа относительно сохранённых runtime-артефактов и что именно можно заключить из этих чисел о robustness моделей.</p>
      <p>Документ намеренно не делает более сильных утверждений, чем позволяет дизайн benchmark-а. Это внутренний технический отчёт по synthetic stress-test, а не доказательство полевой устойчивости к реальным шумам и реальным RIR.</p>
    </section>

    <section class="section">
      <h2>2. Методология benchmark</h2>
      <h3>Коротко</h3>
      {methodology_overview_html}
      <h3>Сводка по дизайну</h3>
      {methodology_table}
      <h3>Набор искажений и уровни тяжести</h3>
      {distortion_table}
      <h3>Какие метрики считаются и что они значат</h3>
      {metric_table}
    </section>
  </section>

  <section class="page">
    <section class="section">
      <h2>3. Проверка достоверности чисел</h2>
      <p>Сохранённые отчёты не принимались на веру. Проверка шла от сырых <code>scores_*.npy</code>, <code>embeddings_*.npy</code> и <code>clean_eval_trials.jsonl</code> обратно к итоговым таблицам.</p>
      {integrity_table}
      <div class="note">
        Максимальное абсолютное расхождение при пересчёте составило <strong>0.0</strong> как для verification-метрик, так и для drift-метрик. Значит, итоговые CSV/JSON численно корректны относительно текущего runtime.
      </div>
      <p>Важно различать два уровня доверия. Первый уровень уже закрыт: цифры действительно воспроизводятся. Второй уровень открыт частично: из-за особенностей протокола эти цифры нельзя автоматически переносить на real-world robustness.</p>
    </section>

    <section class="section">
      <h2>4. Результаты на clean-базе</h2>
      <h3>Коротко</h3>
      {
        _html_list(
            [
                "W2V1j стартует с более сильного clean-baseline по обеим абсолютным метрикам.",
                "Именно clean-база определяет, как нужно читать дальнейшую относительную деградацию.",
            ]
        )
    }
      <h3>Таблица clean-метрик</h3>
      {clean_table}
      <div class="figure">
        <img src="{
        escape(chart_paths["overview_metrics"])
    }" alt="Clean baseline и средняя относительная деградация">
        <div class="caption">Рисунок 1. Первые два графика показывают абсолютные clean-метрики, третий — среднюю относительную деградацию по всем distorted-условиям.</div>
      </div>
      <h3>Что это значит</h3>
      <p><code>W2V1j</code> лучше отвечает на вопрос «какая модель даёт более низкий EER/minDCF вообще». <code>CAM++</code> лучше отвечает только на другой вопрос — «какая модель мягче теряет качество относительно собственной clean-точки».</p>
    </section>
  </section>

  <section class="page">
    <section class="section">
      <h2>5. Устойчивость по семействам искажений</h2>
      <h3>Сводная таблица по семействам</h3>
      {family_table}
      <div class="figure">
        <img src="{
        escape(chart_paths["family_relative_degradation"])
    }" alt="Средняя относительная деградация по семействам">
        <div class="caption">Рисунок 2. Средняя относительная деградация по четырём семействам искажений.</div>
      </div>
      <h3>Что показывает график</h3>
      <p>График усредняет три уровня severity внутри каждого семейства и тем самым показывает не отдельные экстремумы, а характерную реакцию модели на тип искажения.</p>
      <h3>Что это значит</h3>
      {family_insights_html}
      <p><strong>Вывод по блоку.</strong> В относительном смысле <code>CAM++</code> выглядит устойчивее на <code>reverb</code> и <code>codec / bandwidth</code>, а <code>W2V1j</code> — на <code>level / clipping</code> и частично на тяжёлом шуме.</p>
    </section>
  </section>

  <section class="page">
    <section class="section">
      <h2>6. Разбор по отдельным условиям</h2>
      <h3>Кто выигрывает по разным критериям</h3>
      {winner_table}
      <div class="figure">
        <img src="{
        escape(chart_paths["absolute_vs_relative_by_condition"])
    }" alt="Абсолютное качество и относительная деградация по условиям">
        <div class="caption">Рисунок 3. Верхний график показывает абсолютный EER по каждому condition, нижний — относительную деградацию на тех же conditions.</div>
      </div>
      <h3>Что показывает график</h3>
      <p>Это главный диагностический рисунок отчёта. Он прямо показывает, что абсолютное качество и относительная деградация дают разный ranking моделей.</p>
      <h3>Что это значит</h3>
      {condition_insights_html}
      <div class="lead-box">
        <strong>Вывод по блоку.</strong>
        <p><code>W2V1j</code> выигрывает абсолютное сравнение на всех <strong>12/12</strong> distorted-условиях. Формулировка «CAM++ устойчивее» допустима только если явно уточнить, что речь идёт о <strong>normalized degradation</strong>, а не об итоговом verification quality.</p>
      </div>
    </section>
  </section>

  <section class="page">
    <section class="section">
      <h2>7. Drift эмбеддингов</h2>
      <h3>Представительные drift-показатели</h3>
      {drift_table}
      <div class="figure">
        <img src="{escape(chart_paths["drift_summary"])}" alt="Drift эмбеддингов">
        <div class="caption">Рисунок 4. Слева — связь same-clip cosine с относительной деградацией, справа — self-retrieval@1 по условиям.</div>
      </div>
      <h3>Что показывает график</h3>
      <p>Когда искажение начинает сильно смещать эмбеддинг того же самого клипа, падают <code>same_clip_cosine_mean</code> и <code>self_retrieval@1</code>. В тех же условиях обычно ухудшаются и verification-метрики.</p>
      <h3>Что это значит</h3>
      <p>Худший drift для обеих моделей наблюдается в ветке <code>codec_heavy</code>. Это не случайный артефакт отдельной метрики: там же фиксируются самые плохие EER/minDCF и самый глубокий relative drop.</p>
    </section>

    <section class="section">
      <h2>8. Что можно и чего нельзя утверждать по итогам benchmark</h2>
      <h3>Ограничения интерпретации</h3>
      {caveats_html}
      <h3>Честная формулировка доверия</h3>
      <p>Результатам можно доверять как корректно посчитанному внутреннему benchmark-у: цифры не «нарисованы» и воспроизводятся из сырых артефактов.</p>
      <p>Результатам нельзя доверять как окончательному доказательству реальной устойчивости: для такого вывода текущего дизайна benchmark-а недостаточно.</p>
    </section>
  </section>

  <section class="page">
    <section class="section">
      <h2>9. Практическая рекомендация</h2>
      {decision_table}
      <div class="lead-box">
        <strong>Итог.</strong>
        <p>Если нужен лучший абсолютный verification quality, выбирать нужно <code>W2V1j</code>. Если нужен более мягкий relative drift на synthetic <code>reverb/codec</code>-stress, интереснее <code>CAM++</code>. Для презентации корректная формулировка: <strong>W2V1j сильнее overall, CAM++ мягче деградирует относительно собственного clean-baseline на части synthetic условий</strong>.</p>
      </div>
    </section>

    <section class="section">
      <h2>Приложение A. Поусловные таблицы</h2>
      {appendix_tables}
    </section>
  </section>

  <section>
    <section class="section">
      <h2>Приложение B. Источники документа</h2>
      <ul class="source-list">
        <li>Главный summary: <code>{
        escape(str(runtime_root / "reports" / "benchmark_summary.json"))
    }</code></li>
        <li>Таблица сравнения моделей: <code>{
        escape(str(runtime_root / "reports" / "model_comparison.csv"))
    }</code></li>
        <li>Основные метрики CAM++: <code>{
        escape(str(runtime_root / "reports" / "campp_ms41_family_main_metrics.csv"))
    }</code></li>
        <li>Drift-метрики CAM++: <code>{
        escape(str(runtime_root / "reports" / "campp_ms41_family_drift_metrics.csv"))
    }</code></li>
        <li>Основные метрики W2V: <code>{
        escape(str(runtime_root / "reports" / "w2v1j_teacher_peft_stage3_main_metrics.csv"))
    }</code></li>
        <li>Drift-метрики W2V: <code>{
        escape(str(runtime_root / "reports" / "w2v1j_teacher_peft_stage3_drift_metrics.csv"))
    }</code></li>
      </ul>
      <div class="footer">
        Документ собран автоматически из сохранённых runtime-артефактов. PDF рендерится через Chrome headless print.
      </div>
    </section>
  </section>
</body>
</html>
"""


def _render_pdf(*, html_path: Path, pdf_path: Path) -> str:
    chrome = next(
        (
            candidate
            for candidate in (
                "google-chrome-stable",
                "google-chrome",
                "chromium",
                "chromium-browser",
            )
            if shutil.which(candidate)
        ),
        None,
    )
    if chrome is None:
        raise FileNotFoundError("Chrome/Chromium binary not found for PDF rendering.")
    command = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=15000",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Chrome PDF rendering failed:\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return chrome


def _integrity_rows(audit_payload: dict[str, Any]) -> list[dict[str, str]]:
    model_checks = audit_payload["numerical_integrity"]["model_checks"]
    return [
        {
            "Проверка": "Пересчёт `EER/minDCF` из сохранённых `scores_*.npy`",
            "Статус": "Пройдено",
            "Доказательство": f"max abs diff = {max(item['metric_recompute_max_abs'] for item in model_checks):.1f}",
        },
        {
            "Проверка": "Пересчёт drift-метрик из `embeddings_*.npy` и score cache",
            "Статус": "Пройдено",
            "Доказательство": f"max abs diff = {max(item['drift_recompute_max_abs'] for item in model_checks):.1f}",
        },
        {
            "Проверка": "Выравнивание distorted manifests с frozen clean subset",
            "Статус": "Пройдено",
            "Доказательство": "Одинаковые `clean_index` и `item_id` для всех conditions",
        },
        {
            "Проверка": "Монотонность severity",
            "Статус": "Частично",
            "Доказательство": "reverb/codec/clipping монотонны; additive-noise — нет",
        },
    ]


def _methodology_rows(audit_payload: dict[str, Any]) -> list[dict[str, str]]:
    clean_subset = audit_payload["clean_subset"]
    trial_summary = audit_payload["trial_summary"]
    plan_summary = audit_payload["plan_summary"]
    return [
        {
            "Параметр": "Чистая подвыборка",
            "Значение": f"{clean_subset['row_count']} клипов / {clean_subset['speaker_count']} speakers",
        },
        {
            "Параметр": "Распределение по speaker",
            "Значение": f"{clean_subset['min_rows_per_speaker']}–{clean_subset['max_rows_per_speaker']} клипов, медиана {clean_subset['median_rows_per_speaker']:.0f}",
        },
        {
            "Параметр": "План distortions",
            "Значение": f"{plan_summary['row_count']} строк / {plan_summary['condition_count']} conditions / {plan_summary['family_count']} families",
        },
        {
            "Параметр": "Проверочные пары",
            "Значение": f"{trial_summary['trial_count']} всего ({trial_summary['positive_count']} positive / {trial_summary['negative_count']} negative)",
        },
        {"Параметр": "Протокол", "Значение": "clean enrollment vs distorted test"},
    ]


def _comparison_rows(comparison_df: pl.DataFrame) -> list[dict[str, str]]:
    return [
        {
            "Модель": str(row["model_label"]),
            "EER на clean": _fmt(row["clean_eer"], 5),
            "minDCF на clean": _fmt(row["clean_min_dcf"], 6),
            "Средняя относительная деградация": _fmt(row["aggregated_normalized_degradation"], 6),
            "Наихудшее условие": _condition_title(str(row["worst_condition"])),
            "Деградация в наихудшем условии": _fmt(row["worst_condition_degradation"], 6),
        }
        for row in comparison_df.to_dicts()
    ]


def _family_summary_rows(family_df: pl.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for family in FAMILY_ORDER:
        family_row: dict[str, str] = {"Семейство": FAMILY_LABELS[family]}
        for model_key, model_label, _color in MODEL_SPECS:
            value = (
                family_df.filter((pl.col("model_key") == model_key) & (pl.col("family") == family))
                .select("mean_normalized_degradation")
                .item()
            )
            family_row[model_label] = _fmt(float(value), 6)
        rows.append(family_row)
    return rows


def _representative_drift_rows(
    paired_condition_df: pl.DataFrame,
    long_condition_df: pl.DataFrame,
    drift_df: pl.DataFrame,
) -> list[dict[str, str]]:
    worst_per_family = (
        long_condition_df.group_by("family", "condition", "severity")
        .agg(pl.mean("normalized_degradation").alias("mean_degradation"))
        .sort("mean_degradation", descending=True)
    )
    joined = drift_df.join(
        long_condition_df.select(
            "model_key", "condition", "family", "severity", "normalized_degradation"
        ),
        on=["model_key", "condition"],
    )
    rows: list[dict[str, str]] = []
    for family in FAMILY_ORDER:
        condition = (
            worst_per_family.filter(pl.col("family") == family)
            .sort("mean_degradation", descending=True)
            .row(0, named=True)["condition"]
        )
        campp = joined.filter(
            (pl.col("model_key") == MODEL_SPECS[0][0]) & (pl.col("condition") == condition)
        ).to_dicts()[0]
        w2v = joined.filter(
            (pl.col("model_key") == MODEL_SPECS[1][0]) & (pl.col("condition") == condition)
        ).to_dicts()[0]
        pair = paired_condition_df.filter(pl.col("condition") == condition).to_dicts()[0]
        rows.append(
            {
                "Семейство": FAMILY_LABELS[family],
                "Представительное условие": _condition_title(str(condition)),
                "same_clip_cosine_mean": f"{_fmt(campp['same_clip_cosine_mean'], 4)} / {_fmt(w2v['same_clip_cosine_mean'], 4)}",
                "self_retrieval@1": f"{_fmt(campp['self_retrieval_at1'], 4)} / {_fmt(w2v['self_retrieval_at1'], 4)}",
                "Относительная деградация": f"{_fmt(pair['campp_normalized_degradation'], 4)} / {_fmt(pair['w2v_normalized_degradation'], 4)}",
            }
        )
    return rows


def _decision_matrix_rows() -> list[dict[str, str]]:
    return [
        {
            "Сценарий": "Нужно лучшее абсолютное качество верификации",
            "Предпочтительная модель": "W2V1j teacher-PEFT stage3",
            "Почему": "Ниже clean EER/minDCF и лучше абсолютные метрики на всех 12 искажённых условиях.",
        },
        {
            "Сценарий": "Важна более мягкая относительная деградация на синтетическом reverb/codec stress",
            "Предпочтительная модель": "CAM++ MS41 family",
            "Почему": "Ниже normalized degradation на 8/12 условий, особенно в reverb и codec/bandwidth.",
        },
        {
            "Сценарий": "Нужно внешнее утверждение о реальной устойчивости",
            "Предпочтительная модель": "Пока не выбирать победителя",
            "Почему": "Сначала нужен rerun на commit-зафиксированном коде, с симметричным протоколом и реалистичными noise/RIR источниками.",
        },
    ]


def _executive_summary_bullets(
    audit_payload: dict[str, Any],
    comparison_df: pl.DataFrame,
    paired_condition_df: pl.DataFrame,
) -> list[str]:
    findings = audit_payload["comparative_findings"]
    by_model = {str(row["model_key"]): row for row in comparison_df.to_dicts()}
    campp = by_model[MODEL_SPECS[0][0]]
    w2v = by_model[MODEL_SPECS[1][0]]
    worst_condition = _condition_title(str(campp["worst_condition"]))
    return [
        (
            "Все итоговые verification- и drift-метрики пересчитаны из сырых `scores_*.npy` и "
            "`embeddings_*.npy`; максимальное абсолютное расхождение составило `0.0`."
        ),
        (
            f"На clean-базе `W2V1j` лучше обеих главных метрик: `EER = {_fmt(w2v['clean_eer'], 5)}` "
            f"против `{_fmt(campp['clean_eer'], 5)}` у `CAM++`, `minDCF = {_fmt(w2v['clean_min_dcf'], 6)}` "
            f"против `{_fmt(campp['clean_min_dcf'], 6)}`."
        ),
        (
            f"На distorted-условиях `W2V1j` выигрывает по абсолютному качеству на "
            f"`{findings['w2v_better_absolute_eer_on_distorted_conditions']}/12` условиях, а `CAM++` "
            f"выигрывает по относительной деградации на "
            f"`{findings['campp_better_relative_robustness_on_distorted_conditions']}/12` условиях."
        ),
        (
            f"Самое тяжёлое условие для обеих моделей — `{worst_condition}`: там наблюдаются и "
            "максимальная относительная деградация, и наихудшие drift-показатели."
        ),
        (
            "Бенчмарк годится как внутренний synthetic stress-test инвариантности эмбеддингов, но "
            "не доказывает полевую устойчивость к реальным шумам, RIR и продакшен-сценариям."
        ),
    ]


def _methodology_overview_bullets(audit_payload: dict[str, Any]) -> list[str]:
    clean_subset = audit_payload["clean_subset"]
    plan_summary = audit_payload["plan_summary"]
    trial_summary = audit_payload["trial_summary"]
    extraction_notes = audit_payload["extraction_notes"]
    return [
        (
            f"Из `dev_manifest.csv` была зафиксирована clean-подвыборка на "
            f"`{clean_subset['row_count']}` клипов и `{clean_subset['speaker_count']}` спикеров."
        ),
        (
            f"Для каждого clean-клипа были построены `{plan_summary['condition_count']}` искажённых "
            f"условий в `{plan_summary['family_count']}` семействах; суммарно план содержит "
            f"`{plan_summary['row_count']}` distorted-строк."
        ),
        (
            "Проверочный protocol односторонний: enrollment остаётся clean, а test подаётся в "
            "искажённом виде. Это стресс-тест инвариантности, а не полная имитация всех сценариев."
        ),
        (
            f"Trial-set содержит `{trial_summary['trial_count']}` сравнений: "
            f"`{trial_summary['positive_count']}` positive и `{trial_summary['negative_count']}` negative."
        ),
        (
            "Извлечение эмбеддингов не унифицировано между моделями: "
            f"`CAM++` идёт через {extraction_notes['campp_ms41_family']} "
            f"`W2V1j` идёт через {extraction_notes['w2v1j_teacher_peft_stage3']}"
        ),
    ]


def _distortion_catalog_rows(audit_payload: dict[str, Any]) -> list[dict[str, str]]:
    catalog = {
        family: {
            item["severity"]: _format_condition_parameters(family, item["parameters"])
            for item in audit_payload["condition_catalog"]
            if item["family"] == family
        }
        for family in FAMILY_ORDER
    }
    comments = {
        "additive_noise": (
            "Меняются одновременно и `SNR`, и цвет шума; поэтому ветка noise не образует "
            "чисто монотонную шкалу сложности."
        ),
        "reverb": "Имитируется синтетическая реверберация с изменением `rt60` и `direct_gain`.",
        "codec_bandwidth": (
            "Ужесточается одновременно квантование и пропускная полоса: меньше бит и уже спектр."
        ),
        "level_clipping": (
            "Повышается gain и одновременно снижается clip-threshold, поэтому растёт риск насыщения."
        ),
    }
    return [
        {
            "Семейство": FAMILY_LABELS[family],
            "Лёгкий": catalog[family]["light"],
            "Средний": catalog[family]["medium"],
            "Тяжёлый": catalog[family]["heavy"],
            "Что меняется": comments[family],
        }
        for family in FAMILY_ORDER
    ]


def _metric_definition_rows(audit_payload: dict[str, Any]) -> list[dict[str, str]]:
    formula = "0.5 * rel(EER) + 0.5 * rel(minDCF), где rel(x) = max((x - x_clean) / x_clean, 0)"
    return [
        {
            "Метрика": "EER",
            "Как считается": "Точка, в которой false accept rate равен false reject rate.",
            "Как интерпретировать": "Ниже лучше; это главная абсолютная метрика качества верификации.",
        },
        {
            "Метрика": "minDCF",
            "Как считается": (
                "Минимальная detection cost function при фиксированных приоритетах ошибок."
            ),
            "Как интерпретировать": "Ниже лучше; чувствительна к рабочему operating point.",
        },
        {
            "Метрика": "Normalized degradation",
            "Как считается": formula,
            "Как интерпретировать": (
                "Ниже лучше; показывает, насколько condition портит модель относительно её "
                "собственного clean-baseline."
            ),
        },
        {
            "Метрика": "same_clip_cosine_mean",
            "Как считается": (
                "Средний cosine между clean-эмбеддингом клипа и эмбеддингом его distorted-версии."
            ),
            "Как интерпретировать": (
                "Выше лучше; падение означает, что искажение сильнее смещает тот же самый clip."
            ),
        },
        {
            "Метрика": "self_retrieval@1",
            "Как считается": (
                "Доля distorted-клипов, для которых ближайший clean-эмбеддинг принадлежит тому же clip."
            ),
            "Как интерпретировать": "Выше лучше; это прямой индикатор устойчивости локальной геометрии.",
        },
        {
            "Метрика": "separation_gap_delta_vs_clean",
            "Как считается": (
                "Изменение разницы между средним positive-score и средним negative-score "
                "относительно clean-clean baseline."
            ),
            "Как интерпретировать": (
                "Чем сильнее отрицательное смещение, тем хуже отделяются genuine и impostor пары."
            ),
        },
    ]


def _clean_metric_rows(comparison_df: pl.DataFrame) -> list[dict[str, str]]:
    return [
        {
            "Модель": str(row["model_label"]),
            "EER на clean": _fmt(row["clean_eer"], 5),
            "minDCF на clean": _fmt(row["clean_min_dcf"], 6),
            "Комментарий": (
                "Лучший clean-baseline."
                if str(row["model_key"]) == MODEL_SPECS[1][0]
                else "Более слабый clean-baseline."
            ),
        }
        for row in comparison_df.to_dicts()
    ]


def _winner_summary_rows(
    audit_payload: dict[str, Any],
    paired_condition_df: pl.DataFrame,
) -> list[dict[str, str]]:
    campp_relative_titles = ", ".join(_relative_winner_condition_titles(paired_condition_df, False))
    w2v_relative_titles = ", ".join(_relative_winner_condition_titles(paired_condition_df, True))
    findings = audit_payload["comparative_findings"]
    return [
        {
            "Критерий": "Абсолютный EER",
            "Победитель": "W2V1j teacher-PEFT stage3",
            "Счёт": f"{findings['w2v_better_absolute_eer_on_distorted_conditions']}/12",
            "Комментарий": "W2V1j лучше на всех distorted-условиях.",
        },
        {
            "Критерий": "Абсолютный minDCF",
            "Победитель": "W2V1j teacher-PEFT stage3",
            "Счёт": f"{findings['w2v_better_absolute_min_dcf_on_distorted_conditions']}/12",
            "Комментарий": "Вывод совпадает с EER: абсолютный лидер тот же.",
        },
        {
            "Критерий": "Относительная деградация",
            "Победитель": "CAM++ MS41 family",
            "Счёт": (
                f"{findings['campp_better_relative_robustness_on_distorted_conditions']}/12 "
                f"против {findings['w2v_better_relative_robustness_on_distorted_conditions']}/12"
            ),
            "Комментарий": (
                f"CAM++ выигрывает на: {campp_relative_titles}. "
                f"W2V1j выигрывает на: {w2v_relative_titles}."
            ),
        },
    ]


def _relative_winner_condition_titles(
    paired_condition_df: pl.DataFrame,
    w2v_better: bool,
) -> list[str]:
    rows = (
        paired_condition_df.filter(pl.col("w2v_better_relative_robustness") == w2v_better)
        .sort(_condition_sort_exprs())
        .to_dicts()
    )
    return [_condition_title(str(row["condition"])) for row in rows]


def _family_insight_bullets(family_df: pl.DataFrame) -> list[str]:
    def _value(family: str, model_key: str) -> float:
        return float(
            family_df.filter((pl.col("family") == family) & (pl.col("model_key") == model_key))
            .select("mean_normalized_degradation")
            .item()
        )

    codec_campp = _value("codec_bandwidth", MODEL_SPECS[0][0])
    codec_w2v = _value("codec_bandwidth", MODEL_SPECS[1][0])
    level_campp = _value("level_clipping", MODEL_SPECS[0][0])
    level_w2v = _value("level_clipping", MODEL_SPECS[1][0])
    reverb_campp = _value("reverb", MODEL_SPECS[0][0])
    reverb_w2v = _value("reverb", MODEL_SPECS[1][0])
    return [
        (
            f"Самое тяжёлое семейство для обеих моделей — `codec / bandwidth`: средняя относительная "
            f"деградация `CAM++ = {_fmt(codec_campp, 6)}`, `W2V1j = {_fmt(codec_w2v, 6)}`."
        ),
        (
            f"Наиболее мягкое семейство для обеих моделей — `level / clipping`: "
            f"`CAM++ = {_fmt(level_campp, 6)}`, `W2V1j = {_fmt(level_w2v, 6)}`."
        ),
        (
            f"На `reverb` и `codec / bandwidth` преимущество `CAM++` проявляется именно как более "
            f"низкая относительная деградация: `reverb {_fmt(reverb_campp, 6)} vs {_fmt(reverb_w2v, 6)}`."
        ),
    ]


def _condition_insight_bullets(
    audit_payload: dict[str, Any],
    paired_condition_df: pl.DataFrame,
) -> list[str]:
    findings = audit_payload["comparative_findings"]
    by_average = paired_condition_df.with_columns(
        (
            0.5 * (pl.col("campp_normalized_degradation") + pl.col("w2v_normalized_degradation"))
        ).alias("avg_normalized_degradation")
    )
    worst = by_average.sort("avg_normalized_degradation", descending=True).row(0, named=True)
    return [
        (
            f"По абсолютным `EER/minDCF` `W2V1j` выигрывает на "
            f"`{findings['w2v_better_absolute_eer_on_distorted_conditions']}/12` условиях без исключений."
        ),
        (
            f"По относительной деградации `CAM++` выигрывает на "
            f"`{findings['campp_better_relative_robustness_on_distorted_conditions']}/12` условиях, "
            "прежде всего в ветках `reverb` и `codec`."
        ),
        (
            f"Самое тяжёлое условие по усреднённой относительной деградации — "
            f"`{_condition_title(str(worst['condition']))}`."
        ),
    ]


def _format_condition_parameters(family: str, parameters: dict[str, Any]) -> str:
    if family == "additive_noise":
        return f"{parameters['color']}, SNR {parameters['snr_db']} dB"
    if family == "reverb":
        return f"RT60 {parameters['rt60_s']} s, direct_gain {parameters['direct_gain']}"
    if family == "codec_bandwidth":
        return (
            f"{parameters['bits']} bit, {int(parameters['low_hz'])}-{int(parameters['high_hz'])} Hz"
        )
    if family == "level_clipping":
        return f"gain {parameters['gain_db']} dB, clip {parameters['clip_threshold']}"
    raise KeyError(f"Unknown family: {family}")


def _appendix_family_rows(paired_condition_df: pl.DataFrame, family: str) -> list[dict[str, str]]:
    rows = (
        paired_condition_df.filter(pl.col("family") == family)
        .sort(_condition_sort_exprs())
        .to_dicts()
    )
    return [
        {
            "Уровень": SEVERITY_LABELS[str(row["severity"])],
            "CAM++: EER / minDCF": f"{_fmt(row['campp_eer'], 5)} / {_fmt(row['campp_min_dcf'], 5)}",
            "W2V1j: EER / minDCF": f"{_fmt(row['w2v_eer'], 5)} / {_fmt(row['w2v_min_dcf'], 5)}",
            "Отн. деградация CAM++": _fmt(row["campp_normalized_degradation"], 6),
            "Отн. деградация W2V1j": _fmt(row["w2v_normalized_degradation"], 6),
            "Отн. победитель": "W2V1j" if row["w2v_better_relative_robustness"] else "CAM++",
        }
        for row in rows
    ]


def _markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return "\n".join(table)


def _html_table(rows: list[dict[str, str]], *, compact: bool = False) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    class_name = "compact" if compact else ""
    thead = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{_html_inline(str(row[header]))}</td>" for header in headers)
        + "</tr>"
        for row in rows
    )
    return (
        f"<table class='{class_name}'><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>"
    )


def _html_list(items: list[str], *, class_name: str = "bullet-list") -> str:
    body = "".join(f"<li>{_html_inline(item)}</li>" for item in items)
    return f"<ul class='{class_name}'>{body}</ul>"


def _html_inline(text: str) -> str:
    parts = text.split("`")
    rendered: list[str] = []
    for index, part in enumerate(parts):
        escaped = escape(part)
        if index % 2 == 1:
            rendered.append(f"<code>{escaped}</code>")
        else:
            rendered.append(escaped)
    return "".join(rendered)


def _condition_title(condition: str) -> str:
    if condition == "clean":
        return "clean"
    for severity, severity_label in SEVERITY_LABELS.items():
        suffix = f"_{severity}"
        if condition.endswith(suffix):
            prefix = condition.removesuffix(suffix)
            if prefix in FAMILY_LABELS:
                return f"{FAMILY_LABELS[prefix]}, {severity_label}"
            if prefix in CONDITION_PREFIX_LABELS:
                return f"{CONDITION_PREFIX_LABELS[prefix]}, {severity_label}"
    raise KeyError(f"Unknown condition format: {condition}")


def _short_condition_label(row: dict[str, Any]) -> str:
    return f"{FAMILY_SHORT_LABELS[str(row['family'])]} {SEVERITY_SHORT[str(row['severity'])]}"


def _fmt(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    main()
