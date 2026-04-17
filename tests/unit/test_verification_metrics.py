from __future__ import annotations

from pathlib import Path

from kryptonite.eval import compute_verification_metrics, load_verification_score_rows


def test_compute_verification_metrics_reports_perfect_separation() -> None:
    metrics = compute_verification_metrics(
        [
            {"label": 1, "score": 0.9},
            {"label": 1, "score": 0.8},
            {"label": 0, "score": 0.2},
            {"label": 0, "score": 0.1},
        ]
    )

    assert metrics.trial_count == 4
    assert metrics.positive_count == 2
    assert metrics.negative_count == 2
    assert metrics.eer == 0.0
    assert metrics.min_dcf == 0.0


def test_compute_verification_metrics_matches_known_overlap_case() -> None:
    metrics = compute_verification_metrics(
        [
            {"label": 1, "score": 0.9},
            {"label": 1, "score": 0.2},
            {"label": 0, "score": 0.8},
            {"label": 0, "score": 0.1},
        ]
    )

    assert metrics.eer == 0.5
    assert metrics.eer_threshold == 0.8
    assert metrics.min_dcf == 0.5
    assert metrics.min_dcf_threshold == 0.9


def test_load_verification_score_rows_reads_jsonl(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.jsonl"
    score_path.write_text(
        "\n".join(
            [
                '{"label": 1, "score": 0.9}',
                '{"label": 0, "score": 0.2}',
            ]
        )
        + "\n"
    )

    rows = load_verification_score_rows(score_path)

    assert rows == [{"label": 1, "score": 0.9}, {"label": 0, "score": 0.2}]
