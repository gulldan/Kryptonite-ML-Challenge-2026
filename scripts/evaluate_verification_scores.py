"""Compute EER and normalized minDCF for baseline score files."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.eval import compute_verification_metrics, load_verification_score_rows

app = typer.Typer(add_completion=False, help=__doc__)

SCORES_OPTION = typer.Option(
    ...,
    "--scores",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Path to a JSONL score file with `label` and `score` fields.",
)
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    help="Output format: text or json.",
    case_sensitive=False,
)
P_TARGET_OPTION = typer.Option(0.01, "--p-target", help="Target prior used for minDCF.")
C_MISS_OPTION = typer.Option(1.0, "--c-miss", help="Miss cost used for minDCF.")
C_FA_OPTION = typer.Option(1.0, "--c-fa", help="False-accept cost used for minDCF.")


@app.command()
def main(
    scores: Path = SCORES_OPTION,
    output: str = OUTPUT_OPTION,
    p_target: float = P_TARGET_OPTION,
    c_miss: float = C_MISS_OPTION,
    c_fa: float = C_FA_OPTION,
) -> None:
    rows = load_verification_score_rows(scores)
    metrics = compute_verification_metrics(
        rows,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )

    if output == "json":
        typer.echo(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")

    typer.echo(
        "\n".join(
            [
                "Verification score evaluation complete",
                f"Scores: {scores}",
                f"Trials: {metrics.trial_count}",
                f"Positives: {metrics.positive_count}",
                f"Negatives: {metrics.negative_count}",
                f"EER: {metrics.eer:.6f}",
                f"EER threshold: {metrics.eer_threshold:.6f}",
                f"MinDCF@Ptarget={metrics.p_target:.4f}: {metrics.min_dcf:.6f}",
                f"MinDCF threshold: {metrics.min_dcf_threshold:.6f}",
            ]
        )
    )


if __name__ == "__main__":
    app()
