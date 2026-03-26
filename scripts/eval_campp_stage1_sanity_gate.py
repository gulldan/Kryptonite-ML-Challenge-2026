"""Re-evaluate the stage-1 sanity gate on an existing run using centered scores.

Loads saved embeddings from a completed run and re-scores them with global mean
subtraction + length normalization (the geometry the gate always should have used).
Writes a fresh random_init_sanity.json and a short summary report.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)

RUN_DIR_ARG = typer.Argument(
    ...,
    help="Path to the completed stage-1 run directory (contains campp_stage1_encoder.pt).",
)
TRIALS_OPTION = typer.Option(
    Path("artifacts/manifests/ffsvc2022-surrogate/speaker_disjoint_dev_trials.jsonl"),
    "--trials",
    help="Trials manifest to score against.",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    help="Print results without writing random_init_sanity.json.",
)


@app.command()
def main(
    run_dir: Path = RUN_DIR_ARG,
    trials: Path = TRIALS_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    from kryptonite.eval import (
        build_verification_evaluation_report,
        load_verification_score_rows,
        write_verification_evaluation_report,
    )
    from kryptonite.training.speaker_baseline import (
        RANDOM_INIT_SANITY_DIR_NAME,
        RANDOM_INIT_SANITY_FILE_NAME,
        build_sanity_comparison,
        build_sanity_evaluation_summary,
        score_trials,
    )

    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise typer.BadParameter(f"Run directory not found: {run_dir}")

    # Load trials once — shared by both scored passes.
    trial_rows = [
        json.loads(line) for line in trials.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    def eval_pass(
        embeddings_dir: Path,
        metadata_jsonl: Path,
        label: str,
    ) -> tuple[object, object]:
        """Score one pass with centering, return (score_summary, verification_report)."""
        reeval_dir = run_dir / "reeval_centered" / label
        reeval_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = [
            json.loads(line)
            for line in metadata_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ss = score_trials(
            output_root=reeval_dir,
            trials_path=trials,
            metadata_rows=metadata_rows,
            trial_rows=trial_rows,
            embeddings_path=embeddings_dir / "dev_embeddings.npz",
            center_embeddings=True,
        )
        score_rows = load_verification_score_rows(ss.scores_path)
        report = write_verification_evaluation_report(
            build_verification_evaluation_report(
                score_rows,
                scores_path=ss.scores_path,
                trial_rows=trial_rows,
                metadata_rows=metadata_rows,
            ),
            output_root=reeval_dir,
        )
        return ss, report

    sanity_dir = run_dir / RANDOM_INIT_SANITY_DIR_NAME

    typer.echo("Scoring random-init embeddings with centering…")
    init_ss, init_report = eval_pass(
        embeddings_dir=sanity_dir,
        metadata_jsonl=sanity_dir / "dev_embedding_metadata.jsonl",
        label="random_init",
    )

    typer.echo("Scoring trained embeddings with centering…")
    final_ss, final_report = eval_pass(
        embeddings_dir=run_dir,
        metadata_jsonl=run_dir / "dev_embedding_metadata.jsonl",
        label="trained",
    )

    init_summary = build_sanity_evaluation_summary(
        label="random_init",
        output_root=run_dir / "reeval_centered" / "random_init",
        score_summary=init_ss,
        verification_report=init_report,
    )
    final_summary = build_sanity_evaluation_summary(
        label="trained",
        output_root=run_dir / "reeval_centered" / "trained",
        score_summary=final_ss,
        verification_report=final_report,
    )
    comparison = build_sanity_comparison(initial=init_summary, final=final_summary)

    typer.echo("")
    typer.echo(f"Gating rule : {comparison.gating_rule}")
    typer.echo(
        "Score-gap   : "
        f"{init_summary.score_gap:.6f} → {final_summary.score_gap:.6f}  "
        f"improved={comparison.score_gap_improved}"
    )
    typer.echo(
        "EER         : "
        f"{init_summary.eer:.6f} → {final_summary.eer:.6f}  "
        f"improved={comparison.eer_improved}"
    )
    typer.echo(
        "MinDCF      : "
        f"{init_summary.min_dcf:.6f} → {final_summary.min_dcf:.6f}  "
        f"improved={comparison.min_dcf_improved}"
    )
    typer.echo(f"Gate passed : {comparison.passed}")

    if not dry_run:
        sanity_path = run_dir / RANDOM_INIT_SANITY_FILE_NAME
        sanity_path.write_text(
            json.dumps(comparison.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        typer.echo(f"\nUpdated: {sanity_path}")
    else:
        typer.echo("\n(dry-run: random_init_sanity.json not updated)")


if __name__ == "__main__":
    app()
