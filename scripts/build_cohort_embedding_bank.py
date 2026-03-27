"""Build a reproducible cohort-embedding bank from exported embedding artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.config import load_project_config
from kryptonite.eval import CohortEmbeddingBankSelection, build_cohort_embedding_bank

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPTION = typer.Option(
    Path("configs/base.toml"),
    "--config",
    help="Path to the base ProjectConfig TOML file.",
)
EMBEDDINGS_OPTION = typer.Option(
    ...,
    "--embeddings",
    exists=True,
    dir_okay=False,
    help="Path to the source embeddings .npz/.npy artifact.",
)
METADATA_OPTION = typer.Option(
    ...,
    "--metadata",
    exists=True,
    dir_okay=False,
    help="Path to the aligned metadata .jsonl/.csv/.parquet artifact.",
)
OUTPUT_DIR_OPTION = typer.Option(
    Path("artifacts/eval/cohort-bank"),
    "--output-dir",
    help="Directory where the cohort bank artifacts will be written.",
)
INCLUDE_ROLE_OPTION = typer.Option(
    None,
    "--include-role",
    help="Keep only metadata rows with these role values. Can be repeated.",
)
INCLUDE_SPLIT_OPTION = typer.Option(
    None,
    "--include-split",
    help="Keep only metadata rows with these split values. Can be repeated.",
)
INCLUDE_DATASET_OPTION = typer.Option(
    None,
    "--include-dataset",
    help="Keep only metadata rows with these dataset values. Can be repeated.",
)
TRIALS_OPTION = typer.Option(
    None,
    "--trials",
    exists=True,
    dir_okay=False,
    help=(
        "Verification trial JSONL file(s) whose utterance ids should be excluded from the "
        "cohort bank when possible. Can be repeated."
    ),
)
VALIDATION_MANIFEST_OPTION = typer.Option(
    None,
    "--validate-disjoint-speakers-against",
    exists=True,
    dir_okay=False,
    help=(
        "Manifest JSONL file(s) whose speaker ids must stay disjoint from the cohort bank. "
        "Can be repeated."
    ),
)
STRICT_SPEAKER_DISJOINTNESS_OPTION = typer.Option(
    True,
    "--strict-speaker-disjointness/--no-strict-speaker-disjointness",
    help="Fail the build when selected cohort speakers overlap the validation manifests.",
)
ALLOW_TRIAL_OVERLAP_FALLBACK_OPTION = typer.Option(
    True,
    "--allow-trial-overlap-fallback/--no-allow-trial-overlap-fallback",
    help=(
        "If trial exclusions would empty the cohort bank, keep the pre-exclusion rows instead "
        "and record that fallback in the summary."
    ),
)
MIN_EMBEDDINGS_PER_SPEAKER_OPTION = typer.Option(
    1,
    "--min-embeddings-per-speaker",
    min=1,
    help="Minimum number of kept utterance embeddings required per speaker.",
)
MAX_EMBEDDINGS_PER_SPEAKER_OPTION = typer.Option(
    None,
    "--max-embeddings-per-speaker",
    min=1,
    help="Optional cap on the number of kept utterance embeddings per speaker.",
)
MAX_EMBEDDINGS_OPTION = typer.Option(
    None,
    "--max-embeddings",
    min=1,
    help="Optional global cap on the number of kept cohort embeddings.",
)
POINT_ID_FIELD_OPTION = typer.Option(
    "atlas_point_id",
    "--point-id-field",
    help="Metadata field aligned with the source .npz point ids.",
)
EMBEDDINGS_KEY_OPTION = typer.Option(
    "embeddings",
    "--embeddings-key",
    help="Array key to use when the source embeddings artifact is a .npz file.",
)
IDS_KEY_OPTION = typer.Option(
    "point_ids",
    "--ids-key",
    help=("Optional point-id array key to use when the source embeddings artifact is a .npz file."),
)
OVERRIDE_OPTION = typer.Option(
    None,
    "--override",
    help="ProjectConfig override in dotted.key=value form. Can be repeated.",
)
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    help="Output format: text or json.",
    case_sensitive=False,
)


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    embeddings: Path = EMBEDDINGS_OPTION,
    metadata: Path = METADATA_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    include_role: list[str] | None = INCLUDE_ROLE_OPTION,
    include_split: list[str] | None = INCLUDE_SPLIT_OPTION,
    include_dataset: list[str] | None = INCLUDE_DATASET_OPTION,
    trials: list[Path] | None = TRIALS_OPTION,
    validate_disjoint_speakers_against: list[Path] | None = VALIDATION_MANIFEST_OPTION,
    strict_speaker_disjointness: bool = STRICT_SPEAKER_DISJOINTNESS_OPTION,
    allow_trial_overlap_fallback: bool = ALLOW_TRIAL_OVERLAP_FALLBACK_OPTION,
    min_embeddings_per_speaker: int = MIN_EMBEDDINGS_PER_SPEAKER_OPTION,
    max_embeddings_per_speaker: int | None = MAX_EMBEDDINGS_PER_SPEAKER_OPTION,
    max_embeddings: int | None = MAX_EMBEDDINGS_OPTION,
    point_id_field: str = POINT_ID_FIELD_OPTION,
    embeddings_key: str = EMBEDDINGS_KEY_OPTION,
    ids_key: str | None = IDS_KEY_OPTION,
    override: list[str] | None = OVERRIDE_OPTION,
    output: str = OUTPUT_OPTION,
) -> None:
    project = load_project_config(config_path=config, overrides=override or [])
    built = build_cohort_embedding_bank(
        project_root=project.paths.project_root,
        output_root=output_dir,
        embeddings_path=embeddings,
        metadata_path=metadata,
        selection=CohortEmbeddingBankSelection(
            include_roles=tuple(include_role or ()),
            include_splits=tuple(include_split or ()),
            include_datasets=tuple(include_dataset or ()),
            min_embeddings_per_speaker=min_embeddings_per_speaker,
            max_embeddings_per_speaker=max_embeddings_per_speaker,
            max_embeddings=max_embeddings,
            trial_paths=tuple(str(path) for path in (trials or [])),
            validation_manifest_paths=tuple(
                str(path) for path in (validate_disjoint_speakers_against or [])
            ),
            strict_speaker_disjointness=strict_speaker_disjointness,
            allow_trial_overlap_fallback=allow_trial_overlap_fallback,
            point_id_field=point_id_field,
            embeddings_key=embeddings_key,
            ids_key=ids_key,
        ),
    )

    if output == "json":
        typer.echo(json.dumps(built.to_dict(), indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")

    typer.echo(
        "\n".join(
            [
                "Cohort embedding bank build complete",
                f"Output root: {built.output_root}",
                f"Embeddings: {built.embeddings_path}",
                f"Metadata: {built.metadata_parquet_path}",
                f"Summary: {built.summary_path}",
                f"Selected embeddings: {built.summary.selected_row_count}",
                f"Selected speakers: {built.summary.selected_speaker_count}",
                f"Trial-overlap fallback used: {built.summary.trial_overlap_fallback_used}",
                (
                    "Validation speaker overlap: "
                    + (
                        ", ".join(built.summary.overlapping_validation_speakers)
                        if built.summary.overlapping_validation_speakers
                        else "none"
                    )
                ),
            ]
        )
    )


if __name__ == "__main__":
    app()
