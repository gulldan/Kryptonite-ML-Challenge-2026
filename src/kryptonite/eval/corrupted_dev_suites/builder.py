"""Build reproducible corrupted dev suites from a clean dev manifest."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kryptonite.config import NormalizationConfig, SilenceAugmentationConfig
from kryptonite.data.audio_io import write_audio_file
from kryptonite.data.audio_loader import AudioLoadRequest, load_audio
from kryptonite.data.manifest_artifacts import (
    build_file_artifact,
    write_manifest_inventory,
    write_tabular_artifact,
)
from kryptonite.data.schema import ManifestRow
from kryptonite.deployment import resolve_project_path

from .audio import (
    apply_codec_transform,
    apply_distance_transform,
    apply_noise_transform,
    apply_reverb_transform,
    apply_silence_transform,
    load_codec_candidates,
    load_distance_candidates,
    load_noise_candidates,
    load_reverb_candidates,
    load_silence_candidates,
    pick_weighted_candidate,
    stable_rng,
)
from .models import (
    BuiltCorruptedSuite,
    CorruptedDevSuiteSpec,
    CorruptedDevSuitesPlan,
    CorruptedDevSuitesReport,
)

CATALOG_JSON_NAME = "corrupted_dev_suites_catalog.json"
CATALOG_MARKDOWN_NAME = "corrupted_dev_suites_catalog.md"
SUITE_SUMMARY_JSON_NAME = "suite_summary.json"
SUITE_SUMMARY_MARKDOWN_NAME = "suite_summary.md"


@dataclass(frozen=True, slots=True)
class _SourceUtterance:
    row: ManifestRow
    payload: dict[str, object]
    item_id: str
    source_audio_basename: str
    suite_audio_basename: str


def build_corrupted_dev_suites(
    *,
    project_root: Path | str,
    plan: CorruptedDevSuitesPlan,
    normalization_config: NormalizationConfig,
    silence_config: SilenceAugmentationConfig,
    plan_path: Path | str | None = None,
    noise_manifest_path: Path | str | None = None,
    rir_manifest_path: Path | str | None = None,
    room_config_manifest_path: Path | str | None = None,
    codec_plan_path: Path | str = "configs/corruption/codec-bank.toml",
    far_field_plan_path: Path | str = "configs/corruption/far-field-bank.toml",
    ffmpeg_path: str = "ffmpeg",
) -> CorruptedDevSuitesReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    output_root = resolve_project_path(str(project_root_path), plan.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_manifest_path = resolve_project_path(str(project_root_path), plan.source_manifest_path)
    source_rows = _load_source_manifest(source_manifest_path)
    source_trial_paths = _resolve_trial_manifest_paths(
        project_root=project_root_path,
        source_manifest_path=source_manifest_path,
        configured_paths=plan.trial_manifest_paths,
    )
    source_trial_rows = _load_trial_rows(source_trial_paths)

    resolved_noise_manifest_path = _resolve_optional_path(
        project_root_path,
        explicit=noise_manifest_path,
        default="artifacts/corruptions/noise-bank/manifests/noise_bank_manifest.jsonl",
    )
    resolved_rir_manifest_path = _resolve_optional_path(
        project_root_path,
        explicit=rir_manifest_path,
        default=_resolve_first_existing(
            project_root_path,
            "artifacts/corruptions/rir-bank/manifests/rir_bank_manifest.jsonl",
            "artifacts/corruptions/rir-bank*/manifests/rir_bank_manifest.jsonl",
        ),
    )
    resolved_room_config_manifest_path = _resolve_optional_path(
        project_root_path,
        explicit=room_config_manifest_path,
        default=_resolve_first_existing(
            project_root_path,
            "artifacts/corruptions/rir-bank/manifests/room_simulation_configs.jsonl",
            "artifacts/corruptions/rir-bank*/manifests/room_simulation_configs.jsonl",
        ),
    )
    resolved_codec_plan_path = resolve_project_path(str(project_root_path), str(codec_plan_path))
    resolved_far_field_plan_path = resolve_project_path(
        str(project_root_path), str(far_field_plan_path)
    )

    built_suites = tuple(
        _build_single_suite(
            project_root=project_root_path,
            suite_spec=suite_spec,
            plan_seed=plan.seed,
            source_rows=source_rows,
            source_trial_rows=source_trial_rows,
            suite_root=output_root / suite_spec.suite_id,
            normalization_config=normalization_config,
            silence_config=silence_config,
            noise_manifest_path=resolved_noise_manifest_path,
            rir_manifest_path=resolved_rir_manifest_path,
            room_config_manifest_path=resolved_room_config_manifest_path,
            codec_plan_path=resolved_codec_plan_path,
            far_field_plan_path=resolved_far_field_plan_path,
            ffmpeg_path=ffmpeg_path,
        )
        for suite_spec in plan.suites
    )

    catalog_json_path = output_root / CATALOG_JSON_NAME
    catalog_markdown_path = output_root / CATALOG_MARKDOWN_NAME
    report = CorruptedDevSuitesReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        plan_path=(
            _relative_to_project(
                resolve_project_path(str(project_root_path), str(plan_path)), project_root_path
            )
            if plan_path is not None
            else None
        ),
        source_manifest_path=_relative_to_project(source_manifest_path, project_root_path),
        source_trial_manifest_paths=tuple(
            _relative_to_project(path, project_root_path) for path in source_trial_paths
        ),
        output_root=_relative_to_project(output_root, project_root_path),
        seed=plan.seed,
        suites=built_suites,
        catalog_json_path=_relative_to_project(catalog_json_path, project_root_path),
        catalog_markdown_path=_relative_to_project(catalog_markdown_path, project_root_path),
    )
    catalog_json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    catalog_markdown_path.write_text(_render_catalog_markdown(report) + "\n")
    return report


def _build_single_suite(
    *,
    project_root: Path,
    suite_spec: CorruptedDevSuiteSpec,
    plan_seed: int,
    source_rows: tuple[_SourceUtterance, ...],
    source_trial_rows: dict[Path, list[dict[str, object]]],
    suite_root: Path,
    normalization_config: NormalizationConfig,
    silence_config: SilenceAugmentationConfig,
    noise_manifest_path: Path | None,
    rir_manifest_path: Path | None,
    room_config_manifest_path: Path | None,
    codec_plan_path: Path,
    far_field_plan_path: Path,
    ffmpeg_path: str,
) -> BuiltCorruptedSuite:
    suite_root.mkdir(parents=True, exist_ok=True)
    audio_root = suite_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    candidates = _load_suite_candidates(
        project_root=project_root,
        suite_spec=suite_spec,
        silence_config=silence_config,
        noise_manifest_path=noise_manifest_path,
        rir_manifest_path=rir_manifest_path,
        room_config_manifest_path=room_config_manifest_path,
        codec_plan_path=codec_plan_path,
        far_field_plan_path=far_field_plan_path,
    )
    if not candidates:
        raise ValueError(f"No candidates available for suite {suite_spec.suite_id!r}.")

    request = AudioLoadRequest.from_config(normalization_config)
    manifest_rows: list[dict[str, object]] = []
    severity_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    basename_map: dict[str, str] = {}

    for source_row in source_rows:
        rng = stable_rng(seed=plan_seed, namespace=suite_spec.suite_id, item_id=source_row.item_id)
        loaded = load_audio(
            source_row.row.audio_path,
            project_root=project_root,
            request=request,
        )
        candidate = pick_weighted_candidate(candidates, rng=rng)
        outcome = _apply_suite_candidate(
            project_root=project_root,
            suite_spec=suite_spec,
            candidate=candidate,
            waveform=loaded.waveform,
            sample_rate_hz=loaded.sample_rate_hz,
            rng=rng,
            ffmpeg_path=ffmpeg_path,
        )
        output_path = audio_root / source_row.suite_audio_basename
        write_audio_file(
            path=output_path,
            waveform=outcome.waveform,
            sample_rate_hz=outcome.sample_rate_hz,
            output_format=normalization_config.output_format,
            pcm_bits_per_sample=normalization_config.output_pcm_bits_per_sample,
        )
        basename_map[source_row.source_audio_basename] = source_row.suite_audio_basename
        severity_counts[outcome.severity] += 1
        candidate_counts[outcome.candidate_id] += 1
        manifest_rows.append(
            _build_suite_manifest_row(
                source=source_row,
                suite_spec=suite_spec,
                project_root=project_root,
                output_path=output_path,
                sample_rate_hz=outcome.sample_rate_hz,
                num_channels=int(outcome.waveform.shape[0]),
                duration_seconds=round(
                    float(outcome.waveform.shape[-1]) / float(outcome.sample_rate_hz),
                    6,
                ),
                candidate_id=outcome.candidate_id,
                severity=outcome.severity,
                metadata=outcome.metadata,
            )
        )

    manifest_path = suite_root / "dev_manifest.jsonl"
    manifest_artifact = write_tabular_artifact(
        name="dev_manifest",
        kind="data_manifest",
        rows=manifest_rows,
        jsonl_path=manifest_path,
        project_root=str(project_root),
    )
    trial_artifacts = []
    copied_trial_paths: list[str] = []
    for source_trial_path, rows in source_trial_rows.items():
        target_path = suite_root / source_trial_path.name
        translated_rows = [_translate_trial_row(row=row, basename_map=basename_map) for row in rows]
        artifact = write_tabular_artifact(
            name=source_trial_path.stem,
            kind="trial_list",
            rows=translated_rows,
            jsonl_path=target_path,
            project_root=str(project_root),
            field_order=("label", "left_audio", "right_audio"),
        )
        trial_artifacts.append(artifact)
        copied_trial_paths.append(artifact.jsonl_path)

    inventory_path = suite_root / "manifest_inventory.json"
    suite_summary_json_path = suite_root / SUITE_SUMMARY_JSON_NAME
    suite_summary_markdown_path = suite_root / SUITE_SUMMARY_MARKDOWN_NAME
    provisional_suite = BuiltCorruptedSuite(
        suite_id=suite_spec.suite_id,
        family=suite_spec.family,
        description=suite_spec.description,
        seed=plan_seed,
        utterance_count=len(manifest_rows),
        speaker_count=len({str(row["speaker_id"]) for row in manifest_rows}),
        total_duration_seconds=_manifest_total_duration(manifest_rows),
        severity_counts=dict(severity_counts),
        candidate_counts=dict(candidate_counts),
        output_root=_relative_to_project(suite_root, project_root),
        audio_root=_relative_to_project(audio_root, project_root),
        manifest_path=manifest_artifact.jsonl_path,
        inventory_path=_relative_to_project(inventory_path, project_root),
        suite_summary_json_path=_relative_to_project(suite_summary_json_path, project_root),
        suite_summary_markdown_path=_relative_to_project(suite_summary_markdown_path, project_root),
        trial_manifest_paths=tuple(copied_trial_paths),
    )
    suite_summary_json_path.write_text(
        json.dumps(provisional_suite.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    suite_summary_markdown_path.write_text(_render_suite_markdown(provisional_suite) + "\n")
    write_manifest_inventory(
        dataset=str(manifest_rows[0]["dataset"]),
        inventory_path=inventory_path,
        project_root=str(project_root),
        manifest_tables=(manifest_artifact,),
        auxiliary_tables=tuple(trial_artifacts),
        auxiliary_files=(
            build_file_artifact(
                name="suite_summary_json",
                kind="metadata",
                path=suite_summary_json_path,
                project_root=str(project_root),
            ),
            build_file_artifact(
                name="suite_summary_markdown",
                kind="documentation",
                path=suite_summary_markdown_path,
                project_root=str(project_root),
            ),
        ),
    )
    return provisional_suite


def _load_suite_candidates(
    *,
    project_root: Path,
    suite_spec: CorruptedDevSuiteSpec,
    silence_config: SilenceAugmentationConfig,
    noise_manifest_path: Path | None,
    rir_manifest_path: Path | None,
    room_config_manifest_path: Path | None,
    codec_plan_path: Path,
    far_field_plan_path: Path,
) -> tuple[Any, ...]:
    if suite_spec.family == "noise":
        if noise_manifest_path is None:
            return ()
        return load_noise_candidates(
            project_root=project_root,
            manifest_path=noise_manifest_path,
            suite_spec=suite_spec,
        )
    if suite_spec.family == "reverb":
        if rir_manifest_path is None or room_config_manifest_path is None:
            return ()
        return load_reverb_candidates(
            project_root=project_root,
            rir_manifest_path=rir_manifest_path,
            room_config_manifest_path=room_config_manifest_path,
            suite_spec=suite_spec,
        )
    if suite_spec.family == "codec":
        return load_codec_candidates(plan_path=codec_plan_path, suite_spec=suite_spec)
    if suite_spec.family == "distance":
        return load_distance_candidates(plan_path=far_field_plan_path, suite_spec=suite_spec)
    return load_silence_candidates(suite_spec=suite_spec, silence_config=silence_config)


def _apply_suite_candidate(
    *,
    project_root: Path,
    suite_spec: CorruptedDevSuiteSpec,
    candidate: Any,
    waveform: Any,
    sample_rate_hz: int,
    rng: Any,
    ffmpeg_path: str,
) -> Any:
    if suite_spec.family == "noise":
        return apply_noise_transform(
            waveform=waveform,
            sample_rate_hz=sample_rate_hz,
            candidate=candidate,
            rng=rng,
            project_root=project_root,
        )
    if suite_spec.family == "reverb":
        return apply_reverb_transform(
            waveform=waveform,
            sample_rate_hz=sample_rate_hz,
            candidate=candidate,
            rng=rng,
            project_root=project_root,
        )
    if suite_spec.family == "codec":
        return apply_codec_transform(
            waveform=waveform,
            sample_rate_hz=sample_rate_hz,
            candidate=candidate,
            ffmpeg_path=ffmpeg_path,
        )
    if suite_spec.family == "distance":
        return apply_distance_transform(
            waveform=waveform,
            sample_rate_hz=sample_rate_hz,
            candidate=candidate,
        )
    return apply_silence_transform(
        waveform=waveform,
        sample_rate_hz=sample_rate_hz,
        candidate=candidate,
        rng=rng,
    )


def _build_suite_manifest_row(
    *,
    source: _SourceUtterance,
    suite_spec: CorruptedDevSuiteSpec,
    project_root: Path,
    output_path: Path,
    sample_rate_hz: int,
    num_channels: int,
    duration_seconds: float,
    candidate_id: str,
    severity: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    payload = dict(source.payload)
    payload["dataset"] = f"{source.row.dataset}-{suite_spec.suite_id}"
    payload["source_dataset"] = source.row.dataset
    payload["audio_path"] = _relative_to_project(output_path, project_root)
    payload["duration_seconds"] = duration_seconds
    payload["sample_rate_hz"] = sample_rate_hz
    payload["num_channels"] = num_channels
    payload["corruption_suite"] = suite_spec.suite_id
    payload["corruption_family"] = suite_spec.family
    payload["corruption_candidate_id"] = candidate_id
    payload["corruption_severity"] = severity
    payload["source_audio_path"] = source.row.audio_path
    payload["source_audio_basename"] = source.source_audio_basename
    payload["corruption_metadata"] = metadata
    if "target_snr_db" in metadata:
        payload["snr_db"] = metadata["target_snr_db"]
    if "rir_id" in metadata:
        payload["rir_id"] = metadata["rir_id"]
    return payload


def _translate_trial_row(
    *,
    row: dict[str, object],
    basename_map: dict[str, str],
) -> dict[str, object]:
    payload = dict(row)
    for field_name in ("left_audio", "right_audio"):
        value = row.get(field_name)
        if value is None:
            continue
        payload[field_name] = basename_map.get(str(value), str(value))
    return payload


def _load_source_manifest(path: Path) -> tuple[_SourceUtterance, ...]:
    rows: list[_SourceUtterance] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Manifest line {line_number} is not a JSON object.")
        row = ManifestRow.from_mapping(payload, manifest_path=str(path), line_number=line_number)
        source_audio_basename = Path(row.audio_path).name
        suite_audio_basename = f"{Path(source_audio_basename).stem}.wav"
        item_id = row.utterance_id or source_audio_basename
        rows.append(
            _SourceUtterance(
                row=row,
                payload=dict(payload),
                item_id=item_id,
                source_audio_basename=source_audio_basename,
                suite_audio_basename=suite_audio_basename,
            )
        )
    if not rows:
        raise ValueError(f"Source manifest is empty: {path}")
    return tuple(rows)


def _resolve_trial_manifest_paths(
    *,
    project_root: Path,
    source_manifest_path: Path,
    configured_paths: tuple[str, ...],
) -> tuple[Path, ...]:
    if configured_paths:
        return tuple(resolve_project_path(str(project_root), path) for path in configured_paths)
    candidates = (
        source_manifest_path.parent / "official_dev_trials.jsonl",
        source_manifest_path.parent / "speaker_disjoint_dev_trials.jsonl",
    )
    return tuple(path for path in candidates if path.exists())


def _load_trial_rows(paths: tuple[Path, ...]) -> dict[Path, list[dict[str, object]]]:
    return {
        path: [
            payload
            for line in path.read_text().splitlines()
            if line.strip()
            for payload in [json.loads(line)]
            if isinstance(payload, dict)
        ]
        for path in paths
    }


def _resolve_optional_path(
    project_root: Path,
    *,
    explicit: Path | str | None,
    default: Path | str | None,
) -> Path | None:
    if explicit is not None:
        return resolve_project_path(str(project_root), str(explicit))
    if default is None:
        return None
    return resolve_project_path(str(project_root), str(default))


def _resolve_first_existing(project_root: Path, *patterns: str) -> Path | None:
    for pattern in patterns:
        if "*" in pattern:
            matches = sorted(project_root.glob(pattern))
            if matches:
                return matches[0]
            continue
        candidate = project_root / pattern
        if candidate.exists():
            return candidate
    return None


def _relative_to_project(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _manifest_total_duration(rows: list[dict[str, object]]) -> float:
    total = 0.0
    for row in rows:
        value = row.get("duration_seconds")
        if value is None:
            continue
        total += _coerce_float_value(value)
    return round(total, 6)


def _coerce_float_value(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"Expected numeric duration value, got {type(value)!r}")
    return float(value)


def _render_suite_markdown(suite: BuiltCorruptedSuite) -> str:
    lines = [
        f"# {suite.suite_id}",
        "",
        f"- family: `{suite.family}`",
        f"- seed: `{suite.seed}`",
        f"- utterances: `{suite.utterance_count}`",
        f"- speakers: `{suite.speaker_count}`",
        f"- total duration (s): `{suite.total_duration_seconds:.3f}`",
        f"- manifest: `{suite.manifest_path}`",
    ]
    if suite.trial_manifest_paths:
        lines.append("- trials: " + ", ".join(f"`{path}`" for path in suite.trial_manifest_paths))
    lines.extend(
        [
            "",
            "## Severity Counts",
            "",
            _format_key_value_lines(suite.severity_counts),
            "",
            "## Candidate Counts",
            "",
            _format_key_value_lines(suite.candidate_counts),
        ]
    )
    return "\n".join(lines)


def _render_catalog_markdown(report: CorruptedDevSuitesReport) -> str:
    lines = [
        "# Corrupted Dev Suites",
        "",
        f"- generated at: `{report.generated_at}`",
        f"- source manifest: `{report.source_manifest_path}`",
        f"- seed: `{report.seed}`",
        "",
        "## Suites",
        "",
    ]
    for suite in report.suites:
        lines.extend(
            [
                f"### {suite.suite_id}",
                "",
                f"- family: `{suite.family}`",
                f"- manifest: `{suite.manifest_path}`",
                f"- inventory: `{suite.inventory_path}`",
                f"- utterances: `{suite.utterance_count}`",
                f"- speakers: `{suite.speaker_count}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _format_key_value_lines(values: dict[str, int]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(values.items()))


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")
