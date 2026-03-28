from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from kryptonite.config import load_project_config
from kryptonite.eval import (
    build_onnx_parity_report,
    load_onnx_parity_config,
    write_onnx_parity_report,
)
from kryptonite.models import CAMPPlusConfig, CAMPPlusEncoder
from kryptonite.serve.onnx_export import CAMPPONNXExportRequest, export_campp_checkpoint_to_onnx


def test_onnx_parity_report_writes_outputs_and_promotes_metadata(tmp_path: Path) -> None:
    config_path, metadata_path = _write_onnx_parity_fixture(tmp_path, zero_tolerances=False)

    config = load_onnx_parity_config(config_path=config_path)
    report = build_onnx_parity_report(config, config_path=config_path)
    written = write_onnx_parity_report(report)

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert written.summary.passed is True
    assert written.promotion.applied is True
    assert Path(written.report_markdown_path).is_file()
    assert Path(written.audio_rows_path).is_file()
    assert Path(written.trial_rows_path).is_file()
    assert payload["summary"]["passed"] is True
    assert len(payload["variants"]) == 4
    assert metadata["inference_package"]["validated_backends"]["onnxruntime"] is True
    assert metadata["export_validation"]["runtime_backends_promoted"] is True
    assert (
        metadata["export_validation"]["onnx_parity_report_path"]
        == "artifacts/release/current/onnx_parity_report.json"
    )


def test_onnx_parity_report_skips_metadata_promotion_when_tolerances_fail(tmp_path: Path) -> None:
    config_path, metadata_path = _write_onnx_parity_fixture(tmp_path, zero_tolerances=True)

    config = load_onnx_parity_config(config_path=config_path)
    report = build_onnx_parity_report(config, config_path=config_path)
    written = write_onnx_parity_report(report)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert written.summary.passed is False
    assert written.promotion.applied is False
    assert "metadata promotion skipped" in str(written.promotion.error)
    assert metadata["inference_package"]["validated_backends"]["onnxruntime"] is False
    assert metadata["export_validation"]["runtime_backends_promoted"] is False


def _write_onnx_parity_fixture(
    tmp_path: Path,
    *,
    zero_tolerances: bool,
) -> tuple[Path, Path]:
    checkpoint_dir = tmp_path / "artifacts" / "baselines" / "campp" / "run-001"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "campp_encoder.pt"

    torch.manual_seed(0)
    model_config = CAMPPlusConfig(
        feat_dim=16,
        embedding_size=32,
        growth_rate=4,
        bottleneck_scale=2,
        init_channels=8,
        head_channels=4,
        head_res_blocks=(1, 1),
        block_layers=(1, 1, 1),
        block_kernel_sizes=(3, 3, 3),
        block_dilations=(1, 1, 2),
        memory_efficient=False,
    )
    model = CAMPPlusEncoder(model_config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": {},
            "model_config": asdict(model_config),
            "baseline_config": {},
            "speaker_to_index": {"speaker_alpha": 0, "speaker_bravo": 1},
        },
        checkpoint_path,
    )

    project_config = load_project_config(
        config_path=Path("configs/base.toml"),
        overrides=[
            f'paths.project_root="{tmp_path.as_posix()}"',
            "features.num_mel_bins=16",
            "tracking.enabled=false",
            "runtime.num_workers=0",
        ],
    )
    exported = export_campp_checkpoint_to_onnx(
        config=project_config,
        request=CAMPPONNXExportRequest(
            checkpoint_path=str(checkpoint_dir),
            output_root="artifacts/model-bundle-campp-test",
            sample_frame_count=120,
        ),
    )
    metadata_path = tmp_path / exported.metadata_path

    audio_root = tmp_path / "artifacts" / "demo-subset"
    _write_tone(audio_root / "enrollment" / "speaker_alpha-enroll_01.wav", frequency_hz=220.0)
    _write_tone(audio_root / "enrollment" / "speaker_bravo-enroll_01.wav", frequency_hz=330.0)
    _write_tone(audio_root / "test" / "speaker_alpha-test_01.wav", frequency_hz=224.0)
    _write_tone(audio_root / "test" / "speaker_bravo-test_01.wav", frequency_hz=334.0)

    run_root = tmp_path / "artifacts" / "baselines" / "campp" / "run-001"
    _write_jsonl(
        run_root / "dev_trials.jsonl",
        [
            {
                "label": 1,
                "left_id": "speaker_alpha:enroll_01",
                "left_speaker_id": "speaker_alpha",
                "right_id": "speaker_alpha:test_01",
                "right_speaker_id": "speaker_alpha",
            },
            {
                "label": 0,
                "left_id": "speaker_alpha:enroll_01",
                "left_speaker_id": "speaker_alpha",
                "right_id": "speaker_bravo:test_01",
                "right_speaker_id": "speaker_bravo",
            },
            {
                "label": 1,
                "left_id": "speaker_bravo:enroll_01",
                "left_speaker_id": "speaker_bravo",
                "right_id": "speaker_bravo:test_01",
                "right_speaker_id": "speaker_bravo",
            },
            {
                "label": 0,
                "left_id": "speaker_bravo:enroll_01",
                "left_speaker_id": "speaker_bravo",
                "right_id": "speaker_alpha:test_01",
                "right_speaker_id": "speaker_alpha",
            },
        ],
    )
    _write_jsonl(
        run_root / "dev_embedding_metadata.jsonl",
        [
            {
                "trial_item_id": "speaker_alpha:enroll_01",
                "speaker_id": "speaker_alpha",
                "role": "enrollment",
                "audio_path": "datasets/demo/alpha_enroll.wav",
                "demo_subset_path": "artifacts/demo-subset/enrollment/speaker_alpha-enroll_01.wav",
            },
            {
                "trial_item_id": "speaker_bravo:enroll_01",
                "speaker_id": "speaker_bravo",
                "role": "enrollment",
                "audio_path": "datasets/demo/bravo_enroll.wav",
                "demo_subset_path": "artifacts/demo-subset/enrollment/speaker_bravo-enroll_01.wav",
            },
            {
                "trial_item_id": "speaker_alpha:test_01",
                "speaker_id": "speaker_alpha",
                "role": "test",
                "audio_path": "datasets/demo/alpha_test.wav",
                "demo_subset_path": "artifacts/demo-subset/test/speaker_alpha-test_01.wav",
            },
            {
                "trial_item_id": "speaker_bravo:test_01",
                "speaker_id": "speaker_bravo",
                "role": "test",
                "audio_path": "datasets/demo/bravo_test.wav",
                "demo_subset_path": "artifacts/demo-subset/test/speaker_bravo-test_01.wav",
            },
        ],
    )

    config_path = tmp_path / "configs" / "release" / "onnx-parity.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    diff_tolerance_value = "0.0" if zero_tolerances else "0.02"
    metric_tolerance_value = "0.0" if zero_tolerances else "1.0"
    metadata_rows_path = "artifacts/baselines/campp/run-001/dev_embedding_metadata.jsonl"
    config_path.write_text(
        "\n".join(
            [
                'title = "Fixture ONNX parity"',
                'report_id = "fixture-onnx-parity"',
                'summary = "Fixture parity report."',
                f'project_root = "{tmp_path.as_posix()}"',
                'output_root = "artifacts/release/current"',
                "",
                "[artifacts]",
                f'model_bundle_metadata_path = "{exported.metadata_path}"',
                'trial_rows_path = "artifacts/baselines/campp/run-001/dev_trials.jsonl"',
                f'metadata_rows_path = "{metadata_rows_path}"',
                "",
                "[evaluation]",
                "seed = 7",
                "prefer_demo_subset = true",
                "max_trial_count = 4",
                "score_normalize = true",
                "promote_validated_backend = true",
                "",
                "[[variants]]",
                'id = "clean"',
                'kind = "identity"',
                'description = "Clean trials"',
                "",
                "[[variants]]",
                'id = "probe_noise"',
                'kind = "gaussian_noise"',
                'description = "Noisy probe"',
                'apply_to_roles = ["test"]',
                "snr_db = 18.0",
                "",
                "[[variants]]",
                'id = "probe_clip"',
                'kind = "clip"',
                'description = "Clipped probe"',
                'apply_to_roles = ["test"]',
                "pre_gain_db = 8.0",
                "clip_amplitude = 0.7",
                "",
                "[[variants]]",
                'id = "probe_pause"',
                'kind = "pause"',
                'description = "Paused probe"',
                'apply_to_roles = ["test"]',
                "pause_ratio = 0.18",
                "",
                "[tolerances]",
                f"max_chunk_max_abs_diff = {diff_tolerance_value}",
                f"max_pooled_max_abs_diff = {diff_tolerance_value}",
                f"max_pooled_cosine_distance = {diff_tolerance_value}",
                f"max_score_abs_diff = {diff_tolerance_value}",
                f"max_eer_delta = {metric_tolerance_value}",
                f"max_min_dcf_delta = {metric_tolerance_value}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path, metadata_path


def _write_tone(path: Path, *, frequency_hz: float, duration_seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate_hz = 16_000
    time_axis = np.arange(int(sample_rate_hz * duration_seconds), dtype=np.float32) / sample_rate_hz
    waveform = (0.25 * np.sin(2.0 * np.pi * frequency_hz * time_axis)).astype(np.float32)
    sf.write(path, waveform, sample_rate_hz, format="WAV")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
