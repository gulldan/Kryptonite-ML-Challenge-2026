from __future__ import annotations

import torch

from kryptonite.features.campp_official import even_waveform_segments, official_campp_fbank
from kryptonite.models.campp.checkpoint import (
    load_campp_state_and_config,
    remap_official_campp_state_dict,
)


def test_remap_official_campp_state_dict_names() -> None:
    state = {
        "xvector.block1.tdnnd1.nonlinear1.batchnorm.weight": torch.ones(2),
        "xvector.block1.tdnnd1.linear1.weight": torch.ones(2, 2, 1),
        "xvector.block1.tdnnd1.nonlinear2.batchnorm.bias": torch.zeros(2),
        "xvector.block1.tdnnd1.cam_layer.linear_local.weight": torch.ones(2, 2, 3),
        "xvector.block1.tdnnd1.cam_layer.linear1.bias": torch.zeros(1),
        "xvector.block1.tdnnd1.cam_layer.linear2.bias": torch.zeros(2),
    }

    remapped = remap_official_campp_state_dict(state)

    assert "xvector.block1.tdnnd1.nonlinear_in.batchnorm.weight" in remapped
    assert "xvector.block1.tdnnd1.project.weight" in remapped
    assert "xvector.block1.tdnnd1.nonlinear_bottleneck.batchnorm.bias" in remapped
    assert "xvector.block1.tdnnd1.cam.local.weight" in remapped
    assert "xvector.block1.tdnnd1.cam.context_down.bias" in remapped
    assert "xvector.block1.tdnnd1.cam.context_up.bias" in remapped


def test_load_campp_state_and_config_accepts_official_embedding_payload() -> None:
    _, state = load_campp_state_and_config(
        {
            "embedding_model": {
                "xvector.block1.tdnnd1.nonlinear1.batchnorm.weight": torch.ones(2),
            }
        }
    )

    assert set(state) == {"xvector.block1.tdnnd1.nonlinear_in.batchnorm.weight"}


def test_even_waveform_segments_repeats_short_clip() -> None:
    waveform = torch.tensor([1.0, 2.0, 3.0])

    segments = even_waveform_segments(
        waveform,
        sample_rate_hz=2,
        chunk_seconds=3.0,
        segment_count=3,
    )

    assert len(segments) == 1
    assert segments[0].tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_official_campp_fbank_applies_utterance_mean_normalization() -> None:
    waveform = torch.sin(torch.linspace(0.0, 12.0, steps=16_000))

    features = official_campp_fbank(waveform, sample_rate_hz=16_000, num_mel_bins=80)

    assert features.shape[1] == 80
    assert torch.allclose(features.mean(dim=0), torch.zeros(80), atol=1e-5)
