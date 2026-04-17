from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

BASELINE_ROOT = Path(__file__).resolve().parents[2] / "baseline"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_module_attr(module: types.ModuleType, name: str, value: object) -> None:
    module.__dict__[name] = value


@pytest.fixture
def pandas_module():
    return pytest.importorskip("pandas")


def _install_dataset_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.ModuleType("torch")

    class Tensor:
        pass

    _set_module_attr(fake_torch, "Tensor", Tensor)
    _set_module_attr(fake_torch, "tensor", lambda value: value)
    fake_torchaudio = types.ModuleType("torchaudio")
    _set_module_attr(
        fake_torchaudio,
        "load",
        lambda _path: (_ for _ in ()).throw(
            RuntimeError("torchaudio.load is not used by these tests")
        ),
    )
    _set_module_attr(fake_torchaudio, "transforms", types.SimpleNamespace(Resample=object))
    fake_pandas = types.ModuleType("pandas")
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils_data = types.ModuleType("torch.utils.data")
    _set_module_attr(fake_torch_utils_data, "Dataset", object)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_torch_utils_data)


def _install_train_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.ModuleType("torch")
    _set_module_attr(fake_torch, "Tensor", object)
    _set_module_attr(fake_torch, "cat", lambda *_args, **_kwargs: None)
    _set_module_attr(fake_torch, "manual_seed", lambda _seed: None)
    _set_module_attr(fake_torch, "no_grad", lambda: lambda func: func)
    _set_module_attr(
        fake_torch,
        "cuda",
        types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda _seed: None),
    )
    _set_module_attr(
        fake_torch,
        "backends",
        types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False, deterministic=False)),
    )
    _set_module_attr(
        fake_torch,
        "Generator",
        lambda: types.SimpleNamespace(manual_seed=lambda _seed: None),
    )
    _set_module_attr(fake_torch, "optim", types.SimpleNamespace(Optimizer=object, AdamW=object))
    _set_module_attr(
        fake_torch,
        "nn",
        types.SimpleNamespace(Module=object, CrossEntropyLoss=lambda: None),
    )

    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils_data = types.ModuleType("torch.utils.data")
    _set_module_attr(fake_torch_utils_data, "DataLoader", object)
    fake_src = types.ModuleType("src")
    fake_metrics = types.ModuleType("src.metrics")
    _set_module_attr(fake_metrics, "precision_at_k", lambda *_args, **_kwargs: {})
    fake_dataset = types.ModuleType("src.dataset")
    _set_module_attr(fake_dataset, "SpeakerDataset", object)
    fake_model = types.ModuleType("src.model")
    _set_module_attr(fake_model, "ECAPASpeakerId", object)
    fake_tqdm = types.ModuleType("tqdm")
    _set_module_attr(fake_tqdm, "tqdm", lambda iterable, *_args, **_kwargs: iterable)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_torch_utils_data)
    monkeypatch.setitem(sys.modules, "src", fake_src)
    monkeypatch.setitem(sys.modules, "src.metrics", fake_metrics)
    monkeypatch.setitem(sys.modules, "src.dataset", fake_dataset)
    monkeypatch.setitem(sys.modules, "src.model", fake_model)
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)


def _install_calc_metric_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_src = types.ModuleType("src")
    fake_metrics = types.ModuleType("src.metrics")
    _set_module_attr(
        fake_metrics,
        "precision_at_k_from_indices",
        lambda *_args, **_kwargs: {"precision@10": 1.0},
    )
    monkeypatch.setitem(sys.modules, "src", fake_src)
    monkeypatch.setitem(sys.modules, "src.metrics", fake_metrics)


def test_eval_chunking_uses_center_and_evenly_spaced_crops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dataset_stubs(monkeypatch)
    dataset = _load_module(BASELINE_ROOT / "src" / "dataset.py", "organizer_dataset_test")

    waveform = np.arange(10, dtype=np.float32)

    assert dataset.get_chunk(waveform, 4, random_chunk=False).tolist() == [3.0, 4.0, 5.0, 6.0]
    assert dataset.get_eval_chunks(waveform, 4, num_chunks=3).tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0, 6.0],
        [6.0, 7.0, 8.0, 9.0],
    ]


def test_split_data_is_speaker_disjoint_and_keeps_small_speakers_in_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pandas_module,
) -> None:
    pd = pandas_module
    _install_train_stubs(monkeypatch)
    train_module = _load_module(BASELINE_ROOT / "train.py", "organizer_train_test")

    rows = []
    for speaker in ["spk_a", "spk_b", "spk_c", "spk_d", "spk_e"]:
        for index in range(11):
            rows.append({"speaker_id": speaker, "filepath": f"train/{speaker}/{index}.flac"})
    for index in range(3):
        rows.append({"speaker_id": "tiny", "filepath": f"train/tiny/{index}.flac"})
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    train_csv, val_csv = train_module.split_data(
        str(csv_path),
        str(tmp_path),
        train_ratio=0.6,
        seed=2026,
        min_val_utts=11,
    )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    assert set(train_df["speaker_id"]).isdisjoint(set(val_df["speaker_id"]))
    assert "tiny" in set(train_df["speaker_id"])
    assert "tiny" not in set(val_df["speaker_id"])
    assert val_df.groupby("speaker_id").size().min() >= 11


def test_train_metric_improved_respects_mode_and_min_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_train_stubs(monkeypatch)
    train_module = _load_module(BASELINE_ROOT / "train.py", "organizer_train_metric_test")

    assert train_module.metric_improved(0.5, None, mode="max", min_delta=0.01) is True
    assert train_module.metric_improved(0.52, 0.5, mode="max", min_delta=0.01) is True
    assert train_module.metric_improved(0.505, 0.5, mode="max", min_delta=0.01) is False
    assert train_module.metric_improved(0.48, 0.5, mode="min", min_delta=0.01) is True
    assert train_module.metric_improved(0.495, 0.5, mode="min", min_delta=0.01) is False


def test_calc_metrics_validates_template_order_and_index_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pandas_module,
) -> None:
    pd = pandas_module
    _install_calc_metric_stubs(monkeypatch)
    calc_metrics = _load_module(BASELINE_ROOT / "calc_metrics.py", "organizer_calc_metrics_test")

    template = tmp_path / "template.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"filepath": ["a.flac", "b.flac", "c.flac"]}).to_csv(template, index=False)
    pd.DataFrame(
        {
            "filepath": ["a.flac", "b.flac", "c.flac"],
            "neighbours": ["1,2", "0,2", "0,1"],
        }
    ).to_csv(submission, index=False)

    calc_metrics.validate_filepath_order(str(submission), str(template))
    indices = calc_metrics.load_indices(str(submission), expected_columns=2)
    assert indices.tolist() == [[1, 2], [0, 2], [0, 1]]

    bad_order = tmp_path / "bad_order.csv"
    pd.DataFrame(
        {
            "filepath": ["b.flac", "a.flac", "c.flac"],
            "neighbours": ["1,2", "0,2", "0,1"],
        }
    ).to_csv(bad_order, index=False)
    with pytest.raises(ValueError, match="filepath order"):
        calc_metrics.validate_filepath_order(str(bad_order), str(template))

    with pytest.raises(ValueError, match="out-of-range"):
        calc_metrics.validate_indices(np.asarray([[1, 3], [0, 2], [0, 1]], dtype=np.int64))
