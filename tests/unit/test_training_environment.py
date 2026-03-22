from __future__ import annotations

from types import SimpleNamespace

import kryptonite.training.environment as training_environment


def test_build_training_environment_report_marks_missing_required_packages(
    monkeypatch,
) -> None:
    def fake_load_module(module_name: str) -> object:
        if module_name == "torch":
            return SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                version=SimpleNamespace(cuda=None),
                backends=SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: False),
                ),
            )
        raise ImportError(f"{module_name} is not installed")

    monkeypatch.setattr(training_environment, "_load_module", fake_load_module)
    monkeypatch.setattr(
        training_environment,
        "_distribution_version",
        lambda distribution: "2.10.0" if distribution == "torch" else None,
    )

    report = training_environment.build_training_environment_report()

    assert report.passed is False
    assert "onnx" in report.missing_required
    assert "mlflow" in report.missing_required


def test_build_training_environment_report_includes_gpu_probe_when_requested(
    monkeypatch,
) -> None:
    def fake_load_module(module_name: str) -> object:
        if module_name == "torch":
            return SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True),
                version=SimpleNamespace(cuda="12.8"),
                backends=SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: False),
                ),
            )
        if module_name == "onnxruntime":
            return SimpleNamespace(
                get_available_providers=lambda: ["CPUExecutionProvider"],
            )
        if module_name == "tensorrt":
            return SimpleNamespace(Logger=object())
        return SimpleNamespace()

    monkeypatch.setattr(training_environment, "_load_module", fake_load_module)
    monkeypatch.setattr(training_environment, "_distribution_version", lambda _: "1.0.0")

    report = training_environment.build_training_environment_report(require_gpu=True)

    assert report.passed is True
    assert any(probe.distribution == "tensorrt-cu12" for probe in report.packages)
    assert (
        "Missing required packages"
        not in training_environment.render_training_environment_report(report)
    )
