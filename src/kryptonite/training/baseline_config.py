"""Shared config primitives for manifest-backed speaker baseline recipes."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class BaselineDataConfig:
    train_manifest: str
    dev_manifest: str
    output_root: str
    trials_manifest: str | None
    checkpoint_name: str
    generate_demo_artifacts_if_missing: bool = True
    max_train_rows: int | None = None
    max_dev_rows: int | None = None

    def __post_init__(self) -> None:
        if not self.train_manifest.strip():
            raise ValueError("train_manifest must not be empty")
        if not self.dev_manifest.strip():
            raise ValueError("dev_manifest must not be empty")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty")
        if not self.checkpoint_name.strip():
            raise ValueError("checkpoint_name must not be empty")
        if self.max_train_rows is not None and self.max_train_rows <= 0:
            raise ValueError("max_train_rows must be positive when provided")
        if self.max_dev_rows is not None and self.max_dev_rows <= 0:
            raise ValueError("max_dev_rows must be positive when provided")


@dataclass(frozen=True, slots=True)
class BaselineObjectiveConfig:
    classifier_blocks: int = 0
    classifier_hidden_dim: int = 512
    scale: float = 32.0
    margin: float = 0.2
    easy_margin: bool = False

    def __post_init__(self) -> None:
        if self.classifier_blocks < 0:
            raise ValueError("classifier_blocks must be non-negative")
        if self.classifier_hidden_dim <= 0:
            raise ValueError("classifier_hidden_dim must be positive")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative")


@dataclass(frozen=True, slots=True)
class BaselineOptimizationConfig:
    learning_rate: float = 0.1
    min_learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    grad_clip_norm: float | None = 5.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.min_learning_rate < 0.0:
            raise ValueError("min_learning_rate must be non-negative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        if self.momentum < 0.0:
            raise ValueError("momentum must be non-negative")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when provided")


@dataclass(frozen=True, slots=True)
class BaselineProvenanceConfig:
    ruleset: str = "standard"
    initialization: str = "from_scratch"
    teacher_resources: tuple[str, ...] = ()
    pretrained_resources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.ruleset.strip():
            raise ValueError("ruleset must not be empty")
        if self.ruleset not in {"standard", "restricted-rules"}:
            raise ValueError("ruleset must be either 'standard' or 'restricted-rules'")
        if not self.initialization.strip():
            raise ValueError("initialization must not be empty")
        if self.initialization not in {"from_scratch", "pretrained"}:
            raise ValueError("initialization must be either 'from_scratch' or 'pretrained'")

        for field_name, values in (
            ("teacher_resources", self.teacher_resources),
            ("pretrained_resources", self.pretrained_resources),
            ("notes", self.notes),
        ):
            for value in values:
                if not value.strip():
                    raise ValueError(f"{field_name} entries must not be empty")

        if self.initialization == "from_scratch" and self.pretrained_resources:
            raise ValueError(
                "pretrained_resources must stay empty when initialization='from_scratch'"
            )

        if self.ruleset == "restricted-rules":
            if self.initialization != "from_scratch":
                raise ValueError(
                    "restricted-rules baselines must use initialization='from_scratch'"
                )
            if self.teacher_resources:
                raise ValueError("restricted-rules baselines must not declare teacher_resources")
            if self.pretrained_resources:
                raise ValueError("restricted-rules baselines must not declare pretrained_resources")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
