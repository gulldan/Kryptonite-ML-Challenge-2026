"""Core package for the Kryptonite speaker-recognition project."""

from .config import ProjectConfig, load_project_config
from .repro import build_reproducibility_snapshot, run_reproducibility_self_check, set_global_seed
from .tracking import build_tracker, create_run_id

__all__ = [
    "ProjectConfig",
    "build_reproducibility_snapshot",
    "build_tracker",
    "create_run_id",
    "load_project_config",
    "run_reproducibility_self_check",
    "set_global_seed",
    "__version__",
]

__version__ = "0.1.0"
