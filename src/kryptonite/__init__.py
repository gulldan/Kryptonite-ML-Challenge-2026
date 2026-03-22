"""Core package for the Kryptonite speaker-recognition project."""

from .config import ProjectConfig, load_project_config
from .project import ProjectLayout, get_project_layout
from .repro import build_reproducibility_snapshot, run_reproducibility_self_check, set_global_seed
from .tracking import build_tracker, create_run_id

__all__ = [
    "ProjectConfig",
    "ProjectLayout",
    "build_reproducibility_snapshot",
    "build_tracker",
    "create_run_id",
    "get_project_layout",
    "load_project_config",
    "run_reproducibility_self_check",
    "set_global_seed",
    "__version__",
]

__version__ = "0.1.0"
