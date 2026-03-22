"""Core package for the Kryptonite speaker-recognition project."""

from .config import ProjectConfig, load_project_config
from .project import ProjectLayout, get_project_layout

__all__ = [
    "ProjectConfig",
    "ProjectLayout",
    "get_project_layout",
    "load_project_config",
    "__version__",
]

__version__ = "0.1.0"
