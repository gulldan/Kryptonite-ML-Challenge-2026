"""Repository layout helpers shared by scripts and app entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    """Resolved paths for the tracked repository layout."""

    root: Path
    apps: Path
    artifacts: Path
    assets: Path
    configs: Path
    deployment: Path
    docs: Path
    notebooks: Path
    scripts: Path
    src: Path
    tests: Path


def get_project_layout() -> ProjectLayout:
    """Return the repository layout anchored at the project root."""

    root = Path(__file__).resolve().parents[2]
    return ProjectLayout(
        root=root,
        apps=root / "apps",
        artifacts=root / "artifacts",
        assets=root / "assets",
        configs=root / "configs",
        deployment=root / "deployment",
        docs=root / "docs",
        notebooks=root / "notebooks",
        scripts=root / "scripts",
        src=root / "src",
        tests=root / "tests",
    )
