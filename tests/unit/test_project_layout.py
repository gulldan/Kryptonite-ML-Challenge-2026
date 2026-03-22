from kryptonite import __version__, get_project_layout


def test_version_smoke() -> None:
    assert __version__ == "0.1.0"


def test_repository_layout_paths_exist() -> None:
    layout = get_project_layout()

    for path in (
        layout.apps,
        layout.artifacts,
        layout.assets,
        layout.configs,
        layout.deployment,
        layout.docs,
        layout.notebooks,
        layout.scripts,
        layout.src,
        layout.tests,
    ):
        assert path.exists()
