from pathlib import Path


def test_epic_04_closeout_doc_exists_and_covers_corruption_outputs() -> None:
    epic_closeout = Path("docs/archive/epic-04-corruption-bank-closeout.md")
    noise_doc = Path("docs/archive/audio-noise-bank.md")
    rir_doc = Path("docs/archive/audio-rir-bank.md")
    codec_doc = Path("docs/archive/audio-codec-simulation.md")
    far_field_doc = Path("docs/archive/audio-far-field-simulation.md")
    silence_doc = Path("docs/archive/audio-silence-augmentation.md")
    scheduler_doc = Path("docs/archive/audio-augmentation-scheduler.md")
    corrupted_suites_doc = Path("docs/archive/audio-corrupted-dev-suites.md")
    noise_plan = Path("configs/corruption/noise-bank.toml")
    rir_plan = Path("configs/corruption/rir-bank.toml")
    codec_plan = Path("configs/corruption/codec-bank.toml")
    far_field_plan = Path("configs/corruption/far-field-bank.toml")
    suites_plan = Path("configs/corruption/corrupted-dev-suites.toml")

    assert epic_closeout.is_file()
    assert noise_doc.is_file()
    assert rir_doc.is_file()
    assert codec_doc.is_file()
    assert far_field_doc.is_file()
    assert silence_doc.is_file()
    assert scheduler_doc.is_file()
    assert corrupted_suites_doc.is_file()
    assert noise_plan.is_file()
    assert rir_plan.is_file()
    assert codec_plan.is_file()
    assert far_field_plan.is_file()
    assert suites_plan.is_file()

    epic_text = epic_closeout.read_text(encoding="utf-8")
    scheduler_text = scheduler_doc.read_text(encoding="utf-8")
    corrupted_suites_text = corrupted_suites_doc.read_text(encoding="utf-8")

    assert "KVA-471" in epic_text
    for child_issue in (
        "KVA-505",
        "KVA-506",
        "KVA-507",
        "KVA-508",
        "KVA-509",
        "KVA-510",
        "KVA-511",
    ):
        assert child_issue in epic_text
    assert "## Deliverables" in epic_text
    assert "## Validation" in epic_text
    assert "scripts/build_noise_bank.py" in epic_text
    assert "scripts/build_rir_bank.py" in epic_text
    assert "scripts/build_codec_bank.py" in epic_text
    assert "scripts/build_far_field_bank.py" in epic_text
    assert "scripts/augmentation_scheduler_report.py" in epic_text
    assert (
        "scripts/build_corrupted_dev_suites.py --config configs/base.toml --plan "
        "configs/corruption/corrupted-dev-suites.toml" in epic_text
    )

    assert "warmup" in scheduler_text
    assert "steady" in scheduler_text
    assert "dev_snr" in corrupted_suites_text
    assert "dev_silence" in corrupted_suites_text
    assert "build_corrupted_dev_suites.py" in corrupted_suites_text


def test_epic_04_closeout_doc_is_linked_from_archive_index() -> None:
    archive_readme = Path("docs/archive/README.md").read_text(encoding="utf-8")

    assert "docs/archive/audio-corrupted-dev-suites.md" in archive_readme
    assert "docs/archive/epic-04-corruption-bank-closeout.md" in archive_readme
