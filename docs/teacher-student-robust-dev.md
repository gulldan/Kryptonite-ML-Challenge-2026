# Teacher Vs Student Robust Dev

`KVA-532` adds one reproducible report that compares the stretch teacher branch
against student candidates on the same frozen corrupted-dev suites.

## Goal

The repository already had:

- clean-dev verification reports for individual runs;
- a deterministic corrupted-dev suite builder;
- one runnable `teacher-peft` training lane;
- one robust-dev shortlist workflow for `CAM++` only.

What was missing was one family-agnostic comparison step that can take finished
teacher and student runs, re-score them on the same frozen robust-dev suites,
and render one auditable report with quality and cost deltas.

## What The Workflow Does

The checked-in path lives in:

- `configs/eval/teacher-student-robust-dev.example.toml`
- `scripts/build_teacher_student_robust_dev_report.py`
- `src/kryptonite/eval/teacher_student_robust_dev*.py`

The workflow:

1. loads the clean-dev verification report and training summary from each
   candidate run;
2. loads the frozen corrupted-dev suite catalog built by
   `scripts/build_corrupted_dev_suites.py`;
3. re-exports embeddings on every selected corrupted suite using the candidate's
   native checkpoint loader;
4. rebuilds trials, cohort bank, cosine scores, and verification reports for
   every candidate/suite pair;
5. ranks all candidates with one weighted objective over clean + robust metrics
   and renders teacher-vs-student pairwise deltas.

Supported candidate families are:

- `campp`
- `eres2netv2`
- `teacher_peft`

## Command

Build the corrupted suites first if the catalog does not exist yet:

```bash
uv run python scripts/build_corrupted_dev_suites.py \
  --config configs/base.toml \
  --plan configs/corruption/corrupted-dev-suites.toml
```

Then execute the comparison:

```bash
uv sync --dev --group train
uv run python scripts/build_teacher_student_robust_dev_report.py \
  --config configs/eval/teacher-student-robust-dev.example.toml
```

If you run the command outside the repository root, also pass:

```bash
--project-root /path/to/Kryptonite-ML-Challenge-2026
```

## Output Layout

The top-level report lands under the configured `output_root`, for example
`artifacts/eval/teacher-student-robust-dev/`:

- `teacher_student_robust_dev_report.json`
- `teacher_student_robust_dev_report.md`

Per-candidate robust-dev artifacts land under:

- `artifacts/eval/teacher-student-robust-dev/candidates/<candidate-id>/robust_dev/<suite-id>/`

Each suite directory contains the same baseline-compatible evaluation payload:

- exported embeddings and metadata;
- merged or generated trials;
- cohort-bank artifacts;
- score summary JSON;
- verification report JSON and Markdown.

## Notes

- This workflow intentionally re-scores robust suites from checkpoints instead
  of trusting previously cached summary JSON. That keeps the teacher/student
  comparison honest and slice-consistent.
- `teacher_peft` evaluation still depends on the original Hugging Face backbone
  id stored in `checkpoint_metadata.json`; a populated `HUGGINGFACE_HUB_TOKEN`
  may still be required on machines without the backbone cached locally.
- The checked-in config is an example template because the actual candidate
  `run_root` values depend on whichever teacher and student runs you want to
  compare.
