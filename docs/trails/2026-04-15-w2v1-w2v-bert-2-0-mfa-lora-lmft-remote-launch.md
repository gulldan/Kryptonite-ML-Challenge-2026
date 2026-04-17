# 2026-04-15 — W2V1 w2v-BERT 2.0 MFA LoRA LMFT remote launch

- Hypothesis: a large multilingual PTM speaker stack modeled on the recent w2v-BERT 2.0
  SV recipe should be tested as a separate moonshot branch against the current CAM++ /
  ERes2Net work. The intended recipe is three staged runs on challenge-legal manifests:
  stage1 frozen-backbone LoRA + layer adapters + MFA, stage2 joint full fine-tuning after
  LoRA merge, and stage3 LMFT with longer fixed crops and larger ArcMargin.
- Paper / external provenance:
  [arXiv 2510.04213](https://arxiv.org/html/2510.04213),
  upstream code reference `ZXHY-82/w2v-BERT-2.0_SV`.
- Remote target:
  `remote` host `<redacted>`, container `container`, repository
  `<repo-root>`, detached launch on `CUDA_VISIBLE_DEVICES=1`
  (`gpu1`).
- Code / config introduced in this session on top of commit
  `e3f2553fc038d5b3d8ffda13e1cca1caed395460` with an uncommitted diff:
  `src/kryptonite/training/teacher_peft/`, `src/kryptonite/training/__init__.py`,
  `scripts/run_teacher_peft.py`, `scripts/run_teacher_peft_finetune.py`,
  `scripts/run_w2vbert2_sv_moonshot.py`,
  `configs/training/teacher-peft.toml`,
  `configs/training/w2vbert2-mfa-lora-stage1.toml`,
  `configs/training/w2vbert2-mfa-joint-ft-stage2.toml`,
  `configs/training/w2vbert2-mfa-lmft-stage3.toml`.
- Remote sync for this launch copied only the files above to `remote`; `datasets/` and the
  existing remote `artifacts/` tree were left untouched.
- Launch orchestration command executed inside `container`:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 nohup \
  uv run --group train python scripts/run_w2vbert2_sv_moonshot.py \
    --run-id W2V1_w2vbert2_mfa_lora_lmft_20260415T090107Z \
    --gpu-label gpu1 \
    --report-json artifacts/reports/w2vbert2/W2V1_w2vbert2_mfa_lora_lmft_20260415T090107Z_summary.json \
  > artifacts/logs/W2V1_w2vbert2_mfa_lora_lmft_20260415T090107Z.log 2>&1 < /dev/null &
```

- Stable launch artifacts:
  - run id: `W2V1_w2vbert2_mfa_lora_lmft_20260415T090107Z`
  - log: `artifacts/logs/W2V1_w2vbert2_mfa_lora_lmft_20260415T090107Z.log`
  - pid file: `artifacts/logs/W2V1_w2vbert2_mfa_lora_lmft_20260415T090107Z.pid`
  - latest pointer:
    `artifacts/logs/latest_W2V1_w2vbert2_mfa_lora_lmft.txt`
  - final stage summary target:
    `artifacts/reports/w2vbert2/W2V1_w2vbert2_mfa_lora_lmft_20260415T090107Z_summary.json`
- Duplicate launch note: an earlier detached run
  `W2V1_w2vbert2_mfa_lora_lmft_20260415T090049Z` was started during shell debugging and
  then terminated explicitly by killing process group `515102`; only
  `...20260415T090107Z` is the kept run.
- Data / split contract for all three stages:
  - train manifest:
    `artifacts/manifests/pseudo_ms31/ms31_filtered_mixed_train_manifest.jsonl`
  - dev manifest:
    `artifacts/manifests/participants_fixed/dev_manifest.jsonl`
  - organizer dataset root remains
    `<repo-root>/datasets/Для участников`
- Shared runtime / preprocessing:
  - base config: `configs/base.toml`
  - seed `42`, `pythonhashseed=42`, `runtime.device="auto"`, overridden remote GPU via
    `CUDA_VISIBLE_DEVICES=1`
  - `runtime.num_workers=8`
  - normalization target sample rate `16 kHz`
  - VAD policy `none`
  - precision `bf16`
  - gradient checkpointing enabled
  - train short-utterance policy inherited from base config: `repeat_pad`
- Stage design:
  - stage1 config `configs/training/w2vbert2-mfa-lora-stage1.toml`:
    `facebook/w2v-bert-2.0`, frozen feature encoder, LoRA target modules
    `linear_q` / `linear_v`, rank `64`, alpha `128`, dropout `0.0`,
    MFA over all hidden states (`mfa_num_layers=-1`), layer adapters enabled with
    `adapter_dim=128`, ASP pooling, embedding dim `256`, classifier hidden dim `512`,
    ArcMargin scale `32`, margin `0.2`, AdamW + cosine, lr `1e-4`, min lr `1e-5`,
    weight decay `1e-4`, grad accumulation `4`, grad clip `1.0`, warmup `1` epoch,
    batch `8`, eval batch `2`, epochs `4`, fixed 3 s train crops, 6 s eval chunks with
    1 s overlap, output root `artifacts/baselines/w2vbert2-mfa-lora-stage1`.
  - stage2 config `configs/training/w2vbert2-mfa-joint-ft-stage2.toml`:
    initializes from stage1 checkpoint, merges LoRA into the backbone, restores the
    classifier head, unfreezes the encoder, keeps MFA/adapters/ASP, AdamW + cosine,
    lr `1e-5`, min lr `5e-6`, grad accumulation `8`, batch `4`, eval batch `2`,
    epochs `2`, margin `0.2`, output root
    `artifacts/baselines/w2vbert2-mfa-joint-ft-stage2`.
  - stage3 config `configs/training/w2vbert2-mfa-lmft-stage3.toml`:
    initializes from the stage2 checkpoint with classifier restored, keeps the encoder
    trainable, switches to fixed 6 s train crops, raises ArcMargin to `0.4`, AdamW +
    cosine, lr `1e-5`, min lr `5e-6`, grad accumulation `8`, batch `2`, eval batch `2`,
    epochs `1`, output root `artifacts/baselines/w2vbert2-mfa-lmft-stage3`.
- Runtime evidence immediately after launch:
  - log contains the stage1 start banner and the delegated command
    `scripts/run_teacher_peft.py --config configs/training/w2vbert2-mfa-lora-stage1.toml --output json`
  - active kept process tree inside `container`:
    `515286 -> 515289 -> 515290`
  - `nvidia-smi` showed process `515290` on GPU UUID
    `GPU-160f1116-6087-0bb4-5c98-1fc2c166964c` (host GPU index `1`) with
    `3748 MiB` allocated and non-zero utilization, confirming the run is on `gpu1`
  - host GPU index `0` remained occupied by an older unrelated ERes2Net run
    (`pid 502255`), so the moonshot launch did not evict the pre-existing workload.
- Checkpoint / metrics status at this write:
  - stage1/2/3 checkpoint directories are not available yet because the run is still in
    stage1 initialization / early training
  - expected checkpoint roots are the three stage output roots above, each with the
    `teacher_peft` checkpoint name
  - public submission is not started yet
  - public leaderboard score pending
- Follow-up evaluation / submission path:
  - the stage runner writes each stage `training_summary.json`, stage-local dev
    `score_summary`, and the final chained report json once stage3 finishes
  - no public inference or submission command has been launched yet from this branch;
    that decision waits for the first completed stage metrics and checkpoint quality.
- Decision: keep `W2V1_w2vbert2_mfa_lora_lmft_20260415T090107Z` running on `gpu1` as the
  heavy-integration moonshot branch. Revisit once stage1 emits its first durable
  training artifacts; if startup or throughput is unacceptable, reduce duplication and
  inspect worker / batch pressure before touching the model recipe itself.
