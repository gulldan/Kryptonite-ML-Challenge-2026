# FFSVC 2022 Surrogate

## Why This Dataset

For a pre-Dataton pipeline shakeout we want a server-only surrogate that is close to the target task:

- speaker verification
- far-field or device-mismatched audio
- single-channel trials
- official trial files and metadata

`FFSVC 2022` is the closest official benchmark currently available from primary sources. Its dataset page explicitly says:

- the challenge builds on far-field speaker verification;
- the dev set has the same data distribution as the evaluation data;
- the dev/eval artifacts are directly downloadable from the official page;
- the older full `FFSVC2020` main training data now requires manual request, while the public page still exposes only the directly downloadable subset today.

Sources:

- [FFSVC 2022 dataset page](https://ffsvc.github.io/dataset/)
- [FFSVC 2022 site](https://ffsvc.github.io/)

## Important Caveat

This bundle is a surrogate for engineering bring-up, not a challenge-legal training recipe.

The official FFSVC 2022 dataset page says the dev set is for tuning and testing and must not be used for training in the original competition. We are intentionally relaxing that for internal pipeline smoke only, because the Dataton raw data is not yet public and we need to validate:

- ingestion
- metadata parsing
- manifest generation
- speaker-disjoint split logic
- trial generation and scoring
- end-to-end training/evaluation plumbing

Once the real Dataton data is available, the code should swap datasets without changing the pipeline shape.

## What The Plan Downloads

The server-only acquisition plan at `configs/data-acquisition/ffsvc2022-surrogate.toml` downloads:

- FFSVC 2022 dev trials
- FFSVC 2022 dev metadata
- FFSVC 2022 eval trials
- FFSVC 2022 dev WAV archive from Zenodo

It does not automate:

- full `FFSVC2020` train/dev/eval, because the official page now requires email request
- `VoxCeleb`, because the official path is still manual/gated

## Run It

Run this on `gpu-server` from `/mnt/storage/Kryptonite-ML-Challenge-2026`:

```bash
uv run python scripts/acquire_ffsvc2022_surrogate.py --execute
```

Inspect-only mode:

```bash
uv run python scripts/acquire_ffsvc2022_surrogate.py
```

The dataset root is:

```text
/mnt/storage/Kryptonite-ML-Challenge-2026/datasets/ffsvc2022-surrogate
```

## Recommended Next Steps

After acquisition:

1. build a parser for `dev_meta_list.txt`
2. generate unified manifests under `artifacts/manifests/ffsvc2022-surrogate/`
3. derive speaker-disjoint train/dev splits from the official dev metadata
4. reuse official trial files where possible for verification scoring
5. treat this bundle as the engineering stand-in until the Dataton data lands
