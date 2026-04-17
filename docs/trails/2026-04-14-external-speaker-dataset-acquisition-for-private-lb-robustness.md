# 2026-04-14 — External Speaker Dataset Acquisition For Private-LB Robustness

Purpose:

- User clarified that the challenge rules disallow only VoxBlink2, while other external
  datasets are allowed. Because private LB is expected to be about `3x` larger and more
  diverse than public, prepare external far-field/speaker data for a controlled adaptation
  run on top of the current official CAM++ branch.
- Do not download or use VoxBlink2 audio, labels, embeddings, or checkpoints.
- User later clarified that VoxBlink v1 is allowed. Treat VoxBlink v1 as an allowed
  external annotation/audio-reconstruction source, but keep VoxBlink2 fully excluded.

Repository change:

- Added `scripts/download_external_speaker_datasets.py`.
- The downloader covers open CN-Celeb v2, open FFSVC2022 trial/meta files, and restricted
  FFSVC Zenodo records when `ZENODO_ACCESS_TOKEN` is available.
- Restricted FFSVC audio archives are not silently mirrored: without a token the script writes
  access notes under `datasets/ffsvc*/.../REQUEST_ACCESS.md` and reports `skipped`.
- Added `scripts/build_cnceleb_manifests.py` for the next step after extraction: scan
  `datasets/CN-Celeb_flac`, create external train/dev manifests, and optionally append the
  external train split to the participant train manifest for mixed adaptation runs. A partial
  tar header check showed CN-Celeb v2 paths like
  `CN-Celeb_flac/eval/test/id00939-speech-01-079.flac`, so the manifest builder defaults to
  speaker ids from the filename prefix before the first `-`.
- Added `scripts/build_voxblink_v1_audio.py` for official VoxBlink v1 only. The script
  downloads the official v1 annotation bundle, clones the official
  `VoxBlink/ScriptsForVoxBlink` repository for provenance, extracts timestamps/tags, builds
  16 kHz mono WAV utterances from YouTube timestamps with `yt-dlp` plus FFmpeg, and writes
  VoxBlink v1 manifests/mixed manifests. VoxBlink2 support is intentionally absent.
- Added `configs/training/campp-ms31-cnceleb-mixed-lowlr.toml` as the planned GPU1 external
  adaptation config once CN-Celeb is fully downloaded, verified, extracted, and converted to
  manifests. It starts from the MS31 checkpoint, uses the official CAM++ frontend, keeps
  participant dev as the guardrail, and trains for `2` low-LR epochs on participant train plus
  CN-Celeb external train.

Planned remote acquisition:

- Run id: `EXTDATA_external_speaker_download_20260414T0630Z`.
- Execution environment: `<redacted>`.
- Command target: download all known non-VoxBlink2 external speaker datasets supported by the
  script. Open downloads should materialize immediately; restricted FFSVC audio archives need a
  Zenodo token.
- Remote log:
  `artifacts/logs/EXTDATA_external_speaker_download_20260414T0630Z.log`.
- PID file:
  `artifacts/logs/EXTDATA_external_speaker_download_20260414T0630Z.pid`.
- Launch status: started detached on remote at 2026-04-14T06:30Z, PID `479335`.
- Initial progress: FFSVC2022 `trials_dev_keys`, `dev_meta_list`, and `trials_eval` downloaded
  with MD5 verification on remote. CN-Celeb v2 `21G` archive download started from OpenSLR.
  Restricted FFSVC audio Zenodo records require `ZENODO_ACCESS_TOKEN`; without it they will be
  reported as skipped and access notes will be written under `datasets/ffsvc*/...`.
- CN-Celeb primary OpenSLR download timed out early. The downloader was patched with mirror
  fallback and the job was resumed as PID `479467`, log
  `artifacts/logs/EXTDATA_external_speaker_download_20260414T0630Z_resume1.log`, reusing the
  partial archive through `wget -c`.
- The first resume exposed an incomplete-archive edge case: a partial `cn-celeb_v2.tar.gz`
  was treated as extractable and failed with `EOFError`. The downloader was patched to verify
  MD5 before extraction, resume corrupt/incomplete archives instead of unpacking them, and keep
  mirror fallback. The stale `wget` process was stopped, the failed partial extraction
  directory `datasets/CN-Celeb_flac` was removed, and acquisition was relaunched detached as
  PID `479589`, log
  `artifacts/logs/EXTDATA_external_speaker_download_20260414T0630Z_resume2.log`.
- Early `resume2` throughput from the EU OpenSLR mirrors was too low for a 21 GB archive. A
  short range probe showed the China OpenSLR mirror responding better from `remote`, so the
  CN-Celeb URL order was changed to prefer `http://openslr.magicdatatech.com/...` first.
  The job was restarted again against the same partial archive as PID `479830`, log
  `artifacts/logs/EXTDATA_external_speaker_download_20260414T0630Z_resume3.log`.
- A separate restricted-FFSVC check was run immediately to record the access state without
  waiting for CN-Celeb to finish. Report:
  `artifacts/reports/external-speaker-datasets/EXTDATA_ffsvc_restricted_access_notes_20260414T0650Z.json`.
  Result: all restricted FFSVC audio records were `skipped` because `ZENODO_ACCESS_TOKEN` is
  not set; `REQUEST_ACCESS.md` notes were written under the corresponding `datasets/ffsvc*`
  directories.
- `resume3` made progress to `3.8G` of the `21G` CN-Celeb archive, but then stopped after one
  full mirror pass because every OpenSLR mirror eventually returned timeout exit code `4`.
  The downloader was patched with `--download-passes` and `--retry-sleep-seconds` so long
  unstable downloads can continue through repeated mirror cycles until final MD5 verification.
  CN-Celeb was relaunched detached from the same partial archive as
  `EXTDATA_cnceleb_resume_until_md5_20260414T1527Z`, PID `481721`, log
  `artifacts/logs/EXTDATA_cnceleb_resume_until_md5_20260414T1527Z.log`, report
  `artifacts/reports/external-speaker-datasets/EXTDATA_cnceleb_resume_until_md5_20260414T1527Z_status.json`.
- Alternative-source probes were also run from `remote`. `zzj-pro/CN_Celeb_v2` on Hugging Face
  Hub was public but contained only `.gitattributes` (`used_storage=0`), so it is not usable as
  a mirror. Historical `cslt.riit.tsinghua.edu.cn` and guessed `cnceleb.org` URLs were not
  resolvable or returned `404`. Keep the OpenSLR partial resume as the active acquisition path.
- Prepared downstream GPU1 run as a detached watcher:
  `MS33_campp_ms31_cnceleb_mixed_lowlr_public_c4_20260414T1545Z`, PID `482582`, log
  `artifacts/logs/MS33_campp_ms31_cnceleb_mixed_lowlr_public_c4_20260414T1545Z.log`, latest
  pointer `artifacts/logs/latest_MS33_campp_ms31_cnceleb_mixed_lowlr_public_c4.txt`. It waits
  for `datasets/CN-Celeb_flac`, builds `cnceleb_v2_ms31_mixed` manifests with filename-prefix
  speaker ids, guards on at least `100` speakers and `10000` train rows, then runs
  `configs/training/campp-ms31-cnceleb-mixed-lowlr.toml` from the MS31 checkpoint on
  `CUDA_VISIBLE_DEVICES=1`, followed by the standard official CAM++ public C4 tail into
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms33_cnceleb_mixed_20260414T1545Z/`.
- VoxBlink v1 official resource acquisition: the official Google Drive annotation bundle was
  downloaded on remote under `datasets/voxblink_v1/resource`. Present resources include
  `data/utt_clean.txt`, `data/utt_full.txt`, `meta/{spk2gender,spk2lan,spk2loc,utt2dur}`,
  `video_list/spk2videos_{clean,full,test}`, `timestamp.tar.gz`, and `video_tags.tar.gz`.
  `timestamp.tar.gz` and `video_tags.tar.gz` were extracted in place.
- VoxBlink v1 smoke run: `EXTDATA_voxblink_v1_test_audio_smoke2_20260414T1616Z`, PID
  `484314`, log `artifacts/logs/EXTDATA_voxblink_v1_test_audio_smoke2_20260414T1616Z.log`,
  latest pointer `artifacts/logs/latest_EXTDATA_voxblink_v1_test_audio_smoke.txt`. Command:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run python scripts/build_voxblink_v1_audio.py \
  --mode test \
  --step all \
  --workers 4 \
  --max-speakers 8 \
  --max-videos-per-speaker 2 \
  --video-download-timeout-seconds 75 \
  --base-train-manifest artifacts/manifests/participants_fixed/train_manifest.jsonl
```

  The goal is not training yet; it is an access/throughput smoke test for YouTube downloads
  from the remote container and a guard against orphaned `yt-dlp` processes. If it writes WAVs,
  proceed to a larger clean-mode VoxBlink v1 acquisition; if all selected YouTube videos fail
  or time out, keep only the official annotation resource and do not spend GPU time on a
  VoxBlink v1 adaptation branch until a working audio acquisition route is found.
- VoxBlink v1 test-mode smoke result: completed with `0` WAVs. Summary line:
  `{"download_failed": 2, "speaker_id": "id42491"}`. Manifest generation succeeded but
  produced `row_count=0`, `speaker_count=0`. No live `yt-dlp` orphan remained after the run;
  the process-group timeout cleanup is working for the new script path. Because
  `spk2videos_test` effectively contains only one speaker, this is not enough to reject
  VoxBlink v1 audio acquisition.
- Follow-up clean-mode smoke run:
  `EXTDATA_voxblink_v1_clean_audio_smoke_20260414T1620Z`, PID `486611`, log
  `artifacts/logs/EXTDATA_voxblink_v1_clean_audio_smoke_20260414T1620Z.log`, latest pointer
  `artifacts/logs/latest_EXTDATA_voxblink_v1_clean_audio_smoke.txt`. It checks `16` clean
  speakers with one video each, `4` workers, and `75s` per-video timeout. Decision pending:
  if this writes any WAVs, launch a larger VoxBlink v1 clean acquisition; if it also writes
  `0`, treat YouTube materialization from remote as blocked/low-yield until a better route is
  available.
- VoxBlink v1 clean-mode smoke result: completed with `0` WAVs. All `16/16` selected clean
  speakers returned `download_failed=1`; manifest generation produced `row_count=0`,
  `speaker_count=0`. No live `yt-dlp` orphan remained.
- Manual YouTube diagnostic from inside `remote` container: `uvx yt-dlp -F` timed out during
  TLS handshake both for a VoxBlink v1 video id (`FkP2yE--B-Y`) and for a known public YouTube
  control id (`dQw4w9WgXcQ`), eventually killed by a `70s` wrapper timeout. Conclusion:
  direct YouTube materialization from the current remote/container network path is blocked or
  too unreliable. Decision: do not launch a full VoxBlink v1 audio crawl from remote now. Keep
  the official VoxBlink v1 annotation resource and repo script for a future route with working
  YouTube access, cookies/proxy, or a permitted pre-materialized mirror.
