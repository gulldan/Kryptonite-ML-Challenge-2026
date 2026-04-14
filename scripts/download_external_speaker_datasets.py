"""Download external speaker datasets used by challenge adaptation experiments.

This script intentionally excludes VoxBlink2. It downloads open resources directly and
supports restricted FFSVC Zenodo records when a valid Zenodo access token is provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASETS_ROOT = Path("datasets")


@dataclass(frozen=True, slots=True)
class DirectDownload:
    key: str
    description: str
    url: str
    relative_path: str
    mirrors: tuple[str, ...] = ()
    md5: str | None = None
    extract_to: str | None = None


@dataclass(frozen=True, slots=True)
class ZenodoRecord:
    key: str
    description: str
    record_id: str
    target_dir: str
    md5: str | None = None


DIRECT_DOWNLOADS: tuple[DirectDownload, ...] = (
    DirectDownload(
        key="ffsvc2022-dev-trials",
        description="FFSVC2022 development trials with keys.",
        url="https://ffsvc.github.io/assets/ffsvc2022/trials_dev_keys",
        relative_path="ffsvc2022/trials_dev_keys",
        md5="de0a22b157c3f5173a58b2bbc2ee0cab",
    ),
    DirectDownload(
        key="ffsvc2022-dev-meta",
        description="FFSVC2022 development metadata list.",
        url="https://ffsvc.github.io/assets/ffsvc2022/dev_meta_list",
        relative_path="ffsvc2022/dev_meta_list",
        md5="8ad1bdc99529e85be5ed453873a670e2",
    ),
    DirectDownload(
        key="ffsvc2022-eval-trials",
        description="FFSVC2022 evaluation trials without keys.",
        url="https://ffsvc.github.io/assets/ffsvc2022/trials_eval",
        relative_path="ffsvc2022/trials_eval",
        md5="4a2af9dc8132ddd8826439cc3c1e440f",
    ),
    DirectDownload(
        key="cn-celeb",
        description="CN-Celeb v2, OpenSLR 82 external speaker corpus.",
        url="http://openslr.magicdatatech.com/resources/82/cn-celeb_v2.tar.gz",
        relative_path="cn-celeb_v2.tar.gz",
        mirrors=(
            "https://openslr.magicdatatech.com/resources/82/cn-celeb_v2.tar.gz",
            "https://openslr.elda.org/resources/82/cn-celeb_v2.tar.gz",
            "https://openslr.trmal.net/resources/82/cn-celeb_v2.tar.gz",
            "https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz",
        ),
        md5="7ab1b214028a7439e26608b2d5a0336c",
        extract_to="CN-Celeb_flac",
    ),
)


ZENODO_RECORDS: tuple[ZenodoRecord, ...] = (
    ZenodoRecord(
        key="ffsvc2020-supplement-f",
        description="Restricted FFSVC2020 supplementary far-field set, first visit.",
        record_id="6465819",
        target_dir="ffsvc2020_supplement/F",
        md5="768b4544d756dd76f92a9a75cafd3f6f",
    ),
    ZenodoRecord(
        key="ffsvc2020-supplement-s",
        description="Restricted FFSVC2020 supplementary far-field set, second visit.",
        record_id="6462609",
        target_dir="ffsvc2020_supplement/S",
        md5="a3e30342702b3e567ba71b8d9cb8e06e",
    ),
    ZenodoRecord(
        key="ffsvc2020-supplement-t",
        description="Restricted FFSVC2020 supplementary far-field set, third visit.",
        record_id="6465833",
        target_dir="ffsvc2020_supplement/T",
        md5="9e671e0155f7b1e5a161a33f703ca246",
    ),
    ZenodoRecord(
        key="ffsvc2022-dev-wav",
        description="Restricted FFSVC2022 development WAV files.",
        record_id="6466068",
        target_dir="ffsvc2022/dev_wav",
        md5="e7fb015c5c6389da21d8966ce549e59f",
    ),
    ZenodoRecord(
        key="ffsvc2022-eval-wav",
        description="Restricted FFSVC2022 evaluation WAV files.",
        record_id="6461072",
        target_dir="ffsvc2022/eval_wav",
        md5="e3cf9c525a64f53b9fe960cc1705b6f5",
    ),
)


def main() -> None:
    args = _parse_args()
    selected = _resolve_selection(args.dataset)
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    if args.list:
        _print_inventory()
        return

    status: list[dict[str, Any]] = []
    for item in DIRECT_DOWNLOADS:
        if item.key not in selected:
            continue
        status.append(
            _download_direct(
                item,
                force=args.force,
                extract=args.extract,
                download_passes=args.download_passes,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
        )

    token = os.environ.get(args.zenodo_token_env, "").strip()
    for item in ZENODO_RECORDS:
        if item.key not in selected:
            continue
        status.append(
            _download_zenodo_record(
                item,
                token=token,
                force=args.force,
                extract=args.extract,
            )
        )

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    failures = [row for row in status if row["status"] not in {"downloaded", "exists"}]
    if failures and args.fail_on_skip:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset key to download. May be repeated. Use 'all', 'open', "
            "'ffsvc', or 'restricted-ffsvc'."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing archives.")
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract supported archives after download.",
    )
    parser.add_argument(
        "--zenodo-token-env",
        default="ZENODO_ACCESS_TOKEN",
        help="Environment variable containing a Zenodo token for restricted records.",
    )
    parser.add_argument(
        "--report-json",
        default="artifacts/reports/external-speaker-datasets/download_status.json",
    )
    parser.add_argument(
        "--download-passes",
        type=int,
        default=1,
        help="Number of full URL/mirror passes for direct downloads before failing.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between direct download passes after all mirrors fail.",
    )
    parser.add_argument("--fail-on-skip", action="store_true")
    parser.add_argument("--list", action="store_true", help="List known dataset keys and exit.")
    return parser.parse_args()


def _resolve_selection(values: list[str]) -> set[str]:
    if not values:
        values = ["open"]
    direct_keys = {item.key for item in DIRECT_DOWNLOADS}
    zenodo_keys = {item.key for item in ZENODO_RECORDS}
    selected: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if normalized == "all":
            selected.update(direct_keys | zenodo_keys)
        elif normalized == "open":
            selected.update(direct_keys)
        elif normalized == "ffsvc":
            selected.update(key for key in direct_keys | zenodo_keys if key.startswith("ffsvc"))
        elif normalized == "restricted-ffsvc":
            selected.update(zenodo_keys)
        elif normalized in direct_keys | zenodo_keys:
            selected.add(normalized)
        else:
            known = ", ".join(sorted(direct_keys | zenodo_keys))
            raise SystemExit(f"Unknown dataset {value!r}. Known keys: {known}")
    return selected


def _print_inventory() -> None:
    print("Direct/open downloads:")
    for item in DIRECT_DOWNLOADS:
        print(f"  {item.key}: {item.description}")
    print("\nRestricted Zenodo records, token required:")
    for item in ZENODO_RECORDS:
        print(f"  {item.key}: Zenodo record {item.record_id}; {item.description}")


def _download_direct(
    item: DirectDownload,
    *,
    force: bool,
    extract: bool,
    download_passes: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    target = DATASETS_ROOT / item.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        if item.md5 and not _verify_md5(target, item.md5):
            print(f"Existing archive is incomplete or corrupt; resuming {target}")
        else:
            _maybe_extract(target, item.extract_to, extract=extract)
            return _status(item.key, "exists", target)
    if target.exists() and force:
        target.unlink()
    _run_wget_with_fallback(
        (item.url, *item.mirrors),
        target,
        passes=max(1, download_passes),
        retry_sleep_seconds=retry_sleep_seconds,
    )
    verified = _verify_md5(target, item.md5)
    if verified:
        _maybe_extract(target, item.extract_to, extract=extract)
        return _status(item.key, "downloaded", target, md5_ok=True)
    return _status(item.key, "failed", target, md5_ok=False, reason="md5 mismatch")


def _download_zenodo_record(
    item: ZenodoRecord,
    *,
    token: str,
    force: bool,
    extract: bool,
) -> dict[str, Any]:
    target_dir = DATASETS_ROOT / item.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    if not token:
        _write_access_note(item, target_dir)
        return _status(
            item.key,
            "skipped",
            target_dir,
            reason="restricted Zenodo record requires ZENODO_ACCESS_TOKEN",
        )
    try:
        metadata = _fetch_zenodo_record(item.record_id, token=token)
    except Exception as exc:  # noqa: BLE001
        _write_access_note(item, target_dir, error=str(exc))
        return _status(item.key, "skipped", target_dir, reason=str(exc))
    files = metadata.get("files", [])
    if not isinstance(files, list) or not files:
        return _status(item.key, "skipped", target_dir, reason="record has no visible files")

    downloaded: list[str] = []
    md5_results: list[bool] = []
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        key = str(file_info.get("key") or file_info.get("filename") or "").strip()
        if not key:
            continue
        url = _zenodo_file_url(file_info)
        if not url:
            continue
        target = target_dir / Path(key).name
        if target.exists() and not force:
            downloaded.append(str(target))
            md5_results.append(_verify_zenodo_file_checksum(target, file_info))
            _maybe_extract(target, None, extract=extract)
            continue
        if target.exists() and force:
            target.unlink()
        _run_wget(url, target, headers=(f"Authorization: Bearer {token}",))
        downloaded.append(str(target))
        md5_results.append(_verify_zenodo_file_checksum(target, file_info))
        _maybe_extract(target, None, extract=extract)
    record_md5_ok = True
    if len(downloaded) == 1 and item.md5:
        record_md5_ok = _verify_md5(Path(downloaded[0]), item.md5)
    md5_ok = bool(downloaded) and all(md5_results) and record_md5_ok
    return _status(item.key, "downloaded", target_dir, files=downloaded, md5_ok=md5_ok)


def _run_wget(url: str, target: Path, *, headers: Iterable[str] = ()) -> None:
    cmd = [
        "wget",
        "--no-check-certificate",
        "--retry-connrefused",
        "--waitretry=3",
        "--timeout=120",
        "-t",
        "5",
        "-c",
        url,
        "-O",
        str(target),
    ]
    for header in headers:
        cmd.insert(1, f"--header={header}")
    subprocess.run(cmd, check=True)


def _run_wget_with_fallback(
    urls: Iterable[str],
    target: Path,
    *,
    passes: int,
    retry_sleep_seconds: float,
) -> None:
    url_list = tuple(urls)
    errors: list[str] = []
    for pass_index in range(1, passes + 1):
        start_size = target.stat().st_size if target.exists() else 0
        print(f"Download pass {pass_index}/{passes} for {target} starting at {start_size} bytes")
        for url in url_list:
            try:
                _run_wget(url, target)
                return
            except subprocess.CalledProcessError as exc:
                current_size = target.stat().st_size if target.exists() else 0
                errors.append(f"pass={pass_index} {url}: exit={exc.returncode} size={current_size}")
        end_size = target.stat().st_size if target.exists() else 0
        if end_size > start_size:
            print(f"Download pass {pass_index} made progress: {start_size} -> {end_size} bytes")
        if pass_index < passes and retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)
    raise RuntimeError("all download URLs failed: " + "; ".join(errors))


def _fetch_zenodo_record(record_id: str, *, token: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"access_token": token})
    url = f"https://zenodo.org/api/records/{record_id}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Zenodo API response for record {record_id}")
    return payload


def _zenodo_file_url(file_info: dict[str, Any]) -> str:
    links = file_info.get("links")
    if isinstance(links, dict):
        for key in ("self", "download"):
            value = links.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _verify_zenodo_file_checksum(path: Path, file_info: dict[str, Any]) -> bool:
    checksum = str(file_info.get("checksum") or "")
    if not checksum:
        return True
    if ":" in checksum:
        algorithm, expected = checksum.split(":", 1)
    else:
        algorithm, expected = "md5", checksum
    if algorithm.lower() != "md5":
        return True
    return _verify_md5(path, expected)


def _verify_md5(path: Path, expected: str | None) -> bool:
    if not expected:
        return True
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        print(f"MD5 mismatch for {path}: expected {expected}, got {actual}")
        return False
    print(f"MD5 OK for {path}: {actual}")
    return True


def _maybe_extract(path: Path, extract_to: str | None, *, extract: bool) -> None:
    if not extract:
        return
    name = path.name.lower()
    if not (
        name.endswith(".zip")
        or name.endswith(".tar")
        or name.endswith(".tar.gz")
        or name.endswith(".tgz")
    ):
        return
    output_dir = DATASETS_ROOT / extract_to if extract_to else path.with_suffix("")
    if output_dir.exists():
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(output_dir.parent)
    else:
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(path, mode) as archive:
            archive.extractall(output_dir.parent)
    if extract_to and not output_dir.exists():
        print(f"WARNING: expected extracted directory {output_dir} was not created.")


def _write_access_note(item: ZenodoRecord, target_dir: Path, *, error: str = "") -> None:
    note = [
        f"# {item.key}",
        "",
        f"Zenodo record: https://zenodo.org/records/{item.record_id}",
        "",
        "This record is restricted. Request access on Zenodo and rerun with:",
        "",
        "```bash",
        (
            "ZENODO_ACCESS_TOKEN=<token> uv run python "
            f"scripts/download_external_speaker_datasets.py --dataset {item.key}"
        ),
        "```",
        "",
    ]
    if error:
        note.extend(["Last error:", "", f"```text\n{error}\n```", ""])
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "REQUEST_ACCESS.md").write_text("\n".join(note), encoding="utf-8")


def _status(key: str, status: str, path: Path, **extra: Any) -> dict[str, Any]:
    row = {"dataset": key, "status": status, "path": str(path)}
    row.update(extra)
    print(json.dumps(row, sort_keys=True))
    return row


if __name__ == "__main__":
    main()
