"""Download pinned public source files into a raw cache (spec §3).

Cache lives outside the distributable tree. Files are fetched at the immutable revision
recorded in config/sources.yaml, size-capped, written atomically, and hashed. Nothing
downloaded is ever executed.

usage: python -m data_pipeline.ingest_public [source_id ...]
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("WAF_RAW_CACHE", ROOT.parent / "raw_cache"))
MAX_BYTES = 8 * 2**20  # default per-file cap when the audit recorded no size


def sources(root=ROOT):
    return {s["id"]: s for s in yaml.safe_load(open(root / "config/sources.yaml"))["sources"]}


def raw_url(src, path):
    if src["kind"] == "s3_public":  # unsigned public bucket, e.g. the CSE-CIC-IDS2018 AWS open-data bucket
        return f"{src['upstream_url'].rstrip('/')}/{quote(path)}"
    if src["kind"] == "huggingface":
        repo = src["upstream_url"].removeprefix("https://huggingface.co/datasets/")
        return f"https://huggingface.co/datasets/{repo}/resolve/{src['revision']}/{quote(path)}"
    owner_repo = src["upstream_url"].removeprefix("https://github.com/")
    return f"https://raw.githubusercontent.com/{owner_repo}/{src['revision']}/{quote(path)}"


def fetch_prefix(url, dest, n, retries=5):
    """First n bytes of a large file (pinned revision + fixed n = deterministic sample). The last, partial
    line is the parser's job (pandas on_bad_lines / truncation)."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(url, timeout=60, stream=True, headers={"Range": f"bytes=0-{n - 1}"}) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    got = 0
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk[: n - got])
                        got += len(chunk)
                        if got >= n:
                            break
            os.replace(tmp, dest)
            return
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch(url, dest, cap=MAX_BYTES, retries=5):
    """Resumable (HTTP Range on the .part file), size-capped, atomic rename on completion."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        have = tmp.stat().st_size if tmp.exists() else 0
        try:
            hdr = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, timeout=60, stream=True, headers=hdr) as r:
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored Range: restart
                    have = 0
                with open(tmp, "ab" if have else "wb") as f:
                    n = have
                    for chunk in r.iter_content(1 << 20):
                        n += len(chunk)
                        if n > cap:
                            raise ValueError(f"{url} exceeds cap {cap} bytes")
                        f.write(chunk)
            os.replace(tmp, dest)
            return
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def ingest(src):
    """Fetch every file entry (directory entries end with '/' and are handled by their parser)."""
    base = RAW / src["id"] / src["revision"]
    manifest = {}
    for f in src["files"] or []:
        if f["path"].endswith("/"):
            continue
        dest = base / f["path"]
        if f.get("prefix_bytes"):
            fetch_prefix(raw_url(src, f["path"]), dest, f["prefix_bytes"])
        else:
            fetch(raw_url(src, f["path"]), dest, cap=f.get("size_bytes") or MAX_BYTES)
        size = dest.stat().st_size
        want = f.get("prefix_bytes") or f.get("size_bytes")
        if want is not None and size != want:
            raise ValueError(f"{src['id']}:{f['path']} size {size} != recorded {want}")
        h = hashlib.sha256()
        with open(dest, "rb") as fh:  # streamed: GB-scale files never fully in memory
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        manifest[f["path"]] = {"sha256": h.hexdigest(), "size_bytes": size}
    base.mkdir(parents=True, exist_ok=True)
    (base / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


if __name__ == "__main__":
    srcs = sources()
    ids = sys.argv[1:] or [i for i, s in srcs.items() if s["enabled"] and s["kind"] in ("github", "huggingface", "s3_public")]
    for i in ids:
        m = ingest(srcs[i])
        print(f"{i}: {len(m)} files")
