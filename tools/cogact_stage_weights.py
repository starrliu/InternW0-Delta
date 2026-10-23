#!/usr/bin/env python3
"""Download an explicit, signed Azure manifest from stdin without persisting tokens.

The manifest contains destination_root, receipt_path and files (relative, url,
bytes, sha256). Run inside the existing Pod; this tool never manages its lifetime.
Only verified files receive their final names. Existing different files are an error.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def beneath(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or path == root or not path.is_relative_to(root):
        raise ValueError("Destination must be a relative path within its root")
    return path


def run(manifest: dict) -> dict:
    root = Path(manifest["destination_root"]).resolve()
    if not root.is_relative_to(Path("/data/yuming")):
        raise ValueError("Weight destination must be below /data/yuming")
    root.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "destination_root": str(root), "files": []}
    env = dict(os.environ, AZCOPY_CONCURRENCY_VALUE="2", AZCOPY_BUFFER_GB="0.25")
    # No request/response logs or resumable plans containing SAS URLs are retained.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="internw0-azcopy-") as scratch:
        env.update(AZCOPY_LOG_LOCATION=scratch, AZCOPY_JOB_PLAN_LOCATION=scratch)
        for item in manifest["files"]:
            target = beneath(root, item["relative"])
            source = urlparse(item["url"])
            if (source.scheme != "https" or source.netloc != "eaidatashare.blob.core.windows.net"
                    or not source.path.startswith("/v-xingyumliu/internw0-delta/assets/")):
                raise ValueError("Unexpected weight source")
            expected = int(item["bytes"])
            if target.exists():
                if target.stat().st_size != expected or digest(target) != item["sha256"]:
                    raise ValueError("Existing destination differs: " + item["relative"])
                state = "already_verified"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".internw0-partial")
                result = subprocess.run(
                    ["azcopy", "copy", item["url"], str(temporary), "--from-to=BlobLocal",
                     "--overwrite=true", "--check-length=true", "--log-level=NONE",
                     "--output-level=quiet", "--cap-mbps=512"],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10800,
                )
                if result.returncode:
                    raise RuntimeError("AzCopy failed for " + item["relative"])
                if temporary.stat().st_size != expected or digest(temporary) != item["sha256"]:
                    raise RuntimeError("Downloaded checksum mismatch: " + item["relative"])
                temporary.replace(target)
                state = "downloaded_verified"
            report["files"].append({k: item[k] for k in ("relative", "bytes", "sha256")})
            print(json.dumps({"file": item["relative"], "status": state}), flush=True)
    report.update(status="passed", completed_at_unix=time.time())
    return report


def main():
    manifest = json.load(sys.stdin)
    receipt = Path(manifest["receipt_path"]).resolve()
    if not receipt.is_relative_to(Path("/data/yuming")):
        raise ValueError("Receipt must be below /data/yuming")
    try:
        report = run(manifest)
    except Exception as exc:
        # Exception objects from subprocess can include signed command arguments.
        report = {"status": "failed", "error_type": type(exc).__name__}
        print(json.dumps(report), flush=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(receipt.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(receipt)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
