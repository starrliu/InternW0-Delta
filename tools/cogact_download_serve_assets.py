#!/usr/bin/env python3
"""Download the fixed CogACT deployment manifest using an authenticated AzCopy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError(f"Missing file or size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] /
                        "deploy/robotcontrol/assets_step80000.json")
    parser.add_argument("--output", type=Path, default=Path("deploy-artifacts"))
    parser.add_argument("--azcopy", default="azcopy", help="AzCopy executable name or absolute path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="Show files without downloading")
    mode.add_argument("--verify-only", action="store_true", help="Verify local sizes and SHA256 without downloading")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported asset manifest schema")
    base = manifest["base_url"].rstrip("/")
    if base != "https://eaidatashare.blob.core.windows.net/v-xingyumliu":
        raise ValueError("Unexpected Blob account/container")
    root = args.output.resolve()
    planned, destinations = [], set()
    for item in manifest["files"]:
        relative = Path(item["relative"])
        target = (root / relative).resolve()
        if relative.is_absolute() or target == root or not target.is_relative_to(root) or target in destinations:
            raise ValueError(f"Invalid/duplicate destination: {relative}")
        blob = item["blob"]
        parsed = urlparse(blob)
        if (not blob.startswith("internw0-delta/") or ".." in blob.split("/")
                or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment):
            raise ValueError("Invalid Blob path")
        if type(item["bytes"]) is not int or item["bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid size or SHA256")
        planned.append((item, target, base + "/" + blob))
        destinations.add(target)
    if sum(item["bytes"] for item, _, _ in planned) != manifest["total_bytes"]:
        raise ValueError("Manifest total size mismatch")
    for item, target, url in planned:
        if args.list:
            print(f"{item['bytes']:>12}  {item['relative']}\n              {url}")
            continue
        if target.exists() or args.verify_only:
            verify(target, item)
            print(f"Verified: {item['relative']}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download-partial")
        if temporary.is_symlink():
            raise ValueError(f"Refusing a symlink download destination: {temporary}")
        print(f"Downloading: {item['relative']}", flush=True)
        subprocess.run([args.azcopy, "copy", url, str(temporary), "--from-to=BlobLocal",
                        "--overwrite=true", "--check-length=true", "--log-level=ERROR",
                        "--output-level=essential"], check=True)
        verify(temporary, item)
        temporary.replace(target)
        print(f"Verified: {item['relative']}", flush=True)
    print(f"{'Listed' if args.list else 'Verified'} {len(planned)} files; step {manifest['step']}; "
          f"{manifest['total_bytes'] / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
