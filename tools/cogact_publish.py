#!/usr/bin/env python3
"""Upload closed probe artifacts using per-blob SAS URLs supplied on stdin.

The manifest contains ``files`` (local path, destination URL), ``receipt_path``
and optionally ``receipt_url``. URLs are credentials: do not log the manifest.
Checkpoint completion markers are uploaded only after every other file succeeds.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


def redact(value):
    value = re.sub(r'https?://\S+', '[storage URL]', str(value))
    return re.sub(r'(?i)sig=[^&\s]+', 'sig=[redacted]', value)


def remote_size(url):
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(request, timeout=60) as response:
                return int(response.headers['Content-Length'])
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def upload(entry):
    path = Path(entry['path'])
    if entry.get('verify_existing') and remote_size(entry['url']) == path.stat().st_size:
        print(f'[direct-publish] verified-existing {path.parent.name}/{path.name}', flush=True)
        return path.stat().st_size
    result = subprocess.run(
        ['azcopy', 'copy', str(path), entry['url'], '--overwrite=true',
         '--check-length=true', '--log-level=NONE', '--output-type=json'],
        capture_output=True, text=True, timeout=1800,
        env={**os.environ, 'AZCOPY_CONCURRENCY_VALUE': '16'})
    if result.returncode:
        # Error output can contain signed URLs. Preserve only redacted details.
        detail = redact(result.stdout + result.stderr)
        raise RuntimeError(f'{path.name}: azcopy exit {result.returncode}: {detail[-1500:]}')
    size = remote_size(entry['url'])
    if size != path.stat().st_size:
        raise RuntimeError(f'{path.name}: remote size differs from local file')
    print(f'[direct-publish] verified {path.parent.name}/{path.name} {size}', flush=True)
    return size


def main():
    manifest = json.load(sys.stdin)
    receipt_path = Path(manifest['receipt_path'])
    receipt = {'exit_code': 1, 'transport': 'azcopy',
               'result_subdir': manifest['result_subdir']}
    try:
        files = manifest['files']
        ordinary = [e for e in files if Path(e['path']).name != 'complete.json']
        markers = [e for e in files if Path(e['path']).name == 'complete.json']
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            sizes = list(pool.map(upload, ordinary))
        sizes.extend(upload(e) for e in markers)
        receipt.update(exit_code=0, verified_files=len(sizes), verified_bytes=sum(sizes))
    except Exception as error:
        receipt['error'] = redact(error)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    if manifest.get('receipt_url'):
        try:
            upload({'path': str(receipt_path), 'url': manifest['receipt_url']})
        except Exception as error:
            receipt.update(exit_code=1, receipt_upload_error=redact(error))
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print('[direct-publish] receipt ' + json.dumps(receipt), flush=True)
    return receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
