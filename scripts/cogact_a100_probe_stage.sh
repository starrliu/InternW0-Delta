#!/usr/bin/env bash
# Keep distributed checkpoint I/O off the remote filesystem. Publish only after
# workers exit, so a slow upload cannot time out a GPU collective.
set -uo pipefail
: "${AMLT_OUTPUT_DIR:?AMLT_OUTPUT_DIR must be supplied by amlt}"
: "${COGACT_LOCAL_ROOT:?Set the node-local asset and environment root}"
export COGACT_RESULT_SUBDIR=${COGACT_RESULT_SUBDIR:-probe}
export COGACT_PROBE_LOCAL_OUTPUT="$COGACT_LOCAL_ROOT/results/$COGACT_RESULT_SUBDIR"
COGACT_STAGE_PYTHON="$COGACT_LOCAL_ROOT/venv/bin/python"
mkdir -p "$COGACT_PROBE_LOCAL_OUTPUT"
"$COGACT_STAGE_PYTHON" tools/cogact_probe.py sweep \
  --data "$COGACT_LOCAL_ROOT/assets/data" \
  --output "$COGACT_PROBE_LOCAL_OUTPUT" --max-batch 64 --train-steps 200
COGACT_STAGE_EXIT=$?
echo "[publish] local probe exit=$COGACT_STAGE_EXIT; uploading artifacts sequentially"
"$COGACT_STAGE_PYTHON" - <<'PY'
import os
import shutil
from pathlib import Path

source = Path(os.environ['COGACT_PROBE_LOCAL_OUTPUT'])
target = Path(os.environ['AMLT_OUTPUT_DIR']) / os.environ['COGACT_RESULT_SUBDIR']
files = [p for p in source.rglob('*') if p.is_file() and not p.name.endswith('.tmp')]
# Small reports first, then checkpoint tensors, then completion markers. A
# checkpoint is never advertised as durable until all of its files are closed.
files.sort(key=lambda p: (p.name == 'complete.json', p.suffix == '.pt', str(p)))
for path in files:
    relative = path.relative_to(source)
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    shutil.copyfile(path, temporary)
    if temporary.stat().st_size != path.stat().st_size:
        raise RuntimeError(f'Incomplete upload: {relative}')
    temporary.replace(destination)
    if path.suffix == '.pt' or path.name == 'complete.json':
        print('[publish]', relative, path.stat().st_size, flush=True)
print('[publish] complete', flush=True)
PY
COGACT_PUBLISH_EXIT=$?
if [[ "$COGACT_PUBLISH_EXIT" -ne 0 ]]; then
  echo "[publish] failed; node-local artifacts retained at $COGACT_PROBE_LOCAL_OUTPUT"
  exit "$COGACT_PUBLISH_EXIT"
fi
exit "$COGACT_STAGE_EXIT"
