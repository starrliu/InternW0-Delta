#!/usr/bin/env bash
# Execute inside the allocated Pod. This script has no Pod/Job lifecycle actions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
COGACT_PREP_ROOT="${COGACT_PREP_ROOT:-/data/yuming/internw0-delta/preparation-20261023}"
mkdir -p "$COGACT_PREP_ROOT"
trap 'rc=$?; printf "%s\n" "$rc" > "$COGACT_PREP_ROOT/environment.exitcode"' EXIT
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export DS_BUILD_OPS=0
export UV_CACHE_DIR=/data/yuming/internw0-delta/cache/uv
export PIP_CACHE_DIR=/data/yuming/internw0-delta/cache/pip
export UV_CONCURRENT_BUILDS=2
export UV_CONCURRENT_DOWNLOADS=4
if [[ ! -x .venv/bin/python ]]; then
  python -m venv .venv
fi
.venv/bin/python -m pip install uv
.venv/bin/python -m uv pip install --python .venv/bin/python --torch-backend cu128 \
  -e '.[train,cogact,logging]' ninja setuptools
.venv/bin/python -m pip freeze > "$COGACT_PREP_ROOT/pip-freeze.txt"
.venv/bin/python - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import torch
import torchvision
import av
import transformers
result = {
    'torch': torch.__version__, 'torchvision': torchvision.__version__,
    'cuda_build': torch.version.cuda, 'transformers': transformers.__version__,
    'av': av.__version__, 'deepspeed': metadata.version('deepspeed'),
    'scope': 'CPU imports only; GPU validation deferred until cards are available',
}
assert torch.__version__.startswith('2.10.0')
assert torchvision.__version__.startswith('0.25.0')
print(json.dumps(result), flush=True)
Path(os.environ.get('COGACT_PREP_ROOT', '/data/yuming/internw0-delta/preparation-20261023'),
     'environment.json').write_text(json.dumps(result, indent=2) + '\n')
PY
