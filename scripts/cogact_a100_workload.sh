#!/usr/bin/env bash
set -euo pipefail
: "${COGACT_ASSET_DIR:?Set the prepared asset directory on project storage}"
: "${AMLT_OUTPUT_DIR:?AMLT_OUTPUT_DIR must be supplied by amlt}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export DS_BUILD_OPS=0
# This probe supplies PyTorch AdamW and uses no compiled DeepSpeed kernels.
# Do not require nvcc merely to inspect unused inference/quantization builders.
export DS_IGNORE_CUDA_DETECTION=1
export COGACT_LOCAL_ROOT=/tmp/internw0-cogact-a100
mkdir -p "$COGACT_LOCAL_ROOT"
echo '[setup] copying and verifying prepared assets'
python - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
source = Path(os.environ['COGACT_ASSET_DIR'])
destination = Path(os.environ['COGACT_LOCAL_ROOT']) / 'assets'
manifest = json.loads((source / 'assets_manifest.json').read_text())
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()
for relative, expected in manifest.items():
    target = destination / relative
    if target.exists() and target.stat().st_size == expected['bytes'] and digest(target) == expected['sha256']:
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    shutil.copyfile(source / relative, temporary)
    if temporary.stat().st_size != expected['bytes'] or digest(temporary) != expected['sha256']:
        raise RuntimeError('Asset checksum mismatch: ' + relative)
    temporary.replace(target)
    print('[assets] verified', relative, flush=True)
PY
echo '[setup] creating isolated Python environment'
if [[ ! -x "$COGACT_LOCAL_ROOT/venv/bin/python" ]]; then
  python -m venv "$COGACT_LOCAL_ROOT/venv"
fi
COGACT_PYTHON="$COGACT_LOCAL_ROOT/venv/bin/python"
"$COGACT_PYTHON" -m pip install uv
"$COGACT_PYTHON" -m uv pip install --python "$COGACT_PYTHON" --torch-backend cu128 -e '.[train,cogact]' ninja setuptools
export WAM_CHECKPOINT_ROOT="$COGACT_LOCAL_ROOT/assets/checkpoints"
export WAM_VLM_PATH="$WAM_CHECKPOINT_ROOT/RynnBrain1.1-2B"
"$COGACT_PYTHON" -m pip freeze > "$AMLT_OUTPUT_DIR/pip-freeze.txt"
nvidia-smi > "$AMLT_OUTPUT_DIR/gpus.txt"
echo '[run] starting eight-A100 capacity and training probe'
bash scripts/cogact_a100_probe_stage.sh
