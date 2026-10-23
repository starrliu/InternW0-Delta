#!/usr/bin/env bash
# Run inside an existing allocated Pod. Never changes Pod/Job/PID-1 lifecycle.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:?Usage: cogact_b200_run.sh <text|smoke|train> [Hydra overrides]}"
shift
case "$mode" in text|smoke|train) ;; *) echo 'Unknown mode' >&2; exit 2;; esac
export WAM_CHECKPOINT_ROOT=/data/yuming/internw0-delta/checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$WAM_CHECKPOINT_ROOT"
export WAM_VLM_PATH="$WAM_CHECKPOINT_ROOT/RynnBrain1.1-2B"
export WAM_CACHE_ROOT=/data/yuming/internw0-delta/cache
export WAM_OUTPUT_ROOT=/data/yuming/internw0-delta/runs
export HF_HOME=/data/yuming/internw0-delta/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/workspace/internw0-delta-yuming-20261023/.compile-cache
export TRITON_CACHE_DIR=/workspace/internw0-delta-yuming-20261023/.triton-cache
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export NPROC_PER_NODE=8
export WAM_ACCELERATE_CONFIG=scripts/accelerate_configs/accelerate_zero2_ds.yaml
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$WAM_OUTPUT_ROOT"
exec 9>/data/yuming/internw0-delta/gpu-run.lock
flock -n 9 || { echo 'Another InternW0 GPU process owns the run lock' >&2; exit 3; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPUs have existing processes. This script does not stop them.' >&2
  exit 3
fi
case "$mode" in
  text)
    exec .venv/bin/torchrun --standalone --nproc_per_node=8 tools/text_cache.py task=cogact "$@"
    ;;
  smoke)
    exec bash run.sh cogact max_steps=8 log_every=1 save_every=8 eval_every=8 \
      eval_loss_samples_per_rank=1 save_on_train_end=false "$@"
    ;;
  train)
    exec bash run.sh cogact "$@"
    ;;
esac
