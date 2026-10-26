#!/usr/bin/env bash
# Run once in each of two existing Pods. Never manages Kubernetes resources.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${NODE_RANK:?Set 0 on the coordinator Pod and 1 on the peer}"
: "${MASTER_ADDR:?Set the coordinator Pod IP on both Pods}"
: "${RUN_ID:?Use the original training run name}"
: "${RESUME_STATE_DIR:?Use the verified 16-rank full training state}"
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || exit 2
export NNODES=2
export MASTER_PORT="${MASTER_PORT:-29526}"
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
# These eight 400-Gbit adapters connect the two assigned B200 nodes.
export NCCL_IB_HCA='=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1'
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export WANDB_MODE="${WANDB_MODE:-online}"
.venv/bin/python - "$RESUME_STATE_DIR" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
state=json.loads((root/'trainer_state.json').read_text())
assert state['resume_world_size']==16, 'Expected a 16-rank training state'
assert state['resume_global_batch_size']==256, 'Expected global batch 256'
if 'resume_rng_seed' in state:
    receipt=json.loads((root/'reshard_complete.json').read_text())
    assert receipt['step']==state['global_step']
    assert len(receipt['verified_shards'])==16
print('Verified resume step',state['global_step'],flush=True)
PY
exec bash scripts/cogact_b200_run.sh train \
  task=cogact_bs256_100k_2node "resume=$RESUME_STATE_DIR" "$@"
