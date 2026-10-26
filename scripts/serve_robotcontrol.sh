#!/usr/bin/env bash
# Start a single inference process; does not manage Pods or training processes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${WAM_PYTHON:-$PWD/.venv/bin/python}" -m wam.inference.robotcontrol_server "$@"
