#!/usr/bin/env bash
# Keep the scheduler's foreground process alive for the user's debug hold.
set -uo pipefail
: "${AMLT_OUTPUT_DIR:?AMLT_OUTPUT_DIR must be supplied by amlt}"
mkdir -p "$AMLT_OUTPUT_DIR"
COGACT_JOB_LOG=$(mktemp /tmp/cogact-job.XXXXXX.log)
set +e
bash scripts/cogact_a100_workload.sh 2>&1 | tee "$COGACT_JOB_LOG"
COGACT_JOB_EXIT=${PIPESTATUS[0]}
cp "$COGACT_JOB_LOG" "$AMLT_OUTPUT_DIR/workload.log"
printf '{"workload_exit_code":%d,"holding_seconds":72000}\n' "$COGACT_JOB_EXIT" > "$AMLT_OUTPUT_DIR/debug_hold.json"
echo "[debug_hold] workload exit=$COGACT_JOB_EXIT; keeping node for 20 hours"
sleep 72000
exit "$COGACT_JOB_EXIT"
