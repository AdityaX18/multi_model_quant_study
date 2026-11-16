#!/usr/bin/env bash
# The whole job in one line, safe to re-run after a crash (finished work is skipped):
#   nohup scripts/run_full.sh configs/l40_run_a.json > run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export LLAMA_SERVER="${LLAMA_SERVER:-$PWD/.cache/llama.cpp/build/bin/llama-server}"
. .venv/bin/activate
python -m harness.pipeline --config "${1:-configs/l40_run_a.json}"
