#!/usr/bin/env bash
# One-time setup on a Linux machine with an NVIDIA GPU (tested path: Ubuntu + CUDA toolkit installed).
# Builds llama.cpp with CUDA at the same commit used for the CPU smoke run, makes a Python venv,
# and runs the plumbing tests. Takes ~10 minutes, mostly the CUDA build.
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_COMMIT=f3b407c
if [ ! -x .cache/llama.cpp/build/bin/llama-server ]; then
  mkdir -p .cache
  [ -d .cache/llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp .cache/llama.cpp
  git -C .cache/llama.cpp checkout -q "$LLAMA_COMMIT"
  cmake -S .cache/llama.cpp -B .cache/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build .cache/llama.cpp/build --target llama-server -j "$(nproc)"
fi
python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
python -m pytest tests/test_harness.py -q
echo
echo "Ready. Run:"
echo "  export LLAMA_SERVER=$PWD/.cache/llama.cpp/build/bin/llama-server"
echo "  . .venv/bin/activate && python -m harness.pipeline --config configs/l40_full.json"
