# Running the full grid on a big GPU (for Deepesh)

What it is: 3 Qwen2.5 sizes (1.5B, 3B, 7B) × 7 quantization levels (fp16 down to q2_k) × 840 BFCL
tool-calling items × 4 temperatures with repeated samples = 299,880 generations through llama.cpp.
No API keys. Downloads ~78 GB of GGUF files from Hugging Face. Writes everything under `results/`.

## Needs
Linux, NVIDIA driver + CUDA toolkit (`nvcc --version` works), git, cmake, python3 with venv, ~80 GB free disk,
internet. No Hugging Face login or token: the Qwen GGUF repos are public.

## Steps
```bash
git clone https://github.com/AdityaX18/multi_model_quant_study.git && cd multi_model_quant_study
bash scripts/setup_gpu.sh                 # builds llama.cpp with CUDA, venv, 5 tests should pass
nohup bash scripts/run_full.sh configs/l40_full.json > run.log 2>&1 &
tail -f run.log
```
After ~10 minutes the log prints a `[preflight]` line with a time estimate. If it says STOP, send me
diff --git a/core/__init__.py b/core/__init__.py
index 35c47fd..1591c1b 100644
--- a/core/__init__.py
+++ b/core/__init__.py
@@ -1,5 +1,5 @@.

## If you need the GPU back
Kill it any time. Re-running the same command resumes; nothing finished is redone. Models run in the
order 1.5B, 3B, 7B, so a partial run is still useful. To pack what exists so far:
`python -m harness.pipeline --config configs/l40_full.json --stages score pack`

## What to send back
`results/l40-full-v1.tar.gz` (the last line of the log prints its path and size).

## Knobs
`"parallel": 16` in the config = requests in flight. If GPU utilisation (nvidia-smi) is below ~80%,
raise it to 32; if you see out-of-memory, lower it to 8.
