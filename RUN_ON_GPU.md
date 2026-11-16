# Running the grid on a big GPU (for Deepesh)

Two separate runs, so each fits comfortably and you can do them on different days:

| Run | Config | Models | Generations | Download in total | Disk at any moment |
|---|---|---|---|---|---|
| A | `configs/l40_run_a.json` | Qwen2.5 1.5B and 3B, 7 quant levels each | 199,920 | 32 GB | under 7 GB |
| B | `configs/l40_run_b.json` | Qwen2.5 7B, 7 quant levels | 99,960 | 47 GB | under 16 GB |

Each GGUF file is downloaded right before it is used and deleted as soon as its outputs are saved
(`"delete_after_use": true`), so disk never holds more than one file. GPU memory: the largest model is
15 GB of weights plus about 4 GB of cache at 32 parallel requests, so about 20 GB of the 48.

## How the GPU is used
- llama.cpp is built with CUDA (`-DGGML_CUDA=ON`) and every layer is on the GPU (`"gpu_layers": 99`).
- Batching: `"parallel": 32` keeps 32 requests in flight. llama-server decodes all active requests in
  one batched forward pass per token (continuous batching), so this is a batch size of 32.
- The log prints generations per second and GPU utilisation every 200 generations. If utilisation sits
  below ~90%, raise `"parallel"` to 64 in the config and re-run the same command (it resumes).

## Needs
Linux, NVIDIA driver + CUDA toolkit (`nvcc --version` works), git, cmake, python3 with venv, internet.
No Hugging Face login or token: the Qwen GGUF repos are public.

## Steps
```bash
git clone https://github.com/AdityaX18/multi_model_quant_study.git && cd multi_model_quant_study
bash scripts/setup_gpu.sh                 # builds llama.cpp with CUDA, venv, 5 tests should pass
nohup bash scripts/run_full.sh configs/l40_run_a.json > run_a.log 2>&1 &
tail -f run_a.log
```
Later, for the 7B: `nohup bash scripts/run_full.sh configs/l40_run_b.json > run_b.log 2>&1 &`

After a few minutes the log prints two `[preflight]` lines: how many layers went to the GPU, and a time
estimate for the whole run. If it says STOP, or 0 layers were offloaded, send me the log and
`results/l40-run-a/server-*.log`.

## If you need the GPU back
Kill it any time. Re-running the same command resumes; nothing finished is redone. To pack what exists
so far: `python -m harness.pipeline --config configs/l40_run_a.json --stages score pack`

## What to send back
`results/l40-run-a.tar.gz` and later `results/l40-run-b.tar.gz` (the last log line prints the path).
