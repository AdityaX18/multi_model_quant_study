# Does quantization × temperature break small tool-callers?

Sprint 1 (Nov 2026). Applies Conscious Engines' factorial design (arXiv 2606.29581: quantization ×
sampling temperature, on safety refusals) to **tool calling**: Qwen2.5 1.5B / 3B / 7B, seven GGUF
quantization levels, 840 BFCL items, four temperatures with repeated samples.

## One command
```bash
python -m harness.pipeline --config configs/smoke.json      # laptop, ~6 min, 208 generations
python -m harness.pipeline --config configs/l40_full.json   # big GPU, 299,880 generations (RUN_ON_GPU.md)
```
Stages: check → fetch → preflight → generate → score → analyse → pack. Re-running skips finished work.

## Folder map
| Path | What it does | Author |
|---|---|---|
| `configs/*.json` | The experiment: which models, quant levels, items, temperatures, samples | you decide |
| `data/bfcl/` | 840 BFCL v4 items and answer keys (Apache-2.0, from ShishirPatil/gorilla) | copied |
| `harness/bfcl.py` | Load items, convert BFCL schemas to OpenAI tool format | AI, you read |
| `harness/score.py` | Grade one output: correct, or which error | AI, **you read line by line** |
| `harness/llama.py` | Start/stop `llama-server` for one GGUF file, send chat requests | AI, you read |
| `harness/fetch_models.py` | Download GGUF files (also split ones) from Hugging Face | AI, you read |
| `harness/run_grid.py` | The loop: model × quant × item × temperature × sample, parallel, resumable | AI, you read |
| `harness/pipeline.py` | The one command that runs every stage | AI, you read |
| `harness/sustained.py` | Speed under sustained load (TinyBench protocol), separate from accuracy | AI |
| `core/stats.py` | Wilson, McNemar, Holm, paired bootstrap, flip rate, CDI | **you** |
| `core/analysis.py` | scored.jsonl → tables and plots | **you** |
| `tests/test_harness.py` | Tests for the plumbing (5) | AI |
| `tests/test_stats.py`, `tests/test_analysis.py` | Tests for your code, written before the code | **you** |
| `notes/` | prereg, paper notes, decision record, nightly log | you |
| `scripts/`, `Dockerfile`, `RUN_ON_GPU.md` | GPU setup and the overnight run | AI |

## Data flow
```
config ──► fetch ──► models/*.gguf
                         │
items (bfcl.py) ──► generate (run_grid.py + llama.py) ──► results/<run>/generations.jsonl   raw outputs
                                                              │
                                                score (score.py) ──► scored.jsonl         one row per generation
                                                              │
                                              analyse (core/analysis.py + core/stats.py) ──► report/
```
Raw outputs are kept, so a scorer fix only needs `--stages score analyse`, never a new GPU run.

## Why the grid looks like this
- **fp16 is the reference**, not q8_0: every quant level is compared with the unquantized model.
- **Temperature 0 runs twice.** Batched GPU inference can change float sums, so even greedy decoding
  can flip between two identical runs. The two repeats measure that noise floor; a quantization effect
  only counts if it is larger.
- **Same seeds across quant levels**, so samples are paired and paired tests apply.
- **First-token alternatives at T=0** are stored: how close each item was to "call" versus "answer in
  text". That margin is the mechanism behind the temperature interaction.
- **Budget:** generations = Σ quant levels × items × (t0_repeats + (temperatures − 1) × samples).
