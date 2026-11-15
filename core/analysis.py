"""results/<run>/scored.jsonl -> tables and plots in results/<run>/report/.  YOU write this file.

The pipeline calls main(run_dir) as its "analyse" stage. Until you write it, the pipeline skips this
stage and still packs the raw results, so the GPU run never waits for your code.

One row of scored.jsonl looks like:
  {"key": "qwen2.5-1.5b|q4_k_m|simple_12|0.7|3", "model": "qwen2.5-1.5b", "quant": "q4_k_m",
   "item": "simple_12", "category": "simple", "temperature": 0.7, "sample": 3,
   "correct": true, "error": "none", "n_calls": 1, "first_tokens": null, ...}
At temperature 0, "first_tokens" holds the top alternatives for the first 4 generated tokens.

Suggested functions (each one gets tests in tests/test_analysis.py that YOU write first):
  load(run_dir)            -> list of rows
  accuracy_table(rows)     -> accuracy + Wilson interval per (model, quant, temperature, category)
  noise_floor(rows)        -> flip rate between the two temperature-0 repeats of the SAME quant
  flip_table(rows)         -> decision-flip rate of each quant vs the reference, per (model, temperature)
  paired_vs_reference(rows)-> McNemar at T=0, paired bootstrap at T>0, Holm across the family
  interaction(rows)        -> compound degradation index per (model, quant, temperature) with an interval
  margin(rows)             -> first-token "call vs text" margin at the reference, T=0, per item
"""

from __future__ import annotations

from pathlib import Path


def main(run_dir: Path) -> None:
    raise NotImplementedError("write core/analysis.py")
