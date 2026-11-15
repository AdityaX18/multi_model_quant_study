"""Sustained-load speed test following TinyBench's protocol (conscious-engines/TinyBench).

Their paper measures 1 warm-up + 20 back-to-back generations of one fixed prompt and reports the
per-iteration throughput, so throttling shows up as a downward trend rather than hiding in an average.
This script does the same against a llama.cpp server, for any GGUF file, and logs one CSV row per
iteration. On an NVIDIA machine it also samples power and temperature from nvidia-smi.

  git clone https://github.com/conscious-engines/TinyBench ../TinyBench
  python -m harness.sustained --model path/to/model.gguf --label q4_k_m --gpu-layers 99 \
      --prompt ../TinyBench/rtx4050-laptop/prompt.txt

The prompt is theirs and their repo has no licence file, so it is read from your clone, not copied here.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path

from .llama import LlamaServer

ROOT = Path(__file__).resolve().parents[1]


def gpu_sample() -> dict:
    if not shutil.which("nvidia-smi"):
        return {}
    q = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split("\n")[0]
    try:
        p, t, u = [float(x) for x in q.split(",")]
        return {"gpu_power_w": p, "gpu_temp_c": t, "gpu_util_pct": u}
    except ValueError:
        return {}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt", required=True, help="text file with the fixed prompt (TinyBench's prompt.txt)")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gap-s", type=float, default=1.0, help="pause between iterations (TinyBench uses 1 s)")
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--llama-server", default=None)
    a = ap.parse_args(argv)
    out = ROOT / "results" / "sustained"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.label}.csv"
    messages = [{"role": "user", "content": Path(a.prompt).read_text(encoding="utf-8")}]
    with LlamaServer(a.model, gpu_layers=a.gpu_layers, threads=a.threads, binary=a.llama_server,
                     log=out / f"server-{a.label}.log") as srv, path.open("w", newline="") as f:
        w = None
        for i in range(a.iterations + 1):  # iteration 0 is the warm-up and is marked as such
            r = srv.chat(messages, temperature=0.0, max_tokens=a.max_tokens)
            row = {"iteration": i, "warmup": i == 0, "completion_tokens": r["completion_tokens"],
                   "prompt_ms": r["prompt_ms"], "decode_tok_s": r["tokens_per_s"], "wall_s": r["wall_s"], **gpu_sample()}
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)
            f.flush()
            print(row)
            time.sleep(a.gap_s)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
