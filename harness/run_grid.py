"""Generate: every model x quant level x item x temperature x sample, one JSON line per generation.

Output: results/<run>/generations.jsonl. Resumable: a key already on disk is never generated again,
so a crashed job continues where it stopped (run the same command again).

Key = "<model>|<quant>|<item id>|<temperature>|<sample>".

Design choices that matter for the statistics (explained in README "Why the grid looks like this"):
- temperature 0 is run `t0_repeats` times (default 2). With several slots the server batches requests
  together, and batching can change floating-point sums, so even greedy decoding can differ between
  two runs. Two repeats measure that noise floor; any quantization effect must be larger than it.
- temperature > 0 is run `samples` times with seeds seed+0, seed+1, ...; the same seeds are used for
  every quant level, so samples are paired across quant levels.
- at temperature 0 the top alternatives of the first tokens are stored, to measure how close each item
  was to "call the tool" versus "answer in text" (the decision margin).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import bfcl
from .llama import LlamaServer
from .score import score


def items_for(cfg: dict) -> list[dict]:
    out = []
    for cat, n in cfg["items"].items():
        out += bfcl.load(cat, limit=n)
    return out


def plan_for(cfg: dict) -> list[tuple[float, int]]:
    """(temperature, sample index) pairs run for every item."""
    plan = []
    for t in cfg["temperatures"]:
        n = cfg.get("t0_repeats", 2) if t == 0 else cfg["samples"]
        plan += [(t, s) for s in range(n)]
    return plan


def count(cfg: dict) -> int:
    n_items = sum(len(bfcl.load(c, limit=n)) for c, n in cfg["items"].items())
    n_quant = sum(len(m["quants"]) for m in cfg["models"])
    return n_quant * n_items * len(plan_for(cfg))


def done_keys(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("error") != "request_failed":  # failed requests are retried on the next run
                    done.add(row["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def run_one(srv: LlamaServer, cfg: dict, model: str, quant: str, it: dict, t: float, s: int) -> dict:
    key = f"{model}|{quant}|{it['id']}|{t}|{s}"
    try:
        r = srv.chat(it["messages"], tools=it["tools"], temperature=t, seed=cfg.get("seed", 0) + s,
                     max_tokens=cfg.get("max_tokens", 256), top_logprobs=cfg.get("top_logprobs", 5) if t == 0 else 0)
        verdict = score(it, r["tool_calls"])
    except Exception as e:  # recorded, never silently dropped; retried on the next run
        r = {"content": "", "tool_calls": [], "request_error": repr(e)}
        verdict = {"correct": False, "error": "request_failed"}
    return {"key": key, "model": model, "quant": quant, "item": it["id"], "category": it["category"],
            "temperature": t, "sample": s, **verdict, **r}


def generate(cfg: dict, out: Path, model_paths: dict[tuple[str, str], Path], limit_items: int | None = None,
             gen_name: str = "generations.jsonl") -> Path:
    gen_path = out / gen_name
    done = done_keys(gen_path)
    items = items_for(cfg)
    if limit_items:
        items = items[:limit_items]
    plan = plan_for(cfg)
    parallel = cfg.get("parallel", 1)
    lock = threading.Lock()
    t_start = time.time()
    with gen_path.open("a", encoding="utf-8") as f:
        for m in cfg["models"]:
            for quant in m["quants"]:
                todo = [(it, t, s) for it in items for t, s in plan
                        if f"{m['name']}|{quant}|{it['id']}|{t}|{s}" not in done]
                if not todo:
                    continue
                print(f"[generate] {m['name']} {quant}: {len(todo)} to do, {parallel} parallel", flush=True)
                t0 = time.time()
                with LlamaServer(model_paths[(m["name"], quant)], ctx=cfg.get("ctx", 4096), parallel=parallel,
                                 threads=cfg.get("threads"), gpu_layers=cfg.get("gpu_layers", 0),
                                 binary=cfg.get("llama_server"), extra_args=cfg.get("server_args"),
                                 log=out / f"server-{m['name']}-{quant}.log") as srv:
                    with ThreadPoolExecutor(max_workers=parallel) as pool:
                        futs = [pool.submit(run_one, srv, cfg, m["name"], quant, it, t, s) for it, t, s in todo]
                        for i, fut in enumerate(as_completed(futs), 1):
                            row = fut.result()
                            with lock:
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                                f.flush()
                            if i % 200 == 0 or i == len(todo):
                                rate = i / (time.time() - t0)
                                print(f"  {m['name']} {quant}: {i}/{len(todo)}  {rate:.1f} gen/s  "
                                      f"total {(time.time() - t_start) / 60:.1f} min", flush=True)
    return gen_path
