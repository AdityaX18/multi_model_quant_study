"""One command from nothing to results:

  python -m harness.pipeline --config configs/l40_run_a.json

Stages, in order (each one is skipped if its work is already on disk, so re-running is always safe):
  check      llama-server found, GPU visible, items load
  preflight  24 items on the first model: are tool calls coming back at all, how long will the full run take
  generate   the full grid -> results/<name>/generations.jsonl            (harness/run_grid.py)
             each GGUF is downloaded right before use; with "delete_after_use" it is deleted right after
  score      re-grade every row from the raw output -> scored.jsonl       (harness/score.py)
  analyse    tables and plots -> results/<name>/report/                    (core/analysis.py, YOURS)
  pack       results/<name>.tar.gz, the one file to send back

Run a subset with --stages, e.g. `--stages score analyse` after fixing the scorer.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from . import bfcl
from .fetch_models import sizes
from .run_grid import count, generate, items_for
from .score import score

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["check", "preflight", "generate", "score", "analyse", "pack"]


def load_config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text())
    for m in cfg["models"]:
        ref = cfg.get("reference_quant")
        if ref and ref in m["quants"]:  # reference first, so a partial run always has its baseline
            m["quants"] = [ref] + [q for q in m["quants"] if q != ref]
    return cfg


def stage_check(cfg: dict, out: Path) -> None:
    binary = cfg.get("llama_server") or os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
    if not binary:
        raise SystemExit("llama-server not found: put it on PATH, set LLAMA_SERVER, or set \"llama_server\" in the config")
    ver = subprocess.run([binary, "--version"], capture_output=True, text=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    n = count(cfg)
    meta = {"config": cfg, "machine": platform.platform(), "llama_cpp": (ver.stdout + ver.stderr).strip(),
            "gpu": gpu.stdout.strip() if gpu else "none", "generations_planned": n,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[check] llama-server: {meta['llama_cpp'].splitlines()[0] if meta['llama_cpp'] else binary}")
    print(f"[check] GPU: {meta['gpu']}   items: {len(items_for(cfg))}   generations planned: {n:,}")


def stage_preflight(cfg: dict, out: Path) -> None:
    pf = out / "preflight.json"
    if pf.exists():
        print(f"[preflight] already done: {json.loads(pf.read_text())['summary']}")
        return
    m = cfg["models"][0]
    small = dict(cfg, models=[dict(m, quants=[m["quants"][0]])],
                 items={c: 8 for c in cfg["items"]}, temperatures=[0, max(cfg["temperatures"])], samples=2, t0_repeats=1)
    t0 = time.time()
    gen = generate(small, out, gen_name="preflight.jsonl", keep_models=True)
    secs = time.time() - t0
    rows = [json.loads(line) for line in gen.read_text().splitlines()]
    failed = sum(r["error"] == "request_failed" for r in rows)
    calls = sum(bool(r["tool_calls"]) for r in rows if r["category"] != "irrelevance")
    relevant = sum(r["category"] != "irrelevance" for r in rows)
    per_gen = secs / max(len(rows), 1)
    # crude: assume time per generation grows with file size; good enough to decide whether it fits a night
    size = {mm["name"]: sizes(mm["repo"], mm["stem"], mm["quants"]) for mm in cfg["models"]}
    ref_size = size[m["name"]][m["quants"][0]]
    n_per_quant = count(dict(cfg, models=[dict(m, quants=["x"])]))
    est = sum(n_per_quant * per_gen * max(b / ref_size, 0.3) for mm in size.values() for b in mm.values())
    total_gb = sum(b for mm in size.values() for b in mm.values()) / 1e9
    peak_gb = max(b for mm in size.values() for b in mm.values()) / 1e9
    disk = (f"disk: {peak_gb:.0f} GB peak (one file at a time, deleted after use)" if cfg.get("delete_after_use")
            else f"disk: {total_gb:.0f} GB (all files kept)")
    summary = (f"{len(rows)} generations in {secs:.0f} s ({per_gen:.2f} s each incl. download and model load); "
               f"tool calls on {calls}/{relevant} non-irrelevance items; {failed} failed requests; "
               f"full run estimate {est / 3600:.1f} h (rough, upper bound); {disk}")
    log = out / f"server-{m['name']}-{m['quants'][0]}.log"
    offload = [ln for ln in log.read_text(errors="ignore").splitlines() if "offloaded" in ln and "layers" in ln]
    print(f"[preflight] GPU offload: {offload[-1].strip() if offload else 'no offload line in server log'}")
    pf.write_text(json.dumps({"summary": summary, "seconds": secs, "rows": len(rows)}, indent=1))
    print(f"[preflight] {summary}")
    if failed > len(rows) * 0.1 or calls == 0:
        raise SystemExit("[preflight] STOP: requests failing or no tool calls at all; check server log in "
                         f"{out} and the chat template (--jinja) before burning GPU hours")


def stage_score(out: Path) -> Path:
    items = {it["id"]: it for c in bfcl.CATEGORIES for it in bfcl.load(c)}
    src, dst = out / "generations.jsonl", out / "scored.jsonl"
    keep = ("key", "model", "quant", "item", "category", "temperature", "sample", "first_tokens",
            "completion_tokens", "wall_s", "finish_reason")
    n = 0
    with src.open(encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for line in fi:
            r = json.loads(line)
            if r.get("error") == "request_failed":
                continue
            v = score(items[r["item"]], r["tool_calls"])
            row = {k: r.get(k) for k in keep}
            row.update(v, n_calls=len(r["tool_calls"]))
            fo.write(json.dumps(row) + "\n")
            n += 1
    print(f"[score] {n:,} rows -> {dst}")
    return dst


def stage_analyse(out: Path) -> None:
    # Whatever state core/analysis.py is in, the raw results must still get packed, so nothing here is fatal.
    try:
        from core import analysis
        analysis.main(out)
        print(f"[analyse] report in {out / 'report'}")
    except NotImplementedError:
        print("[analyse] skipped: core/analysis.py is not written yet. Raw results are complete.")
    except Exception as e:
        print(f"[analyse] FAILED ({e!r}); raw results are complete and will still be packed.")


def stage_pack(out: Path) -> None:
    tar = out.parent / f"{out.name}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(out, arcname=out.name)
    print(f"[pack] {tar} ({tar.stat().st_size / 1e6:.0f} MB) -- send this file back")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "check" in a.stages:
        stage_check(cfg, out)
    if "preflight" in a.stages:
        stage_preflight(cfg, out)
    if "generate" in a.stages:
        generate(cfg, out)
    if "score" in a.stages:
        stage_score(out)
    if "analyse" in a.stages:
        stage_analyse(out)
    if "pack" in a.stages:
        stage_pack(out)
    print(f"[pipeline] finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
