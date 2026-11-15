"""Download official GGUF quantizations from Hugging Face, including files split into parts.

Called by the pipeline for every (repo, stem, quant) in the config. Can also be run by hand:

  python -m harness.fetch_models --repo Qwen/Qwen2.5-1.5B-Instruct-GGUF --stem qwen2.5-1.5b-instruct \\
      --quants fp16 q8_0 q4_k_m

Larger models are stored in parts ("...-q8_0-00001-of-00003.gguf"). llama-server loads a split model
when given the FIRST part, so `fetch` downloads every part and returns the path of the first one.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def files_for(repo_files: list[str], stem: str, quant: str) -> list[str]:
    """The GGUF file(s) for one quant level, first part first. Pure function, so it is unit-tested."""
    single = f"{stem}-{quant}.gguf"
    if single in repo_files:
        return [single]
    part = re.compile(re.escape(f"{stem}-{quant}-") + r"(\d{5})-of-(\d{5})\.gguf$")
    parts = sorted(f for f in repo_files if part.match(f))
    if not parts:
        raise FileNotFoundError(f"no file for {stem} {quant}; available: {sorted(f for f in repo_files if f.endswith('.gguf'))}")
    return parts


def fetch(repo: str, stem: str, quant: str, dest: Path = MODELS) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    dest.mkdir(parents=True, exist_ok=True)
    names = files_for(list_repo_files(repo), stem, quant)
    paths = [Path(hf_hub_download(repo, n, local_dir=dest)) for n in names]
    return paths[0]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--stem", required=True, help="file name before the quant suffix, e.g. qwen2.5-3b-instruct")
    ap.add_argument("--quants", nargs="+", required=True)
    a = ap.parse_args(argv)
    for q in a.quants:
        p = fetch(a.repo, a.stem, q)
        base = re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", p.name)
        size = sum(f.stat().st_size for f in p.parent.glob(base + "*.gguf"))
        print(f"{q}: {p} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
