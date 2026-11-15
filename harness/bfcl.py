"""Load BFCL v4 single-turn items and turn them into OpenAI-style tool definitions.

BFCL (Berkeley Function Calling Leaderboard) stores Python-flavoured schemas ("dict", "float", "tuple")
and function names with dots ("math.factorial"). OpenAI-style tools need JSON Schema types and names
without dots, so both are converted here and the name mapping is kept for scoring.
Data: https://github.com/ShishirPatil/gorilla (Apache-2.0), copied into data/bfcl/.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "bfcl"
CATEGORIES = {"simple": "BFCL_v4_simple_python.json", "multiple": "BFCL_v4_multiple.json",
              "irrelevance": "BFCL_v4_irrelevance.json"}
_TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string", "integer": "integer",
          "boolean": "boolean", "string": "string", "array": "array", "number": "number", "object": "object"}


def _schema(node):
    """Recursively convert a BFCL parameter schema to JSON Schema."""
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, str):
            out[k] = _TYPES.get(v.lower(), "string")
        elif k == "properties" and isinstance(v, dict):
            out[k] = {name: _schema(sub) for name, sub in v.items()}
        elif k in ("items", "additionalProperties"):
            out[k] = _schema(v)
        else:
            out[k] = v
    return out


def safe_name(name: str) -> str:
    return name.replace(".", "_")


def load(category: str, limit: int | None = None) -> list[dict]:
    """Items as {id, category, messages, tools, names, truth}. truth is None for irrelevance items."""
    path = DATA / CATEGORIES[category]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    truth_path = DATA / "possible_answer" / CATEGORIES[category]
    truth = {}
    if truth_path.exists():
        truth = {r["id"]: r["ground_truth"] for r in
                 (json.loads(line) for line in truth_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    items = []
    for r in rows[:limit]:
        tools = [{"type": "function", "function": {"name": safe_name(f["name"]), "description": f.get("description", ""),
                                                    "parameters": _schema(f.get("parameters", {}))}}
                 for f in r["function"]]
        items.append({"id": r["id"], "category": category, "messages": r["question"][0], "tools": tools,
                      "names": {safe_name(f["name"]): f["name"] for f in r["function"]},
                      "required": {f["name"]: f.get("parameters", {}).get("required", []) for f in r["function"]},
                      "truth": truth.get(r["id"])})
    return items
