"""A small scorer for single-turn BFCL items.

This is a re-implementation of the idea behind BFCL's AST checker, not the official one:
- irrelevance item: correct only if the model makes no tool call;
- simple/multiple item: exactly one call, right function, every argument value among the allowed
  values, no unknown arguments, and a missing argument is fine only when "" is an allowed value.
Strings are compared after lower-casing and dropping spaces and light punctuation, as BFCL does.
Before any number goes public, cross-check a sample against the official `bfcl-eval` checker (Day 2).
"""

from __future__ import annotations

import json
import re

_PUNCT = re.compile(r"[ \,\.\/\-\_\*\^]")


def _norm(v):
    if isinstance(v, str):
        return _PUNCT.sub("", v).lower().replace("'", '"')
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _allowed(value, options) -> bool:
    nv = _norm(value)
    for opt in options:
        if opt == "":
            continue
        if isinstance(opt, dict) and isinstance(value, dict):
            # nested dict: every key's value must be among that key's allowed values
            if all(k in opt and _allowed(v, opt[k] if isinstance(opt[k], list) else [opt[k]]) for k, v in value.items()):
                return True
        elif nv == _norm(opt):
            return True
    return False


def parse_calls(tool_calls: list[dict]) -> tuple[list[tuple[str, dict]], bool]:
    """OpenAI tool_calls -> [(name, args)], plus a flag for arguments that were not valid JSON."""
    out, malformed = [], False
    for c in tool_calls:
        f = c.get("function", {})
        args = f.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                malformed, args = True, {}
        out.append((f.get("name", ""), args if isinstance(args, dict) else {}))
    return out, malformed


def score(item: dict, tool_calls: list[dict]) -> dict:
    """Returns {"correct": bool, "error": one of none/called_when_irrelevant/no_call/extra_call/
    malformed/wrong_function/wrong_args}. `error` is the first failure found, in that order."""
    calls, malformed = parse_calls(tool_calls)
    if item["truth"] is None:
        return {"correct": not calls, "error": "none" if not calls else "called_when_irrelevant"}
    if not calls:
        return {"correct": False, "error": "no_call"}
    if len(calls) != len(item["truth"]):
        return {"correct": False, "error": "extra_call"}
    if malformed:
        return {"correct": False, "error": "malformed"}
    for (name, args), expected in zip(calls, item["truth"]):
        (true_name, allowed), = expected.items()
        if item["names"].get(name, name) != true_name:
            return {"correct": False, "error": "wrong_function"}
        for k, v in args.items():
            if k not in allowed or not _allowed(v, allowed[k]):
                return {"correct": False, "error": "wrong_args"}
        for k, options in allowed.items():
            if k not in args and "" not in options:
                return {"correct": False, "error": "wrong_args"}
    return {"correct": True, "error": "none"}
