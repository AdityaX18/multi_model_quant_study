"""Tests for the plumbing in harness/. They need no model and run in about a second."""

import json

import pytest

from harness import bfcl
from harness.fetch_models import files_for
from harness.run_grid import count, plan_for
from harness.score import score


def call(name, **args):
    return [{"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]


def test_loader_counts_and_schema_conversion():
    simple, irr = bfcl.load("simple"), bfcl.load("irrelevance")
    assert len(simple) == 400 and len(irr) == 240 and len(bfcl.load("multiple")) == 200
    params = simple[0]["tools"][0]["function"]["parameters"]
    assert params["type"] == "object" and params["properties"]["base"]["type"] == "integer"
    assert all("." not in t["function"]["name"] for it in simple for t in it["tools"])
    assert irr[0]["truth"] is None and simple[0]["truth"]


def test_scoring_simple_item():
    it = bfcl.load("simple", limit=1)[0]  # calculate_triangle_area(base=10, height=5, unit optional)
    assert score(it, call("calculate_triangle_area", base=10, height=5))["correct"]
    assert score(it, call("calculate_triangle_area", base=10, height=5, unit="Units"))["correct"]
    assert score(it, call("calculate_triangle_area", base=10.0, height=5))["correct"]
    assert score(it, call("calculate_triangle_area", base=10, height=6))["error"] == "wrong_args"
    assert score(it, call("calculate_triangle_area", base=10))["error"] == "wrong_args"
    assert score(it, call("calculate_triangle_area", base=10, height=5, colour="red"))["error"] == "wrong_args"
    assert score(it, call("area", base=10, height=5))["error"] == "wrong_function"
    assert score(it, [])["error"] == "no_call"
    assert score(it, call("calculate_triangle_area", base=10, height=5) * 2)["error"] == "extra_call"
    bad = [{"type": "function", "function": {"name": "calculate_triangle_area", "arguments": "{base: 10"}}]
    assert score(it, bad)["error"] == "malformed"


def test_scoring_dotted_names_and_irrelevance():
    it = bfcl.load("simple", limit=2)[1]  # math.factorial(number=5), exposed to the model as math_factorial
    assert score(it, call("math_factorial", number=5))["correct"]
    irr = bfcl.load("irrelevance", limit=1)[0]
    assert score(irr, [])["correct"]
    assert score(irr, call("determine_body_mass_index", weight=70, height=1.7))["error"] == "called_when_irrelevant"


def test_files_for_single_and_split():
    repo = ["a-q8_0.gguf", "a-fp16-00002-of-00002.gguf", "a-fp16-00001-of-00002.gguf", "README.md"]
    assert files_for(repo, "a", "q8_0") == ["a-q8_0.gguf"]
    assert files_for(repo, "a", "fp16") == ["a-fp16-00001-of-00002.gguf", "a-fp16-00002-of-00002.gguf"]
    with pytest.raises(FileNotFoundError):
        files_for(repo, "a", "q2_k")


def test_plan_and_count():
    cfg = {"temperatures": [0, 0.7, 1.0], "samples": 3, "t0_repeats": 2,
           "items": {"simple": 10, "irrelevance": 5}, "models": [{"quants": ["fp16", "q4_k_m"]}]}
    assert plan_for(cfg) == [(0, 0), (0, 1), (0.7, 0), (0.7, 1), (0.7, 2), (1.0, 0), (1.0, 1), (1.0, 2)]
    assert count(cfg) == 2 * 15 * 8
