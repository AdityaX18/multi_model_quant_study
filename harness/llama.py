"""Start and stop a llama.cpp server for one GGUF file, and send it chat requests.

llama-server speaks the OpenAI chat-completions format, so the same client code works against vLLM or
Ollama later. `--jinja` turns on the model's own chat template, which is what makes tool calls work.

`parallel` > 1 opens that many slots: the server batches requests from several threads together,
which is how a big GPU stays busy. llama-server divides the context between slots, so the total
context passed with -c is ctx * parallel.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class LlamaServer:
    def __init__(self, model: str | Path, port: int = 8089, ctx: int = 4096, threads: int | None = None,
                 gpu_layers: int = 0, parallel: int = 1, binary: str | None = None,
                 extra_args: list[str] | None = None, log: str | Path | None = None):
        self.model = Path(model)
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.binary = binary or os.environ.get("LLAMA_SERVER", "llama-server")
        self.args = [self.binary, "-m", str(self.model), "--port", str(port), "-c", str(ctx * parallel),
                     "--jinja", "-ngl", str(gpu_layers), "--no-webui", "-np", str(parallel)]
        if threads:
            self.args += ["-t", str(threads)]
        self.args += list(extra_args or [])
        self.log = Path(log) if log else None
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "LlamaServer":
        out = open(self.log, "w") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.args, stdout=out, stderr=subprocess.STDOUT)
        deadline = time.time() + 600  # a 15 GB fp16 file can take minutes to load from a slow disk
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self.proc.returncode}); see {self.log}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        return self
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                time.sleep(0.5)
        raise TimeoutError("llama-server did not become healthy in 600 s")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0,
             seed: int = 0, max_tokens: int = 256, top_logprobs: int = 0, timeout: int = 600) -> dict:
        """One completion. Returns the message, timing from the server's own counters and, if asked,
        the top alternatives for the first generated tokens (used for the decision-margin analysis)."""
        body = {"model": self.model.name, "messages": messages, "temperature": temperature, "seed": seed,
                "max_tokens": max_tokens, "cache_prompt": False}
        if tools:
            body["tools"] = tools
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        wall = time.perf_counter() - t0
        choice = d["choices"][0]
        msg = choice["message"]
        tm = d.get("timings", {})
        out = {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or [],
               "finish_reason": choice.get("finish_reason"), "wall_s": round(wall, 4),
               "prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
               "completion_tokens": d.get("usage", {}).get("completion_tokens"),
               "prompt_ms": tm.get("prompt_ms"), "predicted_ms": tm.get("predicted_ms"),
               "tokens_per_s": tm.get("predicted_per_second")}
        lp = (choice.get("logprobs") or {}).get("content") if top_logprobs else None
        if lp:
            # keep only the first few tokens: the call-or-not decision is made there, and files stay small
            out["first_tokens"] = [{"t": x.get("token"), "lp": round(x.get("logprob", 0.0), 4),
                                    "alt": [[a.get("token"), round(a.get("logprob", 0.0), 4)] for a in x.get("top_logprobs", [])]}
                                   for x in lp[:4]]
        return out
