"""LLM backends with tool calling, plus a deterministic response cache.

Two interchangeable backends expose ``chat(messages, tools) -> dict``:

``OllamaBackend``    the reference setup of the project (``ollama pull
                     qwen2.5:3b`` on a laptop).  Uses the ollama Python client.
``LlamaCppBackend``  runs a GGUF file in-process with llama-cpp-python.  Used
                     where no Ollama server is available.  It
                     renders the model's own chat template (stored in the
                     GGUF) with the tool schemas, and parses the
                     ``<tool_call>{json}</tool_call>`` blocks that Qwen-family
                     models emit — the same convention Ollama uses for them.

Both return ``{"content": str, "tool_calls": [{"name": str, "arguments": dict}],
"latency_s": float}``.

``CachedBackend`` wraps either one.  Calls are keyed by a hash of (backend,
model, messages, tools, options); with temperature 0 and a fixed seed a
re-run replays the stored answer and its *originally measured* latency, so
every figure can be regenerated without the model, and nothing is
re-measured by accident.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


class BackendError(RuntimeError):
    pass


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str, temperature: float = 0.0, seed: int = 0,
                 host: str | None = None, num_ctx: int = 8192) -> None:
        import ollama  # noqa: F401  (fail early if missing)
        self.model = model
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            resp = self._client.chat(model=self.model, messages=messages, tools=tools,
                                     options=self.options)
        except Exception as exc:                       # connection, model missing, …
            raise BackendError(str(exc)) from exc
        msg = resp.message
        calls = []
        for tc in msg.tool_calls or []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"__unparsed__": args}
            calls.append({"name": tc.function.name, "arguments": dict(args or {})})
        return {"content": msg.content or "", "tool_calls": calls,
                "latency_s": time.perf_counter() - t0}


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract ``<tool_call>{"name":…, "arguments":{…}}</tool_call>`` blocks."""
    calls = []
    for block in _TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(block)
            calls.append({"name": str(obj.get("name", "")),
                          "arguments": obj.get("arguments") or {}})
        except json.JSONDecodeError:
            calls.append({"name": "__malformed__", "arguments": {"raw": block[:500]}})
    content = _THINK_RE.sub("", _TOOL_CALL_RE.sub("", text)).strip()
    return content, calls


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, model_path: str, model: str | None = None, temperature: float = 0.0,
                 seed: int = 0, n_ctx: int = 8192, n_threads: int | None = None,
                 max_tokens: int = 384) -> None:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from llama_cpp import Llama
        self.model = model or Path(model_path).stem
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_threads=n_threads,
                         seed=seed, verbose=False)
        tpl = self.llm.metadata.get("tokenizer.chat_template")
        if not tpl:
            raise BackendError("GGUF file has no chat template")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
        self.template = env.from_string(tpl)

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        msgs = []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [{"type": "function",
                                    "function": {"name": c["function"]["name"],
                                                 "arguments": c["function"]["arguments"]}}
                                   for c in m["tool_calls"]]
            msgs.append(m)
        # Qwen3 templates accept enable_thinking; other templates ignore it.
        return self.template.render(messages=msgs, tools=tools, add_generation_prompt=True,
                                    enable_thinking=False)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self.render(messages, tools)
        t0 = time.perf_counter()
        try:
            out = self.llm.create_completion(prompt, max_tokens=self.max_tokens,
                                             temperature=self.temperature, seed=self.seed,
                                             stop=["<|im_end|>", "<|endoftext|>"])
        except Exception as exc:
            raise BackendError(str(exc)) from exc
        text = out["choices"][0]["text"]
        content, calls = parse_tool_calls(text)
        return {"content": content, "tool_calls": calls,
                "latency_s": time.perf_counter() - t0,
                "prompt_tokens": out["usage"]["prompt_tokens"],
                "completion_tokens": out["usage"]["completion_tokens"]}


class CachedBackend:
    """Replay identical calls from a JSONL cache (see module docstring)."""

    def __init__(self, inner: Any, cache_dir: str | Path, backend_name: str, model: str,
                 options: dict[str, Any]) -> None:
        self.inner = inner
        self.model = model
        self.backend_name = backend_name
        self.options = options
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{backend_name}_{model}")
        self.path = Path(cache_dir) / f"{slug}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.store[rec["key"]] = rec["response"]
        self.hits = 0
        self.misses = 0

    def key(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        blob = json.dumps({"b": self.backend_name, "m": self.model, "o": self.options,
                           "msgs": messages, "tools": tools}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        k = self.key(messages, tools)
        if k in self.store:
            self.hits += 1
            return dict(self.store[k], cached=True)
        if self.inner is None:
            raise BackendError("cache miss and no live backend (replay-only mode)")
        resp = self.inner.chat(messages, tools)
        self.misses += 1
        self.store[k] = resp
        with self.path.open("a") as f:
            f.write(json.dumps({"key": k, "response": resp}, default=str) + "\n")
        return dict(resp, cached=False)


_LIVE: dict[tuple, Any] = {}     # one model instance per process (keeps llama.cpp's prefix cache)


def make_backend(agent_cfg: dict[str, Any], replay_only: bool = False) -> CachedBackend:
    kind = agent_cfg["backend"]
    opts = {"temperature": agent_cfg["temperature"], "seed": agent_cfg["seed"]}
    model = agent_cfg["model"]
    inner = None
    if not replay_only:
        key = (kind, model, agent_cfg.get("model_path"), opts["temperature"], opts["seed"])
        if key not in _LIVE:
            if kind == "ollama":
                _LIVE[key] = OllamaBackend(model, **opts)
            elif kind == "llamacpp":
                _LIVE[key] = LlamaCppBackend(agent_cfg["model_path"], model=model,
                                             n_threads=agent_cfg.get("n_threads"), **opts)
            else:
                raise ValueError(f"unknown backend {kind}")
        inner = _LIVE[key]
    cache_dir = Path(agent_cfg["cache_dir"])
    if not cache_dir.is_absolute():                 # relative to the repository root
        cache_dir = Path(__file__).resolve().parents[2] / cache_dir
    return CachedBackend(inner, cache_dir, kind, model, opts)
