"""Anthropic and OpenAI behind one call.

A turn is a list of parts: {"type": "text", "text": ...} or
{"type": "image", "path": ...}. Every request uses the model's default
settings: no temperature (the current Anthropic API rejects it, and the OpenAI
reasoning models ignore it), default thinking and effort. Repeated samples
therefore differ the way they would in production.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass, field

PRICES = {
    # USD per million tokens: input, cached input, output. List prices on
    # 2026-10-29, from the providers' pricing pages.
    "claude-haiku-5-5": (0.10, 0.01, 0.50),
    "claude-sonnet-5-5": (2.00, 0.20, 10.00),
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
}
PRICE_DATE = "2026-10-29"


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: str = ""
    model_id: str = ""
    logprobs: list[tuple[str, float]] | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def cost(self, model: str) -> float:
        p_in, p_cache, p_out = PRICES[model]
        fresh = max(self.input_tokens - self.cached_tokens, 0)
        return (fresh * p_in + self.cached_tokens * p_cache + self.output_tokens * p_out) / 1e6


def provider_of(model: str) -> str:
    return "anthropic" if model.startswith("claude") else "openai"


def _b64(path: str) -> tuple[str, str]:
    mt = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return mt, base64.standard_b64encode(f.read()).decode()


class Anthropic:
    def __init__(self):
        import anthropic
        self.sdk = anthropic
        self.client = anthropic.Anthropic(max_retries=6, timeout=600)

    def _content(self, parts):
        out = []
        for p in parts:
            if p["type"] == "image":
                mt, data = _b64(p["path"])
                out.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
            else:
                out.append({"type": "text", "text": p["text"]})
        return out

    def call(self, model: str, system: str, turns: list[dict], schema: dict | None) -> Response:
        msgs = []
        for t in turns:
            if t["role"] == "assistant":
                # Thinking blocks from the first turn are sent back unchanged.
                msgs.append({"role": "assistant", "content": t.get("raw_content") or t["text"]})
            else:
                msgs.append({"role": "user", "content": self._content(t["parts"])})
        kw = {}
        if schema is not None:
            kw["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        t0 = time.perf_counter()
        r = self.client.messages.create(model=model, max_tokens=16000, system=system, messages=msgs, **kw)
        dt = time.perf_counter() - t0
        text = "".join(b.text for b in r.content if b.type == "text")
        u = r.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        return Response(
            text=text,
            input_tokens=u.input_tokens + cache_read + cache_write,
            cached_tokens=cache_read,
            output_tokens=u.output_tokens,
            latency_s=dt, stop_reason=r.stop_reason or "", model_id=r.model,
            extra={"raw_content": [b.model_dump(exclude_none=True) for b in r.content]},
        )


class OpenAI:
    def __init__(self):
        import openai
        self.client = openai.OpenAI(max_retries=6, timeout=600)

    def _content(self, parts):
        out = []
        for p in parts:
            if p["type"] == "image":
                mt, data = _b64(p["path"])
                out.append({"type": "input_image", "image_url": f"data:{mt};base64,{data}", "detail": "high"})
            else:
                out.append({"type": "input_text", "text": p["text"]})
        return out

    def call(self, model: str, system: str, turns: list[dict], schema: dict | None, logprobs: bool = False) -> Response:
        inp = [{"role": "system", "content": system}]
        for t in turns:
            if t["role"] == "assistant":
                inp.append({"role": "assistant", "content": t["text"]})
            else:
                inp.append({"role": "user", "content": self._content(t["parts"])})
        kw = {}
        if schema is not None:
            kw["text"] = {"format": {"type": "json_schema", "name": "extraction", "schema": schema, "strict": True}}
        if logprobs:
            kw["include"] = ["message.output_text.logprobs"]
        t0 = time.perf_counter()
        r = self.client.responses.create(model=model, input=inp, max_output_tokens=16000, **kw)
        dt = time.perf_counter() - t0
        text = r.output_text or ""
        lps = None
        if logprobs:
            lps = []
            for it in r.output:
                if it.type == "message":
                    for c in it.content:
                        for lp in (getattr(c, "logprobs", None) or []):
                            lps.append((lp.token, lp.logprob))
        u = r.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        return Response(
            text=text, input_tokens=u.input_tokens, cached_tokens=cached, output_tokens=u.output_tokens,
            reasoning_tokens=reasoning, latency_s=dt, stop_reason=r.status or "", model_id=r.model, logprobs=lps,
        )


_CLIENTS: dict[str, object] = {}


def client(model: str):
    p = provider_of(model)
    if p not in _CLIENTS:
        _CLIENTS[p] = Anthropic() if p == "anthropic" else OpenAI()
    return _CLIENTS[p]
