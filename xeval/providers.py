"""Anthropic, OpenAI and Google behind one call.

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
    # Gemini runs use the free tier; cost is what the same tokens would cost
    # at the paid list price (text/image input). Cached input is priced as
    # fresh input, since the runs don't use context caching.
    "gemini-3.1-flash-lite": (0.25, 0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 0.30, 2.50),
    "gemini-3.6-flash": (0.75, 0.75, 3.75),
    "gemini-3.8-flash": (0.75, 0.75, 3.75),
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
    if model.startswith("claude"):
        return "anthropic"
    return "google" if model.startswith("gemini") else "openai"


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


# Free-tier requests per minute, per model. Google shows the current numbers
# only in AI Studio; these were read from the 429 responses on 2026-10-29.
GEMINI_RPM = {"gemini-3.1-flash-lite": 15, "gemini-3.5-flash-lite": 15, "gemini-3.6-flash": 5, "gemini-3.8-flash": 5}


class DailyQuotaExceeded(Exception):
    """The free tier's requests-per-day limit: nothing more today."""


class Google:
    """Gemini through the google-genai SDK, paced to the free tier's
    requests-per-minute limit. Rate-limit and overload errors are retried
    with the server's suggested delay; a daily-quota error stops the run."""

    def __init__(self):
        import threading
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(http_options=types.HttpOptions(timeout=600_000))
        self.lock = threading.Lock()
        self.next_slot: dict[str, float] = {}

    def _wait_turn(self, model: str):
        gap = 60.0 / GEMINI_RPM.get(model, 10)
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot.get(model, now))
            self.next_slot[model] = slot + gap
        if slot > now:
            time.sleep(slot - now)

    def _parts(self, parts):
        t = self.types
        out = []
        for p in parts:
            if p["type"] == "image":
                mt = mimetypes.guess_type(p["path"])[0] or "image/png"
                with open(p["path"], "rb") as f:
                    out.append(t.Part.from_bytes(data=f.read(), mime_type=mt))
            else:
                out.append(t.Part.from_text(text=p["text"]))
        return out

    def call(self, model: str, system: str, turns: list[dict], schema: dict | None) -> Response:
        import re
        from google.genai import errors
        t = self.types
        contents = [t.Content(role="model", parts=[t.Part.from_text(text=x["text"])]) if x["role"] == "assistant"
                    else t.Content(role="user", parts=self._parts(x["parts"])) for x in turns]
        cfg = dict(system_instruction=system, max_output_tokens=16000)
        if schema is not None:
            cfg |= dict(response_mime_type="application/json", response_json_schema=schema)
        for attempt in range(8):
            self._wait_turn(model)
            t0 = time.perf_counter()
            try:
                r = self.client.models.generate_content(model=model, contents=contents, config=t.GenerateContentConfig(**cfg))
                break
            except errors.APIError as e:
                msg = str(e)
                if e.code == 429 and "PerDay" in msg:
                    raise DailyQuotaExceeded(f"{model}: daily quota reached ({', '.join(re.findall(r"'quotaId': '([^']+)'", msg))})") from e
                if e.code not in (429, 500, 503, 504) or attempt == 7:
                    raise
                m = re.search(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'", msg)
                delay = float(next(g for g in m.groups() if g)) if m else min(2 ** attempt * 5, 120)
                time.sleep(delay + 1)
        dt = time.perf_counter() - t0
        u = r.usage_metadata
        thoughts = u.thoughts_token_count or 0
        cand = r.candidates[0] if r.candidates else None
        return Response(
            text=r.text or "", input_tokens=u.prompt_token_count or 0, cached_tokens=u.cached_content_token_count or 0,
            output_tokens=(u.candidates_token_count or 0) + thoughts, reasoning_tokens=thoughts, latency_s=dt,
            stop_reason=str(cand.finish_reason.name if cand and cand.finish_reason else ""), model_id=r.model_version or model,
        )


_CLIENTS: dict[str, object] = {}


def client(model: str):
    p = provider_of(model)
    if p not in _CLIENTS:
        _CLIENTS[p] = {"anthropic": Anthropic, "openai": OpenAI, "google": Google}[p]()
    return _CLIENTS[p]
