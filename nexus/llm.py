"""Model-independent gateway. Callers ask for chat, streaming or embeddings; providers stay behind this file.

Every call is metered into usage_events. Prices are USD per million tokens; a model with no known price
is recorded as cost NULL ("unpriced"), never as zero.
"""
import base64
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import httpx
import numpy as np

from voicerag.embeddings import HashingEmbedder

from . import cache
from .config import keys, settings
from .db import SessionLocal, UsageEvent

log = logging.getLogger("nexus.llm")

PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


class LLMUnavailable(RuntimeError):
    """No provider is configured, or every provider failed."""


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""


def price(model: str) -> tuple[float, float] | None:
    for item in filter(None, settings().price_overrides.split(",")):
        name, _, rates = item.partition("=")
        if name.strip() == model:
            a, _, b = rates.partition(":")
            return float(a), float(b or 0)
    return PRICES.get(model)


def record(workspace_id: str | None, purpose: str, provider: str, model: str, inp: int, out: int) -> None:
    if not workspace_id:
        return
    rates = price(model)
    cost = (inp * rates[0] + out * rates[1]) / 1e6 if rates else None
    try:
        with SessionLocal() as db:
            db.add(UsageEvent(workspace_id=workspace_id, purpose=purpose, provider=provider, model=model,
                              input_tokens=inp, output_tokens=out, cost_usd=cost))
            db.commit()
    except Exception:
        log.exception("usage metering failed")


class OpenAIProvider:
    """Any OpenAI-compatible endpoint."""
    name = "openai"

    def __init__(self):
        k = keys()
        self.model = k.openai_model
        self.base = k.openai_base_url.rstrip("/")
        self.client = httpx.Client(timeout=settings().llm_timeout, headers={"Authorization": f"Bearer {k.openai_api_key}"})

    def chat(self, system: str, messages: list[dict], max_tokens: int, temperature: float, json_mode: bool) -> Completion:
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, *messages]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self.client.post(f"{self.base}/chat/completions", json=body)
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return Completion(d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0),
                          u.get("completion_tokens", 0), self.model, self.name)

    def stream(self, system: str, messages: list[dict], max_tokens: int, temperature: float, usage: dict) -> Iterator[str]:
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "system", "content": system}, *messages]}
        with self.client.stream("POST", f"{self.base}/chat/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                d = json.loads(line[6:])
                if d.get("usage"):
                    usage.update(input=d["usage"].get("prompt_tokens", 0), output=d["usage"].get("completion_tokens", 0))
                for choice in d.get("choices") or []:
                    if tok := choice.get("delta", {}).get("content"):
                        yield tok

    def vision(self, prompt: str, image: bytes, mime: str) -> Completion:
        url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        return self.chat("Describe images accurately and transcribe any text.",
                         [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                       {"type": "image_url", "image_url": {"url": url}}]}],
                         800, 0, False)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        import anthropic

        k = keys()
        self.model = k.voicerag_model
        self.client = anthropic.Anthropic(api_key=k.anthropic_api_key, timeout=settings().llm_timeout)

    def chat(self, system: str, messages: list[dict], max_tokens: int, temperature: float, json_mode: bool) -> Completion:
        if json_mode:
            system += "\nRespond with a single JSON object and nothing else."
        m = self.client.messages.create(model=self.model, max_tokens=max_tokens, system=system, messages=messages)
        return Completion("".join(b.text for b in m.content if b.type == "text"), m.usage.input_tokens,
                          m.usage.output_tokens, self.model, self.name)

    def stream(self, system: str, messages: list[dict], max_tokens: int, temperature: float, usage: dict) -> Iterator[str]:
        with self.client.messages.stream(model=self.model, max_tokens=max_tokens, system=system, messages=messages) as s:
            yield from s.text_stream
            u = s.get_final_message().usage
            usage.update(input=u.input_tokens, output=u.output_tokens)

    def vision(self, prompt: str, image: bytes, mime: str) -> Completion:
        data = base64.b64encode(image).decode()
        return self.chat("Describe images accurately and transcribe any text.", [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}},
            {"type": "text", "text": prompt}]}], 800, 0, False)


@lru_cache
def providers() -> tuple:
    k, chosen = keys(), settings().llm_provider
    available = {}
    if k.openai_api_key:
        available["openai"] = OpenAIProvider
    if k.anthropic_api_key:
        available["anthropic"] = AnthropicProvider
    order = sorted(available, key=lambda n: n != chosen)
    return tuple(available[n]() for n in order)


def available() -> bool:
    return bool(providers())


def chat(system: str, user: str | list[dict], *, purpose: str, workspace_id: str | None = None, max_tokens: int = 800,
         temperature: float = 0.1, json_mode: bool = False) -> Completion:
    """Tries each configured provider in order; raises LLMUnavailable when none can answer."""
    messages = [{"role": "user", "content": user}] if isinstance(user, str) else user
    last = None
    for p in providers():
        try:
            c = p.chat(system, messages, max_tokens, temperature, json_mode)
            record(workspace_id, purpose, c.provider, c.model, c.input_tokens, c.output_tokens)
            return c
        except Exception as e:
            log.warning("%s failed for %s: %s", p.name, purpose, e)
            last = e
    raise LLMUnavailable(f"No language model available ({last})" if last else "No language model is configured")


def chat_json(system: str, user: str, *, purpose: str, workspace_id: str | None = None, max_tokens: int = 1000) -> dict:
    c = chat(system, user, purpose=purpose, workspace_id=workspace_id, max_tokens=max_tokens, temperature=0, json_mode=True)
    m = re.search(r"\{.*\}", c.text, re.S)
    if not m:
        raise ValueError("model did not return JSON")
    return json.loads(m.group(0))


def stream(system: str, messages: list[dict], *, purpose: str, workspace_id: str | None = None,
           max_tokens: int = 900, temperature: float = 0.2) -> Iterator[str]:
    """Falls over to the next provider only if the failure happens before the first token."""
    last = None
    for p in providers():
        usage: dict = {}
        started = False
        try:
            for tok in p.stream(system, messages, max_tokens, temperature, usage):
                started = True
                yield tok
            record(workspace_id, purpose, p.name, p.model, usage.get("input", 0), usage.get("output", 0))
            return
        except Exception as e:
            log.warning("%s stream failed: %s", p.name, e)
            last = e
            if started:
                raise
    raise LLMUnavailable(f"No language model available ({last})" if last else "No language model is configured")


def vision(prompt: str, image: bytes, mime: str, workspace_id: str | None) -> str:
    for p in providers():
        try:
            c = p.vision(prompt, image, mime)
            record(workspace_id, "vision", c.provider, c.model, c.input_tokens, c.output_tokens)
            return c.text
        except Exception as e:
            log.warning("%s vision failed: %s", p.name, e)
    raise LLMUnavailable("Reading images needs a vision-capable model")


# ---- embeddings ---------------------------------------------------------------------------------
class OpenAIEmbedder:
    name = "openai"
    semantic = True

    def __init__(self):
        k = keys()
        self.model = settings().embedding_model
        self.dim = {"text-embedding-3-large": 3072}.get(self.model, 1536)
        self.url = k.openai_base_url.rstrip("/") + "/embeddings"
        self.client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {k.openai_api_key}"})

    def encode(self, texts: list[str], workspace_id: str | None = None) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 96):
            r = self.client.post(self.url, json={"model": self.model, "input": [t[:8000] for t in texts[i:i + 96]]})
            r.raise_for_status()
            d = r.json()
            out += [e["embedding"] for e in sorted(d["data"], key=lambda e: e["index"])]
            record(workspace_id, "embedding", "openai", self.model, d.get("usage", {}).get("prompt_tokens", 0), 0)
        return np.array(out, dtype=np.float32)


class LocalEmbedder:
    name = "local"
    semantic = True

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = settings().local_embedding_model
        self.net = SentenceTransformer(self.model)
        self.dim = self.net.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], workspace_id: str | None = None) -> np.ndarray:
        return self.net.encode(list(texts), normalize_embeddings=True).astype(np.float32)


class HashEmbedder:
    """No download, no key, lexical only: it cannot match paraphrases or across languages."""
    name = "hashing"
    semantic = False
    model = "hashing-2048"
    dim = 2048

    def __init__(self):
        self.impl = HashingEmbedder(self.dim)

    def encode(self, texts: list[str], workspace_id: str | None = None) -> np.ndarray:
        return self.impl.encode(list(texts))


@lru_cache
def embedder():
    choice = settings().embedding_provider or ("openai" if keys().openai_api_key else "hashing")
    return {"openai": OpenAIEmbedder, "local": LocalEmbedder, "hashing": HashEmbedder}[choice]()


def embed_query(text: str, workspace_id: str | None = None) -> np.ndarray:
    e = embedder()
    key = f"emb:{e.model}:{text}"
    hit = cache.get(key)
    if hit is not None:
        return np.frombuffer(base64.b64decode(hit), dtype=np.float32)
    v = e.encode([text], workspace_id)[0]
    cache.set(key, base64.b64encode(v.tobytes()).decode(), ttl=86400)
    return v
