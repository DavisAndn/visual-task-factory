"""Model clients for `vtf models`, plus a per-run answer cache.

- Claude, through the official Anthropic SDK; the key comes from ANTHROPIC_API_KEY.
- GPT models, through OpenAI's API (OPENAI_API_KEY), and open-weight models served by Cerebras
  (CEREBRAS_API_KEY), both through the official openai SDK, since Cerebras's API is
  OpenAI-compatible. Model ids "gpt-..." and "openai/<id>" go to OpenAI, "cerebras/<id>" to
  Cerebras, and anything else to Anthropic.

`vtf models` also reads the keys from a .env file in the current folder. Every model gets the same
system prompt and the same question, with the image before it; `effort` sets how hard the models
that can think about it do.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anthropic
import openai

from .validators.models import ClientConfigError, ModelClient, Reply

DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
SYSTEM = (
    "You answer questions about images. Work the answer out carefully, then reply with only "
    "the final answer, with no explanation."
)
# Models that accept the server-side refusal fallback. If one declines a request (very unlikely
# for a chart question), the API retries it on a fallback model; Reply.model records who answered.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# USD per million input and output tokens, for the cost line in the report. A reply's model is
# priced by the longest name it starts with, so dated snapshots ("gpt-...-2026-10-01") match.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    # List prices from November 2026 pricing trackers; check the provider's own page before quoting.
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.2-sol": (2.00, 10.00),
    "qwen-3.8-27b": (0.99, 1.49),
}

# Providers reached through the openai SDK: where requests go and which key they need.
OPENAI_COMPATIBLE = {
    "openai": (None, "OPENAI_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
}


def cost_usd(reply: Reply) -> float:
    names = [name for name in PRICES if reply.model.startswith(name)]
    price_in, price_out = PRICES[max(names, key=len)] if names else (0.0, 0.0)
    return (reply.input_tokens * price_in + reply.output_tokens * price_out) / 1e6


def provider_of(model: str) -> str:
    """'anthropic', 'openai' or 'cerebras', from a model id as `vtf models -m` takes it."""
    prefix, _, rest = model.partition("/")
    if rest and prefix in OPENAI_COMPATIBLE:
        return prefix
    return "openai" if model.startswith("gpt-") else "anthropic"


def key_present(model: str) -> str | None:
    """None if the key a model needs is set, else the name of the missing variable."""
    provider = provider_of(model)
    if provider == "anthropic":
        found = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        return None if found else "ANTHROPIC_API_KEY"
    variable = OPENAI_COMPATIBLE[provider][1]
    return None if os.environ.get(variable) else variable


def make_client(model: str, effort: str = "high") -> ModelClient:
    if provider_of(model) == "anthropic":
        return ClaudeClient(model, effort)
    return ChatCompletionsClient(model, effort)


def _image_part(image_path: Path) -> tuple[str, str]:
    data = base64.standard_b64encode(Path(image_path).read_bytes()).decode()
    kind = "image/jpeg" if Path(image_path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return kind, data


class ClaudeClient:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "high", max_retries: int = 4) -> None:
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
        self.model = model
        self.effort = effort
        self.name = f"{model} ({effort} effort)"
        self.signature = f"{model}|{effort}|{hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]}"
        self._client = anthropic.Anthropic(max_retries=max_retries)

    def request(self, prompt: str, image_path: Path | None) -> dict[str, Any]:
        """Keyword arguments for the Messages API call. The image goes before the question."""
        content: list[dict[str, Any]] = []
        if image_path is not None:
            kind, data = _image_part(image_path)
            content.append({"type": "image", "source": {"type": "base64", "media_type": kind, "data": data}})
        content.append({"type": "text", "text": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,  # thinking counts toward it; a low cap would cut answers off
            "system": SYSTEM,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": self.effort},
        }
        if self.model in FALLBACK_MODELS:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        return kwargs

    def ask(self, prompt: str, image_path: Path | None) -> Reply:
        kwargs = self.request(prompt, image_path)
        try:
            if "betas" in kwargs:
                response = self._client.beta.messages.create(**kwargs)
            else:
                response = self._client.messages.create(**kwargs)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as err:
            raise ClientConfigError(f"The API key was rejected: {err.message}") from err
        except anthropic.NotFoundError as err:
            raise ClientConfigError(f"Unknown model {self.model!r}: {err.message}") from err
        except anthropic.BadRequestError as err:
            raise ClientConfigError(f"The API rejected the request: {err.message}") from err
        text = ""
        if response.stop_reason != "refusal":
            text = "".join(block.text for block in response.content if block.type == "text")
        return Reply(
            text=text.strip(),
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class ChatCompletionsClient:
    """A GPT model from OpenAI, or an open-weight model served by Cerebras, through the openai SDK's
    Chat Completions API. GPT models get `effort` as their reasoning effort and see the image at
    high detail; the open models run as their provider serves them.
    """

    def __init__(self, model: str, effort: str = "high", max_retries: int = 8) -> None:
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
        self.provider = provider_of(model)
        if self.provider == "anthropic":
            raise ValueError(f"{model!r} is a Claude model; use ClaudeClient")
        self.model = model
        self.api_model = model.partition("/")[2] if "/" in model else model
        self.effort = effort if self.provider == "openai" else None
        self.name = f"{model} ({effort} effort)" if self.effort else model
        prompt_hash = hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]
        self.signature = f"{self.provider}|{self.api_model}|{self.effort or '-'}|{prompt_hash}"
        base_url, variable = OPENAI_COMPATIBLE[self.provider]
        # Rate limits are tight on some plans: the SDK waits and retries on 429s, honouring Retry-After.
        self._client = openai.OpenAI(
            api_key=os.environ.get(variable) or "missing", base_url=base_url, max_retries=max_retries, timeout=900
        )

    def request(self, prompt: str, image_path: Path | None) -> dict[str, Any]:
        """Keyword arguments for the Chat Completions call. The image goes before the question."""
        content: list[dict[str, Any]] = []
        if image_path is not None:
            kind, data = _image_part(image_path)
            image: dict[str, Any] = {"url": f"data:{kind};base64,{data}"}
            if self.provider == "openai":
                image["detail"] = "high"
            content.append({"type": "image_url", "image_url": image})
        content.append({"type": "text", "text": prompt})
        kwargs: dict[str, Any] = {
            "model": self.api_model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            "max_completion_tokens": 16000,  # reasoning counts toward it, as Claude's thinking does
        }
        if self.effort:
            kwargs["reasoning_effort"] = self.effort
        return kwargs

    def ask(self, prompt: str, image_path: Path | None) -> Reply:
        try:
            response = self._client.chat.completions.create(**self.request(prompt, image_path))
        except (openai.AuthenticationError, openai.PermissionDeniedError) as err:
            raise ClientConfigError(f"The {self.provider} API key was rejected: {err.message}") from err
        except openai.NotFoundError as err:
            raise ClientConfigError(f"Unknown model {self.model!r}: {err.message}") from err
        except openai.BadRequestError as err:
            raise ClientConfigError(f"The {self.provider} API rejected the request: {err.message}") from err
        choice = response.choices[0]
        text, stop = choice.message.content or "", choice.finish_reason
        if getattr(choice.message, "refusal", None):
            text, stop = "", "refusal"
        usage = response.usage
        return Reply(
            text=text.strip(),
            model=response.model or self.api_model,
            stop_reason=stop,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


class CachedClient:
    """Answers repeated requests from a JSONL cache, so re-running a check costs nothing.

    The key covers the client's model, effort and system prompt, the prompt, and the image bytes.
    """

    def __init__(self, inner: ModelClient, path: Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                record = json.loads(line)
                self._entries[record["key"]] = record["reply"]

    def key(self, prompt: str, image_path: Path | None) -> str:
        digest = hashlib.sha256(getattr(self.inner, "signature", self.inner.name).encode())
        digest.update(b"\0" + prompt.encode() + b"\0")
        if image_path is not None:
            digest.update(Path(image_path).read_bytes())
        return digest.hexdigest()

    def ask(self, prompt: str, image_path: Path | None) -> Reply:
        key = self.key(prompt, image_path)
        with self._lock:
            hit = self._entries.get(key)
        if hit is not None:
            return Reply(**{**hit, "cached": True})
        reply = self.inner.ask(prompt, image_path)
        record = {"key": key, "reply": asdict(reply)}
        with self._lock:
            self._entries[key] = record["reply"]
            with self.path.open("a") as cache:
                cache.write(json.dumps(record) + "\n")
        return reply
