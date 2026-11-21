"""Claude as a ModelClient, through the official Anthropic SDK, plus a per-run answer cache.

The key comes from ANTHROPIC_API_KEY (`vtf models` also reads it from a .env file in the
current folder). Claude Opus 5.5 always thinks before answering; `effort` sets how hard.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anthropic

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
# USD per million input and output tokens, for the cost line in the report.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def cost_usd(reply: Reply) -> float:
    price_in, price_out = PRICES.get(reply.model, (0.0, 0.0))
    return (reply.input_tokens * price_in + reply.output_tokens * price_out) / 1e6


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
            data = base64.standard_b64encode(Path(image_path).read_bytes()).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
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
