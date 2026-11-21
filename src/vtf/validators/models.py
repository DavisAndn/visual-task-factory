"""Checks that need a language or vision model: the blind baseline and solvability.

A ModelClient answers a prompt, optionally about an image. `vtf models` runs both checks over a
finished run with Claude (see vtf/clients.py); the gates here work with any client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..core.schema import Answer, GateResult, Task

FORMAT = {
    "integer": "Reply with just the number.",
    "number": "Reply with just the number.",
    "label": "Reply with just the name, exactly as it is written in the image.",
    "yes_no": "Reply with just yes or no.",
}
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_TAG = re.compile(r"<[^>]*>")


def final_answer(reply: str) -> str:
    """The part of a reply that holds the final answer.

    Asked for just the answer, models still sometimes show their working first ("335 - 238 = 97",
    then "97"), so scoring reads the last non-empty line, after its last "=", with markup removed.
    """
    lines = [line.strip() for line in _TAG.sub(" ", reply).splitlines() if line.strip()]
    return lines[-1].rsplit("=", 1)[-1] if lines else ""


@dataclass
class Reply:
    text: str
    model: str = ""  # the model that answered; a refusal fallback can differ from the one asked
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False  # answered from the run's cache rather than a new request


class ClientConfigError(RuntimeError):
    """The client can't work at all (bad key, unknown model...): stop instead of retrying per task."""


class ModelClient(Protocol):
    name: str

    def ask(self, prompt: str, image_path: Path | None) -> Reply: ...


def image_prompt(task: Task) -> str:
    return f"{task.question}\n{FORMAT[task.answer.type]}"


def blind_prompt(task: Task) -> str:
    return (
        "This question is about an image you cannot see. Give your best guess anyway.\n"
        f"{task.question}\n{FORMAT[task.answer.type]}"
    )


def score_response(answer: Answer, reply: str) -> bool:
    """Whether a model's free-text reply matches the answer (numbers within tolerance)."""
    text = final_answer(reply).strip().lower().replace("−", "-")
    if answer.type == "yes_no":
        word = re.match(r"[^a-z]*([a-z]+)", text)  # skip markdown or quotes before the first word
        return word is not None and word.group(1) == str(answer.value)
    if answer.type in ("integer", "number"):
        match = _NUMBER.search(text)
        if match is None:
            return False
        value = float(match.group(0).replace(",", ""))
        return abs(value - float(answer.value)) <= answer.tolerance + 1e-9
    return _normalize(text) == _normalize(str(answer.value))


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class BlindBaselineGate:
    """Fails open-ended questions a model answers correctly without the image: they don't test vision.

    Yes/no questions are right half the time by chance, so they are tracked per template in
    the report instead of being failed one by one.
    """

    name = "blind_baseline"

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def check(self, task: Task) -> GateResult:
        reply = self.client.ask(blind_prompt(task), None)
        if task.answer.type != "yes_no" and score_response(task.answer, reply.text):
            return GateResult(
                status="fail",
                code="answerable_blind",
                reason=f"{self.client.name} answered {reply.text!r} without seeing the image",
            )
        return GateResult(status="pass")


class SolvabilityGate:
    """Asks several models with the image. Tasks none of them solve go to human review, not the bin:
    hard but valid tasks are the most valuable ones."""

    name = "solvability"

    def __init__(self, clients: list[ModelClient]) -> None:
        self.clients = clients

    def check(self, task: Task, image_path: Path) -> GateResult:
        prompt = image_prompt(task)
        solved = [c.name for c in self.clients if score_response(task.answer, c.ask(prompt, image_path).text)]
        if not solved:
            return GateResult(status="review", code="unsolved", reason="no model solved it; check it by hand")
        return GateResult(status="pass", reason=f"solved by {', '.join(solved)}")
