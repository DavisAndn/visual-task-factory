"""Checks that need a language or vision model: the blind baseline and model solvability.

Not wired into `vtf generate` yet. To use them, implement ModelClient for your provider (for
example with the hackathon's model credits) and run the gates over a run's tasks.jsonl.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from ..core.schema import Answer, GateResult, Task

BLIND_PROMPT = (
    "You cannot see the image this question is about. Give your best guess anyway.\n"
    "{question}\nReply with only the answer."
)
IMAGE_PROMPT = "{question}\nReply with only the answer."
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


class ModelClient(Protocol):
    name: str

    def ask(self, prompt: str, image_path: Path | None) -> str: ...


def score_response(answer: Answer, reply: str) -> bool:
    """Whether a model's free-text reply matches the answer (numbers within tolerance)."""
    text = reply.strip().lower()
    if answer.type == "yes_no":
        word = re.match(r"[a-z]+", text)
        return word is not None and word.group(0) == str(answer.value)
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
    the report's always-guess baseline instead of being failed one by one.
    """

    name = "blind_baseline"

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def check(self, task: Task) -> GateResult:
        reply = self.client.ask(BLIND_PROMPT.format(question=task.question), None)
        if task.answer.type != "yes_no" and score_response(task.answer, reply):
            return GateResult(
                status="fail",
                code="answerable_blind",
                reason=f"{self.client.name} answered {reply.strip()!r} without seeing the image",
            )
        return GateResult(status="pass")


class SolvabilityGate:
    """Asks several models with the image. Tasks none of them solve go to human review, not the bin:
    hard but valid tasks are the most valuable ones."""

    name = "solvability"

    def __init__(self, clients: list[ModelClient]) -> None:
        self.clients = clients

    def check(self, task: Task, image_path: Path) -> GateResult:
        prompt = IMAGE_PROMPT.format(question=task.question)
        solved = [c.name for c in self.clients if score_response(task.answer, c.ask(prompt, image_path))]
        if not solved:
            return GateResult(status="review", code="unsolved", reason="no model solved it; check it by hand")
        return GateResult(status="pass", reason=f"solved by {', '.join(solved)}")
