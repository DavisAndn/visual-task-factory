"""What an image family provides to plug into the pipeline, and the objects passed between stages."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from .schema import Answer, LayoutManifest


@dataclass
class Question:
    template: str
    skill: str
    text: str
    params: dict[str, Any]
    # Element ids the question and its answer rely on. A trailing '*' means "at least two of these".
    depends_on: list[str]


@dataclass
class Solution:
    answer: Answer
    trace: list[str]
    # Gaps in data units that decide the answer (e.g. top bar minus runner-up). Too small = ambiguous.
    margins: list[float] = field(default_factory=list)


@dataclass
class Candidate:
    """A question about one rendered scene, waiting for the gates."""

    scene: Any
    layout: LayoutManifest
    question: Question
    solution: Solution
    image_key: str
    image_hash: int


class Template(Protocol):
    id: str
    skill: str
    weight: float  # target share of accepted tasks, relative to the other templates

    def applies(self, scene: Any) -> bool: ...

    def instantiate(self, scene: Any, layout: LayoutManifest, rng: random.Random) -> Question | None: ...

    def solve(self, scene: Any, params: dict[str, Any], layout: LayoutManifest) -> Solution: ...


class Family(Protocol):
    name: str
    defaults: dict[str, Any]  # scene settings; a YAML config can override any of them
    templates: list[Template]

    def sample_scene(self, rng: random.Random, cfg: dict[str, Any]) -> Any: ...

    def scene_from_dict(self, data: dict[str, Any]) -> Any: ...

    def render(self, scene: Any) -> tuple[Image.Image, LayoutManifest]: ...

    def facets(self, scene: dict[str, Any]) -> dict[str, str]: ...
