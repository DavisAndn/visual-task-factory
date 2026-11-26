"""Data models shared by every image family."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

# "set": several names, in any order ("T2, T5", or "none"). "time": a clock reading, h:mm.
AnswerType = Literal["integer", "number", "label", "yes_no", "set", "time"]
GateStatus = Literal["pass", "fail", "skip", "review"]
ItemKind = Literal["text", "mark", "occluder"]
Point = tuple[float, float]


class BBox(BaseModel):
    """Axis-aligned box in image pixels, origin at the top-left corner."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def corners(self) -> list[Point]:
        return [(self.x0, self.y0), (self.x1, self.y0), (self.x1, self.y1), (self.x0, self.y1)]

    def inside(self, width: float, height: float, tol: float = 0.5) -> bool:
        return self.x0 >= -tol and self.y0 >= -tol and self.x1 <= width + tol and self.y1 <= height + tol


class LayoutItem(BaseModel):
    """One thing the renderer drew: text, a data mark, or something that can hide marks (a legend)."""

    element_id: str
    kind: ItemKind
    bbox: BBox
    text: str | None = None
    font_px: float | None = None
    rotation: float = 0.0
    quad: list[Point] | None = None  # exact corners of rotated text; its bbox over-covers it

    def polygon(self) -> list[Point]:
        return self.quad or self.bbox.corners()


class LayoutManifest(BaseModel):
    """Everything the renderer drew and where it landed, so gates can check what is visible."""

    width: int
    height: int
    items: list[LayoutItem]
    meta: dict[str, Any] = Field(default_factory=dict)

    def find(self, ref: str) -> list[LayoutItem]:
        """Items with this element id. A trailing '*' matches every id with that prefix."""
        if ref.endswith("*"):
            prefix = ref[:-1]
            return [item for item in self.items if item.element_id.startswith(prefix)]
        return [item for item in self.items if item.element_id == ref]


class Answer(BaseModel):
    value: int | float | str
    type: AnswerType
    tolerance: float = 0.0  # absolute; numbers within it count as correct (minutes, for times)
    unit: str | None = None
    # For "set" answers: every name that could be in the set, so a reply can be searched for them.
    options: list[str] | None = None

    @model_serializer(mode="wrap")
    def _without_unused(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.options is None:  # older answer types stay byte-identical
            data.pop("options", None)
        return data


def set_members(answer: Answer) -> set[str]:
    """The names in a "set" answer."""
    text = str(answer.value)
    return set() if text == "none" else {name.strip() for name in text.split(",") if name.strip()}


_CLOCK = re.compile(r"(\d{1,2})\s*[:.]\s*(\d{2})")


def clock_minutes(text: str) -> int | None:
    """Minutes past 12 o'clock (0-719) for the first h:mm in the text."""
    match = _CLOCK.search(text)
    if match is None or int(match.group(2)) > 59:
        return None
    return int(match.group(1)) % 12 * 60 + int(match.group(2))


def clock_gap(a: int, b: int) -> int:
    """Minutes between two clock readings, the short way round the dial."""
    d = abs(a - b) % 720
    return min(d, 720 - d)


class GateResult(BaseModel):
    status: GateStatus
    code: str | None = None
    reason: str | None = None
    boxes: list[BBox] = Field(default_factory=list)  # where the problem is, for the report


class Task(BaseModel):
    """One packaged task: everything needed to use it, audit it, or rebuild it from its seed."""

    id: str
    family: str
    seed: int
    image: str
    scene_hash: str
    pixel_hash: str
    template: str
    skill: str
    question: str
    answer: Answer
    params: dict[str, Any]
    depends_on: list[str]
    trace: list[str]
    validation: dict[str, GateResult]
    scene: dict[str, Any]
    # Set on photographed twins (vtf photos): the capture settings applied to the rendered image.
    capture: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    def _without_unused(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.capture is None:
            data.pop("capture", None)
        return data


class Pair(BaseModel):
    """A contrast pair: the same question about a minimally edited image, with a different answer."""

    id: str
    a: str  # id of the original task in tasks.jsonl
    b: Task  # the twin task, on the edited image
    change: str  # what was edited, e.g. "Peru in 2023: 55 → 92"
    size: float  # the edit as a share of the value range the reader sees (0.1 = a tenth of the axis)
