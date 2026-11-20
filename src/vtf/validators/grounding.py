"""Grounding: everything a question and its answer depend on must be visible in the image.

Uses the renderer's layout manifest, so the check is about what was actually drawn: each
referenced label or mark must exist, sit inside the image, be big enough, and not be covered
by other text or the legend. Label answers must also appear in the image as legible text.
"""

from __future__ import annotations

import math

from ..core.family import Candidate
from ..core.schema import BBox, GateResult, LayoutItem, LayoutManifest, Point


def overlap_px(p: list[Point], q: list[Point]) -> float:
    """How deeply two convex polygons overlap, in pixels (0 if they don't touch).

    Separating-axis test, so rotated labels are compared by their real shape rather than
    their axis-aligned boxes, which overlap long before the text does.
    """
    depth = math.inf
    for poly in (p, q):
        for k in range(len(poly)):
            (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % len(poly)]
            nx, ny = y2 - y1, x1 - x2
            norm = math.hypot(nx, ny)
            if norm == 0:
                continue
            nx, ny = nx / norm, ny / norm
            pa = [nx * x + ny * y for x, y in p]
            qa = [nx * x + ny * y for x, y in q]
            d = min(max(pa), max(qa)) - max(min(pa), min(qa))
            if d <= 0:
                return 0.0
            depth = min(depth, d)
    return 0.0 if depth is math.inf else depth


Problem = tuple[str, str, list[BBox]]


class GroundingGate:
    name = "grounding"

    def __init__(
        self,
        min_font_px: float = 9.0,
        min_mark_px: float = 3.0,
        overlap_tol_px: float = 1.0,
        clean_text: bool = True,
    ) -> None:
        self.min_font_px = min_font_px
        self.min_mark_px = min_mark_px
        self.tol = overlap_tol_px
        self.clean_text = clean_text  # also require that no text anywhere in the image collides

    def check(self, c: Candidate) -> GateResult:
        layout = c.layout
        if self.clean_text:
            texts = [item for item in layout.items if item.kind == "text"]
            for i, first in enumerate(texts):
                for second in texts[i + 1 :]:
                    if overlap_px(first.polygon(), second.polygon()) > self.tol:
                        reason = f"{first.element_id} overlaps {second.element_id} (clean images required)"
                        return _fail("cluttered_image", reason, [first.bbox, second.bbox])

        for ref in c.question.depends_on:
            found = layout.find(ref)
            if ref.endswith("*"):
                legible = [item for item in found if self._problem(item, layout) is None]
                if len(legible) < 2:
                    what = ref.rstrip(":*")
                    return _fail("missing", f"fewer than two legible {what} labels", [i.bbox for i in found])
                continue
            if not found:
                return _fail("missing", f"{ref} is not drawn", [])
            for item in found:
                problem = self._problem(item, layout)
                if problem:
                    return _fail(*problem)

        answer = c.solution.answer
        if answer.type == "label":
            shown = [item for item in layout.items if item.kind == "text" and item.text == str(answer.value)]
            if not any(self._problem(item, layout) is None for item in shown):
                return _fail("answer_not_visible", f"the answer '{answer.value}' is not legible text in the image", [])
        return GateResult(status="pass")

    def _problem(self, item: LayoutItem, layout: LayoutManifest) -> Problem | None:
        if not item.bbox.inside(layout.width, layout.height):
            return "clipped", f"{item.element_id} is cut off at the image edge", [item.bbox]
        if item.kind == "text":
            if (item.font_px or 0.0) < self.min_font_px:
                return "too_small", f"{item.element_id} text is {item.font_px:.1f}px, under {self.min_font_px:g}px", [item.bbox]
            for other in layout.items:
                if other is item or other.kind != "text":
                    continue
                if overlap_px(item.polygon(), other.polygon()) > self.tol:
                    return "text_overlap", f"{item.element_id} overlaps {other.element_id}", [item.bbox, other.bbox]
            if item.element_id.startswith("val:"):
                own = item.element_id.split(":", 1)[1]
                for mark in layout.items:
                    if mark.kind != "mark" or mark.element_id.split(":", 1)[1] == own:
                        continue
                    if overlap_px(item.polygon(), mark.polygon()) > self.tol:
                        return "label_on_mark", f"{item.element_id} is drawn over {mark.element_id}", [item.bbox, mark.bbox]
        elif min(item.bbox.width, item.bbox.height) < self.min_mark_px:
            return "too_small", f"{item.element_id} is under {self.min_mark_px:g}px and hard to see", [item.bbox]
        if not item.element_id.startswith("legend:"):
            for occluder in layout.items:
                if occluder.kind == "occluder" and overlap_px(item.polygon(), occluder.polygon()) > self.tol:
                    return "occluded", f"{item.element_id} is covered by the {occluder.element_id}", [item.bbox, occluder.bbox]
        return None


def _fail(code: str, reason: str, boxes: list[BBox]) -> GateResult:
    return GateResult(status="fail", code=code, reason=reason, boxes=boxes)
