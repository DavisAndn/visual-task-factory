"""Ambiguity: an answer must be pinned down by what a reader can actually resolve.

Three checks:
- Templates report their deciding gaps as `margins` (top bar minus runner-up, distance from a
  threshold, ...). With exact values printed, any gap at the printed precision is enough.
  Without them, the gap must be a visible share of the value axis, or two readers could disagree.
- Families without a value axis (instruments, diagrams) report the finest detail a reader must
  resolve, in pixels; it must be at least min_feature_px.
- Numbers read off an axis carry a tolerance. If it is too large relative to the answer
  (e.g. a small difference between two large bars), the question can't be answered precisely.
"""

from __future__ import annotations

from ..core.family import Candidate
from ..core.schema import GateResult


class AmbiguityGate:
    name = "ambiguity"

    def __init__(self, visual_gap: float = 0.05, max_relative_tolerance: float = 0.2, min_feature_px: float = 4.0) -> None:
        self.visual_gap = visual_gap
        self.max_relative_tolerance = max_relative_tolerance
        self.min_feature_px = min_feature_px

    def check(self, c: Candidate) -> GateResult:
        meta = c.layout.meta
        margins = c.solution.margins
        if margins:
            if meta.get("value_labels"):
                need, basis = meta["value_precision"] * 0.999, "the printed precision"
            else:
                low, high = meta["value_axis"]
                need, basis = self.visual_gap * (high - low), f"{self.visual_gap:.0%} of the value axis"
            worst = min(margins)
            if worst < need:
                return GateResult(
                    status="fail",
                    code="too_close",
                    reason=f"the deciding gap is {worst:.3g}, under {need:.3g} ({basis})",
                )

        # Families without a value axis report the finest detail a reader must resolve (the gap
        # between scale ticks, between two gears...) in pixels.
        feature = meta.get("min_feature_px")
        if feature is not None and feature < self.min_feature_px:
            return GateResult(
                status="fail",
                code="too_fine",
                reason=f"details the answer depends on are {feature:.1f}px apart, under {self.min_feature_px:g}px",
            )

        answer = c.solution.answer
        if answer.type == "number" and answer.tolerance > 0:
            value = abs(float(answer.value))
            if value == 0 or answer.tolerance / value > self.max_relative_tolerance:
                return GateResult(
                    status="fail",
                    code="imprecise",
                    reason=(
                        f"the answer {value:g} can only be read to ±{answer.tolerance:g}, "
                        f"more than {self.max_relative_tolerance:.0%} of it"
                    ),
                )
        return GateResult(status="pass")
