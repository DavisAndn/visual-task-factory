"""Charts family: bar, horizontal bar, grouped bar, stacked bar, line and pie charts, with computed answers."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from PIL import Image

from ...core.schema import LayoutManifest
from .questions import TEMPLATES, _entity_and_period, _fmt
from .render import render
from .scene import DEFAULTS, ChartScene, sample_scene

# Contrast-pair edits, as a share of the value axis, smallest first.
EDIT_STEPS = (0.08, 0.12, 0.18, 0.25, 0.35, 0.5)


class ChartsFamily:
    name = "charts"
    defaults = DEFAULTS
    templates = TEMPLATES

    def sample_scene(self, rng: random.Random, cfg: dict[str, Any]) -> ChartScene:
        return sample_scene(rng, cfg)

    def scene_from_dict(self, data: dict[str, Any]) -> ChartScene:
        return ChartScene.model_validate(data)

    def render(self, scene: ChartScene) -> tuple[Image.Image, LayoutManifest]:
        return render(scene)

    def perturb(
        self, scene: ChartScene, layout: LayoutManifest, params: dict[str, Any]
    ) -> Iterator[tuple[str, float, ChartScene]]:
        """Edits of one data value, smallest first, for contrast pairs.

        The value axis is pinned to its original range, so the edited image differs from the
        original only in one bar or point (and its printed value). Yields (description, size as
        a share of the axis, edited scene); the caller keeps the first edit that flips the answer.
        """
        if scene.chart_type == "pie":
            yield from self._perturb_pie(scene)
            return
        low, high = layout.meta["value_axis"]
        span = high - low
        named = sorted({params[key] for key in ("series", "from", "to") if key in params})
        series = named or list(range(len(scene.series)))
        stacked = scene.chart_type == "stacked_bar"
        for step in EDIT_STEPS:
            for s in series:
                for i, value in enumerate(scene.series[s].values):
                    for sign in (1, -1):
                        new = round(value + sign * step * span, scene.decimals)
                        # Stay on the chart, with room above for a printed value.
                        top = sum(other.values[i] for other in scene.series) - value + new if stacked else new
                        if new <= 0 or not low + 0.04 * span <= new or top > high - 0.1 * span:
                            continue
                        edited = scene.model_copy(deep=True)
                        edited.series[s].values[i] = new
                        edited.style.value_limits = (low, high)
                        entity, period = _entity_and_period(scene, s, i)
                        change = f"{entity}, {period}: {_fmt(scene, value)} → {_fmt(scene, new)}"
                        yield change, abs(new - value) / span, edited

    @staticmethod
    def _perturb_pie(scene: ChartScene) -> Iterator[tuple[str, float, ChartScene]]:
        """Move a share from one slice to another, so the slices still add up to 100%."""
        values, cats = scene.series[0].values, scene.categories
        for points in (5, 8, 12, 16, 20):  # percentage points moved
            for i in range(len(values)):
                for j in range(len(values)):
                    gain, loss = round(values[i] + points, scene.decimals), round(values[j] - points, scene.decimals)
                    if i == j or loss < 3:
                        continue
                    edited = scene.model_copy(deep=True)
                    edited.series[0].values[i], edited.series[0].values[j] = gain, loss
                    change = (
                        f"{cats[i]} {_fmt(scene, values[i])}% → {_fmt(scene, gain)}%, "
                        f"{cats[j]} {_fmt(scene, values[j])}% → {_fmt(scene, loss)}%"
                    )
                    yield change, points / 100, edited

    def facets(self, scene: dict[str, Any]) -> dict[str, str]:
        """How tasks are broken down in the coverage report and the model evaluation."""
        n = len(scene["categories"])
        return {
            "chart type": scene["chart_type"],
            "values": "printed" if scene["style"]["value_labels"] else "not printed",
            "topic": scene["domain"],
            "categories": "2–4" if n <= 4 else "5–6" if n <= 6 else "7+",
            "label angle": f"{scene['style']['label_rotation']}°",
        }
