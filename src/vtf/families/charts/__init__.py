"""Charts family: bar, horizontal bar, grouped bar and line charts, with computed answers."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image

from ...core.schema import LayoutManifest
from .questions import TEMPLATES
from .render import render
from .scene import DEFAULTS, ChartScene, sample_scene


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

    def facets(self, scene: dict[str, Any]) -> dict[str, str]:
        """How accepted tasks are broken down in the coverage report."""
        return {
            "chart type": scene["chart_type"],
            "values": "printed" if scene["style"]["value_labels"] else "axis only",
            "topic": scene["domain"],
        }
