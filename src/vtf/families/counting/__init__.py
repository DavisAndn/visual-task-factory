"""Counting family: how many points of a dense scatter plot lie on one side of a dashed line.

Counting is where the pilot separated the models most clearly, so its dial is the number of
points, from about 30 to about 800 (levels 1-6). Points are at least 6 px apart, and none sits
within a few pixels of the line, so a careful person can count every one.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import Any, Literal

from matplotlib import rc_context
from PIL import Image
from pydantic import BaseModel

from ...core.family import Question, Solution
from ...core.schema import Answer, BBox, GateResult, LayoutManifest
from ..drawing import INK, RC, Sheet, pick, probe_check

# (fewest points, most points, marker size in pt², figure width in inches) by level.
LEVELS = {1: (25, 35, 36, 8.0), 2: (60, 80, 36, 8.0), 3: (140, 160, 25, 10.0), 4: (280, 320, 16, 12.0),
          5: (480, 550, 12, 12.0), 6: (750, 850, 10, 14.0)}
TOPICS = [
    ("Stores by footfall and margin", "Footfall (index)", "Margin (index)", "stores"),
    ("Patients by age and dose", "Age (index)", "Dose (index)", "patients"),
    ("Machines by load and temperature", "Load (index)", "Temperature (index)", "machines"),
    ("Accounts by activity and balance", "Activity (index)", "Balance (index)", "accounts"),
]
DEFAULTS: dict[str, Any] = {"levels": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "dpi": 110}
PLOT = (0.07, 0.09, 0.9, 0.83)  # axes rectangle, as a share of the figure


class CountScene(BaseModel):
    level: int
    dpi: int = 110
    topic: int
    vertical: bool  # the dashed line is vertical (count left or right of it) or horizontal
    threshold: float
    points: list[tuple[float, float]]  # data units, 0-100 on both axes
    size: float  # marker size, pt²
    width_in: float

    @property
    def height_in(self) -> float:
        return self.width_in * 0.625

    def count(self, side: str) -> int:
        coord = 0 if self.vertical else 1
        above = sum(p[coord] > self.threshold for p in self.points)
        return above if side in ("above", "right") else len(self.points) - above


def sample_scene(rng: random.Random, cfg: dict[str, Any]) -> CountScene:
    level = pick(rng, cfg["levels"])
    lo, hi, size, width = LEVELS[level]
    n = rng.randint(lo, hi)
    dpi = int(cfg.get("dpi", 110))
    pw, ph = width * dpi * PLOT[2], width * 0.625 * dpi * PLOT[3]  # plot area, pixels
    r = math.sqrt(size) / 2 / 72 * dpi  # marker radius, pixels
    vertical = rng.random() < 0.5
    t = rng.choice([35, 40, 45, 50, 55, 60, 65])
    pts: list[tuple[float, float]] = []
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    cell = 2 * r + 6  # 6 px between neighbouring points, so a photo of the plot still keeps them apart
    while len(pts) < n:
        x, y = rng.uniform(r + 3, pw - r - 3), rng.uniform(r + 3, ph - r - 3)
        line = t / 100 * (pw if vertical else ph)
        if abs((x if vertical else y) - line) < r + 5:
            continue
        gx, gy = int(x // cell), int(y // cell)
        near = [p for dx in (-1, 0, 1) for dy in (-1, 0, 1) for p in grid.get((gx + dx, gy + dy), [])]
        if any((x - a) ** 2 + (y - b) ** 2 < cell ** 2 for a, b in near):
            continue
        grid.setdefault((gx, gy), []).append((x, y))
        pts.append((x, y))
    points = [(round(x / pw * 100, 4), round(y / ph * 100, 4)) for x, y in pts]
    return CountScene(level=level, dpi=dpi, topic=rng.randrange(len(TOPICS)), vertical=vertical, threshold=t,
                      points=points, size=size, width_in=width)


def render(scene: CountScene) -> tuple[Image.Image, LayoutManifest]:
    title, xlabel, ylabel, _ = TOPICS[scene.topic]
    with rc_context({**RC, "font.size": 10}):
        sheet = Sheet(scene.width_in, scene.height_in, scene.dpi, (0, 1), (0, 1), equal=False)
        sheet.ax.set_axis_off()
        ax = sheet.fig.add_axes(PLOT)
        xs, ys = [p[0] for p in scene.points], [p[1] for p in scene.points]
        ax.scatter(xs, ys, s=scene.size, color="#4E79A7", edgecolors="white", linewidths=0.5, zorder=3)
        line = ax.axvline(scene.threshold, color="#444444", ls="--", lw=1.3, zorder=2) if scene.vertical else \
            ax.axhline(scene.threshold, color="#444444", ls="--", lw=1.3, zorder=2)
        sheet.mark("line", line, pad_px=2)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        sheet.adopt_text("title", ax.set_title(title))
        sheet.adopt_text("xlabel", ax.set_xlabel(xlabel))
        sheet.adopt_text("ylabel", ax.set_ylabel(ylabel))
        for i, label in enumerate(ax.get_xticklabels() + ax.get_yticklabels()):
            sheet.adopt_text(f"tick:{i}", label)
        sheet.canvas.draw()
        height = sheet.canvas.get_width_height()[1]
        r = math.sqrt(scene.size) / 2 / 72 * scene.dpi
        centers = [ax.transData.transform(p) for p in scene.points]
        sheet.mark_box_px("points", BBox(x0=float(min(c[0] for c in centers) - r), y0=float(height - max(c[1] for c in centers) - r),
                                         x1=float(max(c[0] for c in centers) + r), y1=float(height - min(c[1] for c in centers) + r)))
        probes = []
        for x, y in scene.points[:: max(1, len(scene.points) // 6)]:
            px, py = ax.transData.transform((x, y))
            probes.append({"at": [round(float(px), 1), round(float(sheet.canvas.get_width_height()[1] - py), 1)],
                           "expect": "solid", "what": "a point"})
        # Point centers sit at least 2r + 6 px apart, so neighbouring points are 6 px apart at the edges.
        return sheet.finish({"min_feature_px": 6.0, "probes": probes})


class CountSide:
    id, skill, weight = "count_side", "count many", 1.0

    def applies(self, scene: CountScene) -> bool:
        return True

    def instantiate(self, scene: CountScene, layout: LayoutManifest, rng: random.Random) -> Question:
        side = rng.choice(["right", "left"] if scene.vertical else ["above", "below"])
        noun = TOPICS[scene.topic][3]
        where = f"to the {side} of" if scene.vertical else f"{side}"
        text = f"How many {noun} (points) lie {where} the dashed line?"
        return Question(template=self.id, skill=self.skill, text=text, params={"side": side}, depends_on=["points", "line"])

    def solve(self, scene: CountScene, params: dict[str, Any], layout: LayoutManifest) -> Solution:
        count = scene.count(params["side"])
        trace = [f"{len(scene.points)} points; the line is at {scene.threshold:g}",
                 f"{count} lie {params['side']} it; none is within a few pixels of it"]
        return Solution(Answer(value=count, type="integer"), trace)


class CountingFamily:
    name = "counting"
    defaults = DEFAULTS
    templates = [CountSide()]
    questions_per_scene = 1

    def sample_scene(self, rng: random.Random, cfg: dict[str, Any]) -> CountScene:
        return sample_scene(rng, cfg)

    def scene_from_dict(self, data: dict[str, Any]) -> CountScene:
        return CountScene.model_validate(data)

    def render(self, scene: CountScene) -> tuple[Image.Image, LayoutManifest]:
        return render(scene)

    def check_render(self, scene: CountScene, image: Image.Image, layout: LayoutManifest) -> GateResult:
        return probe_check(image, layout)

    def perturb(self, scene: CountScene, layout: LayoutManifest, params: dict[str, Any]) -> Iterator[tuple[str, float, CountScene]]:
        """Move one point to the other side of the line, mirrored, if it lands clear of the others."""
        coord = 0 if scene.vertical else 1
        for i, p in enumerate(scene.points):
            mirrored = list(p)
            mirrored[coord] = 2 * scene.threshold - p[coord]
            if not 2 <= mirrored[coord] <= 98:
                continue
            others = [q for j, q in enumerate(scene.points) if j != i]
            if any(abs(q[0] - mirrored[0]) < 2.5 and abs(q[1] - mirrored[1]) < 2.5 for q in others):
                continue
            points = list(scene.points)
            points[i] = (mirrored[0], mirrored[1])
            yield f"one point moved across the line ({p[0]:.0f}, {p[1]:.0f})", 1 / len(scene.points), \
                scene.model_copy(update={"points": points})
            return

    def facets(self, scene: dict[str, Any]) -> dict[str, str]:
        n = len(scene["points"])
        return {"points": f"{LEVELS[scene['level']][0]}-{LEVELS[scene['level']][1]}", "level": f"L{scene['level']}",
                "difficulty": f"dots L{scene['level']}", "line": "vertical" if scene["vertical"] else "horizontal",
                "count": "under 100" if n < 100 else "100-400" if n < 400 else "400+"}
