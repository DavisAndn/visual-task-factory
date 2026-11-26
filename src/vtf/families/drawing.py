"""Shared drawing for families that aren't charts: a matplotlib figure that records where every
label and mark lands, so the same gates (legibility, overlap, resolution) apply to them."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from ..core.schema import BBox, GateResult, LayoutItem, LayoutManifest, Point

INK = "#222222"
STEEL = "#3B4754"
RC = {"font.family": "DejaVu Sans", "text.color": INK}


class Sheet:
    """One figure with data axes filling it. Coordinates are data units; y points up."""

    def __init__(self, width_in: float, height_in: float, dpi: int, xlim: tuple[float, float],
                 ylim: tuple[float, float], equal: bool = True, background: str = "white") -> None:
        self.dpi = dpi
        self.fig = Figure(figsize=(width_in, height_in), dpi=dpi, facecolor=background)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        if equal:
            self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.axis("off")
        self._texts: list[tuple[str, Any]] = []
        self._artists: list[tuple[str, Any, float]] = []
        self._polys: list[tuple[str, list[Point]]] = []
        self._boxes: list[tuple[str, BBox]] = []

    def text(self, element_id: str, x: float, y: float, s: str, **kw: Any) -> Any:
        artist = self.ax.text(x, y, s, **kw)
        self._texts.append((element_id, artist))
        return artist

    def adopt_text(self, element_id: str, artist: Any) -> Any:
        """Record text that matplotlib drew itself (a title, an axis label, a tick label)."""
        self._texts.append((element_id, artist))
        return artist

    def mark(self, element_id: str, artist: Any, pad_px: float = 0.0) -> Any:
        """Record a drawn artist (patch, line) as a mark, by its drawn extent. A line's extent leaves
        out its width, so pass half the width as pad_px (a flat line would otherwise be 0 px tall)."""
        self._artists.append((element_id, artist, pad_px))
        return artist

    def mark_box_px(self, element_id: str, box: BBox) -> None:
        """Record a mark by a box already in image pixels (for artists that can't report their extent)."""
        self._boxes.append((element_id, box))

    def mark_polygon(self, element_id: str, points: list[Point]) -> None:
        """Record a mark by its outline in data coordinates (exact for thin or tilted shapes)."""
        self._polys.append((element_id, points))

    def px_per_unit(self) -> float:
        """Pixels per data unit along x (the same along y with an equal aspect)."""
        self.canvas.draw()
        (x0, _), (x1, _) = self.ax.transData.transform([(0.0, 0.0), (1.0, 0.0)])
        return float(x1 - x0)

    def to_px(self, x: float, y: float) -> Point:
        """Data point to image pixels (origin top-left)."""
        _, height = self.canvas.get_width_height()
        dx, dy = self.ax.transData.transform((x, y))
        return float(dx), float(height - dy)

    def finish(self, meta: dict[str, Any]) -> tuple[Image.Image, LayoutManifest]:
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        width, height = self.canvas.get_width_height()
        px_per_pt = self.dpi / 72.0

        def to_box(extent: Any) -> BBox:
            return BBox(x0=float(extent.x0), y0=float(height - extent.y1), x1=float(extent.x1), y1=float(height - extent.y0))

        items: list[LayoutItem] = []
        for element_id, artist in self._texts:
            if artist.get_visible() and artist.get_text().strip():
                items.append(LayoutItem(element_id=element_id, kind="text", text=artist.get_text(),
                                        bbox=to_box(artist.get_window_extent(renderer)),
                                        font_px=round(artist.get_fontsize() * px_per_pt, 2)))
        for element_id, artist, pad in self._artists:
            box = to_box(artist.get_window_extent(renderer))
            box = BBox(x0=box.x0 - pad, y0=box.y0 - pad, x1=box.x1 + pad, y1=box.y1 + pad)
            items.append(LayoutItem(element_id=element_id, kind="mark", bbox=box))
        for element_id, points in self._polys:
            quad = [self.to_px(x, y) for x, y in points]
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            box = BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
            items.append(LayoutItem(element_id=element_id, kind="mark", bbox=box, quad=quad if len(quad) == 4 else None))
        items += [LayoutItem(element_id=element_id, kind="mark", bbox=box) for element_id, box in self._boxes]
        pixels = np.ascontiguousarray(np.asarray(self.canvas.buffer_rgba())[:, :, :3])
        return Image.fromarray(pixels), LayoutManifest(width=width, height=height, items=items, meta=meta)


def pick(rng: Any, weights: dict[str, float] | list[float]) -> Any:
    """A weighted choice that is stable across runs: keys (or 1-based levels) in a fixed order."""
    if isinstance(weights, list):
        keys: list[Any] = list(range(1, len(weights) + 1))
        values = [float(w) for w in weights]
    else:
        keys = sorted(weights)
        values = [float(weights[k]) for k in keys]
    keys, values = [k for k, v in zip(keys, values) if v > 0], [v for v in values if v > 0]
    if not keys:
        raise ValueError("every weight is zero")
    return rng.choices(keys, weights=values, k=1)[0]


def probe_check(image: Image.Image, layout: LayoutManifest) -> GateResult:
    """Checks the pixels agree with the scene at the renderer's probe points: dark where a needle,
    hand, stopper or closed valve should be, light where there should be nothing, and so on. It
    reads the final image, so it catches drawing bugs that the scene data can't show (a gear drawn
    out of mesh, a needle pointing somewhere else)."""
    probes = layout.meta.get("probes") or []
    if not probes:
        return GateResult(status="skip")
    gray = np.asarray(image.convert("L"), dtype=float)
    for probe in probes:
        x, y = probe["at"]
        xi, yi = int(round(x)), int(round(y))
        patch = gray[max(yi - 1, 0) : yi + 2, max(xi - 1, 0) : xi + 2]
        if patch.size == 0:
            return GateResult(status="fail", code="probe_outside", reason=f"the {probe['what']} probe is outside the image")
        level = float(patch.mean())
        ok = {"dark": level < 110, "light": level > 170, "solid": level < 248, "background": level > 248}[probe["expect"]]
        if not ok:
            box = BBox(x0=x - 6, y0=y - 6, x1=x + 6, y1=y + 6)
            return GateResult(status="fail", code="drawing_mismatch",
                              reason=f"expected {probe['expect']} pixels at the {probe['what']}, found gray level {level:.0f}",
                              boxes=[box])
    return GateResult(status="pass")
