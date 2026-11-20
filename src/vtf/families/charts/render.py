"""Draws a chart scene with matplotlib and records where every label and mark landed."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from ...core.schema import BBox, LayoutItem, LayoutManifest, Point
from .scene import ChartScene

INK = "#222222"


@dataclass
class _Drawn:
    value_axis: str  # "x" for horizontal bars, otherwise "y"
    marks: list[tuple[str, Any]] = field(default_factory=list)
    points: list[tuple[str, float, float, float]] = field(default_factory=list)  # id, x, y, marker size (pt)
    values: list[tuple[str, Any]] = field(default_factory=list)
    legend: Any = None
    legend_inside: bool = False


def render(scene: ChartScene) -> tuple[Image.Image, LayoutManifest]:
    """Render a scene. A pure function of the scene: the same scene always gives the same pixels."""
    st = scene.style
    rc = {
        "font.family": st.font_family,
        "font.size": st.font_size,
        "axes.titlesize": st.font_size * 1.2,
        "axes.labelsize": st.font_size,
        "xtick.labelsize": st.font_size * 0.92,
        "ytick.labelsize": st.font_size * 0.92,
        "legend.fontsize": st.font_size * 0.9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#555555",
        "xtick.color": INK,
        "ytick.color": INK,
    }
    with rc_context(rc):
        fig = Figure(figsize=st.figsize, dpi=st.dpi, facecolor=st.background, layout="constrained")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.set_facecolor(st.background)
        drawn = _draw(ax, scene)
        canvas.draw()
        pixels = np.ascontiguousarray(np.asarray(canvas.buffer_rgba())[:, :, :3])
        layout = _layout(canvas, ax, scene, drawn)
    return Image.fromarray(pixels), layout


def _fmt(scene: ChartScene, value: float) -> str:
    return f"{value:.{scene.decimals}f}"


def _draw(ax: Any, scene: ChartScene) -> _Drawn:
    st = scene.style
    n, k = len(scene.categories), len(scene.series)
    pos = np.arange(n, dtype=float)
    top = max(max(s.values) for s in scene.series)
    legend_inside = k > 1 and st.legend_loc != "outside"
    label_kw = {"textcoords": "offset points", "fontsize": st.font_size * 0.85, "zorder": 3}
    tick_kw = {"rotation": st.label_rotation, "ha": "right", "rotation_mode": "anchor"} if st.label_rotation else {}

    if scene.chart_type == "hbar":
        drawn = _Drawn(value_axis="x")
        s = scene.series[0]
        bars = ax.barh(pos, s.values, height=st.bar_width, color=st.palette[0], zorder=2)
        for i, rect in enumerate(bars.patches):
            drawn.marks.append((f"bar:0:{i}", rect))
            if st.value_labels:
                xy = (rect.get_width(), rect.get_y() + rect.get_height() / 2)
                label = ax.annotate(_fmt(scene, s.values[i]), xy, xytext=(3, 0), ha="left", va="center", **label_kw)
                drawn.values.append((f"val:0:{i}", label))
        ax.set_yticks(pos, scene.categories)
        ax.set_ylim(n - 0.4, -0.6)  # first category on top
        ax.set_xlim(0, top * 1.2)
        ax.set_xlabel(scene.value_axis)
        ax.set_ylabel(scene.category_axis)
    elif scene.chart_type == "line":
        drawn = _Drawn(value_axis="y")
        for s_idx, s in enumerate(scene.series):
            color = st.palette[s_idx % len(st.palette)]
            (line,) = ax.plot(pos, s.values, marker="o", markersize=5.5, linewidth=2, color=color, label=s.name, zorder=2)
            for i, value in enumerate(s.values):
                drawn.points.append((f"point:{s_idx}:{i}", float(pos[i]), value, line.get_markersize()))
                if st.value_labels:
                    label = ax.annotate(_fmt(scene, value), (pos[i], value), xytext=(0, 6), ha="center", va="bottom", **label_kw)
                    drawn.values.append((f"val:{s_idx}:{i}", label))
        ax.set_xticks(pos, scene.categories, **tick_kw)
        ax.set_xlim(-0.4, n - 0.6)
        low = min(min(s.values) for s in scene.series)
        pad = max((top - low) * 0.12, 1.0)
        ax.set_ylim(max(0.0, low - pad), top + pad * (3.0 if legend_inside else 1.5))
        ax.set_xlabel(scene.category_axis)
        ax.set_ylabel(scene.value_axis)
    else:  # bar, grouped_bar
        drawn = _Drawn(value_axis="y")
        width = st.bar_width / k
        for s_idx, s in enumerate(scene.series):
            offset = (s_idx - (k - 1) / 2) * width
            color = st.palette[s_idx % len(st.palette)]
            bars = ax.bar(pos + offset, s.values, width=width * 0.92, color=color, label=s.name, zorder=2)
            for i, rect in enumerate(bars.patches):
                drawn.marks.append((f"bar:{s_idx}:{i}", rect))
                if st.value_labels:
                    xy = (rect.get_x() + rect.get_width() / 2, rect.get_height())
                    label = ax.annotate(_fmt(scene, s.values[i]), xy, xytext=(0, 2), ha="center", va="bottom", **label_kw)
                    drawn.values.append((f"val:{s_idx}:{i}", label))
        ax.set_xticks(pos, scene.categories, **tick_kw)
        ax.set_xlim(-0.6, n - 0.4)
        headroom = 1.1 + (0.08 if st.value_labels else 0.0) + (0.35 if legend_inside else 0.0)
        ax.set_ylim(0, top * headroom)
        ax.set_xlabel(scene.category_axis)
        ax.set_ylabel(scene.value_axis)

    ax.set_title(scene.title)
    if st.grid:
        ax.grid(True, axis=drawn.value_axis, color="#d4d4d4", linewidth=0.8)
        ax.set_axisbelow(True)
    if st.hide_spines:
        ax.spines[["top", "right"]].set_visible(False)
    if k > 1:
        if legend_inside:
            drawn.legend = ax.legend(loc=st.legend_loc, frameon=True)
        else:
            drawn.legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        drawn.legend_inside = legend_inside
    return drawn


def _layout(canvas: FigureCanvasAgg, ax: Any, scene: ChartScene, drawn: _Drawn) -> LayoutManifest:
    renderer = canvas.get_renderer()
    width, height = canvas.get_width_height()
    px_per_pt = scene.style.dpi / 72.0
    items: list[LayoutItem] = []

    def to_box(extent: Any) -> BBox:
        # Matplotlib measures from the bottom-left; images are indexed from the top-left.
        return BBox(x0=float(extent.x0), y0=float(height - extent.y1), x1=float(extent.x1), y1=float(height - extent.y0))

    def add_text(element_id: str, text: Any) -> None:
        if not text.get_visible() or not text.get_text().strip():
            return
        items.append(
            LayoutItem(
                element_id=element_id,
                kind="text",
                bbox=to_box(text.get_window_extent(renderer)),
                text=text.get_text(),
                font_px=round(text.get_fontsize() * px_per_pt, 2),
                rotation=float(text.get_rotation()),
                quad=_rotated_quad(text, renderer, height),
            )
        )

    cat_axis, val_axis = (ax.yaxis, ax.xaxis) if drawn.value_axis == "x" else (ax.xaxis, ax.yaxis)
    add_text("title", ax.title)
    add_text("axis:category", cat_axis.label)
    add_text("axis:value", val_axis.label)

    index = {name: i for i, name in enumerate(scene.categories)}
    for label in cat_axis.get_ticklabels():
        if label.get_text() in index:
            add_text(f"cat:{index[label.get_text()]}", label)

    low, high = sorted(float(v) for v in val_axis.get_view_interval())
    ticks: list[float] = []
    for tick in val_axis.get_major_ticks():
        loc = tick.get_loc()
        if loc is None or not (low - 1e-9 <= loc <= high + 1e-9) or not tick.label1.get_visible():
            continue
        add_text(f"vtick:{len(ticks)}", tick.label1)
        ticks.append(float(loc))

    for element_id, text in drawn.values:
        add_text(element_id, text)
    if drawn.legend is not None:
        for s_idx, text in enumerate(drawn.legend.get_texts()):
            add_text(f"legend:{s_idx}", text)
        if drawn.legend_inside:
            box = to_box(drawn.legend.get_window_extent(renderer))
            items.append(LayoutItem(element_id="legend", kind="occluder", bbox=box))

    for element_id, rect in drawn.marks:
        items.append(LayoutItem(element_id=element_id, kind="mark", bbox=to_box(rect.get_window_extent(renderer))))
    for element_id, x, y, size_pt in drawn.points:
        cx, cy = (float(v) for v in ax.transData.transform((x, y)))
        r = size_pt * px_per_pt / 2
        box = BBox(x0=cx - r, y0=height - cy - r, x1=cx + r, y1=height - cy + r)
        items.append(LayoutItem(element_id=element_id, kind="mark", bbox=box))

    step = min((b - a for a, b in zip(ticks, ticks[1:])), default=(high - low) / 5)
    meta = {
        "value_axis": [low, high],
        "value_ticks": ticks,
        "tick_step": step,
        "value_labels": scene.style.value_labels,
        "value_precision": 10.0 ** -scene.decimals,
    }
    return LayoutManifest(width=width, height=height, items=items, meta=meta)


def _rotated_quad(text: Any, renderer: Any, height: int) -> list[Point] | None:
    """Exact corners of rotated text, whose axis-aligned box over-covers it.

    With rotation_mode="anchor", matplotlib aligns the unrotated text to its anchor and then
    rotates it about that anchor, so we measure it flat and rotate the corners ourselves.
    """
    angle = text.get_rotation()
    if angle % 90 == 0 or text.get_rotation_mode() != "anchor":
        return None
    ax_, ay_ = text.get_transform().transform(text.get_unitless_position())
    text.set_rotation(0)
    flat = text.get_window_extent(renderer)
    text.set_rotation(angle)
    theta = math.radians(angle)
    cos, sin = math.cos(theta), math.sin(theta)
    corners = [(flat.x0, flat.y0), (flat.x1, flat.y0), (flat.x1, flat.y1), (flat.x0, flat.y1)]
    quad = []
    for cx, cy in corners:
        dx, dy = cx - ax_, cy - ay_
        rx, ry = ax_ + dx * cos - dy * sin, ay_ + dx * sin + dy * cos
        quad.append((float(rx), float(height - ry)))
    return quad
