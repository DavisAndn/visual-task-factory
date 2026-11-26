"""Draws connection diagrams and records ports, tanks, valves and gears, plus pixel probes: valve
fills must match their state, meshing gears must touch, and gears that don't mesh must not."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from matplotlib import rc_context
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from PIL import Image

from ...core.schema import LayoutManifest
from ..drawing import INK, RC, STEEL, Sheet
from .scene import ConnectionScene, Gears, Valves

CABLE_COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#17BECF"]


def render(scene: ConnectionScene) -> tuple[Image.Image, LayoutManifest]:
    with rc_context(RC):
        if scene.kind == "cables":
            return _cables(scene)
        if scene.kind == "valves":
            return _valves(scene)
        return _gears(scene)


def cable_path(i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """The cable from Panel 1 port i to Panel 2 port j. Every cable shares the same x control points,
    so two cables cross at most once, always as a clean X."""
    t = np.linspace(0, 1, 200)
    bx = [0.5, 4.0, 6.0, 9.5]
    xs = (1 - t) ** 3 * bx[0] + 3 * (1 - t) ** 2 * t * bx[1] + 3 * (1 - t) * t ** 2 * bx[2] + t ** 3 * bx[3]
    s = 3 * t ** 2 - 2 * t ** 3
    return xs, i * (1 - s) + j * s


def _cables(scene: ConnectionScene) -> tuple[Image.Image, LayoutManifest]:
    c = scene.cables
    assert c is not None
    n = len(c.perm)
    height_in = 1.6 + 0.36 * n
    sheet = Sheet(7.0, height_in, scene.dpi, (-1.4, 11.4), (n + 0.4, -1.6), equal=False)
    ax = sheet.ax
    for x in (0, 10):
        ax.add_patch(Rectangle((x - 0.5, -0.6), 1.0, n + 0.2, facecolor="#D5D8DC", edgecolor="#808B96", zorder=1))
    for i in range(n):
        xs, ys = cable_path(i, c.perm[i])
        color = CABLE_COLORS[i % len(CABLE_COLORS)] if c.colored else "#2F3B47"
        line = ax.plot(xs, ys, color=color, lw=2.2, zorder=2, solid_capstyle="round")[0]
        sheet.mark(f"cable:{i}", line, pad_px=2.2 * scene.dpi / 72 / 2)
        sheet.mark(f"lport:{i}", ax.add_patch(Rectangle((0, i - 0.22), 0.5, 0.44, facecolor="#1B2631", zorder=3)))
        sheet.mark(f"rport:{i}", ax.add_patch(Rectangle((9.5, i - 0.22), 0.5, 0.44, facecolor="#1B2631", zorder=3)))
        sheet.text(f"left:{i}", -0.75, i, str(i + 1), ha="right", va="center", fontsize=10, color=INK)
        sheet.text(f"right:{i}", 10.75, i, c.right[i], ha="left", va="center", fontsize=10, color=INK)
    sheet.text("panel:1", 0, -1.05, "Panel 1", ha="center", fontsize=10, fontweight="bold", color=INK)
    sheet.text("panel:2", 10, -1.05, "Panel 2", ha="center", fontsize=10, fontweight="bold", color=INK)
    sheet.canvas.draw()
    port_gap = abs(sheet.to_px(0, 1)[1] - sheet.to_px(0, 0)[1])
    return sheet.finish({"min_feature_px": round(port_gap / 2, 2)})


def _bowtie(mx: float, my: float, horizontal: bool) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    s, w = 0.16, 0.11
    if horizontal:
        tris = [[(mx - s, my + w), (mx - s, my - w), (mx, my)], [(mx + s, my + w), (mx + s, my - w), (mx, my)]]
        box = [(mx - s, my - w), (mx + s, my - w), (mx + s, my + w), (mx - s, my + w)]
    else:
        tris = [[(mx + w, my - s), (mx - w, my - s), (mx, my)], [(mx + w, my + s), (mx - w, my + s), (mx, my)]]
        box = [(mx - w, my - s), (mx + w, my - s), (mx + w, my + s), (mx - w, my + s)]
    return box, tris


def node_xy(v: Valves, node: str) -> tuple[float, float]:
    if node == "P":
        return (v.start[0] - 0.95, v.start[1])
    if node.startswith("T"):
        tank = v.tanks[int(node[1:])]
        return (tank.node[0] + 0.95 * tank.out[0], tank.node[1] + 0.95 * tank.out[1])
    i, j = node.split(",")
    return (float(i), float(j))


def _valves(scene: ConnectionScene) -> tuple[Image.Image, LayoutManifest]:
    v = scene.valves
    assert v is not None
    W, H = v.width, v.height
    sheet = Sheet(1.25 * (W + 2.6), 1.25 * (H + 3.6), scene.dpi, (-1.9, W - 1 + 1.7), (-2.6, H - 1 + 2.0))
    ax = sheet.ax
    degree: Counter[str] = Counter()
    for a, b in v.pipes:
        degree[a] += 1
        degree[b] += 1
        (x0, y0), (x1, y1) = node_xy(v, a), node_xy(v, b)
        ax.plot([x0, x1], [y0, y1], color=STEEL, lw=3, solid_capstyle="round", zorder=1)
    for node, d in degree.items():
        if "," in node and d >= 3:
            ax.add_patch(Circle(node_xy(v, node), 0.08, color=STEEL, zorder=2))
    closed = set(v.closed)
    probe_points: list[tuple[float, float, str, str]] = []
    for p in v.valves:
        (x0, y0), (x1, y1) = (node_xy(v, n) for n in v.pipes[p])
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        horizontal = abs(y1 - y0) < 1e-9
        box, tris = _bowtie(mx, my, horizontal)
        for tri in tris:
            ax.add_patch(Polygon(tri, closed=True, facecolor="#111111" if p in closed else "white", edgecolor="#111111",
                                 lw=1.2, zorder=3))
        sheet.mark_polygon(f"valve:{p}", box)
        # The centroid of one triangle: dark for a closed valve, white for an open one.
        cx, cy = (mx - 0.09, my) if horizontal else (mx, my - 0.09)
        probe_points.append((cx, cy, "dark" if p in closed else "light", f"valve on pipe {p}"))
    for k, tank in enumerate(v.tanks):
        cx, cy = tank.node[0] + 1.25 * tank.out[0], tank.node[1] + 1.25 * tank.out[1]
        sheet.mark(f"tankbox:{k}", ax.add_patch(Rectangle((cx - 0.27, cy - 0.32), 0.54, 0.64, facecolor="#EAF2F8",
                                                          edgecolor=STEEL, lw=1.5, zorder=2)))
        sheet.text(f"tank:{k}", cx, cy, tank.name, ha="center", va="center", fontsize=9, fontweight="bold", color=INK, zorder=3)
    px, py = v.start[0] - 1.25, v.start[1]
    sheet.mark("pumpbody", ax.add_patch(Circle((px, py), 0.3, facecolor="white", edgecolor=STEEL, lw=1.5, zorder=2)))
    ax.add_patch(Polygon([(px - 0.13, py + 0.15), (px - 0.13, py - 0.15), (px + 0.17, py)], closed=True, facecolor=STEEL, zorder=3))
    sheet.text("pump", px, py - 0.45, "P1", ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
    sheet.text("title", -1.8, H - 1 + 1.85, "Cooling-water network: valve line-up", fontsize=11, va="top", color=INK)
    ly = -2.25
    for x, shut, label in ((-1.5, False, "open valve"), (0.6, True, "closed valve")):
        for tri in _bowtie(x, ly, True)[1]:
            ax.add_patch(Polygon(tri, closed=True, facecolor="#111111" if shut else "white", edgecolor="#111111", lw=1.2))
        sheet.text(f"legend:{label.split()[0]}", x + 0.25, ly, label, va="center", fontsize=8.5, color=INK)
    ax.add_patch(Circle((2.8, ly), 0.08, color=STEEL))
    sheet.text("legend:joined", 3.0, ly, "pipes joined", va="center", fontsize=8.5, color=INK)
    scale = sheet.px_per_unit()
    probes = [{"at": [round(c, 1) for c in sheet.to_px(x, y)], "expect": e, "what": w} for x, y, e, w in probe_points]
    return sheet.finish({"min_feature_px": round(0.22 * scale, 2), "probes": probes})


def gear_outline(x: float, y: float, z: int, m: float, phase: float) -> list[tuple[float, float]]:
    R = m * z / 2
    ro, rr, p = R + m, R - 1.25 * m, 2 * math.pi / z
    pts = []
    for i in range(z):
        c = phase + i * p
        for frac, rad in ((-0.5, rr), (-0.25, rr), (-0.12, ro), (0.12, ro), (0.25, rr)):
            pts.append((x + rad * math.cos(c + frac * p), y + rad * math.sin(c + frac * p)))
    return pts


def _gears(scene: ConnectionScene) -> tuple[Image.Image, LayoutManifest]:
    g = scene.gears
    assert g is not None
    W, H = 100.0, 100.0 * 5.5 / 8.0
    sheet = Sheet(8.0, 5.5, scene.dpi, (0, W), (0, H))
    ax = sheet.ax
    m = g.module
    for i, gear in enumerate(g.gears):
        body = ax.add_patch(Polygon(gear_outline(gear.x, gear.y, gear.teeth, m, gear.phase), closed=True,
                                    facecolor="#DCE2E8", edgecolor=STEEL, lw=1.1, zorder=2))
        sheet.mark(f"gearbody:{i}", body)
        ax.add_patch(Circle((gear.x, gear.y), 1.3, facecolor="white", edgecolor=STEEL, lw=1.0, zorder=3))
        sheet.text(f"gear:{i}", gear.x, gear.y + 0.5 * g.radius(i), gear.label, ha="center", va="center", fontsize=12,
                   fontweight="bold", color=INK, zorder=4)
    # The driver's direction: a clockwise arrow on gear A's own face, below its label, so it never
    # crosses another gear or a gap between two.
    a, r = g.gears[0], 0.62 * g.radius(0)
    p0 = (a.x + r * math.cos(math.radians(335)), a.y + r * math.sin(math.radians(335)))
    p1 = (a.x + r * math.cos(math.radians(205)), a.y + r * math.sin(math.radians(205)))
    sheet.mark("arrow", ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                                     mutation_scale=14, lw=2, color="#C0392B", zorder=5)))
    sheet.text("title", 2, H - 2, "Gear train, driven at gear A", fontsize=11, va="top", color=INK)
    scale = sheet.px_per_unit()
    probes, gaps = [], [m]
    for i, gear in enumerate(g.gears):
        for j in range(i):
            other = g.gears[j]
            dx, dy = gear.x - other.x, gear.y - other.y
            dist = math.hypot(dx, dy)
            ux, uy = dx / dist, dy / dist
            if gear.parent == j:
                # Meshing: just past the parent's tooth tips lies the child's solid body (inside its
                # root circle), whatever the teeth's phase. A gear drawn out of mesh leaves it empty.
                x, y = other.x + ux * (g.radius(j) + 1.3 * m), other.y + uy * (g.radius(j) + 1.3 * m)
                probes.append({"at": [round(c, 1) for c in sheet.to_px(x, y)], "expect": "solid",
                               "what": f"mesh of gears {other.label} and {gear.label}"})
                continue
            gap = dist - g.radius(i) - g.radius(j) - 2 * m  # between the tooth tips
            gaps.append(gap)
            if gap < 8 * m:
                mid = g.radius(j) + m + gap / 2
                x, y = other.x + ux * mid, other.y + uy * mid
                if _clear(g, x, y, (i, j)):
                    probes.append({"at": [round(c, 1) for c in sheet.to_px(x, y)], "expect": "background",
                                   "what": f"gap between gears {other.label} and {gear.label}"})
    return sheet.finish({"min_feature_px": round(min(gaps) * scale, 2), "probes": probes})


def _clear(g: Gears, x: float, y: float, skip: tuple[int, int]) -> bool:
    """Whether a point is outside every other gear (so the background should show there)."""
    return all(math.hypot(x - gear.x, y - gear.y) > g.radius(k) + 2 * g.module
               for k, gear in enumerate(g.gears) if k not in skip)
