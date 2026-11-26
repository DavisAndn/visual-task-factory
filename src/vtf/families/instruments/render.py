"""Draws instruments and records their labels, needle, stopper and hands, plus probe points: places
whose pixels must be dark or light if the drawing matches the scene (see drawing.probe_check)."""

from __future__ import annotations

import math
from typing import Any

from matplotlib import rc_context
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
from PIL import Image

from ...core.schema import LayoutManifest
from ..drawing import INK, RC, Sheet
from .scene import InstrumentScene, Scale

RED = "#C0392B"
ROMAN = ["XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]


def render(scene: InstrumentScene) -> tuple[Image.Image, LayoutManifest]:
    with rc_context(RC):
        if scene.kind == "gauge":
            return _gauge(scene)
        if scene.kind == "syringe":
            return _syringe(scene)
        return _clock(scene)


def _dial(sheet: Sheet, scale: Scale, angle: Any, r: float, inward: bool, color: str, label_r: float,
          size: float, prefix: str) -> None:
    for k in range(scale.steps + 1):
        a = angle(k * scale.minor)
        big = k % scale.per_label == 0
        r1 = r - (0.12 if big else 0.06) * (1 if inward else -1)
        sheet.ax.plot([r * math.cos(a), r1 * math.cos(a)], [r * math.sin(a), r1 * math.sin(a)], color=color,
                      lw=1.8 if big else 0.9, zorder=3, solid_capstyle="butt")
        if big:
            sheet.text(f"{prefix}:{k // scale.per_label}", label_r * math.cos(a), label_r * math.sin(a),
                       scale.label(k * scale.minor), ha="center", va="center", fontsize=size, color=color, zorder=3)


def _gauge(scene: InstrumentScene) -> tuple[Image.Image, LayoutManifest]:
    assert scene.scale is not None
    sc, inner = scene.scale, scene.inner
    sheet = Sheet(5.0, 5.0, scene.dpi, (-1.12, 1.12), (-1.12, 1.12))
    ax = sheet.ax
    angle = lambda v: math.radians(225 - 270 * v / sc.top)  # noqa: E731
    ax.add_patch(Circle((0, 0), 1.06, facecolor="#2C3E50", zorder=1))
    ax.add_patch(Circle((0, 0), 0.99, facecolor="white", zorder=2))
    if scene.redline is not None:
        zone = Wedge((0, 0), 0.97, 225 - 270, 225 - 270 * scene.redline / sc.top, width=0.1, facecolor=RED,
                     alpha=0.85, zorder=2.5)
        sheet.mark("redzone", ax.add_patch(zone))
    _dial(sheet, sc, angle, 0.95, True, INK, 0.74 if inner else 0.72, 10 if inner else 11, "glabel")
    if inner is not None:
        _dial(sheet, inner, lambda b: angle(b * scene.factor), 0.5, False, RED, 0.4, 9, "ilabel")
        sheet.text("unit:inner", 0, -0.36, inner.unit, ha="center", fontsize=11, color=RED, zorder=3)
        sheet.text("unit", 0, -0.6, sc.unit, ha="center", fontsize=11, color=INK, zorder=3)
    else:
        sheet.text("unit", 0, -0.42, sc.unit, ha="center", fontsize=12, color=INK, zorder=3)
        sheet.text("title", 0, 0.36, scene.title, ha="center", fontsize=8.5, color="#555555", zorder=3)
    a = angle(scene.value)
    tip, tail, w = 0.9, 0.16, 0.035
    needle = [(tip * math.cos(a), tip * math.sin(a)), (w * math.cos(a + math.pi / 2), w * math.sin(a + math.pi / 2)),
              (-tail * math.cos(a), -tail * math.sin(a)), (w * math.cos(a - math.pi / 2), w * math.sin(a - math.pi / 2))]
    ax.add_patch(Polygon(needle, closed=True, facecolor="#111111", zorder=6))
    sheet.mark_polygon("needle", needle)
    ax.add_patch(Circle((0, 0), 0.06, facecolor="#555555", zorder=7))
    px = sheet.px_per_unit()
    spacing = [0.95 * math.radians(270 * sc.minor / sc.top) * px]
    if inner is not None:
        spacing.append(0.5 * math.radians(270 * inner.minor * scene.factor / sc.top) * px)
    r_light = 0.28 if inner is not None else 0.6  # clear of the inner scale's ticks and labels
    probes = [_probe(sheet, 0.6 * math.cos(a), 0.6 * math.sin(a), "dark", "needle"),
              _probe(sheet, r_light * math.cos(a + 0.35), r_light * math.sin(a + 0.35), "light", "beside the needle")]
    return sheet.finish({"min_feature_px": round(min(spacing), 2), "probes": probes})


def _syringe(scene: InstrumentScene) -> tuple[Image.Image, LayoutManifest]:
    assert scene.scale is not None
    sc = scene.scale
    length = 10.0 if sc.steps <= 50 else 14.0
    scale_in = 0.6 if length == 10.0 else 0.475
    X = lambda v: v / sc.top * length  # noqa: E731
    xd, top, bot = X(scene.value), 0.7, -0.7
    x0, x1 = -2.1, 1.5 * length + 1.2
    sheet = Sheet((x1 - x0) * scale_in, 3.4 * scale_in, scene.dpi, (x0, x1), (-1.5, 1.9))
    ax = sheet.ax
    ax.add_patch(Rectangle((-0.25, bot), length + 0.85, top - bot, facecolor="#F7F9FA", edgecolor="#7F8C8D", lw=1.2, zorder=1))
    sheet.mark("liquid", ax.add_patch(Rectangle((0, bot + 0.06), xd, top - bot - 0.12, facecolor="#BFE0F5",
                                                edgecolor="none", zorder=2)))
    # The thumb press always sits outside the barrel, beyond its flange, as on a real syringe.
    rod_end = max(xd + 0.45 + 0.5 * length, length + 1.0)
    ax.add_patch(Rectangle((xd + 0.45, -0.22), rod_end - xd - 0.45, 0.44, facecolor="#E8EAEB", edgecolor="#95A5A6",
                           lw=1, zorder=2.5))
    ax.add_patch(Rectangle((rod_end, -0.95), 0.3, 1.9, facecolor="#E8EAEB", edgecolor="#95A5A6", lw=1, zorder=2.5))
    stopper = [(xd, bot + 0.04), (xd + 0.45, bot + 0.04), (xd + 0.45, top - 0.04), (xd, top - 0.04)]
    ax.add_patch(Polygon(stopper, closed=True, facecolor="#3D3D3D", edgecolor="#222222", lw=1, zorder=3))
    sheet.mark_polygon("stopper", stopper)
    for dx in (0.12, 0.33):
        ax.plot([xd + dx] * 2, [bot + 0.04, top - 0.04], color="#7A7A7A", lw=1.5, zorder=4)
    ax.add_patch(Rectangle((length + 0.6, -1.15), 0.22, 2.3, facecolor="#F7F9FA", edgecolor="#7F8C8D", lw=1.2, zorder=1.5))
    ax.add_patch(Polygon([(-0.25, 0.35), (-0.25, -0.35), (-0.9, -0.15), (-0.9, 0.15)], closed=True,
                         facecolor="#F7F9FA", edgecolor="#7F8C8D", lw=1.2))
    ax.plot([-2.05, -0.9], [0, 0], color="#95A5A6", lw=1.6)
    half = sc.per_label // 2 if sc.per_label % 2 == 0 else 0
    for k in range(sc.steps + 1):
        x = X(k * sc.minor)
        ln = 0.55 if k % sc.per_label == 0 else 0.4 if half and k % half == 0 else 0.25
        ax.plot([x, x], [top, top - ln], color="#111111", lw=1.0, zorder=5)
        if k % sc.per_label == 0 and k:
            sheet.text(f"slabel:{k // sc.per_label}", x, top + 0.1, sc.label(k * sc.minor), ha="center", va="bottom",
                       fontsize=11, color=INK, zorder=5)
    sheet.text("unit", X(sc.top) + 0.3, top + 0.1, sc.unit, ha="left", va="bottom", fontsize=11, color=INK)
    px = sheet.px_per_unit()
    probes = [_probe(sheet, xd + 0.22, -0.2, "dark", "stopper"), _probe(sheet, max(xd - 0.2, 0.05), -0.2, "light", "liquid")]
    return sheet.finish({"min_feature_px": round(X(sc.minor) * px, 2), "probes": probes})


def _clock(scene: InstrumentScene) -> tuple[Image.Image, LayoutManifest]:
    sheet = Sheet(4.6, 4.6, scene.dpi, (-1.1, 1.1), (-1.1, 1.1))
    ax = sheet.ax
    ang = lambda frac: math.radians(90 - 360 * frac)  # noqa: E731
    ax.add_patch(Circle((0, 0), 1.04, facecolor="#34495E", zorder=1))
    ax.add_patch(Circle((0, 0), 0.97, facecolor="#FDFDFB", zorder=2))
    for k in range(60):
        a, big = ang(k / 60), k % 5 == 0
        r1 = 0.8 if big else 0.87
        ax.plot([0.93 * math.cos(a), r1 * math.cos(a)], [0.93 * math.sin(a), r1 * math.sin(a)], color=INK,
                lw=3 if big else 1, zorder=3, solid_capstyle="butt")
    if scene.numerals != "none":
        for k in range(1, 13):
            a = ang(k / 12)
            text = str(k) if scene.numerals == "arabic" else ROMAN[k % 12]
            sheet.text(f"numeral:{k}", 0.66 * math.cos(a), 0.66 * math.sin(a), text, ha="center", va="center",
                       fontsize=15 if scene.numerals == "arabic" else 12, color=INK, zorder=3)
    position = scene.minute + (scene.seconds or 0) / 60
    hands = [("hour", ((scene.hour % 12) + position / 60) / 12, 0.62 if scene.close_hands else 0.5, 7),
             ("minute", position / 60, 0.78 if scene.close_hands else 0.8, 4)]
    for name, frac, length, lw in hands:
        a = ang(frac)
        ax.plot([-0.1 * math.cos(a), length * math.cos(a)], [-0.1 * math.sin(a), length * math.sin(a)],
                color=INK, lw=lw, solid_capstyle="round", zorder=5)
    if scene.seconds is not None:
        a = ang(scene.seconds / 60)
        ax.plot([-0.18 * math.cos(a), 0.88 * math.cos(a)], [-0.18 * math.sin(a), 0.88 * math.sin(a)],
                color=RED, lw=1.3, zorder=6)
    ax.add_patch(Circle((0, 0), 0.04, facecolor=INK, zorder=7))
    px = sheet.px_per_unit()
    # Hands are recorded by their outline with the line width: a line's drawn extent ignores its
    # width, so a vertical hand would measure 0 px wide.
    for name, frac, length, lw in hands:
        sheet.mark_polygon(f"hand:{name}", _stroke(ang(frac), -0.1, length, lw * scene.dpi / 72 / px))
    if scene.seconds is not None:
        sheet.mark_polygon("hand:second", _stroke(ang(scene.seconds / 60), -0.18, 0.88, 1.3 * scene.dpi / 72 / px))
    probes = [_probe(sheet, 0.4 * math.cos(ang(frac)), 0.4 * math.sin(ang(frac)), "dark", f"{name} hand")
              for name, frac, _, _ in hands]
    return sheet.finish({"min_feature_px": round(0.9 * 2 * math.pi / 60 * px, 2), "probes": probes})


def _stroke(a: float, start: float, end: float, width: float) -> list[tuple[float, float]]:
    """Outline of a straight stroke along angle a, from radius start to end, `width` data units wide."""
    c, s, h = math.cos(a), math.sin(a), width / 2
    nx, ny = -s * h, c * h
    return [(start * c + nx, start * s + ny), (end * c + nx, end * s + ny), (end * c - nx, end * s - ny),
            (start * c - nx, start * s - ny)]


def _probe(sheet: Sheet, x: float, y: float, expect: str, what: str) -> dict[str, Any]:
    px, py = sheet.to_px(x, y)
    return {"at": [round(px, 1), round(py, 1)], "expect": expect, "what": what}
