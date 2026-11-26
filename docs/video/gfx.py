"""Drawing helpers for the demo video: theme, fonts, easing, text, image cards and smooth shapes.

Frames are 1920x1080 RGBA images. Text and cards are rendered once and cached, then composited
every frame at some opacity, which keeps a scene's draw call cheap. Shapes that need smooth edges
(dots, rings, chart lines) are drawn on a supersampled layer and scaled down.
"""

from __future__ import annotations

import textwrap
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
FPS = 30
LANCZOS = Image.Resampling.LANCZOS

BG_TOP, BG_BOTTOM = (9, 15, 30), (20, 31, 55)
INK = (246, 243, 236)
MUTED = (158, 169, 191)
FAINT = (84, 97, 124)
LINE = (52, 64, 90)
ACCENT = (247, 186, 62)
GOOD = (56, 201, 128)
BAD = (243, 88, 88)
PAPER = (250, 249, 246)

_AVENIR = Path("/System/Library/Fonts/Avenir Next.ttc")
_MENLO = Path("/System/Library/Fonts/Menlo.ttc")
_FACES = {"bold": 0, "demi": 2, "medium": 5, "regular": 7, "heavy": 8}
_DEJAVU = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


@lru_cache(maxsize=None)
def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    """Avenir Next in a weight (heavy, bold, demi, medium, regular) or Menlo ("mono"), with DejaVu
    as the fallback off macOS."""
    if weight in ("mono", "mono-bold"):
        if _MENLO.exists():
            return ImageFont.truetype(str(_MENLO), size, index=1 if weight == "mono-bold" else 0)
        return ImageFont.truetype(str(_DEJAVU / "DejaVuSansMono.ttf"), size)
    if _AVENIR.exists():
        return ImageFont.truetype(str(_AVENIR), size, index=_FACES[weight])
    bold = weight in ("heavy", "bold", "demi")
    return ImageFont.truetype(str(_DEJAVU / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


# Easing: all take and return values in [0, 1].
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def ramp(t: float, start: float, duration: float = 0.5) -> float:
    """0 before `start`, rising linearly to 1 over `duration` seconds."""
    if duration <= 0:
        return float(t >= start)
    return clamp((t - start) / duration)


def ease_out(x: float) -> float:
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x: float) -> float:
    x = clamp(x)
    return x * x * (3 - 2 * x)


def back_out(x: float, overshoot: float = 1.4) -> float:
    """Ease out with a small overshoot, for things that pop in."""
    x = clamp(x)
    return 1 + (overshoot + 1) * (x - 1) ** 3 + overshoot * (x - 1) ** 2


def fade(t: float, start: float, duration: float = 0.45, end: float | None = None, out: float = 0.35) -> float:
    """Opacity that fades in at `start` and, if `end` is given, out again from `end`."""
    a = ease_out(ramp(t, start, duration))
    if end is not None:
        a *= 1 - ease_in_out(ramp(t, end, out))
    return a


def lerp(a: float, b: float, x: float) -> float:
    return a + (b - a) * x


def mix(c1: tuple[int, ...], c2: tuple[int, ...], x: float) -> tuple[int, ...]:
    return tuple(round(lerp(a, b, clamp(x))) for a, b in zip(c1, c2))


@lru_cache(maxsize=1)
def background() -> Image.Image:
    """A deep navy gradient with a soft central glow, a vignette and a little grain (grain keeps the
    gradient from banding once it is compressed)."""
    yy = np.linspace(0.0, 1.0, H)[:, None, None]
    xs = np.linspace(-1.0, 1.0, W)[None, :, None]
    ys = np.linspace(-1.0, 1.0, H)[:, None, None]
    top, bottom = np.array(BG_TOP, float), np.array(BG_BOTTOM, float)
    rgb = top * (1 - yy) + bottom * yy
    r2 = xs**2 * 0.55 + (ys + 0.15) ** 2
    rgb = rgb + np.exp(-r2 * 1.8) * np.array([16.0, 22.0, 38.0])
    rgb = rgb * (1 - 0.38 * np.clip(r2 - 0.35, 0, None))
    rgb = rgb + np.random.default_rng(7).normal(0, 1.1, rgb.shape)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def new_frame() -> Image.Image:
    return background().copy()


def with_alpha(im: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 0.999:
        return im
    out = im.copy()
    out.putalpha(im.getchannel("A").point(lambda v: round(v * clamp(alpha))))
    return out


def paste(frame: Image.Image, im: Image.Image, x: float, y: float, alpha: float = 1.0) -> None:
    """Composite `im` with its top-left corner at (x, y), clipped to the frame."""
    if alpha <= 0.003:
        return
    x, y = round(x), round(y)
    left, top = max(0, -x), max(0, -y)
    right, bottom = min(im.width, frame.width - x), min(im.height, frame.height - y)
    if right <= left or bottom <= top:
        return
    if (left, top, right, bottom) != (0, 0, im.width, im.height):
        im = im.crop((left, top, right, bottom))
    frame.alpha_composite(with_alpha(im, alpha), (x + left, y + top))


def place(frame: Image.Image, im: Image.Image, cx: float, cy: float, alpha: float = 1.0, scale: float = 1.0) -> None:
    """Composite `im` centred on (cx, cy), optionally scaled."""
    if alpha <= 0.003 or scale <= 0.01:
        return
    if abs(scale - 1) > 0.002:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.BILINEAR)
    paste(frame, im, cx - im.width / 2, cy - im.height / 2, alpha)


@lru_cache(maxsize=4096)
def text_image(s: str, weight: str, size: int, color: tuple[int, int, int] = INK) -> Image.Image:
    """One line of text on a transparent image whose box spans the font's full ascent and descent."""
    f = font(weight, size)
    ascent, descent = f.getmetrics()
    left, _, right, _ = f.getbbox(s) if s else (0, 0, 0, 0)
    im = Image.new("RGBA", (max(1, right - min(left, 0) + 6), ascent + descent + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((3 - min(left, 0), 2), s, font=f, fill=(*color, 255))
    return im


def text_width(s: str, weight: str, size: int) -> int:
    return round(font(weight, size).getlength(s))


def put_text(
    frame: Image.Image,
    s: str,
    x: float,
    y: float,
    weight: str = "medium",
    size: int = 40,
    color: tuple[int, int, int] = INK,
    alpha: float = 1.0,
    anchor: str = "lt",
) -> tuple[int, int]:
    """Draw one line of text. Anchor: l/m/r horizontally, then t (top) or m (middle) vertically.
    Returns the text image's size."""
    im = text_image(s, weight, size, color)
    dx = {"l": 0, "m": im.width / 2, "r": im.width}[anchor[0]]
    dy = {"t": 0, "m": im.height / 2}[anchor[1]]
    paste(frame, im, x - dx, y - dy, alpha)
    return im.size


def put_lines(
    frame: Image.Image,
    s: str,
    x: float,
    y: float,
    width: int,
    weight: str = "medium",
    size: int = 40,
    color: tuple[int, int, int] = INK,
    alpha: float = 1.0,
    align: str = "l",
    leading: float = 1.28,
) -> float:
    """Wrapped text, `width` characters per line. Returns the y below the last line."""
    for line in textwrap.wrap(s, width):
        put_text(frame, line, x, y, weight, size, color, alpha, anchor=f"{align}t")
        y += size * leading
    return y


def _round_mask(size: tuple[int, int], radius: int, ss: int = 4) -> Image.Image:
    big = Image.new("L", (size[0] * ss, size[1] * ss), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, big.width - 1, big.height - 1), radius * ss, fill=255)
    return big.resize(size, LANCZOS)


@lru_cache(maxsize=32)
def _shadow(size: tuple[int, int], radius: int, spread: int, strength: int) -> Image.Image:
    w, h = size
    layer = Image.new("L", (w + 2 * spread, h + 2 * spread), 0)
    ImageDraw.Draw(layer).rounded_rectangle((spread, spread + spread // 4, spread + w, spread + h + spread // 4), radius, fill=strength)
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    shadow.putalpha(layer.filter(ImageFilter.GaussianBlur(spread / 2.4)))
    return shadow


def frame_card(
    picture: Image.Image, pad: int = 0, radius: int = 16, spread: int = 34, color: tuple[int, int, int] = PAPER
) -> Image.Image:
    """A picture on a rounded card with a soft drop shadow. The result is larger than the card by
    `spread` on each side, so centre it rather than placing its corner."""
    body = Image.new("RGBA", (picture.width + 2 * pad, picture.height + 2 * pad), (*color, 255))
    body.paste(picture.convert("RGB"), (pad, pad))
    body.putalpha(_round_mask(body.size, radius))
    out = _shadow(body.size, radius, spread, 170).copy()
    out.alpha_composite(body, (spread, spread))
    return out


@lru_cache(maxsize=96)
def card(path: str, width: int = 0, height: int = 0, pad: int = 0, radius: int = 16, crop: tuple | None = None) -> Image.Image:
    """An image file as a card, scaled to a width or a height (or to fit both)."""
    src = Image.open(path).convert("RGB")
    if crop:
        src = src.crop(crop)
    inner_w, inner_h = width - 2 * pad if width else 0, height - 2 * pad if height else 0
    if inner_w and inner_h:
        scale = min(inner_w / src.width, inner_h / src.height)
    elif inner_w:
        scale = inner_w / src.width
    else:
        scale = inner_h / src.height
    picture = src.resize((round(src.width * scale), round(src.height * scale)), LANCZOS)
    return frame_card(picture, pad, radius)


class Sketch:
    """A transparent layer for smooth shapes: drawn `ss` times larger, then scaled down."""

    def __init__(self, size: tuple[int, int], ss: int = 3) -> None:
        self.size, self.ss = size, ss
        self.layer = Image.new("RGBA", (size[0] * ss, size[1] * ss), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.layer)

    def _p(self, *xy: float) -> list[float]:
        return [v * self.ss for v in xy]

    def line(self, points: list[tuple[float, float]], color: tuple[int, ...], width: float, joint: bool = True) -> None:
        if len(points) < 2:
            return
        flat = [v * self.ss for p in points for v in p]
        self.draw.line(flat, fill=color, width=max(1, round(width * self.ss)), joint="curve" if joint else None)
        r = width / 2
        for x, y in (points[0], points[-1]):
            self.circle((x, y), r, color)

    def dashed(self, a: tuple[float, float], b: tuple[float, float], color: tuple[int, ...], width: float, dash: float = 12, gap: float = 9) -> None:
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if length == 0:
            return
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            self.draw.line(self._p(a[0] + ux * pos, a[1] + uy * pos, a[0] + ux * end, a[1] + uy * end), fill=color, width=max(1, round(width * self.ss)))
            pos = end + gap

    def circle(self, center: tuple[float, float], r: float, fill: tuple[int, ...] | None = None, outline: tuple[int, ...] | None = None, width: float = 0) -> None:
        x, y = center
        self.draw.ellipse(self._p(x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=round(width * self.ss) if outline else 0)

    def rrect(self, box: tuple[float, float, float, float], radius: float, fill: tuple[int, ...] | None = None, outline: tuple[int, ...] | None = None, width: float = 0) -> None:
        self.draw.rounded_rectangle(self._p(*box), radius * self.ss, fill=fill, outline=outline, width=round(width * self.ss) if outline else 0)

    def polygon(self, points: list[tuple[float, float]], fill: tuple[int, ...]) -> None:
        self.draw.polygon([v * self.ss for p in points for v in p], fill=fill)

    def arc(self, center: tuple[float, float], r: float, start: float, end: float, color: tuple[int, ...], width: float) -> None:
        x, y = center
        self.draw.arc(self._p(x - r, y - r, x + r, y + r), start, end, fill=color, width=max(1, round(width * self.ss)))

    def image(self) -> Image.Image:
        return self.layer.resize(self.size, LANCZOS)


def pill(text: str, weight: str = "demi", size: int = 30, fg: tuple[int, int, int] = INK, bg: tuple[int, ...] = (255, 255, 255, 28), pad_x: int = 22, pad_y: int = 10, outline: tuple[int, ...] | None = None) -> Image.Image:
    """Text in a rounded pill, as an image."""
    label = text_image(text, weight, size, fg)
    w, h = label.width + 2 * pad_x, label.height + 2 * pad_y
    sketch = Sketch((w, h))
    sketch.rrect((1, 1, w - 1, h - 1), h / 2 - 1, fill=bg, outline=outline, width=2 if outline else 0)
    out = sketch.image()
    out.alpha_composite(label, (pad_x, pad_y))
    return out


def check_mark(size: int, color: tuple[int, int, int], good: bool) -> Image.Image:
    """A filled circle with a tick (good) or a cross."""
    s = Sketch((size, size), ss=4)
    s.circle((size / 2, size / 2), size / 2 - 1, fill=(*color, 255))
    w = size * 0.11
    if good:
        s.line([(size * 0.28, size * 0.52), (size * 0.44, size * 0.68), (size * 0.73, size * 0.34)], (255, 255, 255, 255), w)
    else:
        s.line([(size * 0.32, size * 0.32), (size * 0.68, size * 0.68)], (255, 255, 255, 255), w)
        s.line([(size * 0.68, size * 0.32), (size * 0.32, size * 0.68)], (255, 255, 255, 255), w)
    return s.image()


def caption_box(text: str, alpha: float = 1.0) -> Image.Image | None:
    """The subtitle for the bottom of the frame: at most two centred lines on a dark translucent box,
    split near the middle so the two lines are about as long."""
    lines = [text]
    if len(text) > 62:
        spaces = [i for i, c in enumerate(text) if c == " "]
        cut = min(spaces, key=lambda i: abs(i - len(text) / 2)) if spaces else len(text)
        lines = [text[:cut], text[cut + 1 :]] if spaces else textwrap.wrap(text, 66)[:2]
    if not lines or alpha <= 0.003:
        return None
    size = 34
    widths = [text_width(line, "medium", size) for line in lines]
    w, h = max(widths) + 56, round(len(lines) * size * 1.32) + 28
    box = Sketch((w, h), ss=2)
    box.rrect((0, 0, w, h), 14, fill=(6, 10, 20, 178))
    out = box.image()
    y = 14
    for line, lw in zip(lines, widths):
        im = text_image(line, "medium", size, (240, 238, 232))
        out.alpha_composite(im, ((w - im.width) // 2, round(y)))
        y += size * 1.32
    return with_alpha(out, alpha)
