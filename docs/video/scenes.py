"""The video's scenes. A scene draws the frame for time t (seconds into the scene), given when its
cues fall: the words in the narration that the pictures follow. Content stays above y = 930, so
the subtitles at the bottom never cover it."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw

import data
from gfx import (
    ACCENT, BAD, FAINT, GOOD, H, INK, LINE, MUTED, PAPER, W, Sketch, back_out, card, check_mark, clamp,
    ease_in_out, ease_out, fade, frame_card, lerp, mix, paste, pill, place, put_lines, put_text, ramp, text_image,
    text_width, with_alpha,
)

Cue = Callable[..., float]
ASSETS = Path(__file__).resolve().parent / "assets"


def image(run: str, name: str) -> str:
    return str(data.SAMPLES / run / "images" / name)


def task(run: str, task_id: str) -> dict:
    for line in (data.SAMPLES / run / "tasks.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["id"] == task_id:
            return record
    raise KeyError(task_id)


def answer_text(record: dict) -> str:
    value, unit = record["answer"]["value"], record["answer"].get("unit") or ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value} {unit}".strip()


def answered(run: str, name: str) -> tuple[str, str]:
    """An image of a run and the real answer to the first task asked about it."""
    records = [json.loads(line) for line in (data.SAMPLES / run / "tasks.jsonl").read_text().splitlines()]
    mine = [r for r in records if r["image"] == f"images/{name}"]
    # A reading says more than yes or no, and a short one fits under its card.
    best = next((r for r in mine if r["answer"]["type"] != "yes_no" and len(answer_text(r)) <= 16), mine[0])
    return image(run, name), answer_text(best)


def rotated(im: Image.Image, degrees: float) -> Image.Image:
    return im.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=True)


def heading(frame: Image.Image, text: str, t: float, start: float, y: float = 92, size: int = 60, end: float | None = None) -> None:
    a = fade(t, start, 0.5, end)
    put_text(frame, text, W / 2, y + (1 - ease_out(ramp(t, start, 0.6))) * 24, "bold", size, INK, a, "mt")


class Scene:
    hold = 0.0  # seconds of picture after the voice ends, before the next scene
    captions = True

    def draw(self, frame: Image.Image, t: float, T: float, cue: Cue) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------
# 1. Cold open: a syringe every model reads right, then its phone photo, which they don't.


class Hook(Scene):
    def __init__(self, run: str, task_id: str) -> None:
        record = task(run, task_id)
        number = task_id.split("-")[1]
        self.clean = card(image(run, f"{record['family']}-{number}.png"), width=1560)
        self.photo = card(image(run, f"{record['family']}-{number}-photo.jpg"), width=1560)
        self.truth = answer_text(record)
        self.models = data.models()
        self.replies = {m.key: data.answers(run, task_id, m) for m in self.models}
        unit = record["answer"].get("unit") or ""
        self.unit = unit
        self.tags = {"clean": pill("CLEAN DRAWING", "demi", 24, INK, (255, 255, 255, 30)),
                     "photo": pill("PHONE PHOTO", "demi", 24, (20, 20, 20), (*ACCENT, 255))}
        self.per_row = 3 if len(self.models) <= 3 or len(self.models) in (5, 6) else 4
        self.chip_w = 480 if self.per_row == 3 else 440
        self.chips = {m.key: self._chip(m) for m in self.models}

    def _chip(self, m: data.Model) -> Image.Image:
        w, h = self.chip_w, 76
        s = Sketch((w, h))
        s.rrect((1, 1, w - 1, h - 1), 18, fill=(255, 255, 255, 18), outline=(255, 255, 255, 46), width=1.5)
        s.circle((30, h / 2), 8, fill=(*m.color, 255))
        out = s.image()
        name = text_image(m.name, "demi", 27 if self.per_row == 3 else 25, INK)
        out.alpha_composite(name, (50, (h - name.height) // 2))
        return out

    def draw(self, f, t, T, cue):
        snap, right, wrong = cue("snap"), cue("right"), cue("wrong")
        a = fade(t, 0.15)
        put_text(f, "How much liquid is in this syringe?", W / 2, 86, "bold", 60, INK, a, "mt")
        put_text(f, "Read the scale at the edge of the dark stopper nearest the needle.", W / 2, 172, "regular", 30, MUTED, a, "mt")
        cy = 410
        if t < snap:
            place(f, self.clean, W / 2, cy + (1 - ease_out(ramp(t, 0.3, 0.9))) * 40, fade(t, 0.3, 0.8))
            place(f, self.tags["clean"], 250, cy - 205, fade(t, 0.6))
        else:
            k = ease_out(ramp(t, snap, 0.55))
            place(f, rotated(self.photo, lerp(-2.0, 0.0, k)) if k < 0.999 else self.photo, W / 2, cy + 6, 1.0, scale=lerp(1.07, 1.0, k))
            place(f, self.tags["photo"], 250, cy - 214, fade(t, snap + 0.3))
        start, kind = (right, "image") if t < snap else (wrong, "photo")
        per_row, half = self.per_row, self.chip_w / 2
        for i, m in enumerate(self.models):
            row, col = divmod(i, per_row)
            in_row = min(per_row, len(self.models) - row * per_row)
            x = W / 2 + (col - (in_row - 1) / 2) * (self.chip_w + 22)
            y = 690 + row * 96
            appear = fade(t, 0.9 + 0.08 * i) if t < snap else 1.0
            place(f, self.chips[m.key], x, y, appear)
            reveal = ramp(t, start + 0.17 * i, 0.3)
            if reveal > 0:
                text, ok = self.replies[m.key][kind]
                label = f"{text} {self.unit}".strip() if text and not text.endswith(self.unit) else text
                answer = text_image(label or "—", "bold", 29, GOOD if ok else BAD)
                paste(f, answer, x + half - 74 - answer.width, y - answer.height / 2, ease_out(reveal))
                place(f, check_mark(36, GOOD if ok else BAD, ok), x + half - 38, y, ease_out(reveal), scale=back_out(reveal) * 0.999 + 0.001)
        if snap <= t < snap + 0.5:  # the camera flash
            flash = Image.new("RGBA", (W, H), (255, 255, 255, round(220 * (1 - (t - snap) / 0.5) ** 2)))
            f.alpha_composite(flash)
        last = wrong + 0.17 * len(self.models) + 0.6
        put_text(f, f"Correct answer, both times: {self.truth}", W / 2, 690 + ((len(self.models) - 1) // per_row) * 96 + 70,
                 "demi", 34, ACCENT, fade(t, last), "mt")


# ---------------------------------------------------------------------------------------------
# 2. Title card (also the closing card).


def collage(paths: list[str], seed: int = 3) -> Image.Image:
    """Task images scattered on a dark field, for behind the title."""
    rng = random.Random(seed)
    field = Image.new("RGBA", (W + 400, H + 300), (0, 0, 0, 0))
    cols, rows = 5, 3
    for i, path in enumerate(paths[: cols * rows]):
        c, r = i % cols, i // cols
        im = card(path, width=rng.randint(330, 420))
        im = rotated(im, rng.uniform(-9, 9))
        x = c * (field.width / cols) + rng.uniform(-30, 30)
        y = r * (field.height / rows) + rng.uniform(-30, 30)
        field.alpha_composite(im, (round(x), round(y)))
    return with_alpha(field, 0.13)


class Title(Scene):
    def __init__(self, title: str, tagline: str, kicker: str, footer: list[str] | None = None, backdrop: list[str] | None = None, hold: float = 0.0) -> None:
        self.title, self.tagline, self.kicker, self.footer = title, tagline, kicker, footer or []
        self.backdrop = collage(backdrop) if backdrop else None
        self.hold = hold

    def draw(self, f, t, T, cue):
        if self.backdrop is not None:
            drift = t * 14
            paste(f, self.backdrop, -200 - drift, -150 + drift * 0.3, fade(t, 0, 1.2))
        start = cue("title", 0.2)
        put_text(f, self.kicker, W / 2, 330, "demi", 30, ACCENT, fade(t, start, 0.6), "mt")
        k = ease_out(ramp(t, start + 0.15, 0.8))
        put_text(f, self.title, W / 2, 392 + (1 - k) * 30, "heavy", 132, INK, k, "mt")
        line = ease_in_out(ramp(t, start + 0.5, 0.8))
        if line > 0:
            s = Sketch((800, 10))
            half = 380 * line
            s.rrect((400 - half, 2, 400 + half, 8), 3, fill=(*ACCENT, 255))
            place(f, s.image(), W / 2, 578)
        put_text(f, self.tagline, W / 2, 612, "medium", 42, (226, 230, 238), fade(t, start + 0.8, 0.7), "mt")
        for i, line_text in enumerate(self.footer):
            put_text(f, line_text, W / 2, 760 + i * 50, "medium", 32, MUTED, fade(t, start + 1.3 + 0.2 * i), "mt")


# ---------------------------------------------------------------------------------------------
# 3. The problem: AI reads work pictures; benchmarks get used up.


class Pictures(Scene):
    """Cards in a row, each appearing on its cue, then a question over them."""

    def __init__(self, items: list[tuple[str, str, str]], question: str) -> None:
        self.items = [(card(path, height=400, width=410), label, name) for path, label, name in items]
        self.question = question

    def draw(self, f, t, T, cue):
        q = cue("q")
        dim = 1 - 0.65 * ease_in_out(ramp(t, q, 0.5))
        lift = 110 * ease_in_out(ramp(t, q, 0.6))
        n = len(self.items)
        for i, (im, label, name) in enumerate(self.items):
            at = cue(name)
            k = ease_out(ramp(t, at, 0.6))
            x = W / 2 + (i - (n - 1) / 2) * 455
            place(f, im, x, 420 - lift + (1 - k) * 50, k * dim)
            put_text(f, label, x, 662 - lift, "demi", 34, INK, k * dim, "mt")
        a = fade(t, q + 0.2, 0.6)
        put_text(f, "How do you find out where it fails,", W / 2, 690, "bold", 62, INK, a, "mt")
        put_text(f, "before it matters?", W / 2, 772, "bold", 62, ACCENT, fade(t, q + 0.6, 0.6), "mt")


class Benchmark(Scene):
    def __init__(self, stack: list[str], photo: str, stat: list[tuple[str, int, int, tuple[int, int, int]]]) -> None:
        rng = random.Random(5)
        self.stack = [rotated(card(p, width=520), rng.uniform(-9, 9)) for p in stack]
        self.photo = rotated(card(photo, width=760), -5)
        self.stat = stat  # (model name, correct, of, colour)
        self.stamp = rotated(pill("USED UP", "heavy", 84, BAD, (24, 10, 14, 225), 36, 14, outline=(*BAD, 255)), 14)

    def draw(self, f, t, T, cue):
        spent, stat, solved, real = cue("spent"), cue("stat"), cue("solved"), cue("real")
        out = ease_in_out(ramp(t, stat - 0.2, 0.6))
        for i, im in enumerate(self.stack):
            k = ease_out(ramp(t, 0.1 + 0.12 * i, 0.6))
            place(f, im, 560 + i * 18 - out * 900, 460 - i * 14 + (1 - k) * 60, k * (1 - out))
        a = fade(t, 0.4) * (1 - out)
        put_text(f, "A fixed benchmark", 1180, 330, "bold", 64, INK, a, "lt")
        put_lines(f, "A set of images, labelled by hand. Models learn it, ace it, and it's spent.", 1180, 430, 32, "regular", 36, MUTED, a)
        s = ramp(t, spent, 0.35)
        if s > 0:
            place(f, self.stamp, 600 - out * 900, 470, (1 - out) * min(1, s * 2), scale=lerp(1.8, 1.0, ease_out(s)))
        # The stat: three perfect scores on clean charts.
        a = fade(t, stat + 0.1, 0.5, real - 0.1)
        put_text(f, "450 clean charts", W / 2, 170, "demi", 40, MUTED, a, "mt")
        n = len(self.stat)
        for i, (name, correct, of, color) in enumerate(self.stat):
            x = W / 2 + (i - (n - 1) / 2) * 470
            k = ease_out(ramp(t, stat + 0.25 + 0.12 * i, 0.55))
            value = round(100 * correct / of * k)
            put_text(f, f"{value}%", x, 250, "heavy", 140, INK, a * min(1, k * 3), "mt")
            put_text(f, name, x, 440, "demi", 34, color, a * min(1, k * 3), "mt")
        a2 = fade(t, solved, 0.5, real - 0.1)
        put_text(f, "Clean pictures are solved.", W / 2, 600, "bold", 76, ACCENT, a2, "mt")
        r = ease_out(ramp(t, real, 0.7))
        if r > 0:
            place(f, self.photo, lerp(W + 500, 1340, r), 470, r)
            put_text(f, "The real world", 160, 330, "heavy", 92, INK, fade(t, real + 0.2), "lt")
            put_text(f, "isn't clean.", 160, 450, "heavy", 92, ACCENT, fade(t, real + 0.45), "lt")


# ---------------------------------------------------------------------------------------------
# 4. The idea: a factory that manufactures pictures, so it knows every answer.


class Factory(Scene):
    def __init__(self, items: list[tuple[str, str]]) -> None:
        self.cards = [(card(path, height=340, width=500, pad=10), self._tag(label)) for path, label in items]
        self.families = [pill(x, "demi", 30, INK, (255, 255, 255, 26)) for x in ("Charts", "Instruments", "Connection diagrams", "Dense plots")]
        self.speed = 150.0

    @staticmethod
    def _tag(label: str) -> Image.Image:
        tick = check_mark(30, GOOD, True)
        text = text_image(f"Answer: {label}", "demi", 28, GOOD)
        out = Image.new("RGBA", (tick.width + 12 + text.width, max(tick.height, text.height)), (0, 0, 0, 0))
        out.alpha_composite(tick, (0, (out.height - tick.height) // 2))
        out.alpha_composite(text, (tick.width + 12, (out.height - text.height) // 2))
        return out

    def draw(self, f, t, T, cue):
        heading(f, "So I built a factory.", t, cue("factory"))
        belt_y = 520
        a = fade(t, 0.1, 0.8)
        s = Sketch((W, 40), ss=2)
        s.rrect((0, 14, W, 26), 6, fill=(255, 255, 255, 26))
        offset = (t * self.speed) % 60
        x = -60 + offset
        while x < W:
            s.rrect((x, 17, x + 26, 23), 3, fill=(255, 255, 255, 40))
            x += 60
        paste(f, s.image(), 0, belt_y + 180, a)
        gap, x0 = 540, -500 + t * self.speed
        for i in range(len(self.cards) * 3):
            im, tag = self.cards[i % len(self.cards)]
            x = x0 + i * gap - len(self.cards) * gap
            if -320 < x < W + 320:
                place(f, im, x, belt_y, a)
                place(f, tag, x, belt_y + 236, a * fade(t, cue("proof") - 0.3))
        fam = cue("families")
        widths = [p.width for p in self.families]
        total = sum(widths) + 24 * (len(widths) - 1)
        x = (W - total) / 2
        for i, p in enumerate(self.families):
            paste(f, p, x, 220, fade(t, fam + 0.35 * i))
            x += p.width + 24
        proof = cue("proof")
        put_text(f, "Every answer computed from the data.", W / 2, 800, "bold", 50, INK, fade(t, proof + 0.4), "mt")
        put_text(f, "No labelling. No guesswork.", W / 2, 868, "demi", 38, ACCENT, fade(t, cue("guess", proof + 1.6)), "mt")


# ---------------------------------------------------------------------------------------------
# 5. One command, then the checks.


class Terminal(Scene):
    def __init__(self, command: str, output: list[str]) -> None:
        self.command, self.output = command, output

    def draw(self, f, t, T, cue):
        left, top, right, bottom = 190, 190, W - 190, 780
        a = fade(t, 0, 0.4)
        s = Sketch((right - left, bottom - top), ss=2)
        s.rrect((0, 0, right - left, bottom - top), 22, fill=(22, 26, 38, 250), outline=(255, 255, 255, 30), width=1.5)
        for i, color in enumerate(((237, 106, 94), (245, 191, 79), (98, 197, 84))):
            s.circle((36 + i * 30, 30), 8, fill=(*color, 255))
        paste(f, s.image(), left, top, a)
        typed = round(len(self.command) * ramp(t, 0.4, 1.4))
        cursor = "_" if typed < len(self.command) and int(t * 3) % 2 == 0 else ""
        put_text(f, "$", left + 50, top + 78, "mono", 36, (126, 211, 130), a, "lt")
        put_text(f, self.command[:typed] + cursor, left + 96, top + 78, "mono", 36, INK, a, "lt")
        shown = ramp(t, 2.0, 1.0)
        y = top + 156
        for i, line in enumerate(self.output):
            if shown > 0 and i < shown * len(self.output):
                color = (140, 220, 150) if line.lstrip().startswith("Accepted") else (192, 200, 216)
                put_text(f, line[:80], left + 50, y, "mono", 31, color, a, "lt")
            y += 50


class Checks(Scene):
    ITEMS = [
        ("gates", "Gates", "reject anything illegible, ambiguous or duplicated"),
        ("probes", "Pixel probes", "check each drawing matches its data"),
        ("ocr", "OCR read-back", "an outside engine reads the labels again"),
        ("twins", "Contrast twins", "flip one detail, and the answer must flip"),
        ("audit", "Blind human audit", "a person answers from the image alone"),
    ]

    def __init__(self) -> None:
        self.icons = {name: self._icon(name) for name, _, _ in self.ITEMS}

    @staticmethod
    def _icon(name: str) -> Image.Image:
        s = Sketch((120, 120), ss=4)
        c = (*ACCENT, 255)
        if name == "gates":
            s.polygon([(16, 22), (104, 22), (70, 64), (70, 100), (50, 92), (50, 64)], c)
        elif name == "probes":
            s.circle((60, 60), 38, outline=c, width=7)
            for a, b in (((60, 8), (60, 36)), ((60, 84), (60, 112)), ((8, 60), (36, 60)), ((84, 60), (112, 60))):
                s.line([a, b], c, 7)
            s.circle((60, 60), 8, fill=c)
        elif name == "ocr":
            s.rrect((10, 22, 110, 98), 14, outline=c, width=7)
            img = text_image("Aa", "heavy", 50, ACCENT)
            out = s.image()
            out.alpha_composite(img, ((120 - img.width) // 2, (120 - img.height) // 2 - 2))
            return out
        elif name == "twins":
            s.rrect((8, 30, 52, 90), 8, outline=c, width=6)
            s.rrect((68, 30, 112, 90), 8, fill=c)
            s.line([(30, 108), (90, 108)], c, 5)
        else:
            s.circle((60, 38), 20, fill=c)
            s.rrect((24, 66, 96, 112), 22, fill=c)
        return s.image()

    def draw(self, f, t, T, cue):
        heading(f, "Then it tries to break its own work.", t, 0.1, y=110)
        n = len(self.ITEMS)
        for i, (name, title, text) in enumerate(self.ITEMS):
            at = cue(name)
            k = ease_out(ramp(t, at, 0.55))
            x = W / 2 + (i - (n - 1) / 2) * 352
            y = 270 + (1 - k) * 40
            s = Sketch((330, 520), ss=2)
            s.rrect((0, 0, 330, 520), 24, fill=(255, 255, 255, 16), outline=(255, 255, 255, 40), width=1.5)
            place(f, s.image(), x, y + 260, k)
            place(f, self.icons[name], x, y + 104, k)
            put_text(f, title, x, y + 196, "bold", 36, INK, k, "mt")
            put_lines(f, text, x, y + 262, 16, "regular", 31, MUTED, k, align="m")


# ---------------------------------------------------------------------------------------------
# 6. The two dials: size, then photo mode.


def knob(value: float, label: str, readout: str, alpha: float) -> Image.Image:
    size = 440
    s = Sketch((size, size), ss=3)
    c = size / 2
    s.circle((c, c), 168, fill=(255, 255, 255, 14), outline=(255, 255, 255, 60), width=2)
    start, sweep = 135.0, 270.0
    for k in range(28):
        ang = math.radians(start + sweep * k / 27)
        on = k / 27 <= value + 1e-6
        r0, r1 = 182, 206 if k % 3 == 0 else 198
        s.line([(c + r0 * math.cos(ang), c + r0 * math.sin(ang)), (c + r1 * math.cos(ang), c + r1 * math.sin(ang))],
               (*ACCENT, 255) if on else (255, 255, 255, 70), 4)
    s.arc((c, c), 150, start, start + sweep * max(0.001, value), (*ACCENT, 255), 8)
    ang = math.radians(start + sweep * value)
    s.line([(c + 40 * math.cos(ang), c + 40 * math.sin(ang)), (c + 128 * math.cos(ang), c + 128 * math.sin(ang))], (*INK, 255), 12)
    s.circle((c, c), 26, fill=(*ACCENT, 255))
    out = s.image()
    lab = text_image(label, "heavy", 34, ACCENT)
    out.alpha_composite(lab, ((size - lab.width) // 2, size - 64))
    full = Image.new("RGBA", (size, size + 90), (0, 0, 0, 0))
    full.alpha_composite(out, (0, 0))
    read = text_image(readout, "bold", 44, INK)
    full.alpha_composite(read, ((size - read.width) // 2, size + 20))
    return with_alpha(full, alpha)


class Dials(Scene):
    def __init__(self, cables: list[tuple[str, str]], dots: list[tuple[str, str]], morph: list[Image.Image], answer: str) -> None:
        self.cables = [(card(p, width=760, height=780), r) for p, r in cables]
        self.dots = [(card(p, width=860, height=620), r) for p, r in dots]
        self.morph = [frame_card(im, radius=12) for im in morph]
        self.answer = answer
        self.effects = [pill(x, "demi", 28, (20, 20, 20), (*ACCENT, 255)) for x in ("tilt", "glare", "blur", "JPEG")]

    def _step(self, t: float, start: float, n: int, every: float) -> tuple[int, float]:
        x = max(0.0, (t - start) / every)
        i = min(n - 1, int(x))
        return i, (x - i) if i < n - 1 else 1.0

    def draw(self, f, t, T, cue):
        d1, cab, dot, d2, mor, exact = cue("d1"), cue("cables"), cue("dots"), cue("d2"), cue("morph"), cue("exact")
        cx = 1250
        if t < d2:
            # Each sequence runs through all its levels before the next cue, at most 0.6 s a level.
            if t < dot:
                seq, start, base, span, until = self.cables, cab, 0.0, 0.45, dot
            else:
                seq, start, base, span, until = self.dots, dot, 0.5, 0.5, d2 - 0.3
            every = min(0.6, max(0.25, (until - start - 0.35) / (len(seq) - 1)))
            i, frac = self._step(t, start, len(seq), every)
            value = base + span * (i + ease_in_out(min(1, frac * 2.5)) * (i < len(seq) - 1)) / (len(seq) - 1)
            if t < start:
                value, i = base, 0
            out = 1 - ease_in_out(ramp(t, d2 - 0.4, 0.4))
            a = fade(t, d1, 0.5) * out
            place(f, knob(value, "DIAL 1 · SIZE", seq[i][1], 1.0), 470, 540, a)
            im = seq[i][0]
            place(f, im, cx, 500, a)
            if i + 1 < len(seq) and frac > 0.75:  # crossfade into the next level
                place(f, seq[i + 1][0], cx, 500, a * (frac - 0.75) * 4)
        else:
            a = fade(t, d2, 0.5)
            k = ease_in_out(ramp(t, mor, 2.6))
            place(f, knob(k, "DIAL 2 · PHOTO", "phone photo" if k > 0.5 else "clean", 1.0), 470, 540, a)
            pos = k * (len(self.morph) - 1)
            j = int(pos)
            place(f, self.morph[j], cx, 480, a)
            if j + 1 < len(self.morph):
                place(f, self.morph[j + 1], cx, 480, a * (pos - j))
            for i, p in enumerate(self.effects):
                place(f, p, 470 + (i - 1.5) * 128, 868, fade(t, mor + 0.5 * i))
            if t > exact:
                ok = check_mark(46, GOOD, True)
                label = text_image(f"Answer unchanged: {self.answer}", "bold", 36, INK)
                badge = Image.new("RGBA", (label.width + 70, 60), (0, 0, 0, 0))
                badge.alpha_composite(ok, (0, 7))
                badge.alpha_composite(label, (62, 6))
                place(f, badge, cx, 110, fade(t, exact))


# ---------------------------------------------------------------------------------------------
# 7. Results: clean vs photographed, per model; syringes; breaking points.


class Dumbbell(Scene):
    def __init__(self, title: str, subtitle: str = "") -> None:
        self.title, self.subtitle = title, subtitle
        self.rows = sorted(((m, data.same_tasks(m)) for m in data.models()), key=lambda r: -r[1]["photo_clean"])
        low = min(r[1]["photo"] for r in self.rows)
        self.lo = max(0, math.floor((low - 0.06) * 10) / 10)
        self.labels = {m.key: text_image(m.name, "demi", 34, INK) for m, _ in self.rows}

    def x(self, v: float) -> float:
        return lerp(700, 1660, (v - self.lo) / (1 - self.lo))

    def draw(self, f, t, T, cue):
        clean, photo = cue("clean"), cue("photo")
        heading(f, self.title, t, 0.1, y=72, size=54)
        if self.subtitle:
            put_text(f, self.subtitle, W / 2, 150, "medium", 30, MUTED, fade(t, 1.2), "mt")
        n = len(self.rows)
        pitch = min(118, 560 / max(n, 1))
        a = fade(t, 0.3)
        s = Sketch((W, 760), ss=2)
        for g in range(int(round(self.lo * 100)), 101, 10):
            gx = self.x(g / 100)
            s.line([(gx, 210 - 40), (gx, 210 + pitch * n - 10)], (255, 255, 255, 20), 1.5)
        legend_y = 168
        s.circle((706, legend_y), 11, fill=(*INK, 255))
        s.circle((860, legend_y), 11, fill=(*BAD, 255))
        slide = ease_in_out(ramp(t, photo, 1.3))
        for i, (m, r) in enumerate(self.rows):
            y = 210 + pitch * i + pitch / 2 - 20
            k = ease_out(ramp(t, clean + 0.1 * i, 0.5))
            if k <= 0:
                continue
            xc, xp = self.x(r["photo_clean"]), self.x(lerp(r["photo_clean"], r["photo"], slide))
            if slide > 0:
                s.line([(xp, y), (xc, y)], (*BAD, 150), 8)
                s.circle((xp, y), 16, fill=(*BAD, 255))
            s.circle((xc, y), 16 * k, fill=(*INK, 255))
        paste(f, s.image(), 0, 70, a)
        put_text(f, "clean", 726, legend_y + 70, "medium", 28, MUTED, a, "lm")
        put_text(f, "phone photo", 880, legend_y + 70, "medium", 28, MUTED, fade(t, photo), "lm")
        for g in range(int(round(self.lo * 100)), 101, 10):
            put_text(f, f"{g}%", self.x(g / 100), 280 + pitch * n + 10, "medium", 26, FAINT, a, "mt")
        for i, (m, r) in enumerate(self.rows):
            y = 280 + pitch * i + pitch / 2 - 20
            named = ease_out(ramp(t, 2.0 + 0.15 * i, 0.5))  # the models are named first, their scores follow
            k = ease_out(ramp(t, clean + 0.1 * i, 0.5))
            dot = Sketch((24, 24), ss=4)
            dot.circle((12, 12), 10, fill=(*m.color, 255))
            place(f, dot.image(), 160, y, named)
            paste(f, self.labels[m.key], 186, y - self.labels[m.key].height / 2, named)
            put_text(f, f"{r['photo_clean']:.0%}", self.x(r["photo_clean"]) + 30, y, "bold", 30, INK, k, "lm")
            if slide > 0.95:
                drop = round(100 * (r["photo_clean"] - r["photo"]))
                put_text(f, f"{r['photo']:.0%}", self.x(r["photo"]) - 30, y, "bold", 30, BAD, fade(t, photo + 1.2), "rm")
                put_text(f, f"−{drop} pt" + ("s" if drop != 1 else ""), 1740, y, "demi", 28, BAD, fade(t, photo + 1.4), "lm")


class Bars(Scene):
    """Clean vs photographed accuracy, two bars per model, for one kind of picture."""

    def __init__(self, title: str, subtitle: str, rows: list[tuple[data.Model, float, float]]) -> None:
        self.title, self.subtitle, self.rows = title, subtitle, rows

    def draw(self, f, t, T, cue):
        heading(f, self.title, t, 0.1, y=84, size=56)
        put_text(f, self.subtitle, W / 2, 168, "medium", 32, MUTED, fade(t, 0.4), "mt")
        drop = cue("drop")
        n = len(self.rows)
        base, height = 800, 470
        group = min(330, 1500 / n)
        s = Sketch((W, 900), ss=2)
        s.line([(160, base), (W - 160, base)], (255, 255, 255, 60), 2)
        for i, (m, clean, photo) in enumerate(self.rows):
            x = W / 2 + (i - (n - 1) / 2) * group
            k = ease_out(ramp(t, 0.6 + 0.1 * i, 0.7))
            d = ease_in_out(ramp(t, drop + 0.15 * i, 1.0))
            h1 = height * clean * k
            h2 = height * lerp(clean, photo, d) * k
            s.rrect((x - 74, base - h1, x - 6, base), 8, fill=(*INK, 230))
            s.rrect((x + 6, base - h2, x + 74, base), 8, fill=(*BAD, 255))
        paste(f, s.image(), 0, 0, fade(t, 0.3))
        for i, (m, clean, photo) in enumerate(self.rows):
            x = W / 2 + (i - (n - 1) / 2) * group
            k = ease_out(ramp(t, 0.6 + 0.1 * i, 0.7))
            d = ease_in_out(ramp(t, drop + 0.15 * i, 1.0))
            shown = fade(t, 1.4 + 0.1 * i, 0.3)  # values appear once the bars have grown
            size, dx = (30, 40) if n <= 4 else (25, 46)  # smaller and further apart, so two 100% labels don't touch
            put_text(f, f"{clean:.0%}", x - dx, base - height * clean * k - 44, "bold", size, INK, shown, "mt")
            value = lerp(clean, photo, d)
            put_text(f, f"{value:.0%}", x + dx, base - height * value * k - 44, "bold", size, BAD if d > 0 else INK, shown * (d > 0), "mt")
            name = m.name if n <= 4 else m.name.removeprefix("Claude ")
            put_text(f, name, x, base + 22, "demi", 28 if n <= 4 else 24, INK, k, "mt")
        for j, (label, color, start) in enumerate((("clean", INK, 0.6), ("phone photo", BAD, drop))):
            x = W / 2 - 170 + j * 230
            sw = Sketch((26, 26), ss=3)
            sw.rrect((1, 1, 25, 25), 5, fill=(*color, 255))
            place(f, sw.image(), x, 262, fade(t, start))
            put_text(f, label, x + 24, 262, "demi", 30, color, fade(t, start), "lm")


class Breaking(Scene):
    """Accuracy per model and level as a grid of cells: green at 80% or more, amber below that,
    red below 50%. Cells fill in level by level; each model's first cell under 80% is ringed on its
    cue, and a last column shows the same tasks asked without the image."""

    def __init__(self, title: str, run: str, prefix: str, ticks: list[str], unit: str, rings: dict[str, list[str]],
                 lab: str | None = None, note: str = "") -> None:
        self.title, self.ticks, self.unit, self.note = title, ticks, unit, note
        self.levels = data.levels(run, prefix)
        rows = []
        self.asked = {}  # tasks behind each row: models were checked on samples of different sizes
        for m in data.models():
            if lab and m.lab != lab:
                continue
            values = [data.facet(run, "difficulty", v, m).get("with_image") for v in self.levels]
            blind = data.summary(run)["models"][m.key]["blind"]
            rows.append((m, values, blind["correct"] / blind["of"] if blind["of"] else None))
            self.asked[m.key] = data.summary(run)["models"][m.key]["with_image"]["of"]
        breaks = lambda r: next((i for i, v in enumerate(r[1]) if v is not None and v < 0.8), len(r[1]))
        self.rows = sorted(rows, key=lambda r: (-breaks(r), -sum(v or 0 for v in r[1])))
        self.rings = rings  # cue name -> short names of the models whose breaking cell it rings

    @staticmethod
    def color(v: float) -> tuple[int, int, int]:
        return GOOD if v >= 0.8 else ACCENT if v >= 0.5 else BAD

    def draw(self, f, t, T, cue):
        heading(f, self.title, t, 0.1, y=84, size=54)
        start, blind = cue("draw"), cue("blind", None)
        n, cols = len(self.rows), len(self.levels)
        pitch = min(96, 560 / n)
        cell_w, cell_h, gap = 158, pitch - 14, 14
        x0 = 590
        y0 = 250 + (560 - pitch * n) / 2
        a = fade(t, 0.3)
        for i, tick in enumerate(self.ticks):
            put_text(f, tick, x0 + i * (cell_w + gap) + cell_w / 2, y0 - 56, "bold", 34, INK, a, "mt")
        put_text(f, self.unit, x0 + cols * (cell_w + gap) / 2, y0 - 104, "medium", 28, MUTED, a, "mt")
        if blind is not None:
            put_text(f, "no image", x0 + cols * (cell_w + gap) + 30 + cell_w / 2, y0 - 50, "demi", 28, MUTED, fade(t, blind), "mt")
        ringed = {short: cue(name, None) for name, shorts in self.rings.items() for short in shorts}
        s = Sketch((W, H), ss=2)
        labels = []
        for r, (m, values, no_image) in enumerate(self.rows):
            y = y0 + r * pitch
            for i, v in enumerate(values):
                k = ease_out(ramp(t, start + 0.42 * i + 0.05 * r, 0.35))
                if v is None or k <= 0:
                    continue
                x = x0 + i * (cell_w + gap)
                c = self.color(v)
                s.rrect((x, y, x + cell_w, y + cell_h), 12, fill=(*c, round(235 * k)))
                labels.append((f"{v:.0%}", x + cell_w / 2, y + cell_h / 2, k))
            at = ringed.get(m.short)
            first = next((i for i, v in enumerate(values) if v is not None and v < 0.8), None)
            if at is not None and t >= at and first is not None:
                x = x0 + first * (cell_w + gap)
                grow = 10 * (1 - ease_out(ramp(t, at, 0.4)))
                s.rrect((x - 7 - grow, y - 7 - grow, x + cell_w + 7 + grow, y + cell_h + 7 + grow), 16, outline=(*INK, 255), width=5)
            if blind is not None and no_image is not None:
                k = ease_out(ramp(t, blind + 0.05 * r, 0.35))
                x = x0 + cols * (cell_w + gap) + 30
                s.rrect((x, y, x + cell_w, y + cell_h), 12, fill=(*self.color(no_image), round(235 * k)))
                labels.append((f"{no_image:.0%}", x + cell_w / 2, y + cell_h / 2, k))
        paste(f, s.image(), 0, 0)
        for text, x, y, k in labels:
            put_text(f, text, x, y, "bold", 32, (16, 20, 30), k, "mm")
        for r, (m, _, _) in enumerate(self.rows):
            y = y0 + r * pitch + cell_h / 2
            dot = Sketch((24, 24), ss=4)
            dot.circle((12, 12), 10, fill=(*m.color, 255))
            place(f, dot.image(), 110, y, a)
            put_text(f, m.name, 134, y, "demi", 32 if n <= 4 else 28, INK, a, "lm")
            if len(set(self.asked.values())) > 1:
                put_text(f, f"{self.asked[m.key]} tasks", 560, y, "medium", 24, MUTED, a, "rm")
        legend_y = y0 + n * pitch + 24
        if self.note:
            put_lines(f, self.note, 110, legend_y + 70, 92, "medium", 28, MUTED, fade(t, cue("note", start + 1.0)))
        for j, (label, color) in enumerate((("80% or more", GOOD), ("50 to 79%", ACCENT), ("under 50%", BAD))):
            x = x0 + j * 300
            sw = Sketch((26, 26), ss=3)
            sw.rrect((1, 1, 25, 25), 6, fill=(*color, 255))
            place(f, sw.image(), x + 13, legend_y + 16, a)
            put_text(f, label, x + 36, legend_y + 16, "medium", 28, MUTED, a, "lm")


# ---------------------------------------------------------------------------------------------
# 8. Honesty: a model caught a bug in my generator.


class Anecdote(Scene):
    def __init__(self, picture: str, question: str, reply: str, who: str) -> None:
        self.card = card(picture, height=640)
        self.question, self.reply, self.who = question, reply, who

    def draw(self, f, t, T, cue):
        k = ease_out(ramp(t, 0.1, 0.7))
        place(f, self.card, 560, 470 + (1 - k) * 40, k)
        right, fix = cue("right"), cue("fix")
        if t > right:
            r = ease_out(ramp(t, right, 0.6))
            s = Sketch((300, 300), ss=3)
            s.circle((150, 150), 120 + 40 * (1 - r), outline=(*BAD, 255), width=7)
            place(f, s.image(), 560 + 0.105 * self.card.width - 20, 470 + 0.06 * self.card.height, r)
        x = 1080
        put_lines(f, self.question, x, 230, 34, "medium", 34, MUTED, fade(t, cue("ask")))
        put_text(f, f"{self.who}:", x, 400, "demi", 40, MUTED, fade(t, cue("ask") + 0.6), "lt")
        put_text(f, self.reply, x, 452, "heavy", 96, INK, fade(t, cue("ask") + 0.9), "lt")
        put_text(f, "It was right.", x, 610, "bold", 52, GOOD, fade(t, right), "lt")
        put_lines(f, "My generator had drawn gear A out of mesh.", x, 680, 30, "regular", 34, MUTED, fade(t, right + 0.6))
        put_lines(f, "Now every drawing is checked against its data.", x, 800, 30, "demi", 36, ACCENT, fade(t, fix))


# ---------------------------------------------------------------------------------------------
# 9. Why a factory beats a benchmark; who it's for.


class Versus(Scene):
    def __init__(self, rows: list[tuple[str, str, str, str]]) -> None:
        self.rows = rows  # (cue, topic, benchmark, factory)

    def draw(self, f, t, T, cue):
        a = fade(t, 0.1)
        put_text(f, "Fixed benchmark", 690, 130, "bold", 42, MUTED, a, "lt")
        put_text(f, "Visual Task Factory", 1230, 130, "bold", 42, ACCENT, a, "lt")
        n = len(self.rows)
        pitch = min(140, 680 / n)
        for i, (name, topic, old, new) in enumerate(self.rows):
            k = ease_out(ramp(t, cue(name), 0.5))
            y = 230 + i * pitch
            s = Sketch((1680, round(pitch - 18)), ss=2)
            s.rrect((0, 0, 1680, pitch - 18), 18, fill=(255, 255, 255, 14))
            place(f, s.image(), W / 2, y + (pitch - 18) / 2, k)
            mid = y + (pitch - 18) / 2
            put_text(f, topic, 160, mid, "demi", 36, INK, k, "lm")
            place(f, check_mark(40, (120, 128, 146), False), 670, mid, k)
            put_text(f, old, 706, mid, "medium", 34, MUTED, k, "lm")
            place(f, check_mark(40, GOOD, True), 1210, mid, k)
            put_text(f, new, 1248, mid, "demi", 34, INK, k, "lm")


class Value(Scene):
    def __init__(self, cards: list[tuple[str, str, str]]) -> None:
        self.cards = cards  # (cue, who, what)

    def draw(self, f, t, T, cue):
        n = len(self.cards)
        for i, (name, who, what) in enumerate(self.cards):
            k = ease_out(ramp(t, cue(name), 0.6))
            x = W / 2 + (i - (n - 1) / 2) * 760
            s = Sketch((700, 420), ss=2)
            s.rrect((0, 0, 700, 420), 28, fill=(255, 255, 255, 16), outline=(*ACCENT, 120), width=2)
            place(f, s.image(), x, 470 + (1 - k) * 40, k)
            put_text(f, who, x, 320 + (1 - k) * 40, "demi", 34, ACCENT, k, "mt")
            put_lines(f, what, x, 400 + (1 - k) * 40, 24, "bold", 48, INK, k, align="m")
