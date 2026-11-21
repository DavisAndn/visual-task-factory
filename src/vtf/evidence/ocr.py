"""OCR read-back: an independent check that the text a question depends on is legible in the pixels.

The grounding gate trusts the renderer's layout manifest. This check doesn't. It crops every label a
question depends on out of the final PNG, straightens rotated text, and asks Tesseract, an OCR engine
unrelated to any model being evaluated, to read it. A label counts as read back when the OCR text
matches the text the manifest says was drawn there. Each label gets up to two reads: a thresholded
crop, then, if that misses, a grayscale crop (digits only for numbers). Mismatches are kept with their
crops so a person can see whether the label or the OCR engine is at fault.
"""

from __future__ import annotations

import io
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..core.runs import read_tasks
from ..core.schema import LayoutItem, Task
from ..families import get_family

# Characters OCR engines mistake for each other, folded the same way on both sides of a comparison
# (after lowercasing, so "Harlow" and "HarLow" fold alike).
_CONFUSABLE = str.maketrans({"o": "0", "q": "0", "l": "1", "i": "1", "|": "1", "s": "5", "b": "8", "z": "2"})
_NUMERIC = re.compile(r"^[\d.,%\s−-]+$")


def tesseract_path() -> str | None:
    return shutil.which("tesseract")


def tesseract_version() -> str | None:
    path = tesseract_path()
    if path is None:
        return None
    out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
    return (out.stdout or out.stderr).splitlines()[0].strip()


def is_numeric(text: str) -> bool:
    return bool(_NUMERIC.match(text))


def normalize(text: str) -> str:
    text = text.replace("−", "-").replace("–", "-").replace("—", "-").lower().translate(_CONFUSABLE)
    return re.sub(r"[^0-9a-z.%-]", "", text)


def matches(expected: str, got: str) -> bool:
    return normalize(expected) == normalize(got) != ""


def prepare(image: Image.Image, item: LayoutItem, scale: int = 4, soft: bool = False) -> Image.Image | None:
    """Crop one label, mask out its neighbors, level it, and make it dark on light for Tesseract.

    By default the crop is thresholded to pure black on white; with soft=True it stays grayscale,
    which reads small digits better but some words worse, so it is used as a second attempt.
    """
    # Crop tight to the label (its box is the glyphs' extent) so neighboring bars, ticks and
    # gridlines stay out, then pad with the label's own background color.
    box, pad = item.bbox, 1
    left, top = max(0, math.floor(box.x0) - pad), max(0, math.floor(box.y0) - pad)
    right, bottom = min(image.width, math.ceil(box.x1) + pad), min(image.height, math.ceil(box.y1) + pad)
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    rgb = np.asarray(crop)
    ring = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    fill = tuple(int(v) for v in np.median(ring, axis=0))
    if item.quad:  # rotated text: keep only its own slanted strip, then rotate it level
        mask = Image.new("L", crop.size, 0)
        grown = _grow([(x - left, y - top) for x, y in item.quad], pad)
        ImageDraw.Draw(mask).polygon(grown, fill=255)
        crop = Image.composite(crop, Image.new("RGB", crop.size, fill), mask)
        crop = crop.rotate(-item.rotation, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
    crop = ImageOps.expand(crop, border=4, fill=fill)
    gray = crop.convert("L").resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=float)
    low, high = pixels.min(), pixels.max()
    if high - low < 40:
        return None  # nothing legible there
    ink = pixels < (low + high) / 2
    if ink.mean() > 0.5:  # light text on a dark fill: the background is the majority
        ink, pixels = ~ink, 255 - pixels
    out = pixels if soft else np.where(ink, 0, 255)
    return ImageOps.expand(Image.fromarray(out.astype(np.uint8)), border=24, fill=255)


def _grow(quad: list[tuple[float, float]], by: float) -> list[tuple[float, float]]:
    cx, cy = sum(x for x, _ in quad) / len(quad), sum(y for _, y in quad) / len(quad)
    out = []
    for x, y in quad:
        dx, dy = x - cx, y - cy
        norm = math.hypot(dx, dy) or 1.0
        out.append((x + dx / norm * by, y + dy / norm * by))
    return out


def read_text(prepared: Image.Image, digits_only: bool = False) -> str:
    buffer = io.BytesIO()
    prepared.save(buffer, format="PNG")
    command = [tesseract_path() or "tesseract", "stdin", "stdout", "--psm", "7", "-l", "eng"]
    if digits_only:
        command += ["-c", "tessedit_char_whitelist=0123456789.,%-"]
    out = subprocess.run(command, input=buffer.getvalue(), capture_output=True, timeout=60)
    return out.stdout.decode("utf-8", "replace").strip()


def read_label(image: Image.Image, item: LayoutItem) -> tuple[str, int, Image.Image | None]:
    """OCR text for one label, which attempt matched (1 or 2; 0 if neither), and the last crop tried."""
    expected = item.text or ""
    prepared = prepare(image, item)
    got = read_text(prepared) if prepared is not None else ""
    if matches(expected, got):
        return got, 1, prepared
    second = prepare(image, item, soft=True)
    if second is None:
        return got, 0, prepared
    retry = read_text(second, digits_only=is_numeric(expected))
    if matches(expected, retry):
        return retry, 2, second
    return got, 0, prepared


def _labels(task: Task, layout: Any) -> list[LayoutItem]:
    found: dict[str, LayoutItem] = {}
    for ref in task.depends_on:
        for item in layout.find(ref):
            if item.kind == "text" and item.text:
                found[item.element_id] = item
    return list(found.values())


def run_ocr(run_dir: Path, workers: int = 8, progress: Callable[[int], None] | None = None) -> dict[str, Any]:
    """Read back every label the run's accepted tasks depend on; write ocr_checks.jsonl and ocr_summary.json."""
    if tesseract_path() is None:
        raise FileNotFoundError("Tesseract is not installed (macOS: brew install tesseract; Ubuntu: apt install tesseract-ocr)")
    tasks = read_tasks(run_dir / "tasks.jsonl")
    by_image: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_image[task.image].append(task)

    jobs: list[tuple[str, LayoutItem, Image.Image]] = []
    needed: dict[str, list[str]] = {}
    for image_key, group in by_image.items():
        family = get_family(group[0].family)
        _, layout = family.render(family.scene_from_dict(group[0].scene))
        image = Image.open(run_dir / image_key).convert("RGB")
        items: dict[str, LayoutItem] = {}
        for task in group:
            labels = _labels(task, layout)
            needed[task.id] = [item.element_id for item in labels]
            items.update({item.element_id: item for item in labels})
        jobs += [(image_key, item, image) for item in items.values()]

    def check(job: tuple[str, LayoutItem, Image.Image]) -> dict[str, Any]:
        image_key, item, image = job
        got, attempt, prepared = read_label(image, item)
        if progress:
            progress(1)
        return {"image": image_key, "element": item.element_id, "expected": item.text, "got": got,
                "ok": attempt > 0, "attempt": attempt, "_prepared": prepared}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check, jobs))

    crops = run_dir / "ocr"
    crops.mkdir(exist_ok=True)
    for old in crops.glob("*.png"):
        old.unlink()
    misses = []
    for n, result in enumerate(r for r in results if not r["ok"]):
        if result["_prepared"] is not None:
            path = crops / f"miss-{n:04d}.png"
            result["_prepared"].save(path)
            result["crop"] = str(path.relative_to(run_dir))
        misses.append(result)
    for result in results:
        result.pop("_prepared", None)

    ok = {(r["image"], r["element"]) for r in results if r["ok"]}
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"labels": 0, "read_back": 0})
    for r in results:
        kind = r["element"].split(":")[0]
        by_kind[kind]["labels"] += 1
        by_kind[kind]["read_back"] += int(r["ok"])
    fully = sum(all((t.image, e) in ok for e in needed[t.id]) for t in tasks)
    summary = {
        "engine": tesseract_version(),
        "labels": len(results),
        "read_back": len(ok),
        "read_on_second_attempt": sum(r["attempt"] == 2 for r in results),
        "tasks": len(tasks),
        "tasks_fully_read_back": fully,
        "by_kind": dict(sorted(by_kind.items())),
        "misses": [{k: r.get(k) for k in ("image", "element", "expected", "got", "crop")} for r in misses],
    }
    (run_dir / "ocr_checks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (run_dir / "ocr_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
