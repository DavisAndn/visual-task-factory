"""Photo mode: re-shoot a run's tasks as phone photos, keeping each exact answer.

People photograph instruments, screens and printouts and ask AI to read them. `vtf photos` turns
every image of a run into such a photo: the page seen at a slight angle on a desk, lit unevenly
with a soft glare, a little out of focus, with sensor noise and JPEG compression.

The answer can't change. A perspective transform maps straight lines to straight lines and keeps
which tick a needle points at and which pipe a valve sits on. The layout manifest goes through the
same transform, so the grounding and ambiguity gates run again on the photo itself (text that ends
up too small or details too fine are caught); `vtf ocr --photos` then reads the labels back from
the photo pixels. Each photo twin is written to photos.jsonl, next to the clean task it copies.
"""

from __future__ import annotations

import io
import json
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from ..families import get_family
from ..validators import AmbiguityGate, GroundingGate
from .family import Candidate, Question
from .pipeline import json_hash, pixel_hash
from .runs import read_tasks
from .schema import BBox, GateResult, LayoutItem, LayoutManifest, Task

CAPTURE_DEFAULTS: dict[str, Any] = {
    "tilt": 0.05,  # each corner moves up to this share of the image size: a camera a few degrees off square
    "blur": [0.7, 1.0],  # Gaussian blur radius, pixels
    "glare": [35, 55],  # brightness added at the center of a soft highlight
    "shade": 0.2,  # brightness falls by up to this share across the page
    "noise": 5.0,  # sensor noise, standard deviation in gray levels
    "jpeg_quality": 55,
}


def sample_capture(rng: random.Random, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Concrete settings for one photo; stored on each twin, so the photo can be rebuilt exactly."""
    c = {**CAPTURE_DEFAULTS, **(cfg or {})}
    t = c["tilt"]
    return {
        "corners": [[round(rng.uniform(-t, t), 5), round(rng.uniform(-t, t), 5)] for _ in range(4)],
        "shade": [round(rng.uniform(-1, 1), 4), round(rng.uniform(-1, 1), 4), c["shade"]],
        "glare": [round(rng.uniform(0.25, 0.75), 4), round(rng.uniform(0.25, 0.75), 4),
                  round(rng.uniform(0.12, 0.22), 4), round(rng.uniform(*c["glare"]), 2)],
        "blur": round(rng.uniform(*c["blur"]), 3),
        "noise": c["noise"],
        "noise_seed": rng.randrange(1 << 30),
        "jpeg_quality": c["jpeg_quality"],
    }


def _homography(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> np.ndarray:
    """The 3x3 projective transform taking each src point to its dst point."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    h = np.linalg.solve(np.array(rows, float), np.array(rhs, float))
    return np.append(h, 1.0).reshape(3, 3)


def _apply(m: np.ndarray, x: float, y: float) -> tuple[float, float]:
    u, v, w = m @ np.array([x, y, 1.0])
    return float(u / w), float(v / w)


def photograph(image: Image.Image, layout: LayoutManifest, capture: dict[str, Any]) -> tuple[Image.Image, LayoutManifest]:
    """The photo of a rendered image, and its layout manifest mapped onto the photo."""
    _, photo, mapped = shoot(image, layout, capture)
    return photo, mapped


def shoot(image: Image.Image, layout: LayoutManifest, capture: dict[str, Any]) -> tuple[bytes, Image.Image, LayoutManifest]:
    """The photo as JPEG file bytes, the same photo decoded, and the layout mapped onto it."""
    w, h = image.size
    W, H = int(w * 1.16), int(h * 1.22)
    ox, oy = (W - w) / 2, (H - h) / 2
    page = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    quad = [(ox + px + dx * w, oy + py + dy * h) for (px, py), (dx, dy) in zip(page, capture["corners"])]
    forward = _homography(page, quad)
    backward = np.linalg.inv(forward)
    backward = backward / backward[2, 2]
    coeffs = tuple(float(c) for c in backward.flatten()[:8])

    desk = Image.new("RGB", (W, H), (118, 104, 92))
    warped = image.convert("RGB").transform((W, H), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    mask = Image.new("L", (w, h), 255).transform((W, H), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    desk.paste(warped, (0, 0), mask)
    pixels = np.asarray(desk).astype(float)
    yy, xx = np.mgrid[0:H, 0:W]
    gx, gy, shade = capture["shade"]
    pixels *= (1 - shade * ((xx / W - 0.5) * gx + (yy / H - 0.5) * gy + 0.5))[..., None]
    cx, cy, r, glare = capture["glare"]
    cx, cy, r = cx * W, cy * H, r * W
    pixels += (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * glare)[..., None]
    pixels *= np.array([1.03, 1.0, 0.93])  # a warm color cast
    pixels += np.random.default_rng(capture["noise_seed"]).normal(0, capture["noise"], pixels.shape)
    shot = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(capture["blur"]))
    buffer = io.BytesIO()
    shot.save(buffer, "JPEG", quality=capture["jpeg_quality"])
    photo = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return buffer.getvalue(), photo, _map_layout(layout, forward, W, H)


def _local_scale(m: np.ndarray, x: float, y: float) -> float:
    """How much the transform stretches lengths near a point (the smaller of the two directions)."""
    p = _apply(m, x, y)
    dx, dy = _apply(m, x + 1, y), _apply(m, x, y + 1)
    return min(float(np.hypot(dx[0] - p[0], dx[1] - p[1])), float(np.hypot(dy[0] - p[0], dy[1] - p[1])))


def _map_layout(layout: LayoutManifest, m: np.ndarray, W: int, H: int) -> LayoutManifest:
    items = []
    for item in layout.items:
        quad = [_apply(m, x, y) for x, y in item.polygon()]
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        bbox = BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
        center = ((item.bbox.x0 + item.bbox.x1) / 2, (item.bbox.y0 + item.bbox.y1) / 2)
        font = None if item.font_px is None else round(item.font_px * _local_scale(m, *center), 2)
        items.append(LayoutItem(element_id=item.element_id, kind=item.kind, bbox=bbox, text=item.text, font_px=font,
                                rotation=item.rotation, quad=quad if len(quad) == 4 else None))
    meta = dict(layout.meta)
    if meta.get("min_feature_px") is not None:
        corners = [(0, 0), (layout.width, 0), (layout.width, layout.height), (0, layout.height)]
        meta["min_feature_px"] = round(meta["min_feature_px"] * min(_local_scale(m, x, y) for x, y in corners), 2)
    meta.pop("probes", None)  # probe points are for the clean render
    return LayoutManifest(width=W, height=H, items=items, meta=meta)


def task_view(task: Task) -> tuple[Image.Image, LayoutManifest]:
    """The image and layout a task shows: its render, photographed if the task is a photo twin."""
    family = get_family(task.family)
    image, layout = family.render(family.scene_from_dict(task.scene))
    if task.capture:
        return photograph(image, layout, task.capture)
    return image, layout


def make_photos(
    run_dir: Path,
    share: float = 1.0,
    seed: int = 0,
    capture_cfg: dict[str, Any] | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Photograph a share of a run's images; write photos.jsonl and photos_summary.json."""
    cfg = json.loads((run_dir / "config.json").read_text())
    v = cfg["validation"]
    gates = [
        GroundingGate(v["min_font_px"], v["min_mark_px"], v["overlap_tol_px"], v["clean_text"]),
        AmbiguityGate(v["visual_gap"], v["max_relative_tolerance"], v.get("min_feature_px", 4.0)),
    ]
    rng = random.Random(seed)
    tasks = read_tasks(run_dir / "tasks.jsonl")
    images = sorted({t.image for t in tasks})
    chosen = {key for key in images if rng.random() < share}
    made, skipped = 0, Counter()
    with (run_dir / "photos.jsonl").open("w") as out:
        for key in images:
            group = [t for t in tasks if t.image == key]
            if key not in chosen:
                if progress:
                    progress(len(group))
                continue
            family = get_family(group[0].family)
            scene = family.scene_from_dict(group[0].scene)
            image, layout = family.render(scene)
            capture = sample_capture(random.Random(f"{seed}:{key}"), capture_cfg)
            jpeg, photo, photo_layout = shoot(image, layout, capture)
            photo_key = key.replace(".png", "-photo.jpg")  # a photo is a JPEG
            saved = False
            for task in group:
                template = next(t for t in family.templates if t.id == task.template)
                solution = template.solve(scene, task.params, photo_layout)
                question = Question(task.template, task.skill, task.question, task.params, task.depends_on)
                cand = Candidate(scene, photo_layout, question, solution, photo_key, 0)
                results: dict[str, GateResult] = {gate.name: gate.check(cand) for gate in gates}
                failed = next((f"{name}: {r.code}" for name, r in results.items() if r.status == "fail"), None)
                if failed or solution.answer != task.answer:
                    skipped[failed or "answer changed"] += 1
                    continue
                if not saved:
                    (run_dir / photo_key).write_bytes(jpeg)
                    saved = True
                twin = task.model_copy(update={"id": f"{task.id}-photo", "image": photo_key, "pixel_hash": pixel_hash(photo),
                                               "validation": results, "capture": capture})
                out.write(twin.model_dump_json() + "\n")
                made += 1
            if progress:
                progress(len(group))
    summary = {
        "share": share,
        "seed": seed,
        "tasks": sum(1 for t in tasks if t.image in chosen),
        "photos": made,
        "skipped": dict(skipped.most_common()),
        "capture": {**CAPTURE_DEFAULTS, **(capture_cfg or {})},
        "settings_hash": json_hash({**CAPTURE_DEFAULTS, **(capture_cfg or {})}),
    }
    (run_dir / "photos_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
