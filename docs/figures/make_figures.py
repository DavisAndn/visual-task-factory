"""Builds the deck figures from the committed runs (defaults: samples/charts-seed42 and
samples/charts-hard-seed7).

    uv run python docs/figures/make_figures.py [run_dir] [hard_run_dir]

Writes examples.png, contrast-pair.png and pipeline.png next to this file, plus model-accuracy.png
and twin-miss.png when the runs have model results, and families.png, photos.png and
breaking-points.png from the instruments, connections and counting samples.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
INK, MUTED = "#1c1c1e", "#5f5f66"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def _tasks(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "tasks.jsonl").read_text().splitlines()]


def _answer(answer: dict) -> str:
    shown = str(answer["value"])
    if answer["tolerance"]:
        shown += f" ± {answer['tolerance']:g}"
    if answer.get("unit") == "%":
        shown += "%"
    elif answer.get("unit"):
        shown += f" ({answer['unit']})"
    return shown


def examples(run: Path) -> None:
    tasks = _tasks(run)
    order = ["bar", "hbar", "grouped_bar", "stacked_bar", "line", "pie"]
    wanted = {"bar": "rank_then_read", "hbar": "count_above", "grouped_bar": "largest_increase",
              "stacked_bar": "largest_segment", "line": "trend_decreases", "pie": "pie_share"}
    picks = []
    for chart in order:
        group = [t for t in tasks if t["scene"]["chart_type"] == chart]
        picks.append(next((t for t in group if t["template"] == wanted[chart]), group[0]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=120)
    for ax, task in zip(axes.flat, picks):
        ax.imshow(Image.open(run / task["image"]))
        ax.set_axis_off()
        question = "\n".join(textwrap.wrap(task["question"], 58))
        ax.set_title(f"{question}\nAnswer: {_answer(task['answer'])}", fontsize=10.5, loc="left", pad=6)
    fig.suptitle("One task per chart type, from the committed sample run", fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "examples.png", facecolor="white")
    plt.close(fig)


def contrast_pair(run: Path) -> None:
    tasks = {t["id"]: t for t in _tasks(run)}
    pairs = [json.loads(line) for line in (run / "pairs.jsonl").read_text().splitlines()]
    pair = next(p for p in pairs if tasks[p["a"]]["template"] == "compare_pair" and p["size"] <= 0.15)
    original = tasks[pair["a"]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=120)
    for ax, task, side in ((axes[0], original, "Original"), (axes[1], pair["b"], "Twin")):
        ax.imshow(Image.open(run / task["image"]))
        ax.set_axis_off()
        ax.set_title(f"{side}: answer {_answer(task['answer'])}", fontsize=14, loc="left")
    fig.suptitle(
        f"Contrast pair. Same question: “{original['question']}”\n"
        f"One value edited ({pair['change']}, {pair['size']:.0%} of the axis); the axis is pinned so nothing else moves.",
        fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / "contrast-pair.png", facecolor="white")
    plt.close(fig)


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_axis_off()

    def box(x, y, w, h, title, body, face, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=face, ec=edge, lw=1.6))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(x + w / 2, y + h * 0.3, body, ha="center", va="center", fontsize=10, color=MUTED, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": "#6b7280", "lw": 1.6})

    gen_face, gen_edge = "#E8F0FE", "#4E79A7"
    ev_face, ev_edge = "#E6F4EA", "#3A8F4F"
    out_face, out_edge = "#F3F4F6", "#6B7280"
    ax.text(0.3, 8.45, "Generation: every answer comes from the data", fontsize=15, weight="bold")
    steps = [
        ("Seed", "one integer\nper image"),
        ("Scene", "values, labels,\nstyle"),
        ("Render", "image + layout\nrecord"),
        ("Ask", "template picks\nparameters"),
        ("Solve", "answer, trace,\ntolerance"),
        ("Gates", "determinism, grounding,\nambiguity, dedup, balance"),
    ]
    w, gap, y = 2.2, 0.42, 6.3
    for i, (title, body) in enumerate(steps):
        x = 0.3 + i * (w + gap)
        box(x, y, w + (0.3 if title == "Gates" else 0), 1.6, title, body, gen_face, gen_edge)
        if i:
            arrow(x - gap + 0.02, y + 0.8, x - 0.02, y + 0.8)

    # Gates split into accepted and rejected, right under the gates box.
    box(11.0, 4.2, 2.45, 1.25, "Accepted", "tasks.jsonl + images", out_face, out_edge)
    box(13.75, 4.2, 2.05, 1.25, "Rejected", "with the reason\nand red boxes", out_face, out_edge)
    arrow(14.0, 6.28, 12.6, 5.47)  # the gates box spans x 13.4 to 15.9
    arrow(15.2, 6.28, 14.9, 5.47)

    line = {"color": "#6b7280", "lw": 1.6}
    ax.text(0.3, 5.0, "Evidence: checks that don't trust the generator", fontsize=15, weight="bold")
    evidence = [
        ("Contrast pairs", "a minimal edit flips\nthe answer"),
        ("OCR read-back", "Tesseract reads every\nlabel a question needs"),
        ("Model evaluation", "with and without the\nimage, by skill"),
        ("Human audit", "blind answers,\n95% interval"),
    ]
    w, gap, top, bottom = 2.4, 0.25, 3.75, 1.95
    centers = [0.3 + i * (w + gap) + w / 2 for i in range(len(evidence))]
    # One bus from Accepted across the top of the evidence row, one along its bottom to the report.
    ax.plot([11.0, centers[0]], [4.25, 4.25], **line)
    ax.plot([centers[0], centers[-1]], [1.55, 1.55], **line)
    for (title, body), cx in zip(evidence, centers):
        box(cx - w / 2, bottom, w, top - bottom, title, body, ev_face, ev_edge)
        arrow(cx, 4.25, cx, top + 0.03)
        ax.plot([cx, cx], [bottom - 0.02, 1.55], **line)
    mid = (centers[0] + centers[-1]) / 2
    arrow(mid, 1.55, mid, 1.17)
    box(mid - 1.6, 0.2, 3.2, 0.95, "report.html", "one self-contained page per run", out_face, out_edge)
    fig.savefig(HERE / "pipeline.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


SURFACE, GRID, AXIS, SECONDARY = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#52514e"
LIGHT_BLUE, BLUE = "#86b6ef", "#2a78d6"  # one hue, two steps (validated as an ordinal pair)


def _dumbbell(ax, table: dict[str, dict], title: str, order: list[str]) -> None:
    """Without the image (light dot) to with it (dark dot), one row per group, in the given order."""
    ax.set_facecolor(SURFACE)
    labels = []
    for y, name in enumerate(order):
        row = table.get(name)
        labels.append(f"{name.replace('_', ' ')}  ({row['tasks'] if row else 0})")
        if not row:
            continue
        blind, seen = row["blind"] * 100, row["with_image"] * 100
        ax.plot([blind, seen], [y, y], color=AXIS, lw=2, solid_capstyle="round", zorder=1)
        ax.scatter([blind], [y], s=90, color=LIGHT_BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.scatter([seen], [y], s=90, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.text(seen + 2.2, y, f"{seen:.0f}%", va="center", fontsize=11.5, color=INK)
    rows = order
    ax.set_yticks(range(len(rows)), labels)
    ax.tick_params(axis="y", length=0, labelsize=11.5, labelcolor=INK, pad=8)
    ax.tick_params(axis="x", colors=MUTED, labelsize=10.5, length=0)
    ax.set_xlim(-2, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_title(title, loc="left", fontsize=13.5, color=INK, pad=10, weight="bold")


def _pct(counts: dict) -> str:
    return f"{counts['correct'] / counts['of']:.0%}" if counts["of"] else "–"


def model_accuracy(standard: Path, hard: Path) -> bool:
    """Accuracy by skill on the standard set and the hard set, side by side on one scale."""
    paths = [standard / "model_summary.json", hard / "model_summary.json"]
    if not all(p.is_file() for p in paths):
        return False
    def opus_first(path: Path) -> tuple[str, dict]:
        models = json.loads(path.read_text())["models"]
        return next(((k, v) for k, v in models.items() if "opus" in k), next(iter(models.items())))

    (name, std), (_, hrd) = (opus_first(p) for p in paths)
    display = name.replace("claude-opus-5-5", "Claude Opus 5.5").replace("claude-sonnet-5-5", "Claude Sonnet 5.5")
    skills = set(std["by_skill"]) | set(hrd["by_skill"])

    def rank(skill: str) -> tuple[float, float]:
        # Weakest with the image on top; ties broken by how guessable the skill is without it.
        row = hrd["by_skill"].get(skill) or std["by_skill"][skill]
        return (row["with_image"], -std["by_skill"].get(skill, row)["blind"])

    order = sorted(skills, key=rank)
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 9), dpi=120, sharey=False)
    fig.patch.set_facecolor(SURFACE)
    _dumbbell(left, std["by_skill"], f"Standard set ({std['tasks']} tasks)", order)
    _dumbbell(right, hrd["by_skill"], f"Hard set ({hrd['tasks']} tasks): nothing printed, crowded charts", order)
    fig.text(0.012, 0.965, f"{display}: accuracy by skill, with the image and without it", fontsize=17, weight="bold", color=INK)
    def median(s: dict) -> str:
        return f"{s['reading_precision']['median_error']:.1%}" if s.get("reading_precision") else "–"

    fig.text(
        0.012, 0.928,
        f"Standard: {_pct(std['with_image'])} with the image, {_pct(std['blind'])} without. "
        f"Hard: {_pct(hrd['with_image'])} with the image, {_pct(hrd['blind'])} without. Task counts in brackets.",
        fontsize=11.5, color=SECONDARY,
    )
    fig.text(
        0.012, 0.898,
        f"Values that aren't printed were read with a median error of {median(std)} (standard) and {median(hrd)} (hard).",
        fontsize=11.5, color=SECONDARY,
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=BLUE, mec=SURFACE, label="With the image"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=LIGHT_BLUE, mec=SURFACE, label="Without the image"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.985), frameon=False, fontsize=11.5,
               labelcolor=SECONDARY, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=4)
    fig.savefig(HERE / "model-accuracy.png", facecolor=SURFACE)
    plt.close(fig)
    return True


def twin_miss(run: Path, model_prefix: str = "claude-sonnet-5-5") -> bool:
    """A contrast pair a model got half wrong: same question, one value edited, same old answer."""
    checks = run / "model_checks.jsonl"
    if not checks.is_file():
        return False
    records = [json.loads(line) for line in checks.read_text().splitlines()]
    miss = next((r for r in records if r["model"].startswith(model_prefix) and "reply" in r.get("pair", {})
                 and r["image"]["correct"] and not r["pair"]["correct"]), None)
    if miss is None:
        return False
    tasks = {t["id"]: t for t in _tasks(run)}
    pair = next(json.loads(l) for l in (run / "pairs.jsonl").read_text().splitlines() if json.loads(l)["a"] == miss["task"])
    original, twin = tasks[miss["task"]], pair["b"]
    label = miss["model"].split(" ")[0].replace("claude-sonnet-5-5", "Claude Sonnet 5.5").replace("claude-haiku-5-5", "Claude Haiku 5.5")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), dpi=120)
    for ax, task, reply, ok in ((axes[0], original, miss["image"]["reply"], True), (axes[1], twin, miss["pair"]["reply"], False)):
        ax.imshow(Image.open(run / task["image"]))
        ax.set_axis_off()
        said = reply.strip().splitlines()[-1].strip("* ")
        verdict = "right" if ok else "wrong"
        ax.set_title(f"Reference: {_answer(task['answer'])}    {label} said: {said} ({verdict})", fontsize=13, loc="left",
                     color=INK if ok else "#b42318")
    fig.suptitle(
        f"A mistake the contrast pairs caught: \u201c{original['question']}\u201d\n"
        f"Right image: {pair['change']}. The printed values now favor a different answer; the model repeated its first one.",
        fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(HERE / "twin-miss.png", facecolor="white")
    plt.close(fig)
    return True



# Three models, three categorical slots in a fixed order (validated all-pairs for color-blind
# readers); every series is also labelled directly, since the third slot is light on white.
MODEL_COLORS = {"Haiku": "#2a78d6", "Sonnet": "#eb6834", "Opus": "#1baf7a"}
MODEL_ORDER = ["Haiku", "Sonnet", "Opus"]


def _models(run: Path) -> dict[str, dict]:
    path = run / "model_summary.json"
    if not path.is_file():
        return {}
    models = json.loads(path.read_text())["models"]
    short = {}
    for name, m in models.items():
        for label in MODEL_ORDER:
            if f"claude-{label.lower()}" in name:
                short[label] = m
    return short


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=8, weight="bold")
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10.5, length=0)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])


def families(samples: Path) -> bool:
    """One task from each kind of image the factory makes, with its question and computed answer."""
    picks = [("charts-seed42", lambda t: t["scene"].get("chart_type") == "grouped_bar"),
             ("instruments-seed42", lambda t: t["scene"].get("kind") == "syringe" and t["scene"]["level"] == 4),
             ("instruments-seed42", lambda t: t["scene"].get("kind") == "gauge" and t["scene"]["level"] >= 3),
             ("instruments-seed42", lambda t: t["scene"].get("kind") == "clock" and t["scene"]["level"] == 3),
             ("connections-seed42", lambda t: t["scene"].get("kind") == "valves" and t["scene"]["level"] == 2),
             ("connections-seed42", lambda t: t["scene"].get("kind") == "gears" and t["scene"]["level"] == 2),
             ("connections-seed42", lambda t: t["scene"].get("kind") == "cables" and t["scene"]["level"] == 3),
             ("counting-seed42", lambda t: t["scene"].get("level") == 4)]
    chosen = []
    for run, wanted in picks:
        if not (samples / run / "tasks.jsonl").is_file():
            return False
        task = next((t for t in _tasks(samples / run) if wanted(t)), None)
        if task is None:
            return False
        chosen.append((samples / run, task))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=110)
    fig.patch.set_facecolor("white")
    for ax, (run, task) in zip(axes.flat, chosen):
        ax.imshow(Image.open(run / task["image"]))
        ax.set_axis_off()
        question = "\n".join(textwrap.wrap(task["question"], 52)[:3])
        ax.set_title(f"{question}\nAnswer: {_answer(task['answer'])}", fontsize=10.5, loc="left", pad=6)
    fig.suptitle("Charts, instruments, connection diagrams and dense plots: every answer computed from the scene",
                 fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "families.png", facecolor="white")
    plt.close(fig)
    return True


def photos(samples: Path) -> bool:
    """Accuracy on the clean image and on its phone photo, per kind of image and model."""
    rows = []  # (label, {model: (clean, photo, n)})
    for run, facet in (("charts-seed42", None), ("instruments-seed42", "instrument"),
                       ("connections-seed42", "diagram"), ("counting-seed42", None)):
        models = _models(samples / run)
        if not models:
            continue
        if facet is None:
            label = "charts" if run.startswith("charts") else "dense plots"
            rows.append((label, {m: (v["photos"]["clean_correct"] / v["photos"]["of"], v["photos"]["photo_correct"] / v["photos"]["of"],
                                     v["photos"]["of"]) for m, v in models.items() if v.get("photos", {}).get("of")}))
            continue
        kinds = sorted({k for v in models.values() for k in v["by_facet"].get(facet, {})})
        for kind in kinds:
            cells = {}
            for m, v in models.items():
                row = v["by_facet"][facet].get(kind, {})
                if "photo" in row:
                    cells[m] = (row["clean_of_photographed"], row["photo"], row["photo_tasks"])
            rows.append((kind + ("s" if not kind.endswith("s") else ""), cells))
    rows = [r for r in rows if r[1]]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(14, 1.1 + 0.95 * len(rows)), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    offsets = {"Haiku": -0.25, "Sonnet": 0.0, "Opus": 0.25}
    for y, (label, cells) in enumerate(rows):
        for m in MODEL_ORDER:
            if m not in cells:
                continue
            clean, photo, n = cells[m]
            yy = y + offsets[m]
            ax.plot([clean * 100, photo * 100], [yy, yy], color=MODEL_COLORS[m], lw=2, alpha=0.55, zorder=1)
            ax.scatter([clean * 100], [yy], s=55, facecolors=SURFACE, edgecolors=MODEL_COLORS[m], linewidths=1.8, zorder=3)
            ax.scatter([photo * 100], [yy], s=60, color=MODEL_COLORS[m], edgecolors=SURFACE, linewidths=1, zorder=3)
            ax.text(min(clean, photo) * 100 - 1.5, yy, f"{m} {photo:.0%}", ha="right", va="center", fontsize=9.5, color=SECONDARY)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=12, labelcolor=INK, pad=10)
    ax.tick_params(axis="x", colors=MUTED, labelsize=10.5, length=0)
    ax.set_xlim(20, 103)
    ax.set_xticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_title("The same tasks, clean (hollow dot) and photographed with a phone (filled dot)", loc="left",
                 fontsize=14, color=INK, pad=12, weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "photos.png", facecolor=SURFACE)
    plt.close(fig)
    return True


def breaking_points(samples: Path) -> bool:
    """Accuracy against difficulty level, per model: where each one breaks."""
    panels = [("counting-seed42", "dots", "with_image", "Counting dots", ["~30", "~70", "~150", "~300", "~515", "~800"]),
              ("connections-seed42", "cables", "with_image", "Tracing one-color cables", ["8 col.", "8", "16", "24", "32"]),
              ("instruments-seed42", "syringe", "photo", "Reading photographed syringes", ["20-60 mL", "5-10 mL", "3 mL, insulin", "1 mL"])]
    drawn = []
    for run, kind, key, title, ticks in panels:
        models = _models(samples / run)
        series = {}
        for m, v in models.items():
            table = v["by_facet"].get("difficulty", {})
            levels = sorted((int(k.rpartition(" L")[2]), row) for k, row in table.items() if k.startswith(kind + " L"))
            series[m] = [(lvl, row.get(key)) for lvl, row in levels if row.get(key) is not None]
        if any(series.values()):
            drawn.append((title, ticks, series))
    if not drawn:
        return False
    fig, axes = plt.subplots(1, len(drawn), figsize=(6.2 * len(drawn), 5.4), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    axes = axes if len(drawn) > 1 else [axes]
    for ax, (title, ticks, series) in zip(axes, drawn):
        _style(ax, title)
        ax.axhspan(0, 80, color="#f3f2ee", zorder=0)
        ends = []
        for m in MODEL_ORDER:
            points = series.get(m) or []
            if not points:
                continue
            xs, ys = [p[0] for p in points], [p[1] * 100 for p in points]
            ax.plot(xs, ys, color=MODEL_COLORS[m], lw=2, marker="o", ms=7, mec=SURFACE, zorder=3)
            ends.append([ys[-1], xs[-1], m])
        # End labels, nudged apart so models that finish at the same score stay readable.
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 6)
        for y, x, m in ends:
            ax.text(x + 0.12, y, m, va="center", fontsize=10.5, color=INK)
        ax.set_xticks(range(1, len(ticks) + 1), ticks)
        ax.set_xlim(0.7, len(ticks) + 0.8)
    fig.text(0.012, 0.955, "Where each model breaks: accuracy as the dial turns (shaded: below 80%)", fontsize=15,
             weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=3)
    fig.savefig(HERE / "breaking-points.png", facecolor=SURFACE)
    plt.close(fig)
    return True


# Every model, coloured by lab with the light steps of the validated palette (one hue per lab),
# and every row labelled directly with its model's name.
LAB_COLORS = {"claude": "#eb6834", "gpt": "#2a78d6", "qwen": "#e87ba4", "gemma": "#1baf7a"}


def _wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 1.0
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _display(name: str) -> str:
    model = name.split(" ")[0].removeprefix("cerebras/")
    if model.startswith("claude-"):
        family, major, minor = model.split("-")[1:4]
        return f"Claude {family.capitalize()} {major}.{minor}"
    if model.startswith("gpt-"):
        version, tier = model.removeprefix("gpt-").split("-", 1)
        return f"GPT-{version} {tier.capitalize()}"
    if model.startswith("qwen-"):
        _, version, size = model.split("-")
        return f"Qwen {version} {size.upper()}"
    return model


def same_tasks(samples: Path) -> bool:
    """Every model on the tasks they all answered: clean vs photographed, with 95% intervals."""
    runs = ["charts-seed42", "instruments-seed42", "connections-seed42", "counting-seed42"]
    records: dict[str, dict[str, dict]] = {}
    for run in runs:
        path = samples / run / "model_checks.jsonl"
        if not path.is_file():
            return False
        for line in path.read_text().splitlines():
            r = json.loads(line)
            records.setdefault(r["model"], {})[f"{run}/{r['task']}"] = r
    shared = set.intersection(*({k for k, r in recs.items() if "reply" in r.get("image", {})} for recs in records.values()))
    if len(records) < 4 or not shared:
        return False
    rows = []
    for name, recs in records.items():
        clean = photo = clean_p = n_photo = 0
        for key in shared:
            r = recs[key]
            clean += bool(r["image"]["correct"])
            if "reply" in r.get("photo", {}):
                n_photo += 1
                clean_p += bool(r["image"]["correct"])
                photo += bool(r["photo"]["correct"])
        rows.append((_display(name), name, clean_p, photo, n_photo))
    rows.sort(key=lambda r: -r[2])
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.62 * len(rows)), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for y, (label, name, clean_p, photo, n) in enumerate(rows):
        color = next(c for k, c in LAB_COLORS.items() if k in name)
        c, p = clean_p / n * 100, photo / n * 100
        lo, hi = (v * 100 for v in _wilson(photo, n))
        ax.plot([lo, hi], [y + 0.16, y + 0.16], color=color, lw=1.2, alpha=0.6, zorder=1)
        ax.plot([p, c], [y, y], color=color, lw=2, alpha=0.5, zorder=1)
        ax.scatter([c], [y], s=60, facecolors=SURFACE, edgecolors=color, linewidths=1.8, zorder=3)
        ax.scatter([p], [y], s=64, color=color, edgecolors=SURFACE, linewidths=1, zorder=3)
        ax.text(c + 1.2, y, f"{c:.0f}%", va="center", fontsize=10, color=SECONDARY)
        ax.text(p - 1.2, y, f"{p:.0f}%", va="center", ha="right", fontsize=10, color=INK, weight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=12, labelcolor=INK, pad=10)
    ax.tick_params(axis="x", colors=MUTED, labelsize=10.5, length=0)
    low = min(min(r[3] / r[4] for r in rows) * 100 - 12, 70)
    ax.set_xlim(max(0, low), 104)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(len(rows) - 0.5, -0.7)
    words = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
    ax.set_title(f"{words.get(len(rows), len(rows))} models on the same {len(shared)} tasks, clean and photographed",
                 loc="left", fontsize=14, color=INK, pad=30, weight="bold")
    ax.text(0, 1.03, "Hollow dot: clean image. Filled dot: its phone photo. Thin bar: 95% interval of the photo score.",
            transform=ax.transAxes, fontsize=11, color=MUTED)
    fig.tight_layout()
    fig.savefig(HERE / "same-tasks.png", facecolor=SURFACE)
    plt.close(fig)
    return True


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "samples" / "charts-seed42"
    hard = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "samples" / "charts-hard-seed7"
    examples(run)
    contrast_pair(run)
    pipeline()
    model_accuracy(run, hard)
    twin_miss(run)
    samples = ROOT / "samples"
    families(samples)
    photos(samples)
    breaking_points(samples)
    same_tasks(samples)
    print(f"Figures written to {HERE}")


if __name__ == "__main__":
    main()
