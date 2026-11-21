"""Builds the deck figures from the committed runs (defaults: samples/charts-seed42 and
samples/charts-hard-seed7).

    uv run python docs/figures/make_figures.py [run_dir] [hard_run_dir]

Writes examples.png, contrast-pair.png and pipeline.png next to this file, plus model-accuracy.png
and twin-miss.png when the runs have model results.
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
    (name, std), (_, hrd) = (next(iter(json.loads(p.read_text())["models"].items())) for p in paths)
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


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "samples" / "charts-seed42"
    hard = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "samples" / "charts-hard-seed7"
    examples(run)
    contrast_pair(run)
    pipeline()
    model_accuracy(run, hard)
    twin_miss(run)
    print(f"Figures written to {HERE}")


if __name__ == "__main__":
    main()
