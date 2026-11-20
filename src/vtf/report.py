"""Writes report.html for a run: the rejection funnel, coverage, the always-guess baseline, and
real accepted and rejected examples. Example images are embedded so the file stands alone."""

from __future__ import annotations

import base64
import html
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .core.schema import GateResult, Task

CSS = """
:root { --bg:#ffffff; --fg:#1c1c1e; --muted:#6c6c72; --line:#e4e4e7; --card:#f7f7f8; --bad:#d92d20; --good:#067647; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141416; --fg:#ececef; --muted:#a1a1aa; --line:#2e2e33; --card:#1c1c1f; --bad:#f97066; --good:#47cd89; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 28px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 36px 0 8px; }
p.note { color: var(--muted); margin: 0 0 12px; max-width: 760px; }
.muted { color: var(--muted); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 18px; }
.tile { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; background: var(--card); }
.tile b { display: block; font-size: 24px; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-size: 13px; font-weight: 600; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; }
.cols h3 { font-size: 14px; margin: 0 0 4px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 16px; }
.card { border: 1px solid var(--line); border-radius: 10px; background: var(--card); padding: 12px; }
.shot { position: relative; margin-bottom: 10px; }
.shot img { display: block; width: 100%; height: auto; border-radius: 6px; }
.box { position: absolute; border: 2px solid var(--bad); border-radius: 3px; }
.q { font-weight: 600; margin: 0 0 6px; }
.a { margin: 0 0 6px; }
.why { color: var(--bad); margin: 0 0 6px; }
ol { margin: 4px 0 8px; padding-left: 20px; color: var(--muted); font-size: 13px; }
.tag { font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }
"""


def write_report(out_dir: Path, summary: dict[str, Any], accepted: list[Task], rejected: list[Task]) -> Path:
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_e(summary['family'])} tasks, seed {summary['seed']}</title><style>{CSS}</style></head><body><main>",
        _header(summary),
        _funnel(summary),
        _coverage(summary),
        _prior(summary),
        _accepted_examples(out_dir, accepted),
        _rejected_examples(out_dir, rejected),
        "</main></body></html>",
    ]
    path = out_dir / "report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _e(value: Any) -> str:
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[Any]], numeric: set[int] = frozenset()) -> str:
    def cell(tag: str, i: int, value: Any) -> str:
        cls = " class='num'" if i in numeric else ""
        return f"<{tag}{cls}>{_e(value)}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _header(s: dict[str, Any]) -> str:
    det = s["determinism"]
    tiles = [
        ("Accepted tasks", s["accepted"]),
        ("Acceptance rate", f"{s['acceptance_rate']:.0%}"),
        ("Candidates checked", s["candidates"]),
        ("Images rendered", s["scenes_rendered"]),
        ("Seed rebuilds identical", f"{det['passed']}/{det['checked']}"),
    ]
    tile_html = "".join(f"<div class='tile'><b>{_e(v)}</b><span class='muted'>{_e(k)}</span></div>" for k, v in tiles)
    versions = ", ".join(f"{k} {v}" for k, v in s["versions"].items())
    return (
        f"<h1>Visual task factory: {_e(s['family'])}</h1>"
        f"<div class='muted'>Seed {s['seed']} · {s['elapsed_seconds']}s · {_e(versions)}</div>"
        f"<div class='tiles'>{tile_html}</div>"
    )


def _funnel(s: dict[str, Any]) -> str:
    rows = [["Candidates", "", s["candidates"]]]
    rows += [[row["gate"], row["rejected"], row["remaining"]] for row in s["funnel"]]
    reasons = [[reason, count] for reason, count in s["rejection_reasons"].items()]
    return (
        "<h2>Rejection funnel</h2>"
        "<p class='note'>Every candidate goes through every gate; each rejection is counted at the first gate it failed.</p>"
        f"<div class='cols'><div>{_table(['Gate', 'Rejected', 'Remaining'], rows, {1, 2})}</div>"
        f"<div>{_table(['Reason', 'Count'], reasons or [['none', 0]], {1})}</div></div>"
    )


def _coverage(s: dict[str, Any]) -> str:
    blocks = []
    for facet, counts in s["coverage"].items():
        rows = [[value, count] for value, count in counts.items()]
        blocks.append(f"<div><h3>{_e(facet.capitalize())}</h3>{_table(['Value', 'Tasks'], rows, {1})}</div>")
    return "<h2>Coverage of accepted tasks</h2>" f"<div class='cols'>{''.join(blocks)}</div>"


def _prior(s: dict[str, Any]) -> str:
    rows = [
        [
            template,
            p["tasks"],
            p["most_common_answer"],
            f"{p['share']:.0%}",
            f"{p['named_in_question']:.0%}" if "named_in_question" in p else "–",
        ]
        for template, p in s["answer_prior"].items()
    ]
    return (
        "<h2>Guessing without the image</h2>"
        "<p class='note'>Two shortcuts a model could use without looking: always giving a template's most "
        "common answer, or answering with a name the question itself mentions. High shares mean the "
        "template leaks its answer and needs rebalancing.</p>"
        + _table(
            ["Template", "Tasks", "Most common answer", "Always-guess score", "Answer named in question"],
            rows,
            {1, 3, 4},
        )
    )


def _accepted_examples(out_dir: Path, tasks: list[Task], limit: int = 12) -> str:
    by_template: dict[str, list[Task]] = {}
    for task in tasks:
        by_template.setdefault(task.template, []).append(task)
    picked: list[Task] = []
    images: set[str] = set()
    for group in by_template.values():  # one per template, each on a different image when possible
        choice = next((t for t in group if t.image not in images), group[0])
        picked.append(choice)
        images.add(choice.image)
    ids = {t.id for t in picked}
    for task in tasks:
        if len(picked) >= limit:
            break
        if task.id not in ids and task.image not in images:
            picked.append(task)
            images.add(task.image)
    cards = "".join(_card(out_dir, task, None) for task in picked[:limit])
    return f"<h2>Accepted examples</h2><div class='grid'>{cards}</div>"


def _rejected_examples(out_dir: Path, tasks: list[Task], per_reason: int = 2, limit: int = 12) -> str:
    counts: Counter[str] = Counter()
    images: set[str] = set()
    cards = []
    for task in tasks:
        gate, result = next((g, r) for g, r in task.validation.items() if r.status == "fail")
        key = f"{gate}:{result.code}"
        if counts[key] < per_reason and task.image not in images:
            counts[key] += 1
            images.add(task.image)
            cards.append(_card(out_dir, task, (gate, result)))
        if len(cards) >= limit:
            break
    note = "<p class='note'>Red boxes mark the elements that failed the check.</p>"
    return f"<h2>Rejected examples</h2>{note}<div class='grid'>{''.join(cards) or 'None.'}</div>"


def _card(out_dir: Path, task: Task, failure: tuple[str, GateResult] | None) -> str:
    image_path = out_dir / task.image
    with Image.open(image_path) as im:
        width, height = im.size
    data = base64.b64encode(image_path.read_bytes()).decode()
    boxes = ""
    if failure:
        for b in failure[1].boxes:
            style = (
                f"left:{b.x0 / width:.2%};top:{b.y0 / height:.2%};"
                f"width:{(b.x1 - b.x0) / width:.2%};height:{(b.y1 - b.y0) / height:.2%}"
            )
            boxes += f"<div class='box' style='{style}'></div>"
    answer = task.answer
    shown = f"{answer.value}"
    if answer.tolerance:
        shown += f" (± {answer.tolerance:g})"
    if answer.unit:
        shown += f" {answer.unit}"
    trace = "".join(f"<li>{_e(step)}</li>" for step in task.trace)
    if failure:
        gate, result = failure
        verdict = f"<p class='why'>{_e(gate)}: {_e(result.reason)}</p>"
    else:
        verdict = "<p class='tag'>" + " · ".join(f"{_e(g)} {_e(r.status)}" for g, r in task.validation.items()) + "</p>"
    return (
        "<div class='card'>"
        f"<div class='shot'><img src='data:image/png;base64,{data}' alt='{_e(task.question)}'>{boxes}</div>"
        f"<p class='q'>{_e(task.question)}</p>"
        f"<p class='a'>Answer: {_e(shown)}</p>"
        f"{verdict}<ol>{trace}</ol>"
        f"<div class='tag'>{_e(task.id)} · {_e(task.skill)}</div>"
        "</div>"
    )
