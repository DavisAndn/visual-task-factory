"""Writes report.html for a run: the rejection funnel, coverage, shortcut checks, the evidence
gathered after generation (model evaluation, contrast pairs, OCR read-back, human audit), and real
accepted and rejected examples. Example images are embedded so the file stands alone."""

from __future__ import annotations

import base64
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .core.runs import read_json, read_pairs, read_tasks
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
h3 { font-size: 14px; margin: 16px 0 4px; }
p.note { color: var(--muted); margin: 0 0 12px; max-width: 760px; }
.muted { color: var(--muted); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 18px; }
.tile { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; background: var(--card); }
.tile b { display: block; font-size: 24px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-size: 13px; font-weight: 600; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; }
.cols h3 { margin-top: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 16px; }
.card { border: 1px solid var(--line); border-radius: 10px; background: var(--card); padding: 12px; }
.shot { position: relative; margin-bottom: 10px; }
.shot img { display: block; width: 100%; height: auto; border-radius: 6px; }
.box { position: absolute; border: 2px solid var(--bad); border-radius: 3px; }
.q { font-weight: 600; margin: 0 0 6px; }
.a { margin: 0 0 6px; }
.why { color: var(--bad); margin: 0 0 6px; }
.got { margin: 0 0 6px; }
.got.ok { color: var(--good); }
.got.no { color: var(--bad); }
ol { margin: 4px 0 8px; padding-left: 20px; color: var(--muted); font-size: 13px; }
.tag { font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }
.twins { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
"""


def rebuild_report(run_dir: Path) -> Path:
    """Rewrite report.html from the files in a run folder, picking up any evidence added since."""
    summary = read_json(run_dir / "summary.json")
    if summary is None:
        raise FileNotFoundError(f"{run_dir} has no summary.json; run vtf generate first")
    return write_report(run_dir, summary, read_tasks(run_dir / "tasks.jsonl"), read_tasks(run_dir / "rejected.jsonl"))


def write_report(out_dir: Path, summary: dict[str, Any], accepted: list[Task], rejected: list[Task]) -> Path:
    by_id = {task.id: task for task in accepted}
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_e(summary['family'])} tasks, seed {summary['seed']}</title><style>{CSS}</style></head><body><main>",
        _header(summary),
        _funnel(summary),
        _coverage(summary),
        _prior(summary),
        _models(out_dir, by_id),
        _pairs(out_dir, by_id),
        _ocr(out_dir),
        _audit(out_dir, by_id),
        _accepted_examples(out_dir, accepted),
        _rejected_examples(out_dir, rejected),
        "</main></body></html>",
    ]
    path = out_dir / "report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _e(value: Any) -> str:
    return html.escape(str(value))


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def _share(counts: dict[str, int]) -> str:
    return f"{counts['correct']}/{counts['of']} ({counts['correct'] / counts['of']:.0%})" if counts["of"] else "–"


def _table(headers: list[str], rows: list[list[Any]], numeric: set[int] = frozenset()) -> str:
    def cell(tag: str, i: int, value: Any) -> str:
        cls = " class='num'" if i in numeric else ""
        return f"<{tag}{cls}>{_e(value)}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>" for row in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


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


def _models(out_dir: Path, by_id: dict[str, Task]) -> str:
    summary = read_json(out_dir / "model_summary.json")
    if not summary or not summary.get("models"):
        return ""
    models = summary["models"]
    def precision(m: dict[str, Any]) -> str:
        p = m.get("reading_precision")
        return f"{p['median_error']:.1%} (90% within {p['p90_error']:.1%})" if p else "–"

    rows = [
        [
            name,
            m["tasks"],
            _share(m["with_image"]),
            _share(m["blind"]),
            _share(m["blind_open_ended"]),
            _share({"correct": m["pairs"]["both_correct"], "of": m["pairs"]["of"]}),
            precision(m),
            f"${m['cost_usd']:.2f}",
        ]
        for name, m in models.items()
    ]
    overview = _table(
        [
            "Model", "Tasks", "With the image", "Without it", "Without it, open-ended only",
            "Contrast pairs, both right", "Median error reading unprinted values", "Cost",
        ],
        rows,
        {1, 2, 3, 4, 5, 6, 7},
    )

    def breakdown(title: str, tables: dict[str, dict[str, Any]], always_guess: bool = False) -> str:
        values = sorted({v for table in tables.values() for v in table})
        headers = [title, "Tasks"] + [f"{name}: with image" for name in tables] + ["Without image (first model)"]
        if always_guess:
            headers.append("Always-guess")
        first = next(iter(tables.values()))
        body = []
        for value in values:
            row = [value, first.get(value, {}).get("tasks", "")]
            row += [_pct(table.get(value, {}).get("with_image")) for table in tables.values()]
            row.append(_pct(first.get(value, {}).get("blind")))
            if always_guess:
                row.append(_pct(first.get(value, {}).get("always_guess")))
            body.append(row)
        return f"<h3>{_e(title)}</h3>" + _table(headers, body, set(range(1, len(headers))))

    sections = [
        breakdown("Skill", {n: m["by_skill"] for n, m in models.items()}),
        breakdown("Template", {n: m["by_template"] for n, m in models.items()}, always_guess=True),
    ]
    facets = sorted({f for m in models.values() for f in m.get("by_facet", {})})
    for facet in facets:
        sections.append(breakdown(facet.capitalize(), {n: m["by_facet"].get(facet, {}) for n, m in models.items()}))

    replies: dict[tuple[str, str], dict[str, Any]] = {}
    checks = out_dir / "model_checks.jsonl"
    if checks.is_file():
        for line in checks.read_text().splitlines():
            record = json.loads(line)
            replies[(record["model"], record["task"])] = record
    examples = []
    for name, m in models.items():
        cards = []
        for task_id in m["unsolved"][:4]:
            if task_id in by_id:
                got = replies.get((name, task_id), {}).get("image", {}).get("reply", "")
                cards.append(_card(out_dir, by_id[task_id], None, note=(f"{name} answered: {got or '(no answer)'}", False)))
        for task_id in m["guessable"][:2]:
            if task_id in by_id:
                got = replies.get((name, task_id), {}).get("blind", {}).get("reply", "")
                cards.append(_card(out_dir, by_id[task_id], None, note=(f"Without the image, {name} answered: {got}", True)))
        if cards:
            examples.append(f"<h3>Where {_e(name)} went wrong</h3><div class='grid'>{''.join(cards)}</div>")

    sampled = f"{summary['sample']} of {summary['of']} tasks, spread across templates"
    html_parts = [
        "<h2>Model evaluation</h2>",
        "<p class='note'>Each sampled task was asked twice: with its image, and without it. Reference answers "
        "come from the scene data, never from a model, so a wrong model answer is the model's mistake unless "
        "the human audit finds otherwise. Contrast pairs ask the same question about a minimally edited image "
        f"with a different answer; a model that reads the image gets both right. Sample: {_e(sampled)}.</p>",
        overview,
        "<div class='cols'>" + "".join(f"<div>{s}</div>" for s in sections) + "</div>",
    ]
    html_parts += examples
    return "".join(html_parts)


def _pairs(out_dir: Path, by_id: dict[str, Task], limit: int = 6) -> str:
    summary = read_json(out_dir / "pairs_summary.json")
    if not summary:
        return ""
    rows = [[t, r["tasks"], r["pairs"]] for t, r in summary["by_template"].items()]
    table = _table(["Template", "Tasks tried", "Pairs made"], rows, {1, 2})
    cards = []
    seen: set[str] = set()
    for pair in read_pairs(out_dir):
        if len(cards) >= limit:
            break
        original = by_id.get(pair.a)
        if original is None or original.template in seen:  # one example per template
            continue
        seen.add(original.template)
        cards.append(
            "<div class='card'>"
            f"<div class='twins'>{_image(out_dir, original)}{_image(out_dir, pair.b)}</div>"
            f"<p class='q'>{_e(original.question)}</p>"
            f"<p class='a'>Left: {_e(_answer(original))} · Right: {_e(_answer(pair.b))}</p>"
            f"<p class='tag'>Edit: {_e(pair.change)} ({pair.size:.0%} of the axis)</p>"
            "</div>"
        )
    edit = f" The median edit moves one value by {summary['median_edit']:.0%} of the value axis." if summary["median_edit"] else ""
    return (
        "<h2>Contrast pairs</h2>"
        f"<p class='note'>{summary['pairs']} of {summary['tasks']} tasks have a twin: the same question about an image "
        f"where one value was edited so the answer changes, with the axis pinned so nothing else moves.{edit} Both "
        "images pass the same gates. A model that answers from the question text alone can't get both right.</p>"
        f"<div class='cols'><div>{table}</div></div><div class='grid'>{''.join(cards)}</div>"
    )


OCR_KINDS = {
    "cat": "Category and slice labels",
    "legend": "Legend entries",
    "val": "Printed values",
    "pct": "Printed percentages",
    "tot": "Printed stack totals",
    "vtick": "Value-axis ticks",
    "title": "Titles",
}


def _ocr(out_dir: Path) -> str:
    summary = read_json(out_dir / "ocr_summary.json")
    if not summary:
        return ""
    rows = [
        [OCR_KINDS.get(kind, kind), row["labels"], row["read_back"], _pct(row["read_back"] / row["labels"] if row["labels"] else None)]
        for kind, row in summary["by_kind"].items()
    ]
    tiles = [
        ("Labels read back", f"{summary['read_back']}/{summary['labels']}"),
        ("Read-back rate", _pct(summary["read_back"] / summary["labels"] if summary["labels"] else None)),
        ("Tasks with every label read back", f"{summary['tasks_fully_read_back']}/{summary['tasks']}"),
    ]
    tile_html = "".join(f"<div class='tile'><b>{_e(v)}</b><span class='muted'>{_e(k)}</span></div>" for k, v in tiles)
    misses = []
    for miss in summary["misses"][:12]:
        crop = out_dir / miss["crop"] if miss.get("crop") else None
        img = ""
        if crop and crop.is_file():
            img = f"<img src='data:image/png;base64,{base64.b64encode(crop.read_bytes()).decode()}' alt='' style='max-width:100%'>"
        misses.append(
            f"<div class='card'>{img}<p class='a'>Drawn: {_e(miss['expected'])} · OCR read: {_e(miss['got'] or '(nothing)')}</p>"
            f"<div class='tag'>{_e(miss['image'])} · {_e(miss['element'])}</div></div>"
        )
    miss_html = f"<h3>Labels OCR could not read back</h3><div class='grid'>{''.join(misses)}</div>" if misses else ""
    return (
        "<h2>OCR read-back</h2>"
        f"<p class='note'>An independent check of the grounding gate, which trusts the renderer's layout record. Every "
        f"label a question depends on was cropped out of the final PNG and read by {_e(summary['engine'])}, an OCR engine "
        "unrelated to the models being evaluated. A label counts as read back when the OCR text matches what was drawn "
        "(characters OCR commonly confuses, such as O and 0, count as equal). The crops below show what OCR saw.</p>"
        f"<div class='tiles'>{tile_html}</div><div class='cols'><div>"
        + _table(["Label kind", "Labels", "Read back", "Rate"], rows, {1, 2, 3})
        + f"</div></div>{miss_html}"
    )


def _audit(out_dir: Path, by_id: dict[str, Task]) -> str:
    summary = read_json(out_dir / "audit_summary.json")
    if not summary:
        return ""
    r, t = summary["random"], summary["targeted"]
    tiles = []
    if r["audited"]:
        low, high = r["valid_rate_95ci"]
        tiles = [
            ("Random tasks audited", r["audited"]),
            ("Judged valid", f"{r['valid']}/{r['audited']}"),
            ("95% interval for the valid rate", f"{low:.1%} – {high:.1%}"),
            ("Blind answers matching the reference", f"{r['reviewer_agreed']}/{r['audited']}"),
        ]
    tile_html = "".join(f"<div class='tile'><b>{_e(v)}</b><span class='muted'>{_e(k)}</span></div>" for k, v in tiles)
    rows = [
        [name, g["audited"], g["valid"], g["wrong"], g["ambiguous"]]
        for name, g in (("Random sample", r), ("Tasks a model got wrong (targeted)", t))
        if g["audited"]
    ]
    problems = []
    for p in summary["problems"][:12]:
        task = by_id.get(p["id"])
        if task is not None:
            note = f"{p['verdict']}: {p.get('note') or 'no note'} (reviewer answered {p.get('reviewer_answer') or '–'})"
            problems.append(_card(out_dir, task, None, note=(note, False)))
    who = f" by {_e(summary['reviewer'])}" if summary.get("reviewer") else ""
    return (
        "<h2>Human audit</h2>"
        f"<p class='note'>A person{who} answered each sampled task from the image alone, then saw the reference answer and "
        "judged it valid, wrong, or ambiguous. The random sample gives an unbiased rate; tasks a model got wrong were "
        "reviewed separately because that is where a generator mistake would hide.</p>"
        f"<div class='tiles'>{tile_html}</div>"
        + _table(["Group", "Audited", "Valid", "Wrong", "Ambiguous"], rows, {1, 2, 3, 4})
        + (f"<h3>Tasks the reviewer flagged</h3><div class='grid'>{''.join(problems)}</div>" if problems else "")
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


def _image(out_dir: Path, task: Task, failure: tuple[str, GateResult] | None = None) -> str:
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
    return f"<div class='shot'><img src='data:image/png;base64,{data}' alt='{_e(task.question)}'>{boxes}</div>"


def _answer(task: Task) -> str:
    answer = task.answer
    shown = f"{answer.value}"
    if answer.tolerance:
        shown += f" (± {answer.tolerance:g})"
    if answer.unit:
        shown += f" {answer.unit}"
    return shown


def _card(
    out_dir: Path,
    task: Task,
    failure: tuple[str, GateResult] | None,
    note: tuple[str, bool] | None = None,
) -> str:
    trace = "".join(f"<li>{_e(step)}</li>" for step in task.trace)
    if failure:
        gate, result = failure
        verdict = f"<p class='why'>{_e(gate)}: {_e(result.reason)}</p>"
    else:
        verdict = "<p class='tag'>" + " · ".join(f"{_e(g)} {_e(r.status)}" for g, r in task.validation.items()) + "</p>"
    extra = f"<p class='got {'ok' if note[1] else 'no'}'>{_e(note[0])}</p>" if note else ""
    return (
        "<div class='card'>"
        f"{_image(out_dir, task, failure)}"
        f"<p class='q'>{_e(task.question)}</p>"
        f"<p class='a'>Answer: {_e(_answer(task))}</p>"
        f"{extra}{verdict}<ol>{trace}</ol>"
        f"<div class='tag'>{_e(task.id)} · {_e(task.skill)}</div>"
        "</div>"
    )
