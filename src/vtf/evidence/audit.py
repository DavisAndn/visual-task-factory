"""Human audit: a person answers a random sample of tasks blind, then judges the reference answer.

`build_audit` writes audit.html into the run folder: a self-contained page that shows one task at
a time. The reviewer types their own answer from the image first, then sees the reference answer
and its derivation and marks the task valid, wrong, or ambiguous. Progress is kept in the browser;
"Download results" saves a JSON file that `import_audit` turns into audit_summary.json.

The random sample gives an unbiased error rate (with a 95% Wilson interval). Tasks a model got
wrong are added as targeted reviews: they are where generator mistakes would hide, but because
they are not random they are reported separately and never mixed into the rate.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from ..core.runs import read_json, read_tasks
from ..validators.models import score_response
from ..core.schema import Answer
from .models import sample_tasks

VERDICTS = ("valid", "wrong", "ambiguous")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_audit(run_dir: Path, n: int = 100, seed: int = 0, targeted: int = 30, photos: bool = False) -> Path:
    """Write audit.html (or audit-photos.html, for the photo twins): a sample to judge blind."""
    tasks = read_tasks(run_dir / ("photos.jsonl" if photos else "tasks.jsonl"))
    sample = sample_tasks(tasks, n, seed)
    chosen = {t.id for t in sample}
    extra: list[str] = []
    models = (read_json(run_dir / "model_summary.json") or {}).get("models", {})
    for summary in models.values():
        wrong = [f"{i}-photo" for i in summary.get("photos", {}).get("lost", [])] if photos else summary.get("unsolved", [])
        extra += [task_id for task_id in wrong if task_id not in chosen and task_id not in extra]
    by_id = {t.id: t for t in tasks}
    items = [(t, "random") for t in sample] + [(by_id[i], "targeted") for i in extra[:targeted] if i in by_id]
    random.Random(seed).shuffle(items)  # the reviewer can't tell targeted items from random ones

    data = []
    for task, source in items:
        image = base64.b64encode((run_dir / task.image).read_bytes()).decode()
        data.append({
            "id": task.id,
            "source": source,
            "image": f"data:image/{'jpeg' if task.image.endswith('.jpg') else 'png'};base64,{image}",
            "question": task.question,
            "answer": task.answer.model_dump(),
            "trace": task.trace,
            "template": task.template,
        })
    run_id = hashlib.sha256(json.dumps([d["id"] for d in data]).encode()).hexdigest()[:12]
    page = (
        _PAGE.replace("__RUN__", json.dumps(run_dir.name))
        .replace("__KEY__", json.dumps(f"vtf-audit:{run_dir.name}:{'photos:' if photos else ''}{run_id}"))
        .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    )
    path = run_dir / ("audit-photos.html" if photos else "audit.html")
    path.write_text(page, encoding="utf-8")
    return path


def import_audit(run_dir: Path, results_path: Path) -> dict[str, Any]:
    """Check a downloaded results file against the run and write audit_results.json and audit_summary.json
    (photo_audit_*.json when the file came from audit-photos.html)."""
    results = json.loads(Path(results_path).read_text())
    photos = any(str(item.get("id", "")).endswith("-photo") for item in results.get("items", []))
    prefix = "photo_" if photos else ""
    by_id = {t.id: t for t in read_tasks(run_dir / ("photos.jsonl" if photos else "tasks.jsonl"))}
    rows = []
    for item in results["items"]:
        task = by_id.get(item["id"])
        if task is None or item.get("verdict") not in VERDICTS:
            continue
        agrees = score_response(Answer.model_validate(task.answer.model_dump()), item.get("reviewer_answer", ""))
        rows.append({**item, "agrees": agrees, "reference": task.answer.model_dump(), "template": task.template})
    (run_dir / f"{prefix}audit_results.json").write_text(json.dumps({"reviewer": results.get("reviewer"), "items": rows}, indent=2))

    def stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = {v: sum(r["verdict"] == v for r in group) for v in VERDICTS}
        low, high = wilson(count["valid"], len(group))
        return {
            "audited": len(group),
            **count,
            "valid_rate": round(count["valid"] / len(group), 4) if group else None,
            "valid_rate_95ci": [round(low, 4), round(high, 4)],
            "reviewer_agreed": sum(r["agrees"] for r in group),
        }

    random_rows = [r for r in rows if r.get("source") == "random"]
    targeted_rows = [r for r in rows if r.get("source") == "targeted"]
    summary = {
        "reviewer": results.get("reviewer"),
        "random": stats(random_rows),
        "targeted": stats(targeted_rows),
        "problems": [
            {k: r.get(k) for k in ("id", "source", "verdict", "note", "reviewer_answer", "reference", "template")}
            for r in rows
            if r["verdict"] != "valid"
        ],
    }
    (run_dir / f"{prefix}audit_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task audit</title>
<style>
:root { --bg:#fff; --fg:#1c1c1e; --muted:#6c6c72; --line:#e4e4e7; --card:#f7f7f8; --bad:#d92d20; --good:#067647; --accent:#2563eb; }
@media (prefers-color-scheme: dark) { :root { --bg:#141416; --fg:#ececef; --muted:#a1a1aa; --line:#2e2e33; --card:#1c1c1f; --bad:#f97066; --good:#47cd89; --accent:#60a5fa; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 20px; margin: 0; }
.muted { color: var(--muted); }
.bar { height: 6px; background: var(--line); border-radius: 3px; margin: 10px 0 16px; overflow: hidden; }
.bar > div { height: 100%; background: var(--accent); width: 0; }
.card { border:1px solid var(--line); background:var(--card); border-radius:12px; padding:14px; }
.card img { width:100%; height:auto; border-radius:8px; display:block; background:#fff; }
.q { font-weight:600; font-size:17px; margin:12px 0; }
form { display:flex; gap:8px; }
input { flex:1; font:inherit; padding:9px 11px; border-radius:8px; border:1px solid var(--line); background:var(--bg); color:var(--fg); }
button { font:inherit; padding:9px 14px; border-radius:8px; border:1px solid var(--line); background:var(--bg); color:var(--fg); cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.verdicts { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:8px; margin:12px 0 8px; }
.verdicts button.on { outline: 3px solid var(--accent); }
.ok { color: var(--good); } .no { color: var(--bad); }
ol { color: var(--muted); font-size:14px; margin:6px 0; }
nav, footer { display:flex; gap:8px; align-items:center; justify-content:space-between; margin-top:14px; flex-wrap:wrap; }
.help { font-size:14px; }
kbd { border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:12px; }
</style></head>
<body><main>
<h1>Task audit · <span id="run"></span></h1>
<p class="help muted">Answer each question from the image alone, then press <kbd>Enter</kbd> to see the reference answer.
Mark it <b>valid</b> only if the reference answer is right and the image shows enough to be sure.
Keys: <kbd>1</kbd> valid · <kbd>2</kbd> wrong · <kbd>3</kbd> ambiguous · <kbd>←</kbd> <kbd>→</kbd> move.</p>
<div class="bar"><div id="fill"></div></div>
<div class="card">
  <img id="img" alt="">
  <p class="q" id="q"></p>
  <form id="form"><input id="answer" autocomplete="off" placeholder="Your answer"><button class="primary">Check</button></form>
  <div id="reveal" hidden>
    <p id="ref"></p><p id="agree"></p><ol id="trace"></ol>
    <div class="verdicts">
      <button data-v="valid">1 · Valid</button>
      <button data-v="wrong">2 · Reference answer is wrong</button>
      <button data-v="ambiguous">3 · Ambiguous or unreadable</button>
    </div>
    <input id="note" placeholder="Note (optional)">
  </div>
</div>
<nav><button id="prev">← Back</button><span id="pos" class="muted"></span><button id="next">Next →</button></nav>
<footer><span><input id="reviewer" placeholder="Your name (optional)" style="width:220px"></span>
<span><span id="status" class="muted"></span> <button id="export" class="primary">Download results</button></span></footer>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const RUN = __RUN__, KEY = __KEY__;
const items = JSON.parse(document.getElementById("data").textContent);
const $ = (id) => document.getElementById(id);
let state = { pos: 0, results: {}, reviewer: "" };
try { const saved = JSON.parse(localStorage.getItem(KEY)); if (saved) state = saved; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };

const norm = (t) => t.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
function finalAnswer(reply) {  // last non-empty line, after its last "=", markup removed
  const lines = reply.replace(/<[^>]*>/g, " ").split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  return lines.length ? lines[lines.length - 1].split("=").pop() : "";
}
function agrees(answer, reply) {  // same rules as vtf's score_response
  const text = finalAnswer(reply).trim().toLowerCase().replace(/−/g, "-");
  if (answer.type === "yes_no") { const w = text.match(/^[^a-z]*([a-z]+)/); return !!w && w[1] === String(answer.value); }
  if (answer.type === "integer" || answer.type === "number") {
    const m = text.match(/-?\d[\d,]*(?:\.\d+)?|-?\.\d+/); if (!m) return false;
    return Math.abs(parseFloat(m[0].replace(/,/g, "")) - Number(answer.value)) <= answer.tolerance + 1e-9;
  }
  return norm(text) === norm(String(answer.value));
}
function shown(a) { return `${a.value}${a.tolerance ? ` (± ${a.tolerance})` : ""}${a.unit ? ` ${a.unit}` : ""}`; }

function render() {
  const item = items[state.pos], r = state.results[item.id] || {};
  $("run").textContent = RUN;
  $("img").src = item.image; $("q").textContent = item.question;
  $("answer").value = r.reviewer_answer || ""; $("note").value = r.note || "";
  $("reveal").hidden = r.reviewer_answer === undefined;
  if (!$("reveal").hidden) {
    $("ref").innerHTML = `Reference answer: <b>${shown(item.answer)}</b>`;
    const ok = agrees(item.answer, r.reviewer_answer);
    $("agree").textContent = ok ? "Your answer agrees." : "Your answer differs. Look again before you judge.";
    $("agree").className = ok ? "ok" : "no";
    $("trace").innerHTML = item.trace.map((s) => `<li>${s.replace(/[&<>]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}</li>`).join("");
  }
  document.querySelectorAll(".verdicts button").forEach((b) => b.classList.toggle("on", b.dataset.v === r.verdict));
  const done = Object.values(state.results).filter((x) => x.verdict).length;
  $("pos").textContent = `${state.pos + 1} of ${items.length} · ${done} judged`;
  $("fill").style.width = `${(100 * done) / items.length}%`;
  $("reviewer").value = state.reviewer || "";
  if ($("reveal").hidden) $("answer").focus();
}
function record(fields) { const item = items[state.pos]; state.results[item.id] = { ...(state.results[item.id] || {}), ...fields }; save(); }
function move(step) { state.pos = Math.min(items.length - 1, Math.max(0, state.pos + step)); save(); render(); }
function judge(v) {
  const item = items[state.pos]; if ($("reveal").hidden) return;
  record({ verdict: v, note: $("note").value, judged_at: new Date().toISOString() });
  if (state.pos < items.length - 1) move(1); else render();
}
$("form").addEventListener("submit", (e) => {
  e.preventDefault(); record({ reviewer_answer: $("answer").value }); render(); $("answer").blur();  // free the 1/2/3 keys
});
document.querySelectorAll(".verdicts button").forEach((b) => b.addEventListener("click", () => judge(b.dataset.v)));
$("note").addEventListener("change", () => record({ note: $("note").value }));
$("reviewer").addEventListener("change", () => { state.reviewer = $("reviewer").value; save(); });
$("prev").addEventListener("click", () => move(-1)); $("next").addEventListener("click", () => move(1));
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "1" || e.key === "2" || e.key === "3") judge(["valid", "wrong", "ambiguous"][Number(e.key) - 1]);
  if (e.key === "ArrowRight") move(1);
  if (e.key === "ArrowLeft") move(-1);
});
$("export").addEventListener("click", () => {
  const out = { run: RUN, reviewer: state.reviewer || null, exported_at: new Date().toISOString(),
    items: items.filter((i) => state.results[i.id] && state.results[i.id].verdict)
      .map((i) => ({ id: i.id, source: i.source, ...state.results[i.id] })) };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `audit-results-${RUN}.json`; a.click();
  $("status").textContent = `Saved ${out.items.length} judgments.`;
});
render();
</script></body></html>
"""
