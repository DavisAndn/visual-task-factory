"""Command line.

  vtf generate   make a batch of validated tasks
  vtf models     evaluate Claude on a run, with and without the images
  vtf report     rebuild a run's report.html with any evidence gathered since
  vtf rebuild    re-derive one task from its seed and check it comes out identical
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Optional

import typer

from .core.config import load_config
from .core.family import Family
from .core.pipeline import rebuild_task, run
from .core.schema import Task
from .families import get_family
from .report import rebuild_report

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Generate image question-answer tasks with computed answers and machine-checked validity.",
)


def _family(name: str) -> Family:
    try:
        return get_family(name)
    except ValueError as err:
        typer.secho(str(err), fg="red", err=True)
        raise typer.Exit(2) from None


@app.command()
def generate(
    family: Annotated[str, typer.Option(help="Image family to generate.")] = "charts",
    n: Annotated[int, typer.Option("--n", "-n", min=1, help="Number of accepted tasks to produce.")] = 100,
    seed: Annotated[int, typer.Option(help="Run seed. The same seed and config give identical output.")] = 42,
    config: Annotated[
        Optional[Path], typer.Option(exists=True, dir_okay=False, help="YAML file that overrides the defaults.")
    ] = None,
    out: Annotated[Optional[Path], typer.Option(help="Output folder. Default: out/<family>-seed<seed>.")] = None,
) -> None:
    """Generate and validate tasks; write tasks.jsonl, rejected.jsonl, summary.json and report.html."""
    fam = _family(family)
    cfg = load_config(config, fam)
    out_dir = out or Path("out") / f"{family}-seed{seed}"
    with typer.progressbar(length=n, label="Generating") as bar:
        summary = run(fam, n, seed, out_dir, cfg, progress=lambda done: bar.update(done - bar.pos))
    typer.echo(
        f"Accepted {summary['accepted']} of {summary['candidates']} candidates "
        f"from {summary['scenes_rendered']} images in {summary['elapsed_seconds']}s."
    )
    for reason, count in summary["rejection_reasons"].items():
        typer.echo(f"  rejected {count:>4}  {reason}")
    typer.echo(f"Tasks:  {out_dir / 'tasks.jsonl'}")
    typer.echo(f"Report: {out_dir / 'report.html'}")
    if summary["accepted"] < n:
        typer.secho(f"Stopped at the image limit before reaching {n} tasks; raise max_scenes.", fg="yellow")


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=value lines from .env into the environment, without overriding what is already set."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip().strip("'\"")
        if key and value:
            os.environ.setdefault(key, value)


RunDir = Annotated[Path, typer.Argument(exists=True, file_okay=False, help="A run folder made by vtf generate.")]


@app.command()
def models(
    run_dir: RunDir,
    model: Annotated[
        Optional[list[str]], typer.Option("--model", "-m", help="Claude model id. Repeat the option to compare models.")
    ] = None,
    effort: Annotated[str, typer.Option(help="How hard the model thinks: low, medium, high, xhigh or max.")] = "high",
    sample: Annotated[int, typer.Option(min=0, help="Tasks to ask, spread across templates. 0 asks every task.")] = 0,
    seed: Annotated[int, typer.Option(help="Seed for picking the sample.")] = 0,
    workers: Annotated[int, typer.Option(min=1, max=32, help="Requests in flight at once.")] = 8,
) -> None:
    """Evaluate Claude on a run: each sampled task with its image and without it.

    Needs ANTHROPIC_API_KEY, from the environment or a .env file in the current folder. Answers are
    cached in the run folder, so running again (or adding --sample) only pays for new questions.
    """
    from .clients import DEFAULT_MODEL, CachedClient, ClaudeClient
    from .evidence.models import count_jobs, run_model_checks
    from .validators.models import ClientConfigError

    _load_dotenv()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        typer.secho("No API key: set ANTHROPIC_API_KEY, or add ANTHROPIC_API_KEY=... to a .env file here.", fg="red", err=True)
        raise typer.Exit(2)
    if os.environ.get("ANTHROPIC_BASE_URL"):
        typer.secho(f"Note: requests go to {os.environ['ANTHROPIC_BASE_URL']} (ANTHROPIC_BASE_URL is set).", fg="yellow")
    try:
        clients = [CachedClient(ClaudeClient(name, effort), run_dir / "model_cache.jsonl") for name in model or [DEFAULT_MODEL]]
    except ValueError as err:
        typer.secho(str(err), fg="red", err=True)
        raise typer.Exit(2) from None
    total = count_jobs(run_dir, len(clients), sample, seed)
    try:
        with typer.progressbar(length=total, label="Asking") as bar:
            summary = run_model_checks(run_dir, clients, sample, seed, workers, progress=bar.update)
    except ClientConfigError as err:
        typer.secho(str(err), fg="red", err=True)
        raise typer.Exit(1) from None
    for client in clients:
        m = summary["models"][client.name]
        typer.echo(f"{client.name}")
        typer.echo(f"  with the image     {_share(m['with_image'])}")
        typer.echo(f"  without it         {_share(m['blind'])}  (open-ended only: {_share(m['blind_open_ended'])})")
        if m["pairs"]["of"]:
            typer.echo(f"  contrast pairs     {m['pairs']['both_correct']}/{m['pairs']['of']} with both answers right")
        if m.get("reading_precision"):
            p = m["reading_precision"]
            typer.echo(f"  reading error      median {p['median_error']:.1%}, 90% within {p['p90_error']:.1%} (unprinted values)")
        if m["errors"]:
            typer.secho(f"  {m['errors']} requests failed; run the command again to retry them", fg="yellow")
        typer.echo(f"  cost               ${m['cost_usd']:.2f} (${m['spent_this_run_usd']:.2f} new this run)")
    typer.echo(f"Report: {rebuild_report(run_dir)}")


@app.command()
def pairs(
    run_dir: RunDir,
    share: Annotated[float, typer.Option(min=0.0, max=1.0, help="Share of the run's tasks to pair.")] = 1.0,
    seed: Annotated[int, typer.Option(help="Seed for picking which tasks to pair.")] = 0,
) -> None:
    """Make contrast pairs: the same question about a minimally edited image with a different answer."""
    from .core.contrast import make_pairs
    from .core.runs import read_tasks

    total = sum(1 for _ in read_tasks(run_dir / "tasks.jsonl"))
    with typer.progressbar(length=total, label="Pairing") as bar:
        summary = make_pairs(run_dir, share, seed, progress=bar.update)
    edit = f"; median edit {summary['median_edit']:.0%} of the value axis" if summary["median_edit"] else ""
    typer.echo(f"Paired {summary['pairs']} of {summary['tasks']} tasks{edit}.")
    for template, row in summary["by_template"].items():
        typer.echo(f"  {row['pairs']:>4}/{row['tasks']:<4} {template}")
    typer.echo(f"Pairs:  {run_dir / 'pairs.jsonl'}")
    typer.echo(f"Report: {rebuild_report(run_dir)}")


@app.command()
def ocr(
    run_dir: RunDir,
    workers: Annotated[int, typer.Option(min=1, max=32, help="OCR processes at once.")] = 8,
) -> None:
    """Read back every label the run's tasks depend on with Tesseract, independently of the renderer."""
    from .core.runs import read_tasks
    from .evidence.ocr import run_ocr, tesseract_path

    if tesseract_path() is None:
        typer.secho("Tesseract is not installed. macOS: brew install tesseract · Ubuntu: apt install tesseract-ocr", fg="red", err=True)
        raise typer.Exit(2)
    from .families import get_family

    tasks = read_tasks(run_dir / "tasks.jsonl")
    total = 0
    for image in {t.image: t for t in tasks}.values():
        family = get_family(image.family)
        _, layout = family.render(family.scene_from_dict(image.scene))
        ids = {i.element_id for t in tasks if t.image == image.image for ref in t.depends_on for i in layout.find(ref) if i.kind == "text"}
        total += len(ids)
    with typer.progressbar(length=total, label="Reading") as bar:
        summary = run_ocr(run_dir, workers, progress=bar.update)
    rate = summary["read_back"] / summary["labels"] if summary["labels"] else 0
    typer.echo(f"{summary['engine']} read back {summary['read_back']} of {summary['labels']} labels ({rate:.1%}).")
    typer.echo(f"Tasks with every label read back: {summary['tasks_fully_read_back']} of {summary['tasks']}.")
    for miss in summary["misses"][:10]:
        typer.echo(f"  {miss['image']} {miss['element']}: expected {miss['expected']!r}, read {miss['got']!r}")
    typer.echo(f"Report: {rebuild_report(run_dir)}")


@app.command()
def audit(
    run_dir: RunDir,
    n: Annotated[int, typer.Option("--n", "-n", min=1, help="Random tasks to audit, spread across templates.")] = 100,
    seed: Annotated[int, typer.Option(help="Seed for the sample.")] = 0,
    targeted: Annotated[int, typer.Option(min=0, help="Extra tasks a model got wrong, reviewed separately.")] = 30,
    import_file: Annotated[
        Optional[Path], typer.Option("--import", exists=True, dir_okay=False, help="Results file downloaded from audit.html.")
    ] = None,
) -> None:
    """Human audit. Without --import: write audit.html to review a sample blind. With it: score the results."""
    from .evidence.audit import build_audit, import_audit

    if import_file is None:
        path = build_audit(run_dir, n, seed, targeted)
        typer.echo(f"Audit page: {path}")
        typer.echo("Open it in a browser, judge the tasks, click Download results, then run:")
        typer.echo(f"  vtf audit {run_dir} --import <downloaded file>")
        return
    summary = import_audit(run_dir, import_file)
    r = summary["random"]
    if r["audited"]:
        low, high = r["valid_rate_95ci"]
        typer.echo(
            f"Random sample: {r['valid']} of {r['audited']} valid ({r['valid_rate']:.1%}, 95% CI {low:.1%}–{high:.1%}); "
            f"{r['wrong']} wrong, {r['ambiguous']} ambiguous."
        )
        typer.echo(f"Your blind answers matched the reference on {r['reviewer_agreed']} of {r['audited']}.")
    t = summary["targeted"]
    if t["audited"]:
        typer.echo(f"Tasks a model got wrong: {t['valid']} of {t['audited']} valid, {t['wrong']} wrong, {t['ambiguous']} ambiguous.")
    typer.echo(f"Report: {rebuild_report(run_dir)}")


def _share(counts: dict[str, int]) -> str:
    if not counts["of"]:
        return "–"
    return f"{counts['correct']}/{counts['of']} ({counts['correct'] / counts['of']:.0%})"


@app.command()
def report(run_dir: RunDir) -> None:
    """Rebuild report.html for a run, including evidence gathered after generation."""
    typer.echo(f"Report: {rebuild_report(run_dir)}")


@app.command()
def rebuild(
    task_id: Annotated[str, typer.Argument(help="Task id, for example charts-000012-extremum.")],
    tasks: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="tasks.jsonl or rejected.jsonl from a run.")
    ] = Path("samples/charts-seed42/tasks.jsonl"),
) -> None:
    """Rebuild one task from its seed and check that its scene, pixels and answer come out identical."""
    task = next((t for t in map(Task.model_validate_json, tasks.read_text().splitlines()) if t.id == task_id), None)
    if task is None:
        typer.secho(f"No task {task_id} in {tasks}", fg="red", err=True)
        raise typer.Exit(1)
    cfg = json.loads((tasks.parent / "config.json").read_text())
    checks = rebuild_task(task, cfg)
    for name, same in checks.items():
        typer.echo(f"{'same' if same else 'DIFFERENT':<9} {name}")
    raise typer.Exit(0 if all(checks.values()) else 1)
