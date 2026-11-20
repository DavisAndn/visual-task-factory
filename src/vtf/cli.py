"""Command line: `vtf generate` makes a batch of tasks; `vtf rebuild` re-derives one from its seed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from .core.config import load_config
from .core.family import Family
from .core.pipeline import rebuild_task, run
from .core.schema import Task
from .families import get_family

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


@app.command()
def rebuild(
    task_id: Annotated[str, typer.Argument(help="Task id, for example charts-000012-extremum.")],
    tasks: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="tasks.jsonl or rejected.jsonl from a run.")
    ] = Path("out/charts-seed42/tasks.jsonl"),
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
