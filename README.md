# Visual task factory

A reusable pipeline that generates visual question-answer tasks (an image, a question about it,
and the answer) for benchmarking and training vision-language models. Every answer is computed
from data, never by a model, and every task passes machine checks that prove it is valid before
it is kept. Built for Track 2 (Visual Task Factory) of the Innodata AI hackathon.

## Quick start

```bash
uv sync
uv run vtf generate --n 100 --seed 42
open out/charts-seed42/report.html
```

Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -e . pytest`, then
run `vtf` directly. Run the tests with `uv run pytest`.

On a laptop, 100 accepted tasks take about 2 seconds and 1,000 take about 20.

## How it works

**Data first, image second.** Each task starts as a *scene*: structured data (values, labels,
styling) sampled from a seed. Everything else is derived from it.

1. **Sample** a scene from the family's generator, driven by the seed and `configs/charts.yaml`.
2. **Render** it with matplotlib. The renderer also writes a *layout manifest*: every label and
   mark it drew, with its pixel box, font size and, for rotated text, its exact corners.
3. **Ask.** A question template picks its parameters, and a solver computes the answer from the
   scene data. Each question lists `depends_on`, the parts of the image that it and its answer
   rely on (category labels, bars, the legend entry, the value axis...).
4. **Validate** every candidate through the gates below. Failures are kept in `rejected.jsonl`
   with the reason; nothing is silently fixed.
5. **Package** accepted tasks into `tasks.jsonl` with their images, plus `summary.json`,
   `config.json` and a self-contained `report.html`.

| Gate | Rejects | How |
|---|---|---|
| Determinism | Output that changes between runs | 10% of images are rebuilt from their seed; scene and pixel hashes must match. |
| Grounding | Questions or answers that aren't visible | Every `depends_on` element must be drawn, inside the image, set in at least 9 px type if it is text, and not covered by other text or the legend. Label answers must appear in the image as legible text. With `clean_text`, no text anywhere may collide. Rotated labels are compared by their real shape, not their bounding box. |
| Ambiguity | Answers readers could disagree on | The values that decide an answer (top bar vs runner-up, distance from a threshold...) must differ by more than the printed precision, or by 5% of the axis when values aren't printed. Numbers read off an axis must be readable to within 20%. |
| Dedup | Near-copies | A 256-bit image hash; images within 8 bits of an accepted image are rejected. |

The report adds two dataset-level checks for shortcuts a model could take without looking at
the image: the score from always giving a template's most common answer, and how often the
answer is a name that appears in the question itself.

## Question templates

| Template | Skill | Example |
|---|---|---|
| `count_categories` | structure | How many countries does the chart show? |
| `read_value` | retrieval | What was the output for Gas in 2015? Answer in TWh. |
| `extremum` | extremum | Which source had the highest output in 2015? |
| `compare_pair` | comparison | Did Photos have a higher user count than Maps in July? |
| `count_above` | counting | How many sources had an output above 200 TWh in 2015? |
| `difference` | arithmetic | What is the difference in user count between Music and Fitness in July? |
| `rank_then_read` | multi-hop | What was the user count of the app with the third-highest user count in July? |
| `what_if` | counterfactual | If Nuclear's output in 2015 were doubled, which source would have the highest output? |
| `trend_decreases` | trend | How many times did News's user count decrease from one month to the next? |
| `largest_increase` | multi-hop | Which region had the largest increase in revenue from Q4 2023 to Q1 2024? |

Templates are picked per image so accepted tasks stay close to the target shares in the config,
even when some templates are rejected more often than others. Yes/no answers are balanced, and
`what_if` names the eventual winner only about 1 time in n, so the question doesn't give away
its answer.

Visual variation: bar, horizontal bar, grouped bar and line charts; five topics; three fonts;
four palettes; three backgrounds; label rotation; gridlines; legend placement; and values printed
or left to be read off the axis.

## Output

Each line of `tasks.jsonl` is one task (scene data trimmed here):

```json
{
  "id": "charts-000000-compare_pair",
  "seed": 71586830263575,
  "image": "images/charts-000000.png",
  "question": "Did Photos have a higher user count than Maps in July?",
  "answer": {"value": "yes", "type": "yes_no", "tolerance": 0.0, "unit": null},
  "depends_on": ["cat:4", "cat:3", "bar:0:4", "bar:0:3", "title"],
  "trace": ["Photos = 369, Maps = 258", "higher? → yes"],
  "validation": {"determinism": {"status": "pass"}, "grounding": {"status": "pass"}, "...": "..."},
  "scene": {"chart_type": "bar", "categories": ["Mail", "Fitness", "Music", "Maps", "Photos"], "...": "..."}
}
```

Numeric answers read off an axis carry a `tolerance` (5%, as in ChartQA, never tighter than a
tenth of a gridline step); printed values are exact.

## Reproducibility

The same seed and config give byte-identical `tasks.jsonl`. Any single task can be re-derived
and checked:

```bash
uv run vtf rebuild charts-000002-read_value --tasks out/charts-seed42/tasks.jsonl
```

Charts only use fonts that ship with matplotlib, and `uv.lock` pins every dependency, so pixel
hashes should match across machines running the same versions (verified on one machine so far).

## Configuration

`configs/charts.yaml` lists every setting with its default: template shares, chart-type mix,
style options and gate thresholds. Pass it with `--config configs/charts.yaml`.

## Adding an image family

Implement the `Family` protocol in `src/vtf/core/family.py`: `sample_scene`, `render` (returning
the image and its layout manifest), a list of templates with `instantiate` and `solve`, and
`facets` for the coverage report. Register it in `src/vtf/families/__init__.py`. The gates and the
report work unchanged as long as the renderer fills in the manifest and the solvers report
their deciding margins.

## Layout

```
src/vtf/
  cli.py               vtf generate, vtf rebuild
  core/                data models, plugin contract, config, pipeline
  families/charts/     scene sampler, renderer, question templates
  validators/          grounding, ambiguity, dedup; model-based checks
  report.py            HTML report
configs/charts.yaml    every setting with its default
tests/                 pytest suite
```

## Not done yet

- **Model-based checks.** `validators/models.py` has a blind-baseline gate (fails questions a
  model answers without the image) and a solvability gate (sends tasks no model solves to human
  review), tested with a fake client. They need a `ModelClient` for a real provider and a command
  to run them over `tasks.jsonl`.
- **OCR read-back** of the labels each question depends on.
- **Human audit page** to review a random sample and report the error rate.
- **Contrast pairs**: nudge the data so the answer flips and ask the same question of both images.
- **More chart types** (stacked, pie, scatter) and a second family (circuit schematics).
- **Known skew:** `count_categories` answers lean small, because charts with many categories are
  rejected more often for colliding labels.
