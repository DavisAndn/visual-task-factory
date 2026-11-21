# Deck content

Slide by slide, in the order of the Innodata template ("Innodata | IDEA SUBMISSION"). Copy the
template in Google Slides (File → Make a copy), paste each section into its slide, and export as
PDF (under 5 MB). Figures are in [figures/](figures/); rebuild them with
`uv run python docs/figures/make_figures.py`.

## 1. Title

- **Team name:** Plot Twist (a team of one)
- **Team leader name:** Davis Andherson
- **Problem statement:** Visual Task Factory: chart understanding for document AI

## 2. Brief about the idea

A pipeline that turns seeded data into chart images, questions and reference answers that no
model ever sets, checks every task with model-independent evidence, and uses the tasks to map
where frontier multimodal models misread charts.

Figure: `figures/examples.png` (one task per chart type).

## 3. Opportunity

**How it differs from existing chart benchmarks.** Benchmarks such as ChartQA are scraped once and
annotated by people or models, so they leak into training data and their labels are only as good
as the annotators. Here every image is generated, so fresh held-out sets cost nothing and can't
have leaked, and every answer is computed from the data that drew the image.

**How it solves the problem.** Each task comes with proof: the data and the derivation of its
answer, a record of every element drawn, gate results, an OCR read-back of the labels it depends
on, a contrast twin, and a seed that rebuilds it.

**USP.**
- Answers validated independently of the model under test (data, gates, OCR, human audit).
- Contrast pairs: the same question about a minimally edited image whose answer changes, which
  separates reading the image from guessing.
- Shortcut control: answer balancing keeps "always guess the commonest answer" near chance.
- Reproducible: any task rebuilds from its seed, and CI regenerates the committed samples.

## 4. Features

- 6 chart types (bar, horizontal, grouped, stacked, line, pie), 5 topics, 17 templates, 11 skills.
- Gates: determinism, grounding, ambiguity, dedup; answer balancing.
- Evidence: contrast pairs, OCR read-back, blind baseline, model evaluation, human audit.
- One self-contained HTML report per run; JSONL outputs; `vtf rebuild` for any single task.
- A difficulty knob: `configs/charts-hard.yaml` removes printed values and crowds the charts.

**Results (Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5, high effort):** every model answered all 450
tasks (300 standard, 150 hard) correctly, read unprinted values with a median error of 0.2% or
less, and got 2 to 11% of open-ended questions right without the image. Only contrast pairs
separated them: 400, 399 and 398 of 400 pairs with both images right; Sonnet and Haiku each once
repeated their first answer after the deciding printed value changed. Model disagreement also
exposed an ambiguous question and two too-thin margins in our own templates, all fixed.

Figures: `figures/contrast-pair.png`, `figures/model-accuracy.png`, `figures/twin-miss.png`.

## 5. Process flow

Seed → scene data → render image + layout record → question template → answer from the data →
gates → accepted tasks → evidence (pairs, OCR, model evaluation, human audit) → report.

Figure: `figures/pipeline.png`.

## 6. Wireframes

Screenshots of `report.html` (funnel, coverage, model breakdown, OCR, pairs) and `audit.html`
(blind answer, reveal, verdict). Take them from `samples/charts-seed42/report.html` and an audit
page built with `vtf audit samples/charts-seed42`.

## 7. Architecture

- `core/`: the family contract, pipeline, configuration, contrast pairs.
- `families/charts/`: scene sampler, renderer with layout record, 17 templates.
- `validators/`: grounding (polygon overlap on the layout record), ambiguity, dedup, answer scoring.
- `evidence/`: model evaluation (Claude via the Anthropic SDK, cached), OCR read-back (Tesseract),
  human audit.
- `report.py`: one self-contained HTML report that picks up whatever evidence a run has.

A new image family implements five methods and gets every gate, check and report for free.

## 8. Technologies

Python 3.13, uv, matplotlib, NumPy, Pillow, pydantic, Typer, PyYAML, Tesseract OCR, the Anthropic
Python SDK (Claude Opus 5.5), pytest, GitHub Actions.

## 9. Estimated cost

- Generation, contrast pairs and OCR: free, on a laptop (300 tasks in about 6 seconds; OCR of
  1,265 labels in about 17 seconds).
- Model evaluation, every task with and without the image plus every contrast twin (1,300
  requests per model across both sets): Opus 5.5 $4.15, Sonnet 5.5 $2.27, Haiku 5.5 $0.14.
  Replies are cached, so rebuilding results is free.
- Human audit: about 20 minutes per 100 tasks.

## 10. Deliverables

- **Runnable source code:** the repository; `uv sync` then `uv run vtf generate`.
- **Sample task packages:** `samples/charts-seed42` (300 tasks, 271 contrast pairs) and
  `samples/charts-hard-seed7` (150 hard tasks, 129 pairs), each with OCR results, model results
  and a report.
- **Validation test logs:** the CI run's `validation-logs` artifact (pytest log, generation, pairs
  and OCR logs, report) and the gate results stored on every task.
- **Technical documentation:** the README.
- **Live demonstration:** the demo video.
