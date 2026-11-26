# Deck content

Slide by slide, in the order of the Innodata template ("Innodata | IDEA SUBMISSION"). Figures are in
[figures/](figures/); rebuild them with `uv run python docs/figures/make_figures.py`.

## 1. Title

- **Team name:** Plot Twist (a team of one)
- **Team leader name:** Davis Andherson
- **Problem statement:** Visual Task Factory: finding where frontier vision models break, on the
  images people photograph at work

## 2. Brief about the idea

A factory of image questions whose answers are proven, not labelled. It draws charts, medical and
industrial instruments, connection diagrams and dense plots from data, computes every answer from
that data, checks each task without trusting any model, and then turns two dials real life turns:
how much is in the picture, and whether it was taken with a phone.

Figure: `figures/families.png` (one task from each kind of image).

## 3. Opportunity

**The problem with fixed benchmarks.** A benchmark is used up once models ace it, and its labels are
only as good as its annotators. Clean diagrams are largely solved: on 450 clean charts Claude Haiku,
Sonnet and Opus 5.5 scored 100%, and on clean clocks, gauges and syringes 98-100%.

**How the factory differs.** It doesn't stop at one picture. Every task has a scene behind it, so
the factory can make it harder (more cables, finer ticks, more points) or photograph it, and the
answer stays exact. That finds the edge of each model, and keeps making fresh, verified tasks there.

**USP.**
- Photo mode: any task re-shot as a phone photo with the same, provable answer. It cost every model,
  Opus included: photographed syringes fell from 100% to 54% (Haiku), 84% (Sonnet), 76% (Opus).
- Difficulty dials with breaking points: one command reports the level where each model drops
  below 80%.
- Answers no model sets, checked independently: pixel probes, OCR on clean images and photos,
  contrast pairs, independent solvers, a blind human audit.
- Any model, any lab: six models from Anthropic, OpenAI and Alibaba compared on the same tasks.
- One engine, four domains with stated costs of error: a dose misread by a tick, a tank thought
  isolated that isn't, a cable unplugged by mistake.

## 4. Features

- Four families: charts (6 types, 17 templates), instruments (gauges, syringes, clocks), connections
  (patch panels, valve networks, gear trains), counting (dense scatter plots).
- Difficulty levels on every new family; a hard preset for charts.
- `vtf photos`: photographed twins, gates re-run on the photo.
- Gates: determinism, drawing-matches-data probes, grounding, ambiguity and resolution, dedup,
  answer balancing.
- Evidence: contrast pairs, OCR read-back (clean and photo), blind baseline, model evaluation with
  breaking points, human audit.
- One self-contained HTML report per run; `vtf rebuild` re-derives any task.

**Results (990 tasks in four families, Claude Haiku, Sonnet and Opus 5.5, high effort):**

- Clean: 92%, 95% and 97% right; the same tasks photographed: 84%, 90% and 91%.
- Photographed syringes: 54%, 84% and 76% (each 100% clean); 1 mL syringes: 46%, 76% and 61%.
- Counting breaks Haiku and Sonnet at about 300 points and Opus at about 515 (73%); at about 800
  points, Opus gets 13% right.
- Tracing one of 24 identical cables breaks Haiku (43%) and Sonnet (79%); Opus holds until 32 (75%).
- Charts, clocks, gauges, valve networks and gear trains stay at 89-100%, clean or photographed.
- Without the image, 0 to 14% of open-ended questions are answered right; contrast pairs catch the rest.

**Across three labs (GPT-6 Luna, GPT-6.2 Sol and Qwen 3.8 27B on a labelled 255-task sample; all six
models compared on those same tasks):**

- Clean 58% to 96%, photographed 56% to 95%: every model drops when photographed.
- GPT-6.2 Sol ties Opus clean (96%) and holds best photographed (95%): the only model to read all 22
  photographed syringes right. Claude falls to 55-82% on them, GPT-6 Luna to 73%.
- Qwen 3.8 27B (open weights, served by Cerebras) reads instruments to the nearest printed number:
  31% on instruments, 7% on counting.
- A model found a bug in a prototype: Opus said no gear turned, and it was right; pixel probes now
  check every drawing matches its data.

Figures: `figures/photos.png`, `figures/breaking-points.png`, `figures/same-tasks.png`.

## 5. Process flow

Seed → scene → render image + layout record + probes → question → answer from the scene → gates →
accepted tasks → twins (edited, photographed) → evidence (OCR, models, human audit) → report.

Figure: `figures/pipeline.png`.

## 6. Wireframes

Screenshots of `report.html` (breaking points, phone photos, model breakdown) and
`audit-photos.html` (blind answer, reveal, verdict), from `samples/instruments-seed42`.

## 7. Architecture

- `core/`: the family contract, pipeline, configuration, contrast pairs, photo mode.
- `families/`: charts, instruments, connections, counting, and a shared drawing helper that records
  the layout of any matplotlib drawing.
- `validators/`: grounding, ambiguity and resolution, dedup, answer scoring (numbers, names, sets,
  clock times).
- `evidence/`: model evaluation (Claude via the Anthropic SDK; GPT and Cerebras-served models via the
  openai SDK; cached), OCR read-back (Tesseract), human audit.
- `report.py`: one HTML report per run, with breaking points and clean-vs-photo comparisons.

A new family implements five methods and gets every gate, twin, check and report for free.

## 8. Technologies

Python 3.13, uv, matplotlib, NumPy, Pillow, pydantic, Typer, PyYAML, Tesseract OCR, the Anthropic
Python SDK (Claude Haiku, Sonnet and Opus 5.5), the openai Python SDK (GPT-6 Luna and GPT-6.2 Sol;
Qwen 3.8 27B on Cerebras), pytest, GitHub Actions.

## 9. Estimated cost

- Generation, pairs, photos and OCR: free, on a laptop (300 tasks in 5 to 15 seconds; 300 photos in
  about 10 seconds).
- Model evaluation, every task with and without the image plus its contrast and photo twins (about
  3,800 requests per model): Haiku 5.5 $1.23, Sonnet 5.5 $16.77, Opus 5.5 $30.02. The 255-task sample
  for the other three: GPT-6 Luna $0.32, GPT-6.2 Sol $3.16, Qwen 3.8 27B $7.21. Replies are cached,
  so rebuilding results is free.
- Human audit: about 20 minutes per 40 tasks.

## 10. Deliverables

- **Runnable source code:** the repository; `uv sync` then `uv run vtf generate --family instruments`.
- **Sample task packages:** five committed runs in `samples/` (charts standard and hard, instruments,
  connections, counting), each with pairs, photos, OCR, model results and a report.
- **Validation test logs:** the CI run's `validation-logs` artifact and the gate results stored on
  every task.
- **Technical documentation:** the README.
- **Live demonstration:** the demo video.
