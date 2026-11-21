# Submission kit

Everything the Hack2skill prototype form and the Innodata deck template ask for. The form closes
on 6 Dec 2026 at 11:55 PM IST.

## Form fields

| Field | Value |
|---|---|
| Challenge | Visual Task Factory |
| Prototype brief (max 1024 characters) | Below |
| Technology stack (max 1024 characters) | Below |
| GitHub public repository link | The repository URL. **Make the repository public before submitting.** |
| Demo video link | Your upload (outline below) |
| Solution deck | The template, filled from [deck.md](deck.md), exported as a PDF under 5 MB |

### Prototype brief (940 characters)

> A factory for chart-understanding tasks whose answers no model ever sets. Each scene is sampled from a seed, rendered with matplotlib and recorded element by element; 17 templates across 6 chart types ask about it and compute the answer from the data, with a trace and tolerance. Gates reject anything ambiguous, illegible, cluttered, duplicated or not reproducible, and answers are balanced against guessing. Independent evidence: Tesseract reads back 99.7% of the labels questions depend on, contrast pairs flip each answer with a minimal edit, seeds rebuild byte for byte (checked in CI), and a blind human audit gives a valid rate with a 95% interval. Results: Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 each answered all 450 tasks (standard and hard presets), read unprinted values to within a pixel, and got 2-11% of open-ended questions without the image. Only contrast pairs separated them (400, 399 and 398 of 400 pairs both right).

### Technology stack (952 characters)

> Python 3.13 with uv and a locked dependency set. matplotlib (Agg renderer, bundled DejaVu fonts) for deterministic chart rendering; NumPy and Pillow for pixel hashes, crops and image processing; pydantic v2 for the task, layout and answer schemas; Typer for the vtf command line; PyYAML for configs. Validation: a grounding gate using separating-axis polygon overlap on the renderer's layout record, ambiguity and dedup (256-bit dHash) gates, and Tesseract 5 OCR for independent label read-back. Model evaluation: the Anthropic Python SDK with Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 (adaptive thinking, explicit effort, server-side refusal fallback where supported), threaded requests and a per-run reply cache. Human audit: a self-contained HTML/JS page with blind answering and Wilson confidence intervals. Self-contained HTML reports. pytest suite and GitHub Actions CI that checks byte-identical output on Linux against hashes recorded on macOS.

## Demo video outline (about 4 minutes)

1. **The problem (30 s).** Charts carry the numbers in reports; a model that misreads one returns
   a plausible wrong figure that nobody catches. Show a stacked bar and a pie from the sample.
2. **One command (30 s).** `uv run vtf generate --n 300 --seed 42`: 300 tasks in 6 seconds. Open
   `report.html`: the rejection funnel and a rejected example with its red box.
3. **Why the answers are right (60 s).** A task's trace; `vtf rebuild` printing "same" three
   times; the OCR read-back table and the line-through-label miss that led to a new gate rule; a
   contrast pair side by side.
4. **Evaluating models (60 s).** `vtf models` output; the accuracy breakdown by skill and chart
   type; two tasks the model got wrong, and the same model getting one twin of a pair right and
   the other wrong.
5. **Human audit (30 s).** Answer two tasks blind in `audit.html`, import, show the valid rate and
   its interval.
6. **Close (30 s).** What it would take to add a new image family, and the CI run passing.
