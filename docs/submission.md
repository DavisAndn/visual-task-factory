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
| Demo video link | https://github.com/DavisAndn/visual-task-factory/blob/main/docs/demo.mp4 (works once the repository is public) |
| Solution deck | The template, filled from [deck.md](deck.md), exported as a PDF under 5 MB |

### Prototype brief (944 characters)

> A factory for chart-understanding tasks whose answers no model ever sets. Each scene is sampled from a seed, rendered with matplotlib and recorded element by element; 17 templates across 6 chart types ask about it and compute the answer from the data, with a trace and tolerance. Gates reject anything ambiguous, illegible, cluttered, duplicated or not reproducible, and answers are balanced against guessing. Independent evidence: Tesseract reads back 99.7% of the labels questions depend on, contrast pairs flip each answer with a minimal edit, any task rebuilds from its seed (checked in CI), and a blind human audit gives a valid rate with a 95% interval. Results: Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 each answered all 450 tasks (standard and hard presets), read unprinted values to within a pixel, and got 2-11% of open-ended questions without the image. Only contrast pairs separated them (400, 399 and 398 of 400 pairs both right).

### Technology stack (931 characters)

> Python 3.13 with uv and a locked dependency set. matplotlib (Agg renderer, bundled DejaVu fonts) for deterministic chart rendering; NumPy and Pillow for pixel hashes, crops and image processing; pydantic v2 for the task, layout and answer schemas; Typer for the vtf command line; PyYAML for configs. Validation: a grounding gate using separating-axis polygon overlap on the renderer's layout record, ambiguity and dedup (256-bit dHash) gates, and Tesseract 5 OCR for independent label read-back. Model evaluation: the Anthropic Python SDK with Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 (adaptive thinking, explicit effort, server-side refusal fallback where supported), threaded requests and a per-run reply cache. Human audit: a self-contained HTML/JS page with blind answering and Wilson confidence intervals. Self-contained HTML reports. pytest suite and GitHub Actions CI that regenerates the committed samples and checks them.

## Demo video

[demo.mp4](demo.mp4) is a 3.5-minute narrated walkthrough with burned-in captions, in this order:

1. The problem, and one task per chart type.
2. `vtf generate`: 300 tasks in about 6 seconds; the report's rejection funnel and rejected
   examples.
3. `vtf rebuild` on one task.
4. The checks that don't trust the generator: OCR read-back, contrast pairs and the blind audit
   page.
5. `vtf models` on three Claude models, the results, and the contrast twin Sonnet got wrong.
6. The defects the evaluation found in the factory, and the deliverables.

It is built from real command output, report screenshots, the figures and the deck's slides by
[video/make_video.py](video/make_video.py).
