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
| Solution deck | The deck, filled from [deck.md](deck.md), exported as a PDF under 5 MB |

### Prototype brief (965 characters)

> A factory of image questions whose answers are proven, not labelled, built to find where vision models break. It draws four families from data: business charts; instruments (pressure gauges, syringes, clocks); connection diagrams (patch panels, valve networks, gear trains); dense scatter plots. Every answer is computed from the scene. Gates reject anything illegible, ambiguous or duplicated, pixel probes check each drawing matches its data, and OCR, contrast pairs, independent solvers and a blind human audit check the rest. Two dials find each model's edge: size (8 to 32 cables, 30 to 800 points) and photo mode, which re-shoots any task as a phone photo with the same answer. On 990 tasks, Claude Haiku, Sonnet and Opus 5.5 score 92%, 95% and 97% clean but 84%, 90% and 91% photographed; counting breaks Haiku and Sonnet at about 300 points, Opus at about 515. On a shared 255-task sample, GPT-6.2 Sol holds best photographed (95%); Qwen 3.8 27B scores 58%.

### Technology stack (986 characters)

> Python 3.13 with uv (locked dependencies). matplotlib (Agg renderer, bundled fonts) draws every family deterministically; a shared drawing layer records each label's and mark's pixel box. NumPy and Pillow simulate phone photos (perspective, uneven light, glare, blur, noise, JPEG) and run pixel probes, hashes and crops. pydantic v2 models scenes, layouts, tasks and typed answers (numbers with tolerances, names, sets, clock times). Typer CLI: generate, pairs, photos, ocr, models, audit, report, rebuild, compare; PyYAML configs. Validation: overlap checks on the layout record, ambiguity and resolution gates, dHash dedup, drawing-matches-data probes, Tesseract 5 OCR on clean images and photos, independent solvers in pytest. Evaluation: the Anthropic SDK (Claude Haiku, Sonnet, Opus 5.5) and the openai SDK (GPT-6 Luna, GPT-6.2 Sol; Qwen 3.8 27B on Cerebras), threaded, with a per-run reply cache. HTML reports and audit pages; GitHub Actions CI regenerates every committed sample.

## Demo video

[demo.mp4](demo.mp4) is a four-minute pitch with a walkthrough. It opens on a syringe that six
models from three labs all read right as a clean drawing, and half of them misread as a phone
photo. It then covers why fixed benchmarks get used up, the factory and how it checks its own work,
the two dials (size and photo mode), the six models compared on the same tasks, and where each model
breaks. Then the pipeline runs end to end, with real terminal output recorded by
[video/capture_walkthrough.py](video/capture_walkthrough.py): generate, contrast twins, photos, OCR,
model checks (from the cache) and the report. It closes on the bug a model caught in a prototype
and why a factory beats a benchmark. It says on screen and aloud that Claude answered all 990 tasks
and GPT and Qwen a 255-task sample. It is narrated in Indian English by the ElevenLabs voice "Jeevan" (Eleven v4) over an
instrumental from the ElevenLabs Music API. The animation is drawn from the committed samples by
[video/make_video.py](video/make_video.py), with every number on screen read from the samples'
model results. `--stills` and `--silent` preview it without any API calls.
