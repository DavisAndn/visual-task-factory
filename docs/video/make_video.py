"""Builds docs/demo.mp4: a narrated walkthrough made from real command output, the run report, the
audit page, the figures and the deck's slides, with burned-in captions.

    uv run python docs/video/make_video.py

macOS only (narration uses the system voice through `say`); needs ffmpeg. Inputs are made first:
the terminal text in out/media/term/, report and audit screenshots in out/media/, and slide
renders in out/media/slides/.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "out" / "media"
FIGS = ROOT / "docs" / "figures"
WORK = ROOT / "out" / "video"
OUT = ROOT / "docs" / "demo.mp4"

W, H = 1920, 1080
BAND = 150  # caption band height at the bottom
STAGE = (W, H - BAND)
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
SANS = str(FONTS / "DejaVuSans.ttf")
MONO = str(FONTS / "DejaVuSansMono.ttf")
PAPER, NAVY = (246, 245, 241), (20, 32, 58)
VOICE, RATE, TAIL = "Samantha", 172, 0.8  # words per minute; seconds of quiet after each scene

# Each scene: what is on screen, and what the narrator says (also the captions).
SCENES = [
    (("image", MEDIA / "slides" / "cover.png"),
     "This is Visual Task Factory, my entry for Track 2 of the Innodata AI Hackathon. It generates chart "
     "images, questions about them, and reference answers that no model ever sets. It proves every task is "
     "valid, and uses the tasks to measure how well frontier models read charts."),
    (("image", FIGS / "examples.png"),
     "Charts carry the numbers in reports, decks and papers, and a model that misreads one returns a "
     "plausible wrong figure that nobody catches. The factory covers six chart types, five topics and "
     "seventeen question templates, from reading a single value to counterfactuals: if this bar were "
     "doubled, which would lead?"),
    (("terminal", "uv run vtf generate --n 300 --seed 42", "generate"),
     "One command makes the dataset: three hundred tasks from 153 images, in about six seconds. Each "
     "scene is sampled from a seed and drawn with matplotlib, which also records where every label, bar "
     "and line landed. The answer is computed from the scene's data, with a step by step trace and a "
     "tolerance."),
    (("image", MEDIA / "report-top.png"),
     "Every candidate then passes four gates: determinism, grounding, ambiguity and deduplication. Here, "
     "143 of 443 candidates were rejected, each with its reason, and the run report shows the funnel."),
    (("image", MEDIA / "report-rejected.png"),
     "It also shows real rejected examples, with the failing elements boxed in red: labels that collide, "
     "gaps too small to see, and values that can't be read precisely enough."),
    (("terminal", "uv run vtf rebuild charts-000002-read_value", "rebuild"),
     "Any single task can be rebuilt from its seed, and the scene, the pixels and the answer come out the "
     "same."),
    (("terminal", "uv run vtf ocr out/demo-run", "ocr"),
     "Then come the checks that don't trust the generator. Tesseract, an OCR engine unrelated to any model "
     "being evaluated, reads every label each question depends on from the final image: 99.7 percent match. "
     "On its first run, this check found value labels that a line ran straight through, which the layout "
     "check couldn't see, so I fixed the gate."),
    (("image", FIGS / "contrast-pair.png"),
     "Contrast pairs ask the same question about a copy of the image with one value edited, so the answer "
     "changes, while the axis stays pinned and nothing else moves. A model that answers from the question "
     "text, or from habit, can't get both right."),
    (("image", MEDIA / "audit.png"),
     "And a blind human audit: a person answers each task from the image alone, then sees the reference "
     "answer and judges it, and the valid rate comes with a 95 percent confidence interval."),
    (("terminal", "uv run vtf models samples/charts-seed42 -m claude-opus-5-5 -m claude-sonnet-5-5 -m claude-haiku-5-5", "models"),
     "Finally, I evaluated three Claude models, Opus, Sonnet and Haiku 5.5, on 450 tasks, a standard "
     "set and a hard one with nothing printed, asking each task with and without the image, plus every "
     "contrast twin. Replies are cached, so this rerun cost nothing."),
    (("image", MEDIA / "slides" / "results.png"),
     "All three answered every task correctly, reading unprinted values to within about a pixel. Without "
     "the image they got only six to ten percent of open-ended questions right, so the tasks really need "
     "the image."),
    (("image", FIGS / "twin-miss.png"),
     "What separated them was consistency. Here Sonnet answered Metro for both charts, even after South's "
     "printed value rose from 112 to 237. Haiku did the same once. Opus never did."),
    (("image", MEDIA / "slides" / "evidence.png"),
     "The evaluation also found defects in the factory itself: a scorer that read the working instead of the "
     "answer, two ambiguous questions, and two margins that were too thin. Every one was fixed and "
     "re-checked."),
    (("image", MEDIA / "slides" / "deliverables.png"),
     "Everything is reproducible: the code, two evaluated sample runs with their reports, and continuous "
     "integration that regenerates the samples. Chart reading at this level is saturated for current "
     "models, and the factory is built to add the harder task families that will show where they break."),
]


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def fit(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    scale = min(box[0] / image.width, box[1] / image.height)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def stage_image(path: Path) -> Image.Image:
    canvas = Image.new("RGB", STAGE, PAPER)
    picture = fit(Image.open(path).convert("RGB"), (STAGE[0] - 80, STAGE[1] - 60))
    canvas.paste(picture, ((STAGE[0] - picture.width) // 2, (STAGE[1] - picture.height) // 2))
    return canvas


def stage_terminal(command: str, typed: int, output: list[str]) -> Image.Image:
    canvas = Image.new("RGB", STAGE, PAPER)
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = 150, 50, W - 150, STAGE[1] - 40
    draw.rounded_rectangle((left, top, right, bottom), radius=18, fill=(29, 33, 45))
    for i, color in enumerate(((237, 106, 94), (245, 191, 79), (98, 197, 84))):
        draw.ellipse((left + 26 + i * 30, top + 18, left + 44 + i * 30, top + 36), fill=color)
    mono = font(MONO, 30)
    width = (right - left - 80) // 18  # characters per line at this size
    lines = textwrap.wrap("$ " + command[:typed], width, subsequent_indent="  ") or ["$ "]
    y = top + 70
    for i, line in enumerate(lines):
        if i == 0:
            draw.text((left + 40, y), "$", font=mono, fill=(126, 211, 130))
            draw.text((left + 40 + 36, y), line[2:], font=mono, fill=(236, 239, 244))
        else:
            draw.text((left + 40, y), line, font=mono, fill=(236, 239, 244))
        y += 42
    y += 14
    # Long output gets tighter lines and a smaller font, so it stays inside the window.
    pitch = min(40.0, (bottom - 28 - y) / max(len(output), 1))
    small = font(MONO, min(30, int(pitch * 0.78)))
    for line in output:
        draw.text((left + 40, round(y)), line[:width], font=small, fill=(190, 199, 214))
        y += pitch
    return canvas


def caption_chunks(text: str) -> list[str]:
    """Sentences, with long ones split at a comma or colon near the middle, at most two lines each."""
    chunks = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence) <= 150:
            chunks.append(sentence)
            continue
        cut = min((m.start() for m in re.finditer(r"[,:;]", sentence)), key=lambda i: abs(i - len(sentence) / 2))
        chunks += [sentence[: cut + 1], sentence[cut + 1 :].strip()]
    return chunks


def frame(stage: Image.Image, caption: str) -> Image.Image:
    canvas = Image.new("RGB", (W, H), NAVY)
    canvas.paste(stage, (0, 0))
    draw = ImageDraw.Draw(canvas)
    sans = font(SANS, 34)
    lines = textwrap.wrap(caption, 92)[:2]
    y = STAGE[1] + (BAND - 46 * len(lines)) // 2
    for line in lines:
        width = draw.textlength(line, font=sans)
        draw.text(((W - width) / 2, y), line, font=sans, fill=(243, 241, 234))
        y += 46
    return canvas


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "frames").mkdir(parents=True)
    concat, audio_list, n = [], [], 0
    total = 0.0
    for s, (visual, text) in enumerate(SCENES):
        voice = WORK / f"scene{s:02d}.aiff"
        run("say", "-v", VOICE, "-r", str(RATE), "-o", str(voice), text)
        spoken = float(run("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(voice)))
        length = spoken + TAIL
        padded = WORK / f"scene{s:02d}.wav"
        run("ffmpeg", "-y", "-v", "error", "-i", str(voice), "-af", f"apad=whole_dur={length:.3f}", "-ar", "48000", "-ac", "1", str(padded))
        audio_list.append(f"file '{padded}'")

        # Visual events: (start time, stage image).
        if visual[0] == "image":
            events = [(0.0, stage_image(visual[1]))]
        else:
            _, command, name = visual
            output = (MEDIA / "term" / f"{name}.txt").read_text().splitlines()
            steps = 10
            events = [(0.0, stage_terminal(command, 0, []))]
            events += [(0.3 + 1.2 * k / steps, stage_terminal(command, round(len(command) * k / steps), [])) for k in range(1, steps + 1)]
            events.append((1.9, stage_terminal(command, len(command), output)))
        # Caption events, timed in proportion to their length.
        chunks = caption_chunks(text)
        sizes = [len(c) for c in chunks]
        starts, t = [], 0.0
        for size in sizes:
            starts.append(t)
            t += spoken * size / sum(sizes)
        changes = sorted({e[0] for e in events} | set(starts))
        for i, start in enumerate(changes):
            end = changes[i + 1] if i + 1 < len(changes) else length
            stage = [e[1] for e in events if e[0] <= start][-1]
            caption = [c for c, st in zip(chunks, starts) if st <= start][-1]
            path = WORK / "frames" / f"f{n:04d}.png"
            frame(stage, caption).save(path)
            concat += [f"file '{path}'", f"duration {end - start:.3f}"]
            n += 1
        total += length
    concat.append(concat[-2])  # the concat demuxer needs the last frame listed twice
    (WORK / "frames.txt").write_text("\n".join(concat) + "\n")
    (WORK / "audio.txt").write_text("\n".join(audio_list) + "\n")
    run("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(WORK / "audio.txt"), str(WORK / "audio.wav"))
    run(
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(WORK / "frames.txt"),
        "-i", str(WORK / "audio.wav"), "-fps_mode", "cfr", "-r", "30", "-c:v", "libx264", "-preset", "medium",
        "-crf", "21", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart", str(OUT),
    )
    print(f"{OUT.relative_to(ROOT)}: {total / 60:.1f} min, {OUT.stat().st_size / 1e6:.1f} MB, {n} frames")


if __name__ == "__main__":
    main()
