"""Builds docs/demo.mp4: a narrated pitch for Visual Task Factory, animated from the committed
samples' real images and model results, with a music bed and burned-in subtitles.

    uv run python docs/video/make_video.py            # the full video
    uv run python docs/video/make_video.py --stills   # free preview: key frames on estimated timing
    uv run python docs/video/make_video.py --silent   # free preview: the whole video, no audio

The narration is one take of the whole script by the ElevenLabs voice "Jeevan" (Eleven v4). Its
timing endpoint says when every character is spoken, so scene changes, animations and subtitles
follow the voice: a {cue} in the script marks the word a picture waits for, and [tags] direct the
delivery without being spoken or shown. The music is an instrumental from the ElevenLabs Music
API, ducked under the voice. Both need ELEVENLABS_API_KEY (from the environment or the .env file)
and are cached in out/narration/, so rebuilding with an unchanged script costs nothing. Every
number on screen is read from samples/*/model_summary.json by data.py, and the walkthrough shows
the real commands and output that capture_walkthrough.py recorded in assets/.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import scenes as sc  # noqa: E402
from gfx import FPS, H, W, caption_box, new_frame, paste  # noqa: E402

from vtf.cli import _load_dotenv  # noqa: E402

ROOT = data.ROOT
WORK = ROOT / "out" / "video"
CACHE = ROOT / "out" / "narration"  # kept apart from WORK, which every build clears
OUT = ROOT / "docs" / "demo.mp4"

MODEL = "eleven_v4"
VOICE = "mCQMfsqGDT6IDkEKR20a"  # "Jeevan - Expressive and Confident" from the Voice Library: Indian English
FORMAT = "wav_44100"  # lossless; needs the Pro plan
SETTINGS = {"stability": 0.45, "similarity_boost": 0.75}
SEED = 42
RATE = 48000
LEAD, GAP, END = 0.5, 0.3, 2.2  # seconds of quiet: before the voice starts, before each later scene, after it ends
FADE = 0.35  # crossfade between scenes, seconds
TAG = re.compile(r"\[[^\]]*\]\s*")  # audio tags direct the delivery; they aren't spoken or captioned
NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
MUSIC_SECONDS = 270  # one bed, longer than the video and faded out at its end, so a new take doesn't need new music
CUE = re.compile(r"\{(\w+)\}")
MUSIC_PROMPT = (
    "Instrumental background music for a confident technology pitch video. Modern and optimistic: soft "
    "pulsing synth arpeggio, warm pads, light piano and gentle electronic percussion, about 100 BPM. Opens "
    "sparse and curious, builds steadily, lifts with energy in the middle, then resolves warmly at the end. "
    "Steady and unobtrusive under a spoken voice-over, no vocals, no sudden drops."
)


def script() -> list[tuple[sc.Scene, str]]:
    """The scenes in order, each with what the narrator says over it."""
    inst, conn, cnt, charts = "instruments-seed42", "connections-seed42", "counting-seed42", "charts-seed42"
    img = sc.image
    claude = [m for m in data.models() if m.lab == "Anthropic"]
    clean_charts = []
    for m in claude:
        right = sum(data.summary(r)["models"][m.key]["with_image"]["correct"] for r in ("charts-seed42", "charts-hard-seed7"))
        total = sum(data.summary(r)["models"][m.key]["with_image"]["of"] for r in ("charts-seed42", "charts-hard-seed7"))
        clean_charts.append((m.name, right, total, m.color))
    models = data.models()

    def say(n: int) -> str:
        return NUMBERS.get(n, str(n))

    # The cold open: a syringe every model reads right clean, then misreads photographed.
    hook_id = data.hook_task(inst)
    hook = sc.task(inst, hook_id)
    value = hook["answer"]["value"]
    spoken = f"{int(value) if float(value).is_integer() else value} {'millilitres' if hook['answer']['unit'] == 'mL' else hook['answer']['unit']}"
    wrong = sum(not data.answers(inst, hook_id, m)["photo"][1] for m in models)
    wrong_phrase = ("every one of them gets it wrong" if wrong == len(models) else "half of them get it wrong"
                    if 2 * wrong == len(models) else f"{say(wrong)} of the {say(len(models))} get it wrong")
    # The same tasks for every model: each claim below is checked against the data before it is spoken.
    same = {m.key: data.same_tasks(m) for m in models}
    shared = same[models[0].key]["tasks"]
    assert all(r["photo"] < r["photo_clean"] for r in same.values()), "not every model drops when photographed"
    lo, hi = (round(100 * f(r["photo_clean"] for r in same.values())) for f in (min, max))
    n_syr, syringes = data.same_facet(inst, lambda f: f["instrument"] == "syringe")
    perfect = [(m, photo) for m, clean, photo in syringes if clean == 1.0]
    held = [m for m, photo in perfect if photo == 1.0]
    slipped = [photo for m, photo in perfect if photo < 1.0]
    assert len(held) == 1 and all(photo > 0 for photo in slipped)
    big = [t for t in data.common(cnt) if data.checks(cnt)[(models[0].key, t)]["facets"]["difficulty"] in ("dots L5", "dots L6")]
    others = [m for m in models if m.lab != "Anthropic"]
    on_big = [data.tally(cnt, m, "image", big)[0] for m in others]
    note = (f"{', '.join(m.name for m in others[:-1])} and {others[-1].name} answered {len(data.common(cnt))} of these "
            f"tasks; on the {len(big)} with 500 or more points they got {', '.join(map(str, on_big[:-1]))} and {on_big[-1]} right.")
    assert max(on_big) <= 0.25 * len(big)
    morph_task = sc.task(inst, "instruments-000007-gauge_read")
    backdrop = [img(charts, "charts-000009.png"), img(inst, "instruments-000009.png"), img(conn, "connections-000016.png"),
                img(cnt, "counting-000002.png"), img(inst, "instruments-000012.png"), img(conn, "connections-000008.png"),
                img(charts, "charts-000002.png"), img(conn, "connections-000015.png"), img(inst, "instruments-000007.png"),
                img(charts, "charts-000005.png"), img(inst, "instruments-000050.png"), img(conn, "connections-000013.png"),
                img(cnt, "counting-000001.png"), img(charts, "charts-000000.png"), img(inst, "instruments-000036.png")]
    count = NUMBERS[len(models)]
    labs = sorted({m.lab for m in models})
    who = f"{count} Claude models" if labs == ["Anthropic"] else f"{count} models from {NUMBERS[len(labs)]} labs"
    return [
        (sc.Hook(inst, hook_id),
         f"[curious, measured] Here's a syringe. How much liquid is in it? {{right}}Ask {who.replace(' models', ' AI models', 1)}, and every one of them reads "
         f"it right: {spoken}. [short pause] Now take a photo of the same syringe, {{snap}}the way a phone would see it. "
         f"Same syringe. Same answer. {{wrong}}[quietly] And {wrong_phrase}."),
        (sc.Title("Visual Task Factory", "Proven answers. Real-world pictures. Every model's breaking point.",
                  "TEAM PLOT TWIST  ·  INNODATA AI HACKATHON  ·  TRACK 2", backdrop=backdrop, hold=0.4),
         "{title}[confident] That gap is exactly what Visual Task Factory is built to find."),
        (sc.Pictures([(img(charts, "charts-000002.png"), "Charts in reports", "charts"),
                      (img(inst, "instruments-000009.png"), "Gauges on plant floors", "gauges"),
                      (img(inst, "instruments-000006.png"), "Syringes on wards", "syringes"),
                      (img(conn, "connections-000008.png"), "Cables in server rooms", "cables")],
                     "How do you find out where it fails, before it matters?"),
         "AI already reads the pictures we work with: {charts}charts in reports, {gauges}gauges on plant floors, "
         "{syringes}syringes on hospital wards, {cables}cables in server rooms. {q}So how do you find out where it fails, "
         "before it matters?"),
        (sc.Benchmark([img(charts, f"charts-00000{i}.png") for i in (0, 5, 9, 2, 1)], img(charts, "charts-000002-photo.jpg"),
                      clean_charts),
         "The usual answer is a benchmark: a fixed set of images, labelled by hand. But models catch up, {spent}and the "
         "benchmark gets used up. {stat}On 450 clean charts, all three Claude models scored a perfect 100 percent. "
         "{solved}Clean pictures are solved. {real}But the real world isn't clean."),
        (sc.Factory([sc.answered(run, name) for run, name in (
            (inst, "instruments-000050.png"), (conn, "connections-000002.png"), (charts, "charts-000009.png"),
            (inst, "instruments-000012.png"), (cnt, "counting-000002.png"), (conn, "connections-000015.png"),
            (inst, "instruments-000009.png"), (conn, "connections-000008.png"))]),
         "[building energy] {factory}So I built a factory. It doesn't collect pictures. It manufactures them, from data: "
         "{families}charts, instruments, connection diagrams and dense plots. {proof}And because it drew every picture, it "
         "knows every answer. {guess}No labelling. No guesswork."),
        (sc.Checks(),
         "Then the factory tries to break its own work. {gates}Gates reject anything illegible, ambiguous or duplicated. "
         "{probes}Pixel probes check each drawing matches its data. {ocr}An OCR engine reads the labels back. {twins}Contrast "
         "twins flip one detail, so the answer must flip too. {audit}And a person can audit any batch, blind."),
        (sc.Dials([(img(conn, f"connections-0000{n}.png"), r) for n, r in
                   (("27", "8 cables"), ("26", "8, one colour"), ("08", "16 cables"), ("00", "24 cables"), ("13", "32 cables"))],
                  [(img(cnt, f"counting-0000{n}.png"), r) for n, r in
                   (("04", "30 points"), ("00", "70 points"), ("15", "150 points"), ("08", "300 points"), ("02", "515 points"),
                    ("01", "800 points"))],
                  morph_frames(inst, "instruments-000007"), sc.answer_text(morph_task)),
         "[energised] {d1}Then come the two dials. Dial one: how much is in the picture. {cables}Eight cables become "
         "thirty-two. {dots}Thirty points become eight hundred. {d2}Dial two: photo mode. {morph}Any task, re-shot the way a "
         "phone would: tilted, glared, blurred, compressed. {exact}And the answer stays exact, because the factory knows "
         "where every pixel went."),
        (sc.Dumbbell(f"The same {shared} tasks: clean, then photographed",
                     f"Claude answered all 990 tasks; GPT and Qwen a sample of {shared}. All six are compared on those {shared}."),
         f"Then I put {who} through the factory. Claude took all 990 tasks; GPT and Qwen, a sample of {shared}. "
         f"{{clean}}On those same {shared} tasks, clean images score from {lo} to {hi} percent. {{photo}}Photograph them, "
         "and every model drops."),
        (sc.Bars("Syringes, photographed", f"{n_syr} syringe readings every model answered, clean and photographed", syringes),
         f"Syringes show it most clearly. On {say(n_syr)} readings, {say(len(perfect))} of the {say(len(models))} models read "
         f"every clean syringe right. {{drop}}Photographed, {say(len(slipped))} of those {say(len(perfect))} slip, to as low as "
         f"{round(100 * min(slipped))} percent. [surprised] And here's a plot twist: the one that held is {held[0].name}. "
         f"It read all {say(n_syr)}, and beat every Claude model."),
        (sc.Breaking("Counting points: where each Claude model breaks", cnt, "dots", ["30", "70", "150", "300", "515", "800"],
                     "points in the plot (about)", {"lower": ["Haiku", "Sonnet"], "opus": ["Opus"]}, lab="Anthropic", note=note),
         "{draw}Turn up the size dial, and every model hits a wall. {lower}Claude Haiku and Sonnet break at about 300 points. "
         "{opus}Opus holds until about 500. At 800, even Opus gets just one in eight right. {note}On the biggest plots, GPT "
         "and Qwen hit the same wall. {blind}And without the image? Not one count right, from any model."),
        (sc.Walkthrough(json.loads((sc.ASSETS / "walkthrough.json").read_text()), str(sc.ASSETS / "report.png")),
         "{gen}And it's all one pipeline you can run yourself, on a laptop or in Docker. Here it is, end to end: three "
         "hundred tasks in seconds. {pairs}Their contrast twins. {photos}Their phone photos. {ocr}An OCR check of every "
         "label. {models}Then the models, with every reply cached, so this rerun cost nothing. {report}And one report holds "
         "it all: every task, every answer, and where each model breaks."),
        (sc.Anecdote(str(sc.ASSETS / "gear-out-of-mesh.png"), "Gear A is driven clockwise. Which gears turn counterclockwise?",
                     "“None.”", "Claude Opus 5.5"),
         "[warm] The models even keep me honest. {ask}In an early prototype, I asked which gears turn counterclockwise, "
         "and Opus said: none. {right}It was right. My generator had drawn gear A out of mesh. {fix}So now, every drawing "
         "is checked against its data."),
        (sc.Versus([("answers", "Answers", "labelled by hand", "computed and proven"),
                    ("life", "When models catch up", "start over", "turn the dial"),
                    ("edge", "Result", "one score", "each model's breaking point"),
                    ("domain", "A new domain", "a new labelling project", "five methods")]),
         "That's why a factory beats a benchmark. {answers}Its answers are proven, not labelled. {life}When models catch "
         "up, you don't start over: you turn the dial. {edge}Instead of one score, you get each model's breaking point. "
         "{domain}And a new domain is just five methods."),
        (sc.Value([("build", "For teams that build models", "An endless supply of hard, verified data"),
                   ("deploy", "For teams that deploy them", "A map of where their model breaks")]),
         "{build}For teams that build models, that's an endless supply of hard, verified data. {deploy}For teams that "
         "deploy them, it's a map of where their model breaks, before their users find out."),
        (sc.Title("Visual Task Factory", "Turn the dial. Find the edge.", "CLEAN PICTURES ARE SOLVED. THE REAL WORLD IS NEXT.",
                  ["github.com/DavisAndn/visual-task-factory", "Team Plot Twist  ·  Innodata AI Hackathon  ·  Track 2"],
                  backdrop=backdrop[::-1]),
         "{title}Clean pictures are solved. The real world is next. This is Visual Task Factory, from team Plot Twist. "
         "[confident] Turn the dial. Find the edge."),
    ]


def morph_frames(run: str, name: str, steps: int = 14) -> list[Image.Image]:
    """The clean image of a task photographed with gradually stronger camera settings, ending at the
    settings of its real photo twin."""
    from vtf.core.capture import shoot
    from vtf.core.runs import read_photos

    photos = read_photos(data.SAMPLES / run)
    twin = next(t for k, t in photos.items() if k.startswith(name))
    from vtf.families import get_family

    family = get_family(twin.family)
    picture, layout = family.render(family.scene_from_dict(twin.scene))
    full = twin.capture
    frames = []
    for i in range(steps):
        k = i / (steps - 1)
        capture = {
            **full,
            "corners": [[c[0] * k, c[1] * k] for c in full["corners"]],
            "shade": [full["shade"][0], full["shade"][1], full["shade"][2] * k],
            "glare": [*full["glare"][:3], full["glare"][3] * k],
            "blur": max(0.01, full["blur"] * k),
            "noise": full["noise"] * k,
            "jpeg_quality": round(95 - (95 - full["jpeg_quality"]) * k),
        }
        _, photo, _ = shoot(picture, layout, capture)
        photo.thumbnail((900, 760))
        frames.append(photo)
    return frames


# ---------------------------------------------------------------------------------------------
# Script parsing and timing.


def parse(text: str) -> tuple[str, dict[str, int]]:
    """The text to speak (cues removed, tags kept) and where in it each cue falls."""
    spoken, cues, pos = "", {}, 0
    for match in CUE.finditer(text):
        spoken += text[pos : match.start()]
        cues[match.group(1)] = len(spoken)
        pos = match.end()
    spoken += text[pos:]
    return spoken, cues


def caption_chunks(text: str) -> list[str]:
    """Sentences, with long ones split at a comma, colon or semicolon near the middle."""
    chunks = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        pending = [sentence]
        while pending:
            piece = pending.pop(0)
            marks = [m.start() for m in re.finditer(r"[,:;]", piece)]
            if len(piece) <= 96 or not marks:
                chunks.append(piece)
                continue
            cut = min(marks, key=lambda i: abs(i - len(piece) / 2))
            pending[:0] = [piece[: cut + 1], piece[cut + 1 :].strip()]
    return [c for c in chunks if c]


def plain(text: str) -> tuple[str, list[int]]:
    """Text without audio tags, and for each of its characters, its index in `text`."""
    out, where, pos = "", [], 0
    for tag in [*TAG.finditer(text), None]:
        stop = tag.start() if tag else len(text)
        out += text[pos:stop]
        where += range(pos, stop)
        pos = tag.end() if tag else stop
    return out, where


def captions(text: str) -> list[tuple[int, str]]:
    """Subtitle chunks of a scene, each with the index in the spoken text where it starts."""
    flat, where = plain(text)
    out, at = [], 0
    for chunk in caption_chunks(flat):
        at = flat.index(chunk, at)
        out.append((where[at], chunk))
        at += len(chunk)
    return out


class Timeline:
    """When each scene starts and ends, when its cues and subtitles fall, and the voice track."""

    def __init__(self) -> None:
        self.starts: list[float] = []
        self.lengths: list[float] = []
        self.cues: list[dict[str, float]] = []
        self.subs: list[list[tuple[float, str]]] = []
        self.voice: np.ndarray | None = None

    @property
    def total(self) -> float:
        return self.starts[-1] + self.lengths[-1]


def estimate(scenes: list[tuple[sc.Scene, str]], cps: float = 14.5) -> Timeline:
    """Timing guessed from text length, for the free previews."""
    tl, clock = Timeline(), 0.0
    for s, (scene, text) in enumerate(scenes):
        spoken, cues = parse(text)
        flat, where = plain(spoken)
        index = {w: i for i, w in enumerate(where)}

        def at(k: int) -> float:
            i = next((index[j] for j in range(k, len(spoken)) if j in index), len(flat))
            pauses = len(re.findall(r"[.!?]\s", flat[:i])) * 0.35
            return (LEAD if s == 0 else GAP) + i / cps + pauses

        length = at(len(spoken)) + scene.hold + (END if s == len(scenes) - 1 else 0.25)
        tl.starts.append(clock)
        tl.lengths.append(length)
        tl.cues.append({name: at(k) for name, k in cues.items()})
        tl.subs.append([(0.0 if i == 0 else at(k), chunk) for i, (k, chunk) in enumerate(captions(spoken))])
        clock += length
    return tl


# ---------------------------------------------------------------------------------------------
# Narration and music, from ElevenLabs, cached.


def _key() -> str:
    _load_dotenv(ROOT / ".env")
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("No ElevenLabs key: set ELEVENLABS_API_KEY, or add ELEVENLABS_API_KEY=... to .env.")
    return key


def _post(url: str, body: dict) -> bytes:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"xi-api-key": _key(), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"ElevenLabs answered {error.code}: {error.read().decode(errors='replace')}") from None


def narrate(text: str) -> tuple[Path, dict]:
    """One take of `text` and its character timing, from the cache or else from ElevenLabs."""
    body = {"text": text, "model_id": MODEL, "language_code": "en", "voice_settings": SETTINGS, "seed": SEED}
    name = hashlib.sha256(json.dumps([VOICE, FORMAT, body], sort_keys=True).encode()).hexdigest()[:16]
    audio, timing = CACHE / f"{name}.wav", CACHE / f"{name}.json"
    if audio.exists() and timing.exists():
        return audio, json.loads(timing.read_text())
    print(f"Narrating {len(text)} characters with ElevenLabs...", flush=True)
    reply = json.loads(_post(f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}/with-timestamps?output_format={FORMAT}", body))
    CACHE.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(base64.b64decode(reply["audio_base64"]))
    timing.write_text(json.dumps(reply["alignment"]))
    return audio, reply["alignment"]


def compose(seconds: float) -> Path:
    """An instrumental of about the given length, from the cache or else from ElevenLabs Music."""
    body = {"prompt": MUSIC_PROMPT, "music_length_ms": int(seconds * 1000), "model_id": "music_v1", "force_instrumental": True}
    name = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"music-{name}.mp3"
    if not path.exists():
        print(f"Composing {seconds:.0f} s of music with ElevenLabs...", flush=True)
        audio = _post("https://api.elevenlabs.io/v1/music?output_format=mp3_44100_192", body)
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
    return path


def char_times(text: str, alignment: dict) -> list[tuple[float, float] | None]:
    """When each character of `text` is spoken in the take: (start, end) seconds, or None if unmatched."""
    spoken, times = "", []
    for chars, start, end in zip(alignment["characters"], alignment["character_start_times_seconds"], alignment["character_end_times_seconds"]):
        spoken += chars
        times += [(start, end)] * len(chars)
    if spoken == text:
        return times
    matched: list[tuple[float, float] | None] = [None] * len(text)
    for a, b, size in difflib.SequenceMatcher(None, text, spoken, autojunk=False).get_matching_blocks():
        matched[a : a + size] = times[b : b + size]
    return matched


def decode(path: Path, channels: int = 1) -> np.ndarray:
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", str(channels), "-ar", str(RATE), "-"],
                         check=True, capture_output=True).stdout
    out = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    return out.reshape(-1, channels) if channels > 1 else out


def quietest(voice: np.ndarray, start: float, end: float) -> float:
    """The middle of the longest quiet stretch of the take between two times, in seconds."""
    hop = RATE // 100
    piece = voice[round(start * RATE) : round(end * RATE)].astype(np.float64)
    frames = piece[: len(piece) // hop * hop].reshape(-1, hop)
    level = np.sqrt((frames**2).mean(axis=1))
    quiet = level < 0.06 * np.sqrt((voice.astype(np.float64) ** 2).mean())
    best, first = (0, 0), None
    for i, q in enumerate([*quiet, False]):
        if q and first is None:
            first = i
        elif not q and first is not None:
            best = max(best, (first, i), key=lambda run: run[1] - run[0])
            first = None
    middle = (best[0] + best[1]) / 2 if best[1] > best[0] else int(np.argmin(level)) + 0.5
    return start + middle * hop / RATE


def faded(piece: np.ndarray, seconds: float = 0.01) -> np.ndarray:
    n = min(round(seconds * RATE), len(piece) // 2)
    ramp = np.linspace(0.0, 1.0, n)
    out = piece.copy()
    out[:n] *= ramp
    out[len(out) - n :] *= ramp[::-1]
    return out


def narrated(scenes: list[tuple[sc.Scene, str]]) -> Timeline:
    """Timing from one narrated take, cut into scenes at the pauses between them."""
    parsed = [parse(text) for _, text in scenes]
    texts = [p[0] for p in parsed]
    script_text = "\n\n".join(texts)
    offsets = [sum(len(t) + 2 for t in texts[:s]) for s in range(len(texts))]
    take, alignment = narrate(script_text)
    times = char_times(script_text, alignment)
    voice = decode(take)

    def letters(s: int) -> list[tuple[float, float]]:
        tagged = {k for tag in TAG.finditer(texts[s]) for k in range(tag.start(), tag.end())}
        found = [times[offsets[s] + k] for k, c in enumerate(texts[s]) if k not in tagged and c.isalnum()]
        return [t for t in found if t]

    def when(s: int, k: int) -> float:
        return next(t[0] for t in times[offsets[s] + k :] if t)

    cuts = [0.0]
    cuts += [quietest(voice, letters(s - 1)[-1][0] - 0.2, letters(s)[0][1] + 0.4) for s in range(1, len(scenes))]
    cuts += [len(voice) / RATE]
    tl, track, clock = Timeline(), [], 0.0
    for s, (scene, _) in enumerate(scenes):
        quiet = LEAD if s == 0 else GAP
        tail = scene.hold + (END if s == len(scenes) - 1 else 0.0)
        piece = faded(voice[round(cuts[s] * RATE) : round(cuts[s + 1] * RATE)])
        track += [np.zeros(round(quiet * RATE), np.float32), piece, np.zeros(round(tail * RATE), np.float32)]
        length = quiet + len(piece) / RATE + tail
        shift = quiet - cuts[s]
        tl.starts.append(clock)
        tl.lengths.append(length)
        tl.cues.append({name: when(s, k) + shift for name, k in parsed[s][1].items()})
        tl.subs.append([(0.0 if i == 0 else when(s, k) + shift, chunk) for i, (k, chunk) in enumerate(captions(texts[s]))])
        clock += length
    tl.voice = np.concatenate(track)
    return tl


def mix(voice: np.ndarray, music_path: Path) -> np.ndarray:
    """Stereo: the voice in the middle, the music under it, ducked while the voice speaks."""
    music = decode(music_path, channels=2)
    n = len(voice)
    if len(music) < n:
        music = np.concatenate([music, np.zeros((n - len(music), 2), np.float32)])
    music = music[:n]
    music = music / (np.sqrt((music**2).mean()) + 1e-9) * 10 ** (-20 / 20)  # music at about -20 dBFS RMS
    hop = RATE // 50
    frames = voice[: n // hop * hop].reshape(-1, hop)
    active = (np.sqrt((frames**2).mean(axis=1)) > 0.02).astype(np.float32)
    # Bridge pauses up to about 0.7 s, so the music doesn't swell between sentences, only in longer breaks.
    active = np.convolve(active, np.ones(35) / 35, mode="same") > 0.03
    env = np.zeros(len(active), np.float32)
    level = 0.0
    for i, on in enumerate(active):
        target = 1.0 if on else 0.0
        level += (target - level) * (0.3 if target > level else 0.03)  # fast attack, slow release
        env[i] = level
    env = np.repeat(env, hop)
    env = np.concatenate([env, np.full(n - len(env), env[-1] if len(env) else 0.0, np.float32)])
    # With the voice at about -22 dBFS: music about 18 dB under it while it speaks, 10 dB in the breaks.
    gain = 10 ** (np.interp(env, [0, 1], [-12.0, -20.0]) / 20)
    fade_in, fade_out = np.clip(np.arange(n) / (1.5 * RATE), 0, 1), np.clip((n - np.arange(n)) / (3.0 * RATE), 0, 1)
    music = music * (gain * fade_in * fade_out)[:, None]
    out = music + voice[:, None] * 0.9
    return np.clip(out, -1, 1)


# ---------------------------------------------------------------------------------------------
# Frames.


def render(scenes: list[tuple[sc.Scene, str]], tl: Timeline, t: float) -> Image.Image:
    """The frame at time t of the whole video."""
    s = max(i for i, start in enumerate(tl.starts) if start <= t + 1e-9)
    local = t - tl.starts[s]
    scene = scenes[s][0]
    frame = new_frame()
    scene.draw(frame, local, tl.lengths[s], cue_lookup(tl, s))
    if s > 0 and local < FADE:  # crossfade from the previous scene's last frame
        before = new_frame()
        prev = scenes[s - 1][0]
        prev.draw(before, tl.lengths[s - 1], tl.lengths[s - 1], cue_lookup(tl, s - 1))
        frame = Image.blend(before, frame, local / FADE)
    if scene.captions:
        chunks = tl.subs[s]
        current = [c for start, c in chunks if start <= local + 1e-9]
        if current:
            box = caption_box(current[-1])
            if box is not None:
                paste(frame, box, (W - box.width) / 2, H - box.height - 36)
    return frame


def cue_lookup(tl: Timeline, s: int):
    cues = tl.cues[s]

    def cue(name: str, default: float | None = 0.0) -> float | None:
        if name in cues:
            return cues[name]
        if default is None or name == "title":
            return default
        raise KeyError(f"scene {s} has no cue {{{name}}}")

    return cue


def stills(scenes: list[tuple[sc.Scene, str]], tl: Timeline, out_dir: Path) -> None:
    """A contact sheet per scene: frames just after each cue and at the end."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, (scene, _) in enumerate(scenes):
        moments = sorted({0.6, *[c + 1.4 for c in tl.cues[s].values()], tl.lengths[s] - 0.05})
        moments = [m for m in moments if m < tl.lengths[s]]
        thumbs = [render(scenes, tl, tl.starts[s] + m).convert("RGB").resize((W // 3, H // 3)) for m in moments]
        cols = 3
        sheet = Image.new("RGB", (cols * W // 3, ((len(thumbs) + cols - 1) // cols) * H // 3), (0, 0, 0))
        for i, im in enumerate(thumbs):
            sheet.paste(im, ((i % cols) * W // 3, (i // cols) * H // 3))
        sheet.save(out_dir / f"scene-{s:02d}-{type(scene).__name__.lower()}.png")
    print(f"Stills in {out_dir}")


def encode(scenes: list[tuple[sc.Scene, str]], tl: Timeline, audio: Path | None, out: Path) -> None:
    frames = round(tl.total * FPS)
    command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio is not None:
        command += ["-i", str(audio), "-c:a", "aac", "-b:a", "192k", "-ar", str(RATE)]
    command += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", str(out)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(frames):
        proc.stdin.write(render(scenes, tl, i / FPS).convert("RGB").tobytes())
        if i % (FPS * 10) == 0:
            print(f"  {i / FPS:5.0f} s of {tl.total:.0f} s", flush=True)
    proc.stdin.close()
    if proc.wait():
        raise SystemExit("ffmpeg failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stills", action="store_true", help="contact sheets on estimated timing, no API calls")
    parser.add_argument("--silent", action="store_true", help="the whole video on estimated timing, without audio")
    parser.add_argument("--audio-only", action="store_true", help="rebuild the soundtrack and swap it into docs/demo.mp4")
    args = parser.parse_args()
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    scenes = script()
    for s, (_, text) in enumerate(scenes):
        missing = set(CUE.findall(text)) ^ set(parse(text)[1])
        assert not missing, (s, missing)
    if args.stills or args.silent:
        tl = estimate(scenes)
        print(f"Estimated length {tl.total / 60:.1f} min")
        if args.stills:
            stills(scenes, tl, WORK / "stills")
        else:
            encode(scenes, tl, None, WORK / "silent.mp4")
            print(f"{(WORK / 'silent.mp4').relative_to(ROOT)}")
        return
    tl = narrated(scenes)
    assert tl.voice is not None
    timeline = [{"scene": type(scene).__name__, "start": round(tl.starts[s], 3), "length": round(tl.lengths[s], 3),
                 "cues": {k: round(v, 3) for k, v in tl.cues[s].items()}, "subtitles": tl.subs[s]}
                for s, (scene, _) in enumerate(scenes)]
    (WORK / "timeline.json").write_text(json.dumps(timeline, indent=1))
    music = compose(max(MUSIC_SECONDS, tl.total + 1.0))
    mixed = mix(tl.voice, music)
    raw = WORK / "mix.wav"
    with wave.open(str(raw), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((mixed * 32767).astype(np.int16).tobytes())
    final = WORK / "mix-norm.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(RATE), str(final)], check=True)
    if args.audio_only:  # keep the rendered pictures, replace the soundtrack
        swapped = WORK / "swapped.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(OUT), "-i", str(final), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-ar", str(RATE), "-shortest", "-movflags", "+faststart", str(swapped)], check=True)
        shutil.move(swapped, OUT)
        print(f"{OUT.relative_to(ROOT)}: new soundtrack")
        return
    print(f"Rendering {tl.total:.0f} s of video...", flush=True)
    encode(scenes, tl, final, OUT)
    print(f"{OUT.relative_to(ROOT)}: {tl.total / 60:.1f} min, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
