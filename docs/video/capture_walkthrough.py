"""Records the end-to-end run the video's walkthrough scene shows: the real commands, run from a
clean folder, with their real output, and a screenshot of the report they make.

    uv run python docs/video/capture_walkthrough.py

It writes docs/video/assets/walkthrough.json (each command and what it printed) and
docs/video/assets/report.png. The model step asks Claude Opus 5.5 and GPT-6.2 Sol about 75 tasks;
the run is the committed instruments sample made again from its seed, so their replies are copied
from that sample's cache and nothing is paid for. If any image differs from the committed one, it
stops before asking, rather than sending a request.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"
RUN = "out/instruments"
SAMPLE = ROOT / "samples" / "instruments-seed42"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
COMMANDS = [
    ("gen", f"uv run vtf generate --family instruments --n 300 --out {RUN}"),
    ("pairs", f"uv run vtf pairs {RUN}"),
    ("photos", f"uv run vtf photos {RUN}"),
    ("ocr", f"uv run vtf ocr {RUN} --photos"),
    ("models", f"uv run vtf models {RUN} -m claude-opus-5-5 -m gpt-6.2-sol --sample 75"),
]
PROGRESS = re.compile(r"^\s*\w+\s+\[[#\- ]*\]")  # typer progress bars


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_images() -> bool:
    """Whether every image of the new run is byte-for-byte the committed sample's."""
    made = sorted((ROOT / RUN / "images").iterdir())
    return bool(made) and all((SAMPLE / "images" / p.name).is_file() and digest(p) == digest(SAMPLE / "images" / p.name)
                              for p in made)


def run(command: str) -> list[str]:
    result = subprocess.run(command.split(), cwd=ROOT, capture_output=True, text=True, check=True)
    lines = []
    for raw in result.stdout.splitlines():
        line = raw.split("\r")[-1].rstrip()
        if PROGRESS.match(line):
            line = re.sub(r"\[[#\- ]*\]", "[" + "#" * 36 + "]", line)
        if line:
            lines.append(line.replace(str(ROOT) + "/", ""))
    return lines


def main() -> None:
    shutil.rmtree(ROOT / RUN, ignore_errors=True)
    steps = []
    for name, command in COMMANDS:
        if name == "models":
            if not same_images():
                raise SystemExit("The new run's images differ from the committed sample's; stopping before any paid request.")
            shutil.copy(SAMPLE / "model_cache.jsonl", ROOT / RUN / "model_cache.jsonl")
        output = run(command)
        if name == "models" and not any("$0.00 new this run" in line for line in output):
            raise SystemExit("A model reply was not cached; check the cost lines above.")
        steps.append({"cue": name, "command": command, "output": output})
        print(f"$ {command}\n" + "\n".join(output) + "\n")
    steps.append({"cue": "report", "command": f"open {RUN}/report.html", "output": []})
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "walkthrough.json").write_text(json.dumps(steps, indent=1) + "\n")
    # The report as a browser shows it, 1600 px wide and tall enough for its first sections.
    subprocess.run([str(CHROME), "--headless=new", "--hide-scrollbars", "--window-size=1600,7200",
                    f"--screenshot={ASSETS / 'report.png'}", (ROOT / RUN / "report.html").as_uri()],
                   check=True, capture_output=True)
    print(f"Wrote {ASSETS / 'walkthrough.json'} and {ASSETS / 'report.png'}")


if __name__ == "__main__":
    main()
