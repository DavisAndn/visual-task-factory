# Visual Task Factory in a container: Python 3.13, Tesseract for the OCR read-back, and the locked
# dependencies. Build once, then run any vtf command (the README's "Run it in Docker" has more):
#
#   docker build -t vtf .
#   docker run --rm -v "$PWD/out:/app/out" vtf generate --family instruments --n 300
#   docker run --rm --entrypoint pytest vtf -q
#
# Model API keys are never part of the image: pass them at run time with --env-file .env.
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "uv==0.11.32"

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PATH="/app/.venv/bin:$PATH"

# The locked dependencies first, so a code change doesn't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project

COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY samples ./samples
RUN uv sync --locked

ENTRYPOINT ["vtf"]
CMD ["--help"]
