FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

COPY src/cli/ ./cli/
COPY src/minijax/ ./minijax/

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "cli"]
