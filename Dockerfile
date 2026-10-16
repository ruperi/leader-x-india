FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY app.py ./app.py

RUN uv sync --frozen --no-dev

RUN uv run python -m leader_x.synthetic --output-dir data/synthetic --database data/leader_x.duckdb

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
