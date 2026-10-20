# One image, several roles (api | worker | scorer | ui), chosen by the command.
# Multi-stage: dependencies are resolved from uv.lock in a builder; the runtime image has no
# compiler, no uv cache and runs as a non-root user.
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra ui
COPY src ./src
RUN uv sync --frozen --no-dev --extra ui

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* && useradd --uid 10001 --create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 AML_DATA_DIR=/data
USER app
EXPOSE 8000 8001 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; \
urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1
CMD ["aml-api", "serve"]
