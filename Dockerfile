FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY talanton ./talanton

RUN pip install --no-cache-dir ".[cli,openai]"

ENTRYPOINT ["talanton"]
CMD ["--help"]
