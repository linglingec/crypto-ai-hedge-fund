FROM python:3.11-slim

# system deps (gcc + OpenMP runtime for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./

RUN uv pip install -r pyproject.toml --system

RUN pip install --no-cache-dir jupyter ipykernel

COPY . .

RUN chmod +x docker/entrypoint.sh

ENV PYTHONPATH=/app

EXPOSE 8888
EXPOSE 8000

ENTRYPOINT ["docker/entrypoint.sh"]
CMD []
