FROM python:3.11-slim

# system deps (gcc + OpenMP runtime for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# fast installer
RUN pip install --no-cache-dir uv

WORKDIR /app

# >>> copy files that pyproject references (README.md) BEFORE install
COPY pyproject.toml README.md ./

# install project deps into system env
RUN uv pip install -r pyproject.toml --system

RUN pip install --no-cache-dir jupyter ipykernel

# now copy the rest of the project (code, notebooks, data)
COPY . .

ENV PYTHONPATH=/app

EXPOSE 8888
CMD ["python", "-m", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token="]
