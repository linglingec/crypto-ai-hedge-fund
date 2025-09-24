#!/usr/bin/env sh
set -e

# MODE=api → запустить REST API, иначе — Jupyter Notebook
if [ "$MODE" = "api" ]; then
  exec uvicorn src.api.server:app --host 0.0.0.0 --port 8000
else
  exec python -m notebook --ip=0.0.0.0 --allow-root --NotebookApp.token=
fi