#!/usr/bin/env bash
set -e

MODE="${MODE:-$1}"   # accept MODE from env or first CLI arg

case "$MODE" in
  api)
    exec uvicorn src.api.server:app --host 0.0.0.0 --port 8000
    ;;
  notebook|lab)
    # Jupyter Lab / Notebook (choose one)
    exec jupyter lab --ip=0.0.0.0 --port 8888 --allow-root --no-browser
    # or: exec jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root --no-browser
    ;;
  *)
    echo "Unknown mode: '$MODE'. Use MODE=api or MODE=notebook." >&2
    exit 1
    ;;
esac
