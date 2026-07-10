#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x "./roberta-env/bin/python" ]]; then
  echo "Error: roberta-env is missing. Create it first with: python -m venv roberta-env"
  exit 1
fi

exec ./roberta-env/bin/python -m streamlit run app.py "$@"
