#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CANDIDATES=("$PYTHON_BIN")
else
    PYTHON_CANDIDATES=(python3.13 python3.12 python3.11)
fi
PYTHON_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: Python 3.11-3.13 is unavailable." >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating $VENV_DIR with $($PYTHON_BIN --version)"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PIP=("$VENV_DIR/bin/python" -m pip --disable-pip-version-check)

echo "Installing NumPy and pandas ..."
"${PIP[@]}" install --retries 10 --timeout 120 \
    "numpy>=2.0,<3.0" "pandas>=2.2,<3.0"

echo "Installing CPU-only PyTorch ..."
"${PIP[@]}" install --retries 10 --timeout 120 \
    "torch>=2.8,<3.0" --index-url https://download.pytorch.org/whl/cpu

echo "Installing FastAPI and the local package ..."
"${PIP[@]}" install --retries 10 --timeout 120 "fastapi[standard]>=0.116,<0.117"
"${PIP[@]}" install --no-deps --editable .

"$VENV_DIR/bin/python" - <<'PY'
import fastapi
import numpy
import pandas
import torch

print(f"NumPy {numpy.__version__}")
print(f"pandas {pandas.__version__}")
print(f"PyTorch {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"FastAPI {fastapi.__version__}")
PY

echo "Setup complete. Next: .venv/bin/resolveai download-data"
