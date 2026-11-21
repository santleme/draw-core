#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

_venv_ready() {
    [ -x ".venv/bin/draw" ] && .venv/bin/python -c '
import cv2, onnxruntime, draw
from draw.app import _gui_backend_compiled
assert _gui_backend_compiled(), "headless opencv"
' >/dev/null 2>&1
}

if ! _venv_ready; then
    echo "Preparando ambiente do Draw (primeira execucao)..."
    if command -v uv >/dev/null 2>&1; then
        uv sync --extra gui --extra hand
        if ! _venv_ready; then
            uv pip uninstall opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
            uv pip install --reinstall "opencv-contrib-python>=4.8,<5.1"
        fi
    else
        python3 -m venv .venv
        .venv/bin/pip install --upgrade pip
        .venv/bin/pip install -e ".[gui,hand]"
        if ! _venv_ready; then
            .venv/bin/pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
            .venv/bin/pip install --force-reinstall "opencv-contrib-python>=4.8,<5.1"
        fi
    fi
fi

exec .venv/bin/draw "$@"
