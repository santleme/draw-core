#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

case "$(uname -s 2>/dev/null || echo Linux)" in
    MINGW*|MSYS*|CYGWIN*)
        VENV_PYTHON=".venv/Scripts/python.exe"
        VENV_PIP=".venv/Scripts/pip.exe"
        VENV_DRAW=".venv/Scripts/draw.exe"
        ;;
    *)
        VENV_PYTHON=".venv/bin/python"
        VENV_PIP=".venv/bin/pip"
        VENV_DRAW=".venv/bin/draw"
        ;;
esac

_venv_ready() {
    [ -x "$VENV_DRAW" ] && "$VENV_PYTHON" -c '
import cv2, onnxruntime, mediapipe, draw
from draw.app import _gui_backend_compiled
assert _gui_backend_compiled(), "headless opencv"
' >/dev/null 2>&1
}

if ! _venv_ready; then
    echo "Preparando ambiente do Draw (primeira execucao)..."
    if command -v uv >/dev/null 2>&1; then
        uv sync --extra gui --extra hand
        if ! _venv_ready; then
            uv pip uninstall --python "$VENV_PYTHON" opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
            uv pip install --python "$VENV_PYTHON" --reinstall "opencv-contrib-python>=4.8,<5.1"
        fi
    else
        python3 -m venv .venv
        "$VENV_PIP" install --upgrade pip
        "$VENV_PIP" install -e ".[gui,hand]"
        if ! _venv_ready; then
            "$VENV_PIP" uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
            "$VENV_PIP" install --force-reinstall "opencv-contrib-python>=4.8,<5.1"
        fi
    fi
fi

exec "$VENV_DRAW" "$@"
