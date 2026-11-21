#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

_resolve_venv_paths() {
    case "$(uname -s 2>/dev/null || echo Linux)" in
        MINGW*|MSYS*|CYGWIN*)
            if [ -x ".venv/Scripts/python.exe" ]; then
                VENV_PYTHON=".venv/Scripts/python.exe"
                VENV_DRAW=".venv/Scripts/draw.exe"
            elif [ -x ".venv/bin/python" ]; then
                VENV_PYTHON=".venv/bin/python"
                VENV_DRAW=".venv/bin/draw"
            else
                VENV_PYTHON=""
                VENV_DRAW=""
            fi
            ;;
        *)
            if [ -x ".venv/bin/python" ]; then
                VENV_PYTHON=".venv/bin/python"
                VENV_DRAW=".venv/bin/draw"
            else
                VENV_PYTHON=""
                VENV_DRAW=""
            fi
            ;;
    esac
}

_venv_ready() {
    _resolve_venv_paths
    [ -n "$VENV_DRAW" ] && [ -x "$VENV_DRAW" ] && "$VENV_PYTHON" -c '
import cv2, onnxruntime, draw
from draw.app import _gui_backend_compiled
assert _gui_backend_compiled(), "headless opencv"
' >/dev/null 2>&1
}

if ! _venv_ready; then
    echo "Preparando ambiente do Draw (primeira execucao)..."
    if command -v uv >/dev/null 2>&1; then
        uv sync --extra gui --extra hand
        _resolve_venv_paths
        if ! _venv_ready; then
            uv pip uninstall --python "$VENV_PYTHON" opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
            uv pip install --python "$VENV_PYTHON" --reinstall "opencv-contrib-python>=4.8,<5.1"
        fi
    else
        python3 -m venv .venv
        _resolve_venv_paths
        "$VENV_PYTHON" -m pip install --upgrade pip
        "$VENV_PYTHON" -m pip install -e ".[gui,hand]"
        if ! _venv_ready; then
            "$VENV_PYTHON" -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
            "$VENV_PYTHON" -m pip install --force-reinstall "opencv-contrib-python>=4.8,<5.1"
        fi
    fi
fi

_resolve_venv_paths
exec "$VENV_DRAW" "$@"
