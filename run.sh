#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

case "$(uname -s 2>/dev/null || echo Linux)" in
    MINGW*|MSYS*|CYGWIN*)
        HOST_KIND="win"
        if [ -x ".venv/bin/python" ] && [ ! -x ".venv/Scripts/python.exe" ]; then
            VENV_DIR=".venv-win"
        else
            VENV_DIR=".venv"
        fi
        ;;
    *)
        HOST_KIND="posix"
        if [ -x ".venv/Scripts/python.exe" ] && [ ! -x ".venv/bin/python" ]; then
            VENV_DIR=".venv-linux"
        else
            VENV_DIR=".venv"
        fi
        ;;
esac

_resolve_venv_paths() {
    if [ "$HOST_KIND" = "win" ]; then
        if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
            VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
            VENV_DRAW="$VENV_DIR/Scripts/draw.exe"
        elif [ -x "$VENV_DIR/bin/python" ]; then
            VENV_PYTHON="$VENV_DIR/bin/python"
            VENV_DRAW="$VENV_DIR/bin/draw"
        else
            VENV_PYTHON=""
            VENV_DRAW=""
        fi
    else
        if [ -x "$VENV_DIR/bin/python" ]; then
            VENV_PYTHON="$VENV_DIR/bin/python"
            VENV_DRAW="$VENV_DIR/bin/draw"
        else
            VENV_PYTHON=""
            VENV_DRAW=""
        fi
    fi
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
        UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --extra gui --extra hand
        _resolve_venv_paths
        if ! _venv_ready; then
            uv pip uninstall --python "$VENV_PYTHON" opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
            uv pip install --python "$VENV_PYTHON" --reinstall "opencv-contrib-python>=4.8,<5.1"
        fi
    else
        python3 -m venv "$VENV_DIR"
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
