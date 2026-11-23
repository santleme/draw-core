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

_gui_ready() {
    _resolve_venv_paths
    [ -n "$VENV_DRAW" ] && [ -x "$VENV_DRAW" ] && "$VENV_PYTHON" -c '
import cv2, onnxruntime, draw
from draw.app import _gui_backend_compiled
assert _gui_backend_compiled(), "headless opencv"
assert hasattr(cv2, "ximgproc"), "opencv sem modulos contrib"
' >/dev/null 2>&1
}

_drop_extra_cv2_providers() {
    # A second registered cv2 provider is a time bomb: the next exact
    # `uv sync` uninstalls one dist and deletes the shared cv2 files.
    # When the GUI works the files are already the contrib ones — the
    # extra provider only needs to lose what it owns alone: its RECORD
    # entries minus every file a retained distribution also lists, and
    # only inside this venv's own purelib/platlib — registrations
    # inherited from system or user site-packages are never touched.
    # No wheel, no network, and nothing shared is ever removed, so a
    # working env stays launchable even offline on a cold cache.
    _resolve_venv_paths
    [ -n "$VENV_PYTHON" ] && "$VENV_PYTHON" -c '
import shutil, sysconfig
from pathlib import Path
from importlib.metadata import distributions

extra = {
    "opencv-contrib-python-headless",
    "opencv-python",
    "opencv-python-headless",
}
roots = {
    Path(sysconfig.get_path(key)).resolve()
    for key in ("purelib", "platlib")
}
dists = [
    d for d in distributions()
    if (p := getattr(d, "_path", None)) is not None
    and Path(p).resolve().parent in roots
]
names = {d.metadata["Name"].lower() for d in dists}
if "opencv-contrib-python" in names:
    keep = {
        Path(d.locate_file(f)).resolve()
        for d in dists
        if d.metadata["Name"].lower() not in extra
        for f in (d.files or ())
    }
    for d in dists:
        if d.metadata["Name"].lower() not in extra:
            continue
        root = Path(d._path).resolve().parent
        parents = set()
        for f in d.files or ():
            target = Path(d.locate_file(f)).resolve()
            if target in keep or root not in target.parents:
                continue
            if target.is_file() or target.is_symlink():
                target.unlink(missing_ok=True)
                parents.update(target.parents)
        for folder in sorted(
            (p for p in parents if root in p.parents),
            key=lambda p: -len(p.parts),
        ):
            try:
                folder.rmdir()
            except OSError:
                pass
        shutil.rmtree(d._path, ignore_errors=True)
' >/dev/null 2>&1 || true
}

_hand_ready() {
    "$VENV_PYTHON" -c 'import mediapipe' >/dev/null 2>&1
}

_hand_retry_due() {
    "$VENV_PYTHON" -c '
import sys, time
from pathlib import Path

try:
    age = time.time() - (Path(sys.prefix) / ".no_hand").stat().st_mtime
except FileNotFoundError:
    sys.exit(0)
sys.exit(0 <= age < 24 * 60 * 60)
'
}

if command -v uv >/dev/null 2>&1; then
    USE_UV=1
else
    USE_UV=0
fi

_install_gui() {
    if [ "$USE_UV" = 1 ]; then
        UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --inexact --extra gui \
            --no-install-package opencv-contrib-python-headless "$@"
    else
        local extras=gui
        if [ "$#" -gt 0 ]; then
            extras=gui,hand
        fi
        # pip nao le [tool.uv.sources]: sem o membro instalado antes, a
        # dependencia "draw-dral" do draw-core nao resolve fora do
        # workspace. Instalado primeiro, ela ja consta como satisfeita.
        "$VENV_PYTHON" -m pip install -e "./packages/draw-dral" || return "$?"
        "$VENV_PYTHON" -m pip install -e ".[$extras]"
    fi
}

_prepare_pip() {
    if [ "$USE_UV" = 0 ]; then
        python3 -m venv "$VENV_DIR" || return "$?"
        _resolve_venv_paths
        "$VENV_PYTHON" -m pip install --upgrade pip
    fi
}

_repair_opencv() {
    # Force the locked contrib wheel back on top without uninstalling:
    # --reinstall-package rewrites its files in place, so a working GUI
    # survives a failed attempt, and the extra cv2 dist registrations
    # are dropped metadata-only at the end. --locked --offline resolves
    # from uv.lock alone and reinstalls straight from the wheel cache;
    # go online only for a missing wheel.
    if [ "$USE_UV" = 1 ]; then
        UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --locked --offline \
            --inexact --extra gui --reinstall-package opencv-contrib-python \
            --no-install-package opencv-contrib-python-headless ||
            UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --inexact \
                --extra gui --reinstall-package opencv-contrib-python \
                --no-install-package opencv-contrib-python-headless ||
            return "$?"
    else
        "$VENV_PYTHON" -m pip install --no-deps --force-reinstall \
            "opencv-contrib-python>=4.8,<5.1" || return "$?"
    fi
    _drop_extra_cv2_providers
}

if _gui_ready; then
    _drop_extra_cv2_providers
    if ! _hand_ready; then
        BASE_VENV_DIR="$VENV_DIR"
        VENV_DIR="$BASE_VENV_DIR-hand"
        if _gui_ready && _hand_ready; then
            _drop_extra_cv2_providers
            exec "$VENV_DRAW" "$@"
        fi
        VENV_DIR="$BASE_VENV_DIR"
        _resolve_venv_paths
        if _hand_retry_due; then
            # Keep the working GUI environment available if optional setup fails.
            if (
                VENV_DIR="$BASE_VENV_DIR-hand"
                _prepare_pip &&
                    _install_gui --extra hand &&
                    { _gui_ready || { _repair_opencv && _gui_ready; }; } &&
                    _hand_ready
            ); then
                VENV_DIR="$BASE_VENV_DIR-hand"
            else
                touch "$BASE_VENV_DIR/.no_hand" || true
            fi
        fi
    fi
else
    if [ -d "$VENV_DIR" ]; then
        echo "Reparando o OpenCV do ambiente do Draw..."
    else
        echo "Preparando ambiente do Draw (primeira execucao)..."
    fi
    _prepare_pip
    if _install_gui --extra hand; then
        rm -f "$VENV_DIR/.no_hand"
    else
        _install_gui || exit "$?"
        touch "$VENV_DIR/.no_hand"
    fi
    if ! _gui_ready; then
        _repair_opencv
    fi
fi

_resolve_venv_paths
_drop_extra_cv2_providers
exec "$VENV_DRAW" "$@"
