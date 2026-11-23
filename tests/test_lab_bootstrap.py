"""Fumaça: mains de lab/ executáveis por caminho direto, fora da raiz.

Cada script de lab/ precisa funcionar via ``python lab/<nome>.py`` de
qualquer cwd — o bootstrap no topo insere raiz + src + draw-dral no
sys.path quando ``__package__`` está vazio. A regressão coberta aqui:
``falsify`` e ``eval_scenario`` inseriam só ``src`` e morriam com
``ModuleNotFoundError: No module named 'lab'`` fora da raiz.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_LAB_MAINS = [
    "bench_ink.py",
    "calibrate_desk.py",
    "corpus/__main__.py",
    "desk_capture.py",
    "diag_live.py",
    "eval_capture.py",
    "eval_scenario.py",
    "export_vectors.py",
    "falsify.py",
    "gen_synthetic.py",
    "parity_e2e.py",
    "quantize_page_clean.py",
    "replay_lab.py",
    "seg_scene_audit.py",
    "sim_source.py",
    "sim_writer.py",
]


@pytest.mark.parametrize("script", _LAB_MAINS)
def test_lab_main_help_from_foreign_cwd(tmp_path, script):
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "lab" / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
