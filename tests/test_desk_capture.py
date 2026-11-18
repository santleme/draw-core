"""lab/desk_capture.py: tap de quadros, dumps periodicos e o driver.

Mesmo padrao de test_eval_capture: fonte de mentira, detector
grampeado e modelo ONNX de fixture — exercita o driver de ponta a
ponta (Pipeline real + PageCleanWorker) sem camera e sem threads.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab import desk_capture as dc  # noqa: E402
from lab import eval_capture as ec  # noqa: E402

onnx_fixtures = pytest.importorskip("onnx_fixtures")  # noqa: E402


class _Frames:
    """VideoCapture de mentira: quadros numerados e get() de fps."""

    def __init__(self, frames=4, fps=10.0):
        self._frames = frames
        self._index = 0
        self._fps = fps
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self._fps

    def read(self):
        if self._index >= self._frames:
            return False, None
        self._index += 1
        return True, np.full((8, 8, 3), self._index, np.uint8)

    def release(self):
        self.released = True


class _NoGet(_Frames):
    """Fonte estilo hvcam/ajudante: sem get() de VideoCapture."""

    get = None


def test_tap_memoriza_ultimo_quadro_e_limita():
    tap = dc._TappedCapture(_Frames(frames=3), max_frames=2)
    ok, first = tap.read()
    assert ok and tap.last_frame is first
    ok, second = tap.read()
    assert ok and tap.last_frame is second
    assert tap.read() == (False, None)
    assert tap.frames_read == 2
    assert tap.get(5) == 10.0
    tap.release()


def test_tap_sem_get_devolve_zero():
    """Fonte sem get() devolve 0.0: run_pipeline pede --fps, nao quebra."""
    tap = dc._TappedCapture(_NoGet(frames=1))
    assert tap.get(5) == 0.0
    ok, _frame = tap.read()
    assert ok and tap.frames_read == 1


class _FakeLoop:
    def __init__(self, mask):
        self.clean_mask = mask
        self.detection_worker = SimpleNamespace(sequence=7)
        self.tracker = SimpleNamespace(_frames_since_seen=3)
        self.last_tracker_event = "drop"


class _FakeWorker:
    def __init__(self, corners):
        self.anchor_corners = corners
        self.canvas_epoch = 2

    def stats(self):
        return {"passes_total": 11}


def _dump_setup(tmp_path, mask, corners, every=2):
    tap = dc._TappedCapture(_Frames(frames=1))
    tap.read()
    loop = _FakeLoop(mask)
    worker = _FakeWorker(corners)
    lines = []
    dump = dc._DumpEvery(tmp_path, tap, every=every, log=lines.append)
    return dump, loop, worker, lines


def test_dump_escreve_scene_e_canvas(tmp_path):
    mask = np.zeros((8, 8), np.uint8)
    mask[2:5, 3:6] = 1
    corners = np.array([[1, 1], [6, 1], [6, 6], [1, 6]], np.float32)
    dump, loop, worker, lines = _dump_setup(tmp_path, mask, corners)

    dump(loop, worker, 0)
    import cv2

    scene = cv2.imread(str(tmp_path / "scene_00000.png"))
    canvas = cv2.imread(str(tmp_path / "canvas_00000.png"), cv2.IMREAD_GRAYSCALE)
    assert scene is not None and canvas is not None
    assert canvas[3, 4] == 255 and canvas[0, 0] == 0
    assert np.count_nonzero(canvas == 255) == 9
    assert lines and "anchor=sim" in lines[-1] and "conf=9" in lines[-1]
    assert "passes=11" in lines[-1] and "det=7" in lines[-1]
    assert "miss=3" in lines[-1] and "evt=drop" in lines[-1]

    dump(loop, worker, 1)  # 1 % every != 0: nada novo
    assert len(list(tmp_path.glob("scene_*.png"))) == 1


def test_dump_sem_mascara_nao_escreve_canvas(tmp_path):
    corners = np.array([[1, 1], [6, 1], [6, 6], [1, 6]], np.float32)
    dump, loop, worker, lines = _dump_setup(tmp_path, None, corners)
    loop2 = _FakeLoop(None)
    dump(loop2, worker, 0)
    assert (tmp_path / "scene_00000.png").exists()
    assert not (tmp_path / "canvas_00000.png").exists()
    assert "conf=0" in lines[-1]


def test_dump_sem_ancora_marca_nao(tmp_path):
    dump, loop, worker, lines = _dump_setup(tmp_path, None, None)
    dump(loop, worker, 0)
    assert "anchor=NAO" in lines[-1]


def test_dump_every_invalido(tmp_path):
    tap = dc._TappedCapture(_Frames(1))
    with pytest.raises(ValueError, match="every"):
        dc._DumpEvery(tmp_path, tap, every=0)


class _SheetCapture(_Frames):
    """Folha clara com um traco escuro, como _FakeCapture do eval."""

    def read(self):
        if self._index >= self._frames:
            return False, None
        self._index += 1
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[10:150, 20:200] = 220
        frame[60:80, 60:160] = 20
        return True, frame


def test_run_end_to_end_grava_dumps_e_sessao(tmp_path, monkeypatch):
    onnx_path = onnx_fixtures.page_clean_model(
        tmp_path / "pc.onnx", patch=256, dynamic_batch=True
    )
    # Sem o binario default instalado, o detector classico grampeado
    # e o caminho — mesmo arranjo de test_run_pipeline_dirige_pipeline_real.
    monkeypatch.setattr(ec, "_DEFAULT_SHEET_MODEL", tmp_path / "missing.onnx")
    from draw import paper as paper_module

    corners = np.array([[20, 10], [200, 10], [200, 150], [20, 150]], np.float32)
    monkeypatch.setattr(
        paper_module,
        "detect_sheet_with_confidence",
        lambda frame: (corners.copy(), "paper"),
    )
    out = tmp_path / "desk"
    lines = []
    result = dc.run(
        _SheetCapture(frames=40),
        out_dir=out,
        every=10,
        fps=10,
        page_clean_model=str(onnx_path),
        hand_model="__none__",
        verify=None,
        log=lines.append,
    )
    assert result is not None and result["frames"] == 40
    scenes = sorted(out.glob("scene_*.png"))
    assert [p.name for p in scenes] == [
        "scene_00000.png",
        "scene_00010.png",
        "scene_00020.png",
        "scene_00030.png",
    ]
    # O modelo didatico confirma o traco: canvas periodico e final.
    assert sorted(out.glob("canvas_*.png")) and (out / "canvas_final.png").exists()
    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert session["frames"] == 40 and session["status"] == "ok"
    assert session["interrupted"] is False
    assert session["registration_count"] == len(result["registrations"])
    assert "mask" not in session and "anchors" not in session
    assert any("anchor=sim" in line and "conf=" in line for line in lines)


def test_run_sem_fps_em_fonte_sem_get(tmp_path):
    """Fonte estilo hvcam sem --fps: erro pede a flag, nao AttributeError."""
    with pytest.raises(ValueError, match="--fps"):
        dc.run(
            _NoGet(frames=5),
            out_dir=tmp_path / "desk",
            page_clean_model="missing.onnx",
            hand_model="__none__",
            verify=None,
            log=lambda *_a: None,
        )


def test_run_interrompida_grava_sessao_parcial(tmp_path, monkeypatch):
    """Ctrl+C no meio: session.json registra o quanto foi lido."""
    onnx_path = onnx_fixtures.page_clean_model(
        tmp_path / "pc.onnx", patch=256, dynamic_batch=True
    )
    monkeypatch.setattr(ec, "_DEFAULT_SHEET_MODEL", tmp_path / "missing.onnx")

    class _Boom:
        def isOpened(self):
            return True

        def get(self, prop):
            return 10.0

        def read(self):
            raise KeyboardInterrupt

        def release(self):
            pass

    result = dc.run(
        _Boom(),
        out_dir=tmp_path / "desk",
        page_clean_model=str(onnx_path),
        hand_model="__none__",
        verify=None,
        log=lambda *_a: None,
    )
    assert result is None
    session = json.loads(
        (tmp_path / "desk" / "session.json").read_text(encoding="utf-8")
    )
    assert session["interrupted"] is True and session["status"] == "interrupted"


def test_main_every_invalido_sai_2():
    with pytest.raises(SystemExit) as err:
        dc.main(["--every", "0"])
    assert err.value.code == 2


def test_default_out_dir_sob_captures():
    out = dc._default_out_dir()
    assert out.parent == Path("captures") and out.name.startswith("desk-")
