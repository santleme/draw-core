"""lab/desk_capture.py: tap de quadros, dumps periodicos e o driver.

Mesmo padrao de test_eval_capture: fonte de mentira, detector
grampeado e modelo ONNX de fixture — exercita o driver de ponta a
ponta (Pipeline real + PageCleanWorker) sem camera e sem threads.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
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


class _Scripted:
    """Fonte com roteiro de leituras; esgotado o roteiro, falha sempre."""

    def __init__(self, script):
        self._script = list(script)
        self._index = 0
        self.reads = 0

    def isOpened(self):
        return True

    def read(self):
        self.reads += 1
        if self._index < len(self._script):
            item = self._script[self._index]
            self._index += 1
            return item
        return False, None

    def release(self):
        pass


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


def test_tap_recupera_falha_transitoria_em_fonte_viva():
    """hvcam/camera: um read falho entre dois bons nao encerra a corrida."""
    f = np.zeros((4, 4, 3), np.uint8)
    tap = dc._TappedCapture(
        _Scripted([(True, f), (False, None), (True, f), (True, f)]),
        live_retries=3,
    )
    for _ in range(3):
        ok, frame = tap.read()
        assert ok and frame is f  # False interno -> retentado -> quadro
    assert tap.frames_read == 3
    # Roteiro esgotado = fonte morta: esgota o orcamento e sobe erro,
    # nunca um fim de fluxo "ok".
    with pytest.raises(RuntimeError, match="fonte ao vivo"):
        tap.read()


def test_tap_arquivo_nao_retenta_eof():
    """Fonte de arquivo (live_retries=0): primeiro False ja e fim."""
    f = np.zeros((4, 4, 3), np.uint8)
    cap = _Scripted([(True, f)])
    tap = dc._TappedCapture(cap, live_retries=0)
    ok, frame = tap.read()
    assert ok and frame is f
    assert tap.read() == (False, None)
    assert cap.reads == 2


def test_real_fps_medio_so_tempo_da_fonte():
    """O denominador e so o tempo dentro de read() — pipeline nao conta."""
    clock = [100.0]

    class _SlowRead:
        def isOpened(self):
            return True

        def read(self):
            clock[0] += 0.05  # 20 fps de entrega
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    tap = dc._TappedCapture(_SlowRead(), now=lambda: clock[0])
    for _ in range(4):
        tap.read()
        clock[0] += 1.0  # trabalho do pipeline entre as leituras
    assert tap.real_fps() == pytest.approx(20.0)


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


def test_dump_falha_de_imwrite_vira_oserror(tmp_path, monkeypatch):
    """cv2.imwrite devolvendo False nao passa batido: sobe OSError."""
    import cv2

    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: False)
    dump, loop, worker, _lines = _dump_setup(tmp_path, np.ones((8, 8), np.uint8), None)
    with pytest.raises(OSError, match="scene_00000"):
        dump(loop, worker, 0)


def test_dump_retem_mascara_mais_recente(tmp_path):
    """last_mask acompanha cada quadro, inclusive transicao para None."""
    mask = np.ones((8, 8), np.uint8)
    dump, loop, worker, _ = _dump_setup(tmp_path, mask, None, every=99)
    dump(loop, worker, 1)  # quadro fora do ciclo: mascara ainda assim retida
    assert dump.last_mask is mask
    loop.clean_mask = None
    dump(loop, worker, 2)
    assert dump.last_mask is None


def test_run_recusa_diretorio_com_artefatos(tmp_path):
    """--out reutilizado nao mistura sessoes: recusa, nao apaga."""
    out = tmp_path / "desk"
    out.mkdir()
    (out / "scene_00000.png").write_bytes(b"")
    (out / "session.json").write_text("{}")
    with pytest.raises(ValueError, match="artefatos"):
        dc.run(
            _Frames(frames=1),
            out_dir=out,
            page_clean_model="x.onnx",
            hand_model="__none__",
            verify=None,
            log=lambda *_a: None,
        )
    assert (out / "scene_00000.png").exists()  # evidencia intocada


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
    assert session["canvas_final"] is False
    assert not (tmp_path / "desk" / "canvas_final.png").exists()


def test_run_interrompida_salva_canvas_mais_recente(tmp_path, monkeypatch):
    """Ctrl+C depois de tinta confirmada entre dumps: canvas_final sai."""
    onnx_path = onnx_fixtures.page_clean_model(
        tmp_path / "pc.onnx", patch=256, dynamic_batch=True
    )
    monkeypatch.setattr(ec, "_DEFAULT_SHEET_MODEL", tmp_path / "missing.onnx")
    from draw import paper as paper_module

    corners = np.array([[20, 10], [200, 10], [200, 150], [20, 150]], np.float32)
    monkeypatch.setattr(
        paper_module,
        "detect_sheet_with_confidence",
        lambda frame: (corners.copy(), "paper"),
    )

    class _InterruptAfter:
        def __init__(self, frames):
            self._frames = frames
            self._index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return 10.0

        def read(self):
            if self._index >= self._frames:
                raise KeyboardInterrupt
            self._index += 1
            frame = np.zeros((240, 320, 3), np.uint8)
            frame[10:150, 20:200] = 220
            frame[60:80, 60:160] = 20
            return True, frame

        def release(self):
            pass

    # 40 quadros a 10 fps com a ultima varredura em 30: a tinta
    # confirmada depois do ultimo dump so sobrevive via last_mask.
    result = dc.run(
        _InterruptAfter(40),
        out_dir=tmp_path / "desk",
        every=15,
        fps=10,
        page_clean_model=str(onnx_path),
        hand_model="__none__",
        verify=None,
        log=lambda *_a: None,
    )
    assert result is None
    canvas = cv2.imread(
        str(tmp_path / "desk" / "canvas_final.png"), cv2.IMREAD_GRAYSCALE
    )
    assert canvas is not None and (canvas == 255).any()
    session = json.loads(
        (tmp_path / "desk" / "session.json").read_text(encoding="utf-8")
    )
    assert session["interrupted"] is True and session["canvas_final"] is True


def test_main_every_invalido_sai_2():
    with pytest.raises(SystemExit) as err:
        dc.main(["--every", "0"])
    assert err.value.code == 2


def test_default_out_dir_sob_captures():
    out = dc._default_out_dir()
    assert out.parent == Path("captures") and out.name.startswith("desk-")
