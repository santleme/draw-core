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
    # Roteiro esgotado = fonte morta: esgota o orcamento e sobe erro
    # distinto — nunca um fim de fluxo "ok".
    with pytest.raises(dc.LiveSourceExhausted, match="fonte ao vivo"):
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


def test_real_fps_medio_cadencia_entre_chegadas():
    """fps = quadros/espacamento entre retornos — a cadencia observada."""
    clock = [100.0]

    class _BufferedRead:
        """Read instantaneo (backend bufferizado): tempo de read ~0."""

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    tap = dc._TappedCapture(_BufferedRead(), now=lambda: clock[0])
    for _ in range(4):
        tap.read()
        clock[0] += 0.5  # pipeline lenta entre as chegadas
    # Retornos em 100.0/100.5/101.0/101.5: 3 gaps em 1.5 s = 2 fps —
    # a cadencia efetiva, nao um "read instantaneo" inflado.
    assert tap.real_fps() == pytest.approx(2.0)


def test_src_fps_usa_carimbos_do_emissor():
    """hvcam/ajudante: a cadencia vem de last_capture_ts_ms, nao do relogio."""
    clock = [100.0]

    class _Stamped:
        def __init__(self):
            self.last_capture_ts_ms = None
            self._ts = 0

        def isOpened(self):
            return True

        def read(self):
            self._ts += 33  # emissor a ~30 fps
            self.last_capture_ts_ms = self._ts
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    tap = dc._TappedCapture(_Stamped(), now=lambda: clock[0])
    for _ in range(4):
        tap.read()
        clock[0] += 1.0  # pipeline lenta nao afeta a cadencia do emissor
    assert tap.src_fps() == pytest.approx(30.3, rel=0.01)


class _StampedDropped:
    """Fonte carimbada: ~30 fps com saltos de seq contabilizados."""

    def __init__(self, drop_at=None):
        self.last_capture_ts_ms = None
        self.source_dropped = 0
        self._ts = 0
        self._n = 0
        self._drop_at = drop_at

    def isOpened(self):
        return True

    def read(self):
        self._n += 1
        if self._n == self._drop_at:
            self.source_dropped += 2  # dois frames se perderam no caminho
        self._ts += 33
        self.last_capture_ts_ms = self._ts
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        pass


def test_src_fps_conta_quadros_descartados():
    """Salto de seq do emissor nao derruba a taxa: emitidos = lidos + drop."""
    tap = dc._TappedCapture(_StampedDropped(drop_at=3))
    for _ in range(4):
        tap.read()
    # Emitidos 6 em 99 ms ~= 50.5 fps — nao os 30.3 dos entregues.
    assert tap.src_fps() == pytest.approx(50.5, rel=0.01)


def test_src_fps_ignora_drops_anteriores_ao_inicio():
    """Drops antes do 1o quadro utilizavel ficam fora da janela da taxa.

    A falha e exercida de verdade: read() falha duas vezes (JPEG
    indecodificavel, como HelperCapture faz) acumulando 4 drops —
    que entram no baseline. Um drop DENTRO da janela continua
    contando.
    """

    class _FlakyStart(_StampedDropped):
        def read(self):
            if self._n < 2:
                self._n += 1
                self.source_dropped += 2  # dois registros mortos cada
                return False, None
            return super().read()

    src = _FlakyStart(drop_at=5)  # drop dentro da janela, no 3o quadro
    tap = dc._TappedCapture(src)
    assert tap.read() == (False, None)
    assert tap.read() == (False, None)
    for _ in range(4):
        assert tap.read()[0]
    # Janela: 4 entregues + 2 drops = 6 emitidos em 99 ms; os 4 drops
    # de startup ficam fora — senao a taxa inflava para ~90 fps.
    assert tap.src_fps() == pytest.approx(50.5, rel=0.01)


def test_src_fps_none_sem_carimbo():
    tap = dc._TappedCapture(_Frames(frames=2))
    tap.read()
    assert tap.src_fps() is None


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


def test_dump_retem_loop_vivo_para_finalizacao(tmp_path):
    """dump.loop expoe a mascara VIVA: reset depois do dump e honrado."""
    mask = np.ones((8, 8), np.uint8)
    dump, loop, worker, _ = _dump_setup(tmp_path, mask, None, every=99)
    dump(loop, worker, 1)
    assert dump.loop is loop
    loop.clean_mask = None  # pipeline limpou depois do dump
    assert dump.loop.clean_mask is None


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


def test_artefatos_ignoram_imagens_de_outras_ferramentas(tmp_path):
    """Checkpoints do falsify (scene_<fase>_<n>.png) nao sao dumps do desk."""
    out = tmp_path / "desk"
    out.mkdir()
    (out / "scene_settled_42.png").write_bytes(b"")
    (out / "canvas_phase1.png").write_bytes(b"")
    assert not dc._desk_artifacts(out)
    (out / "scene_00042.png").write_bytes(b"")
    assert dc._desk_artifacts(out)


def test_artefatos_detectam_symlink_pendurado(tmp_path):
    """Link pendurado tem exists()==False — a guarda confere o NOME."""
    out = tmp_path / "desk"
    out.mkdir()
    (out / "canvas_final.png").symlink_to(tmp_path / "fora.png")
    assert dc._desk_artifacts(out)


def test_live_retries_file_url_e_arquivo(tmp_path):
    """file:// e so outra grafia de caminho: EOF real, sem retry."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    assert dc._live_retries(str(video)) == 0
    assert dc._live_retries(f"file://{video}") == 0
    assert dc._live_retries("tcp://10.0.0.1") == dc._LIVE_READ_RETRIES
    assert dc._live_retries("hvcam://") == dc._LIVE_READ_RETRIES
    assert dc._live_retries(0) == dc._LIVE_READ_RETRIES


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


class _InterruptAfter:
    """Fonte que interrompe no quadro N; traco B entra apos change_at."""

    def __init__(self, frames, change_at=None):
        self._frames = frames
        self._change_at = change_at
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
        frame[60:80, 60:160] = 20  # traco A
        if self._change_at is not None and self._index > self._change_at:
            frame[100:120, 100:180] = 20  # traco B, depois do ultimo dump
        return True, frame

    def release(self):
        pass


def _stub_detector(monkeypatch, tmp_path):
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
    return onnx_path


def test_run_interrompida_salva_canvas_mais_recente(tmp_path, monkeypatch):
    """Ctrl+C com tinta confirmada depois do ultimo dump: entra no final."""
    onnx_path = _stub_detector(monkeypatch, tmp_path)
    result = dc.run(
        _InterruptAfter(45, change_at=31),
        out_dir=tmp_path / "desk",
        every=15,
        fps=10,
        page_clean_model=str(onnx_path),
        hand_model="__none__",
        verify=None,
        log=lambda *_a: None,
    )
    assert result is None
    out = tmp_path / "desk"
    canvas = cv2.imread(str(out / "canvas_final.png"), cv2.IMREAD_GRAYSCALE)
    assert canvas is not None and (canvas == 255).any()
    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert session["interrupted"] is True and session["canvas_final"] is True


def test_run_fonte_morta_grava_evidencia_e_sobe_erro(tmp_path, monkeypatch):
    """Retries esgotados: session.json + canvas_final saem antes do erro."""
    onnx_path = _stub_detector(monkeypatch, tmp_path)
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[10:150, 20:200] = 220
    frame[60:80, 60:160] = 20
    tap_script = [(True, frame)] * 45  # depois: (False, None) para sempre
    out = tmp_path / "desk"
    with pytest.raises(dc.LiveSourceExhausted):
        dc.run(
            _Scripted(tap_script),
            out_dir=out,
            every=15,
            fps=10,
            live_retries=3,
            page_clean_model=str(onnx_path),
            hand_model="__none__",
            verify=None,
            log=lambda *_a: None,
        )
    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert session["status"] == "source_failed"
    assert session["interrupted"] is False and "fonte ao vivo" in session["error"]
    assert session["frames"] == 45
    canvas = cv2.imread(str(out / "canvas_final.png"), cv2.IMREAD_GRAYSCALE)
    assert canvas is not None and (canvas == 255).any()


def test_finalizacao_usa_mascara_viva_do_loop(tmp_path, monkeypatch):
    """Ctrl+C depois de publicacao nova ou reset: o loop vivo decide."""

    def fake_pipeline(mask_after):
        def _run(_tap, *, on_frame, **_kw):
            loop = _FakeLoop(np.zeros((8, 8), np.uint8))
            worker = _FakeWorker(None)
            for i in range(3):
                _tap.read()
                on_frame(loop, worker, i)
            loop.clean_mask = mask_after  # publica/reseta apos o ultimo on_frame
            raise KeyboardInterrupt

        return _run

    mask_b = np.zeros((8, 8), np.uint8)
    mask_b[5:] = 1
    monkeypatch.setattr(dc, "run_pipeline", fake_pipeline(mask_b))
    out = tmp_path / "pub"
    assert dc.run(_Frames(3), out_dir=out, verify=None, log=lambda *_a: None) is None
    canvas = cv2.imread(str(out / "canvas_final.png"), cv2.IMREAD_GRAYSCALE)
    assert canvas is not None
    assert np.count_nonzero(canvas == 255) == 24  # mask_b, nao a vazia

    monkeypatch.setattr(dc, "run_pipeline", fake_pipeline(None))
    out2 = tmp_path / "reset"
    assert dc.run(_Frames(3), out_dir=out2, verify=None, log=lambda *_a: None) is None
    assert not (out2 / "canvas_final.png").exists()  # reset honrado
    session = json.loads((out2 / "session.json").read_text(encoding="utf-8"))
    assert session["canvas_final"] is False


class _SheetMove:
    """Folha some por `absent` quadros e volta em outro lugar (reset)."""

    def __init__(self, frames, absent):
        self._frames = frames
        self._absent = absent
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
        if not (self._absent[0] <= self._index <= self._absent[1]):
            if self._index > self._absent[1]:
                frame[60:200, 110:300] = 220  # folha B, deslocada
            else:
                frame[10:150, 20:200] = 220  # folha A
                frame[60:80, 60:160] = 20
        return True, frame

    def release(self):
        pass


def _sheet_bbox_detector(frame):
    """Detecta a folha pelo bbox dos pixels claros do proprio frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 150)
    if len(ys) < 500:
        return None, None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    return corners, "paper"


def test_run_interrompida_reflete_mascara_viva(tmp_path, monkeypatch):
    """Canvas final sai da mascara VIVA do loop, nao de retrato do dump.

    A mascara satura na pagina (o fixture confirma quase tudo), entao
    pixel-a-pixel nao separa "tinta tardia" — a prova de frescor e o
    teste com loop stubado abaixo. Aqui o e2e cobre o caminho real:
    drop + reabertura com identidade preservada restauram a mascara,
    e a finalizacao grava exatamente a mascara viva do loop.
    """
    onnx_path = onnx_fixtures.page_clean_model(
        tmp_path / "pc.onnx", patch=256, dynamic_batch=True
    )
    monkeypatch.setattr(ec, "_DEFAULT_SHEET_MODEL", tmp_path / "missing.onnx")
    from draw import paper as paper_module

    monkeypatch.setattr(
        paper_module, "detect_sheet_with_confidence", _sheet_bbox_detector
    )
    out = tmp_path / "desk"
    result = dc.run(
        _SheetMove(58, absent=(31, 50)),
        out_dir=out,
        every=15,
        fps=10,
        page_clean_model=str(onnx_path),
        hand_model="__none__",
        verify=None,
        log=lambda *_a: None,
    )
    assert result is None
    # Reabertura com identidade preservada: o final sai da mascara
    # viva do loop (o estado real do pipeline na hora do Ctrl+C,
    # seja ela restaurada ou zerada pela geracao nova).
    canvas = cv2.imread(str(out / "canvas_final.png"), cv2.IMREAD_GRAYSCALE)
    assert canvas is not None
    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert session["interrupted"] is True and session["canvas_final"] is True


def test_main_every_invalido_sai_2():
    with pytest.raises(SystemExit) as err:
        dc.main(["--every", "0"])
    assert err.value.code == 2


def test_default_out_dir_sob_captures():
    out = dc._default_out_dir()
    assert out.parent == Path("captures") and out.name.startswith("desk-")
