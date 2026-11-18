"""Interface de producao: deteccao combinada, --check e CLI (draw/app.py).

Cobre o que o app acrescenta sobre os modulos: a decisao combinada
neural+classico com veto de presenca e extras de mascara, a checagem
de modelos do --check e a validacao de argumentos do main().
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from draw import app as app_module  # noqa: E402
from draw import config  # noqa: E402
from draw import paper as paper_module  # noqa: E402
from draw.infra import manifest as manifest_module  # noqa: E402
from draw.perception import PRESENCE_VETO_LOGIT  # noqa: E402


def _frame(h=120, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _corners():
    return np.array([[10, 10], [150, 10], [150, 110], [10, 110]], dtype=np.float32)


class _FakeNeural:
    """Detector neural de mentira: saida e atributos programados."""

    def __init__(self, corners=None, verdict=None, logit=None, mask=None):
        self._out = (corners, verdict, mask)
        self.last_presence_logit = logit
        self.last_coverage_mask = None
        self.last_hand_mask = None
        self.calls = []

    def __call__(self, frame, *, return_mask=True):
        self.calls.append(return_mask)
        return self._out


@pytest.fixture(autouse=True)
def _restore_format():
    yield
    config.apply_format("a4")


@pytest.fixture
def classic(monkeypatch):
    """Classico programavel: devolve (corners, verdict) por chamada."""
    state = {"result": (None, None), "calls": 0}

    def fake(frame):
        state["calls"] += 1
        return state["result"]

    monkeypatch.setattr(paper_module, "detect_sheet_with_confidence", fake)
    return state


def test_neural_fecha_quad_e_publica_mascaras(classic):
    frame = _frame()
    neural = _FakeNeural(
        _corners(), "paper", logit=3.0, mask=np.ones((120, 160), np.uint8)
    )
    neural.last_coverage_mask = np.full((30, 40), 255, np.uint8)
    neural.last_hand_mask = np.zeros((30, 40), np.uint8)
    neural.last_hand_mask[10:12, 10:12] = 255
    corners, verdict, trace, extras = app_module._detect_sheet_combined_traced(
        frame, neural
    )
    assert corners is not None and verdict == "paper"
    assert trace["source"] == "neural"
    assert classic["calls"] == 0
    # Mascaras sobem retificadas na grade do quadro.
    assert extras["coverage_mask"].shape == frame.shape[:2]
    assert extras["hand_mask"].shape == frame.shape[:2]
    assert neural.calls == [True]


def test_cobertura_quase_vazia_nao_sobe(classic):
    neural = _FakeNeural(
        _corners(), "paper", logit=3.0, mask=np.ones((120, 160), np.uint8)
    )
    coverage = np.zeros((120, 160), np.uint8)
    coverage[0, 0] = 255  # 1 px: abaixo do piso de 5%
    neural.last_coverage_mask = coverage
    _, _, _, extras = app_module._detect_sheet_combined_traced(_frame(), neural)
    assert "coverage_mask" not in extras


def test_logit_forte_veta_o_classico(classic):
    classic["result"] = (_corners(), "paper")
    neural = _FakeNeural(None, None, logit=PRESENCE_VETO_LOGIT, mask=None)
    corners, verdict, trace, _ = app_module._detect_sheet_combined_traced(
        _frame(), neural
    )
    assert (corners, verdict) == (None, None)
    assert trace["source"] == "veto"
    assert classic["calls"] == 0


def test_logit_fraco_deixa_o_classico_falar(classic):
    classic["result"] = (_corners(), "blank_maybe")
    neural = _FakeNeural(None, None, logit=-1.0, mask=None)
    corners, verdict, trace, _ = app_module._detect_sheet_combined_traced(
        _frame(), neural
    )
    assert corners is not None and verdict == "blank_maybe"
    assert trace["source"] == "classic"
    assert classic["calls"] == 1


def test_presenca_sem_posicao_devolve_paper_sem_cantos(classic):
    neural = _FakeNeural(None, None, logit=2.0, mask=None)
    corners, verdict, trace, _ = app_module._detect_sheet_combined_traced(
        _frame(), neural
    )
    assert corners is None and verdict == "paper"
    assert trace["source"] == "presence_only"


def test_sem_neural_e_classico_puro_e_extras_limpos(classic):
    classic["result"] = (_corners(), "paper")
    corners, verdict, trace, extras = app_module._detect_sheet_combined_traced(
        _frame(), None
    )
    assert corners is not None and verdict == "paper"
    assert trace["source"] == "classic" and trace["neural_ran"] is False
    # extras sai sempre (mesmo vazio): e o que limpa a cobertura e a
    # mascara de mao anteriores no resultado publicado.
    assert extras == {}


def test_check_models_avisa_modelo_ausente(monkeypatch, tmp_path, capsys):
    fake = {"modelo_ausente_xyz.onnx": {"sha256": "0" * 64, "bytes": 1}}
    monkeypatch.setattr(
        manifest_module, "load_manifest", lambda path=None: (Path("m.json"), fake)
    )
    monkeypatch.setattr(manifest_module, "DEFAULT_MODELS_DIR", tmp_path)
    assert app_module._check_models() is True
    out = capsys.readouterr().out
    assert "modelo_ausente_xyz.onnx" in out
    assert "Release" in out


def test_check_models_falha_em_divergencia(monkeypatch, tmp_path, capsys):
    model = tmp_path / "modelo.onnx"
    model.write_bytes(b"binario")
    fake = {"modelo.onnx": {"sha256": "f" * 64, "bytes": 7}}
    monkeypatch.setattr(
        manifest_module, "load_manifest", lambda path=None: (Path("m.json"), fake)
    )
    monkeypatch.setattr(manifest_module, "DEFAULT_MODELS_DIR", tmp_path)
    # O conftest rebaixa o strict globalmente; aqui o cenario e o
    # padrao de producao (estrito): divergencia e FAIL.
    monkeypatch.setattr(manifest_module, "strict_enabled", lambda: True)
    monkeypatch.setattr(
        manifest_module,
        "verify_model",
        lambda path, **kw: (_ for _ in ()).throw(RuntimeError("sha diverge")),
    )
    assert app_module._check_models() is False
    assert "FAIL" in capsys.readouterr().out


def test_main_check_chama_run_check(monkeypatch):
    seen = {}

    def fake_check(source, helper_token_file=None):
        seen["s"] = source
        return 0

    monkeypatch.setattr(app_module, "run_check", fake_check)
    monkeypatch.setattr(sys, "argv", ["draw", "--check", "--source", "3"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 0
    assert seen["s"] == 3


def test_main_check_com_url_passa_url(monkeypatch):
    seen = {}

    def fake_check(source, helper_token_file=None):
        seen["s"] = source
        return 0

    monkeypatch.setattr(app_module, "run_check", fake_check)
    monkeypatch.setattr(
        sys, "argv", ["draw", "--check", "--source", "http://10.0.0.1:8080/video"]
    )
    with pytest.raises(SystemExit):
        app_module.main()
    assert seen["s"] == "http://10.0.0.1:8080/video"


def test_main_markers_rejeita_flags_de_modelo(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["draw", "--markers", "--neural", "x.onnx"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2


def test_main_neural_sem_caminho_e_erro(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["draw", "--neural"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2


def test_main_formato_invalido_e_erro(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["draw", "--format", "paisagem"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2


def test_main_defaults_chamam_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.update(kw))
    monkeypatch.setattr(sys, "argv", ["draw", "--save-dir", "out"])
    app_module.main()
    assert seen["source"] == "0"
    assert seen["save_dir"] == "out"
    assert seen["markers_mode"] is False
    # Sem --page-clean, o modelo padrao resolve dentro de run/PageCleanInk.
    assert seen["page_clean_model"] is None


def test_main_erro_de_execucao_sai_2(monkeypatch):
    def boom(**kw):
        raise RuntimeError("camera morta")

    monkeypatch.setattr(app_module, "run", boom)
    monkeypatch.setattr(sys, "argv", ["draw"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2


def test_save_stamp_unico_por_processo():
    stamps = {app_module._save_stamp() for _ in range(20)}
    assert len(stamps) == 20


def test_require_gui_backend_levanta_sem_highgui(monkeypatch):
    monkeypatch.setattr(app_module, "_gui_backend_compiled", lambda: False)
    with pytest.raises(RuntimeError, match="draw-core\[gui\]"):
        app_module._require_gui_backend()


def test_require_gui_backend_levanta_sem_display(monkeypatch):
    import subprocess

    monkeypatch.setattr(app_module, "_gui_backend_compiled", lambda: True)

    def _aborted(*a, **kw):
        # O Qt morre por sinal sem display: rc != 0, sem excecao.
        return subprocess.CompletedProcess(a[0], -6, "", "")

    monkeypatch.setattr(app_module.subprocess, "run", _aborted)
    with pytest.raises(RuntimeError, match="display"):
        app_module._require_gui_backend()


def test_require_gui_backend_ok_com_display(monkeypatch):
    import subprocess

    calls = []
    monkeypatch.setattr(app_module, "_gui_backend_compiled", lambda: True)
    monkeypatch.setattr(
        app_module.subprocess,
        "run",
        lambda cmd, **kw: (
            calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", "")
        ),
    )
    app_module._require_gui_backend()
    # A sonda de janela roda isolada: Qt pode abortar o filho, nunca o app.
    assert len(calls) == 1
    assert calls[0][1] == "-c"
    assert "namedWindow" in calls[0][2]


def test_gui_backend_compiled_le_buildinfo(monkeypatch):
    fake_info = (
        "General configuration for OpenCV 4.9.0\n"
        "  GUI:\n"
        "    QT:                          YES (ver 5.15.13)\n"
        "    GTK+:                        NO\n"
        "  Media I/O:\n"
        "    JPEG:                        YES\n"
    )
    monkeypatch.setattr(app_module.cv2, "getBuildInformation", lambda: fake_info)
    assert app_module._gui_backend_compiled() is True

    headless = fake_info.replace("YES (ver 5.15.13)", "NO").replace(
        "JPEG:                        YES", "JPEG:                        NO"
    )
    monkeypatch.setattr(app_module.cv2, "getBuildInformation", lambda: headless)
    assert app_module._gui_backend_compiled() is False


def test_check_models_divergencia_nao_estrita_vira_aviso(monkeypatch, tmp_path, capsys):
    # --allow-unverified: mismatch vira AVISO, nunca "confere com o
    # manifesto" — era o bug que imprimia PASS com hash errado.
    model = tmp_path / "modelo.onnx"
    model.write_bytes(b"binario")
    fake = {"modelo.onnx": {"sha256": "f" * 64, "bytes": 7}}
    monkeypatch.setattr(
        manifest_module, "load_manifest", lambda path=None: (Path("m.json"), fake)
    )
    monkeypatch.setattr(manifest_module, "DEFAULT_MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest_module, "strict_enabled", lambda: False)
    monkeypatch.setattr(
        manifest_module,
        "verify_model",
        lambda path, **kw: (_ for _ in ()).throw(RuntimeError("sha diverge")),
    )
    assert app_module._check_models() is True
    out = capsys.readouterr().out
    assert "AVISO" in out and "confere com o manifesto" not in out


def test_run_check_fora_do_linux_degrada(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(app_module.sys, "platform", "darwin")
    monkeypatch.setattr(app_module, "_require_gui_backend", lambda: None)
    monkeypatch.setattr(app_module, "_check_models", lambda: True)
    monkeypatch.setattr(
        app_module,
        "_check_camera_open",
        lambda source, label: seen.setdefault("open", (source, label)) or True,
    )
    # Sem /dev/video* e sem grp no caminho: nao deve explodir.
    assert app_module.run_check(0) == 0
    assert seen["open"] == (0, "camera 0")
    assert "AVISO" in capsys.readouterr().out


def test_run_check_avisa_sem_backend_gui(monkeypatch, capsys):
    monkeypatch.setattr(
        app_module,
        "_require_gui_backend",
        lambda: (_ for _ in ()).throw(RuntimeError("sem GUI")),
    )
    monkeypatch.setattr(app_module, "_check_models", lambda: True)
    monkeypatch.setattr(app_module, "_check_camera_open", lambda *a: False)
    monkeypatch.setattr(app_module.sys, "platform", "darwin")
    assert app_module.run_check(0) == 1
    assert "draw-core[gui]" in capsys.readouterr().out


def test_gui_backend_compiled_gui_none(monkeypatch):
    monkeypatch.setattr(
        app_module.cv2,
        "getBuildInformation",
        lambda: "General configuration\n  GUI: NONE\n  Media I/O:\n    JPEG: YES\n",
    )
    assert app_module._gui_backend_compiled() is False


def test_gui_backend_compiled_secao_irma_nao_conta(monkeypatch):
    # Media I/O irmã também é indentada: um YES fora da seção GUI
    # não pode marcar a roda headless como "com janela".
    info = (
        "General configuration\n"
        "  GUI:\n"
        "    QT:                          NO\n"
        "    GTK+:                        NO\n"
        "  Media I/O:\n"
        "    JPEG:                        YES\n"
        "    PNG:                         YES\n"
    )
    monkeypatch.setattr(app_module.cv2, "getBuildInformation", lambda: info)
    assert app_module._gui_backend_compiled() is False


class _PacedStallFeed:
    """Fonte ritmada: so entrega o quadro k+1 depois do k consumido.

    LatestFrameCapture guarda so o pacote mais novo: sem ritmo a bomba
    publica tudo antes do laco acordar e quadros do meio nunca sao
    processados — o teste de limpeza passaria sem nunca ter cacheado
    uma folha. O consumo e medido pelas chamadas de waitKey: cada
    iteracao do laco chama waitKey uma vez. Esgotados os quadros, o
    read() trava como um stall TCP; release() destrava para o join da
    bomba nao pagar o teto de 2 s.
    """

    def __init__(self, frames):
        self._frames = list(frames)
        self._served = 0
        self.consumed = 0
        self.stalled = False
        self._released = False

    def read(self):
        while self._served and self.consumed < self._served:
            if self._released:
                return False, None
            time.sleep(0.002)
        if self._served < len(self._frames):
            frame = self._frames[self._served]
            self._served += 1
            return True, frame
        # O while de ritmo garante: aqui todos os quadros ja foram
        # servidos E consumidos — todo waitKey dali em diante e do
        # ramo de stall.
        self.stalled = True
        while not self._released:
            time.sleep(0.01)
        return False, None

    def release(self):
        self._released = True


def _patch_marker_run(monkeypatch, feed, keys):
    """Sobe run() headless em --markers: janela e teclas por script.

    A ordem das chamadas de waitKey e deterministica: a bomba so publica
    o quadro seguinte depois do anterior consumido (um waitKey por
    quadro) e, com a fonte travada, o laco cai no ramo de stall a cada
    _FRAME_WAIT_SECONDS consumindo uma tecla por iteracao.
    """

    def fake_warp(frame):
        # Quadro com o primeiro pixel preto = folha; qualquer outro
        # = ausencia (warp falha, como numa mesa sem marcadores).
        if frame[0, 0, 0] == 0:
            warped = np.full((60, 40, 3), 200, np.uint8)
            return warped, {i: (5, 5) for i in config.MARKER_IDS}
        return None, {}

    def wait_key(_ms):
        feed.consumed += 1
        if not feed.stalled:
            # Quadro ainda por servir/consumir: nunca gasta tecla do
            # script — 's' e 'q' so caem no ramo de stall, senao um
            # timeout esporio antes do ultimo quadro deslocava o 's'
            # para o save normal e o teste passava sem cobrir o stall.
            return -1
        return next(key_iter, ord("q"))

    monkeypatch.setattr(app_module, "_require_gui_backend", lambda: None)
    monkeypatch.setattr(app_module, "open_capture", lambda *a, **k: feed)
    monkeypatch.setattr(paper_module, "warp_paper", fake_warp)
    monkeypatch.setattr(app_module.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(app_module.cv2, "destroyAllWindows", lambda: None)
    key_iter = iter(keys)
    monkeypatch.setattr(app_module.cv2, "waitKey", wait_key)


def test_markers_salva_folha_exibida_durante_stall(monkeypatch, tmp_path):
    """'s' no stall com folha em cena salva a ultima folha retificada.

    Sem o cache, o ramo de espera reportava "nada a salvar" enquanto
    a folha seguia visivel na janela.
    """
    feed = _PacedStallFeed([_frame(), _frame()])
    _patch_marker_run(monkeypatch, feed, [ord("s"), ord("q")])

    app_module.run(source=0, save_dir=str(tmp_path), markers_mode=True)

    saved = list(tmp_path.glob("canvas_*.png"))
    assert len(saved) == 1
    out = cv2.imread(str(saved[0]))
    assert out is not None and out.shape == (60, 40, 3)
    assert int(out[0, 0, 0]) == 200


def test_markers_stall_apos_quadro_sem_folha_nao_salva(monkeypatch, tmp_path, capsys):
    """Quadro processado sem folha limpa o cache: 's' no stall nao salva.

    O cache nao pode devolver uma folha que saiu de cena — senao o 's'
    gravaria uma imagem que a janela ja nao mostra.
    """
    no_sheet = np.full((120, 160, 3), 255, np.uint8)
    feed = _PacedStallFeed([_frame(), _frame(), no_sheet])
    _patch_marker_run(monkeypatch, feed, [ord("s"), ord("q")])

    app_module.run(source=0, save_dir=str(tmp_path), markers_mode=True)

    assert list(tmp_path.glob("canvas_*.png")) == []
    assert "Nada a salvar" in capsys.readouterr().out
