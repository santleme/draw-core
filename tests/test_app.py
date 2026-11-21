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
    monkeypatch.setattr(
        manifest_module,
        "ensure_models",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert app_module._check_models() is True
    out = capsys.readouterr().out
    assert "modelo_ausente_xyz.onnx" in out
    assert "Release" in out


def test_check_models_baixa_modelo_ausente_e_valida(monkeypatch, tmp_path, capsys):
    payload = b"binario"
    sha = manifest_module.hashlib.sha256(payload).hexdigest()
    fake = {
        "page_clean_unet.onnx": {
            "sha256": sha,
            "bytes": len(payload),
            "approved_by": "teste",
        }
    }
    manifest_file = tmp_path / "manifest.json"
    import json

    manifest_file.write_text(json.dumps({"schema": 1, "models": fake}))
    models_dir = tmp_path / "models"
    monkeypatch.setattr(
        manifest_module,
        "load_manifest",
        lambda path=None: (manifest_file, fake),
    )
    monkeypatch.setattr(manifest_module, "DEFAULT_MODELS_DIR", models_dir)

    def fake_ensure(names=None, **kw):
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / "page_clean_unet.onnx"
        target.write_bytes(payload)
        return [target]

    monkeypatch.setattr(manifest_module, "ensure_models", fake_ensure)
    assert app_module._check_models() is True
    out = capsys.readouterr().out
    assert "PASS: page_clean_unet.onnx confere com o manifesto" in out


def test_provision_official_models_isolates_failures_and_handles_explicit_official_paths(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(manifest_module, "DEFAULT_MODELS_DIR", tmp_path)
    requested = []

    def fake_ensure(names):
        requested.extend(names)
        if names == ["page_clean_unet.onnx"]:
            raise RuntimeError("falha no primeiro download")
        return []

    monkeypatch.setattr(manifest_module, "ensure_models", fake_ensure)
    explicit_official_ink = str(tmp_path / "page_clean_unet.onnx")
    custom_hand = str(tmp_path / "outro_dir" / "hand_landmarker.task")
    app_module._provision_official_models(
        neural_model=None,
        page_clean_model=explicit_official_ink,
        hand_model=custom_hand,
    )
    assert requested == ["page_clean_unet.onnx", "draw_seg_v3_dyn.onnx"]


def test_check_models_falha_em_limiar_invalido(monkeypatch, tmp_path, capsys):
    """Binário confere mas runtime_threshold quebrado: FAIL, não PASS."""
    model = tmp_path / "page_clean_unet.onnx"
    model.write_bytes(b"binario")
    entry = {"sha256": "f" * 64, "bytes": 7, "runtime_threshold": 1.5}
    fake = {"page_clean_unet.onnx": entry}
    monkeypatch.setattr(
        manifest_module, "load_manifest", lambda path=None: (Path("m.json"), fake)
    )
    monkeypatch.setattr(manifest_module, "DEFAULT_MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest_module, "verify_model", lambda path, **kw: entry)
    assert app_module._check_models() is False
    out = capsys.readouterr().out
    assert "FAIL" in out and "runtime_threshold" in out
    assert "PASS" not in out


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
    with pytest.raises(RuntimeError, match=r"draw-core\[gui\]"):
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
    monkeypatch.setattr(app_module, "_open_window", lambda fullscreen: None)
    monkeypatch.setattr(app_module.cv2, "resizeWindow", lambda *a, **k: None)
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


def _reset_camera_menu(monkeypatch, current=0):
    state = app_module._camera_menu
    monkeypatch.setitem(state, "open", False)
    monkeypatch.setitem(state, "items", [])
    monkeypatch.setitem(state, "current", current)
    monkeypatch.setitem(state, "busy", None)
    monkeypatch.setitem(state, "request", None)
    return state


def test_camera_label_formats_each_kind_of_source():
    assert app_module._camera_label(2) == "Camera 2"
    assert app_module._camera_label("http://192.168.0.10:8080/video") == (
        "Rede: 192.168.0.10:8080"
    )
    assert app_module._camera_label("captures/sessao.drawcap") == "sessao.drawcap"


def test_list_cameras_adds_the_current_source_when_it_is_not_a_device(monkeypatch):
    monkeypatch.setattr(
        app_module, "_video_nodes", lambda: ["/dev/video0", "/dev/video2"]
    )
    assert app_module._list_cameras(0) == [("Camera 0", 0), ("Camera 2", 2)]
    url = "http://host/video"
    assert app_module._list_cameras(url)[0] == ("Rede: host", url)


def test_camera_click_opens_the_menu_and_selects_another_camera(monkeypatch):
    state = _reset_camera_menu(monkeypatch)
    monkeypatch.setattr(
        app_module, "_video_nodes", lambda: ["/dev/video0", "/dev/video2"]
    )
    monkeypatch.setattr(app_module, "_window_size", lambda: (1600, 640))
    layout = app_module._camera_layout()
    cx, cy, _ = layout["dot"]
    assert app_module._camera_click(cx, cy) is True
    assert state["open"] and len(state["items"]) == 2
    layout = app_module._camera_layout()
    rx, ry, _, _ = layout["rows"][1]
    assert app_module._camera_click(rx + 3, ry + 3) is True
    assert not state["open"] and state["request"] == 2


def test_camera_click_on_the_current_camera_or_outside_requests_nothing(monkeypatch):
    state = _reset_camera_menu(monkeypatch)
    monkeypatch.setattr(
        app_module, "_video_nodes", lambda: ["/dev/video0", "/dev/video2"]
    )
    monkeypatch.setattr(app_module, "_window_size", lambda: (1600, 640))
    cx, cy, _ = app_module._camera_layout()["dot"]
    app_module._camera_click(cx, cy)
    rx, ry, _, _ = app_module._camera_layout()["rows"][0]
    assert app_module._camera_click(rx + 3, ry + 3) is True
    assert state["request"] is None and not state["open"]
    app_module._camera_click(cx, cy)
    assert app_module._camera_click(5, 5) is False
    assert not state["open"] and state["request"] is None


def _reset_settings(monkeypatch):
    state = app_module._settings
    monkeypatch.setitem(state, "open", False)
    monkeypatch.setitem(
        state, "on", {key: False for key, _ in app_module._SETTINGS_ITEMS}
    )
    monkeypatch.setitem(app_module._camera_menu, "open", False)
    monkeypatch.setattr(app_module, "_window_size", lambda: (1600, 640))
    return state


def test_settings_click_opens_and_toggles_options_without_closing(monkeypatch):
    state = _reset_settings(monkeypatch)
    cx, cy, _ = app_module._settings_layout()["gear"]
    assert app_module._settings_click(cx, cy) is True and state["open"]
    rx, ry, _, _ = app_module._settings_layout()["rows"][1]
    assert app_module._settings_click(rx + 3, ry + 3) is True
    assert state["open"] and state["on"]["latency"] is True
    assert app_module._settings_click(rx + 3, ry + 3) is True
    assert state["on"]["latency"] is False
    assert app_module._settings_click(5, 5) is False and not state["open"]


def test_hud_values_only_show_what_is_enabled(monkeypatch):
    state = _reset_settings(monkeypatch)
    monkeypatch.setitem(app_module._hud, "captured_at", None)
    monkeypatch.setitem(app_module._hud, "fps", 14.5)
    monkeypatch.setitem(app_module._mouse, "pos", (10, 20))
    assert app_module._hud_values() == {"probe": None, "latency": None, "fps": None}
    state["on"].update(probe=True, fps=True)
    values = app_module._hud_values()
    assert values["probe"] == (10, 20) and values["fps"] == 14.5


def test_apply_fullscreen_only_calls_cv2_when_the_option_changes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        app_module.cv2, "setWindowProperty", lambda *args: calls.append(args)
    )
    monkeypatch.setattr(app_module, "_fullscreen_applied", [False])
    monkeypatch.setitem(app_module._settings["on"], "fullscreen", False)
    app_module._apply_fullscreen()
    assert calls == []
    app_module._settings["on"]["fullscreen"] = True
    app_module._apply_fullscreen()
    app_module._apply_fullscreen()
    assert [c[2] for c in calls] == [app_module.cv2.WINDOW_FULLSCREEN]
    app_module._settings["on"]["fullscreen"] = False
    app_module._apply_fullscreen()
    assert [c[2] for c in calls][-1] == app_module.cv2.WINDOW_NORMAL
    assert len(calls) == 2


def test_mouse_events_are_mapped_back_to_layout_coordinates(monkeypatch):
    monkeypatch.setitem(app_module._view_map, "scale", 0.5)
    monkeypatch.setitem(app_module._view_map, "offset", (100, 40))
    monkeypatch.setitem(app_module._mouse, "pos", None)
    app_module._on_mouse(app_module.cv2.EVENT_MOUSEMOVE, 300, 200, 0, None)
    assert app_module._mouse["pos"] == (400, 320)


def test_window_size_uses_the_real_rect_down_to_a_small_floor(monkeypatch):
    monkeypatch.setattr(app_module, "_window_sized", {True})
    monkeypatch.setattr(app_module, "_window_rect", lambda: (400, 200))
    assert app_module._window_size() == (400, 200)
    monkeypatch.setattr(app_module, "_window_rect", lambda: (300, 100))
    assert app_module._window_size() == (1600, 640)


def test_camera_menu_with_many_cameras_scrolls_and_selects_absolute_rows(monkeypatch):
    state = _reset_camera_menu(monkeypatch)
    monkeypatch.setattr(app_module, "_window_size", lambda: (1600, 640))
    monkeypatch.setattr(
        app_module,
        "_list_cameras",
        lambda current: [(f"Camera {i}", i) for i in range(30)],
    )
    monkeypatch.setitem(state, "scroll", 0)
    cx, cy, _ = app_module._camera_layout()["dot"]
    assert app_module._camera_click(cx, cy) is True
    fx = app_module._camera_fx()
    layout = app_module._camera_layout()
    assert len(fx["items"]) == layout["visible"] < 30
    monkeypatch.setitem(state, "scroll", 3)
    layout = app_module._camera_layout()
    assert layout["first"] == 3
    rx, ry, _, _ = layout["rows"][0]
    assert app_module._camera_click(rx + 3, ry + 3) is True
    assert state["request"] == 3


def test_open_ready_source_requires_a_first_frame(monkeypatch):
    class _Dead:
        released = 0

        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            _Dead.released += 1

    monkeypatch.setattr(app_module, "open_capture", lambda *a, **k: _Dead())
    with pytest.raises(RuntimeError, match="nao entregou quadros"):
        app_module._open_ready_source("x", ready_seconds=0.3)

    class _Live(_Dead):
        def read(self):
            return True, np.zeros((8, 8, 3), np.uint8)

    monkeypatch.setattr(app_module, "open_capture", lambda *a, **k: _Live())
    source = app_module._open_ready_source("x", ready_seconds=2.0)
    try:
        assert source.wait_next(0, timeout=1.0) is not None
    finally:
        source.stop()


def test_waiting_view_keeps_the_controls_available():
    from draw import view as view_module

    waiting = app_module._waiting_view()
    view = view_module.compose_view(waiting, view_module.blank_sheet())
    assert waiting.shape == (720, 1280, 3) and view.shape == (640, 1600, 3)


def test_camera_menu_arrows_scroll_within_bounds_without_selecting(monkeypatch):
    state = _reset_camera_menu(monkeypatch)
    monkeypatch.setattr(app_module, "_window_size", lambda: (1600, 640))
    monkeypatch.setattr(
        app_module,
        "_list_cameras",
        lambda current: [(f"Camera {i}", i) for i in range(30)],
    )
    cx, cy, _ = app_module._camera_layout()["dot"]
    app_module._camera_click(cx, cy)
    layout = app_module._camera_layout()
    ux, uy, _, _ = layout["scroll_up"]
    dx, dy, _, _ = layout["scroll_down"]
    assert app_module._camera_click(ux + 2, uy + 2) is True and state["scroll"] == 0
    for _ in range(100):
        assert app_module._camera_click(dx + 2, dy + 2) is True
    layout = app_module._camera_layout()
    assert state["scroll"] == 30 - layout["visible"] == layout["first"]
    assert state["open"] and state["request"] is None
    assert app_module._camera_click(ux + 2, uy + 2) is True
    assert state["scroll"] == 30 - layout["visible"] - 1, "sobe 1 de imediato"


def test_mouse_wheel_events_are_ignored_by_the_handler(monkeypatch):
    monkeypatch.setitem(app_module._view_map, "scale", 1.0)
    monkeypatch.setitem(app_module._view_map, "offset", (0, 0))
    monkeypatch.setitem(app_module._mouse, "pos", None)
    _reset_camera_menu(monkeypatch)
    app_module._camera_menu["open"] = True
    app_module._on_mouse(app_module.cv2.EVENT_MOUSEWHEEL, 5, 5, 7 << 16, None)


def test_main_orientation_paisagem_deita_a_grade(monkeypatch):
    seen = []
    monkeypatch.setattr(
        app_module,
        "run",
        lambda **kw: seen.append(
            (config.PAPER_ORIENTATION, config.PAPER_W, config.PAPER_H)
        ),
    )
    monkeypatch.setattr(sys, "argv", ["draw", "--orientation", "paisagem"])
    app_module.main()
    assert seen[-1] == ("paisagem", 1200, 850)
    config.apply_format("a4")


def test_main_orientation_paisagem_recusa_markers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["draw", "--markers", "--orientation", "paisagem"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2


def _run_main(monkeypatch, *argv):
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(sys, "argv", ["draw", *argv])
    app_module.main()
    return seen[-1]


def test_main_sem_flags_usa_o_preset_de_producao(monkeypatch):
    pipeline = _run_main(monkeypatch)["pipeline_config"]
    assert pipeline == app_module.PipelineConfig()
    assert pipeline.ink == app_module.InkConfig.production()
    assert pipeline.ink_auto_gain is True
    assert pipeline.detect_wake is True
    ink = pipeline.ink
    assert ink.ruling_removal and ink.hysteresis and ink.motion_veto
    assert ink.pass_interval_s == 0.15
    assert ink.vote_score_strong > 1.9


def test_main_legacy_ink_volta_ao_comportamento_anterior(monkeypatch):
    pipeline = _run_main(monkeypatch, "--legacy-ink")["pipeline_config"]
    assert pipeline == app_module.PipelineConfig(
        ink=app_module.InkConfig(), ink_auto_gain=False, detect_wake=False
    )


def test_main_no_ink_ruling_tira_so_a_pauta(monkeypatch):
    pipeline = _run_main(monkeypatch, "--no-ink-ruling")["pipeline_config"]
    assert pipeline.ink == app_module.InkConfig.production(ruling_removal=False)
    assert pipeline.ink_auto_gain and pipeline.detect_wake


def test_main_ink_vote_e_interval_ajustam_o_preset(monkeypatch):
    ink = _run_main(monkeypatch, "--ink-vote", "0.9", "--ink-interval", "0.4")[
        "pipeline_config"
    ].ink
    assert ink == app_module.InkConfig.production(vote_level=0.9, pass_interval_s=0.4)


def test_main_spread_mode_entra_na_pipeline_config(monkeypatch):
    pipeline = _run_main(monkeypatch, "--spread-mode", "always")["pipeline_config"]
    assert pipeline.spread_mode == "always"


def test_main_markers_recusa_spread_mode_explicito(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["draw", "--markers", "--spread-mode", "auto"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2


def test_main_spread_mode_explicito_sinaliza_run(monkeypatch):
    for argv in (["--spread-mode", "auto"], ["--spread-mode", "always"]):
        kw = _run_main(monkeypatch, *argv)
        assert kw["spread_mode_explicit"] is True


def test_main_sem_spread_mode_nao_sinaliza_explicit(monkeypatch):
    assert _run_main(monkeypatch)["spread_mode_explicit"] is False


@pytest.mark.parametrize(
    "spread_mode,explicit,aviso",
    [
        ("always", False, "spread_mode='always' nao aplicado"),
        ("auto", True, "spread_mode='auto' nao aplicado"),
        ("auto", False, None),
    ],
)
def test_run_warns_when_spread_mode_has_no_neural_detector(
    monkeypatch, tmp_path, capsys, spread_mode, explicit, aviso
):
    from draw.settings import PipelineConfig

    monkeypatch.setattr(app_module, "_apply_opencv_thread_cap", lambda: None)
    monkeypatch.setattr(app_module, "_require_gui_backend", lambda: None)
    monkeypatch.setattr(app_module, "_open_window", lambda _fullscreen: None)
    monkeypatch.setattr(app_module, "_provision_official_models", lambda *_args: None)
    monkeypatch.setattr(app_module, "_DEFAULT_SHEET_MODEL", tmp_path / "missing.onnx")
    monkeypatch.setattr(app_module.cv2, "destroyAllWindows", lambda: None)

    class _FailingPageClean:
        def __init__(self, *_args, **_kwargs):
            pass

        def load(self):
            raise RuntimeError("stop before camera setup")

    monkeypatch.setattr(app_module, "PageCleanInk", _FailingPageClean)

    with pytest.raises(RuntimeError, match="modelo pagina->limpa nao carregou"):
        app_module.run(
            pipeline_config=PipelineConfig(spread_mode=spread_mode),
            spread_mode_explicit=explicit,
        )

    out = capsys.readouterr().out
    assert (aviso in out) if aviso is not None else ("nao aplicado" not in out)


def test_main_markers_sem_flags_usa_o_preset_legado(monkeypatch):
    pipeline = _run_main(monkeypatch, "--markers")["pipeline_config"]
    assert pipeline.ink == app_module.InkConfig()
    assert pipeline.ink_auto_gain is False and pipeline.detect_wake is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--markers", "--ink-vote", "0.85"],
        ["--markers", "--ink-interval", "0.15"],
        ["--markers", "--no-ink-ruling"],
        ["--markers", "--legacy-ink"],
        ["--legacy-ink", "--no-ink-ruling"],
        ["--ink-interval", "0"],
        ["--ink-vote", "1.5"],
        ["--ink-gain"],
        ["--ink-ruling"],
        ["--ink-hysteresis"],
        ["--ink-motion-veto"],
        ["--detect-wake"],
        ["--experimental-all"],
    ],
)
def test_main_combinacoes_e_flags_removidas_saem_com_2(monkeypatch, argv):
    monkeypatch.setattr(app_module, "run", lambda **kw: None)
    monkeypatch.setattr(sys, "argv", ["draw", *argv])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2
