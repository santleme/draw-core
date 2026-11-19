"""Interface de producao: janela, teclas e montagem do pipeline.

Uso:
  python -m draw                                  # webcam local (indice 0)
  python -m draw --check                          # valida modelos e camera e sai
  python -m draw --fullscreen                     # tela cheia, sem a moldura
  python -m draw --markers                        # folha-modelo impressa
  python -m draw --neural modelo.onnx             # outro detector de folha
  python -m draw --page-clean modelo.onnx         # outro modelo de tinta
  python -m draw --hand-model modelo.task         # outro modelo de mao
  python -m draw --source 1                       # outra webcam local
  python -m draw --source http://192.168.0.10:8080/video   # camera via rede
  python -m draw --source sessao.drawcap          # gravacao do ajudante
  python -m draw --source tcp://192.168.0.10      # ajudante ao vivo
      --helper-token-file token.txt               # token do ajudante
  python -m draw --source hvcam://                # webcam do Windows via vsock
  python -m draw --source hvcam+tcp://10.0.0.2    # idem, fallback TCP

Sem nenhuma flag, o caminho e o de producao: folha pelo detector
neural (models/draw_seg_v3_dyn.onnx, com o classico de reserva) e
tinta pelo modelo pagina->limpa (models/page_clean_unet.onnx), em
duas camadas na trabalhadora: a provisoria aparece na tela na
primeira passada e a confirmada, que e o canvas dos consumidores,
entra pelo voto entre passadas. Os binarios moram em models/ na
raiz do repositorio e sao conferidos contra assets/manifest.json
antes de abrir (sha256 + tamanho); --allow-unverified rebaixa a
recusa para aviso.

A fonte via rede (MJPEG/RTSP) e o caminho recomendado no WSL. O WSL nao
enxerga webcams USB do Windows por padrao, mas enxerga streams de rede.
A alternativa nativa e a fonte hvcam://, que puxa os quadros da webcam
do Windows por Hyper-V socket (hvcam-host.exe, santleme/hvcam) — sem
IP, sem firewall e sem usbipd.
Assim todo o processamento (e o ML futuro) fica no Linux. A camera so
envia os frames. Ela pode ser um celular, outro computador ou uma camera IP.

Teclas: s = salvar o canvas como PNG, f = capturar um frame cru,
g = ligar ou desligar a captura automatica de frames, q/Esc = sair.
"""

import argparse
import dataclasses
import glob
import itertools
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import camera as camera_module
from . import config
from . import paper as paper_module
from . import perception as neural_module
from . import view as view_module
from .camera import (
    LatestFrameCapture,
    PngWriter,
    is_wsl,
    open_capture,
    parse_source,
)
from .hand import HandMasker, hand_model_available
from .hand_presence import HandPresenceWorker, make_hand_presence
from .hvcam import HvcamCapture, is_hvcam_source
from .infra import manifest as manifest_module
from .infra import session_report as session_report_module
from .infra import telemetry as telemetry_module
from .infra.safe_io import ensure_private_dir
from .ink import render as ink_render
from .ink.config import InkConfig
from .ink.worker import PageCleanWorker
from .page_clean import PageCleanInk, resolve_threshold
from .perception import NeuralSheetDetector
from .pipeline import INK_WORKER_FAILED, Pipeline
from .protocol import HelperCapture, ProtocolError, is_helper_source
from .settings import PipelineConfig
from .tracker import SheetDetectionWorker, _corners_json

# Tempo que o aviso de fonte morta fica na janela antes da saida.
# Sem ele, a morte da fonte e um congelamento silencioso. O waitKey
# devolve antes se qualquer tecla for apertada.
_DEAD_NOTICE_MS = 3000
# Janela curta de espera por quadro novo: longa demais e uma fonte
# em stall TCP (HelperCapture com stall_windows) congela as teclas
# de saida dentro de um unico read().
_FRAME_WAIT_SECONDS = 0.25

# Teto do pool paralelo do OpenCV. Sem teto, o pool abre com as
# CPUs logicas do host (16 no Ryzen) e vira o terceiro disputante
# dos nucleos fisicos, junto dos pools ORT do detector e da tinta
# (oversubscription medida na mesa do Lucas, 2026-11-02: CPU
# desenhando 550,8% com os pools 4+8+16; teto combinado 2+4 no
# ORT cortou para 289,8%, passada igual).
_OPENCV_THREAD_CAP = 4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SHEET_MODEL = _REPO_ROOT / "models" / "draw_seg_v3_dyn.onnx"

_CLASSIC_TRACE_VERDICT = {"paper": "corners", "blank_maybe": "blank_maybe"}


def _apply_opencv_thread_cap():
    """Limita o pool do OpenCV ao teto combinado. Devolve o teto."""
    cap = max(1, min(_OPENCV_THREAD_CAP, os.cpu_count() or _OPENCV_THREAD_CAP))
    cv2.setNumThreads(cap)
    return cap


def _detect_sheet_combined_traced(frame, neural_detector):
    """Detecção do modo sem marcadores, com o rastro da aquisição.

    Neural primário, clássico de reserva — a decisão do legado,
    linha a linha. Sem detector neural, é o clássico puro. Com
    detector, ele decide primeiro; fechando o quadrilátero, vale o
    veredito dele. Não fechando, a reserva clássica só fala quando a
    neural não negou a folha com força: logit de presença até
    PRESENCE_VETO_LOGIT devolve (None, None) direto. Logit negativo
    fraco (acima do limiar) não veta: a reserva segue valendo para a
    folha real que a neural subdispara num setup novo.

    Terceiro desfecho: se nenhum quadrilátero fechou mas o logit de
    presença é positivo, devolve (None, "paper"): presença sem
    posição. O rastreador lê isso como "a folha está lá, segura o
    quadrilátero anterior e renova o relógio de perda".

    O terceiro elemento da saída é o rastro (presença, fonte do
    desfecho, vereditos parciais) que alimenta o evento de aquisição
    da telemetria. O quarto são os extras: sempre um dict (mesmo
    vazio) para que uma detecção sem máscaras LIMPE a cobertura e a
    máscara de mão anteriores no resultado publicado.
    """
    trace = {
        "neural_ran": neural_detector is not None,
        "presence_logit": None,
        "veto_logit": None,
        "neural_corners": None,
        "neural_verdict": None,
        "classic_ran": False,
        "classic_verdict": None,
        "classic_corners": None,
    }
    extras = {}

    def _done(corners, verdict, source):
        trace["source"] = source
        trace["combined_corners"] = _corners_json(corners)
        trace["combined_verdict"] = verdict
        # extras sai sempre (mesmo vazio): é o que deixa a detecção
        # clássica ou sem máscara LIMPAR a cobertura anterior no
        # laço, em vez de deixá-la defasada.
        return corners, verdict, trace, extras

    def _classic():
        corners, verdict = paper_module.detect_sheet_with_confidence(frame)
        trace["classic_ran"] = True
        trace["classic_verdict"] = (
            _CLASSIC_TRACE_VERDICT.get(verdict, verdict)
            if corners is not None
            else None
        )
        trace["classic_corners"] = _corners_json(corners)
        return corners, verdict

    if neural_detector is not None:
        trace["veto_logit"] = float(neural_module.PRESENCE_VETO_LOGIT)
        # A detecção pede a máscara sempre: a cobertura vai no
        # extras para o laço retificar com a homografia do warp de
        # tinta; a máscara de mão alimenta a oclusão semântica.
        corners, verdict, mask = neural_detector(frame, return_mask=True)
        if corners is not None and mask is not None:
            # Máscara quase vazia não é evidência de folha: sem o
            # piso, uma detecção ruim transformaria a cobertura num
            # voto sempre-zero e secaria a tinta inteira. A
            # cobertura é papel+tinta+mão — só o fundo fica fora.
            coverage_mask = getattr(neural_detector, "last_coverage_mask", None)
            if (
                coverage_mask is not None
                and np.count_nonzero(coverage_mask) > coverage_mask.size * 0.05
            ):
                frame_h, frame_w = frame.shape[:2]
                if coverage_mask.shape != (frame_h, frame_w):
                    coverage_mask = cv2.resize(
                        coverage_mask,
                        (frame_w, frame_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                extras["coverage_mask"] = coverage_mask
            # Máscara semântica de mão (mesma remessa, mesmo frame):
            # segundo sinal de oclusão para o canal visible do voto
            # de tinta.
            hand_mask = getattr(neural_detector, "last_hand_mask", None)
            if hand_mask is not None:
                frame_h, frame_w = frame.shape[:2]
                if hand_mask.shape != (frame_h, frame_w):
                    hand_mask = cv2.resize(
                        hand_mask,
                        (frame_w, frame_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                extras["hand_mask"] = hand_mask
        logit = neural_detector.last_presence_logit
        trace["presence_logit"] = None if logit is None else float(logit)
        trace["neural_corners"] = _corners_json(corners)
        trace["neural_verdict"] = verdict
        if corners is not None:
            return _done(corners, verdict, "neural")
        if logit is not None and logit <= neural_module.PRESENCE_VETO_LOGIT:
            return _done(None, None, "veto")
        corners, verdict = _classic()
        if corners is None and logit is not None and logit > 0:
            return _done(None, "paper", "presence_only")
        return _done(corners, verdict, "classic" if corners is not None else "none")
    corners, verdict = _classic()
    return _done(corners, verdict, "classic" if corners is not None else "none")


def _detect_sheet_combined(frame, neural_detector):
    corners, verdict = _detect_sheet_combined_traced(frame, neural_detector)[:2]
    return corners, verdict


def run_check(source=0, helper_token_file=None):
    """Validacao da camera, porta por porta. Retorna 0 ou 1.

    O `source` importa: o indice pedido e o testado, e uma URL e
    aberta como URL.
    """
    print(f"PASS: OpenCV disponivel. Versao {cv2.__version__}.")
    try:
        _require_gui_backend()
    except RuntimeError:
        print(
            "AVISO: backend de janela (HighGUI) indisponivel; a janela "
            "do Draw nao abre. Correcao: pip install 'draw-core[gui]'."
        )
    else:
        print("PASS: backend de janela (HighGUI) disponivel.")
    all_ok = _check_models()

    if not isinstance(source, int):
        if is_hvcam_source(source):
            return 0 if _check_hvcam_source(source) and all_ok else 1
        if is_helper_source(source):
            return (
                0 if _check_helper_source(source, helper_token_file) and all_ok else 1
            )
        return 0 if _check_remote_source(source) and all_ok else 1

    if not sys.platform.startswith("linux"):
        print(
            "AVISO: as checagens de /dev/video* e de permissoes Unix "
            "valem so no Linux; testando a abertura direta da camera."
        )
        return 0 if _check_camera_open(source, f"camera {source}") and all_ok else 1

    nodes = _video_nodes()
    if not nodes:
        print("FAIL: nenhum /dev/video* existe neste sistema.")
        if is_wsl():
            print(
                "  Correcao: anexe a camera com "
                "bash scripts/setup_camera_wsl.sh. A alternativa e uma "
                "camera via rede com --source."
            )
        else:
            print("  Correcao: conecte uma webcam a este computador.")
        return 1
    print(f"PASS: dispositivos de video encontrados: {', '.join(nodes)}.")

    node = f"/dev/video{source}"
    if node not in nodes:
        print(
            f"FAIL: {node} nao existe. Os dispositivos deste sistema "
            f"sao: {', '.join(nodes)}."
        )
        print(
            "  Correcao: use --source com um dos indices acima, ou "
            "conecte a camera pedida."
        )
        return 1

    access = os.access(node, os.R_OK | os.W_OK)
    group_note = ""
    try:
        import grp

        gid = os.stat(node).st_gid
        try:
            group_name = grp.getgrgid(gid).gr_name
        except KeyError:
            group_name = str(gid)
        in_group = gid in os.getgroups()
        group_note = (
            f" Grupo dono de {node}: {group_name}. "
            f"Este processo esta nesse grupo: {'sim' if in_group else 'nao'}."
        )
    except Exception:
        group_note = ""
    if access:
        print(f"PASS: este processo le e escreve em {node}.{group_note}")
    else:
        print(
            f"FAIL: este processo nao tem permissao de leitura e "
            f"escrita em {node}.{group_note}"
        )
        print(
            "  Correcao definitiva: sudo usermod -aG video $USER "
            "(vale nas proximas sessoes)."
        )
        print('  Correcao nesta sessao: sg video -c "python -m draw".')
        all_ok = False

    if not _check_camera_open(source, node):
        all_ok = False

    return 0 if all_ok else 1


def _check_camera_open(source, label):
    """Abre a fonte local e espera um quadro por 3 s. True/False."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(
            f"FAIL: nao consegui abrir {label}. Outro programa "
            "provavelmente esta usando a camera."
        )
        return False
    camera_module._configure_local_capture(capture)
    frame = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            break
        time.sleep(0.1)
    capture.release()
    if frame is not None:
        print(
            f"PASS: quadro lido de {label}. "
            f"Formato (altura, largura, canais): {frame.shape}."
        )
        return True
    print(
        f"FAIL: {label} abriu, mas nenhum quadro chegou em 3 "
        "segundos. A camera pode estar ocupada ou travada."
    )
    return False


def _check_helper_source(source, token_file=None):
    """Abre uma fonte do ajudante (.drawcap/tcp://) e confere um quadro."""
    print(f"PASS: fonte do ajudante informada: {source}.")
    try:
        # stall_windows=0 + prazo de registro: o check mede UM quadro
        # dentro de um deadline total — retentar esticaria a janela,
        # e so o timeout por recv deixaria um frame pingando passar.
        capture = HelperCapture(
            source,
            timeout=5.0,
            token_file=token_file,
            stall_windows=0,
            record_timeout_s=5.0,
        )
    except (ProtocolError, OSError) as error:
        print(f"FAIL: {error}")
        return 0
    if not capture.isOpened():
        print(
            "FAIL: nao consegui abrir a fonte do ajudante. Confira se "
            "o arquivo .drawcap existe ou se o ajudante esta escutando "
            "na porta TCP."
        )
        return 0
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        print("FAIL: a fonte abriu, mas nenhum quadro chegou em 5 segundos.")
        return 0
    print(
        "PASS: quadro lido da fonte do ajudante. "
        f"Formato (altura, largura, canais): {frame.shape}."
    )
    return 1


def _check_hvcam_source(source):
    """Abre uma fonte hvcam:// e confere que um quadro chega. 0/1."""
    print(f"PASS: fonte hvcam informada: {source}.")
    capture = HvcamCapture(source, timeout_ms=5000)
    if not capture.isOpened():
        if capture.open_error is not None:
            print(f"FAIL: {capture.open_error}")
        else:
            print(
                "FAIL: nao consegui conectar na fonte hvcam. Confira se o "
                "hvcam-host.exe esta rodando no Windows; fora do WSL use "
                "--source hvcam+tcp://<host>:<porta>."
            )
        return 0
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        print("FAIL: a fonte abriu, mas nenhum quadro chegou em 5 segundos.")
        return 0
    print(
        "PASS: quadro lido da fonte hvcam. "
        f"Formato (altura, largura, canais): {frame.shape}."
    )
    return 1


def _check_remote_source(url):
    """Abre uma fonte de rede e confere que um quadro chega. 0/1."""
    print(f"PASS: fonte de rede informada: {url}.")
    capture = cv2.VideoCapture(url)
    if not capture.isOpened():
        print(
            "FAIL: nao consegui abrir a fonte de rede. Confira a URL, "
            "se a camera esta transmitindo e se este computador "
            "alcanca o endereco."
        )
        return 0
    frame = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            break
        time.sleep(0.1)
    capture.release()
    if frame is None:
        print("FAIL: a fonte abriu, mas nenhum quadro chegou em 5 segundos.")
        return 0
    print(
        "PASS: quadro lido da fonte de rede. "
        f"Formato (altura, largura, canais): {frame.shape}."
    )
    return 1


def _check_models():
    """Confere os modelos contra assets/manifest.json. True se tudo bate.

    Sem manifesto (o caso de um clone novo), imprime o aviso e
    devolve True: nada a verificar nao e falha. Um .onnx/.task em
    models/ sem entrada no manifesto e FAIL no modo estrito (padrao)
    e AVISO com DRAW_MODEL_STRICT=0.
    """
    manifest_path, models = manifest_module.load_manifest()
    ok = True
    strict = manifest_module.strict_enabled()
    for extra in manifest_module.unregistered_model_files():
        label = "FAIL" if strict else "AVISO"
        print(
            f"{label}: {extra.name} esta em models/ e nao tem entrada no "
            "manifesto; a carga dele e recusada no modo estrito."
        )
        ok = ok and not strict
    if not models:
        print(
            f"AVISO: {manifest_path} sem modelos registrados; "
            "nenhum binario foi conferido."
        )
        return ok
    for name in sorted(models):
        path = manifest_module.DEFAULT_MODELS_DIR / name
        if not path.exists():
            print(
                f"AVISO: {name} esta no manifesto e nao esta em models/. "
                "Baixe os modelos oficiais pela Release do repositorio."
            )
            continue
        try:
            # strict=True aqui de propósito: mismatch de hash/tamanho
            # nunca sai como "confere com o manifesto" — vira FAIL no
            # modo estrito e AVISO com --allow-unverified.
            entry = manifest_module.verify_model(path, strict=True)
        except RuntimeError as error:
            print(f"{'FAIL' if strict else 'AVISO'}: {error}")
            ok = ok and not strict
            continue
        if entry is None:
            print(f"AVISO: {name} nao tem entrada no manifesto.")
            continue
        try:
            # Calibração inválida derruba o load() da tinta; o
            # preflight tem de dizer isso antes, não um PASS.
            resolve_threshold(entry)
        except RuntimeError as error:
            print(f"FAIL: {name}: {error}")
            ok = False
            continue
        print(
            f"PASS: {name} confere com o manifesto "
            f"({entry.get('approved_by', 'sem régua registrada')})."
        )
    return ok


def _video_nodes():
    return sorted(
        path for path in glob.glob("/dev/video*") if path[len("/dev/video") :].isdigit()
    )


_SNAPSHOT_SEQ = itertools.count()


def _save_stamp():
    """Nome único por processo: epoch em ms + contador monotônico.

    Segundos inteiros colidem quando o usuário salva duas vezes no
    mesmo segundo; milissegundos ainda colidem na fila do PngWriter —
    o contador zera a chance sem depender de resolução de relógio.
    """
    return f"{time.time_ns() // 1_000_000}_{next(_SNAPSHOT_SEQ)}"


_GUI_PROBE = (
    "import cv2;"
    "cv2.namedWindow('draw-gui-probe', cv2.WINDOW_NORMAL);"
    "cv2.destroyWindow('draw-gui-probe')"
)


def _gui_backend_compiled():
    """True se a roda instalada tem backend de janela compilado.

    A seção GUI de cv2.getBuildInformation lista QT/GTK/Win32/Cocoa
    com YES/NO; a roda headless sai de fábrica com tudo NO. Formato
    desconhecido não bloqueia — a sonda de display decide em seguida.
    """
    info = cv2.getBuildInformation()
    match = re.search(
        r"^(?P<indent>[ \t]*)GUI:[ \t]*(?P<inline>\S*)", info, re.MULTILINE
    )
    if match is None:
        return True  # formato desconhecido: não bloqueia
    inline = match.group("inline")
    if inline:
        # Valor inline explícito (GUI: NONE, e variantes de build
        # que listam o backend na mesma linha).
        return inline.upper() not in ("NONE", "NO", "OFF", "FALSE", "0")
    # Seção multilinha: entradas mais indentadas que o cabeçalho GUI.
    # Seções irmãs (Media I/O etc.) também são indentadas — o corte é
    # pelo recuo do próprio "GUI:", não por "linha sem espaço".
    base = len(match.group("indent").expandtabs())
    for line in info[match.end() :].splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base:
            break
        if ":" in line and "YES" in line:
            return True
    return False


def _require_gui_backend():
    """Falha cedo e claro quando a janela do Draw nao pode abrir.

    Dois motivos distintos: a roda headless nao compila HighGUI
    (detectado pelo build info, sem abrir janela) e um build COM GUI
    sem display utilizavel — o Qt aborta o processo inteiro nesse
    caso, entao a sonda de janela roda num subprocesso descartavel.
    """
    if not _gui_backend_compiled():
        raise RuntimeError(
            "O OpenCV instalado nao tem backend de janela (HighGUI) "
            "compilado — a roda headless nao abre cv2.imshow. Para "
            "desenhar na tela: pip install 'draw-core[gui]'. Sem tela "
            "(CI/servidor), use as ferramentas de lab/."
        )
    try:
        result = subprocess.run(
            [sys.executable, "-c", _GUI_PROBE],
            timeout=15,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is None or result.returncode != 0:
        raise RuntimeError(
            "O OpenCV tem backend de janela compilado, mas nenhum "
            "display utilizavel (o Qt encerraria o processo ao abrir "
            "a janela). Conecte um display, use X11 forwarding ou "
            "rode as ferramentas de lab/ que nao precisam de tela."
        )


_WINDOW = "Draw"
_window_sized = set()


_mouse = {}


_UI_TICK_SECONDS = 0.016
_HOVER_SECONDS = 0.22
_WORDMARK_SECONDS = 0.6
_wordmark_state = {"progress": 0.0, "origin": None, "time": None, "inside": False}
_button_state = {"progress": [], "pressed": None, "queue": [], "time": None}


def _button_at(x, y):
    for i, (bx, by, bw, bh) in enumerate(view_module.button_rects(_window_size())):
        if bx <= x < bx + bw and by - 3 <= y < by + bh:
            return i
    return None


_CAMERA_MENU_SECONDS = 0.18
_SWITCH_READY_SECONDS = 10.0
_TOGGLE_SECONDS = 0.35
_REACT_SECONDS = 0.9
_camera_menu = {
    "open": False,
    "progress": 0.0,
    "dot": 0.0,
    "rows": [],
    "items": [],
    "current": None,
    "busy": None,
    "notice": None,
    "notice_until": 0.0,
    "request": None,
    "time": None,
    "scroll": 0,
    "cam_shape": (512, 512, 3),
}


def _camera_label(source):
    if isinstance(source, int):
        return f"Camera {source}"
    text = str(source)
    if "://" in text:
        return "Rede: " + text.split("://", 1)[1].split("/")[0][:20]
    return os.path.basename(text)[:22] or text[:22]


def _list_cameras(current):
    """Cameras para o menu: os /dev/video* e a fonte atual, se for outra."""
    items = [
        (_camera_label(index), index)
        for index in (int(node[len("/dev/video") :]) for node in _video_nodes())
    ]
    if current not in [source for _, source in items]:
        items.insert(0, (_camera_label(current), current))
    return items


def _camera_layout():
    count = len(_camera_menu["items"])
    return view_module.camera_menu_layout(
        _window_size(),
        _camera_menu["cam_shape"],
        view_module.blank_sheet().shape,
        count,
        _camera_menu["scroll"],
    )


def _camera_click(x, y):
    """Clique na bolinha ou no menu de cameras. True se o clique foi usado."""
    state = _camera_menu
    layout = _camera_layout()
    if state["open"]:
        step = view_module.scroll_hit(layout, (x, y))
        if step is not None:
            top = layout["count"] - layout["visible"]
            state["scroll"] = max(0, min(top, layout["first"] + step))
            return True
        row = view_module.row_at(layout, (x, y))
        on_dot = view_module.point_in_dot(layout, (x, y))
        state["open"] = False
        row = None if row is None else layout["first"] + row
        if row is not None and row < len(state["items"]):
            source = state["items"][row][1]
            if source != state["current"] and state["busy"] is None:
                state["request"] = source
            return True
        return on_dot
    if view_module.point_in_dot(layout, (x, y)):
        state["items"] = _list_cameras(state["current"])
        state["scroll"] = 0
        state["open"] = True
        _settings["open"] = False
        return True
    return False


def _camera_fx():
    """Estado animado da bolinha e do menu de cameras para o desenho."""
    state = _camera_menu
    now = time.monotonic()
    last, state["time"] = state["time"], now
    dt = 0.0 if last is None else now - last
    pos = _mouse.get("pos")
    layout = _camera_layout()
    step = dt / _CAMERA_MENU_SECONDS
    on_dot = pos is not None and view_module.point_in_dot(layout, pos)
    state["dot"] += max(-step, min(step, (1.0 if on_dot else 0.0) - state["dot"]))
    target = 1.0 if state["open"] else 0.0
    state["progress"] += max(-step, min(step, target - state["progress"]))
    count = len(state["items"])
    if len(state["rows"]) != count:
        state["rows"] = [0.0] * count
    hovered = (
        view_module.row_at(layout, pos) if pos is not None and state["open"] else None
    )
    if hovered is not None:
        hovered += layout["first"]
    for i in range(count):
        goal = 1.0 if hovered == i else 0.0
        state["rows"][i] += max(-step, min(step, goal - state["rows"][i]))
    items = []
    for (label, source), hover in zip(state["items"], state["rows"], strict=True):
        if source == state["busy"]:
            label = "Abrindo..."
        elif source == state["notice"] and now < state["notice_until"]:
            label = "Nao abriu"
        items.append(
            {"label": label, "selected": source == state["current"], "hover": hover}
        )
    first = layout["first"]
    return {
        "dot": state["dot"],
        "opened": state["open"],
        "progress": state["progress"],
        "items": items[first : first + layout["visible"]],
        "count": count,
        "first": first,
    }


_SETTINGS_ITEMS = (
    ("probe", "Coordenadas e RGB"),
    ("latency", "Latencia"),
    ("fps", "FPS"),
    ("fullscreen", "Tela cheia"),
)
_settings = {
    "open": False,
    "progress": 0.0,
    "gear": 0.0,
    "rows": [0.0] * len(_SETTINGS_ITEMS),
    "on_p": [0.0] * len(_SETTINGS_ITEMS),
    "react": [1.0] * len(_SETTINGS_ITEMS),
    "dir": [1] * len(_SETTINGS_ITEMS),
    "time": None,
    "on": {key: False for key, _ in _SETTINGS_ITEMS},
}
_hud = {"captured_at": None, "fps": None}
_fullscreen_applied = [False]


def _settings_layout():
    return view_module.settings_menu_layout(_window_size(), len(_SETTINGS_ITEMS))


def _settings_click(x, y):
    """Clique na engrenagem ou no menu de opcoes. True se foi usado."""
    state = _settings
    layout = _settings_layout()
    on_gear = view_module.point_in_gear(layout, (x, y))
    if state["open"]:
        row = view_module.row_at(layout, (x, y))
        if row is not None and row < len(_SETTINGS_ITEMS):
            key = _SETTINGS_ITEMS[row][0]
            state["on"][key] = not state["on"][key]
            state["dir"][row] = 1 if state["on"][key] else -1
            state["react"][row] = 0.0
            return True
        state["open"] = False
        return on_gear
    if on_gear:
        state["open"] = True
        _camera_menu["open"] = False
        return True
    return False


def _settings_fx():
    """Estado animado da engrenagem e do menu de opcoes para o desenho."""
    state = _settings
    now = time.monotonic()
    last, state["time"] = state["time"], now
    step = 0.0 if last is None else (now - last) / _CAMERA_MENU_SECONDS
    pos = _mouse.get("pos")
    layout = _settings_layout()
    on_gear = pos is not None and view_module.point_in_gear(layout, pos)
    state["gear"] += max(-step, min(step, (1.0 if on_gear else 0.0) - state["gear"]))
    target = 1.0 if state["open"] else 0.0
    state["progress"] += max(-step, min(step, target - state["progress"]))
    hovered = (
        view_module.row_at(layout, pos) if pos is not None and state["open"] else None
    )
    items = []
    for i, (key, label) in enumerate(_SETTINGS_ITEMS):
        goal = 1.0 if hovered == i else 0.0
        state["rows"][i] += max(-step, min(step, goal - state["rows"][i]))
        dt = 0.0 if last is None else now - last
        slide = dt / _TOGGLE_SECONDS
        aim = 1.0 if state["on"][key] else 0.0
        state["on_p"][i] += max(-slide, min(slide, aim - state["on_p"][i]))
        state["react"][i] = min(1.0, state["react"][i] + dt / _REACT_SECONDS)
        items.append(
            {
                "label": label,
                "on": state["on"][key],
                "hover": state["rows"][i],
                "on_p": state["on_p"][i],
                "dir": state["dir"][i],
                "react": state["react"][i],
            }
        )
    return {
        "gear": state["gear"],
        "opened": state["open"],
        "progress": state["progress"],
        "items": items,
    }


def _hud_values():
    """Valores do painel da camera: so o que esta ligado na engrenagem."""
    on = _settings["on"]
    captured_at = _hud["captured_at"]
    latency = None
    if on["latency"] and captured_at is not None:
        latency = (time.monotonic() - captured_at) * 1000.0
    return {
        "probe": _mouse.get("pos") if on["probe"] else None,
        "latency": latency,
        "fps": _hud["fps"] if on["fps"] else None,
    }


_view_map = {"scale": 1.0, "offset": (0, 0)}


def _on_mouse(event, x, y, flags, param):
    scale, (ox, oy) = _view_map["scale"], _view_map["offset"]
    x, y = int((x - ox) / scale), int((y - oy) / scale)
    _mouse["pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONUP and (_settings_click(x, y) or _camera_click(x, y)):
        _button_state["pressed"] = None
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        _button_state["pressed"] = _button_at(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        pressed, _button_state["pressed"] = _button_state["pressed"], None
        if pressed is not None and pressed == _button_at(x, y):
            keys = view_module.footer_buttons()
            _button_state["queue"].append(ord(keys[pressed][2]))


def _button_fx():
    """Progresso de hover (0..1) por botao, suavizado no tempo."""
    count = len(view_module.footer_buttons())
    progress = _button_state["progress"]
    if len(progress) != count:
        progress[:] = [0.0] * count
    now = time.monotonic()
    last = _button_state["time"]
    _button_state["time"] = now
    step = 0.0 if last is None else (now - last) / _HOVER_SECONDS
    pos = _mouse.get("pos")
    hovered = _button_at(*pos) if pos is not None else None
    for i in range(count):
        target = 1.0 if hovered == i else 0.0
        progress[i] += max(-step, min(step, target - progress[i]))
    return list(progress)


def _ui_state():
    """Estado visual da interface fora da camera: pisca, botoes, mouse, janela."""
    return {
        "rec_on": _rec_on(),
        "buttons": _button_fx(),
        "pressed": _button_state["pressed"],
        "size": _window_size(),
        "pos": _mouse.get("pos"),
        "wordmark": _wordmark_fx(),
        "camera": _camera_fx(),
        "settings": _settings_fx(),
        "hud": _hud_values(),
    }


def _ui_signature(ui):
    return (
        ui["rec_on"],
        tuple(round(v, 3) for v in ui["buttons"]),
        ui["pressed"],
        ui["size"],
        ui["pos"],
        round(ui["wordmark"][0], 3),
        round(ui["camera"]["dot"], 3),
        round(ui["camera"]["progress"], 3),
        tuple(round(item["hover"], 3) for item in ui["camera"]["items"]),
        tuple(item["label"] for item in ui["camera"]["items"]),
        round(ui["settings"]["gear"], 3),
        round(ui["settings"]["progress"], 3),
        tuple(
            (round(item["hover"], 3), round(item["on_p"], 3), round(item["react"], 3))
            for item in ui["settings"]["items"]
        ),
        None if ui["hud"]["latency"] is None else int(ui["hud"]["latency"] // 25),
        None if ui["hud"]["fps"] is None else round(ui["hud"]["fps"], 1),
    )


def _wordmark_fx():
    """Progresso do hover no nome DRAW e ponto onde o mouse o tocou."""
    state = _wordmark_state
    now = time.monotonic()
    last, state["time"] = state["time"], now
    step = 0.0 if last is None else (now - last) / _WORDMARK_SECONDS
    pos = _mouse.get("pos")
    inside = False
    if pos is not None:
        bx, by, bw, bh = view_module.wordmark_rect(_window_size())
        inside = bx <= pos[0] < bx + bw and by <= pos[1] < by + bh
    if inside and not state["inside"]:
        state["origin"] = pos
    state["inside"] = inside
    target = 1.0 if inside else 0.0
    state["progress"] += max(-step, min(step, target - state["progress"]))
    return state["progress"], state["origin"]


def _next_key():
    """Tecla do waitKey ou, se nao houver, um clique de botao pendente."""
    key = cv2.waitKey(1) & 0xFF
    queue = _button_state["queue"]
    if key == 0xFF and queue:
        return queue.pop(0)
    return key


def _rec_on():
    """Alterna o ponto de gravacao a cada 1 s."""
    return int(time.monotonic() / 1.0) % 2 == 0


def _window_rect():
    """Largura e altura da area de desenho da janela, ou None sem janela."""
    try:
        _, _, w, h = cv2.getWindowImageRect(_WINDOW)
    except cv2.error:
        return None
    return w, h


def _window_size():
    """Tamanho atual da area de desenho da janela (padrao antes de abrir)."""
    default = (1600, 640)
    rect = _window_rect() if _window_sized else None
    return rect if rect and rect[0] >= 320 and rect[1] >= 160 else default


def _open_window(fullscreen):
    """Cria a janela do Draw: sem barras do Qt, redimensionavel, com mouse."""
    cv2.namedWindow(
        _WINDOW,
        cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO,
    )
    _window_sized.clear()
    _settings["on"]["fullscreen"] = fullscreen
    _fullscreen_applied[0] = False
    cv2.setMouseCallback(_WINDOW, _on_mouse)


def _ui_wait_seconds():
    """Espera por quadro: curta com o mouse na janela, para a interface fluir."""
    return _UI_TICK_SECONDS if _mouse.get("pos") is not None else _FRAME_WAIT_SECONDS


def _apply_fullscreen():
    """Liga ou desliga a tela cheia, que esconde a moldura da janela."""
    want = _settings["on"]["fullscreen"]
    if _fullscreen_applied[0] != want:
        _fullscreen_applied[0] = want
        cv2.setWindowProperty(
            _WINDOW,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if want else cv2.WINDOW_NORMAL,
        )


def _open_ready_source(target, helper_token_file=None, ready_seconds=None, rotation=0):
    """Abre a fonte e so a devolve depois do primeiro quadro; senao levanta.

    A fonte antiga so e trocada por uma que ja entregou quadro: uma
    camera que abre mas nao le deixa a atual no ar. O orcamento de
    ready_seconds conta desde o inicio da abertura (a recuperacao de
    exposicao do open_capture pode gastar varios segundos); a espera
    pelo primeiro quadro usa o que sobrou, com piso de 1 s. A abertura
    em si nao e cancelavel, mas roda numa thread e nao trava a janela.
    """
    started = time.monotonic()
    capture = open_capture(target, helper_token_file=helper_token_file)
    candidate = LatestFrameCapture(capture, rotation=rotation).start()
    budget = _SWITCH_READY_SECONDS if ready_seconds is None else ready_seconds
    remaining = max(1.0, budget - (time.monotonic() - started))
    if candidate.wait_next(0, timeout=remaining) is None:
        candidate.stop(force=False)
        raise RuntimeError("a camera abriu mas nao entregou quadros")
    return candidate


def _waiting_view():
    """Quadro de camera vazio com o aviso, para a interface seguir completa."""
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i, text in enumerate(
        ("Aguardando quadros da fonte", "A leitura esta lenta ou parada.")
    ):
        cv2.putText(
            image,
            text,
            (320, 330 + i * 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0 if i == 0 else 0.7,
            (230, 235, 235) if i == 0 else (150, 156, 154),
            2 if i == 0 else 1,
            cv2.LINE_AA,
        )
    return image


def _show(image):
    """Mostra o quadro na janela sem deixar o Qt reescalar a imagem.

    A vista tem tamanho fixo; maximizada, o Qt a esticaria com filtro
    barato. Aqui o redimensionamento e feito com interpolacao de
    qualidade, mantendo a proporcao e preenchendo o resto com o fundo.
    Na primeira exibicao a janela abre no tamanho nativo da vista.
    """
    _apply_fullscreen()
    ih, iw = image.shape[:2]
    if not _window_sized:
        cv2.resizeWindow(_WINDOW, iw, ih)
        _window_sized.add(True)
    w, h = _window_rect() or (0, 0)
    _view_map.update(scale=1.0, offset=(0, 0))
    if w > 0 and h > 0 and (w, h) != (iw, ih):
        scale = min(w / iw, h / ih)
        nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
        image = cv2.resize(image, (nw, nh), interpolation=interp)
        dx, dy = max(0, w - nw), max(0, h - nh)
        _view_map.update(scale=nw / iw, offset=(dx // 2, dy // 2))
        image = cv2.copyMakeBorder(
            image,
            dy // 2,
            dy - dy // 2,
            dx // 2,
            dx - dx // 2,
            cv2.BORDER_CONSTANT,
            value=view_module.COLOR_BG_STUDIO,
        )
    cv2.imshow(_WINDOW, image)


def _report_format_suggestion(telemetry, suggestion):
    telemetry.event("format_suggestion", **suggestion)
    ratio = suggestion["ratio"]
    ratio_text = "indisponivel" if ratio is None else f"{ratio:.4f}"
    label = suggestion["suggestion"]
    if label == "serie_a":
        label = (
            "serie A (proporcao 0,71) — tamanho nao estimado; "
            "A4 assumido se nada for escolhido"
        )
    elif label is None:
        label = f"abstencao ({suggestion['reason']})"
    print(f"Sugestao de formato: proporcao {ratio_text}; {label}.")


def run(
    source=0,
    save_dir="captures",
    markers_mode=False,
    neural_model=None,
    hand_model=None,
    page_clean_model=None,
    helper_token_file=None,
    fullscreen=False,
    rotation=0,
    pipeline_config=None,
):
    _apply_opencv_thread_cap()
    _require_gui_backend()
    _open_window(fullscreen)

    # Recursos que o finally derruba; None = ainda nao iniciado.
    # Depois do probe de GUI, TODO o setup roda dentro do try: uma
    # falha em qualquer ponto (pasta sem permissao, modelo, camera)
    # desce pelo finally liberando o que ja subiu.
    ink_worker = None
    hand_masker = None
    hand_presence_worker = None
    detection_worker = None
    frame_source = None
    png_writer = None
    telemetry = None
    session_report = None
    try:
        neural_detector = None
        if neural_model is None and not markers_mode and _DEFAULT_SHEET_MODEL.exists():
            neural_model = str(_DEFAULT_SHEET_MODEL)
        if neural_model is not None and not markers_mode:
            neural_detector = NeuralSheetDetector(
                neural_model, verify=manifest_module.verify_model
            )
            neural_detector.load()
            print(
                f"Detector neural ativo: {neural_model}. "
                "O classico cobre os frames em que o neural nao ve folha."
            )
        elif not markers_mode:
            print(
                "Aviso: sem detector neural de folha (modelo nao "
                "encontrado); usando so o detector classico, que nao "
                "enxerga a folha em varias mesas. Baixe os modelos "
                "oficiais pela Release do repositorio."
            )

        page_clean = None
        if not markers_mode:
            page_clean = PageCleanInk(
                page_clean_model, verify=manifest_module.verify_model
            )
            try:
                page_clean.load()
                print(
                    f"Tinta pagina->limpa ativa: {page_clean.model_path} "
                    f"(limiar {page_clean.threshold}). Canvas por uniao "
                    "simples das mascaras."
                )
            except (RuntimeError, FileNotFoundError) as error:
                raise RuntimeError(
                    f"O modelo pagina->limpa nao carregou ({error}); sem "
                    "ele nao ha tinta no caminho novo. Baixe os modelos "
                    "oficiais pela Release do repositorio."
                ) from error

        ink_worker = None
        hand_masker = None
        if page_clean is not None:
            hand_masker = (
                HandMasker(hand_model, verify=manifest_module.verify_model)
                if hand_model_available(hand_model)
                else None
            )
            ink_worker = PageCleanWorker(
                page_clean,
                hand_masker,
                config=None if pipeline_config is None else pipeline_config.ink,
            ).start()
            print(
                "Tinta em thread propria: o laco principal nao espera "
                "o modelo (passadas com cadencia de ~250 ms)."
            )

        clean_hand_presence = None
        hand_presence_worker = None
        if page_clean is not None and not markers_mode:
            clean_hand_presence = make_hand_presence(
                hand_model, verify=manifest_module.verify_model
            )
            if clean_hand_presence is not None:
                print(
                    "Porta de ociosidade pela mao ativa no caminho "
                    "pagina->limpa: sem mao, a tinta sonda ate a saida "
                    "confirmada assentar, descansa sem passadas e a "
                    "geometria da tinta congela."
                )
                # Amostragem fora do laco (T4): o laco submete e colhe.
                hand_presence_worker = HandPresenceWorker(clean_hand_presence).start()

        def detect_sheet(frame):
            return _detect_sheet_combined_traced(frame, neural_detector)

        capture = open_capture(
            parse_source(source), helper_token_file=helper_token_file
        )
        frame_source = LatestFrameCapture(capture, rotation=rotation).start()
        detection_worker = None
        if not markers_mode:
            # coverage=True na trabalhadora: os extras da deteccao
            # (mascara de cobertura e de mao) sempre sao publicados — a
            # mascara de mao alimenta a oclusao semantica mesmo quando
            # a cobertura esta desligada no pipeline.
            detection_worker = SheetDetectionWorker(detect_sheet, coverage=True).start()
        png_writer = PngWriter()
        # Capturas, telemetria e relatórios são privados: a pasta ganha
        # um .gitignore "*" qualquer que seja o --save-dir.
        save_path = ensure_private_dir(save_dir)
        telemetry = telemetry_module.SessionTelemetry(
            save_path / "telemetry.jsonl"
        ).start()
        format_metadata = {
            "paper_format": config.PAPER_FORMAT,
            "paper_orientation": config.PAPER_ORIENTATION,
            "page_grid": [config.PAPER_W, config.PAPER_H],
            "mm_per_px": config.mm_per_px(),
        }
        telemetry.update(**format_metadata)
        telemetry.event("session_open", **format_metadata)
        print(
            f"Formato da sessao: {config.PAPER_FORMAT} ({config.PAPER_ORIENTATION}), "
            f"grade {config.PAPER_W}x{config.PAPER_H}, "
            f"{config.mm_per_px():.6f} mm/px."
        )
        print(
            "Telemetria de sessao ligada: 1 registro/s em "
            f"{save_path / 'telemetry.jsonl'} (arquivo local e rotativo)."
        )

        # Relatorio de sessao: um JSON por corrida com cabecalho
        # (modelos com sha, abertura da camera, porta da mao) e os
        # agregados da sessao, para uma corrida que falha deixar dito
        # o que rodou — a telemetria bruta sozinha nao diz.
        def _model_header(path):
            if path is None:
                return None
            try:
                return {
                    "arquivo": str(path),
                    "sha256": manifest_module.sha256_of(path),
                }
            except OSError:
                return {"arquivo": str(path), "sha256": None}

        session_report = session_report_module.SessionReport(
            save_path / "session-reports",
            {
                "inicio_epoch": time.time(),
                **format_metadata,
                "fonte": str(source),
                "modo_marcadores": bool(markers_mode),
                "modelo_folha": _model_header(
                    neural_model if neural_detector is not None else None
                ),
                "modelo_tinta": _model_header(
                    page_clean.model_path if page_clean is not None else None
                ),
                "tinta_modo": (
                    "page-clean" if page_clean is not None else "marcadores"
                ),
                "mao_disponivel": clean_hand_presence is not None,
                "camera_abertura": camera_module._LAST_OPEN_PROBE,
            },
        ).start()
        print(
            "Relatorio de sessao ligado: resumo desta corrida em "
            f"{session_report.path} (atualizado a cada 15 s e no fim)."
        )
        if markers_mode:
            print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
        else:
            print("Draw v0 rodando. Mostre uma folha para a camera.")
        print(
            "Teclas: s = salvar canvas, f = capturar frame, "
            "g = captura automatica (desligada), q/Esc = sair."
        )

        frames_path = save_path / "frames"
        frames_saved = 0
        auto_capture = False
        loop_frames = 0
        tel_window_start = time.monotonic()
        tel_window_frames = 0
        tel_loop_fps = 0.0
        tel_detect_ms = None
        tel_sheet_present = False

        def save_raw_frame(img):
            frames_path.mkdir(parents=True, exist_ok=True)
            out = frames_path / f"frame_{int(time.time() * 1000)}.png"

            def _saved(path):
                nonlocal frames_saved
                frames_saved += 1
                print(f"Frame salvo em {path} (total: {frames_saved}).")

            png_writer.save(out, img, _saved)

        def handle_capture_key(key, img):
            nonlocal auto_capture
            if key == ord("f"):
                save_raw_frame(img)
            elif key == ord("g"):
                auto_capture = not auto_capture
                if auto_capture:
                    print(
                        "Captura automatica ligada: um frame a cada "
                        f"10 quadros vai para {frames_path}."
                    )
                else:
                    print("Captura automatica desligada.")

        format_suggester = paper_module.FormatSuggester()

        def save_canvas_snapshot(fallback_panel=None):
            """Salva a camada confirmada (nunca a provisoria).

            Com folha em cena, `fallback_panel` e a folha retificada
            quando ainda nao ha tinta confirmada. Sem folha, salva
            so o canvas confirmado retido (canvas_persist) — a folha
            em branco e a provisoria nunca viram arquivo.
            """
            if pipeline is not None and pipeline.clean_mask is not None:
                # O snapshot e um consumidor de canvas: ele salva a
                # camada confirmada, nunca o painel com a tinta
                # provisoria.
                save_image = ink_render.render_canvas(pipeline.clean_mask)
            elif fallback_panel is not None:
                save_image = fallback_panel
            else:
                print("Nada a salvar: sem canvas confirmado.")
                return
            save_path.mkdir(parents=True, exist_ok=True)
            out = save_path / f"canvas_{_save_stamp()}.png"
            png_writer.save(
                out, save_image, lambda path: print(f"Canvas salvo em {path}")
            )

        def save_page_snapshot(prefix, mask):
            save_path.mkdir(parents=True, exist_ok=True)
            snapshot_path = save_path / f"{prefix}_{_save_stamp()}.png"
            png_writer.save(
                snapshot_path,
                ink_render.render_canvas(mask),
                lambda path: print(f"Pagina anterior salva em {path}"),
            )

        def emit_acquisition(record):
            telemetry.event("acquisition", **record)

        pipeline = None
        tracker = None
        if not markers_mode:
            pipeline = Pipeline(
                detection_worker=detection_worker,
                ink_worker=ink_worker,
                hand_presence_worker=hand_presence_worker,
                hand_masker=hand_masker,
                format_suggester=format_suggester,
                on_format_suggestion=lambda suggestion: _report_format_suggestion(
                    telemetry, suggestion
                ),
                on_acquisition=emit_acquisition,
                save_page_snapshot=save_page_snapshot,
                config=pipeline_config,
            )
            tracker = pipeline.tracker

        last_seq = 0
        last_view = None
        frame = None
        # Cache da ultima folha retificada do modo marcadores: sem
        # pipeline de tinta nao ha mascara confirmada, entao o 's'
        # durante um stall so tem o que salvar com este fallback.
        last_marker_warped = None
        # Ultimo quadro da camera ja composto: a interface (pisca, botoes,
        # mouse, janela) segue viva com ele enquanto a camera nao entrega
        # quadro novo, e uma travada da camera fica so dentro da camera.
        ui_inputs = None
        ui_signature = None

        _camera_menu["current"] = parse_source(source)
        seq_offset = 0
        reset_source = False
        switch = {"thread": None, "result": None}

        def start_switch(target):
            _camera_menu["busy"] = target

            def work():
                try:
                    candidate = _open_ready_source(
                        target, helper_token_file, rotation=rotation
                    )
                    switch["result"] = (target, candidate, None)
                except Exception as error:
                    switch["result"] = (target, None, error)

            switch["thread"] = threading.Thread(target=work, daemon=True)
            switch["thread"].start()

        def service_camera_switch():
            nonlocal frame_source, last_seq, seq_offset, reset_source
            target = _camera_menu["request"]
            if target is not None and switch["thread"] is None:
                _camera_menu["request"] = None
                start_switch(target)
            result = switch["result"]
            if result is None:
                return
            switch["result"], switch["thread"] = None, None
            target, candidate, error = result
            _camera_menu["busy"] = None
            if error is not None:
                print(f"Nao consegui abrir {_camera_label(target)}: {error}")
                _camera_menu["notice"] = target
                _camera_menu["notice_until"] = time.monotonic() + 2.5
                return
            old, frame_source = frame_source, candidate
            old.stop(force=False)
            seq_offset += last_seq
            last_seq = 0
            reset_source = True
            _camera_menu["current"] = target
            print(f"Camera trocada para {_camera_label(target)}.")

        def render_ui(status, canvas):
            nonlocal last_view, ui_signature
            ui = _ui_state()
            _camera_menu["cam_shape"] = status.shape
            ui_signature = _ui_signature(ui)
            last_view = view_module.compose_view(
                status,
                canvas,
                size=ui["size"],
                rec_on=ui["rec_on"],
                buttons=ui["buttons"],
                pressed=ui["pressed"],
                wordmark=ui["wordmark"],
                camera_menu=ui["camera"],
                settings_menu=ui["settings"],
                hud=ui["hud"],
            )
            _show(last_view)

        while True:
            service_camera_switch()
            packet = frame_source.wait_next(last_seq, timeout=_ui_wait_seconds())
            if packet is None and not frame_source.dead:
                # Fonte viva sem quadro novo na janela: a leitura pode
                # estar dentro de um stall TCP prolongado. Antes do
                # primeiro quadro, mostra o aviso de espera (a janela ja
                # existe, mas fica vazia); depois dele, a interface
                # segue viva com o ultimo quadro. As teclas sao
                # atendidas sem realimentar o quadro velho no pipeline.
                if ui_inputs is None:
                    ui_inputs = (_waiting_view(), view_module.blank_sheet())
                    render_ui(*ui_inputs)
                elif _ui_signature(_ui_state()) != ui_signature:
                    render_ui(*ui_inputs)
                key = _next_key()
                if key == ord("s"):
                    # Marcadores nao tem mascara confirmada: o salvavel
                    # visivel na tela de espera e a folha retificada.
                    save_canvas_snapshot(last_marker_warped if markers_mode else None)
                if frame is not None:
                    handle_capture_key(key, frame)
                if key in (ord("q"), 27):
                    break
                continue
            if packet is None:
                # A thread de captura declarou a fonte morta (falhas
                # seguidas de leitura ou excecao no read) e wait_next
                # acordou sem quadro. O aviso vai para o console e para
                # a janela, sobre a ultima vista, e fica _DEAD_NOTICE_MS
                # ou ate uma tecla: so o console seria um congelamento
                # silencioso para quem esta olhando a janela.
                print(
                    "A camera abriu, mas parou de entregar quadros. "
                    "Feche outros programas usando a camera e tente de novo."
                )
                _show(view_module.dead_source_view(last_view))
                cv2.waitKey(_DEAD_NOTICE_MS)
                break
            frame = packet.image
            last_seq = packet.sequence
            _hud["captured_at"] = packet.captured_at
            if seq_offset:
                packet = dataclasses.replace(packet, sequence=last_seq + seq_offset)
            if reset_source:
                reset_source = False
                if pipeline is not None:
                    pipeline.reset_source(packet.captured_at)

            loop_frames += 1
            tel_window_frames += 1
            tel_now = time.monotonic()
            if tel_now - tel_window_start >= 1.0:
                tel_loop_fps = round(
                    tel_window_frames / (tel_now - tel_window_start), 2
                )
                _hud["fps"] = tel_loop_fps
                tel_window_start = tel_now
                tel_window_frames = 0
            tel_fields = {
                "loop_fps": tel_loop_fps,
                "detect_ms": tel_detect_ms,
                "sheet_present": tel_sheet_present,
                "state": "sheet" if tel_sheet_present else "no_sheet",
            }
            tel_fields.update(frame_source.stats())
            if ink_worker is not None:
                tel_fields.update(ink_worker.stats())
            if detection_worker is not None:
                tel_fields.update(detection_worker.stats())
            telemetry.update(**tel_fields)
            session_report.update(**tel_fields)
            if auto_capture and loop_frames % 10 == 0:
                save_raw_frame(frame)

            if markers_mode:
                warped, markers = paper_module.warp_paper(frame)
                sheet_present = warped is not None
                tel_sheet_present = sheet_present
                if sheet_present:
                    suggestion = format_suggester.observe(
                        [markers[i] for i in config.MARKER_IDS],
                        frame.shape,
                        sheet_boundary=False,
                    )
                    if suggestion is not None:
                        _report_format_suggestion(telemetry, suggestion)
            else:
                markers = {}
                warped = None
                loop_outcome = pipeline.step(packet)
                tel_detect_ms = pipeline.detect_ms
                sheet_present = pipeline.sheet_present
                tel_sheet_present = sheet_present
                if loop_outcome == INK_WORKER_FAILED:
                    _show(
                        view_module.error_view(
                            last_view,
                            "A tinta pagina->limpa falhou",
                            [
                                "O canvas parou de atualizar.",
                                "Veja a mensagem no terminal e reinicie o app.",
                            ],
                        ),
                    )
                    cv2.waitKey(_DEAD_NOTICE_MS)
                    break

            status = frame.copy()
            if markers_mode:
                for x, y in markers.values():
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
            elif tracker.corners is not None:
                view_corners = (
                    pipeline.display_corners
                    if pipeline.display_corners is not None
                    else tracker.corners
                )
                outline = view_corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(status, [outline], True, (0, 255, 0), 2)
                for x, y in view_corners:
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
            if not sheet_present:
                if markers_mode:
                    last_marker_warped = None  # quadro processado sem folha
                    status_text = f"Mostre a folha-modelo ({len(markers)}/4 marcadores)"
                else:
                    status_text = "Mostre uma folha para a camera"
                cv2.putText(
                    status,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                panel = (
                    pipeline.accumulated_canvas
                    if pipeline is not None and pipeline.accumulated_canvas is not None
                    else view_module.blank_sheet()
                )
                render_ui(status, panel)
                ui_inputs = (status, panel)
                key = _next_key()
                if key == ord("s"):
                    # Sem folha, so o canvas confirmado retido salva.
                    save_canvas_snapshot()
                handle_capture_key(key, frame)
                if key in (ord("q"), 27):
                    break
                continue
            else:
                if not markers_mode:
                    if tracker.verdict == "blank_maybe":
                        status_text = "Folha detectada (baixa confianca)"
                        status_color = (0, 255, 255)
                    else:
                        status_text = "Folha detectada"
                        status_color = (0, 255, 0)
                    cv2.putText(
                        status,
                        status_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        status_color,
                        2,
                    )
                if markers_mode:
                    # A heuristica de tinta do legado nao migrou: no
                    # modo de marcadores o painel mostra a folha
                    # retificada como esta — e e ela que o 's' salva,
                    # inclusive num stall (cache retido do ultimo warp).
                    canvas = warped
                    last_marker_warped = warped
                elif pipeline.accumulated_canvas is not None:
                    canvas = pipeline.accumulated_canvas
                else:
                    # Tinta assincrona: ate a trabalhadora publicar a
                    # primeira mascara desta folha, o painel e a folha
                    # em branco. Rodar o modelo aqui, sincrono, era o
                    # congelamento que a trabalhadora existe para tirar.
                    canvas = view_module.blank_sheet()
                render_ui(status, canvas)
                ui_inputs = (status, canvas)

                key = _next_key()
                if key == ord("s"):
                    save_canvas_snapshot(canvas)
                handle_capture_key(key, frame)
                if key in (ord("q"), 27):
                    break
                continue

    finally:
        # Epilogo sempre roda: captura liberada, trabalhadora de
        # tinta parada, fila de PNG drenada e janela destruida
        # mesmo quando o laco morre por excecao. Sem o finally, um
        # erro no meio do laco vazava a camera aberta e a thread
        # daemon ficava lendo o dispositivo ate o processo sair.
        # Os `is not None` cobrem falha de setup antes de cada
        # recurso subir: o try abrange do primeiro load ao ultimo loop.
        if frame_source is not None:
            frame_source.stop()
        if detection_worker is not None:
            detection_worker.stop()
            # Aquisicoes que nenhum quadro do rastreador consumiu
            # (latest-wins ou fim de sessao) tambem viram evento.
            if telemetry is not None:
                for trace in detection_worker.take_traces():
                    telemetry.event(
                        "acquisition",
                        **{
                            **trace,
                            "frame_index": None,
                            "consumed_by_tracker": False,
                            "tracker_before": None,
                            "tracker_after": None,
                            "tracker_event": None,
                        },
                    )
        if hand_presence_worker is not None:
            hand_presence_worker.stop()
        if ink_worker is not None:
            ink_worker.stop()
        if hand_masker is not None:
            # O masker da trabalhadora fecha aqui, depois do join:
            # determinístico, nunca pelo __del__ do MediaPipe.
            hand_masker.close()
        if telemetry is not None:
            telemetry.close()
        if session_report is not None:
            session_report.close()
        if png_writer is not None:
            png_writer.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Build headless ou GUI indisponivel: nao ha o que destruir.
            pass


def main():
    parser = argparse.ArgumentParser(description="Draw, papel para o digital")
    parser.add_argument(
        "--source",
        default="0",
        help="indice da webcam local (0, 1...), URL do stream "
        "(ex.: http://192.168.0.10:8080/video), gravacao .drawcap, "
        "stream tcp://host[:porta] do ajudante ou camera do Windows "
        "via hvcam:// (hvcam+tcp://host fora do WSL). Padrao: 0",
    )
    parser.add_argument(
        "--helper-token-file",
        default=None,
        metavar="ARQUIVO",
        help="arquivo com o token do ajudante Windows para "
        "--source tcp:// (bridge/PROTOCOL.md). Sem ele, fontes "
        "tcp:// sao recusadas no handshake",
    )
    parser.add_argument(
        "--markers",
        action="store_true",
        help="usa a folha-modelo impressa com marcadores ArUco",
    )
    parser.add_argument(
        "--neural",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="detector neural (ONNX) como primario, com o classico de "
        "reserva. Sem a flag, o padrao e models/draw_seg_v3_dyn.onnx "
        "quando o arquivo existe",
    )
    parser.add_argument(
        "--page-clean",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="modelo pagina->limpa (ONNX), a fonte da tinta. Sem "
        "caminho, usa models/page_clean_unet.onnx",
    )
    parser.add_argument(
        "--hand-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.task",
        help="modelo de mao (MediaPipe Hand Landmarker) para a porta "
        "de ociosidade e a oclusao no voto de tinta. Sem caminho, "
        "usa models/hand_landmarker.task quando o arquivo existe",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="valida modelos e camera e sai, sem abrir janela",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="abre em tela cheia, sem a moldura da janela (liga e desliga "
        "tambem na engrenagem)",
    )
    parser.add_argument("--save-dir", default="captures")
    parser.add_argument(
        "--rotate",
        type=int,
        choices=camera_module.INPUT_ROTATIONS,
        default=0,
        help=(
            "gira o quadro da camera na captura (180 = webcam invertida na "
            "mesa); preview, quad e canvas giram juntos"
        ),
    )
    parser.add_argument(
        "--orientation",
        choices=config.ORIENTATIONS,
        default="retrato",
        help=(
            "orientacao da grade da pagina: paisagem para caderno aberto ou "
            "folha deitada (evita espremer o conteudo largo na grade em pe)"
        ),
    )
    parser.add_argument(
        "--ink-gain",
        action="store_true",
        help=(
            "ganho digital na pagina retificada antes da tinta: clareia o "
            "papel escuro quando a camera esta no teto de exposicao de 30 fps "
            "(PipelineConfig.ink_auto_gain)"
        ),
    )
    parser.add_argument(
        "--ink-vote",
        type=float,
        default=None,
        metavar="NIVEL",
        help=(
            "voto estrito da tinta ao vivo (InkConfig.strict_vote), 0,634-0,99: "
            "derruba a pauta que o page_clean v2 deixa passar, ao custo de "
            "afinar o traco (medido: 0,85 e o ponto equilibrado). So com a "
            "imagem bem exposta: com papel escuro quebra as letras"
        ),
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="carrega modelos divergentes ou fora do manifesto com "
        "aviso, em vez de recusar (equivale a DRAW_MODEL_STRICT=0)",
    )
    parser.add_argument(
        "--format",
        choices=list(config.FORMATS),
        default=None,
        help=(
            "formato da folha (padrao: a4); em retrato, a menos que combinado "
            "com --orientation paisagem"
        ),
    )
    args = parser.parse_args()
    if args.allow_unverified:
        manifest_module.allow_unverified()
        print(
            "AVISO: --allow-unverified ativo; modelos divergentes ou fora do "
            "manifesto serao carregados apenas com aviso."
        )
    try:
        if args.markers and args.orientation != "retrato":
            parser.error(
                "--orientation paisagem nao combina com --markers (folha-modelo A4 em pe)"
            )
        config.apply_format(args.format or "a4", args.orientation)
    except ValueError as error:
        parser.error(str(error))
    # Validado ANTES do --check: o mesmo nivel invalido impede a sessao ao
    # vivo, entao o preflight nao pode sair com sucesso (review do PR #53).
    ink_config = InkConfig()
    if args.ink_vote is not None:
        if args.markers:
            parser.error(
                "--ink-vote nao combina com --markers (sem tinta no modo marcadores)"
            )
        try:
            ink_config = InkConfig.strict_vote(args.ink_vote)
        except ValueError as error:
            parser.error(str(error))
    pipeline_config = PipelineConfig(ink=ink_config, ink_auto_gain=args.ink_gain)
    if args.check:
        parsed = parse_source(args.source)
        sys.exit(run_check(parsed, helper_token_file=args.helper_token_file))
    neural_model = args.neural
    page_clean_model = args.page_clean
    hand_model = args.hand_model
    if args.markers and (
        neural_model is not None
        or page_clean_model is not None
        or hand_model is not None
    ):
        parser.error("--neural/--page-clean/--hand-model nao combinam com --markers")
    if args.markers and args.ink_gain:
        parser.error(
            "--ink-gain nao combina com --markers (sem tinta no modo marcadores)"
        )
    if neural_model is not None and not neural_model:
        parser.error("--neural sem caminho: passe o arquivo .onnx")
    if page_clean_model is not None and not page_clean_model:
        page_clean_model = None  # padrao: models/page_clean_unet.onnx
    if hand_model is not None and not hand_model:
        hand_model = None  # padrao: models/hand_landmarker.task
    try:
        run(
            source=args.source,
            save_dir=args.save_dir,
            markers_mode=args.markers,
            neural_model=neural_model,
            hand_model=hand_model,
            page_clean_model=page_clean_model,
            helper_token_file=args.helper_token_file,
            fullscreen=args.fullscreen,
            rotation=args.rotate,
            pipeline_config=pipeline_config,
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
