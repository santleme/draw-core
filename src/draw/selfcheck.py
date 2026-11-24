"""Preflight da producao: --check valida GUI, modelos e camera, porta a porta.

`run_check` e o doctor do `python -m draw --check`: imprime PASS/FAIL/AVISO
por etapa e devolve 0 ou 1. `_require_gui_backend` tambem guarda o `run()`:
o Qt aborta o processo inteiro sem display utilizavel, entao a janela so
abre depois da sonda de subprocesso aqui.
"""

import os
import re
import subprocess
import sys
import time

import cv2

from . import camera as camera_module
from .camera import _video_nodes, is_wsl
from .hvcam import HvcamCapture, is_hvcam_source
from .infra import manifest as manifest_module
from .page_clean import resolve_threshold
from .protocol import HelperCapture, ProtocolError, is_helper_source

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
        if source >= 0 and is_wsl():
            candidate, hv_cap = camera_module.open_wsl_hvcam_fallback(
                source, timeout_ms=5000
            )
            if hv_cap is not None:
                ok, frame = hv_cap.read()
                hv_cap.release()
                if ok and frame is not None:
                    print(
                        f"PASS: quadro lido da fonte hvcam ({candidate}). "
                        f"Formato (altura, largura, canais): {frame.shape}."
                    )
                    return 0 if all_ok else 1
                print(f"FAIL: a fonte {candidate} abriu, mas nenhum quadro chegou.")
                return 1
        print("FAIL: nenhum /dev/video* existe neste sistema.")
        if is_wsl():
            print(
                "  Correcao: no Windows rode 'hvcam-host.exe --reverse' "
                "(santleme/hvcam) ou anexe a camera com 'usbipd bind "
                "--busid <BUS>' (admin) + 'usbipd attach --wsl "
                "--busid <BUS>' (BUSID sai de 'usbipd list'). "
                "A alternativa e uma camera via rede com --source."
            )
        else:
            print("  Correcao: conecte uma webcam a este computador.")
        return 1
    print(f"PASS: dispositivos de video encontrados: {', '.join(nodes)}.")

    node = f"/dev/video{source}"
    if node not in nodes:
        if source == 0:
            for alt_node in nodes:
                alt_idx = int(alt_node[len("/dev/video") :])
                if os.access(alt_node, os.R_OK | os.W_OK) and _check_camera_open(
                    alt_idx, alt_node
                ):
                    return 0 if all_ok else 1
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
    models_ok = all_ok
    if access:
        print(f"PASS: este processo le e escreve em {node}.{group_note}")
        if _check_camera_open(source, node):
            return 0 if models_ok else 1

    if source == 0:
        for alt_node in nodes:
            if alt_node == node:
                continue
            alt_idx = int(alt_node[len("/dev/video") :])
            if os.access(alt_node, os.R_OK | os.W_OK) and _check_camera_open(
                alt_idx, alt_node
            ):
                return 0 if models_ok else 1

    if not access:
        print(
            f"FAIL: este processo nao tem permissao de leitura e "
            f"escrita em {node}.{group_note}"
        )
        print(
            "  Correcao definitiva: sudo usermod -aG video $USER "
            "(vale nas proximas sessoes)."
        )
        print('  Correcao nesta sessao: sg video -c "python -m draw".')
    return 1


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
    try:
        while time.time() < deadline:
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
                break
            time.sleep(0.1)
    except cv2.error as error:
        print(f"FAIL: a leitura de {label} quebrou com erro do OpenCV ({error}).")
        return False
    finally:
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
    try:
        while time.time() < deadline:
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
                break
            time.sleep(0.1)
    except cv2.error as error:
        print(f"FAIL: a leitura da fonte de rede quebrou ({error}).")
        return 0
    finally:
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
    e AVISO com DRAW_MODEL_STRICT=0. Se algum modelo registrado estiver
    ausente em models/, tenta baixa-lo da Release oficial antes de
    conferir.
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
            try:
                manifest_module.ensure_models(
                    [name],
                    models_dir=manifest_module.DEFAULT_MODELS_DIR,
                    manifest_path=manifest_path,
                )
            except RuntimeError as error:
                print(
                    f"AVISO: {name} esta no manifesto e nao esta em models/ "
                    f"({error}). Baixe os modelos oficiais pela Release do repositorio."
                )
                continue
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
