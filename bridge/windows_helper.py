# ============================================================
#  CAPTURA IMPLEMENTADA (cv2/DirectShow); INJECAO AINDA ESBOCO.
#
#  Estado de cada metade, conferido na primeira sessao real:
#  - Rede (handshake, accept, bind, timestamp): coberta por
#    tests/test_bridge.py com sockets reais de loopback.
#  - Captura (capture_frames): implementada via cv2.VideoCapture
#    DirectShow + imencode, exercida em webcam real no Windows.
#  - Injecao de cursor (SendInput/ctypes): NUNCA dirigiu uma
#    sessao real. So a forma esta testada — conferir cada funcao
#    antes de confiar nas coordenadas.
#
#  Regras do projeto que este arquivo respeita por desenho:
#  - NENHUM ML no Windows. Este ajudante captura e injeta;
#    todo o processamento e todo o ML ficam no WSL.
#  - Injecao: somente biblioteca padrao (ctypes). Captura: cv2
#    (importado tarde — sem cv2, o modulo importa e injeta normal).
# ============================================================
"""Esboço do ajudante Draw no Windows (ver bridge/PROTOCOL.md).

Duas metades:

1. Injeção de cursor via SendInput (ctypes), DPI-aware, com
   coordenadas absolutas normalizadas 0..65535 sobre o desktop
   virtual inteiro. É a metade esboçável sem hardware: só depende
   de estruturas Win32 documentadas.

2. Laço captura -> JPEG -> socket conforme o PROTOCOL.md. A
   captura usa cv2.VideoCapture (DirectShow) no modo nativo da
   camera — negociar FOURCC/resolucao em tempo de execucao trava
   drivers UVC reais, entao --width/--height sao opt-in e FOURCC
   nao se expoe — e cv2.imencode faz o JPEG do frame. O carimbo
   sai na entrega do frame ao socket (mede a entrega, não a
   exposição do sensor; a diferença entra no relatorio quando o
   passo de latencia rodar).

Uso pretendido (no Windows, quando testado):
  set DRAW_HELPER_TOKEN=<token compartilhado com o WSL>
  python windows_helper.py --port 8765 [--host <IP do vEthernet (WSL)>]

Segurança (PROTOCOL.md, "Authentication handshake"): o bind padrão é
127.0.0.1; escutar em todas as interfaces exige --allow-all-interfaces
e emite aviso. Toda conexão precisa apresentar o token antes do
primeiro frame; sem token configurado, o ajudante gera um e o mostra
uma única vez no console.
"""

import argparse
import hmac
import ipaddress
import os
import queue
import secrets
import selectors
import socket
import struct
import sys
import threading
import time

# --- Constantes do protocolo (espelham bridge/PROTOCOL.md) ---

MAGIC = b"DRAW"
VERSION = 1
HEADER = struct.Struct("<4sHHQqI")  # 28 bytes, little-endian
DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"

HELLO_MAGIC = b"DRAH"
HELLO_VERSION = 1
HELLO = struct.Struct("<4sHH")  # 8 bytes, little-endian
ACCEPT = b"DROK"
TOKEN_ENV = "DRAW_HELPER_TOKEN"
TOKEN_MIN_LEN = 16
TOKEN_MAX_LEN = 256
HANDSHAKE_TIMEOUT_S = 5.0

# --- Constantes Win32 (documentadas; conferir na sessao real) ---

# SetProcessDpiAwarenessContext: valor -4 =
# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2.
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
# shcore.SetProcessDpiAwareness: 2 = PROCESS_PER_MONITOR_DPI_AWARE.
PROCESS_PER_MONITOR_DPI_AWARE = 2

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


def _require_windows():
    if sys.platform != "win32":
        raise SystemExit(
            "Este esboço só roda no Windows. No WSL, quem consome o "
            "fluxo é tools/helper_receiver.py."
        )


def _load_user32():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = [
        wintypes.UINT,
        ctypes.POINTER(_input_structs()[0]),
        ctypes.c_int,
    ]
    user32.SendInput.restype = wintypes.UINT
    return user32


_STRUCTS_CACHE = None


def _input_structs():
    """Define (uma única vez) as estruturas INPUT/MOUSEINPUT do SendInput.

    O cache importa: o ctypes distingue classes definidas em chamadas
    diferentes, e o argtypes do SendInput precisa ser a MESMA classe
    INPUT que _send_mouse instancia.
    """
    global _STRUCTS_CACHE
    if _STRUCTS_CACHE is not None:
        return _STRUCTS_CACHE
    import ctypes
    from ctypes import wintypes

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    class _INPUTunion(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", wintypes.DWORD),
            ("u", _INPUTunion),
        ]

    _STRUCTS_CACHE = (INPUT, MOUSEINPUT)
    return _STRUCTS_CACHE


def set_dpi_awareness():
    """Declara o processo per-monitor DPI-aware (v2, com recuos).

    Sem isso, o Windows escala as coordenadas de processos nao
    cientes e o cursor cai deslocado em monitores com escala
    diferente de 100%. Ordem: API moderna (Win10 1703+), depois
    shcore, depois a legada. Retorna o nome da rota que aceitou.
    """
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    try:
        ok = user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        )
        if ok:
            return "SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)"
    except AttributeError:
        pass
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        hr = shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
        if hr == 0:
            return "SetProcessDpiAwareness(PER_MONITOR_DPI_AWARE)"
    except (AttributeError, OSError):
        pass
    if user32.SetProcessDPIAware():
        return "SetProcessDPIAware (legado)"
    return "nenhuma rota aceitou; coordenadas podem sair escaladas"


def _send_mouse(flags, nx=0, ny=0):
    """Envia um evento de mouse absoluto normalizado (0..65535)."""
    import ctypes

    INPUT, _MOUSEINPUT = _input_structs()
    user32 = _load_user32()
    event = INPUT()
    event.type = INPUT_MOUSE
    event.u.mi.dx = int(nx)
    event.u.mi.dy = int(ny)
    event.u.mi.mouseData = 0
    event.u.mi.dwFlags = flags
    event.u.mi.time = 0
    event.u.mi.dwExtraInfo = None
    sent = user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(
            f"SendInput enviou {sent} de 1 evento (erro {ctypes.get_last_error()})"
        )


def move_cursor_normalized(nx, ny):
    """Move o cursor para (nx, ny) em coordenadas normalizadas.

    nx, ny em 0..65535 sobre o DESKTOP VIRTUAL inteiro (todos os
    monitores), gracas a MOUSEEVENTF_VIRTUALDESK. Quem chama
    converte pixel -> normalizado uma unica vez, na borda.
    """
    flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    _send_mouse(flags, nx, ny)


def pixel_to_normalized(px, py):
    """Converte um pixel do desktop virtual para 0..65535.

    Usa as metricas do desktop virtual (origem pode ser negativa
    com monitor a esquerda/acima do principal). Sem VIRTUALDESK na
    leitura das metricas, a conversao erra em setups multimonitor.
    """
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    origin_x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    origin_y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    nx = round((px - origin_x) * 65535 / max(width - 1, 1))
    ny = round((py - origin_y) * 65535 / max(height - 1, 1))
    return (
        min(max(nx, 0), 65535),
        min(max(ny, 0), 65535),
    )


def click(down=True):
    """Pressiona ou solta o botao esquerdo na posicao atual."""
    _send_mouse(MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP)


# --- Captura: cv2/DirectShow ---


class CaptureError(RuntimeError):
    """Falha da camera/pipeline de captura.

    Derruba so o atendimento daquele cliente — o ajudante residente
    volta ao accept e tenta de novo na proxima conexao. Subclasse de
    RuntimeError para quem so quer o tipo generico.
    """


STALL_TIMEOUT_S = 5.0


def _enqueue_put(out, item, stop):
    """Enfileira respeitando backpressure; sai cedo se o consumidor morreu.

    O protocolo e push — cada frame produzido precisa chegar ao fio
    (frame pulado = furo na gravacao .drawcap), entao nada se
    descarta: fila cheia segura o produtor, que retoma quando o
    consumidor drena. O recheco do stop impede um consumidor morto
    de prender a thread.
    """
    while not stop.is_set():
        try:
            out.put(item, timeout=0.2)
            return
        except queue.Full:
            continue


def _camera_jpeg_producer(index, width, height, quality, out, stop):
    """Corpo da thread produtora: dona do VideoCapture, enfileira JPEGs.

    A captura roda fora da thread consumidora de proposito: um
    cap.read() do DirectShow que nunca retorna so encalha ESTE
    daemon — nunca o laco de accept do ajudante residente.
    """
    cap = None
    try:
        import cv2  # tardio: opcional; so a captura precisa dele

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            _enqueue_put(
                out, CaptureError(f"camera {index} indisponivel ou em uso"), stop
            )
            return
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                _enqueue_put(
                    out, CaptureError(f"leitura da camera {index} falhou"), stop
                )
                return
            ok, jpeg = cv2.imencode(".jpg", frame, (cv2.IMWRITE_JPEG_QUALITY, quality))
            if ok:
                _enqueue_put(out, jpeg.tobytes(), stop)
    except Exception as exc:  # noqa: BLE001 - o consumidor precisa do erro real
        # Sem isso, uma excecao do produtor (ex.: imencode) morria com
        # a thread e o consumidor reportava stall — o erro real so
        # existia no traceback da thread.
        _enqueue_put(
            out, CaptureError(f"captura na camera {index} falhou: {exc}"), stop
        )
    finally:
        if cap is not None:
            cap.release()


def capture_frames(
    index=0,
    width=None,
    height=None,
    quality=85,
    stall_timeout=STALL_TIMEOUT_S,
    _producer=_camera_jpeg_producer,
):
    """Produz (jpeg_bytes) por frame capturado via cv2/DirectShow.

    O VideoCapture vive numa thread produtora daemon alimentando uma
    fila pequena com backpressure (fila cheia segura o produtor —
    ver _enqueue_put).
    O gerador consome com prazo: `stall_timeout` segundos sem frame
    nenhum = camera travada numa chamada que nunca retorna, e isso
    vira CaptureError como qualquer outra falha — so o cliente cai,
    o ajudante volta ao accept. Foi o modo de falha visto em
    hardware: read() do DirectShow pode bloquear para sempre, e sem
    watchdog o helper inteiro congelava (cliente seguinte nem
    completava o handshake).

    O carimbo sai na entrega (quem envia, em `_handle_client`), nao
    na exposicao do sensor — ressalva declarada do cabecalho. `cv2`
    e dependencia so desta metade: importa-se tarde para a injecao
    de cursor (stdlib pura) continuar utilizavel sem ele.

    width/height sao None por padrao e NAO se negocia FOURCC:
    numa webcam real (DirectShow, 2026) cap.set(FOURCC=MJPG,
    1280x720) travou dentro da chamada de driver — processo
    impossivel de matar e dispositivo refem ate reboot. Pedir
    formato ao driver so e seguro quando o operador confirmou
    suporte naquele hardware; por isso --width/--height sao opt-in
    no CLI e o FOURCC nao se expoe de jeito nenhum.

    `_producer`/`stall_timeout` existem para os testes injetarem um
    produtor falso e um prazo curto, sem cv2 nem camera.
    """
    pending = queue.Queue(maxsize=2)
    stop = threading.Event()
    thread = threading.Thread(
        target=_producer,
        args=(index, width, height, quality, pending, stop),
        daemon=True,
    )
    thread.start()
    try:
        while True:
            try:
                item = pending.get(timeout=stall_timeout)
            except queue.Empty as empty:
                raise CaptureError(
                    f"camera {index} parou de entregar frames "
                    f"(>{stall_timeout:g}s sem leitura)"
                ) from empty
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
        # Em produtor saudavel o join e imediato; num read()
        # encalhado o daemon vaza em vez de segurar o encerramento —
        # a camera pode ficar refem ate o processo inteiro sair
        # (limite do cv2: nao ha como abortar um read() bloqueado).
        thread.join(timeout=1.0)


def resolve_bind_host(host=DEFAULT_HOST, allow_all_interfaces=False):
    """Endereço de bind: loopback por padrão, wildcard só com opt-in.

    Um IP específico (ex.: o do adaptador vEthernet (WSL) no modo NAT)
    é aceito sem flag, pois não expõe as demais interfaces. "0.0.0.0",
    "::" ou vazio exigem allow_all_interfaces=True e geram aviso.
    """
    host = (host or "").strip()
    wildcard = host in ("", "0.0.0.0", "::")
    if not wildcard:
        try:
            wildcard = ipaddress.ip_address(host).is_unspecified
        except ValueError:
            import socket

            try:
                infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            except (OSError, UnicodeError):
                infos = []
            for info in infos:
                try:
                    addr = ipaddress.ip_address(info[4][0])
                except ValueError:
                    continue
                mapped = getattr(addr, "ipv4_mapped", None)
                if addr.is_unspecified or (
                    mapped is not None and mapped.is_unspecified
                ):
                    wildcard = True
                    break
    if wildcard:
        if not allow_all_interfaces:
            raise SystemExit(
                f"Recusado: bind em {host or '0.0.0.0'!r} expoe o ajudante a "
                "toda a rede. Use 127.0.0.1 (WSL espelhado) ou o IP do "
                "adaptador vEthernet (WSL); para expor de proposito, passe "
                "--allow-all-interfaces."
            )
        print(
            "AVISO DE SEGURANCA: escutando em todas as interfaces. Qualquer "
            "maquina que alcance esta porta pode tentar o handshake; o canal "
            "nao tem TLS. Restrinja a porta no firewall do Windows.",
            file=sys.stderr,
        )
        return host or "0.0.0.0"
    return host


def resolve_token(token=None, token_file=None):
    """Token como bytes: argumento, --token-file, ambiente ou gerado."""
    if token is None and token_file:
        with open(token_file, encoding="utf-8") as handle:
            token = handle.read().strip()
    if token is None:
        token = os.environ.get(TOKEN_ENV) or None
    if token is None:
        token = secrets.token_urlsafe(32)
        print(
            f"Token gerado para esta sessao (defina {TOKEN_ENV} igual no WSL):\n"
            f"  {token}"
        )
    if isinstance(token, str):
        token = token.encode("utf-8")
    if not TOKEN_MIN_LEN <= len(token) <= TOKEN_MAX_LEN:
        raise SystemExit(
            f"token do ajudante precisa ter {TOKEN_MIN_LEN}..{TOKEN_MAX_LEN} bytes"
        )
    return token


def _recv_exact(conn, size, deadline=None, clock=time.monotonic):
    data = bytearray()
    while len(data) < size:
        if deadline is not None:
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("prazo total do handshake excedido")
            conn.settimeout(remaining)
        chunk = conn.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def authenticate_client(conn, token, timeout=HANDSHAKE_TIMEOUT_S, clock=time.monotonic):
    """Lê HELLO + token do cliente; True e ACCEPT se o token confere.

    Falha (magic, versão, tamanho, token errado, timeout) devolve
    False sem responder nada; quem chamou fecha a conexão. O timeout
    é um prazo TOTAL monotônico para o handshake inteiro — um par
    lento que manda 1 byte antes de cada recv não pode segurar o
    accept serial além dele.
    """
    previous = conn.gettimeout()
    deadline = clock() + timeout
    try:
        hello = _recv_exact(conn, HELLO.size, deadline, clock)
        if hello is None:
            return False
        magic, version, length = HELLO.unpack(hello)
        if magic != HELLO_MAGIC or version != HELLO_VERSION:
            return False
        if not TOKEN_MIN_LEN <= length <= TOKEN_MAX_LEN:
            return False
        offered = _recv_exact(conn, length, deadline, clock)
        if offered is None or not hmac.compare_digest(offered, token):
            return False
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        conn.settimeout(remaining)
        conn.sendall(ACCEPT)
        return True
    except OSError:
        return False
    finally:
        conn.settimeout(previous)


def _handle_client(conn, frames, start_ns, clock=time.perf_counter_ns):
    """Envia frames para um cliente autenticado; devolve o total enviado.

    Um OSError/ConnectionError no envio (o WSL desconectou) propaga
    para quem chamou, que fecha a conexao e volta a escutar.
    """
    frame_id = 0
    for jpeg in frames:
        capture_ts_us = (clock() - start_ns) // 1000
        header = HEADER.pack(MAGIC, VERSION, 0, frame_id, capture_ts_us, len(jpeg))
        conn.sendall(header + jpeg)
        frame_id += 1
    return frame_id


def _accept_loop(
    server,
    token,
    make_frames,
    start_ns,
    clock=time.perf_counter_ns,
    log=print,
    stop=None,
):
    """Laco residente: aceita, autentica e serve um cliente por vez.

    O ajudante e residente (PROTOCOL.md, "Transporte"): por padrao so
    sai por Ctrl+C/sinal. Falha de handshake fecha a conexao; queda do
    cliente no meio do fluxo fecha e volta ao accept. A captura e
    reaberta por cliente (make_frames() a cada conexao) — nenhum
    estado de captura e compartilhado entre clientes. O `stop`
    opcional (threading.Event) da uma saida cooperativa, checada
    entre accepts (e acorda imediatamente via socketpair) — usada por
    testes e por um desligamento limpo.
    """
    if stop is not None and stop.is_set():
        return
    prev_timeout = server.gettimeout()
    server.setblocking(False)
    wake_r = wake_w = watcher = None
    wake_lock = threading.Lock()
    wake_closed = False
    if stop is not None:
        wake_r, wake_w = socket.socketpair()
        wake_w.setblocking(False)
        wake_r.setblocking(False)

        def _watch():
            stop.wait()
            with wake_lock:
                if not wake_closed:
                    try:
                        wake_w.send(b"\x00")
                    except OSError:
                        pass

        watcher = threading.Thread(
            target=_watch, name="helper-accept-wake", daemon=True
        )
        try:
            watcher.start()
        except BaseException:
            for sock in (wake_w, wake_r):
                try:
                    sock.close()
                except OSError:
                    pass
            raise
    try:
        with selectors.DefaultSelector() as sel:
            try:
                sel.register(server, selectors.EVENT_READ)
                if wake_r is not None:
                    sel.register(wake_r, selectors.EVENT_READ)
            except (OSError, ValueError) as exc:
                if stop is None or not stop.is_set():
                    log(f"Accept interrompido: {exc}")
                return
            while True:
                if stop is not None and stop.is_set():
                    return
                try:
                    events = sel.select(timeout=0.5)
                except (OSError, ValueError) as exc:
                    if stop is None or not stop.is_set():
                        log(f"Accept interrompido: {exc}")
                    return
                ready = {key.fileobj for key, _ in events}
                if stop is not None and (stop.is_set() or wake_r in ready):
                    return
                if server not in ready:
                    continue
                try:
                    conn, addr = server.accept()
                except BlockingIOError:
                    continue
                except OSError:
                    return
                # No Windows (Winsock), socket aceito herda modo nao-bloqueante.
                conn.setblocking(True)
                if not authenticate_client(conn, token):
                    log(f"Conexao de {addr[0]} recusada: handshake/token invalido.")
                    conn.close()
                    continue
                frames = None
                try:
                    # Dentro do try de proposito: uma factory eagerciada que
                    # levanta CaptureError/OSError ao abrir a camera precisa
                    # cair na mesma recuperacao das falhas de iteracao —
                    # so a sessao morre, o helper volta ao accept.
                    frames = make_frames()
                    sent = _handle_client(conn, frames, start_ns, clock)
                    log(f"Cliente {addr[0]} desconectou apos {sent} frames.")
                except CaptureError as exc:
                    log(f"Captura falhou para {addr[0]} ({exc}); voltando a escutar.")
                except (OSError, ConnectionError):
                    log(f"Conexao com {addr[0]} caiu; voltando a escutar.")
                finally:
                    # Sem isso, um OSError no sendall abandonava o gerador de
                    # captura — o finally dele (stop da thread produtora +
                    # release do VideoCapture) so rodava no GC, e a camera
                    # podia ficar refem ate la.
                    close = getattr(frames, "close", None)
                    if close is not None:
                        close()
                    conn.close()
    finally:
        try:
            server.settimeout(prev_timeout)
        except OSError:
            pass
        if wake_w is not None:
            with wake_lock:
                wake_closed = True
                try:
                    wake_w.close()
                except OSError:
                    pass
            try:
                wake_r.close()
            except OSError:
                pass
            if stop.is_set():
                watcher.join(timeout=1.0)


def serve(
    port=DEFAULT_PORT,
    host=DEFAULT_HOST,
    token=None,
    allow_all_interfaces=False,
    frames=None,
    make_frames=None,
    clock=time.perf_counter_ns,
    stop=None,
):
    """Laco principal: escuta, autentica o WSL, envia frames.

    O ajudante escuta e o WSL conecta (PROTOCOL.md, "Transporte").
    Conexoes sem o token sao fechadas antes de qualquer frame.
    frame_id comeca em 0 a cada sessao; a epoca do timestamp e o
    inicio do processo (start_ns abaixo). O laco e residente: cada
    cliente caido/recusado volta ao accept; so um sinal encerra.
    frames/make_frames/clock/stop existem para os testes (e a linha
    de comando) injetarem fonte, fabrica de captura, relogio e uma
    saida cooperativa.
    """
    bind_host = resolve_bind_host(host, allow_all_interfaces)
    token = resolve_token(token)
    set_dpi_awareness()
    start_ns = clock()
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    server = socket.socket(family, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_host, port))
    server.listen(1)
    print(f"Ajudante Draw escutando em {bind_host}:{port}.")
    if make_frames is None:
        make_frames = (lambda: frames) if frames is not None else capture_frames
    _accept_loop(server, token, make_frames, start_ns, clock, stop=stop)


def main():
    _require_windows()
    parser = argparse.ArgumentParser(
        description="Ajudante Draw no Windows (captura cv2; injecao esboco)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="endereco de bind (padrao 127.0.0.1; no WSL NAT, o IP do "
        "adaptador vEthernet (WSL))",
    )
    parser.add_argument(
        "--allow-all-interfaces",
        action="store_true",
        help="permite bind em 0.0.0.0/:: (expoe a porta a rede; com aviso)",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help=f"arquivo com o token compartilhado (padrao: ${TOKEN_ENV})",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="indice do dispositivo de captura"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="largura pedida ao driver (opt-in; ver capture_frames)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="altura pedida ao driver (opt-in; ver capture_frames)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85, metavar="0-100")
    args = parser.parse_args()
    serve(
        args.port,
        host=args.host,
        token=resolve_token(token_file=args.token_file),
        allow_all_interfaces=args.allow_all_interfaces,
        make_frames=lambda: capture_frames(
            args.camera, args.width, args.height, args.jpeg_quality
        ),
    )


if __name__ == "__main__":
    main()
