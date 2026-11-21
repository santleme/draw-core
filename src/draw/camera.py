"""Captura: thread de quadros, abertura da fonte e escrita de PNG.

A thread de captura guarda so o quadro mais novo (a fila nao
acumula: quem consome pega o mais recente e os intermediarios
morrem). A abertura local e supervisionada — configura o formato,
sonda a exposicao e tenta de novo antes de declarar a fonte morta;
o desfecho da ultima abertura fica em _LAST_OPEN_PROBE, para o
cabecalho do relatorio de sessao. O escritor de PNG drena a fila
em thread propria para o salvamento nunca travar o laco.

Extraido de draw/app.py na modularizacao de 2026-11-05 (repositório
legado).
"""

import glob
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

from . import hvcam as hvcam_module
from .contracts import FramePacket
from .hvcam import HvcamCapture, is_hvcam_source
from .protocol import HelperCapture, is_helper_source

# Rotacoes de entrada suportadas (graus): 180 cobre a webcam invertida
# sobre a mesa; 90/270 mudariam a proporcao do quadro e nao tem uso.
INPUT_ROTATIONS = (0, 180)

# Limite de falhas seguidas de leitura que declara a fonte morta.
# Era a contagem do laco principal (15 read() falhos seguidos e o app
# avisava e saia); com a leitura numa thread propria, a contagem mora
# na thread e o desfecho visivel e o mesmo.
_CAPTURE_MAX_FAILURES = 15


# Pausa entre tentativas de leitura no caminho de falha. Sem ela, um
# driver que devolve False na hora faz a thread girar uma CPU inteira
# enquanto a camera nao volta; o laco antigo girava sem dormir.
_READ_RETRY_SECONDS = 0.005


# Teto da espera pela drenagem da fila de PNG na saida do app.
_PNG_DRAIN_SECONDS = 5.0


class LatestFrameCapture:
    """Captura em thread separada que guarda so o quadro mais novo.

    Uma thread le da captura sem parar e publica cada quadro como
    contracts.FramePacket: imagem, instante de captura (monotonico,
    tirado na thread logo que o read() retorna), numero de sequencia
    e a contagem de quadros
    descartados (publicados e substituidos antes de alguem consumir).
    Quem consome espera por sequencia nova numa Condition: nao gira
    em vazio e nunca reprocessa o mesmo quadro. Os quadros que o
    driver empilha enquanto o laco processa nao viram fila aqui: o
    consumidor pega sempre o mais novo, e e isso que tira o teto da
    latencia de captura.

    Falha de leitura tem um equivalente declarado do comportamento
    antigo. O laco antigo contava as falhas seguidas de read() e, com
    _CAPTURE_MAX_FAILURES, imprimia o aviso e saia; abaixo do limite,
    girava sem imshow e sem waitKey, com a janela congelada. Aqui a
    thread conta as falhas seguidas e, no limite, marca a fonte como
    morta e acorda quem espera; quem espera recebe None e o laco
    imprime o mesmo aviso e sai. Abaixo do limite, nenhum quadro novo
    chega e o consumidor fica bloqueado na espera: janela congelada e
    teclas sem resposta, como antes. Uma leitura boa zera a contagem.
    No modo drenagem, grab() falso e retrieve() falso contam na
    mesma contagem, com o mesmo limite e o mesmo desfecho.
    """

    def __init__(self, capture, rotation=0):
        if rotation not in INPUT_ROTATIONS:
            raise ValueError(
                f"rotacao de entrada {rotation!r} invalida; aceitas: {INPUT_ROTATIONS}"
            )
        self._capture = capture
        # Webcam montada de cabeca para baixo na mesa: o quadro gira
        # aqui, na origem, antes de qualquer consumidor — preview,
        # deteccao, quad, warp, tinta e o 's' de salvar enxergam o
        # mesmo espaco ja orientado (nada de coordenada desalinhada).
        self._rotation = rotation
        self._frame_rotation = rotation
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._pump, name="draw-capture", daemon=True
        )
        self._frame = None
        self._seq = 0
        self._taken_seq = 0
        self._captured_at = 0.0
        self._dropped = 0
        self._failures = 0
        self._dead = False
        self._stopped = False
        self._released = False
        self._release_lock = threading.Lock()
        self._error = None
        # Metadados do emissor (HelperCapture): o último id/ts publicado
        # no pacote e o contador de descarte na fonte, quando a fonte
        # o expõe (saltos de frame_id contados no adaptador, sobre
        # todos os registros — decode ruim não vira descarte falso).
        self._source_frame_id = None
        self._source_ts_ms = None
        self._has_source_dropped = hasattr(self._capture, "source_dropped")

    def start(self):
        self._thread.start()
        return self

    @property
    def rotation(self):
        with self._condition:
            return self._rotation

    def set_rotation(self, rotation):
        """Atualiza a rotacao de entrada (0 ou 180) em tempo de execucao."""
        if rotation not in INPUT_ROTATIONS:
            raise ValueError(
                f"rotacao de entrada {rotation!r} invalida; aceitas: {INPUT_ROTATIONS}"
            )
        with self._condition:
            self._rotation = rotation

    def _pump(self):
        # Um driver que levanta (cv2 levanta em varios modos de falha)
        # matava a thread com traceback na saida de erro e um
        # PytestUnhandledThreadExceptionWarning na suite. O desfecho
        # visivel tem de ser o mesmo das falhas de read(): fonte
        # morta, aviso na janela e saida limpa.
        try:
            self._pump_loop()
        except Exception as error:  # noqa: BLE001 - vai para o aviso
            with self._condition:
                self._error = error
        finally:
            with self._condition:
                if not self._stopped:
                    self._dead = True
                self._condition.notify_all()
            self._release_capture()

    def _pump_loop(self):
        while True:
            with self._condition:
                if self._stopped:
                    return
            ok, frame = self._capture.read()
            if ok and frame is not None:
                # Carimbo da thread de captura, tirado assim que o
                # read() retorna — o consumidor ve o instante de
                # publicacao, nao o do proprio relogio. Antes da
                # rotacao: o HUD de latencia mede da leitura em
                # diante, e o tempo de girar o quadro e latencia.
                captured_at = time.monotonic()
                with self._condition:
                    rotation = self._rotation
                if rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                # Identidade do emissor quando a fonte a expõe
                # (HelperCapture): entra no pacote e alimenta a
                # contagem de descarte na origem.
                source_frame_id = getattr(self._capture, "last_frame_id", None)
                source_ts_ms = getattr(self._capture, "last_capture_ts_ms", None)
                with self._condition:
                    if self._stopped:
                        return
                    if self._seq > self._taken_seq:
                        self._dropped += 1
                    self._seq += 1
                    self._frame = frame
                    self._frame_rotation = rotation
                    self._captured_at = captured_at
                    self._source_frame_id = source_frame_id
                    self._source_ts_ms = source_ts_ms
                    self._failures = 0
                    self._condition.notify_all()
            else:
                with self._condition:
                    if self._stopped:
                        return
                    self._failures += 1
                    if self._failures >= _CAPTURE_MAX_FAILURES:
                        self._dead = True
                        self._condition.notify_all()
                        return
                time.sleep(_READ_RETRY_SECONDS)

    def wait_next(self, after_seq, timeout=None):
        """Espera um quadro com sequencia maior que after_seq.

        Devolve um contracts.FramePacket (image, captured_at,
        sequence) em sucesso. Devolve None quando a fonte morreu,
        quando stop() foi chamado ou quando o timeout vence sem
        quadro novo; quem chama distingue por dead.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    (self._seq > after_seq and self._frame_rotation == self._rotation)
                    or self._dead
                    or self._stopped
                ),
                timeout=timeout,
            )
            if self._seq > after_seq and self._frame_rotation == self._rotation:
                self._taken_seq = self._seq
                return FramePacket(
                    image=self._frame,
                    captured_at=self._captured_at,
                    sequence=self._seq,
                    source_frame_id=self._source_frame_id,
                    source_ts_ms=self._source_ts_ms,
                )
            return None

    @property
    def dead(self):
        with self._condition:
            return self._dead

    @property
    def dropped(self):
        with self._condition:
            return self._dropped

    def stats(self):
        """Contadores para a telemetria de sessao (1 registro/s).

        capture_dropped: publicados substituidos antes do consumo
        (a fila do slot). capture_source_dropped: quadros que o
        emissor descartou na fonte (saltos de frame_id), presente so
        quando a fonte publica identidade propria.
        """
        with self._condition:
            fields = {
                "capture_seq": self._seq,
                "capture_dropped": self._dropped,
            }
            if self._has_source_dropped:
                fields["capture_source_dropped"] = self._capture.source_dropped
            return fields

    @property
    def failures(self):
        with self._condition:
            return self._failures

    def _release_capture(self):
        """Libera a captura uma unica vez, de qualquer thread."""
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._capture.release()

    def stop(self, force=True):
        """Para a thread e libera a captura.

        Com force (padrao), o release() tenta destravar um read()
        pendurado no driver. Se mesmo assim a thread nao voltar, o
        join tem teto e a thread e daemon: a saida do app nao depende
        dela. Com force=False (troca de fonte com o app rodando), o
        release nunca corre durante um read(): a propria thread da
        captura libera o driver quando o read() voltar, e quem chama
        nao espera por ela.
        """
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if not force:
            return
        self._release_capture()
        self._thread.join(timeout=2.0)


class PngWriter:
    """Escritora de PNG em thread unica, para o imwrite sair do laco.

    Medido no laboratorio (draw-lab/bottleneck-2026-10-30/loop): o
    imwrite sincrono de um frame cru custa 50 a 71 ms (zlib) e o do
    canvas, 10 a 12 ms; dentro do laco, cada salvamento era um
    engasgo inteiro. save() copia a imagem e enfileira; a escritora
    grava em ordem e roda o callback de confirmacao depois da
    escrita, que e onde moram o contador e o print de "salvo".
    close() enfileira o fim da fila e espera a drenagem com teto de
    _PNG_DRAIN_SECONDS: a saida nao trava num disco patologico, e o
    que nao drenar a tempo fica por conta da thread daemon.
    """

    def __init__(self):
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._work, name="draw-png-writer", daemon=True
        )
        self._thread.start()

    def save(self, path, image, on_done=None):
        self._queue.put((Path(path), image.copy(), on_done))

    def _work(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            path, image, on_done = item
            # imwrite devolve False (e pode levantar) em disco cheio,
            # caminho inválido ou codec ausente. Sem este guarda, a
            # thread da escritora morria e todo save seguinte ficava
            # na fila para sempre, com o close() esperando o teto de
            # _PNG_DRAIN_SECONDS.
            try:
                written = cv2.imwrite(str(path), image)
            except Exception as error:  # noqa: BLE001 - vira aviso
                written = False
                error_text = str(error)
            else:
                error_text = "imwrite devolveu False"
            if not written:
                print(f"Aviso: nao consegui salvar {path} ({error_text}).")
                continue
            if on_done is not None:
                try:
                    on_done(path)
                except Exception as error:  # noqa: BLE001 - vira aviso
                    print(f"Aviso: confirmacao de {path} falhou ({error}).")

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=_PNG_DRAIN_SECONDS)


def parse_source(value):
    """'0'/'1' viram indice de webcam local; qualquer outra coisa e URL/arquivo."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def is_wsl():
    """Detecta WSL pelo nome do kernel em /proc (ex.: '...-microsoft-standard-WSL2')."""
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _configure_local_capture(capture):
    # MJPG comprimido e 720p: o formato cru padrao (YUYV em 1080p)
    # estoura a banda do USB virtual do usbip no WSL e a entrega de
    # quadros trava com timeout de select() no V4L2.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Fila do driver no minimo e fps pedido explicitamente
    # (auditoria de 2026-11-02): na mesa do Lucas a fila V4L2 mediu
    # profundidade 4 — ate ~200-300 ms de quadro velho quando o
    # consumo atrasa, somados a toda a latencia do pipeline. A
    # thread de captura ja e latest-frame do lado do app; isto pede
    # o mesmo do lado do driver. As duas propriedades sao pedidos:
    # backend que nao suporta ignora o set() em silencio, e o
    # formato/resolucao acima nao mudam. O efeito so e mensuravel
    # com a camera dele; sem ela, o pedido e o comportamento seguro.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FPS, 30)


# Supervisor de abertura da camera local (medido na mesa do Lucas,
# 2026-11-02): a webcam barata via usbipd tem loteria de exposicao por
# abertura -- as vezes abre estourada (brilho medio ~149 contra os
# ~106-121 saudaveis) e fica assim a sessao inteira: o logit de
# presenca do detector neural vai a negativo, o veto de presenca
# devolve (None, None) e o app nao reconhece folha nenhuma
# (reproduzido na fiacao real: 293/293 vereditos None em 30 s;
# minutos depois, mesma cena e mesmo codigo, 128/131 paper).
# Travar a exposicao por propriedade nao funciona nela (o set() de
# CAP_PROP_AUTO_EXPOSURE devolve False via V4L2/usbipd) e
# pre-processar o quadro estourado nao recupera a pagina clipada
# (gama, escala de brilho e CLAHE medidos piores que o cru nos
# quadros estourados salvos). O que resta: transmitir ate a exposicao
# convergir e, se a abertura ficar estourada, reabrir -- cada
# reabertura e um bilhete novo da loteria (4 aberturas seguidas
# medidas assentadas; a convergencia natural levou ~1,9 s numa
# abertura e ~4 s noutra).
_EXPOSURE_SETTLED_MEAN = 135.0


_EXPOSURE_BLOWN_MEAN = 140.0


_EXPOSURE_PROBE_SECONDS = 6.0


_EXPOSURE_MAX_ATTEMPTS = 3


# Desfecho da ultima abertura local, para o cabecalho do relatorio
# de sessao: sem ele, uma corrida que nasce morta nao deixa dito
# nem como a camera abriu. None = fonte sem supervisor (URL).
_LAST_OPEN_PROBE = None
_LAST_OPEN_DELIVERED = False
_LAST_PROBE_DELIVERED = False


def _record_open_probe(mean, attempts, outcome, delivered=None):
    global _LAST_OPEN_PROBE, _LAST_OPEN_DELIVERED
    _LAST_OPEN_PROBE = {
        "brilho_medio": None if mean is None else round(float(mean), 1),
        "tentativas": attempts,
        "desfecho": outcome,
    }
    _LAST_OPEN_DELIVERED = (
        bool(mean is not None) if delivered is None else bool(delivered)
    )


def _probe_exposure(capture, seconds=_EXPOSURE_PROBE_SECONDS, now=time.monotonic):
    """Transmite por ate `seconds` e devolve o brilho medio recente.

    Fecha cedo quando 3 quadros seguidos ficam abaixo de
    _EXPOSURE_SETTLED_MEAN (exposicao assentada). Devolve None se a
    leitura falhar: quem chama trata como "sem evidencia de estouro"
    e o caminho existente de fonte morta cuida do resto.
    """
    global _LAST_PROBE_DELIVERED
    _LAST_PROBE_DELIVERED = False
    deadline = now() + seconds
    recent = []
    while now() < deadline:
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        _LAST_PROBE_DELIVERED = True
        recent.append(float(frame.mean()))
        del recent[:-3]
        if len(recent) == 3 and sum(recent) / 3 < _EXPOSURE_SETTLED_MEAN:
            return sum(recent) / 3
    if not recent:
        return None
    return sum(recent) / len(recent)


def _supervise_local_open(source, capture, capture_factory):
    """Reabre a camera local se a abertura ficar estourada.

    Devolve a captura a usar: a original assentada, uma reaberta
    assentada, ou a ultima tentativa (com aviso) se todas estourarem.
    """
    global _LAST_PROBE_DELIVERED
    attempt = 1
    while True:
        _LAST_PROBE_DELIVERED = False
        mean = _probe_exposure(capture)
        delivered = _LAST_PROBE_DELIVERED or (mean is not None)
        if mean is None or mean < _EXPOSURE_BLOWN_MEAN:
            if mean is not None and attempt > 1:
                print(
                    f"Camera reaberta: exposicao assentada (brilho medio {mean:.0f})."
                )
            _record_open_probe(
                mean,
                attempt,
                "sem_evidencia" if mean is None else "assentada",
                delivered=delivered,
            )
            return capture
        if attempt >= _EXPOSURE_MAX_ATTEMPTS:
            print(
                "Aviso: camera segue com exposicao alta (brilho medio "
                f"{mean:.0f}) apos {attempt} aberturas; o reconhecimento "
                "de folha pode falhar ate a exposicao assentar."
            )
            _record_open_probe(mean, attempt, "estourada_seguiu", delivered=delivered)
            return capture
        print(
            "Camera abriu com exposicao alta (brilho medio "
            f"{mean:.0f}); reabrindo (tentativa "
            f"{attempt + 1}/{_EXPOSURE_MAX_ATTEMPTS})..."
        )
        capture.release()
        attempt += 1
        capture = capture_factory(source)
        if not capture.isOpened():
            _record_open_probe(mean, attempt, "reabertura_falhou", delivered=False)
            return capture
        _configure_local_capture(capture)


def _local_open_hint(source):
    """Diagnostico por porta: nomeia a primeira condicao que impede a abertura."""
    node = f"/dev/video{source}"
    if not sys.platform.startswith("linux"):
        return (
            "Nenhuma webcam local respondeu nesse indice. "
            "Confira se a webcam esta conectada e livre."
        )
    if not os.path.exists(node):
        hint = f"A camera nao esta conectada a este sistema: {node} nao existe. "
        if is_wsl():
            hint += (
                "No Windows, rode 'hvcam-host.exe --reverse' (santleme/hvcam) "
                "ou anexe a camera ao WSL com: bash scripts/setup_camera_wsl.sh. "
                "A alternativa e uma camera via rede: draw "
                "--source http://<ip-da-camera>:8080/video."
            )
        else:
            hint += "Confira se a webcam esta conectada a este computador."
        return hint
    if not os.access(node, os.R_OK | os.W_OK):
        return (
            f"{node} existe, mas este processo nao tem permissao de "
            "leitura e escrita nele. A porta que falhou e a permissao. "
            "Correcao definitiva: sudo usermod -aG video $USER "
            "(vale nas proximas sessoes). "
            'Correcao nesta sessao: sg video -c "python3 main.py".'
        )
    return (
        f"{node} existe e este processo tem permissao nele, mas a "
        "abertura falhou. Outro programa provavelmente esta usando a "
        "camera. Feche esse programa e tente de novo."
    )


_WSL_HVCAM_CANDIDATES = ("hvcam://", "hvcam+reverse://")


def open_wsl_hvcam_fallback(source, *, timeout_ms=None, connect_timeout_ms=None):
    """Tenta abrir a camera do Windows via hvcam quando WSL nao tem /dev/video*.

    Compartilhado entre `open_capture` e `run_check` para que `draw --check`
    valide exatamente o mesmo fallback que `draw` usa na abertura ao vivo.
    Se nenhum hvcam-host.exe estiver rodando no Windows, provisiona e sobe
    `hvcam-host.exe --reverse` automaticamente via interop do WSL, escolhendo
    a primeira webcam disponivel quando `source == 0`.
    Devolve `(source_url, HvcamCapture)` ou `(None, None)`.
    """
    if (
        not isinstance(source, int)
        or source < 0
        or not sys.platform.startswith("linux")
        or not is_wsl()
    ):
        return None, None
    if os.path.exists("/dev/video0") or _video_nodes():
        return None, None
    print("WSL sem /dev/video*: conectando na camera do Windows via hvcam...")
    kwargs = {}
    if timeout_ms is not None:
        kwargs["timeout_ms"] = timeout_ms
    if connect_timeout_ms is not None:
        kwargs["connect_timeout_ms"] = connect_timeout_ms

    # 1. Tenta hvcam:// primeiro (instantaneo se o usuario ja registrou o servico).
    if source == 0:
        hv_cap = HvcamCapture("hvcam://", **kwargs)
        if hv_cap.isOpened():
            print("Camera do Windows conectada via hvcam://.")
            return "hvcam://", hv_cap
        hv_cap.release()

    # 2. Sobe hvcam-host.exe --reverse automaticamente (baixa se ausente e
    #    escolhe a primeira webcam funcional quando source == 0).
    if HvcamCapture is hvcam_module.HvcamCapture:
        managed_kwargs = {}
        if timeout_ms is not None:
            managed_kwargs["timeout_ms"] = timeout_ms
        if connect_timeout_ms is not None:
            managed_kwargs["connect_timeout_ms"] = connect_timeout_ms
        try:
            hv_cap, dev_idx, dev_name = hvcam_module.open_managed_reverse(
                device=source,
                try_all=(source == 0),
                **managed_kwargs,
            )
        except RuntimeError as error:
            print(f"Aviso: inicio automatico do hvcam-host.exe falhou ({error}).")
            hv_cap = None
            dev_idx = None
            dev_name = None
        if hv_cap is not None:
            print(
                f"Camera do Windows conectada via hvcam-host.exe "
                f"([{dev_idx}] {dev_name})."
            )
            return "hvcam+reverse://", hv_cap

    # 3. Fallback manual: escuta hvcam+reverse:// apenas quando o indice padrao
    #    (0) foi pedido. Para um --source N explicito, um host manual no Windows
    #    nao informa qual dispositivo esta servindo e poderia entregar outra camera.
    if source != 0:
        return None, None
    print("Aguardando hvcam-host.exe --reverse conectar no Windows (ate 35 s)...")
    hv_cap = HvcamCapture("hvcam+reverse://", **kwargs)
    if hv_cap.isOpened():
        print("Camera do Windows conectada via hvcam+reverse://.")
        return "hvcam+reverse://", hv_cap
    hv_cap.release()
    return None, None


_RECOVERY_WINDOW_SECONDS = 0.35


def _confirm_stream_alive(
    capture,
    *,
    window_s=_RECOVERY_WINDOW_SECONDS,
    retry_s=_READ_RETRY_SECONDS,
    now=time.monotonic,
    sleep=time.sleep,
):
    """Confirma que uma captura que falhou no warm-up volta a entregar quadro.

    Usa uma janela curta com pausa entre falhas (como o LatestFrameCapture)
    para nao descartar uma camera que da falhas imediatas durante 20-50 ms
    no aquecimento, mantendo teto baixo para cameras realmente travadas.
    """
    deadline = now() + window_s
    while now() < deadline:
        ok, frame = capture.read()
        if ok and frame is not None:
            return True
        remaining = deadline - now()
        if remaining <= 0:
            return False
        sleep(min(retry_s, remaining))
    return False


def _try_other_local_cameras(source, capture_factory):
    """Quando o indice padrao (0) falha, tenta outras webcams locais disponiveis.

    So aceita um candidato depois de confirmar que ele efetivamente entregou
    e continua entregando quadros (se houve falha no warm-up, exige retomada
    antes de aceitar; camera travada e liberada para testar a proxima).
    """
    if source != 0:
        return None
    if sys.platform.startswith("linux"):
        candidates = [
            int(node[len("/dev/video") :])
            for node in _video_nodes()
            if int(node[len("/dev/video") :]) != 0
        ]
    else:
        candidates = [1, 2, 3, 4]
    for idx in candidates:
        cap = capture_factory(idx)
        if cap.isOpened():
            _configure_local_capture(cap)
            cap = _supervise_local_open(idx, cap, capture_factory)
            probe = _LAST_OPEN_PROBE
            settled = probe is not None and probe.get("brilho_medio") is not None
            alive = settled or (_LAST_OPEN_DELIVERED and _confirm_stream_alive(cap))
            if cap.isOpened() and alive:
                print(f"Camera {source} indisponivel; usando camera local {idx}.")
                return cap
        cap.release()
    return None


def open_capture(source, capture_factory=None, helper_token_file=None):
    global _LAST_OPEN_PROBE
    _LAST_OPEN_PROBE = None
    if is_hvcam_source(source):
        # Fonte hvcam (santleme/hvcam): webcam do lado Windows por
        # vsock (hvcam://) ou TCP (hvcam+tcp://host). A conexao falha
        # cedo quando o hvcam-host nao esta no ar.
        capture = HvcamCapture(source)
        if capture.isOpened():
            return capture
        detail = capture.open_error
        capture.release()
        if detail is not None:
            raise RuntimeError(f"Nao consegui abrir a fonte hvcam: {detail}")
        raise RuntimeError(
            f"Nao consegui abrir a fonte hvcam: {source!r}. Confira se "
            "o hvcam-host.exe esta rodando no Windows e, fora do WSL, "
            "use --source hvcam+tcp://<host>:<porta>."
        )
    if is_helper_source(source):
        # Fonte do ajudante Windows (bridge/PROTOCOL.md): arquivo
        # .drawcap gravado ou stream tcp:// ao vivo. AuthenticationError
        # sobe direto — o operador precisa ver que o token faltou ou
        # foi recusado, nao um "nao consegui abrir" generico.
        capture = HelperCapture(source, token_file=helper_token_file)
        if capture.isOpened():
            return capture
        capture.release()
        raise RuntimeError(
            f"Nao consegui abrir a fonte do ajudante: {source!r}. "
            "Confira se o arquivo .drawcap existe e e legivel, ou se o "
            "ajudante esta escutando na porta TCP (o token vem de "
            "--helper-token-file; veja bridge/PROTOCOL.md)."
        )
    if capture_factory is None:
        capture_factory = cv2.VideoCapture
    default_factory = capture_factory is cv2.VideoCapture
    capture = capture_factory(source)
    if capture.isOpened():
        if isinstance(source, int):
            _configure_local_capture(capture)
            capture = _supervise_local_open(source, capture, capture_factory)
            if not capture.isOpened():
                # A reabertura de recuperacao falhou: a supervisao
                # devolve a captura fechada, e o desfecho aqui e o
                # mesmo erro de abertura de sempre.
                capture.release()
            else:
                return capture
        else:
            return capture
    hint = f"Nao consegui abrir a fonte de video: {source!r}. "
    if isinstance(source, int):
        if default_factory:
            other = _try_other_local_cameras(source, capture_factory)
            if other is not None:
                return other
            _candidate, hv_cap = open_wsl_hvcam_fallback(source)
            if hv_cap is not None:
                return hv_cap
        hint += _local_open_hint(source)
    else:
        hint += (
            "Confira se a URL esta certa, se a camera esta transmitindo e "
            "se este computador alcanca o endereco dela na rede."
        )
    raise RuntimeError(hint)


def _video_nodes():
    return sorted(
        path for path in glob.glob("/dev/video*") if path[len("/dev/video") :].isdigit()
    )
