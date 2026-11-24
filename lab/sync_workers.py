"""Adaptadores síncronos compartilhados das trabalhadoras do laço.

replay_lab, sim_writer, parity_e2e e eval_capture dirigem o
draw.pipeline sem threads: a mesma decisão da trabalhadora real roda
no submit e o poll devolve o resultado mais novo — o contrato
submit/poll dos workers de detecção e presença.

O driver de paridade do legado (parity_legacy_driver.py) NÃO usa
este módulo: roda num subprocesso com PYTHONPATH do repo legado e
fica auto-contido de propósito. A ordenação de diretório dele
espelha ``frame_sort_key`` inline (mesma chave, duplicada lá), então
os dois lados leem o mesmo quadro no mesmo índice — inclusive sem
zero-padding ou além do quadro 9999, quando a captura f%04d já não
alinha mais com a ordem léxica.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from lab.corpus.clip import IMG_EXTS, frame_sort_key


def iter_frames(source, limit):
    """Frames de um diretório de imagens ou de um vídeo.

    Diretório: a ordem é a numérica do sufixo do nome (f2 antes de
    f10) — a lexica os invertia e o replay consumia os quadros fora
    de sequência."""
    path = Path(source)
    if path.is_dir():
        files = sorted(
            (p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS),
            key=frame_sort_key,
        )
        for i, f in enumerate(files):
            if i >= limit:
                break
            frame = cv2.imread(str(f))
            if frame is not None:
                yield i, frame
        return
    capture = cv2.VideoCapture(str(path))
    try:
        index = 0
        while index < limit:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        capture.release()


class StepClock:
    """Relógio de cena: quem dirige o laço atribui `t`.

    Atribua `indice / fps` (divisão): `indice * (1 / fps)` pode cair um
    ulp abaixo do limiar de confirmação do tracker (0,6 s) e deixar o
    candidato pendente no fim de um clipe curto.
    """

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class SyncDetector:
    """Detecção combinada real (mesma decisão de draw.app), síncrona.

    Contrato do SheetDetectionWorker sem thread: submit roda a
    detecção na hora; poll devolve o resultado mais novo.
    `keep_history` guarda `traces`/`seq_frames` (só o replay lê); sem
    ele, capturas longas não acumulam rastro por quadro.
    """

    def __init__(self, detector, detect_ms=1.0, keep_history=False):
        self._detector = detector
        self._detect_ms = detect_ms
        self._keep_history = keep_history
        self.result = None
        self.sequence = 0
        self.error = None
        self.traces = []
        self.seq_frames = []
        self.frame_hint = None

    def submit(self, frame, captured_at, frame_index=None):
        from draw.app import _detect_sheet_combined_traced
        from draw.contracts import DetectionResult

        outcome = _detect_sheet_combined_traced(frame, self._detector)
        corners, verdict, trace = outcome[0], outcome[1], outcome[2]
        extras = outcome[3] if len(outcome) > 3 else None
        self.sequence += 1
        self.result = DetectionResult(
            sequence=self.sequence,
            corners=corners,
            verdict=verdict,
            captured_at=captured_at,
            detect_ms=self._detect_ms,
            coverage_mask=(extras.get("coverage_mask") if extras else None),
            hand_mask=(extras.get("hand_mask") if extras else None),
        )
        if self._keep_history:
            self.traces.append(trace)
            self.seq_frames.append(self.frame_hint)

    def poll(self, after_seq):
        if self.result is not None and self.result.sequence > after_seq:
            return self.result
        return None

    def take_traces(self, upto_result_seq=None):
        return []


class SyncPresence:
    """Presença de mão síncrona — mesmo protocolo da worker real."""

    def __init__(self, presence, strict=False):
        self.presence = presence
        self.strict = strict
        self.result = None
        self.sequence = 0
        self.error = None
        self.real_samples = 0

    def submit(self, frame, captured_at, *, source_id):
        from draw.contracts import HandPresenceResult

        try:
            sample = self.presence.sample(
                frame, captured_at * 1000.0, source_id=source_id
            )
        except Exception as error:
            if self.strict:
                # Paridade: amostra perdida não pode virar "mão medida".
                raise
            # Como a worker do app: o laço degrada para sinal indisponível.
            self.error = error
            return
        self.sequence += 1
        self.real_samples += int(sample.present is not None)
        self.result = HandPresenceResult(self.sequence, captured_at, source_id, sample)

    def poll(self, after_seq):
        if self.result is not None and self.result.sequence > after_seq:
            return self.result
        return None


# ------------------------------------------------------------------
# Montagem compartilhada do laço síncrono
# ------------------------------------------------------------------

_NO_HAND = object()
"""Sentinela do "sem pilha de mão" de build_sync_pipeline.

Distingue o default (replay/sim/bench: nenhum objeto de mão é
criado) de ``hand_model=None``, que resolve o modelo convencional
como o app: masker e presença podem nascer dele."""


@dataclass(frozen=True, slots=True)
class SyncBundle:
    """O laço dirigido montado, com os recursos vivos que ele segura.

    ``close()`` fecha masker e presença (os landmarkers MediaPipe) de
    forma idempotente — é o finally de cada driver, no lugar do
    __del__ que vazava landmarker em vídeo longo. Detecção e tinta
    não guardam recurso a fechar.
    """

    clock: StepClock
    detection_worker: object
    presence_worker: object
    ink_worker: object
    pipeline: object
    page_clean: object
    hand_masker: object
    presence: object
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def close(self):
        """Fecha masker e presença; a segunda chamada não faz nada."""
        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        if self.hand_masker is not None:
            self.hand_masker.close()
        if self.presence is not None:
            self.presence.close()


def build_sync_pipeline(
    *,
    verify,
    config,
    sheet_model=None,
    ink_model=None,
    hand_model=_NO_HAND,
    detector=None,
    page_clean=None,
    detection_worker=None,
    detect_ms=1.0,
    keep_history=False,
    presence_strict=False,
):
    """Monta o Pipeline + as trabalhadoras síncronas dos drivers do lab.

    É a fiação comum de eval_capture, parity_e2e, replay_lab e
    sim_writer — os laços por quadro seguem sob medida em cada um.

    verify é o callable de integridade dos modelos (o pipeline injeta
    draw.infra.manifest.verify_model; None é o opt-out explícito das
    bancadas). config é o PipelineConfig que alinha Pipeline e
    PageCleanWorker no mesmo preset.

    Detecção, por precedência: ``detection_worker`` pronto (o
    OracleDet do sim) > ``detector`` crua (embrulhada no SyncDetector)
    > ``sheet_model`` (NeuralSheetDetector construído e carregado aqui)
    > nada (o SyncDetector roda o caminho clássico). ``sheet_model``
    informado constrói e carrega SEMPRE — mesmo quando um
    detection_worker o vence: o braço oráculo do sim_writer carrega o
    binário real sem usá-lo, e essa verificação é fail-closed.

    Tinta: ``page_clean`` injetado vence (não se chama load() no
    injetado — quem monta é quem carrega); senão PageCleanInk recebe
    ``ink_model`` CRU: None resolve DEFAULT_MODEL_PATH internamente e
    str(None) quebraria o caminho.

    Mão: ``hand_model=_NO_HAND`` (o default) não cria nada da pilha.
    Qualquer outro valor — inclusive ``None`` — segue os gates
    convencionais do app: o masker nasce via ``hand_model_available``
    e a presença vai sem gate para ``make_hand_presence``, que se
    auto-gateia pela mesma função. Com ``None`` os dois podem nascer
    do default convencional (o comportamento antigo do eval); um Path
    concreto (a paridade, que sempre resolve um) segue direto.
    ``presence_strict=True`` reprova a amostra perdida como erro (a
    régua de paridade exige: amostra perdida não pode virar "mão
    medida").

    Todos os imports ``draw.*`` ficam DENTRO do corpo, resolvidos a
    cada chamada: bancadas e testes instrumentam os módulos (o
    bench_ink espioa ``pipeline.Pipeline``; a paridade e o eval
    grampeiam ``PageCleanWorker``/``NeuralSheetDetector``/``Pipeline``)
    — um import no topo fixaria a classe real antes da interceptação.
    O SyncDetector.submit já lazy-importa pela mesma razão.
    """
    from draw import hand as hand_module
    from draw.hand_presence import make_hand_presence
    from draw.ink.worker import PageCleanWorker
    from draw.page_clean import PageCleanInk
    from draw.perception import NeuralSheetDetector
    from draw.pipeline import Pipeline

    clock = StepClock()

    if page_clean is None:
        page_clean = PageCleanInk(ink_model, verify=verify)
        page_clean.load()

    if hand_model is _NO_HAND:
        hand_masker = None
        presence = None
    else:
        hand_masker = (
            hand_module.HandMasker(hand_model, verify=verify)
            if hand_module.hand_model_available(hand_model)
            else None
        )
        presence = make_hand_presence(hand_model, verify=verify)
    presence_worker = (
        SyncPresence(presence, strict=presence_strict) if presence is not None else None
    )
    ink_worker = PageCleanWorker(
        page_clean,
        hand_masker,
        clock=clock,
        config=config.ink,
        ink_divnorm=config.ink_divnorm,
    )

    if sheet_model is not None:
        built_detector = NeuralSheetDetector(str(sheet_model), verify=verify)
        built_detector.load()
        if detector is None:
            detector = built_detector
    if detection_worker is None:
        detection_worker = SyncDetector(
            detector, detect_ms=detect_ms, keep_history=keep_history
        )
    pipeline = Pipeline(
        detection_worker=detection_worker,
        ink_worker=ink_worker,
        hand_presence_worker=presence_worker,
        hand_masker=hand_masker,
        clock=clock,
        log=lambda *_a: None,
        config=config,
    )
    return SyncBundle(
        clock=clock,
        detection_worker=detection_worker,
        presence_worker=presence_worker,
        ink_worker=ink_worker,
        pipeline=pipeline,
        page_clean=page_clean,
        hand_masker=hand_masker,
        presence=presence,
    )


def step_sync_frame(pipeline, ink_worker, packet):
    """Um quadro do laço dirigido: step + pump + colheita, com erro explícito.

    ``step`` já devolve uma colheita (None ou INK_WORKER_FAILED);
    ``pump`` devolve True quando uma passada rodou — só então a
    colheita extra publica a passada nova no mesmo quadro (sem
    passada, collect_ink é um poll_display vazio). Um worker.error ou
    um outcome não-None encerram o laço como RuntimeError. Devolve
    ``pumped`` para o driver medir a passada deste quadro.
    """
    outcome = pipeline.step(packet)
    pumped = ink_worker.pump()
    if pumped and outcome is None:
        outcome = pipeline.collect_ink()
    if ink_worker.error is not None:
        raise RuntimeError("trabalhadora de tinta falhou") from ink_worker.error
    if outcome is not None:
        raise RuntimeError(f"laço do Draw encerrou: {outcome}")
    return pumped
