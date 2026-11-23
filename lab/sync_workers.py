"""Adaptadores síncronos compartilhados das trabalhadoras do laço.

replay_lab, sim_writer, parity_e2e e eval_capture dirigem o
draw.pipeline sem threads: a mesma decisão da trabalhadora real roda
no submit e o poll devolve o resultado mais novo — o contrato
submit/poll dos workers de detecção e presença.

O driver de paridade do legado (parity_legacy_driver.py) NÃO usa
este módulo: roda num subprocesso com PYTHONPATH do repo legado e
fica auto-contido de propósito.
"""

from pathlib import Path

import cv2

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def iter_frames(source, limit):
    """Frames de um diretório de imagens ou de um vídeo."""
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
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
