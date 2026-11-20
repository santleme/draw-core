"""Cadencia adaptativa do detector com sentinela de bordas (Detect-then-Wake).

Quando a folha assenta por 3 deteccoes consecutivas sem salto, o
DetectWakeGate reduz o detector neural para a cadencia de verificacao
(1 Hz por padrao) e monitora as 4 bordas do quad fora da mao em < 0,1 ms
por quadro. Se a folha mexer, o sentinela acorda o detector no proprio
quadro, sem esperar o timer de 1 s.
"""

import sys
import time

import cv2
import numpy as np
import pytest

from draw import app as app_module
from draw import contracts, pipeline, tracker
from draw.settings import PipelineConfig


def _sheet_frame(corners, *, shape=(480, 640, 3), bg=50, fg=220):
    img = np.full(shape, bg, dtype=np.uint8)
    cv2.fillConvexPoly(img, np.round(corners).astype(np.int32), (fg, fg, fg))
    return img


BASE_CORNERS = np.float32([[120, 80], [520, 80], [520, 400], [120, 400]])


def test_settles_after_three_stable_detections_and_skips_between_heartbeats():
    gate = tracker.DetectWakeGate(settled_interval_s=1.0, settle_detections=3)
    frame = _sheet_frame(BASE_CORNERS)

    for i in range(2):
        gate.on_detection(BASE_CORNERS, frame, captured_at=i * 0.2)
        assert not gate.settled
        assert gate.should_detect(frame, i * 0.2 + 0.03, base_should_detect=True)
        assert not gate.should_detect(frame, i * 0.2 + 0.06, base_should_detect=False)

    # 3a deteccao estavel assenta o portao
    gate.on_detection(BASE_CORNERS, frame, captured_at=0.4)
    assert gate.settled

    # Quadros estaticos intermediarios sao pulados (< 1 s)
    assert not gate.should_detect(frame, 0.6, base_should_detect=True)
    assert not gate.should_detect(frame, 0.9, base_should_detect=True)
    assert gate.skipped_submits == 2

    # Heartbeat em 1,0 s dispara exatamente uma vez (nao spamma enquanto em voo)
    assert gate.should_detect(frame, 1.4, base_should_detect=False)
    assert gate.heartbeat_events == 1
    assert not gate.should_detect(frame, 1.43, base_should_detect=True)


def test_sentinel_wakes_immediately_when_sheet_moves():
    gate = tracker.DetectWakeGate(settled_interval_s=1.0, settle_detections=3)
    still = _sheet_frame(BASE_CORNERS)
    for i in range(3):
        gate.on_detection(BASE_CORNERS, still, captured_at=i * 0.2)
    assert gate.settled

    # Folha escorrega 8 px: sentinela acorda no proprio quadro (0,5 s < 1,4 s)
    moved = _sheet_frame(BASE_CORNERS + np.float32([8.0, 0.0]))
    assert gate.should_detect(moved, 0.5, base_should_detect=False)
    assert not gate.settled
    assert gate.wake_events == 1


def test_sentinel_ignores_global_exposure_shift_and_hand_on_one_edge():
    gate = tracker.DetectWakeGate(settled_interval_s=1.0, settle_detections=3)
    still = _sheet_frame(BASE_CORNERS, bg=50, fg=200)
    for i in range(3):
        gate.on_detection(BASE_CORNERS, still, captured_at=i * 0.2)
    assert gate.settled

    # Variacao global de brilho (+25) nao dispara falso alarme
    brighter = np.clip(still.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    assert not gate.should_detect(brighter, 0.6, base_should_detect=True)

    # Mao cobrindo uma borda inteira (mascarada por hand_mask) tambem nao acorda
    with_hand = still.copy()
    with_hand[60:110, 100:540] = 130
    hand_mask = np.zeros(still.shape[:2], dtype=np.uint8)
    hand_mask[55:115, 95:545] = 255
    assert not gate.should_detect(
        with_hand, 0.7, base_should_detect=True, hand_mask=hand_mask
    )
    assert gate.settled


def test_sentinel_wakes_when_hand_covers_too_many_border_samples():
    gate = tracker.DetectWakeGate(settled_interval_s=1.0, settle_detections=3)
    still = _sheet_frame(BASE_CORNERS)
    for i in range(3):
        gate.on_detection(BASE_CORNERS, still, captured_at=i * 0.2)
    assert gate.settled

    # Mascara de mao cobrindo quase tudo: nao fica cego, acorda por seguranca
    huge_hand = np.full(still.shape[:2], 255, dtype=np.uint8)
    assert gate.should_detect(still, 0.5, base_should_detect=False, hand_mask=huge_hand)
    assert not gate.settled


def test_sentinel_latency_is_sub_millisecond_on_1080p():
    gate = tracker.DetectWakeGate(settled_interval_s=1.0, settle_detections=3)
    corners_1080 = np.float32([[300, 150], [1600, 150], [1600, 950], [300, 950]])
    frame = _sheet_frame(corners_1080, shape=(1080, 1920, 3))
    for i in range(3):
        gate.on_detection(corners_1080, frame, captured_at=i * 0.2)
    assert gate.settled
    t0 = time.perf_counter()
    for _ in range(50):
        assert not gate.should_detect(frame, 0.5, base_should_detect=False)
    per_frame_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    assert per_frame_ms < 0.5


def test_pipeline_default_never_builds_gate_and_opt_in_skips_when_settled(monkeypatch):
    # Paridade: com detect_wake=False o portao nem e construido.
    def _boom(*_a, **_kw):
        raise AssertionError("DetectWakeGate construido com detect_wake=False")

    class _DummyDet:
        error = None

        def __init__(self):
            self.submits = 0
            self._seq = 0
            self._pending = None

        def submit(self, frame, captured_at, frame_index=None):
            self.submits += 1
            self._seq += 1
            self._pending = contracts.DetectionResult(
                sequence=self._seq,
                corners=BASE_CORNERS.copy(),
                verdict="paper",
                captured_at=captured_at,
                detect_ms=10.0,
            )

        def poll(self, after_seq):
            if self._pending is not None and self._pending.sequence > after_seq:
                return self._pending
            return None

        def take_traces(self, upto_result_seq=None):
            return []

    class _DummyInk:
        error = None

        def new_generation(self, preserve_identity=False):
            return 1

        def submit(self, *_a, **_kw):
            return None

        def set_hand_presence(self, *_a, **_kw):
            return None

        def poll_display(self, *_a, **_kw):
            return None

    monkeypatch.setattr(tracker, "DetectWakeGate", _boom)
    pipe_off = pipeline.Pipeline(
        detection_worker=_DummyDet(),
        ink_worker=_DummyInk(),
        config=PipelineConfig(detect_wake=False),
    )
    assert pipe_off._detect_wake is None


def test_pipeline_opt_in_skips_when_settled_and_wakes_on_move():
    class _DummyDet:
        error = None

        def __init__(self):
            self.submits = 0
            self._seq = 0
            self._pending = None

        def submit(self, frame, captured_at, frame_index=None):
            self.submits += 1
            self._seq += 1
            self._pending = contracts.DetectionResult(
                sequence=self._seq,
                corners=BASE_CORNERS.copy(),
                verdict="paper",
                captured_at=captured_at,
                detect_ms=10.0,
            )

        def poll(self, after_seq):
            if self._pending is not None and self._pending.sequence > after_seq:
                return self._pending
            return None

        def take_traces(self, upto_result_seq=None):
            return []

    class _DummyInk:
        error = None

        def new_generation(self, preserve_identity=False):
            return 1

        def submit(self, *_a, **_kw):
            return None

        def set_hand_presence(self, *_a, **_kw):
            return None

        def poll_display(self, *_a, **_kw):
            return None

    now = [100.0]
    det = _DummyDet()
    pipe = pipeline.Pipeline(
        detection_worker=det,
        ink_worker=_DummyInk(),
        config=PipelineConfig(detect_wake=True, detect_wake_interval_s=1.0),
        clock=lambda: now[0],
    )
    still = _sheet_frame(BASE_CORNERS)
    # 30 quadros estaticos a 30 fps (1 s): assenta apos 3 deteccoes e pula o resto
    for seq in range(30):
        pipe.step(contracts.FramePacket(still, now[0], seq))
        now[0] += 1.0 / 30.0
    assert pipe._detect_wake.settled
    assert pipe._detect_wake.skipped_submits >= 2
    submits_before_move = det.submits

    # Movimento da folha acorda o detector no proprio quadro
    moved = _sheet_frame(BASE_CORNERS + np.float32([10.0, 0.0]))
    pipe.step(contracts.FramePacket(moved, now[0], 30))
    assert det.submits == submits_before_move + 1
    assert not pipe._detect_wake.settled
    assert pipe._detect_wake.wake_events == 1


def test_settled_tracking_keeps_feeding_ink_and_hand_presence():
    # Regressao (review #69): com detect_wake assentado em 1 Hz, a tinta e
    # a presenca de mao nao podem cair para 1 Hz — seguem recebendo quadros
    # na cadencia ativa (--ink-interval 0,15 s) sobre o quad assentado.
    from draw.ink.config import InkConfig

    class _DummyDet:
        error = None

        def __init__(self):
            self.submits = 0
            self._seq = 0
            self._pending = None

        def submit(self, frame, captured_at, frame_index=None):
            self.submits += 1
            self._seq += 1
            self._pending = contracts.DetectionResult(
                sequence=self._seq,
                corners=BASE_CORNERS.copy(),
                verdict="paper",
                captured_at=captured_at,
                detect_ms=10.0,
            )

        def poll(self, after_seq):
            if self._pending is not None and self._pending.sequence > after_seq:
                return self._pending
            return None

        def take_traces(self, upto_result_seq=None):
            return []

    class _CountingInk:
        error = None

        def __init__(self):
            self.submits = 0

        def new_generation(self, preserve_identity=False):
            return 1

        def submit(self, *_a, **_kw):
            self.submits += 1

        def set_hand_presence(self, *_a, **_kw):
            return None

        def poll_display(self, *_a, **_kw):
            return None

    class _CountingHand:
        error = None

        def __init__(self):
            self.submits = 0

        def submit(self, *_a, **_kw):
            self.submits += 1

        def poll(self, *_a, **_kw):
            return None

    now = [100.0]
    det = _DummyDet()
    ink = _CountingInk()
    hand = _CountingHand()
    pipe = pipeline.Pipeline(
        detection_worker=det,
        ink_worker=ink,
        hand_presence_worker=hand,
        config=PipelineConfig(
            ink=InkConfig(pass_interval_s=0.15),
            detect_wake=True,
            detect_wake_interval_s=1.0,
        ),
        clock=lambda: now[0],
    )
    still = _sheet_frame(BASE_CORNERS)
    # 60 quadros (2,0 s a 30 fps): detector cai para ~5 chamadas, mas tinta
    # e mao seguem recebendo >= 10 quadros (>= 5 Hz).
    for seq in range(60):
        pipe.step(contracts.FramePacket(still, now[0], seq))
        now[0] += 1.0 / 30.0
    assert pipe._detect_wake.settled
    assert det.submits <= 6
    assert ink.submits >= 10
    assert hand.submits >= 10


def test_inflight_move_during_third_detection_wakes_immediately():
    # Regressao (review #69): se a folha mexer enquanto a 3a deteccao estavel
    # esta em voo, a referencia do sentinela precisa vir do quadro submetido
    # (nao do quadro ja movido), para acordar imediatamente no quadro seguinte.
    class _DelayedDet:
        error = None

        def __init__(self):
            self.submits = 0
            self._seq = 0
            self._queue = []

        def submit(self, frame, captured_at, frame_index=None):
            self.submits += 1
            self._seq += 1
            self._queue.append(
                contracts.DetectionResult(
                    sequence=self._seq,
                    corners=BASE_CORNERS.copy(),
                    verdict="paper",
                    captured_at=captured_at,
                    detect_ms=10.0,
                )
            )

        def poll(self, after_seq):
            if self._queue and self._queue[0].sequence > after_seq:
                return self._queue.pop(0)
            return None

        def take_traces(self, upto_result_seq=None):
            return []

    class _DummyInk:
        error = None

        def new_generation(self, preserve_identity=False):
            return 1

        def submit(self, *_a, **_kw):
            return None

        def set_hand_presence(self, *_a, **_kw):
            return None

        def poll_display(self, *_a, **_kw):
            return None

    now = [100.0]
    det = _DelayedDet()
    pipe = pipeline.Pipeline(
        detection_worker=det,
        ink_worker=_DummyInk(),
        config=PipelineConfig(detect_wake=True, detect_wake_interval_s=1.0),
        clock=lambda: now[0],
    )
    still = _sheet_frame(BASE_CORNERS)
    moved = _sheet_frame(BASE_CORNERS + np.float32([12.0, 0.0]))

    # Roda quadros estaticos ate o exato quadro em que a 3a deteccao e submetida
    seq = 0
    while det.submits < 3:
        pipe.step(contracts.FramePacket(still, now[0], seq))
        now[0] += 1.0 / 30.0
        seq += 1
    assert not pipe._detect_wake.settled
    # No quadro seguinte a 3a deteccao chega, mas a folha JA mexeu nesse quadro:
    # a referencia guardada deve ser a do quadro submetido (still), nao moved.
    pipe.step(contracts.FramePacket(moved, now[0], seq))
    now[0] += 1.0 / 30.0
    assert pipe._detect_wake.settled
    submits_before = det.submits
    # No quadro seguinte, sentinela compara moved vs still e acorda ja!
    pipe.step(contracts.FramePacket(moved, now[0], seq + 1))
    assert det.submits == submits_before + 1
    assert not pipe._detect_wake.settled


def test_hand_lift_with_stale_seg_hand_mask_wakes_detector_before_confirming():
    # Regressao (review #69): se a mao sair entre heartbeats enquanto
    # _hand_mask_det ainda marca a mao da ultima deteccao, o pipeline acorda
    # o detector e nao submete uma passada de confirmacao mascarada pela
    # mascara velha.
    hand_blob = np.zeros((480, 640), dtype=np.uint8)
    hand_blob[200:260, 250:320] = 255  # mao no centro da folha (longe das bordas)

    class _DetWithHand:
        error = None

        def __init__(self):
            self.submits = 0
            self._seq = 0
            self._pending = None

        def submit(self, frame, captured_at, frame_index=None):
            self.submits += 1
            self._seq += 1
            self._pending = contracts.DetectionResult(
                sequence=self._seq,
                corners=BASE_CORNERS.copy(),
                verdict="paper",
                captured_at=captured_at,
                detect_ms=10.0,
                hand_mask=hand_blob.copy(),
            )

        def poll(self, after_seq):
            if self._pending is not None and self._pending.sequence > after_seq:
                return self._pending
            return None

        def take_traces(self, upto_result_seq=None):
            return []

    class _CountingInk:
        error = None

        def __init__(self):
            self.submits = 0

        def new_generation(self, preserve_identity=False):
            return 1

        def submit(self, *_a, **_kw):
            self.submits += 1

        def set_hand_presence(self, *_a, **_kw):
            return None

        def poll_display(self, *_a, **_kw):
            return None

    class _HandExit:
        error = None

        def __init__(self):
            self._seq = 0
            self.next_state = None

        def submit(self, *_a, **_kw):
            return None

        def poll(self, after_seq):
            if self.next_state is not None and self._seq > after_seq:
                return (self._seq, self.next_state, False)
            return None

    now = [100.0]
    det = _DetWithHand()
    ink = _CountingInk()
    hand = _HandExit()
    pipe = pipeline.Pipeline(
        detection_worker=det,
        ink_worker=ink,
        hand_presence_worker=hand,
        config=PipelineConfig(detect_wake=True, detect_wake_interval_s=1.0),
        clock=lambda: now[0],
    )
    still = _sheet_frame(BASE_CORNERS)
    for seq in range(16):
        pipe.step(contracts.FramePacket(still, now[0], seq))
        now[0] += 1.0 / 30.0
    assert pipe._detect_wake.settled

    # Mao sai entre heartbeats: _detect_wake acorda e nao submete tinta
    # mascarada pelo _hand_mask_det velho neste quadro.
    ink_before = ink.submits
    hand._seq += 1
    hand.next_state = False
    pipe.step(contracts.FramePacket(still, now[0], 16))
    assert not pipe._detect_wake.settled
    assert ink.submits == ink_before


def test_late_masked_heartbeat_after_hand_lift_does_not_hide_finished_stroke():
    # Regressao (review #69): um heartbeat capturado com a mao ainda sobre
    # o traco retorna DEPOIS de o MediaPipe ja ter reportado ausencia da mao.
    # A mascara defasada (capturada antes da saida da mao) e descartada e o
    # detector acorda uma vez; o traco exposto chega ao canvas confirmado no
    # PageCleanWorker real sem precisar de outro movimento da mao.
    from draw.ink.config import InkConfig
    from draw.ink.worker import PageCleanWorker

    class _DarkIsInk:
        threshold = 0.5

        def ink_prob(self, warped):
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            return ((255.0 - gray) / 255.0).astype(np.float32)

    hand_blob = np.zeros((480, 640), dtype=np.uint8)
    hand_blob[200:280, 260:380] = 255

    class _ManualDet:
        error = None

        def __init__(self):
            self.submits = 0
            self._seq = 0
            self.queued = None

        def submit(self, frame, captured_at, frame_index=None):
            self.submits += 1
            self._seq += 1
            self.queued = contracts.DetectionResult(
                sequence=self._seq,
                corners=BASE_CORNERS.copy(),
                verdict="paper",
                captured_at=captured_at,
                detect_ms=10.0,
                hand_mask=None,
            )

        def poll(self, after_seq):
            if self.queued is not None and self.queued.sequence > after_seq:
                return self.queued
            return None

        def take_traces(self, upto_result_seq=None):
            return []

    class _ManualHand:
        error = None

        def __init__(self):
            self._seq = 0
            self.state = None

        def submit(self, *_a, **_kw):
            return None

        def poll(self, after_seq):
            if self.state is not None and self._seq > after_seq:
                return (self._seq, self.state, False)
            return None

    now = [100.0]
    det = _ManualDet()
    hand = _ManualHand()
    ink_worker = PageCleanWorker(
        _DarkIsInk(),
        config=InkConfig(pass_interval_s=0.15),
        clock=lambda: now[0],
    )
    pipe = pipeline.Pipeline(
        detection_worker=det,
        ink_worker=ink_worker,
        hand_presence_worker=hand,
        config=PipelineConfig(
            ink=InkConfig(pass_interval_s=0.15),
            detect_wake=True,
            detect_wake_interval_s=1.0,
        ),
        clock=lambda: now[0],
    )
    still = _sheet_frame(BASE_CORNERS)
    # 1) Assenta folha limpa
    for seq in range(16):
        pipe.step(contracts.FramePacket(still, now[0], seq))
        ink_worker.pump()
        now[0] += 0.2
    assert pipe._detect_wake.settled

    # 2) Usuario escreve um traco no centro da folha sob a regiao do hand_blob
    with_stroke = still.copy()
    cv2.line(with_stroke, (280, 240), (360, 240), (15, 15, 15), 6)
    # 1o voto limpo ja com a mao saindo -> MediaPipe reporta ausencia ANTES
    # do heartbeat atrasado chegar com hand_blob.
    pre_exit_ts = now[0] - 0.1
    hand._seq += 1
    hand.state = False
    pipe.step(contracts.FramePacket(with_stroke, now[0], 16))
    ink_worker.pump()
    now[0] += 0.2

    # 3) Heartbeat atrasado (capturado em pre_exit_ts < hand_exit_at) chega
    # carregando hand_blob sobre o traco. A mascara velha precisa ser
    # descartada para a passada de confirmacao confirmar o traco!
    det._seq += 1
    det.queued = contracts.DetectionResult(
        sequence=det._seq,
        corners=BASE_CORNERS.copy(),
        verdict="paper",
        captured_at=pre_exit_ts,
        detect_ms=10.0,
        hand_mask=hand_blob.copy(),
    )
    pipe.step(contracts.FramePacket(with_stroke, now[0], 17))
    ink_worker.pump()
    now[0] += 0.2
    pipe.collect_ink()

    # Roda mais uma passada limpa (detector acordou) para fechar a confirmacao
    for seq in (18, 19, 20):
        pipe.step(contracts.FramePacket(with_stroke, now[0], seq))
        ink_worker.pump()
        now[0] += 0.2
    pipe.collect_ink()
    assert pipe.clean_mask is not None
    assert int(np.count_nonzero(pipe.clean_mask)) > 0


def test_cached_hand_mask_on_hand_exit_still_confirms_final_stroke_under_production():
    """Regressao (review #74): sob PipelineConfig() (InkConfig.production),
    quando o detector ja tem uma mascara de mao em cache cobrindo o traco novo
    no momento em que a mao sai e uma passada mascarada ainda roda apos a
    saida antes da limpeza da mascara, as passadas limpas seguintes precisam
    confirmar o traco sem novo movimento da mao e depois entrar em descanso."""
    from draw.ink.worker import PageCleanWorker

    class _DarkIsInk:
        threshold = 0.7

        def ink_prob(self, warped):
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            return np.where(gray < 80, 0.98, 0.0).astype(np.float32)

    hand_blob = np.zeros((480, 640), dtype=np.uint8)
    hand_blob[200:280, 260:380] = 255

    class _ManualDet:
        error = None

        def __init__(self):
            self._seq = 0
            self.queued = None
            self.current_hand_mask = None

        def submit(self, frame, captured_at, frame_index=None):
            self._seq += 1
            mask = (
                None
                if self.current_hand_mask is None
                else self.current_hand_mask.copy()
            )
            self.queued = contracts.DetectionResult(
                sequence=self._seq,
                corners=BASE_CORNERS.copy(),
                verdict="paper",
                captured_at=captured_at,
                detect_ms=10.0,
                hand_mask=mask,
            )

        def poll(self, after_seq):
            if self.queued is not None and self.queued.sequence > after_seq:
                return self.queued
            return None

        def take_traces(self, upto_result_seq=None):
            return []

    class _ManualHand:
        error = None

        def __init__(self):
            self._seq = 0
            self.state = True
            self.moved = True

        def submit(self, *_a, **_kw):
            return None

        def poll(self, after_seq):
            if self._seq > after_seq:
                return (self._seq, self.state, self.moved)
            return None

    now = [100.0]
    det = _ManualDet()
    hand = _ManualHand()
    cfg = PipelineConfig()
    ink_worker = PageCleanWorker(_DarkIsInk(), config=cfg.ink, clock=lambda: now[0])
    pipe = pipeline.Pipeline(
        detection_worker=det,
        ink_worker=ink_worker,
        hand_presence_worker=hand,
        config=cfg,
        clock=lambda: now[0],
    )
    still = _sheet_frame(BASE_CORNERS)
    # Bloco base no canto superior para amadurecer a pagina e o voto.
    cv2.rectangle(still, (150, 120), (210, 160), (10, 10, 10), -1)
    for seq in range(18):
        hand._seq += 1
        hand.moved = seq < 4
        pipe.step(contracts.FramePacket(still, now[0], seq))
        ink_worker.pump()
        now[0] += 0.16
    pipe.collect_ink()
    base_px = int(np.count_nonzero(pipe.clean_mask))
    assert base_px > 0

    # Mao cobre o centro (hand_blob em cache no detector) e desenha traco novo.
    with_stroke = still.copy()
    cv2.rectangle(with_stroke, (295, 225), (345, 255), (10, 10, 10), -1)
    # 2 leituras claras do traco novo antes de a mascara de mao cobri-lo
    # (soma 2 * 0,98 = 1,96 < 2,55: ainda nao confirmado).
    for seq in (18, 19):
        hand._seq += 1
        hand.state = True
        hand.moved = True
        pipe.step(contracts.FramePacket(with_stroke, now[0], seq))
        ink_worker.pump()
        now[0] += 0.16
    pipe.collect_ink()
    assert int(np.count_nonzero(pipe.clean_mask)) == base_px

    # Detector coloca hand_blob em cache no pipeline (_hand_mask_det) e
    # submete um warp mascarado que fica pendente na trabalhadora quando a mao
    # sai — de modo que a 1a passada APOS a saida da mao roda mascarada!
    det.current_hand_mask = hand_blob
    pipe._detect_wake.wake()
    for seq in (20, 21):
        pipe.step(contracts.FramePacket(with_stroke, now[0], seq))
        now[0] += 0.04
    assert pipe._hand_mask_det is not None and np.any(pipe._hand_mask_det)
    assert ink_worker._pending is not None

    # Mao sai: a 1a passada apos a saida consome o warp mascarado (empurrando
    # 0,0 na janela do traco novo).
    passes_before_exit = ink_worker.stats()["passes_total"]
    hand._seq += 1
    hand.state = False
    hand.moved = False
    now[0] += 0.16
    pipe.step(contracts.FramePacket(with_stroke, now[0], 22))
    assert ink_worker.pump() is True
    assert ink_worker.stats()["passes_total"] == passes_before_exit + 1
    pipe.collect_ink()
    assert int(np.count_nonzero(pipe.clean_mask)) == base_px

    # Agora a deteccao acordada termina sem mao (current_hand_mask = None) e
    # as passadas limpas seguintes rodam sem nenhum movimento adicional da mao.
    det.current_hand_mask = None
    for seq in (23, 24, 25, 26):
        now[0] += 0.16
        pipe.step(contracts.FramePacket(with_stroke, now[0], seq))
        ink_worker.pump()
    pipe.collect_ink()
    assert int(np.count_nonzero(pipe.clean_mask)) > base_px
    # Avanca o relogio para fechar as passadas quietas e entrar em descanso.
    for seq in range(27, 35):
        now[0] += 1.05
        pipe.step(contracts.FramePacket(with_stroke, now[0], seq))
        ink_worker.pump()
    assert ink_worker.stats()["resting"] is True


def test_cli_default_is_on_and_config_validation(monkeypatch):
    with pytest.raises(ValueError):
        PipelineConfig(detect_wake_interval_s=0.0)
    with pytest.raises(ValueError):
        PipelineConfig(detect_wake_interval_s=6.0)

    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(sys, "argv", ["draw"])
    app_module.main()
    assert seen[-1]["pipeline_config"].detect_wake is True

    monkeypatch.setattr(sys, "argv", ["draw", "--legacy-ink"])
    app_module.main()
    assert seen[-1]["pipeline_config"].detect_wake is False
