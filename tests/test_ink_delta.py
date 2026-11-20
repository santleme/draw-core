"""InkDelta: o que a tinta confirmada ganhou, via PageCleanWorker.poll_delta.

Canal aditivo ao poll(): mesma versao, sem mexer em cadencia nem voto.
Passadas sincronas (pump) com relogio de cena; mapas de tinta roteirizados.
"""

import numpy as np
import pytest

from draw.contracts import InkDelta, WarpedPage
from draw.ink import worker as worker_mod

H, W = 40, 60
STROKE_A = (slice(5, 8), slice(5, 25))
STROKE_B = (slice(30, 33), slice(35, 55))


class _Script:
    threshold = 0.7

    def __init__(self):
        self.strokes = []

    def ink_prob(self, _warped):
        prob = np.zeros((H, W), np.float32)
        for box in self.strokes:
            prob[box] = 0.99
        return prob


class _Rig:
    def __init__(self, monkeypatch=None, history=None, corners=None):
        if history is not None:
            monkeypatch.setattr(worker_mod, "_DELTA_HISTORY", history)
        self.t = [100.0]
        self.ink = _Script()
        self.worker = worker_mod.PageCleanWorker(self.ink, clock=lambda: self.t[0])
        self.generation = self.worker.new_generation()
        self.corners = corners
        self.frame_t = 0.0
        self.version = 0

    def step(self, *strokes):
        """Uma passada com a lista de tracos visiveis."""
        self.ink.strokes = list(strokes)
        self.t[0] += 1.0
        self.frame_t += 0.1
        self.worker.submit(
            WarpedPage(
                np.zeros((H, W, 3), np.uint8),
                self.generation,
                self.corners,
                captured_at=self.frame_t,
            )
        )
        assert self.worker.pump()

    def delta(self, after=None):
        return self.worker.poll_delta(
            self.generation, self.version if after is None else after
        )


def _published(rig):
    snap = rig.worker.poll(rig.generation, -1)
    return None if snap is None else snap.version


def test_first_publication_of_a_generation_is_full():
    rig = _Rig()
    rig.step()
    delta = rig.delta(after=-1)
    assert isinstance(delta, InkDelta)
    assert delta.full is True
    assert delta.bbox is None and delta.mask.shape == (0, 0)
    assert delta.page_grid == (H, W)
    assert delta.generation == rig.generation


def test_new_stroke_arrives_as_cropped_incremental_delta():
    rig = _Rig()
    rig.step()
    rig.version = _published(rig)
    rig.step(STROKE_A)
    assert rig.delta() is None  # 1a passada: ainda so provisoria
    rig.step(STROKE_A)  # 2a passada forte confirma
    delta = rig.delta()
    assert delta is not None and delta.full is False
    assert delta.bbox == (5, 8, 5, 25)
    assert delta.mask.shape == (3, 20) and (delta.mask == 255).all()
    assert delta.since_version == rig.version
    assert delta.version == _published(rig)
    assert delta.removed_px == 0
    assert delta.captured_at == pytest.approx(rig.frame_t)
    assert delta.mm_per_px is None or delta.mm_per_px > 0
    assert not delta.mask.flags.writeable


def test_nothing_new_returns_none_and_consumer_catches_up():
    rig = _Rig()
    rig.step()
    rig.step(STROKE_A)
    rig.step(STROKE_A)
    latest = _published(rig)
    assert rig.worker.poll_delta(rig.generation, latest) is None


def test_lagging_consumer_gets_the_merged_union_of_deltas():
    rig = _Rig()
    rig.step()
    rig.version = _published(rig)
    rig.step(STROKE_A)
    rig.step(STROKE_A)
    rig.step(STROKE_A, STROKE_B)
    rig.step(STROKE_A, STROKE_B)
    delta = rig.delta()  # nunca consumiu entre as duas confirmacoes
    assert delta.full is False
    assert delta.bbox == (5, 33, 5, 55)
    canvas = np.zeros((H, W), bool)
    y0, y1, x0, x1 = delta.bbox
    canvas[y0:y1, x0:x1] = delta.mask > 0
    expected = np.zeros((H, W), bool)
    expected[STROKE_A] = expected[STROKE_B] = True
    assert (canvas == expected).all()


def test_history_overflow_degrades_to_full_replace(monkeypatch):
    rig = _Rig(monkeypatch, history=1)
    rig.step()
    rig.version = _published(rig)
    rig.step(STROKE_A)
    rig.step(STROKE_A)
    rig.step(STROKE_A, STROKE_B)
    rig.step(STROKE_A, STROKE_B)
    delta = rig.delta()
    assert delta.full is True
    assert delta.bbox == (5, 33, 5, 55)


def _publish(worker, gen, version, mask, captured_at, anchor=None):
    """Publica `mask` como a versao `version` e registra o delta."""
    worker._version = version
    worker._published = (gen, version, mask)
    worker._record_delta(gen, version, mask, captured_at, anchor)


def _strokes():
    canvas = np.zeros((H, W), np.uint8)
    canvas[STROKE_A] = 255
    shrunk = canvas.copy()
    shrunk[5:6, 5:25] = 0  # uma linha do traco sumiu
    return canvas, shrunk


def test_removed_ink_makes_the_interval_full():
    rig = _Rig()
    worker, gen = rig.worker, rig.generation
    canvas, shrunk = _strokes()
    _publish(worker, gen, 1, canvas, 0.1)
    _publish(worker, gen, 2, shrunk, 0.2)
    delta = worker.poll_delta(gen, 1)
    assert delta.full is True
    assert delta.removed_px == 20
    assert delta.bbox == (6, 8, 5, 25)  # o canvas inteiro que sobrou
    assert delta.captured_at == pytest.approx(0.2)


def test_initial_poll_counts_the_erasures_in_the_ring():
    # Review #61: com after_version=-1 o anel completo da geracao nao pode
    # ser tratado como historico faltante (a versao 0 nunca foi publicada).
    rig = _Rig()
    worker, gen = rig.worker, rig.generation
    canvas, shrunk = _strokes()
    _publish(worker, gen, 7, canvas, 0.1)  # versoes seguem entre geracoes
    _publish(worker, gen, 8, shrunk, 0.2)
    delta = worker.poll_delta(gen, -1)
    assert delta.full is True and delta.removed_px == 20


def test_moved_anchor_without_mask_change_is_a_full_delta():
    # Review #61: re-ancorar move a tinta no espaco da camera mesmo com a
    # mascara igual; um delta incremental vazio esconderia isso.
    rig = _Rig()
    worker, gen = rig.worker, rig.generation
    canvas, _ = _strokes()
    anchor_a = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    anchor_b = anchor_a + np.float32(3.0)
    _publish(worker, gen, 1, canvas, 0.1, anchor_a)
    _publish(worker, gen, 2, canvas, 0.2, anchor_b)
    delta = worker.poll_delta(gen, 1)
    assert delta.full is True
    assert np.allclose(delta.anchor_corners, anchor_b)
    assert delta.bbox == (5, 8, 5, 25)  # o canvas inteiro, para substituir


def test_unchanged_anchor_keeps_the_delta_incremental():
    rig = _Rig()
    worker, gen = rig.worker, rig.generation
    canvas, _ = _strokes()
    anchor = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    more = canvas.copy()
    more[STROKE_B] = 255
    _publish(worker, gen, 1, canvas, 0.1, anchor)
    _publish(worker, gen, 2, more, 0.2, anchor.copy())
    delta = worker.poll_delta(gen, 1)
    assert delta.full is False and delta.bbox == (30, 33, 35, 55)


def test_delta_not_ready_yet_returns_none():
    # A passada publica a versao sob o lock e calcula o delta logo depois,
    # fora dele: nesse intervalo o consumidor tenta de novo.
    rig = _Rig()
    worker, gen = rig.worker, rig.generation
    canvas, _ = _strokes()
    _publish(worker, gen, 1, canvas, 0.1)
    worker._version = 2
    worker._published = (gen, 2, canvas)  # sem _record_delta ainda
    assert worker.poll_delta(gen, 1) is None


def test_new_generation_restarts_with_a_full_delta():
    rig = _Rig()
    rig.step()
    rig.step(STROKE_A)
    rig.step(STROKE_A)
    old = rig.generation
    rig.generation = rig.worker.new_generation()
    assert rig.worker.poll_delta(old, -1) is None  # geracao morta
    rig.step(STROKE_B)
    rig.step(STROKE_B)
    delta = rig.worker.poll_delta(rig.generation, -1)
    assert delta.full is True
    assert delta.generation == rig.generation
    assert delta.bbox is not None and delta.bbox[0] >= 30  # sem a tinta velha


def test_anchor_corners_travel_with_the_delta():
    corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    rig = _Rig(corners=corners)
    rig.step()
    delta = rig.delta(after=-1)
    assert delta.anchor_corners is not None
    assert delta.anchor_corners.shape == (4, 2)
    assert np.allclose(delta.anchor_corners, corners)


def test_poll_is_untouched_by_delta_bookkeeping():
    rig = _Rig()
    rig.step()
    rig.step(STROKE_A)
    rig.step(STROKE_A)
    snap = rig.worker.poll(rig.generation, -1)
    assert snap.confirmed.shape == (H, W)
    assert (snap.confirmed[STROKE_A] == 255).all()


def test_contract_validation():
    with pytest.raises(ValueError):
        WarpedPage(np.zeros((H, W, 3), np.uint8), 0, captured_at=float("nan"))
    assert WarpedPage(np.zeros((H, W, 3), np.uint8), 0).captured_at is None
    base = dict(
        generation=0,
        since_version=0,
        version=1,
        captured_at=None,
        page_grid=(H, W),
        mm_per_px=None,
        anchor_corners=None,
        removed_px=0,
        full=False,
    )
    with pytest.raises(ValueError):
        InkDelta(bbox=(0, 2, 0, 2), mask=np.zeros((3, 3), np.uint8), **base)
    with pytest.raises(ValueError):
        InkDelta(bbox=None, mask=np.zeros((2, 2), np.uint8), **base)
    ok = InkDelta(bbox=None, mask=np.zeros((0, 0), np.uint8), **base)
    assert ok.bbox is None


def test_poll_delta_waits_briefly_for_a_record_still_being_computed():
    import threading
    import time

    rig = _Rig()
    worker, gen = rig.worker, rig.generation
    canvas, _ = _strokes()
    _publish(worker, gen, 1, canvas, 0.1)
    worker._version = 2
    worker._published = (gen, 2, canvas)  # publicado; o registro vem ja ja

    assert worker.poll_delta(gen, 1) is None  # sem espera: tenta de novo depois

    def late():
        time.sleep(0.03)
        worker._record_delta(gen, 2, canvas, 0.2, None)

    thread = threading.Thread(target=late)
    thread.start()
    started = time.perf_counter()
    delta = worker.poll_delta(gen, 1, wait=1.0)
    elapsed = time.perf_counter() - started
    thread.join()
    assert delta is not None and delta.version == 2
    assert elapsed < 0.5  # acordou pelo registro, nao pelo prazo
    # sem registro nenhum, a espera respeita o prazo
    worker._version = 3
    worker._published = (gen, 3, canvas)
    started = time.perf_counter()
    assert worker.poll_delta(gen, 2, wait=0.05) is None
    assert time.perf_counter() - started < 0.5
