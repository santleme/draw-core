"""Guarda fotometrica pelo worker real (issue #129): mao sem mascara."""

import time

import numpy as np

from draw.contracts import WarpedPage
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig

H, W = 256, 384
Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)
WITNESS = (slice(16, 112), slice(16, 160))  # ~13,8k px: passa o piso
COVERED = (slice(144, 208), slice(224, 336))
HAND = (slice(128, 224), slice(208, 352))


class _Scripted:
    threshold = 0.7

    def __init__(self):
        self.content = np.zeros((H, W), np.float32)

    def ink_prob(self, _warped):
        return self.content


def _paper():
    image = np.empty((H, W, 3), np.uint8)
    image[:] = (200, 205, 210)
    return image


def _prob(*, covered_ink=True, extra=None):
    prob = np.zeros((H, W), np.float32)
    prob[WITNESS] = 0.95
    if covered_ink:
        prob[COVERED] = 0.95
    if extra is not None:
        prob[extra] = 0.95
    return prob


def _pass(worker, generation, image):
    before = worker.stats()["passes_total"]
    worker.submit(WarpedPage(image, generation, Q0, np.ones((H, W), bool)))
    deadline = time.monotonic() + 5.0
    while worker.stats()["passes_total"] <= before:
        assert time.monotonic() < deadline, "a passada nao executou"
        time.sleep(0.001)


def _run(monkeypatch, config):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _Scripted()
    worker = worker_mod.PageCleanWorker(ink, config=config).start()
    generation = worker.new_generation()
    ink.content = _prob()
    for _ in range(5):
        _pass(worker, generation, _paper())
    confirmed = worker.poll(generation, 0).confirmed
    assert confirmed[COVERED].all()
    return worker, ink, generation


def _hand_image():
    image = _paper()
    image[HAND] = (60, 62, 64)  # mao/sombra escura que o MediaPipe nao viu
    return image


def test_unmasked_dark_hand_never_erases_confirmed_stroke(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig())
    try:
        ink.content = _prob(covered_ink=False)
        for _ in range(14):
            _pass(worker, generation, _hand_image())
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[WITNESS].all()
        # Mao sai, o traco continua la e tinta nova confirma.
        new = (slice(224, 240), slice(16, 120))
        ink.content = _prob(extra=new)
        for _ in range(12):
            _pass(worker, generation, _paper())
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[new].all()
    finally:
        worker.stop()


def test_control_without_guard_erases_under_the_same_hand(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig(paper_guard=False))
    try:
        ink.content = _prob(covered_ink=False)
        for _ in range(14):
            _pass(worker, generation, _hand_image())
        assert not worker.poll(generation, 0).confirmed[COVERED].any()
    finally:
        worker.stop()


def test_paper_really_erased_still_removes_with_guard(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig())
    try:
        ink.content = _prob(covered_ink=False)
        for _ in range(14):
            _pass(worker, generation, _paper())
        snap = worker.poll(generation, 0)
        assert not snap.confirmed[COVERED].any()
        assert snap.confirmed[WITNESS].all()
    finally:
        worker.stop()
