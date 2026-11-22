"""Guarda fotometrica pelo worker real (issue #129): mao sem mascara."""

import dataclasses
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


def _pass(worker, generation, image, visible=None):
    before = worker.stats()["passes_total"]
    if visible is None:
        visible = np.ones((H, W), bool)
    worker.submit(WarpedPage(image, generation, Q0, visible))
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


def _no_idle_probe(monkeypatch):
    """Sem sonda ociosa: so as passadas submetidas votam (sem corrida)."""
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 3600.0)
    time.sleep(0.1)


def _pass_until(worker, generation, image, done, limit):
    """Submete ate done(snapshot) ou o limite: parar a tempo, porque a
    trabalhadora descansa de verdade quando nao ha mais o que votar."""
    for _ in range(limit):
        if done(worker.poll(generation, 0)):
            return
        _pass(worker, generation, image)


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


def test_black_frame_after_a_reference_freezes_instead_of_erasing(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig())
    try:
        ink.content = _prob(covered_ink=False)
        black = np.zeros((H, W, 3), np.uint8)
        for _ in range(14):
            _pass(worker, generation, black)
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
    finally:
        worker.stop()


def test_post_exit_confirm_window_is_rearmed_while_a_shadow_recovers(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig(post_exit_full_window=True))
    try:
        _no_idle_probe(monkeypatch)
        worker.set_hand_presence(True)
        for _ in range(2):
            _pass(worker, generation, _hand_image())
        worker.set_hand_presence(False)
        window = worker._post_exit_confirm_passes()
        assert window > 1
        _pass(worker, generation, _paper())  # sombra saiu: papel em recuperacao
        assert worker._confirm_passes == window
        assert worker._confirm_masked_skips >= 1
    finally:
        worker.stop()


def test_permanent_border_does_not_rearm_the_window(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig(post_exit_full_window=True))
    try:
        worker.set_hand_presence(True)
        worker.set_hand_presence(False)
        image = _paper()
        image[:, :64] = (60, 62, 64)  # borda escura permanente, sem tinta nela
        for _ in range(2):
            _pass(worker, generation, image)
        assert worker._confirm_masked_skips == 0
    finally:
        worker.stop()


def _mature(monkeypatch, config, *, idle_probe=False):
    worker, ink, generation = _run(monkeypatch, config)
    for _ in range(14):  # pagina madura e parada: o descanso esta ao alcance
        _pass(worker, generation, _paper())
    if not idle_probe:
        _no_idle_probe(monkeypatch)
    return worker, ink, generation


def test_stroke_hidden_on_the_first_post_exit_pass_still_confirms(monkeypatch):
    """Sombra ainda cobre o traco na 1a passada pos-saida (modelo ve 0,0) e
    some na seguinte: a janela cheia de votos precisa sobrar apos o hold."""
    config = dataclasses.replace(
        InkConfig.production(), page_model=False, paper_guard=True
    )
    worker, ink, generation = _mature(monkeypatch, config)
    try:
        worker.set_hand_presence(True)
        late = (slice(224, 240), slice(16, 120))
        shadow = _paper()
        shadow[(slice(208, 256), slice(0, 160))] = (60, 62, 64)
        for _ in range(3):
            _pass(worker, generation, shadow)
        worker.set_hand_presence(False)
        ink.content = _prob()  # o modelo ainda nao ve o traco sob a sombra
        _pass(worker, generation, shadow)
        ink.content = _prob(extra=late)
        _pass_until(
            worker,
            generation,
            _paper(),
            lambda snap: snap.confirmed[late].all(),
            config.occlusion_hold_passes + 8,
        )
        assert worker.poll(generation, 0).confirmed[late].all()
    finally:
        worker.stop()


def test_stroke_erased_under_a_shadow_is_removed_after_hand_exit(monkeypatch):
    """Apagou um traco confirmado sob a sombra da mao: ao levantar a mao, o
    hold barra a remocao, mas o descanso nao pode chegar antes dela."""
    config = dataclasses.replace(
        InkConfig.production(), page_model=False, paper_guard=True
    )
    worker, ink, generation = _mature(monkeypatch, config, idle_probe=True)
    try:
        worker.set_hand_presence(True)
        ink.content = _prob(covered_ink=False)
        for _ in range(3):
            _pass(worker, generation, _hand_image())
        worker.set_hand_presence(False)
        # Sombra ainda persiste em parte da janela pos-saida antes de limpar.
        for _ in range(2):
            _pass(worker, generation, _hand_image())
        _pass_until(
            worker,
            generation,
            _paper(),
            lambda snap: not snap.confirmed[COVERED].any(),
            config.occlusion_hold_passes + 12,
        )
        snap = worker.poll(generation, 0)
        assert not snap.confirmed[COVERED].any()
        assert snap.confirmed[WITNESS].all()
        _pass_until(
            worker,
            generation,
            _paper(),
            lambda _snap: worker.stats()["resting"],
            6,
        )
        assert worker.stats()["resting"] is True
    finally:
        worker.stop()


def test_lingering_shadow_across_all_post_exit_passes_still_confirms_and_border_rests(
    monkeypatch,
):
    """Sombra nao mascarada persiste por todas as 3 passadas pos-saida
    enquanto o modelo detecta o traco: o orcamento congelado mantem o worker
    acordado ate limpar e confirmar, e uma borda escura permanente ainda deixa
    o worker descansar."""
    config = dataclasses.replace(
        InkConfig.production(), page_model=False, paper_guard=True
    )
    worker, ink, generation = _mature(monkeypatch, config, idle_probe=True)
    try:
        border_paper = _paper()
        border_paper[:, 336:] = (60, 62, 64)
        late = (slice(224, 240), slice(16, 120))
        shadow = border_paper.copy()
        shadow[(slice(208, 256), slice(0, 160))] = (60, 62, 64)

        worker.set_hand_presence(True)
        ink.content = _prob(extra=late)
        for _ in range(2):
            _pass(worker, generation, shadow)

        worker.set_hand_presence(False)
        for _ in range(worker._post_exit_confirm_passes()):
            _pass(worker, generation, shadow)

        assert not worker.poll(generation, 0).confirmed[late].any()
        assert not worker.stats()["resting"]

        _pass_until(
            worker,
            generation,
            border_paper,
            lambda snap: snap.confirmed[late].all(),
            config.occlusion_hold_passes + 8,
        )
        assert worker.poll(generation, 0).confirmed[late].all()

        _pass_until(
            worker,
            generation,
            border_paper,
            lambda _snap: worker.stats()["resting"],
            6,
        )
        assert worker.stats()["resting"] is True
    finally:
        worker.stop()


def test_uncertain_reference_after_valid_frame_freezes_instead_of_reusing_old_reference(
    monkeypatch,
):
    """Amostras visiveis insuficientes (< _MIN_SAMPLES) apos uma referencia
    valida congelam a passada inteira em vez de reutilizar _paper_ref."""
    worker, ink, generation = _run(monkeypatch, InkConfig())
    try:
        late = (slice(220, 244), slice(16, 40))
        ink.content = _prob(extra=late)
        visible = np.zeros((H, W), bool)
        visible[late] = True
        for _ in range(6):
            _pass(worker, generation, _paper(), visible=visible)
        assert not worker.poll(generation, 0).confirmed[late].any()
    finally:
        worker.stop()
