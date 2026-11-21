"""Registro da tinta na geometria ancora da geracao.

O defeito (regua de replay, 2026-11-05, draw-lab/parallel-2026-
11-05/repassada): a tinta assentada era REIMPRESSA sem fim. O
acumulador vive numa grade fixa, mas cada passada retifica a
pagina com o quad do momento — e o quad anda (vies de canto
pela mao, deriva da mistura do rastreador, tremor de +-1 px).
A evidencia entrava na posicao da geometria de cada passada e
o voto confirmava a copia de cada alinhamento: a confirmada
degenerava para a uniao de todas as alinhagens (na pagina real
da regua: 20.137 px com geometria congelada contra 45.528 px
com tremor de +-1 px e 71.548 px com vies sustentado).

O contrato novo:

1. A matriz de registro leva o mapa da passada para a grade
   da ancora (identidade quando o quad nao andou).
2. Com os cantos informados no submit, a tinta assentada NAO
   cresce sob geometria que anda: o canvas final e o da
   geometria congelada, pixel a pixel.
3. Tinta nova continua entrando na hora sob geometria que
   anda: provisoria na 1a passada, confirmada pelo voto.
4. Sem cantos (chamadores antigos), nada muda: a evidencia
   entra na propria geometria, como antes do registro.
5. Cantos malformados sao ignorados, nunca derrubam a thread.
6. No replay de recuperacao, mover a mesma pagina reusa o canvas
   confirmado pixel a pixel; conteudo divergente abre canvas limpo.
"""

import time

import cv2
import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import accumulator as accumulator_mod
from draw.ink import registration as registration_mod
from draw.ink import worker as worker_mod

H, W = 60, 80
Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)


BLOCK_A = (slice(10, 20), slice(10, 20))
BLOCK_B = (slice(35, 45), slice(55, 65))


def _shift(img, dx, dy):
    if dx == 0 and dy == 0:
        return img
    out = np.zeros_like(img)
    h, w = img.shape
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    xd0, xd1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    yd0, yd1 = max(0, dy), min(h, h + dy)
    out[yd0:yd1, xd0:xd1] = img[ys0:ys1, xs0:xs1]
    return out


def _content(with_b=False):
    prob = np.zeros((H, W), dtype=np.float32)
    prob[BLOCK_A] = 0.95
    if with_b:
        prob[BLOCK_B] = 0.95
    return prob


class _WanderingInk:
    """Tinta falsa: devolve o conteudo do mundo deslocado pelo
    offset de geometria corrente, como o modelo real devolveria
    ao retificar a pagina com o quad deslocado."""

    threshold = 0.7

    def __init__(self):
        self.offset = (0, 0)
        self.with_b = False
        self.calls = 0

    def ink_prob(self, _warped):
        self.calls += 1
        return _shift(_content(self.with_b), *self.offset)


# Deslocamentos de geometria da sequencia: deriva de ida e
# volta de ate 3 px, sustentada por algumas passadas em cada
# degrau (o regime medido da cadeia rastreador+mediana).
WANDER = (
    [(0, 0)] * 4
    + [(1, 0)] * 4
    + [(2, 1)] * 4
    + [(3, 1)] * 6
    + [(2, 0)] * 4
    + [(1, 0)] * 4
    + [(0, 0)] * 4
)


def _drive(worker, generation, ink, offsets, with_corners=True):
    """Submete uma passada por offset e espera ela executar."""
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    for off in offsets:
        ink.offset = off
        before = worker.stats()["passes_total"]
        if with_corners:
            corners = Q0 - np.array(off, dtype=np.float32)
            worker.submit(WarpedPage(dummy, generation, corners))
        else:
            worker.submit(WarpedPage(dummy, generation))
        deadline = time.monotonic() + 5.0
        while worker.stats()["passes_total"] <= before:
            if time.monotonic() > deadline:
                raise AssertionError("a passada nao executou")
            time.sleep(0.001)


def _static_confirmed(passes):
    acc = accumulator_mod.LayeredInkAccumulator(0.7)
    result = None
    for _ in range(passes):
        result = acc.add(_content())
    return result["confirmed"]


def test_registration_matrix_identity_for_same_corners():
    matrix = registration_mod._registration_matrix(Q0, Q0, (H, W))
    assert registration_mod._registration_is_identity(matrix)


def test_registration_matrix_undoes_geometry_shift():
    pattern = np.zeros((H, W), dtype=np.float32)
    pattern[15:25, 20:35] = 0.9
    pattern[40, 60] = 1.0
    # A camera anda por t: o quad da passada e Q0 + t e o
    # conteudo aparece deslocado por -t na pagina da passada.
    t = (3, -2)
    corners = Q0 + np.array(t, dtype=np.float32)
    prob_pass = _shift(pattern, -t[0], -t[1])
    matrix = registration_mod._registration_matrix(Q0, corners, (H, W))
    assert not registration_mod._registration_is_identity(matrix)
    registered = cv2.warpPerspective(
        prob_pass,
        matrix,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderValue=0.0,
    )
    inner = (slice(4, -4), slice(4, -4))
    assert np.max(np.abs(registered[inner] - pattern[inner])) < 1e-4


@pytest.mark.slow
def test_settled_ink_not_reprinted_under_wandering_geometry(
    monkeypatch,
):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _WanderingInk()
    worker = worker_mod.PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, ink, WANDER)
        got = worker.poll(generation, 0)
        assert got is not None
        _version, canvas = got.version, got.confirmed
        expected = _static_confirmed(len(WANDER))
        assert int(np.count_nonzero(canvas)) == int(np.count_nonzero(expected))
        assert np.array_equal(canvas > 0, expected > 0)
    finally:
        worker.stop()


@pytest.mark.slow
def test_new_ink_confirms_fast_under_wandering_geometry(
    monkeypatch,
):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _WanderingInk()
    worker = worker_mod.PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, ink, WANDER[:12])
        settled = worker.poll(generation, 0)
        assert settled is not None
        assert np.count_nonzero(settled.confirmed[BLOCK_B]) == 0

        # Tinta nova entra no mundo com a geometria andando.
        ink.with_b = True
        _drive(worker, generation, ink, WANDER[12:13])
        shown = worker.poll_display(generation, 0)
        assert shown is not None
        _v, confirmed, provisional = shown.version, shown.confirmed, shown.provisional
        assert (
            np.count_nonzero(confirmed[BLOCK_B])
            + np.count_nonzero(provisional[BLOCK_B])
        ) == 100, "a tinta nova aparece na 1a passada (provisoria)"
        _drive(worker, generation, ink, WANDER[13:15])
        got = worker.poll(generation, 0)
        assert got is not None
        assert np.count_nonzero(got.confirmed[BLOCK_B]) == 100, (
            "a tinta nova confirma pelo voto em ate 3 passadas"
        )
        # E o bloco assentado nao engordou com a viagem.
        assert np.count_nonzero(got.confirmed[BLOCK_A]) == 100
    finally:
        worker.stop()


@pytest.mark.slow
def test_moved_page_recovery_reuses_identical_canvas(monkeypatch):
    """Drop/reaparecimento com geometria nova e so re-registro."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _WanderingInk()
    worker = worker_mod.PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, ink, WANDER)
        original = worker.poll(generation, 0)
        assert original is not None
        canvas_before = original.confirmed.copy()
        assert np.count_nonzero(canvas_before) == 100

        # A folha desaparece do tracker e volta deslocada. A nova
        # geração do app pede ao worker para provar identidade pelo
        # conteúdo já confirmado, registrado na grade antiga.
        generation = worker.new_generation(preserve_identity=True)
        ink.offset = (2, -1)
        _drive(worker, generation, ink, [(2, -1)])
        recovered = worker.poll(generation, 0)
        assert recovered is not None
        assert np.array_equal(recovered.confirmed, canvas_before), (
            "mover a mesma página deve reusar o canvas sem acumular uma cópia deslocada"
        )
        identity = next(
            record
            for record in worker.telemetry_snapshot()["records"]
            if record["generation"] == generation
            and record["page_identity_decision"] is not None
        )
        assert identity["page_identity_decision"] == "same_page"
        assert identity["page_identity_recall"] == 1.0
        assert identity["page_identity_reference_px"] == 100
        assert identity["page_identity_geometry_registered"]
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert events[-1]["threshold"] == 0.65
    finally:
        worker.stop()


@pytest.mark.slow
def test_changed_page_content_starts_a_clean_canvas(monkeypatch):
    """Conteudo sem identidade com o anterior nao herda os tracos."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)

    class _NextPageInk(_WanderingInk):
        next_page = False

        def ink_prob(self, _warped):
            self.calls += 1
            if self.next_page:
                result = np.zeros((H, W), dtype=np.float32)
                result[BLOCK_B] = 0.95
                return result
            return _content()

    ink = _NextPageInk()
    worker = worker_mod.PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, ink, [(0, 0), (0, 0)])
        original = worker.poll(generation, 0)
        assert original is not None
        assert np.count_nonzero(original.confirmed[BLOCK_A]) == 100

        generation = worker.new_generation(preserve_identity=True)
        ink.next_page = True
        for _ in range(2):
            _drive(worker, generation, ink, [(0, 0)])
            assert (
                worker._identity_events[-1]["decision"] == "deferred_switch_consensus"
            )
            np.testing.assert_array_equal(
                worker.poll(generation, 0).confirmed, original.confirmed
            )
        _drive(worker, generation, ink, [(0, 0)])
        shown = worker.poll_display(generation, 0)
        assert shown is not None
        _version, confirmed, provisional = (
            shown.version,
            shown.confirmed,
            shown.provisional,
        )
        assert np.count_nonzero(confirmed[BLOCK_A]) == 0
        assert np.count_nonzero(provisional[BLOCK_A]) == 0
        assert np.count_nonzero(provisional[BLOCK_B]) == 100
        identity = next(
            record
            for record in worker.telemetry_snapshot()["records"]
            if record["generation"] == generation
            and record["page_identity_decision"] == "new_page"
        )
        assert identity["page_identity_decision"] == "new_page"
        assert identity["page_identity_recall"] == 0.0
        assert identity["page_identity_reference_px"] == 100
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
    finally:
        worker.stop()


@pytest.mark.slow
def test_without_corners_evidence_stays_unregistered(monkeypatch):
    """Sem cantos, o comportamento antigo segue: a geometria
    que anda re-projeta e a confirmada cresce — o registro e
    dos cantos, nao um efeito colateral da trabalhadora."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _WanderingInk()
    worker = worker_mod.PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, ink, WANDER, with_corners=False)
        got = worker.poll(generation, 0)
        assert got is not None
        grown = int(np.count_nonzero(got.confirmed))
        assert grown > int(np.count_nonzero(_static_confirmed(len(WANDER))))
    finally:
        worker.stop()


@pytest.mark.slow
def test_malformed_corners_are_ignored(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _WanderingInk()
    worker = worker_mod.PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        dummy = np.zeros((H, W, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            WarpedPage(dummy, generation, np.zeros((3, 2)))
        with pytest.raises(ValueError):
            WarpedPage(dummy, generation, "nao-e-quad")
        worker.submit(WarpedPage(dummy, generation))
        deadline = time.monotonic() + 5.0
        while worker.stats()["passes_total"] < 1:
            if time.monotonic() > deadline:
                raise AssertionError("a passada nao executou")
            time.sleep(0.001)
        assert worker.error is None
        assert worker.poll(generation, 0) is not None
    finally:
        worker.stop()


def test_registered_visibility_stays_inside_the_pass_footprint():
    """Reaquisicao com a folha noutra pose: a mascara de camera limpa
    projetaria "visivel" na grade inteira da ancora, mas a prob so
    cobre a pegada que M alcanca — fora dela nao houve observacao, e
    marca-la leria tinta velha fora do recorte como ausencia (review
    do PR #118)."""
    shape = (100, 140)
    prob = np.ones(shape, np.float32)
    anchor = np.array([[120, 60], [500, 60], [500, 300], [120, 300]], np.float32)
    registration = anchor + np.array([60, 0], np.float32)
    camera = np.ones((480, 640), bool)
    _prob, visible = registration_mod._register_ink_probability(
        prob,
        anchor,
        registration,
        visible_cam=camera,
        pass_corners=registration,
    )
    matrix = registration_mod._registration_matrix(anchor, registration, shape)
    footprint = (
        cv2.warpPerspective(
            np.ones(shape, np.uint8),
            matrix,
            (shape[1], shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    )
    assert footprint.any() and not footprint.all()
    assert visible is not None
    assert np.array_equal(visible, footprint)


def test_camera_visibility_follows_the_live_quad_through_deadband():
    """Banda morta segurando o quad de registro (R) enquanto a passada
    usa o quad ao vivo (C): a mascara tem de percorrer M @ C, nao A —
    senao desalinha ~tau px da tinta que protege (review do PR #118)."""
    shape = (100, 140)
    prob = np.ones(shape, np.float32)
    anchor = np.array([[120, 60], [500, 60], [500, 300], [120, 300]], np.float32)
    registration = anchor + np.array([60, 0], np.float32)
    live = registration + np.array([9, 0], np.float32)  # < tau=10px
    camera = np.ones((480, 640), bool)
    camera[:, :340] = False  # metade esquerda ocluida
    _prob, visible = registration_mod._register_ink_probability(
        prob,
        anchor,
        registration,
        visible_cam=camera,
        pass_corners=live,
    )
    size = (shape[1], shape[0])
    matrix = registration_mod._registration_matrix(anchor, registration, shape)
    footprint = (
        cv2.warpPerspective(
            np.ones(shape, np.uint8),
            matrix,
            size,
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    )
    expected = (
        cv2.warpPerspective(
            camera.astype(np.uint8),
            matrix @ registration_mod._corners_page_matrix(live, *size),
            size,
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    ) & footprint
    assert np.array_equal(visible, expected)
    # Nao-vacuo: a projecao antiga por A sozinha teria deslocado a
    # fronteira de oclusao junto com a tinta.
    old = (
        cv2.warpPerspective(
            camera.astype(np.uint8),
            registration_mod._corners_page_matrix(anchor, *size),
            size,
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    ) & footprint
    assert not np.array_equal(visible, old)


def test_off_crop_occlusion_does_not_hold_the_post_exit_window():
    """Mao na mesa fora do quad nao e folha mascarada: com a mascara
    de camera (issue #100), o rearme pos-saida olha o recorte do
    quad, nao o frame inteiro (review do PR #118)."""
    from draw.ink.config import InkConfig

    ink = _WanderingInk()
    worker = worker_mod.PageCleanWorker(ink, config=InkConfig.production())
    generation = worker.new_generation()
    worker.set_hand_presence(True)
    worker.set_hand_presence(False)
    assert worker._awaiting_clear_post_exit
    quad = np.array([[100, 100], [400, 100], [400, 400], [100, 400]], np.float32)
    visible = np.ones((480, 640), bool)
    visible[430:, 550:] = False  # mao na mesa, fora do quad
    worker.submit(
        WarpedPage(np.zeros((60, 80, 3), np.uint8), generation, quad, visible)
    )
    assert not worker._awaiting_clear_post_exit
    # O mesmo recorte governa o is_masked da passada: oclusao fora do
    # quad nao gasta a janela de confirmacao.
    assert worker_mod._crop_all_visible(visible, quad)
    inside = visible.copy()
    inside[200, 200] = False  # oclusao DENTRO do quad mascara a passada
    assert not worker_mod._crop_all_visible(inside, quad)


def test_off_crop_occlusion_lets_the_post_exit_countdown_advance():
    """A janela pos-saida tem de AVANCAR com a passada clara: com a
    mascara de camera, o is_masked so conta oclusao dentro do quad —
    oclusao na mesa nao pode rearmar o contador a cada passada
    (review do PR #118)."""
    from draw.ink.config import InkConfig

    ink = _WanderingInk()
    worker = worker_mod.PageCleanWorker(ink, config=InkConfig.production())
    generation = worker.new_generation()
    worker.set_hand_presence(True)
    worker.set_hand_presence(False)
    quad = np.array([[100, 100], [400, 100], [400, 400], [100, 400]], np.float32)
    visible = np.ones((480, 640), bool)
    visible[430:, 550:] = False  # mao na mesa, fora do quad
    passes = worker._post_exit_confirm_passes()
    worker.submit(
        WarpedPage(np.zeros((60, 80, 3), np.uint8), generation, quad, visible)
    )
    assert worker.pump()
    assert worker._confirm_passes == passes - 1, (
        f"contador {worker._confirm_passes} — a oclusao fora do "
        "recorte rearmou a janela em vez de deixa-la avancar"
    )
