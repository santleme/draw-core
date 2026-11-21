"""Estabilização do rastreador de folha (auditoria de 2026-11-02).

Nos frames reais de uma cena parada, os cantos crus do detector
andam dezenas de pixels por frame e os cantos exibidos andavam
junto (passo médio de 9 px, máximo de 101 px, medido em
laboratório pelo agente). Estes testes pinam as três defesas:

1. banda morta: detecção filtrada a até _SHEET_DEADBAND_PX do
   quad atual confirma presença sem mover o quad um pixel;
2. mediana das últimas 3 detecções: um salto isolado de um frame
   não move o quad;
3. candidato a salto fixo: detecções errantes que "andam" não
   confirmam salto — só uma posição sustentada pelo tempo de
   persistência confirma, e aí o quad vai para ela.
"""

import numpy as np
import pytest

from draw import tracker as tracker_module

BASE = np.array(
    [[200.0, 100.0], [1000.0, 120.0], [1050.0, 650.0], [150.0, 680.0]],
    dtype=np.float32,
)


def _feed(tracker, corners, verdict, times):
    for t in times:
        tracker.update(np.asarray(corners, dtype=np.float32), verdict, True, t)


def test_small_noise_keeps_the_displayed_quad_within_a_pixel():
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    rng = np.random.default_rng(42)
    for i in range(1, 40):
        noisy = (BASE + rng.normal(0.0, 1.0, BASE.shape)).astype(np.float32)
        tracker.update(noisy, "paper", True, i * 0.2)
        distance = tracker_module.corner_distance(tracker.corners, BASE)
        assert distance <= 1.5, (
            f"ruído de ±1 px arrastou o quad exibido {distance:.2f} px: "
            "dentro da banda morta o quad não se move, e fora dela a "
            "mediana limita o passo a uma fração do desvio"
        )


def test_single_frame_spike_does_not_move_the_quad():
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    spiked = BASE + np.array([120.0, 0.0], dtype=np.float32)
    tracker.update(spiked, "paper", True, 0.2)
    tracker.update(BASE.copy(), "paper", True, 0.4)
    tracker.update(BASE.copy(), "paper", True, 0.6)
    assert tracker_module.corner_distance(tracker.corners, BASE) <= (
        tracker_module._SHEET_DEADBAND_PX
    )


def test_wandering_candidate_never_confirms_a_jump():
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    # Dois modos errantes alternados, cada um longe do quad e um
    # do outro: nenhum sustenta posição por 0,3 s.
    mode_a = BASE + np.array([150.0, 0.0], dtype=np.float32)
    mode_b = BASE + np.array([-140.0, 30.0], dtype=np.float32)
    t = 0.2
    for i in range(12):
        tracker.update(mode_a if i % 2 == 0 else mode_b, "paper", True, t)
        t += 0.2
    assert tracker_module.corner_distance(tracker.corners, BASE) <= (
        tracker_module._SHEET_DEADBAND_PX
    )


def test_sustained_new_position_still_confirms():
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    moved = BASE + np.array([150.0, 0.0], dtype=np.float32)
    _feed(tracker, moved, "paper", [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    assert tracker_module.corner_distance(tracker.corners, moved) <= (
        tracker_module._SHEET_NEAR_LIMIT_PX
    )


def test_big_jump_needs_the_longer_persistence():
    """Salto >100 px sustentado por 0,5 s ainda não confirma
    (assinatura de troca de modo do detector); sustentado por
    mais de 1 s, confirma — mover a folha de verdade paga o
    tempo, não perde o movimento."""
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    moved = BASE + np.array([150.0, 0.0], dtype=np.float32)
    _feed(tracker, moved, "paper", [0.2, 0.4, 0.6])
    assert tracker_module.corner_distance(tracker.corners, BASE) <= (
        tracker_module._SHEET_DEADBAND_PX
    )
    _feed(tracker, moved, "paper", [0.8, 1.0, 1.2, 1.4, 1.6])
    assert tracker_module.corner_distance(tracker.corners, moved) <= (
        tracker_module._SHEET_NEAR_LIMIT_PX
    )


def test_blank_maybe_confirms_under_noisy_detections():
    """Folha branca (blank_maybe) sob ruído que antes nunca
    confirmava: no código anterior, com σ=20 px o relógio do
    candidato resetava a cada detecção e 6 s não bastavam
    (simulador da auditoria, antes/depois medidos). Com o filtro
    de mediana, confirma no mínimo desenhado de 0,6 s."""
    tracker = tracker_module.SheetTracker()
    rng = np.random.default_rng(7)
    confirmed_at = None
    for i in range(40):
        noisy = (BASE + rng.normal(0.0, 20.0, BASE.shape)).astype(np.float32)
        tracker.update(noisy, "blank_maybe", True, i * 0.15)
        if tracker.corners is not None:
            confirmed_at = i * 0.15
            break
    assert confirmed_at is not None and confirmed_at <= 3.0, (
        f"blank_maybe sob ruído σ=20 não confirmou (t={confirmed_at})"
    )


def test_blank_candidate_does_not_survive_absence():
    """blank_maybe que volta apos ausencia nao confirma na hora.

    Candidato pendente morre numa deteccao negativa mesmo antes da
    primeira confirmacao: sem a limpeza, o _pending_since velho
    satisfaz o dobro de persistencia no primeiro quadro da volta.
    """
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "blank_maybe", True, 0.0)
    assert tracker.has_pending
    tracker.update(None, None, True, 2.0)  # deteccao rodou, nada visto
    assert not tracker.has_pending
    event = tracker.update(BASE.copy(), "blank_maybe", True, 5.0)
    assert event is None
    assert tracker.corners is None
    # Confirma so depois do dobro de persistencia do candidato novo.
    event = tracker.update(
        BASE.copy(),
        "blank_maybe",
        True,
        5.0 + 2 * tracker_module._SHEET_JUMP_PERSIST_SECONDS + 0.05,
    )
    assert event == "seen"


def test_paper_without_quad_keeps_blank_candidate_alive():
    """'paper' sem quadrilatero nao reinicia a persistencia do
    candidato blank_maybe.

    Antes da primeira confirmacao, o retorno de 'so presenca' exige
    corners confirmados; o paper-sem-quad caia no ramo de limpeza
    pre-confirmacao e apagava candidato, timer e janela da mediana.
    So deteccao negativa de verdade (veredito != 'paper') limpa.
    """
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "blank_maybe", True, 0.0)
    assert tracker.has_pending
    # Detector afirma papel mas nao fecha quad: presenca sem posicao.
    tracker.update(None, "paper", True, 0.3)
    assert tracker.has_pending
    # O candidato original confirma no prazo dobrado de blank_maybe
    # contado desde a PRIMEIRA deteccao, nao do reinicio.
    event = tracker.update(
        BASE.copy(),
        "blank_maybe",
        True,
        2 * tracker_module._SHEET_JUMP_PERSIST_SECONDS + 0.05,
    )
    assert event == "seen"


def test_returning_sheet_does_not_inherit_stale_candidate_corners():
    """Folha em outra posicao nao herda cantos do candidato morto.

    A janela do filtro tambem faz parte do estado de candidato: sem
    a limpeza, a mediana mistura a posicao antiga com a nova e o
    candidato renasce puxado para os cantos antigos.
    """
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "blank_maybe", True, 0.0)
    tracker.update(None, None, True, 2.0)
    moved = BASE + np.array([300.0, 200.0], dtype=np.float32)
    tracker.update(moved, "blank_maybe", True, 3.0)
    assert tracker.has_pending
    distance = tracker_module.corner_distance(tracker._pending, moved)
    assert distance <= 1.5, (
        f"candidato novo nasceu {distance:.1f} px longe da deteccao: "
        "a janela do filtro herdou os cantos antigos"
    )


def test_camera_pixel_scale_uses_the_frame_diagonal():
    """(h, w) -> diagonal do quadro / diagonal de referencia 1280x720."""
    scale = tracker_module.camera_pixel_scale
    assert scale((720, 1280)) == pytest.approx(1.0)
    assert scale((360, 640)) == pytest.approx(0.5)
    assert scale((1440, 2560)) == pytest.approx(2.0)
    # Retrato: a diagonal ignora a orientacao do quadro.
    assert scale((1280, 720)) == pytest.approx(1.0)


def test_deadband_scales_with_pixel_scale():
    """Com pixel_scale=2.0 a banda morta efetiva e 4 px: uma
    deteccao a 3 px confirma presenca sem mover o quad. Na
    referencia (1,0) os mesmos 3 px passam da banda de 2 px e
    misturam."""
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    shifted = BASE + np.array([3.0, 0.0], dtype=np.float32)
    # Deteccoes repetidas: a mediana devolve os 3 px inteiros.
    for i in range(1, 4):
        event = tracker.update(shifted.copy(), "paper", True, i * 0.2, pixel_scale=2.0)
        assert event == "seen"
    assert tracker_module.corner_distance(tracker.corners, BASE) == 0.0
    # Referencia 1,0: fora da banda, os 3 px misturam e o quad anda.
    tracker_ref = tracker_module.SheetTracker()
    tracker_ref.update(BASE.copy(), "paper", True, 0.0)
    for i in range(1, 4):
        tracker_ref.update(shifted.copy(), "paper", True, i * 0.2)
    assert tracker_module.corner_distance(tracker_ref.corners, BASE) > 0.0


def test_near_limit_scales_with_pixel_scale():
    """Deteccao a ~60 px: com pixel_scale=2.0 cabe no limite de
    vizinhanca escalado (80 px) e mistura sem candidato; na
    referencia passa dos 40 px e abre candidato a salto."""
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    shifted = BASE + np.array([60.0, 0.0], dtype=np.float32)
    for t in (0.1, 0.2, 0.3):
        event = tracker.update(shifted.copy(), "paper", True, t, pixel_scale=2.0)
        assert event == "seen"
    assert not tracker.has_pending
    assert tracker_module.corner_distance(tracker.corners, BASE) > 10.0
    # Referencia 1,0: 60 px > 40 abre candidato que pede persistencia.
    tracker_ref = tracker_module.SheetTracker()
    tracker_ref.update(BASE.copy(), "paper", True, 0.0)
    for t in (0.1, 0.2, 0.3):
        tracker_ref.update(shifted.copy(), "paper", True, t)
    assert tracker_ref.has_pending
    assert tracker_module.corner_distance(tracker_ref.corners, BASE) <= 15.0


def test_big_jump_threshold_scales_with_pixel_scale():
    """Candidato a ~150 px: com pixel_scale=2.0 fica abaixo do
    salto grande escalado (200 px) e confirma em 0,3 s; na
    referencia passa dos 100 px e exige 1,0 s de persistencia."""
    tracker = tracker_module.SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    moved = BASE + np.array([150.0, 0.0], dtype=np.float32)
    events = [
        tracker.update(moved.copy(), "paper", True, t, pixel_scale=2.0)
        for t in (0.1, 0.2, 0.3, 0.4, 0.5)
    ]
    # O candidato fixa em t=0,2 e confirma ~0,3 s depois; a
    # distancia de confirmacao (~127 px) nao marca salto grande.
    assert events[-1] == "seen"
    assert tracker.big_jump_seq == 0
    # Referencia 1,0: o mesmo candidato e salto grande — pendente
    # em 0,5 s, confirma so depois de 1,0 s de persistencia.
    tracker_ref = tracker_module.SheetTracker()
    tracker_ref.update(BASE.copy(), "paper", True, 0.0)
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        tracker_ref.update(moved.copy(), "paper", True, t)
    assert tracker_ref.has_pending
    event = tracker_ref.update(moved.copy(), "paper", True, 1.3)
    assert event == "seen"
    assert tracker_ref.big_jump_seq == 1


def test_update_rejects_non_positive_or_non_finite_pixel_scale():
    tracker = tracker_module.SheetTracker()
    for bad in (0.0, -1.5, float("nan"), float("inf"), "dois"):
        with pytest.raises(ValueError):
            tracker.update(BASE.copy(), "paper", True, 0.0, pixel_scale=bad)
    # Escala invalida nao altera o estado nem a escala corrente.
    assert tracker.corners is None
    assert tracker.diagnostic_state()["pixel_scale"] == 1.0


def test_diagnostic_state_reports_pixel_scale():
    tracker = tracker_module.SheetTracker()
    assert tracker.diagnostic_state()["pixel_scale"] == 1.0
    tracker.update(BASE.copy(), "paper", True, 0.0, pixel_scale=2.0)
    assert tracker.diagnostic_state()["pixel_scale"] == 2.0
    tracker.update(BASE.copy(), "paper", True, 0.2)
    assert tracker.diagnostic_state()["pixel_scale"] == 1.0
