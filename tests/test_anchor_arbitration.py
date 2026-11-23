"""Arbitragem de ancora por conteudo (v3, frente 19 de 2026-11-10).

O defeito (FALHAS M2): no pouso sustentado, migrar o estado pela
inversa do registro (v1) PIORA a duplicacao (0,69-1,14 contra
0,44-0,77 da base), porque a inversa assume a tinta parada na
camera; no modo espurio, trocar so a ancora erra o contrario
(0,74). As hipoteses sao indistinguiveis pelo quad. A v3 arbitra
pelo conteudo (Sp = massa da prob sobre o canvas confirmado na
colocacao identidade; Sc = na colocacao registrada) num streak de
assentamento, e — o mecanismo que faltava na v2 — a passada de
evento aberto NAO vota: a copia deslocada confirmava pelo voto
antes de qualquer veredito.

Contratos:
1. Pouso (Sp >> Sc): exatamente 1 troca de ancora, estado
   confirmado byte-identico ao pre-evento.
2. Espurio (Sc >> Sp): nenhuma acao, ancora intacta, canvas = base.
3. Segmentacao: passadas de meio-movimento (inconsistentes) nao
   fecham o streak; a troca so vem apos a pose assentar.
4. Abstencao: nenhuma confirmacao nasce de passada de evento.
5. Sem massa para julgar: troca por default (custo zero nas duas
   hipoteses).
"""

import cv2
import numpy as np
import pytest

from draw.ink import registration as registration_mod
from draw.ink import worker as worker_mod
from rr import BASE, SHAPE, _drive, _page, _ProbIsWarp, _run, _settled

STEP = BASE + np.float32([120.0, 0.0])


@pytest.mark.slow
def test_landing_swaps_anchor_without_touching_state(fast):
    content = _page()
    sequence = _settled(content) + [(content, STEP.copy()) for _ in range(3)]
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing"]
    assert events[0]["score_page"] > 0.5
    # O contrato e a separacao (na bancada real, Sc <= 0,012); aqui a
    # pagina miniatura tem 85 px de largura e o degrau de 120 px
    # degenera a homografia, entao Sc mede 0,06 — a margem segue
    # larga.
    assert events[0]["score_page"] - events[0]["score_camera"] > 0.4
    np.testing.assert_allclose(worker._anchor_corners, STEP)
    # O estado nao se moveu: o canvas confirmado e o mesmo que as
    # passadas assentadas construiram (a tinta esta na grade da
    # pagina; quem anda e a ancora).
    _w2, truth, _p2 = _run(_settled(content))
    assert np.array_equal(confirmed, truth)


@pytest.mark.slow
def test_spurious_mode_keeps_anchor_and_canvas(fast):
    content = _page()
    # O quad mente: a prob chega distorcida pela homografia inversa
    # do registro; registrada contra a ancora, ela volta ao lugar.
    matrix = registration_mod._registration_matrix(BASE, STEP, SHAPE)
    distorted = cv2.warpPerspective(
        content, np.linalg.inv(matrix), (SHAPE[1], SHAPE[0])
    ).astype(np.float32)
    sequence = _settled(content) + [(distorted, STEP.copy()) for _ in range(3)]
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot.get("anchor_arbitration_actions", 0) == 0
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["spurious"]
    assert events[0]["score_camera"] > 0.5
    assert events[0]["score_page"] < 0.05
    np.testing.assert_allclose(worker._anchor_corners, BASE)
    _w2, truth, _p2 = _run(_settled(content))
    assert np.array_equal(confirmed, truth)


@pytest.mark.slow
def test_midmovement_passes_do_not_close_the_streak(fast):
    content = _page()
    ramp = [
        (content, (BASE + np.float32([dx, 0.0])).astype(np.float32))
        for dx in (30.0, 60.0, 90.0)
    ]
    settled_step = [(content, STEP.copy()) for _ in range(3)]
    worker, _confirmed, _provisional = _run(_settled(content) + ramp + settled_step)
    snapshot = worker.telemetry_snapshot()
    # As passadas da rampa abstiveram e nenhuma acao nasceu delas;
    # a unica troca vem do streak assentado no fim.
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert len(events) == 1 and events[0]["verdict"] == "landing"
    records = snapshot["records"]
    abstained = [bool(r.get("arb_abstained", False)) for r in records]
    assert abstained[4:7] == [True, True, True]


@pytest.mark.slow
def test_open_event_births_no_confirmation(fast):
    content = _page()
    other = np.zeros(SHAPE, np.float32)
    other[10:14, 10:40] = 1.0  # tinta nova, fora do canvas confirmado
    sequence = _settled(content) + [(other, STEP.copy()) for _ in range(2)]
    worker = worker_mod.PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, sequence)
        _disp = worker.poll_display(generation, 0)
        _version, confirmed, _provisional = (
            _disp.version,
            _disp.confirmed,
            _disp.provisional,
        )
    finally:
        worker.stop()
    _w2, truth, _p2 = _run(_settled(content))
    assert np.array_equal(confirmed, truth), (
        "passada de evento aberto nao pode confirmar tinta nova"
    )


def _permute(img, k):
    # A construcao FISICA da frente 23 (independente do codigo sob
    # teste): a prob que o warp posicional entrega quando a folha
    # assentou rodada de k x 90 graus.
    k %= 4
    if k == 0:
        return img
    if k == 2:
        return np.rot90(img, 2).copy()
    rotated = np.rot90(img, 1 if k == 1 else -1)
    return cv2.resize(
        rotated, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2])
def test_rotated_landing_relabels_anchor_by_k(fast, k):
    # Frente 23/24: sob rotacao, a correspondencia posicional nao
    # distingue "folha rodada" de "quad errado" — o conteudo sim,
    # pontuado como permutacao da imagem entregue. Geometria
    # dedicada onde a pagina ENCHE o quadro entregue (como no
    # pipeline real, em que a prob e o warp da pagina): o veredito
    # tem que ser landing_rotated com o k fisico, a ancora carrega
    # os rotulos re-rotulados, o estado nao se move e as passadas
    # seguintes (ainda entregues rodadas) confirmam sobre o MESMO
    # canvas — e esse ultimo ponto que fixa a direcao do roll.
    full = np.array(
        [[0.0, 0.0], [199.0, 0.0], [199.0, 99.0], [0.0, 99.0]], dtype=np.float32
    )
    moved = full + np.array([30.0, 10.0], dtype=np.float32)
    content = np.zeros(SHAPE, dtype=np.float32)
    content[15:22, 30:110] = 1.0
    content[45:70, 55:63] = 1.0
    content[75:82, 120:180] = 1.0
    rotated = _permute(content, k)
    sequence = (
        [(content, full.copy()) for _ in range(3)]
        + [(rotated, moved.copy()) for _ in range(3)]
        + [(rotated, moved.copy()) for _ in range(3)]
    )
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing_rotated"]
    assert events[0]["rotation_k"] == k
    np.testing.assert_allclose(worker._anchor_corners, np.roll(moved, k, axis=0))
    _tw, truth, _p2 = _run([(content, full.copy()) for _ in range(3)])
    # Retencao exata (nada do canvas se perde) e IoU alto; a
    # igualdade byte a byte dos outros contratos nao esta
    # disponivel aqui: a miniatura reamostra a prob de forma
    # anisotropica e a permutacao acontece antes dessa reamostragem
    # num braco e nao no outro — a borda de ~1 px e artefato da
    # miniatura; quem julga o numero real e a bancada da frente 23.
    assert np.count_nonzero(truth & ~confirmed) == 0
    inter = np.count_nonzero(confirmed & truth)
    union = np.count_nonzero(confirmed | truth)
    assert inter / union >= 0.85


@pytest.mark.slow
def test_straight_landing_after_rotated_landing(fast):
    # Frente 27 (pousos_3_rot): a era 3 — pouso RETO sobre a grade
    # rolada — julga-se pela hipotese-pagina compensada (a prob
    # des-rotada pelo roll parado: 0,193 medidos na captura contra
    # 0,002 do Sc) e a ancora nova e a mediana do streak rolada
    # pelo roll parado, que se preserva. Gemeo dos contratos de
    # pouso rodado, mesma construcao fisica.
    full = np.array(
        [[0.0, 0.0], [199.0, 0.0], [199.0, 99.0], [0.0, 99.0]], dtype=np.float32
    )
    moved = full + np.array([30.0, 10.0], dtype=np.float32)
    moved2 = full + np.array([70.0, 25.0], dtype=np.float32)
    content = np.zeros(SHAPE, dtype=np.float32)
    content[15:22, 30:110] = 1.0
    content[45:70, 55:63] = 1.0
    content[75:82, 120:180] = 1.0
    rotated = _permute(content, 2)
    sequence = (
        [(content, full.copy()) for _ in range(3)]
        + [(rotated, moved.copy()) for _ in range(6)]
        + [(rotated, moved2.copy()) for _ in range(6)]
    )
    worker, confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 2
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing_rotated", "landing"]
    assert events[1]["score_page_roll_compensated"] is True
    assert worker._anchor_roll == 2
    np.testing.assert_allclose(worker._anchor_corners, np.roll(moved2, 2, axis=0))
    _tw, truth, _p2 = _run([(content, full.copy()) for _ in range(3)])
    assert np.count_nonzero(truth & ~confirmed) == 0
    inter = np.count_nonzero(confirmed & truth)
    union = np.count_nonzero(confirmed | truth)
    assert inter / union >= 0.85


@pytest.mark.slow
def test_no_mass_defaults_to_anchor_swap(fast):
    content = _page()
    empty = np.zeros(SHAPE, np.float32)
    # A geracao assenta numa pagina vazia (ancora nasce, nada
    # confirma) e a folha anda antes de qualquer tinta: sem massa
    # para julgar, a troca e por default.
    sequence = [(empty, BASE.copy()) for _ in range(3)] + [
        (content, STEP.copy()) for _ in range(3)
    ]
    worker, _confirmed, _provisional = _run(sequence)
    snapshot = worker.telemetry_snapshot()
    assert snapshot["anchor_arbitration_actions"] == 1
    events = snapshot["anchor_arbitration_events"]
    assert [e["verdict"] for e in events] == ["landing_default"]
    np.testing.assert_allclose(worker._anchor_corners, STEP)


def _shifted(img, dx, dy):
    out = np.zeros_like(img)
    h, w = img.shape
    y0, y1, x0, x1 = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = img[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return out


@pytest.mark.slow
@pytest.mark.parametrize(
    "residual", [(0, 0), (4, 0), (3, -5)], ids=["T0", "T4x", "T3_5"]
)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotated_landing_with_an_active_residual(fast, k, residual):
    """O resíduo entra depois da permutação: T . Pi_k(P)."""
    dx, dy = residual
    full = np.array(
        [[0.0, 0.0], [199.0, 0.0], [199.0, 99.0], [0.0, 99.0]], dtype=np.float32
    )
    moved = full + np.array([30.0, 10.0], dtype=np.float32)
    content = np.zeros(SHAPE, dtype=np.float32)
    content[15:22, 30:110] = 1.0
    content[45:70, 55:63] = 1.0
    content[75:82, 120:180] = 1.0
    evidence = _shifted(content, -dx, -dy)
    rotated = _permute(evidence, k)
    worker = worker_mod.PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, [(content, full.copy())])
        worker._identity_residual = (dx, dy)
        _drive(worker, generation, [(evidence, full.copy()) for _ in range(3)])
        seen = len(worker.telemetry_snapshot()["anchor_arbitration_events"])
        _drive(worker, generation, [(rotated, moved.copy()) for _ in range(6)])
        events = worker.telemetry_snapshot()["anchor_arbitration_events"][seen:]
        assert [event["verdict"] for event in events] == ["landing_rotated"]
        assert events[0]["rotation_k"] == k
        assert events[0]["scores_cyclic"][k] >= 0.9
        np.testing.assert_allclose(worker._anchor_corners, np.roll(moved, k, axis=0))
        confirmed = worker.poll_display(generation, 0).confirmed
    finally:
        worker.stop()
    _tw, truth, _p2 = _run([(content, full.copy()) for _ in range(3)])
    assert np.count_nonzero(truth & ~confirmed) == 0
    inter = np.count_nonzero(confirmed & truth)
    union = np.count_nonzero(confirmed | truth)
    assert inter / union >= 0.85
