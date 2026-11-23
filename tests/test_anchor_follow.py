"""Mobilidade de ancora por divergencia assentada (mecanismo fixo).

Frente 34 (2026-11-10; 34-desenho-ancora-movel.md): a vigia v3
fechou o split errado do puxao e expos a pergunta sem dono — quem
move a ancora quando ninguem divide a pagina? Sem resposta, a
ancora congela na pose de abertura, a vigia de identidade fica
cega a ~150 px pelo resto da geracao (F33 G1/G3) e a arbitragem
cicla sem veredito com o canvas congelado (livelock da F33).
O FOLLOW e a regra do Lucas (2026-11-01) como mecanismo: mover o
caderno = so re-registro por substituicao — ancora := mediana dos
quads do streak assentado, estado INTOCADO (a grade ja e a da
folha; deformar o estado pela homografia do raio da camera e o
carry da migracao v1, reprovado no FALHAS M2).

Contratos:
1. Divergencia que a arbitragem nao fecha (deriva lenta
   assentada) => exatamente 1 re-ancoragem, ancora na mediana do
   streak, canvas byte-identico ao pre-evento (preservacao
   1,0 >= 0,98) e poll() publica na passada da troca.
2. Tremor (<= ~3 px de grade) e spikes de 1 passada => 0
   re-ancoragens.
3. Abstencao: nenhuma confirmacao nasce de passada de evento
   aberto.
4. Deriva lenta (degraus assentados, consistencia par-a-par
   nunca satisfeita) re-ancora — o caso que a arbitragem nao
   fecha; com a arbitragem ligada ela nao produz veredito e o
   FOLLOW destrava.
5. Arbitragem tem o primeiro veredito: pouso decisivo pre-empta
   (0 re-ancoragens); spurious retem o episodio (0
   re-ancoragens, ancora intacta); indeterminate nao decide e o
   FOLLOW dispara — e a tinta volta a votar na pose nova.
6. Identidade pendente (recuperacao em quarentena ou juiz
   armado) trava a re-ancoragem: streak 0, 0 re-ancoragens.
7. O mecanismo nao dispara no replay assentado do GOLDEN (canvas
   byte-identico ao baseline).
"""

import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import rr
from draw.contracts import WarpedPage
from draw.ink import anchor as anchor_mod
from draw.ink import registration as registration_mod
from draw.ink import worker as worker_mod


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


discipline = _load(
    "test_reg_discipline_follow",
    Path(__file__).resolve().parent / "test_reg_discipline.py",
)

SHAPE = (120, 85)
BASE = rr.base_quad(SHAPE)
# Deslocamento sustentado: ~120 px de camera (~30 px da grade).
STEP = BASE + np.float32([120.0, 0.0])


class _ProbIsWarp:
    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.005)
    return monkeypatch


def _content():
    _drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=3)
    return settled[0].astype(np.float32)


def _drive(worker, generation, sequence):
    for prob, corners in sequence:
        before = worker.stats()["passes_total"]
        worker.submit(WarpedPage(prob, generation, corners))
        deadline = time.monotonic() + 5.0
        while worker.stats()["passes_total"] <= before:
            assert time.monotonic() < deadline, "a passada nao executou"
            time.sleep(0.001)


def _run(sequence, stop=True, config=None):
    worker = worker_mod.PageCleanWorker(_ProbIsWarp(), config=config).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, sequence)
        _disp = worker.poll_display(generation, 0)
        _version, confirmed, _provisional = (
            _disp.version,
            _disp.confirmed,
            _disp.provisional,
        )
    except BaseException:
        worker.stop()  # nunca entrega um worker vivo para ninguém
        raise
    if stop:
        worker.stop()
    return worker, generation, confirmed


def _settled(content, passes=4):
    return [(content, BASE.copy()) for _ in range(passes)]


def test_constants_reuse_the_migration_and_settled_specs():
    assert (
        anchor_mod._FOLLOW_DISPLACEMENT_PX
        == anchor_mod._ANCHOR_MIGRATION_DISPLACEMENT_PX
        == 8.0
    )
    assert anchor_mod._FOLLOW_PASSES == anchor_mod._ANCHOR_MIGRATION_PASSES == 3
    assert anchor_mod._FOLLOW_SETTLE_PX == 6.0


# --- 1. divergencia que a arbitragem nao fecha: 1 re-ancoragem ---


@pytest.mark.slow
def test_drift_divergence_reanchors_once_state_untouched(fast):
    """Com a arbitragem sempre ligada, a unica divergencia que o
    FOLLOW fecha e a deriva lenta: cada degrau assentado fica a
    <=6 px do anterior (o streak do FOLLOW cresce) mas a >6 px do
    primeiro quad do streak da arbitragem a cada 2 passadas (a
    consistencia par-a-par nunca fecha — nenhum veredito nasce e a
    geometria decide o episodio)."""
    content = _content()
    # settle 4 + deriva de 5 px/passo: o evento abre no meio do
    # creep (degrau de ~30 px de camera passa dos 8 px de grade)
    # e o FOLLOW dispara na 3a assentada do streak — medido no
    # indice 11 (streak {+30,+35,+40}) — e a publicacao e
    # forcada nela.
    sequence = _creep_sequence(content, steps=11)
    worker = worker_mod.PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, sequence[:11])
        _snap = worker.poll(generation, 0)
        version_mid, canvas_mid = _snap.version, _snap.confirmed
        _drive(worker, generation, sequence[11:12])
        got = worker.poll(generation, version_mid)
        _drive(worker, generation, sequence[12:])
        snapshot = worker.telemetry_snapshot()
        anchor = worker._anchor_corners.copy()
        _disp = worker.poll_display(generation, 0)
        _version, confirmed, _provisional = (
            _disp.version,
            _disp.confirmed,
            _disp.provisional,
        )
    finally:
        worker.stop()
    assert snapshot["anchor_reanchors"] == 1
    (event,) = snapshot["anchor_reanchor_events"]
    assert event["generation"] == generation
    assert event["streak"] == anchor_mod._FOLLOW_PASSES
    assert event["displacement_px"] > anchor_mod._FOLLOW_DISPLACEMENT_PX
    # A ancora anda ~35 px de camera (mediana do streak {+30,+35,
    # +40} no meio da deriva), nao os >100 px do salto unico.
    assert event["anchor_delta_px"] > 25
    # O evento da arbitragem ficou aberto o episodio inteiro e a
    # re-ancoragem o fechou — e o que deixa a geometria decidir.
    assert event["closed_arb_event"] is True
    # A consistencia par-a-par da arbitragem nunca fecha sob
    # deriva: nenhum veredito, nenhuma acao.
    assert snapshot["anchor_arbitration_actions"] == 0
    assert snapshot["anchor_arbitration_events"] == []
    np.testing.assert_allclose(anchor, BASE + np.float32([35.0, 0.0]), atol=1.5)
    reanchored = [bool(r.get("follow_reanchored", False)) for r in snapshot["records"]]
    assert reanchored == [False] * 11 + [True] + [False] * 3
    assert got is not None, "poll() nao publicou na passada da re-ancoragem"
    _v, canvas_fire = got.version, got.confirmed
    assert np.array_equal(canvas_fire, canvas_mid), (
        "a troca de ancora nao pode deformar o canvas (carry = folha)"
    )
    # Nada do canvas assentado se perdeu (a mancha de borda do
    # registro por quads que derivam e fisica do harness).
    _w2, _g2, truth = _run(_settled(content))
    assert np.count_nonzero(truth & ~confirmed) == 0


# --- 2. tremor e spikes nao re-ancoram ---------------------------------


@pytest.mark.slow
def test_jitter_and_single_pass_spikes_never_reanchor(fast):
    content = _content()
    rng = np.random.default_rng(1)
    quads = []
    for i in range(16):
        quad = BASE + rng.uniform(-8, 8, size=(4, 2)).astype(np.float32)
        if i % 3 == 1:
            quad = BASE + np.float32([200.0, 0.0])
        quads.append(quad)
    for quad in quads:
        if not np.allclose(quad, BASE + np.float32([200.0, 0.0])):
            matrix = registration_mod._registration_matrix(BASE, quad, SHAPE)
            assert anchor_mod._matrix_grid_displacement(matrix, SHAPE) <= 3.0
    worker, _generation, _confirmed = _run(
        _settled(content, passes=2) + [(content, q) for q in quads], stop=False
    )
    try:
        snapshot = worker.telemetry_snapshot()
    finally:
        worker.stop()
    assert snapshot["anchor_reanchors"] == 0
    assert snapshot["anchor_reanchor_events"] == []
    assert max(r["follow_streak"] for r in snapshot["records"]) == 0


# --- 3. abstencao: passada de evento nao confirma -----------------------


@pytest.mark.slow
def test_open_event_births_no_confirmation(fast):
    content = _content()
    other = np.zeros(SHAPE, np.float32)
    other[10:14, 10:40] = 1.0  # tinta nova, fora do canvas confirmado
    # Salto + 1 assentada: o evento esta aberto (streak 1 < 3) e
    # nenhuma das passadas deslocadas pode confirmar a tinta nova.
    sequence = _settled(content) + [(other, STEP.copy()) for _ in range(2)]
    _worker, _generation, confirmed = _run(sequence)
    _w2, _g2, truth = _run(_settled(content))
    assert np.array_equal(confirmed, truth), (
        "passada de evento aberto nao pode confirmar tinta nova"
    )


# --- 4. deriva lenta: o FOLLOW fecha o que a arbitragem nao fecha -------


def _creep_sequence(content, steps=10):
    return _settled(content) + [
        (content, (BASE + np.float32([5.0 * k, 0.0])).astype(np.float32))
        for k in range(1, steps + 1)
    ]


@pytest.mark.slow
def test_slow_creep_reanchors(fast):
    content = _content()
    worker, _generation, confirmed = _run(_creep_sequence(content), stop=False)
    try:
        snapshot = worker.telemetry_snapshot()
        anchor = worker._anchor_corners.copy()
    finally:
        worker.stop()
    assert snapshot["anchor_reanchors"] == 1
    (event,) = snapshot["anchor_reanchor_events"]
    assert event["anchor_delta_px"] > 25
    # A ancora seguiu a deriva (mediana do streak no meio do creep).
    assert 30.0 < float(np.max(np.linalg.norm(anchor - BASE, axis=1))) < 55.0
    # Nada do canvas assentado se perde. A mancha de borda do
    # registro por quads que derivam e fisica do harness (a prob
    # enviada nao muda com o quad): na medicao original com o
    # mecanismo desligado a mesma sequencia manchava 703 px — o
    # FOLLOW nao pode manchar mais que isso.
    _w2, _g2, truth = _run(_settled(content))
    assert np.count_nonzero(truth & ~confirmed) == 0
    assert np.count_nonzero(confirmed & ~truth) <= 703


@pytest.mark.slow
def test_slow_creep_with_arbitration_verdictless_follow_unlocks(fast):
    content = _content()
    worker, _generation, _confirmed = _run(_creep_sequence(content), stop=False)
    try:
        snapshot = worker.telemetry_snapshot()
    finally:
        worker.stop()
    # A consistencia par-a-par da arbitragem nunca fecha sob
    # deriva: nenhum veredito, nenhuma acao — e o FOLLOW destrava.
    assert snapshot.get("anchor_arbitration_events", []) == []
    assert snapshot.get("anchor_arbitration_actions", 0) == 0
    assert snapshot["anchor_reanchors"] == 1


# --- 5. precedencia da arbitragem ---------------------------------------


@pytest.mark.slow
def test_decisive_landing_preempts_follow(fast):
    content = _content()
    # O pouso da bancada de arbitragem: o veredito landing sai na
    # 3a passada deslocada e a ancora anda pelo conteudo.
    sequence = _settled(content) + [(content, STEP.copy()) for _ in range(4)]
    worker, _generation, _confirmed = _run(sequence, stop=False)
    try:
        snapshot = worker.telemetry_snapshot()
        anchor = worker._anchor_corners.copy()
    finally:
        worker.stop()
    assert [e["verdict"] for e in snapshot["anchor_arbitration_events"]] == ["landing"]
    assert snapshot["anchor_arbitration_actions"] == 1
    assert snapshot["anchor_reanchors"] == 0
    np.testing.assert_allclose(anchor, STEP)


@pytest.mark.slow
def test_spurious_verdict_holds_the_episode(fast):
    content = _content()
    # O quad mente (construcao da bancada de arbitragem): o
    # conteudo decide spurious e o episodio fica retido — sem
    # re-ancoragem mesmo com passadas assentadas de sobra.
    matrix = registration_mod._registration_matrix(BASE, STEP, SHAPE)
    distorted = cv2.warpPerspective(
        content, np.linalg.inv(matrix), (SHAPE[1], SHAPE[0])
    ).astype(np.float32)
    sequence = _settled(content) + [(distorted, STEP.copy()) for _ in range(6)]
    worker, _generation, confirmed = _run(sequence, stop=False)
    try:
        snapshot = worker.telemetry_snapshot()
        anchor = worker._anchor_corners.copy()
    finally:
        worker.stop()
    verdicts = [e["verdict"] for e in snapshot["anchor_arbitration_events"]]
    assert verdicts and all(v == "spurious" for v in verdicts)
    assert snapshot["anchor_reanchors"] == 0
    assert any(r.get("follow_hold") for r in snapshot["records"])
    np.testing.assert_allclose(anchor, BASE)
    _w2, _g2, truth = _run(_settled(content))
    assert np.array_equal(confirmed, truth)


@pytest.mark.slow
def test_indeterminate_verdict_lets_follow_fire_and_voting_resumes(fast):
    # Canvas modesto (>= piso da arbitragem) e conteudo deslocado
    # que nao casa com nenhuma colocacao: Sp ~ Sc ~ 0.
    modest = np.zeros(SHAPE, np.float32)
    modest[20:32, 10:60] = 1.0
    # Bloco disjunto do snapshot sob as CINCO colocacoes (pagina,
    # camera e os tres rolls ciclicos — medido: 0,0 nas cinco).
    other = np.zeros(SHAPE, np.float32)
    other[0:12, 10:35] = 1.0
    # Degrau moderado (40 px de camera, ~10,7 px de grade > 8): a
    # homografia do degrau de 120 px degenera na pagina miniatura
    # e o braco ciclico casaria por acidente.
    small_step = BASE + np.float32([40.0, 0.0])
    sequence = (
        [(modest, BASE.copy()) for _ in range(4)]
        + [(other, small_step.copy()) for _ in range(4)]
        + [(other, small_step.copy()) for _ in range(3)]
    )
    worker, _generation, confirmed = _run(sequence, stop=False)
    try:
        snapshot = worker.telemetry_snapshot()
        anchor = worker._anchor_corners.copy()
    finally:
        worker.stop()
    verdicts = [e["verdict"] for e in snapshot["anchor_arbitration_events"]]
    assert verdicts and all(v == "indeterminate" for v in verdicts)
    assert snapshot.get("anchor_arbitration_actions", 0) == 0
    assert snapshot["anchor_reanchors"] == 1
    (event,) = snapshot["anchor_reanchor_events"]
    assert event["closed_arb_event"] is True
    np.testing.assert_allclose(anchor, small_step)
    # Depois da re-ancoragem a tinta da pose nova volta a votar e
    # confirma sobre o canvas antigo (que nao se moveu).
    assert np.count_nonzero(confirmed[0:12, 10:35]) == 12 * 25
    assert np.count_nonzero(confirmed[20:32, 10:60]) == 12 * 50


# --- 6. identidade pendente trava ---------------------------------------


def _drive_hidden(worker, generation, prob, corners, count):
    hidden = np.zeros(prob.shape, bool)
    for _ in range(count):
        before = worker.stats()["passes_total"]
        worker.submit(WarpedPage(prob, generation, corners, hidden))
        deadline = time.monotonic() + 5.0
        while worker.stats()["passes_total"] <= before:
            assert time.monotonic() < deadline, "a passada nao executou"
            time.sleep(0.001)


@pytest.mark.slow
def test_pending_identity_blocks_follow(fast):
    content = _content()
    worker = worker_mod.PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, _settled(content, passes=6))
        generation = worker.new_generation(preserve_identity=True)
        # Recuperacao em quarentena (oclusao total): a decisao de
        # identidade esta pendente e o FOLLOW nao pode contar.
        _drive_hidden(worker, generation, content, STEP.copy(), 4)
        snapshot = worker.telemetry_snapshot()
        pending = worker._identity_recovery_requested
    finally:
        worker.stop()
    assert pending, "a quarentena por oclusao seguia em curso"
    assert snapshot["anchor_reanchors"] == 0
    assert all(
        r["follow_streak"] == 0
        for r in snapshot["records"]
        if r["generation"] == generation
    )


@pytest.mark.slow
def test_identity_judgment_is_not_crossed(fast):
    # Canvas denso (> piso de 2000 px da vigia) para o juiz armar.
    dense = np.zeros(SHAPE, np.float32)
    for row in range(8, 112, 10):
        dense[row : row + 4, 5:80] = 1.0
    worker = worker_mod.PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, [(dense, BASE.copy()) for _ in range(3)])
        generation = worker.new_generation(preserve_identity=True)
        _drive_hidden(worker, generation, dense, STEP.copy(), 4)
        snapshot = worker.telemetry_snapshot()
        pending = worker._identity_recovery_requested
    finally:
        worker.stop()
    assert pending, "o julgamento seguia em quarentena (oclusao)"
    assert snapshot["anchor_reanchors"] == 0
    assert all(
        r["follow_streak"] == 0
        for r in snapshot["records"]
        if r["generation"] == generation
    )


# --- 7. paridade no replay assentado -------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("scenario", sorted(discipline.GOLDEN))
def test_follow_never_triggers_on_settled_jitter_replay(fast, scenario):
    digest, snapshot = discipline._parity_canvas(scenario)
    assert snapshot["anchor_reanchors"] == 0
    assert digest == discipline.GOLDEN[scenario]
