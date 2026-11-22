"""F-11 Block 2: identidade por geracao no worker real, sem tracker paralelo."""

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import identity as identity_mod
from draw.ink import worker as worker_mod

SHAPE = (120, 100)
QUAD = np.float32([[0, 0], [100, 0], [100, 120], [0, 120]])


def page(other=False):
    prob = np.zeros(SHAPE, np.float32)
    prob[80:115, 65:95] = 0.95 if other else 0.0
    if not other:
        prob[10:65, 10:60] = 0.95
    return prob


class Model:
    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0)
    instance = worker_mod.PageCleanWorker(Model()).start()
    try:
        yield instance
    finally:
        instance.stop()


def drive(worker, generation, prob=None, corners=QUAD, visible=None):
    with worker._condition:
        before = worker._passes_total
        worker.submit(
            WarpedPage(page() if prob is None else prob, generation, corners, visible)
        )
        assert worker._condition.wait_for(
            lambda: worker._passes_total > before or worker._stopped, timeout=5
        ), "a passada nao executou"
        assert worker.error is None
        assert worker._passes_total == before + 1
    return worker.telemetry_snapshot()["records"][-1]


def settle(worker):
    generation = worker.new_generation()
    for _ in range(4):
        drive(worker, generation)
    canvas = worker.poll(generation, 0).confirmed
    np.testing.assert_array_equal(canvas > 0, page() > 0.7)
    return canvas


def last_event(worker):
    return worker.telemetry_snapshot()["page_identity_events"][-1]


def test_occlusion_defers_and_preserves_canvas(worker):
    canvas = settle(worker)
    anchor = worker._anchor_corners.copy()
    generation = worker.new_generation(preserve_identity=True)
    reference = worker._identity_reference
    for count in range(1, 4):
        record = drive(
            worker,
            generation,
            corners=QUAD + [30, 15],
            visible=np.zeros(SHAPE, bool),
        )
        event = last_event(worker)
        assert event["decision"] == "deferred_occlusion"
        assert event["quarantine_reason"] == "occlusion"
        assert event["deferred"] is True
        assert event["quarantine_frames"] == count
        assert record["page_identity_decision"] == event["decision"]
        assert record["quarantine_frames"] == count
        assert worker._identity_reference is reference
        assert worker._identity_recovery_requested
        np.testing.assert_array_equal(worker._anchor_corners, anchor)
        np.testing.assert_array_equal(worker._identity_reference_corners, anchor)
        # O desenho mora na referencia, nao no canvas publicavel:
        # a passada abstida so publica o que esta geracao viu.
        np.testing.assert_array_equal(worker._identity_reference, canvas)
        assert not np.any(worker.poll(generation, 0).confirmed)

    drive(worker, generation)
    event = last_event(worker)
    assert event["decision"] == "same_page"
    assert event["deferred"] is False
    assert event["quarantine_frames"] == 0
    assert worker._identity_defer_count == 0
    assert worker._identity_reference is None
    assert worker._identity_reference_corners is None
    assert not worker._identity_recovery_requested
    np.testing.assert_array_equal(worker._anchor_corners, anchor)
    np.testing.assert_array_equal(worker.poll(generation, 0).confirmed, canvas)


def test_second_drop_does_not_merge_intermediate_sheet(worker):
    canvas_a = settle(worker)
    # 1a queda: a referencia salva fica pendente; a folha B entra
    # com a mao cobrindo o miolo das marcas de A — so um anel das
    # marcas antigas segue visivel, entao a passada adia por
    # oclusao e VOTA: a tinta intermediaria confirma no acumulador.
    gen_b = worker.new_generation(preserve_identity=True)
    visible_b = np.ones(SHAPE, bool)
    visible_b[12:63, 12:58] = False
    b_confirmed = 0
    for _ in range(identity_mod._PAGE_IDENTITY_MAX_DEFERRALS):
        drive(worker, gen_b, prob=page(other=True), visible=visible_b)
        event = last_event(worker)
        assert event["decision"] == "deferred_occlusion"
        assert event["quarantine_reason"] == "occlusion"
        confirmed = worker._accumulator._confirmed
        if confirmed is not None:
            b_confirmed = max(
                b_confirmed, int(np.count_nonzero(confirmed[80:115, 65:95]))
            )
    assert b_confirmed > 0
    # 2a queda com a referencia pendente: a geracao intermediaria
    # zera junto — a referencia e a unica dona do desenho de A e a
    # tinta de B nao pode fundir na restauracao.
    gen_a = worker.new_generation(preserve_identity=True)
    np.testing.assert_array_equal(worker._identity_reference, canvas_a)
    confirmed = worker._accumulator._confirmed
    assert confirmed is None or not np.any(confirmed)
    # A retorna: o same_page restaura exatamente o desenho salvo —
    # nada da folha intermediaria sobrevive na restauracao.
    drive(worker, gen_a)
    assert last_event(worker)["decision"] == "same_page"
    np.testing.assert_array_equal(worker.poll(gen_a, 0).confirmed, canvas_a)


def test_fragment_match_does_not_restore_unevaluated_reference(worker):
    # Caminho legado (sem mascara de visibilidade): a substituta
    # cobre so ~40% do quad e traz UMA marca que coincide com a
    # faixa da referencia dentro da pegada — recall e containment
    # fecham 1,0 sobre a fatia avaliada, mas 57% do desenho salvo
    # nunca foi comparado. Restaurar o resto nao e honesto: adia
    # sem votar ate o timeout resolver pelo caminho seguro.
    ink_a = np.zeros(SHAPE, np.float32)
    ink_a[10:65, 10:30] = 0.95  # fora da pegada da substituta
    ink_a[10:65, 84:99] = 0.95  # faixa que a pegada parcial alcanca
    for _ in range(4):
        drive(worker, 0, prob=ink_a)
    canvas_a = worker.poll(0, 0).confirmed
    assert np.count_nonzero(canvas_a) > 0
    gen_b = worker.new_generation(preserve_identity=True)
    part_quad = np.float32([[60, 0], [100, 0], [100, 120], [60, 120]])
    fragment = np.zeros(SHAPE, np.float32)
    fragment[10:65, 60:96] = 0.95  # coincide com a faixa avaliavel
    for _ in range(identity_mod._PAGE_IDENTITY_MAX_DEFERRALS):
        drive(worker, gen_b, prob=fragment, corners=part_quad)
        event = last_event(worker)
        assert event["decision"] == "deferred_occlusion"
        assert event["quarantine_reason"] == "no_overlap"
        np.testing.assert_array_equal(worker._identity_reference, canvas_a)
        assert not np.any(worker.poll(gen_b, 0).confirmed)
    drive(worker, gen_b, prob=fragment, corners=part_quad)
    assert last_event(worker)["decision"] == "quarantine_timeout"
    assert worker._identity_reference is None
    for _ in range(3):
        drive(worker, gen_b, prob=fragment, corners=part_quad)
    confirmed = worker.poll(gen_b, 0).confirmed
    # So a marca do fragmento confirma; o trecho do desenho salvo
    # que a pegada nunca avaliou nao e herdado.
    assert np.count_nonzero(confirmed) > 0
    assert not np.any(confirmed[:, :55])


def test_hand_over_majority_holds_then_restores_same_page(worker):
    canvas_a = settle(worker)
    gen = worker.new_generation(preserve_identity=True)
    # Mesma folha, mesma pose, com a mao sobre 60% da tinta: a
    # fatia visivel casa, mas o desenho inteiro so volta quando a
    # comparacao cobrir metade do dominio — enquanto isso o hold
    # abstem o voto SEM queimar a quarentena (uma substituta com
    # marcas coincidentes na parte visivel nao pode herdar a tinta
    # escondida, e o desenho salvo nao vira wipe no timeout).
    visible = np.ones(SHAPE, bool)
    visible[10:43, 10:60] = False
    for _ in range(3):
        drive(worker, gen, visible=visible)
        event = last_event(worker)
        assert event["decision"] == "deferred_occlusion"
        assert event["quarantine_reason"] == "occlusion"
        assert event["quarantine_frames"] == 0
        np.testing.assert_array_equal(worker._identity_reference, canvas_a)
        assert not np.any(worker.poll(gen, 0).confirmed)
    # Mao levanta: a referencia inteira e comparada — same_page
    # resolve e o desenho volta intacto.
    drive(worker, gen)
    assert last_event(worker)["decision"] == "same_page"
    np.testing.assert_array_equal(worker.poll(gen, 0).confirmed, canvas_a)


def test_replacement_with_matching_visible_marks_holds(worker):
    canvas_a = settle(worker)
    gen = worker.new_generation(preserve_identity=True)
    # Substituta na mesma pose: a metade visivel repete as marcas
    # de A, a mao cobre a regiao onde a tinta difere. O match na
    # fatia vista nao distingue retorno de substituicao — o hold
    # abstem sem queimar a quarentena e a referencia segue intacta.
    replacement = np.zeros(SHAPE, np.float32)
    replacement[43:65, 10:60] = 0.95
    visible = np.ones(SHAPE, bool)
    visible[10:43, 10:60] = False
    for _ in range(3):
        drive(worker, gen, prob=replacement, visible=visible)
        event = last_event(worker)
        assert event["decision"] == "deferred_occlusion"
        assert event["quarantine_reason"] == "occlusion"
        assert event["quarantine_frames"] == 0
        np.testing.assert_array_equal(worker._identity_reference, canvas_a)
        assert not np.any(worker.poll(gen, 0).confirmed)
    # Mao levanta mostrando papel em branco onde A tinha tinta: o
    # consenso de troca resolve — a tinta escondida de A nunca
    # chega ao canvas da substituta.
    for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
        drive(worker, gen, prob=replacement)
        if last_event(worker)["decision"] == "new_page":
            break
    assert last_event(worker)["decision"] == "new_page"
    for _ in range(4):
        drive(worker, gen, prob=replacement)
    confirmed = worker.poll(gen, 0).confirmed
    assert np.count_nonzero(confirmed) > 0
    assert not np.any(confirmed[10:43, 10:60])


def test_deepening_occlusion_stays_inside_hold(worker):
    canvas_a = settle(worker)
    gen = worker.new_generation(preserve_identity=True)
    # 1a passada: mao sobre 60% da tinta — o match parcial engata o
    # hold. Nas seguintes a mao cobre 80%: a passada cai no gate de
    # baixa visibilidade, que QUEIMA a quarentena — mas depois do
    # hold engatado nao e evidencia nova: segue protegido ate a mao
    # sair. Sem o hold pegajoso o timeout apagava o desenho salvo.
    visible = np.ones(SHAPE, bool)
    visible[10:43, 10:60] = False
    drive(worker, gen, visible=visible)
    assert last_event(worker)["decision"] == "deferred_occlusion"
    visible[10:54, 10:60] = False
    for _ in range(identity_mod._PAGE_IDENTITY_MAX_DEFERRALS + 1):
        drive(worker, gen, visible=visible)
        event = last_event(worker)
        assert event["decision"] == "deferred_occlusion"
        assert event["quarantine_reason"] == "occlusion"
        assert event["quarantine_frames"] == 0
        np.testing.assert_array_equal(worker._identity_reference, canvas_a)
        assert not np.any(worker.poll(gen, 0).confirmed)
    drive(worker, gen)
    assert last_event(worker)["decision"] == "same_page"
    np.testing.assert_array_equal(worker.poll(gen, 0).confirmed, canvas_a)


def test_quarter_coverage_holds_instead_of_restoring(worker):
    # Os dois pisos de 50% em cascata deixavam passar um quarto:
    # pegada alcanca ~60% da referencia E a mao esconde ~40% do
    # alcancado → so ~36% da tinta julgavel foi comparada. O piso
    # de visibilidade mede contra a referencia julgavel inteira —
    # segurando (hold, sem queimar) ate a mao revelar o resto.
    ink_a = np.zeros(SHAPE, np.float32)
    ink_a[10:65, 10:20] = 0.95  # fora da pegada da passada
    ink_a[10:65, 60:98] = 0.95  # dentro da pegada parcial
    for _ in range(4):
        drive(worker, 0, prob=ink_a)
    canvas_a = worker._accumulator._confirmed.copy()
    gen = worker.new_generation(preserve_identity=True)
    part_quad = np.float32([[60, 0], [100, 0], [100, 120], [60, 120]])
    visible = np.ones(SHAPE, bool)
    visible[10:32, 60:99] = False
    for _ in range(3):
        drive(worker, gen, prob=ink_a, corners=part_quad, visible=visible)
        event = last_event(worker)
        assert event["decision"] == "deferred_occlusion"
        assert event["quarantine_reason"] == "occlusion"
        assert event["quarantine_frames"] == 0
        np.testing.assert_array_equal(worker._identity_reference, canvas_a)
        assert not np.any(worker.poll(gen, 0).confirmed)
    drive(worker, gen, prob=ink_a, corners=part_quad)
    assert last_event(worker)["decision"] == "same_page"
    np.testing.assert_array_equal(worker.poll(gen, 0).confirmed, canvas_a)


def test_near_quarter_coverage_still_holds(worker):
    # O canto que a cascata deixava aberto de verdade: dominio
    # alcancado pouco acima do piso de fragmento (~58% da
    # referencia julgavel) e visivel pouco acima de metade disso
    # (~51% do dominio → ~30% da referencia). Abaixo de ~25%
    # nao ha cenario: o dominio cai sob o piso geometrico e a
    # passada ja adia por no_overlap — so a banda entre um
    # quarto e metade era aceita como match. Aqui a referencia
    # segura como hold ate a mao revelar o resto.
    ink_a = np.zeros(SHAPE, np.float32)
    ink_a[10:65, 10:36] = 0.95  # 1430 px — fora da pegada
    ink_a[10:65, 60:96] = 0.95  # 1980 px — dentro da pegada
    for _ in range(4):
        drive(worker, 0, prob=ink_a)
    canvas_a = worker._accumulator._confirmed.copy()
    gen = worker.new_generation(preserve_identity=True)
    part_quad = np.float32([[60, 0], [100, 0], [100, 120], [60, 120]])
    # Mao cobre a faixa esquerda inteira (band A) e o topo da
    # banda da pegada: sobram 1008 px = ~30% da referencia
    # julgavel (~51% do dominio) — a cascata restaurava.
    visible = np.ones(SHAPE, bool)
    visible[10:65, 0:60] = False
    visible[10:37, 60:99] = False
    for _ in range(3):
        drive(worker, gen, prob=ink_a, corners=part_quad, visible=visible)
        event = last_event(worker)
        assert event["decision"] == "deferred_occlusion"
        assert event["quarantine_reason"] == "occlusion"
        assert event["quarantine_frames"] == 0
        np.testing.assert_array_equal(worker._identity_reference, canvas_a)
        assert not np.any(worker.poll(gen, 0).confirmed)
    # Sem a mao: o dominio inteiro e comparado — same_page
    # restaura o desenho salvo exato.
    drive(worker, gen, prob=ink_a, corners=part_quad)
    assert last_event(worker)["decision"] == "same_page"
    np.testing.assert_array_equal(worker.poll(gen, 0).confirmed, canvas_a)


def test_deferral_fields_declared_in_all_states(worker):
    # Recuperacao sem referencia tambem declara os campos de quarentena.
    generation = worker.new_generation(preserve_identity=True)
    drive(worker, generation)
    assert last_event(worker)["decision"] == "new_page"
    settle(worker)
    generation = worker.new_generation(preserve_identity=True)
    drive(worker, generation, visible=np.zeros(SHAPE, bool))
    assert last_event(worker)["decision"] == "deferred_occlusion"
    drive(worker, generation)
    assert last_event(worker)["decision"] == "same_page"
    generation = worker.new_generation(preserve_identity=True)
    for _ in range(2):
        drive(worker, generation, page(other=True))
        assert last_event(worker)["decision"] == "deferred_switch_consensus"
    drive(worker, generation, page(other=True))
    assert last_event(worker)["decision"] == "new_page"
    drive(worker, generation, page(other=True))
    assert worker.telemetry_snapshot()["records"][-1]["page_identity_decision"] is None
    for record in (
        worker.telemetry_snapshot()["records"]
        + worker.telemetry_snapshot()["page_identity_events"]
    ):
        assert "quarantine_reason" in record
        assert type(record["deferred"]) is bool
        assert type(record["quarantine_frames"]) is int


def test_legitimate_page_turn_commits(worker):
    canvas = settle(worker)
    anchor = worker._anchor_corners.copy()
    generation = worker.new_generation(preserve_identity=True)
    reference = worker._identity_reference
    for count in range(1, 3):
        drive(worker, generation, page(other=True))
        assert last_event(worker)["decision"] == "deferred_switch_consensus"
        assert worker._identity_switch_votes == count
        assert worker._identity_defer_count == 0
        assert worker._identity_reference is reference
        np.testing.assert_array_equal(worker._identity_reference_corners, anchor)
        np.testing.assert_array_equal(worker._anchor_corners, anchor)
        # Abstido pelo consenso de troca, o desenho mora na
        # referencia e o canvas publicado segue vazio.
        np.testing.assert_array_equal(worker._identity_reference, canvas)
        assert not np.any(worker.poll(generation, 0).confirmed)
    drive(worker, generation, page(other=True))
    event = last_event(worker)
    assert event["decision"] == "new_page"
    assert event["quarantine_frames"] == 0
    assert not event["deferred"]
    assert worker._identity_reference is None
    assert worker._identity_reference_corners is None
    assert not worker._identity_recovery_requested
    assert worker._identity_switch_votes == 0
    assert worker._identity_defer_count == 0
    for _ in range(3):
        drive(worker, generation, page(other=True))
    np.testing.assert_array_equal(
        worker.poll(generation, 0).confirmed > 0, page(other=True) > 0.7
    )


def test_switch_consensus_preserves_state_and_abstains(worker):
    canvas = settle(worker)
    anchor = worker._anchor_corners.copy()
    worker.set_hand_presence(False)
    generation = worker.new_generation(preserve_identity=True)
    reference = worker._identity_reference
    moved = QUAD + [3, -2]
    for count in range(1, 3):
        drive(worker, generation, page(other=True), corners=moved)
        assert last_event(worker)["decision"] == "deferred_switch_consensus"
        assert worker._identity_switch_votes == count
        assert worker._identity_defer_count == 0
        assert worker._identity_reference is reference
        assert worker._identity_recovery_requested
        np.testing.assert_array_equal(worker._identity_reference_corners, anchor)
        np.testing.assert_array_equal(worker._anchor_corners, anchor)
        # Abstido pelo consenso de troca, o desenho mora na
        # referencia e o canvas publicado segue vazio.
        np.testing.assert_array_equal(worker._identity_reference, canvas)
        assert not np.any(worker.poll(generation, 0).confirmed)
        assert not np.any(worker._accumulator._window[-1])
        with worker._condition:
            worker._generation_passes = worker_mod._REST_MIN_GENERATION_PASSES
            worker._quiet_passes = worker_mod._REST_QUIET_PASSES
            worker._confirm_once = False
            assert not worker._at_rest()
    drive(worker, generation, page(other=True), corners=moved)
    assert last_event(worker)["decision"] == "new_page"
    np.testing.assert_array_equal(worker._anchor_corners, moved)
    assert np.any(worker._accumulator._window[-1])
    assert not np.any(worker._accumulator._confirmed)


@pytest.mark.parametrize("blank", [False, True])
def test_switch_consensus_publishes_reset_canvas(worker, blank):
    settle(worker)
    generation = worker.new_generation(preserve_identity=True)
    candidate = np.zeros(SHAPE, np.float32) if blank else page(other=True)
    for _ in range(2):
        drive(worker, generation, candidate)
    version = worker.poll(generation, 0).version
    display_version = worker.poll_display(generation, 0).version
    publishes = worker.telemetry_snapshot()["publishes"]
    drive(worker, generation, candidate)
    assert last_event(worker)["decision"] == "new_page"
    published = worker.poll(generation, version)
    display = worker.poll_display(generation, display_version)
    assert published is not None
    assert display is not None
    assert not np.any(published.confirmed)
    assert not np.any(display.confirmed)
    np.testing.assert_array_equal(display.provisional > 0, candidate > 0.7)
    assert worker.telemetry_snapshot()["publishes"] == publishes + 1


def test_dirty_pass_breaks_switch_consensus(worker):
    settle(worker)
    generation = worker.new_generation(preserve_identity=True)
    reference = worker._identity_reference
    for _ in range(2):
        drive(worker, generation, page(other=True))
    assert worker._identity_switch_votes == 2
    drive(
        worker,
        generation,
        page(other=True),
        visible=np.zeros(SHAPE, bool),
    )
    assert last_event(worker)["decision"] == "deferred_occlusion"
    assert worker._identity_switch_votes == 0
    assert worker._identity_defer_count == 1
    for count in range(1, 3):
        drive(worker, generation, page(other=True))
        assert last_event(worker)["decision"] == "deferred_switch_consensus"
        assert worker._identity_switch_votes == count
        assert worker._identity_defer_count == 1
        assert worker._identity_reference is reference
    drive(worker, generation, page(other=True))
    assert last_event(worker)["decision"] == "new_page"
    assert worker._identity_switch_votes == 0
    assert worker._identity_defer_count == 0
    assert worker._identity_reference is None


def test_same_page_ends_pending_switch_without_consensus(worker):
    canvas = settle(worker)
    anchor = worker._anchor_corners.copy()
    generation = worker.new_generation(preserve_identity=True)
    drive(worker, generation, page(other=True))
    assert worker._identity_switch_votes == 1
    drive(worker, generation)
    assert last_event(worker)["decision"] == "same_page"
    assert worker._identity_switch_votes == 0
    assert worker._identity_reference is None
    np.testing.assert_array_equal(worker._anchor_corners, anchor)
    np.testing.assert_array_equal(worker.poll(generation, 0).confirmed, canvas)


def test_switch_consensus_telemetry_declares_deferral_fields(worker):
    settle(worker)
    generation = worker.new_generation(preserve_identity=True)
    for decision in (
        "deferred_switch_consensus",
        "deferred_switch_consensus",
        "new_page",
    ):
        record = drive(worker, generation, page(other=True))
        event = worker._identity_events[-1]
        assert event["decision"] == record["page_identity_decision"] == decision
        for entry in (event, record):
            assert entry["quarantine_frames"] == 0
            assert entry["quarantine_reason"] is None
            assert entry["deferred"] is False


@pytest.mark.parametrize("preserve", [False, True])
def test_new_generation_resets_switch_votes(worker, preserve):
    settle(worker)
    generation = worker.new_generation(preserve_identity=True)
    for _ in range(2):
        drive(worker, generation, page(other=True))
    assert worker._identity_switch_votes == 2
    generation = worker.new_generation(preserve_identity=preserve)
    assert worker._identity_switch_votes == 0
    drive(worker, generation, page(other=True))
    if preserve:
        assert last_event(worker)["decision"] == "deferred_switch_consensus"
        assert worker._identity_switch_votes == 1
    else:
        assert worker._identity_switch_votes == 0


def test_switch_consensus_does_not_spend_or_reset_quarantine_budget(worker):
    canvas = settle(worker)
    generation = worker.new_generation(preserve_identity=True)
    for count in range(1, identity_mod._PAGE_IDENTITY_MAX_DEFERRALS + 2):
        for _ in range(2):
            drive(worker, generation, page(other=True))
            assert last_event(worker)["decision"] == "deferred_switch_consensus"
            assert worker._identity_defer_count == count - 1
            # Abstido, o desenho mora na referencia — o canvas
            # publicado nao o carrega.
            np.testing.assert_array_equal(worker._identity_reference, canvas)
            assert not np.any(worker.poll(generation, 0).confirmed)
        drive(
            worker,
            generation,
            page(other=True),
            visible=np.zeros(SHAPE, bool),
        )
        assert worker._identity_switch_votes == 0
        assert last_event(worker)["quarantine_frames"] == count
        assert last_event(worker)["decision"] != "new_page"
    assert last_event(worker)["decision"] == "quarantine_timeout"
    assert worker._identity_reference is None
    assert worker._identity_defer_count == 0
    assert worker._identity_switch_votes == 0
    assert worker._anchor_corners is None


@pytest.mark.parametrize("limit", [None, 2])
def test_quarantine_timeout(worker, monkeypatch, limit):
    if limit is not None:
        monkeypatch.setattr(identity_mod, "_PAGE_IDENTITY_MAX_DEFERRALS", limit)
    maximum = identity_mod._PAGE_IDENTITY_MAX_DEFERRALS
    canvas = settle(worker)
    # A quarentena precisa continuar mesmo com a mao ausente e a saida quieta.
    worker.set_hand_presence(False)
    generation = worker.new_generation(preserve_identity=True)
    version = 0
    for count in range(1, maximum + 2):
        record = drive(
            worker,
            generation,
            page(other=True),
            visible=np.zeros(SHAPE, bool),
        )
        event = last_event(worker)
        assert event["quarantine_frames"] == count
        assert record["quarantine_frames"] == count
        assert event["quarantine_reason"] == "occlusion"
        assert event["decision"] != "new_page"
        assert event["deferred"] is True
        if count <= maximum:
            assert worker._identity_reference is not None
            # O desenho mora na referencia, nao no canvas
            # publicavel — nada velho pode vazar para o consumidor
            # enquanto a identidade esta pendente.
            np.testing.assert_array_equal(worker._identity_reference, canvas)
            _snap = worker.poll(generation, 0)
            version, published = _snap.version, _snap.confirmed
            assert not np.any(published)
    assert last_event(worker)["decision"] == "quarantine_timeout"
    assert worker._identity_reference is None
    assert worker._identity_reference_corners is None
    assert not worker._identity_recovery_requested
    assert worker._identity_defer_count == 0
    assert worker._anchor_corners is None
    lost_lock = worker.poll(generation, version)
    assert lost_lock is not None
    assert not np.any(lost_lock.confirmed)
    display = worker.poll_display(generation, 0)
    assert display is not None
    assert not np.any(display.confirmed)
    assert not np.any(display.provisional)
    assert worker.error is None
    for _ in range(4):
        drive(worker, generation, page(other=True))
    np.testing.assert_array_equal(worker._anchor_corners, QUAD)
    np.testing.assert_array_equal(
        worker.poll(generation, 0).confirmed > 0, page(other=True) > 0.7
    )
    assert last_event(worker)["decision"] == "quarantine_timeout"


@pytest.mark.parametrize("preserve", [False, True])
def test_new_generation_resets_quarantine_count(worker, preserve):
    settle(worker)
    generation = worker.new_generation(preserve_identity=True)
    drive(worker, generation, visible=np.zeros(SHAPE, bool))
    assert worker._identity_defer_count == 1
    generation = worker.new_generation(preserve_identity=preserve)
    assert worker._identity_defer_count == 0
    drive(worker, generation)
    assert worker._identity_defer_count == 0
    if preserve:
        assert last_event(worker)["decision"] == "same_page"
