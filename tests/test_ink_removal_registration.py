import cv2
import numpy as np
import pytest

import rr
from draw.contracts import WarpedPage
from draw.ink import accumulator as accumulator_mod
from draw.ink import occlusion as occlusion_mod
from draw.ink import registration as registration_mod
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig
from rr import _ProbIsWarp


def ink():
    prob = np.zeros((100, 120), np.float32)
    prob[15:45, 15:55] = 0.95
    prob[65:75, 65:85] = 0.95
    return prob


def _big_ink():
    prob = np.zeros((260, 400), np.float32)
    prob[10:190, 10:200] = 0.95  # ~34k px confirmados
    return prob


def confirmed(acc, prob):
    visible = np.ones(prob.shape, bool)
    for _ in range(16):
        result = acc.add(prob, visible, paper_exposed=visible)
    return result["confirmed"].copy()


class _MutablePageClean:
    threshold = 0.7

    def __init__(self, prob):
        self.prob = prob

    def ink_prob(self, _image):
        return self.prob


def test_erasure_is_published_and_does_not_resurrect():
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = ink()
    original = confirmed(acc, prob)
    erased = prob.copy()
    erased[65:75, 65:85] = 0
    visible = np.ones(prob.shape, bool)
    for _ in range(registration_mod._REMOVAL_PASSES - 1):
        result = acc.add(erased, visible, paper_exposed=visible)
        np.testing.assert_array_equal(result["confirmed"], original)
    result = acc.add(erased, visible, paper_exposed=visible)
    assert result["confirmed_changed"] and result["removed_confirmed_px"] == 200
    for _ in range(20):
        result = acc.add(erased, visible, paper_exposed=visible)
        assert not result["confirmed"][65:75, 65:85].any()
        assert not result["provisional"][65:75, 65:85].any()
    assert np.all(result["confirmed"][15:45, 15:55] == 255)
    confirmed(acc, prob)
    assert np.all(acc._confirmed[65:75, 65:85] == 255)  # redraw still works


@pytest.mark.parametrize(
    "mode", ["occlusion", "no_visibility", "turn", "registration", "flicker"]
)
def test_negative_evidence_fails_closed(mode):
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = ink()
    original = confirmed(acc, prob)
    erased = prob.copy()
    erased[65:75, 65:85] = 0
    visible = np.ones(prob.shape, bool)
    if mode == "occlusion":
        visible[65:75, 65:85] = False
    elif mode == "no_visibility":
        visible = None
    elif mode == "turn":
        erased[:] = 0
    for i in range(20):
        current = prob if mode == "flicker" and i % 2 else erased
        result = acc.add(
            current,
            visible,
            paper_exposed=visible,
            removal_allowed=mode != "registration",
        )
    np.testing.assert_array_equal(result["confirmed"], original)


def drive(sequence, config=None):
    clock = rr.SceneClock()
    worker = rr.make_ink_worker(_ProbIsWarp(), clock, config=config)
    gen = worker.new_generation()
    for prob, corners in sequence:
        clock.now += 1
        worker.submit(
            WarpedPage(
                prob,
                gen,
                corners,
                rr.camera_mask(np.ones(prob.shape, bool), corners),
            )
        )
        assert worker.pump()
        assert worker.error is None
    return worker


def test_live_telemetry_records_pass_fields():
    prob = ink()
    worker = drive([(prob, rr.base_quad(prob.shape))] * 8)
    record = worker.telemetry_snapshot()["records"][-1]
    assert record["sequence"] == 8 and record["pass_gap_s"] == 1
    assert record["registration_is_identity"]
    assert "arb_event_open" in record and "follow_streak" in record
    assert "new_confirmed_px" in record and "removed_confirmed_px" in record


def test_worker_publishes_erasure_through_the_real_path():
    prob = ink()
    page = np.full((*prob.shape, 3), 220, dtype=np.uint8)
    erased = prob.copy()
    erased[65:75, 65:85] = 0

    clock = rr.SceneClock()
    page_clean = _MutablePageClean(prob)
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(prob.shape)
    visible = np.ones((720, 1280), bool)
    for current in [prob] * 16 + [erased] * 10:
        page_clean.prob = current
        clock.now += 1
        worker.submit(WarpedPage(page, generation, corners, visible=visible))
        assert worker.pump()
        assert worker.error is None
    assert not worker._accumulator._confirmed[65:75, 65:85].any()
    assert np.all(worker._accumulator._confirmed[15:45, 15:55])
    records = worker.telemetry_snapshot()["records"]
    assert sum(r["removed_confirmed_px"] for r in records) == 200
    assert not worker._at_rest() or worker._accumulator._removal_streak.max() == 0


def test_erasure_advances_on_unobstructed_page_without_hand_mask():
    """Caminho clássico (detecção de folha sem modelo de mão): com a
    página à vista e a guarda confirmando papel, a remoção corre — o
    "sem bloco" fotométrico vira evidência de visibilidade, não buraco
    (Devin Review do PR #144)."""
    prob = ink()
    page = np.full((*prob.shape, 3), 220, dtype=np.uint8)
    erased = prob.copy()
    erased[65:75, 65:85] = 0

    clock = rr.SceneClock()
    page_clean = _MutablePageClean(prob)
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(prob.shape)
    for current in [prob] * 16 + [erased] * 10:
        page_clean.prob = current
        clock.now += 1
        worker.submit(WarpedPage(page, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert not worker._accumulator._confirmed[65:75, 65:85].any()
    assert np.all(worker._accumulator._confirmed[15:45, 15:55])
    records = worker.telemetry_snapshot()["records"]
    assert sum(r["removed_confirmed_px"] for r in records) == 200


def test_worker_confirms_fill_after_sustained_ink_evidence():
    """E2E do preenchimento: depois que a referência do papel claro
    está de pé, uma área pintada de escuro confirma como tinta — a
    guarda para de vetar adições com a evidência sustentada, em vez
    de congelar a região para sempre (Devin Review do PR #144)."""
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    filled = paper.copy()
    filled[24:96, 24:96] = 60
    prob_fill = np.zeros(shape, np.float32)
    prob_fill[24:96, 24:96] = 0.95

    clock = rr.SceneClock()
    page_clean = _MutablePageClean(np.zeros(shape, np.float32))
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)

    def pass_page(image, prob):
        page_clean.prob = prob
        clock.now += 1
        worker.submit(WarpedPage(image, generation, corners))
        assert worker.pump()
        assert worker.error is None

    for _ in range(3):
        pass_page(paper, np.zeros(shape, np.float32))
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES - 1):
        pass_page(filled, prob_fill)
        assert not worker._accumulator._confirmed[24:96, 24:96].any()
    for _ in range(4):
        pass_page(filled, prob_fill)
    assert worker._accumulator._confirmed[24:96, 24:96].all()


def test_page_recovery_defers_when_shadow_covers_prior_ink():
    """O bloqueio fotométrico também rege o reencontro: a sombra que
    o detector de mão perdeu esconde a tinta anterior — a identidade
    adia em vez de condenar a página (Devin Review do PR #144)."""
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    prob = np.zeros(shape, np.float32)
    prob[40:50, 60:140] = 0.95  # traço comprido dentro da futura sombra
    shadowed = paper.copy()
    shadowed[24:96, 48:144] = 60  # bloco 3x4 tiles sobre o traço
    dark = np.zeros(shape, np.float32)

    clock = rr.SceneClock()
    page_clean = _MutablePageClean(prob)
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    for _ in range(6):
        page_clean.prob = prob
        clock.now += 1
        worker.submit(WarpedPage(paper, generation, corners))
        assert worker.pump()
        assert worker.error is None

    generation = worker.new_generation(preserve_identity=True)
    page_clean.prob = dark
    clock.now += 1
    worker.submit(WarpedPage(shadowed, generation, corners))
    assert worker.pump()
    assert worker.error is None
    event = worker.telemetry_snapshot()["page_identity_events"][-1]
    assert event["decision"] == "deferred_occlusion"


def test_witness_floor_unblocks_stale_canvas():
    """retained < 65% mas >= piso: a testemunha existe e a remocao corre.

    O catch-22 medido no sim-hand: mais de ~35% do canvas velho e os
    proprios pixels errados impediam a fracao de existir — retained
    0,33% e a saida congelava com a folha limpa a vista. Com o piso,
    4k px de re-testemunho comprovam o modelo vivo e a evidencia
    negativa dele conta.
    """
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = _big_ink()
    confirmed(acc, prob)
    _canvas = int((acc._confirmed > 0).sum())
    # Copia defasada: metade da tinta "some" (stale), metade segue.
    stale = prob.copy()
    stale[10:190, 10:110] = 0.0
    visible = np.ones(prob.shape, bool)
    # retained ~17k de ~34k = ~50% < 65% mas >= 4k -> testemunhada.
    for _ in range(registration_mod._REMOVAL_PASSES):
        acc.add(stale, visible, paper_exposed=visible)
    assert not acc._confirmed[10:190, 10:110].any()
    assert np.all(acc._confirmed[10:190, 110:200] == 255)


def test_witness_floor_keeps_total_darkness_closed():
    """Modelo escurecendo por inteiro segue vedado: nada de varredura."""
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = _big_ink()
    original = confirmed(acc, prob)
    dark = np.zeros_like(prob)
    visible = np.ones(prob.shape, bool)
    for _ in range(2 * registration_mod._REMOVAL_PASSES):
        acc.add(dark, visible, paper_exposed=visible)
    np.testing.assert_array_equal(acc._confirmed > 0, original > 0)


def test_absence_streak_survives_unwitnessed_and_occlusion():
    """Passada sem testemunha ou pixel ocluido seguram o streak.

    A ausencia do F-10 precisa ser persistente, nao consecutiva:
    em cena real a testemunha oscila e a mao cobre a regiao —
    interromper a contagem a cada buraco congelava a copia
    defasada para sempre.
    """
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = _big_ink()
    confirmed(acc, prob)
    stale = prob.copy()
    stale[10:190, 10:60] = 0.0  # regiao que vai ser removida
    visible = np.ones(prob.shape, bool)
    occluded = visible.copy()
    occluded[10:190, 10:60] = False
    for _ in range(3):
        acc.add(stale, visible, paper_exposed=visible)  # 3 testemunhadas ausentes
    acc.add(
        stale, visible, paper_exposed=visible, removal_allowed=False
    )  # sem info: segura
    acc.add(stale, occluded, paper_exposed=occluded)  # regiao ocluida: segura tambem
    assert (acc._confirmed > 0)[10:190, 10:60].any()
    for _ in range(registration_mod._REMOVAL_PASSES - 3):
        acc.add(stale, visible, paper_exposed=visible)  # completa as 6 testemunhadas
    assert not acc._confirmed[10:190, 10:60].any()
    assert np.all(acc._confirmed[10:190, 60:200] == 255)


def test_paper_exposure_guard_freezes_a_dark_block_when_hand_mask_misses():
    guard = occlusion_mod.PaperExposureGuard()
    baseline = np.full((120, 160, 3), 220, dtype=np.uint8)
    visible = np.ones(baseline.shape[:2], dtype=bool)

    baseline_assessment = guard.observe(baseline, visible)
    assert baseline_assessment.paper_exposed is True
    assert baseline_assessment.visible is None
    baseline_assessment = guard.observe(baseline, visible)
    assert baseline_assessment.paper_exposed is True
    assert baseline_assessment.visible is None
    shadow = baseline.copy()
    shadow[24:96, 24:96] = 75
    assessment = guard.observe(shadow, visible)  # visible=True simulates a miss

    assert not assessment.paper_exposed[50:65, 50:65].any()
    assert not assessment.visible[50:65, 50:65].any()
    assert assessment.paper_exposed[0:12, 0:12].all()
    baseline_assessment = guard.observe(baseline, visible)
    assert baseline_assessment.paper_exposed is True
    assert baseline_assessment.visible is None

    first_frame_guard = occlusion_mod.PaperExposureGuard()
    first_frame_hand = baseline.copy()
    first_frame_hand[24:96, 24:96] = 75
    first_frame_exposed = first_frame_guard.observe(first_frame_hand, visible)
    assert not first_frame_exposed.paper_exposed[30:60, 30:60].any()
    assert not first_frame_exposed.visible[30:60, 30:60].any()

    color_guard = occlusion_mod.PaperExposureGuard()
    baseline_assessment = color_guard.observe(baseline, visible)
    assert baseline_assessment.paper_exposed is True
    assert baseline_assessment.visible is None
    colored_object = baseline.copy()
    colored_object[24:96, 24:96] = (255, 200, 190)  # quase mesma luminância
    color_exposed = color_guard.observe(colored_object, visible)
    assert not color_exposed.visible[30:60, 30:60].any()


def test_global_dimming_is_absorbed_not_blocked():
    """Queda global de >20% (luz da sala, auto-exposição): as
    referências cedem ao nível novo — a página não congela atrás de
    um bloco único (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    bright = np.full((120, 160, 3), 220, dtype=np.uint8)
    visible = np.ones(bright.shape[:2], dtype=bool)
    assert guard.observe(bright, visible).paper_exposed is True

    dim = np.full((120, 160, 3), 150, dtype=np.uint8)
    assessment = guard.observe(dim, visible)
    assert assessment.visible is None
    assert assessment.paper_exposed is True

    # Com o nível novo absorvido, um bloco escuro real segue lido
    # como oclusão — a absorção não anestesia a guarda.
    shadow = dim.copy()
    shadow[24:96, 24:96] = 60
    assessment = guard.observe(shadow, visible)
    assert assessment.visible is not None
    assert not assessment.visible[60, 60]


def test_global_color_shift_is_absorbed_not_blocked():
    """Mudança global de temperatura de cor (luz azul sobre papel
    neutro): o croma é canal/grayscale e passa de 2 — com o teto
    antigo na referência de cor sobrava um resíduo permanente e a
    página inteira ficava vetada enquanto a luz durasse (Devin
    Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    neutral = np.full((120, 160, 3), 220, dtype=np.uint8)
    visible = np.ones(neutral.shape[:2], dtype=bool)
    assert guard.observe(neutral, visible).paper_exposed is True

    # BGR (190, 65, 55): grayscale ~76, croma do canal B ~2.5.
    blue_lit = np.full((120, 160, 3), (190, 65, 55), dtype=np.uint8)
    for _ in range(2):
        assessment = guard.observe(blue_lit, visible)
        assert assessment.visible is None
        assert assessment.paper_exposed is True

    # Sob a luz nova, um recorte que foge do ambiente — papel no tom
    # neutro antigo — segue bloqueado: a absorção não anestesia a
    # guarda.
    pale = blue_lit.copy()
    pale[24:96, 24:96] = (220, 220, 220)
    assessment = guard.observe(pale, visible)
    assert assessment.visible is not None
    assert not assessment.visible[60, 60]
    assert not assessment.paper_exposed[60, 60]

    # E o tom azulado num recorte local sobre papel neutro continua
    # sendo anomalia — nunca vira referência global.
    local = occlusion_mod.PaperExposureGuard()
    assert local.observe(neutral, visible).paper_exposed is True
    patch = neutral.copy()
    patch[24:96, 24:96] = (190, 65, 55)
    assessment = local.observe(patch, visible)
    assert assessment.visible is not None
    assert not assessment.visible[60, 60]
    assert not assessment.paper_exposed[60, 60]


def test_global_dimming_preserves_a_simultaneous_local_shadow():
    """O fator global da sala não pode ensinar junto uma sombra
    local: papel_exposed fica falso durante toda a janela de remoção."""
    guard = occlusion_mod.PaperExposureGuard()
    bright = np.full((240, 240, 3), 220, dtype=np.uint8)
    visible = np.ones(bright.shape[:2], dtype=bool)
    guard.observe(bright, visible)

    dim_shadow = np.full_like(bright, 150)
    dim_shadow[96:120, 96:120] = 60
    for _ in range(registration_mod._REMOVAL_PASSES + 2):
        assessment = guard.observe(dim_shadow, visible)
        assert not assessment.paper_exposed[100:110, 100:110].any()


def test_global_dimming_does_not_relearn_a_smooth_cover_over_print():
    """Uma capa lisa que aparece sobre uma área texturada junto com
    a queda global de luz ainda difere da referência do próprio tile.
    A semelhança com os tiles brancos vizinhos não pode apagar esse
    veto e autorizar remoção sob a capa (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    bright = np.full((120, 168, 3), 220, dtype=np.uint8)
    # Um tile impresso isolado, cercado por tiles em branco. A capa
    # lisa vai combinar com esses vizinhos, mas não com a referência
    # própria do tile impresso.
    for row in range(48, 72, 4):
        bright[row : row + 1, 72:96] = 175
    visible = np.ones(bright.shape[:2], dtype=bool)
    blank_ink = np.zeros(bright.shape[:2], np.float32)
    guard.observe(bright, visible, ink_prob=blank_ink)

    dimmed_cover = np.full_like(bright, 150)
    for _ in range(registration_mod._REMOVAL_PASSES + 2):
        assessment = guard.observe(dimmed_cover, visible, ink_prob=blank_ink)
        if isinstance(assessment.paper_exposed, (bool, np.bool_)):
            assert not assessment.paper_exposed
        else:
            assert not assessment.paper_exposed[48:72, 72:96].any()


def test_global_dimming_flattens_the_former_bright_region():
    """A lâmpada direcional apaga e a página nivela no brilho do lado
    escuro: a razão mediana escalava junto o gradiente antigo e a
    faixa antes mais clara virava veto permanente — residual pequeno
    demais para reabrir a absorção e fora do alcance do refresh, que
    só corrige tiles sem anomalia. O nivelamento reancora esses tiles
    no nível medido e solta a página (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    # Grade exata de tiles: sem reduzir 160 px em 7 colunas, o
    # reamostrador mistura a borda clara com papel e muda a textura
    # aprendida do próprio tile.
    lit = np.full((120, 168, 3), 190, dtype=np.uint8)
    lit[24:96, 24:96] = 250  # faixa sob a lâmpada (bloco 3x3 tiles)
    visible = np.ones(lit.shape[:2], dtype=bool)
    assert guard.observe(lit, visible).paper_exposed is True
    assert guard.observe(lit, visible).paper_exposed is True

    transient_texture = lit.copy()
    region = transient_texture[24:96, 24:96]
    yy, xx = np.indices(region.shape[:2])
    region[:] = np.where(((xx + yy) % 2)[..., None], 220, 255)
    transient = guard.observe(transient_texture, visible)
    assert not transient.paper_exposed[24:96, 24:96].all()

    dim = np.full_like(lit, 140)  # lâmpada apagada
    for _ in range(2):
        assessment = guard.observe(dim, visible)
        assert assessment.visible is None
        assert np.asarray(assessment.paper_exposed).all()


def test_fill_recovery_exempts_supported_pixels_across_tile_borders():
    """Um preenchimento deslocado da grade recupera também pixels
    fortes das tiles de borda e sobrevive à dilatação dos vizinhos."""
    guard = occlusion_mod.PaperExposureGuard()
    paper = np.full((120, 168, 3), 220, dtype=np.uint8)
    guard.observe(paper)
    filled = paper.copy()
    filled[37:83, 37:83] = 60
    prob = np.zeros((120, 168), np.float32)
    prob[37:83, 37:83] = 0.95

    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES):
        assessment = guard.observe(filled, ink_prob=prob)

    assert assessment.visible is not None
    assert assessment.visible[37:83, 37:83].all()
    assert not assessment.paper_exposed[37:83, 37:83].any()


def test_texture_mismatch_blocks_removal_under_a_paper_colored_ruler():
    """Uma régua clara esconde a pauta e imita a luma/croma do papel.

    A variância local muda mesmo quando a cor média permanece igual, então
    a guarda não autoriza apagar o traço que o modelo deixou de ver.
    """
    shape = (120, 168)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    for row in range(12, shape[0], 24):
        cv2.line(paper, (0, row), (shape[1] - 1, row), (175, 175, 175), 1)
    ruler = paper.copy()
    ruler[24:96, 24:120] = 220

    old = np.zeros(shape, dtype=np.float32)
    old[8:20, 8:20] = 0.95  # tinta-testemunha fora da régua
    old[52:60, 48:80] = 0.95  # traço confirmado sob a régua
    erased = old.copy()
    erased[52:60, 48:80] = 0.0

    class _MutablePageClean:
        threshold = 0.7

        def __init__(self):
            self.prob = np.zeros(shape, np.float32)

        def ink_prob(self, _image):
            return self.prob

    clock = rr.SceneClock()
    page_clean = _MutablePageClean()
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    for _ in range(4):
        clock.now += 1
        worker.submit(WarpedPage(paper, generation, corners))
        assert worker.pump()
        assert worker.error is None

    page_clean.prob = old
    for _ in range(16):
        clock.now += 1
        worker.submit(WarpedPage(paper, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert worker._accumulator._confirmed[52:60, 48:80].all()

    page_clean.prob = erased
    for _ in range(occlusion_mod._RECOVERY_WAIT_PASSES + 4):
        clock.now += 1
        worker.submit(WarpedPage(ruler, generation, corners))
        assert worker.pump()
        assert worker.error is None

    assert worker._accumulator._confirmed[52:60, 48:80].all()

    # A queda global de luz não pode ensinar a textura da régua como papel.
    dim_ruler = np.rint(ruler.astype(np.float32) * (160.0 / 220.0)).astype(np.uint8)
    for _ in range(registration_mod._REMOVAL_PASSES + 2):
        clock.now += 1
        worker.submit(WarpedPage(dim_ruler, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert worker._accumulator._confirmed[52:60, 48:80].all()


def test_marked_ruler_does_not_erase_the_covered_stroke():
    """Régua cor de papel COM marcas: o modelo lê os traços da régua
    como tinta e a amostra ponderada os descarta — sem o veto de
    tinta nova, a textura restante casava com o papel e a falta
    apagava o traço escondido (Devin Review do PR #144)."""
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    target = (slice(52, 60), slice(48, 80))  # traço que a régua cobre
    witness = (slice(8, 20), slice(8, 20))  # testemunha de remoção
    page = paper.copy()
    page[target] = 60
    page[witness] = 60

    old = np.zeros(shape, np.float32)
    old[target] = 0.95
    old[witness] = 0.95
    marked = np.zeros(shape, np.float32)
    marked[witness] = 0.95

    ruler = paper.copy()
    ticks = np.zeros(shape, dtype=bool)
    for c in range(52, 120, 16):
        ticks[24:96, c] = True  # marcas escuras nos tiles da régua
    ruler[ticks] = 60
    marked[ticks] = 0.9  # o modelo lê as marcas como tinta

    class _MutablePageClean:
        threshold = 0.7

        def __init__(self):
            self.prob = np.zeros(shape, np.float32)

        def ink_prob(self, _image):
            return self.prob

    clock = rr.SceneClock()
    page_clean = _MutablePageClean()
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    for _ in range(4):
        clock.now += 1
        worker.submit(WarpedPage(paper, generation, corners))
        assert worker.pump()
        assert worker.error is None

    page_clean.prob = old
    for _ in range(16):
        clock.now += 1
        worker.submit(WarpedPage(page, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert worker._accumulator._confirmed[target].all()

    page_clean.prob = marked
    # Passadas testemunhadas de sobra depois dos ticks confirmarem:
    # sem o veto, a remoção (6 passadas) já teria andado — o traço
    # sobreviver por menos que isso provava nada (Devin Review do
    # PR #144).
    for _ in range(registration_mod._REMOVAL_PASSES * 2):
        clock.now += 1
        worker.submit(WarpedPage(ruler, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert worker._accumulator._confirmed[target].all()
    records = worker.telemetry_snapshot()["records"]
    assert sum(r["removed_confirmed_px"] for r in records) == 0

    # A mudança global de luz não pode limpar o veto dos ticks locais.
    dimmed_ruler = np.rint(ruler.astype(np.float32) * (160.0 / 220.0)).astype(np.uint8)
    for _ in range(registration_mod._REMOVAL_PASSES * 2):
        clock.now += 1
        worker.submit(WarpedPage(dimmed_ruler, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert worker._accumulator._confirmed[target].all()


def test_partial_erasure_of_a_dense_patch_still_counts_cleared_pixels():
    """Metade do tile apagada: a cobertura insuficiente para APRENDER
    a referência não pode vetar o voto de remoção dos pixels limpos.
    O veto por pixel (dark_ink) já protege a metade ainda grafite
    (Devin Review do PR #144).

    O tile do patch nunca aprendeu referência (58% de tinta já na
    primeira observação), então a evidência corre por pixel."""
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    patch = (slice(24, 48), slice(0, 14))  # ~58% do tile (1,0)
    page = paper.copy()
    page[patch] = 160
    witness = (slice(60, 120), slice(88, 160))
    page[witness] = 60

    prob_full = np.zeros(shape, np.float32)
    prob_full[patch] = 0.95
    prob_full[witness] = 0.95
    cleared = (slice(24, 48), slice(12, 14))
    prob_partial = prob_full.copy()
    prob_partial[cleared] = 0.0
    erased_page = page.copy()
    erased_page[cleared] = 220  # a borda apagada volta ao papel

    class _MutablePageClean:
        threshold = 0.7

        def __init__(self):
            self.prob = prob_full

        def ink_prob(self, _image):
            return self.prob

    clock = rr.SceneClock()
    page_clean = _MutablePageClean()
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    for _ in range(16):
        clock.now += 1
        worker.submit(WarpedPage(page, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert worker._accumulator._confirmed[patch].all()

    page_clean.prob = prob_partial
    for _ in range(registration_mod._REMOVAL_PASSES + 2):
        clock.now += 1
        worker.submit(WarpedPage(erased_page, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert not worker._accumulator._confirmed[cleared].any()
    assert worker._accumulator._confirmed[24:48, 0:12].all()


def test_partial_erasure_on_a_tile_learned_with_ink_counts_cleared_pixels():
    """Tile aprendido JÁ com traço: a baseline grava a fração de tinta
    presente no aprendizado — apagar metade dela não é tinta nova, e a
    metade limpa prova papel e vota a remoção. Sem a comparação com a
    baseline, qualquer resto ≥1% segurava o tile inteiro e a borda
    apagada nunca saía do canvas (Devin Review do PR #144)."""
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    # Tile (1,0) cobre linhas 24:48 e colunas ~0:23; a banda ocupa 6
    # colunas (~26% do tile) — aprende a referência com ela.
    patch = (slice(24, 48), slice(0, 6))
    page = paper.copy()
    page[patch] = 160
    witness = (slice(60, 120), slice(88, 160))
    page[witness] = 60

    prob_full = np.zeros(shape, np.float32)
    prob_full[patch] = 0.95
    prob_full[witness] = 0.95
    cleared = (slice(24, 48), slice(3, 6))  # metade da banda apagada
    prob_partial = prob_full.copy()
    prob_partial[cleared] = 0.0
    erased_page = page.copy()
    erased_page[cleared] = 220

    class _MutablePageClean:
        threshold = 0.7

        def __init__(self):
            self.prob = prob_full

        def ink_prob(self, _image):
            return self.prob

    clock = rr.SceneClock()
    page_clean = _MutablePageClean()
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    for _ in range(16):
        clock.now += 1
        worker.submit(WarpedPage(page, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert worker._accumulator._confirmed[patch].all()

    page_clean.prob = prob_partial
    for _ in range(registration_mod._REMOVAL_PASSES + 2):
        clock.now += 1
        worker.submit(WarpedPage(erased_page, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert not worker._accumulator._confirmed[cleared].any()
    assert worker._accumulator._confirmed[24:48, 0:3].all()
    assert worker._accumulator._confirmed[witness].all()


def test_erasing_preexisting_pencil_stroke_releases_texture_veto():
    """A referência de textura usa o fundo do tile, não o grafite presente.

    A imagem também muda quando a ponta é apagada; o pixel de lápis volta ao
    papel, a textura converge para a referência e a votação de remoção anda.
    """
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    with_stroke = paper.copy()
    stroke = (slice(36, 38), slice(24, 48))  # linha de 2 px em um tile
    with_stroke[stroke] = 60

    ink_full = np.zeros(shape, dtype=np.float32)
    ink_full[stroke] = 0.95
    ink_full[8:20, 8:20] = 0.95  # testemunha que continua na folha
    ink_erased = ink_full.copy()
    ink_erased[stroke] = 0.0
    erased_page = paper.copy()

    class _MutablePageClean:
        threshold = 0.7

        def __init__(self):
            self.prob = ink_full

        def ink_prob(self, _image):
            return self.prob

    clock = rr.SceneClock()
    page_clean = _MutablePageClean()
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    for _ in range(16):
        clock.now += 1
        worker.submit(WarpedPage(with_stroke, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert worker._accumulator._confirmed[stroke].all()

    page_clean.prob = ink_erased
    for _ in range(registration_mod._REMOVAL_PASSES + 2):
        clock.now += 1
        worker.submit(WarpedPage(erased_page, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert not worker._accumulator._confirmed[stroke].any()


def test_tile_hidden_at_startup_learns_its_local_rule_texture():
    """A tile first hidden by a hand learns ruled paper from its row neighbors."""
    shape = (120, 168)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    for row in range(12, shape[0], 24):
        cv2.line(paper, (0, row), (shape[1] - 1, row), (175, 175, 175), 1)

    visible = np.ones(shape, dtype=bool)
    visible[24:48, 24:48] = False
    hand = paper.copy()
    hand[24:48, 24:48] = 60
    guard = occlusion_mod.PaperExposureGuard()
    guard.observe(hand, visible)

    pencil_page = paper.copy()
    stroke = (slice(28, 30), slice(28, 46))
    pencil_page[stroke] = 60
    ink = np.zeros(shape, dtype=np.float32)
    ink[stroke] = 0.95
    assert not guard.observe(pencil_page, ink_prob=ink).paper_exposed[stroke].any()

    erased_page = paper.copy()
    clear_read = guard.observe(erased_page, ink_prob=np.zeros_like(ink))
    # Página limpa pode devolver o True escalar (papel todo exposto)
    # ou a máscara — as duas formas têm de incluir o traço apagado.
    assert clear_read.paper_exposed is True or clear_read.paper_exposed[stroke].all()


def test_hidden_blank_tile_uses_the_local_majority_beside_print():
    """Um único vizinho impresso não pode ensinar textura falsa ao tile
    em branco que ficou escondido na primeira observação."""
    shape = (168, 168)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    printed = paper.copy()
    for row in range(26, 47, 4):
        cv2.line(printed, (48, row), (71, row), (190, 190, 190), 1)

    visible = np.ones(shape, dtype=bool)
    hidden_tile = (slice(24, 48), slice(24, 48))
    visible[hidden_tile] = False
    hand = printed.copy()
    hand[hidden_tile] = 60
    guard = occlusion_mod.PaperExposureGuard()
    guard.observe(hand, visible)

    assessment = guard.observe(paper, ink_prob=np.zeros(shape, np.float32))
    assert np.asarray(assessment.paper_exposed)[hidden_tile].all()

    stroke = (slice(30, 32), slice(28, 46))
    pencil = paper.copy()
    pencil[stroke] = 60
    ink = np.zeros(shape, np.float32)
    ink[stroke] = 0.95
    guard.observe(pencil, ink_prob=ink)
    erased = guard.observe(paper, ink_prob=np.zeros_like(ink))
    assert np.asarray(erased.paper_exposed)[stroke].all()


def test_texture_is_stable_under_small_quantized_lighting_drift():
    """A quantização diferente do tile mean não inventa textura em papel limpo."""
    shape = (120, 168)
    yy, xx = np.indices(shape)
    paper = (220 + ((xx + 3 * yy) % 5 == 0)).astype(np.uint8)
    guard = occlusion_mod.PaperExposureGuard()
    guard.observe(paper)
    guard.observe(paper)

    drifted = np.clip(paper.astype(np.int16) + 1, 0, 255).astype(np.uint8)
    for _ in range(4):
        assessment = guard.observe(drifted)
        assert np.asarray(assessment.paper_exposed).all()


def test_fill_region_recovers_additions_but_never_proves_paper():
    """Preenchimento: o veto de adição cai com evidência de tinta
    sustentada e a mão fora do tile; o de remoção fica — papel
    exposto nunca é provado sobre o preenchimento."""
    guard = occlusion_mod.PaperExposureGuard()
    paper = np.full((120, 168, 3), 210, dtype=np.uint8)
    for _ in range(2):
        assert guard.observe(paper).paper_exposed is True

    filled = paper.copy()
    filled[24:96, 0:72] = 60  # tiles (1:4, 0:3): o preenchimento
    filled[24:96, 96:168] = 60  # tiles (1:4, 4:7): oclusor escuro
    ink = np.zeros((120, 168), np.float32)
    ink[24:96, 0:72] = 0.95  # o modelo lê o preenchimento como tinta

    first = guard.observe(filled, ink_prob=ink)
    assert not first.visible[48, 48]
    assert not first.visible[48, 120]
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES - 1):
        last = guard.observe(filled, ink_prob=ink)
    assert last.visible[48, 48]
    assert not last.visible[48, 120]
    assert not last.paper_exposed[48, 48]
    assert not last.paper_exposed[48, 120]


def test_fill_streak_only_counts_with_a_clear_hand():
    """A contagem de evidência não anda com a mão sobre o tile —
    a adoção do preenchimento exige a mão fora do caminho."""
    guard = occlusion_mod.PaperExposureGuard()
    paper = np.full((96, 96, 3), 210, dtype=np.uint8)
    guard.observe(paper)

    filled = paper.copy()
    filled[24:96, 24:96] = 60
    ink = np.zeros((96, 96), np.float32)
    ink[24:96, 24:96] = 0.95
    hand = np.ones((96, 96), dtype=bool)
    hand[24:96, 24:96] = False
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES + 1):
        covered = guard.observe(filled, hand, ink_prob=ink)
    assert not covered.visible[48, 48]

    clear = np.ones((96, 96), dtype=bool)
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES):
        last = guard.observe(filled, clear, ink_prob=ink)
    # A única anomalia era o preenchimento: adotado, o bloco esvazia e
    # a região volta a votar (à vista ou mascarada só onde restou
    # veto) — mas papel exposto continua não provado sobre ela.
    assert last.visible is None or last.visible[48, 48]
    assert not last.paper_exposed[48, 48]


def test_fill_recovery_counts_evidence_at_the_caller_vote_level():
    """O nível de "tinta forte" vem do chamador: um preenchimento
    sustentado a 0,6 — abaixo do piso histórico de 0,7, mas dentro do
    que o voto confirma (vote_hit 0,5) — adota quando a guarda mede a
    evidência no nível do voto; no piso fixo o mesmo bloco ficava
    vetado para sempre (Devin Review do PR #144)."""
    paper = np.full((120, 168, 3), 220, dtype=np.uint8)
    filled = paper.copy()
    filled[24:96, 24:96] = 60
    ink = np.zeros((120, 168), np.float32)
    ink[24:96, 24:96] = 0.6

    vetoed = occlusion_mod.PaperExposureGuard()
    vetoed.observe(paper)
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES + 2):
        blocked = vetoed.observe(filled, ink_prob=ink)
    assert not blocked.visible[48, 48]

    guard = occlusion_mod.PaperExposureGuard()
    guard.observe(paper)
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES):
        assessment = guard.observe(filled, ink_prob=ink, ink_support_threshold=0.5)
    assert assessment.visible is None or assessment.visible[48, 48]
    assert not assessment.paper_exposed[48, 48]


def test_fill_support_rejects_a_thin_strong_ink_band():
    """Faixa fina e forte não é preenchimento: 4 px de altura a 0,9
    cobrem ~17% dos pixels do tile (passam a fração 0,15) mas a média
    do tile fica ~0,15 — assinatura de borda de sombra parada, não de
    área pintada. O piso de média por tile segura a adoção e o veto
    de adição permanece (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    paper = np.full((120, 168, 3), 220, dtype=np.uint8)
    guard.observe(paper)
    shadow = paper.copy()
    shadow[24:96, 24:96] = 60  # sombra parada: tiles (1:4, 1:4) anômalos
    ink = np.zeros((120, 168), np.float32)
    ink[40:44, 24:96] = 0.9  # faixa fina lida como tinta dentro da sombra
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES + 2):
        assessment = guard.observe(shadow, ink_prob=ink, ink_support_threshold=0.5)
    assert assessment.visible is not None
    assert not assessment.visible[40:44, 24:96].any()


def test_narrow_dark_fill_recovers_from_one_row_of_dense_tile_support():
    """A 3x4 anomaly may have only one dense support row: its partially
    filled edge rows still belong to that real fill. Dense evidence in
    the blocking component seeds recovery (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    paper = np.full((120, 192, 3), 220, dtype=np.uint8)
    guard.observe(paper)

    filled = paper.copy()
    filled[41:79, 48:144] = 10  # uma linha inteira e 7 px nas bordas
    ink = np.zeros((120, 192), np.float32)
    ink[41:79, 48:144] = 0.95
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES):
        assessment = guard.observe(filled, ink_prob=ink)

    assert assessment.visible is not None
    assert assessment.visible[41:79, 48:144].all()
    assert not assessment.paper_exposed[41:79, 48:144].any()


def test_fill_edge_partially_covering_a_tile_still_votes():
    """Borda de preenchimento cobrindo ~1/4 do tile de borda: 6 das
    24 linhas a 0,95 dão média ~0,2375 no tile — o piso de média
    (que existe para barrar faixas finas de sombra) nunca deixa a
    borda adotar, e as 6 linhas pintadas do topo ficavam vetadas
    para sempre mesmo com o miolo confirmando. O miolo adotado
    ancora a reabertura: os pixels escuros e fortes conectados a
    ele votam, o papel descoberto do mesmo tile segue vetado, e a
    faixa fina isolada dentro de outra sombra — sem âncora
    adotada — continua bloqueada (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    paper = np.full((240, 240, 3), 220, dtype=np.uint8)
    guard.observe(paper)

    filled = paper.copy()
    filled[42:120, 48:144] = 10  # topo cobre 6/24 px do tile de borda
    filled[168:240, 24:120] = 60  # sombra isolada (bloco 3x4 tiles)
    ink = np.zeros((240, 240), np.float32)
    ink[42:120, 48:144] = 0.95
    ink[180:184, 24:120] = 0.9  # faixa fina e forte dentro da sombra

    first = guard.observe(filled, ink_prob=ink)
    assert not first.visible[48, 96]  # antes da adoção o veto vale
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES):
        assessment = guard.observe(filled, ink_prob=ink)

    assert assessment.visible is not None
    assert assessment.visible[42:120, 48:144].all()  # borda vota junto
    # O papel descoberto do mesmo tile de borda segue vetado.
    assert not assessment.visible[24:42, 48:144].any()
    assert not assessment.visible[180:184, 24:120].any()  # faixa sem âncora
    assert not assessment.visible[200, 60]  # sombra pura segue vetada
    assert not assessment.paper_exposed[42:120, 48:144].any()


def test_hand_masked_gap_does_not_bridge_fill_to_shadow_edge():
    """Ponte fantasma por baixo da mão: a extensão por componente
    conexo corria sobre o mapa cru de dark_ink — uma mão tampando o
    fosso entre o preenchimento e uma faixa escura dentro de uma
    sombra fazia a faixa votar como parte do preenchimento. A
    conexão só pode atravessar pixels realmente à vista
    (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    paper = np.full((240, 240, 3), 220, dtype=np.uint8)
    guard.observe(paper)

    scene = paper.copy()
    scene[42:120, 48:144] = 10  # preenchimento adotável
    scene[42:240, 136:150] = 10  # corredor escuro: a ponte sob a mão
    scene[168:240, 144:240] = 60  # sombra isolada (bloco 3x4 tiles)
    ink = np.zeros((240, 240), np.float32)
    ink[42:120, 48:144] = 0.95
    ink[120:170, 136:150] = 0.9  # falso ink só no trecho coberto pela mão
    ink[168:172, 144:216] = 0.9  # faixa cruza o limite da máscara da mão

    visible = np.ones((240, 240), bool)
    visible[120:170, 100:200] = False  # mão tampa o corredor no meio

    guard.observe(scene, visible=visible, ink_prob=ink)
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES):
        assessment = guard.observe(scene, visible=visible, ink_prob=ink)

    assert assessment.visible[42:120, 48:144].all()  # miolo vota
    assert not assessment.visible[170, 180]  # faixa não vota: ponte cortada
    assert not assessment.visible[200, 192]  # sombra segue vetada


def test_missed_confirmed_ink_cannot_bridge_fill_recovery_to_shadow():
    """Tinta confirmada que o modelo perdeu pode vetar apagamento, mas
    não pode conectar um fill adotado à borda falsa de uma sombra nem
    liberar pixels dela para novas adições (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    paper = np.full((240, 240, 3), 220, dtype=np.uint8)
    guard.observe(paper)

    scene = paper.copy()
    fill = (slice(48, 120), slice(48, 120))
    shadow = (slice(168, 240), slice(144, 240))
    scene[fill] = 10
    scene[shadow] = 60
    confirmed = np.zeros((240, 240), dtype=bool)
    confirmed[84:90, 120:180] = True
    confirmed[84:180, 174:180] = True
    scene[confirmed] = 10
    ink = np.zeros((240, 240), np.float32)
    ink[fill] = 0.95
    ink[180:184, 168:240] = 0.9  # faixa fina que não deve ser adotada

    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES):
        assessment = guard.observe(scene, ink_prob=ink, confirmed_ink=confirmed)

    assert assessment.visible is not None
    assert assessment.visible[48:120, 48:120].all()
    assert not assessment.visible[180:184, 168:240].any()
    assert not assessment.paper_exposed[confirmed].any()


def test_uncertain_photometry_does_not_advance_removal_streak():
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = _big_ink()
    confirmed(acc, prob)
    missing = prob.copy()
    missing[10:190, 10:110] = 0.0
    visible = np.ones(prob.shape, bool)  # MediaPipe missed the hand
    exposed = visible.copy()
    exposed[10:190, 10:110] = False  # photo guard sees a dark block

    for _ in range(2 * registration_mod._REMOVAL_PASSES):
        acc.add(missing, visible, paper_exposed=exposed)

    assert np.all(acc._confirmed[10:190, 10:110] == 255)
    assert not acc._removal_streak[10:190, 10:110].any()


def test_clear_ink_read_resets_streak_even_where_paper_is_unproven():
    """Leitura clara zera o streak mesmo com papel não provado.

    O exemplo do Devin Review do PR #144: o veto dark_ink da
    guarda (tinta forte sobre luma escura nunca prova papel) casa
    com todo pixel de traço real e tira o paper_exposed dele
    sempre que ALGUM tile da página está anômalo — a leitura
    clara segurava o streak em vez de zerá-lo, e faltas
    espalhadas somavam até apagar tinta confirmada.
    """
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = _big_ink()
    confirmed(acc, prob)
    target = (slice(10, 60), slice(10, 60))
    visible = np.ones(prob.shape, bool)
    unproven = visible.copy()
    unproven[target] = False  # o veto dark_ink sobre o traço lido
    misread = prob.copy()
    misread[target] = 0.05

    acc.add(misread, visible, paper_exposed=visible)  # falta conta
    assert np.all(acc._removal_streak[target] == 1)
    # Leitura clara com o traço fora do paper_exposed: evidência
    # positiva — o streak zera, não segura.
    acc.add(prob, visible, paper_exposed=unproven)
    assert not acc._removal_streak[target].any()

    for _ in range(registration_mod._REMOVAL_PASSES + 2):
        result = acc.add(misread, visible, paper_exposed=visible)
        assert result["removed_confirmed_px"] == 0
        result = acc.add(prob, visible, paper_exposed=unproven)
        assert result["removed_confirmed_px"] == 0
        assert not acc._removal_streak[target].any()
    assert np.all(acc._confirmed[target] == 255)


def test_dark_ink_veto_keeps_resetting_the_streak_end_to_end():
    """Caso do review pelo caminho real: a falta testemunhada conta
    quando o traço some da câmera (streak sobe), e a leitura forte
    sob veto fotométrico zera o streak em vez de segurá-lo —
    faltas espalhadas não apagam tinta confirmada
    (Devin Review do PR #144)."""
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    stroke = (slice(8, 12), slice(8, 88))  # traço fino e escuro
    page = paper.copy()
    page[stroke] = 60  # luma escura: o veto dark_ink o descreve
    shadow = (slice(48, 96), slice(96, 144))  # anomalia 2x3 tiles
    shadowed = page.copy()
    shadowed[shadow] = 60

    # Testemunha longe dos tiles do traço: a amostragem INTER_AREA
    # sangra a tinta dela para os tiles de borda e o veto de tinta
    # nova (correto) impediria o traço de contar falta ali.
    witness = (slice(48, 120), slice(8, 72))  # ~4,6k px: testemunha
    prob_full = np.zeros(shape, np.float32)
    prob_full[witness] = 0.95
    prob_full[stroke] = 0.95
    prob_misread = prob_full.copy()
    prob_misread[stroke] = 0.05

    clock = rr.SceneClock()
    page_clean = _MutablePageClean(prob_full)
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    visible = np.ones((720, 1280), bool)

    def pass_page(image, prob):
        page_clean.prob = prob
        clock.now += 1
        worker.submit(WarpedPage(image, generation, corners, visible=visible))
        assert worker.pump()
        assert worker.error is None

    for _ in range(8):
        pass_page(page, prob_full)
    assert worker._accumulator._confirmed[stroke].all()

    # O traço some da CÂMERA (não só do modelo) na passada de falta: o
    # tile volta a parecer papel, a falta testemunhada conta e o streak
    # sobe — depois a leitura forte sob o veto dark_ink tem de zerar um
    # streak que existe de verdade.
    erased = paper.copy()
    for _ in range(registration_mod._REMOVAL_PASSES + 2):
        pass_page(erased, prob_misread)
        assert np.all(worker._accumulator._removal_streak[stroke] == 1)
        # Traço lido a 0,95 sob página anômala: veto dark_ink tira o
        # paper_exposed, mas a evidência positiva zera o streak — não o
        # segura até apagar (Devin Review do PR #144).
        pass_page(shadowed, prob_full)
        assert not worker._accumulator._removal_streak[stroke].any()

    assert worker._accumulator._confirmed[stroke].all()
    assert worker._accumulator._confirmed[witness].all()
    records = worker.telemetry_snapshot()["records"]
    assert sum(r["removed_confirmed_px"] for r in records) == 0


def test_hand_mask_dropouts_cannot_erase_or_add_hand_edges_and_exit_confirms():
    shape = (240, 320)
    old = np.zeros(shape, np.float32)
    old[90:106, 90:120] = 0.95
    old[10:25, 10:35] = 0.95  # tinta-testemunha distante da mão
    new = old.copy()
    new[180:196, 260:284] = 0.95
    shadow = np.full((*shape, 3), 220, dtype=np.uint8)
    hand = shadow.copy()
    hand[48:144, 48:168] = 75
    hand_edge = old.copy()
    hand_edge[60:70, 145:160] = 0.95
    covered_old = hand_edge.copy()
    covered_old[90:106, 90:120] = 0.0

    clock = rr.SceneClock()
    page_clean = _MutablePageClean(old)
    worker = rr.make_ink_worker(page_clean, clock, config=InkConfig())
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    camera_to_page = registration_mod._corners_page_matrix(corners, shape[1], shape[0])

    def camera_visibility_for(page_visibility):
        return cv2.warpPerspective(
            page_visibility.astype(np.uint8),
            np.linalg.inv(camera_to_page),
            (1280, 720),
            flags=cv2.INTER_NEAREST,
            borderValue=1,
        ).astype(bool)

    all_visible = np.ones(shape, bool)

    def pass_page(image, probability, visible):
        page_clean.prob = probability
        clock.now += 1
        worker.submit(
            WarpedPage(
                image,
                generation,
                corners,
                visible=camera_visibility_for(visible),
            )
        )
        assert worker.pump()
        assert worker.error is None

    for _ in range(16):
        pass_page(shadow, old, all_visible)
    original = worker._accumulator._confirmed.copy()
    assert original[90:106, 90:120].all()

    hand_visible = all_visible.copy()
    hand_visible[48:144, 48:168] = False
    for index in range(registration_mod._REMOVAL_PASSES * 2):
        # A periodic all-visible result models 1–3 missed masks while
        # the image itself keeps the photometric guard active.
        visible = all_visible if index % 4 == 0 else hand_visible
        pass_page(hand, covered_old, visible)

    result = worker._accumulator
    np.testing.assert_array_equal(result._confirmed, original)
    assert not result._provisional[48:144, 48:168].any()
    assert not result._confirmed[60:70, 145:160].any()

    for _ in range(8):
        pass_page(shadow, new, all_visible)
    assert result._confirmed[90:106, 90:120].all()
    assert result._confirmed[180:196, 260:284].all()


def test_global_dimming_releases_tiles_hidden_by_the_hand():
    """A absorção de shift global escala TODAS as referências
    conhecidas — inclusive as escondidas pela mão. Sem isso o tile
    coberto ficava claro demais e, ao reaparecer no nível novo,
    virava um bloco escuro permanente (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    bright = np.full((120, 160, 3), 220, dtype=np.uint8)
    clear = np.ones(bright.shape[:2], dtype=bool)
    for _ in range(2):
        assert guard.observe(bright, clear).paper_exposed is True

    hand = clear.copy()
    hand[24:96, 24:96] = False  # 3x3 tiles sob a mão
    dim = np.full((120, 160, 3), 150, dtype=np.uint8)
    absorbed = guard.observe(dim, hand)
    assert absorbed.visible is None  # shift absorvido, não bloqueio

    revealed = guard.observe(dim, clear)
    assert revealed.visible is None
    assert revealed.paper_exposed is True


def test_fill_recovery_keeps_the_worker_awake_past_the_exit_window():
    """A janela pós-saída do voto legado tem 1 passada, mas o streak
    de preenchimento precisa de _FILL_RECOVERY_PASSES limpas: sem a
    trava, a trabalhadora dormia no meio da contagem e a tinta
    adotável nunca votava (Devin Review do PR #144)."""
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    filled = paper.copy()
    filled[24:96, 24:96] = 60
    prob_fill = np.zeros(shape, np.float32)
    prob_fill[24:96, 24:96] = 0.95

    clock = rr.SceneClock()
    page_clean = _MutablePageClean(np.zeros(shape, np.float32))
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    clear = np.ones(shape, dtype=bool)
    hand_covered = clear.copy()
    hand_covered[24:96, 24:96] = False

    def pass_page(image, prob, page_visible):
        page_clean.prob = prob
        clock.now += 1
        worker.submit(
            WarpedPage(
                image,
                generation,
                corners,
                visible=rr.camera_mask(page_visible, corners),
            )
        )
        ran = worker.pump()
        assert worker.error is None
        return ran

    worker.set_hand_presence(True)
    for _ in range(2):
        assert pass_page(filled, prob_fill, hand_covered)
    worker.set_hand_presence(False)

    ran = [
        pass_page(filled, prob_fill, clear)
        for _ in range(occlusion_mod._FILL_RECOVERY_PASSES + 3)
    ]
    assert all(ran)  # a porta de descanso não fechou durante o streak
    assert worker._accumulator._confirmed[24:96, 24:96].all()


def test_fill_recovery_uses_active_cadence_after_hand_exit():
    """A carência que mantém a worker acordada não pode reduzir a leitura a 1 Hz."""
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    filled = paper.copy()
    filled[24:96, 24:96] = 60
    ink_prob = np.zeros(shape, dtype=np.float32)
    ink_prob[24:96, 24:96] = 0.95

    class _MutablePageClean:
        threshold = 0.7

        def ink_prob(self, _image):
            return ink_prob

    worker = rr.make_ink_worker(_MutablePageClean(), rr.SceneClock())
    worker._paper_exposure.observe(paper)
    worker._paper_exposure.observe(filled, ink_prob=ink_prob)
    assert worker._paper_exposure.recovery_pending
    worker.set_hand_presence(False)
    worker._confirm_passes = 0

    interval = worker._pass_interval()
    assert interval == worker_mod._PAGE_CLEAN_INTERVAL_SECONDS
    assert interval != worker_mod._IDLE_PROBE_SECONDS


def test_page_recovery_ignores_false_ink_inside_the_photometric_veto():
    """Tinta falsa lida DENTRO do bloqueio fotométrico não entra na
    contenção nem fabrica recall dilatada: sem a interseção com a
    máscara à vista, a sombra acusava troca de página mesmo com a
    tinta anterior intacta à vista (Devin Review do PR #144)."""
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    prob = np.zeros(shape, np.float32)
    prob[10:20, 5:20] = 0.95  # traço real fora da futura sombra
    shadowed = paper.copy()
    shadowed[24:96, 48:144] = 60  # bloco 3x4 tiles
    ghost = prob.copy()
    ghost[30:90, 60:130] = 0.95  # "tinta" lida sob a sombra

    clock = rr.SceneClock()
    page_clean = _MutablePageClean(prob)
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    for _ in range(6):
        page_clean.prob = prob
        clock.now += 1
        worker.submit(WarpedPage(paper, generation, corners))
        assert worker.pump()
        assert worker.error is None

    generation = worker.new_generation(preserve_identity=True)
    page_clean.prob = ghost
    clock.now += 1
    worker.submit(WarpedPage(shadowed, generation, corners))
    assert worker.pump()
    assert worker.error is None
    event = worker.telemetry_snapshot()["page_identity_events"][-1]
    assert event["decision"] == "same_page"


def test_wrong_shape_visible_fails_closed_instead_of_open():
    """Máscara em geometria errada é bug do chamador: veta tudo —
    a passada não vota tinta de graça onde a oclusão está em dúvida."""
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = ink()
    bad = np.ones((10, 10), bool)
    for _ in range(6):
        result = acc.add(prob, bad)
    assert not result["confirmed"].any()


def test_wrong_shape_visible_returns_none_and_keeps_recovery_state():
    """Máscara fora da grade do warp: abstém sem varrer streaks."""
    guard = occlusion_mod.PaperExposureGuard()
    page = np.full((96, 144, 3), 220, np.uint8)
    page[24:72, 24:72] = (60, 62, 64)
    ink_prob = np.zeros((96, 144), np.float32)
    ink_prob[24:72, 24:72] = 0.95
    visible = np.ones((96, 144), bool)
    guard.observe(page, visible, ink_prob=ink_prob, ink_support_threshold=0.7)
    streak = guard._ink_streak.copy()
    assert streak.any()
    assert guard.observe(page, np.ones((8, 8), bool), ink_prob=ink_prob) is None
    np.testing.assert_array_equal(guard._ink_streak, streak)


def test_moved_marks_hold_the_new_ink_veto_when_total_coverage_drops():
    """Régua com ticks MENORES que o traço aprendido: a fração total
    de tinta cai abaixo da baseline e o veto por fração não dispara.
    A posição é o teste: ticks em pixels novos são marcas do oclusor
    e o traço coberto segue protegido (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    visible = np.ones(shape, dtype=bool)
    stroke = (slice(30, 33), slice(50, 68))  # 3x18px no tile (1,2)
    page = paper.copy()
    page[stroke] = 60
    prob = np.zeros(shape, np.float32)
    prob[stroke] = 0.95
    for _ in range(3):
        guard.observe(page, visible, ink_prob=prob, ink_support_threshold=0.7)
    # tile (1,2): linhas 24:48, colunas ~46:69 — aprendeu limpo com o
    # traço (~10% de baseline).
    assert guard._texture_learned_clean[1, 2]

    ruler = paper.copy()
    ticks = np.zeros(shape, dtype=bool)
    ticks[30:46, 55] = True
    ticks[30:46, 60] = True  # ticks finos: ~32px < 54px do traço
    ruler[ticks] = 60
    marked = np.zeros(shape, np.float32)
    marked[ticks] = 0.9  # o traço some da leitura; só os ticks ficam
    for _ in range(6):
        assessment = guard.observe(
            ruler, visible, ink_prob=marked, ink_support_threshold=0.7
        )
    assert not np.asarray(assessment.paper_exposed)[stroke].any()


def test_l_shaped_shadow_corner_cannot_seed_fill_recovery():
    """Duas faixas finas se encontrando num canto têm bbox 3x3 e
    passavam no portão 2-D: sem um tile denso de semente o componente
    não pode adotar (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    shape = (120, 168)  # grade exata de 24px
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    visible = np.ones(shape, dtype=bool)
    for _ in range(2):
        guard.observe(paper, visible)

    shadowed = paper.copy()
    shadowed[24:96, 24:96] = 150  # sombra 3x3 tiles
    bands = np.zeros(shape, np.float32)
    bands[24:31, 24:96] = 0.95  # faixa de 7px no topo
    bands[24:96, 24:31] = 0.95  # faixa de 7px na esquerda (L)
    # tile reto de faixa: 7/24 = 29% de suporte (mean ~0,28) — passa
    # no piso de suporte; tile de canto: 287/576 = 49,8% < 50% denso.
    for _ in range(6):
        guard.observe(shadowed, visible, ink_prob=bands, ink_support_threshold=0.7)
    assert not guard._ink_streak.any()
    assert not guard._fill_adopted.any()


def test_lamp_edge_tile_reanchors_when_the_light_flattens():
    """Tile de borda de lâmpada: textura aprendida veio de um degrau
    de luma (maioria dos pixels no lado escuro). Quando a luz nivela a
    luma escala junto com a página mas a textura some — mismatch
    explicado pelo próprio deslocamento, e o tile reancora em vez de
    vetar remoção para sempre. A baseline de tinta gravada não sobe
    no reancoramento (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    shape = (120, 160)
    visible = np.ones(shape, dtype=bool)
    lit = np.full((*shape, 3), 220, dtype=np.uint8)
    # tile (2,3): linhas 48:72, colunas ~69:92 — degrau de luma:
    # 2 colunas a 250, o resto a 190 (lado escuro domina).
    lit[:, 69:71] = 250
    lit[:, 71:92] = 190
    lit[46:52, 71:92] = 60  # linha de tinta ~15% do tile
    prob = np.zeros(shape, np.float32)
    prob[46:52, 71:92] = 0.9
    for _ in range(3):
        guard.observe(lit, visible, ink_prob=prob, ink_support_threshold=0.7)
    assert guard._texture_learned_clean[2, 3]
    baseline = float(guard._texture_ink_baseline[2, 3])
    assert 0.0 < baseline < 0.18

    # Lâmpada apaga: a página inteira nivela e o degrau some — a
    # textura que cai foi criada pelo próprio deslocamento.
    dim = np.full_like(lit, 140)
    dim[46:53, 71:92] = 40  # a tinta cresce um fio dentro do jitter
    dim_prob = np.zeros(shape, np.float32)
    dim_prob[46:53, 71:92] = 0.9
    assessment = guard.observe(
        dim, visible, ink_prob=dim_prob, ink_support_threshold=0.7
    )
    # A região do tile SEM tinta reabre (os pixels de tinta seguem
    # protegidos pelo veto dark_ink, por desenho).
    assert np.asarray(assessment.paper_exposed)[55:72, 71:92].all()
    # O reancoramento faz mínimo, não sobrescrita: a baseline gravada
    # não pode subir para a cobertura corrente (Devin Review #144).
    assert guard._texture_ink_baseline[2, 3] <= baseline


def test_moved_marks_keep_veto_through_the_worker():
    """Cenário do review na trabalhadora: traço confirmado coberto
    por régua cujos ticks somam menos área que o traço — a remoção
    não pode contar o apagamento falso (Devin Review do PR #144)."""
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    target = (slice(30, 33), slice(50, 68))  # traço coberto
    witness = (slice(80, 92), slice(8, 20))  # testemunha
    page = paper.copy()
    page[target] = 60
    page[witness] = 60

    prob_strokes = np.zeros(shape, np.float32)
    prob_strokes[target] = 0.95
    prob_strokes[witness] = 0.95

    ruler = paper.copy()
    ticks = np.zeros(shape, dtype=bool)
    ticks[30:46, 55] = True
    ticks[30:46, 60] = True
    ruler[ticks] = 60
    prob_ruler = np.zeros(shape, np.float32)
    prob_ruler[ticks] = 0.9
    prob_ruler[witness] = 0.95

    class _MutablePageClean:
        threshold = 0.7

        def __init__(self):
            self.prob = prob_strokes

        def ink_prob(self, _image):
            return self.prob

    clock = rr.SceneClock()
    page_clean = _MutablePageClean()
    worker = rr.make_ink_worker(page_clean, clock)
    generation = worker.new_generation()
    corners = rr.base_quad(shape)
    for _ in range(4):
        clock.now += 1
        worker.submit(WarpedPage(paper, generation, corners))
        assert worker.pump()
        assert worker.error is None
    for _ in range(16):
        clock.now += 1
        worker.submit(WarpedPage(page, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert worker._accumulator._confirmed[target].all()

    page_clean.prob = prob_ruler
    for _ in range(registration_mod._REMOVAL_PASSES * 2):
        clock.now += 1
        worker.submit(WarpedPage(ruler, generation, corners))
        assert worker.pump()
        assert worker.error is None
    assert worker._accumulator._confirmed[target].all()


def test_diagonal_stroke_seeds_fill_recovery():
    """Traço diagonal sólido (~18px de grossura, ~65% de cada tile que
    cruza): o blob enche ~100% do retângulo ORIENTADO — a semente não
    pode medir o bbox alinhado ao eixo (~8%), senão o traço nunca
    confirma. O canto em L segue oco em qualquer orientação (Devin
    Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    shape = (240, 240)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    visible = np.ones(shape, dtype=bool)
    for _ in range(2):
        guard.observe(paper, visible)

    dark = paper.copy()
    dark[96:168, 48:192] = 60  # sombra ~3x6 tiles, fora do shift global
    prob = np.zeros(shape, np.float32)
    prob[104:160, 56:184] = 0.0
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    diag = (
        (np.abs((xx - 48) - (yy - 96)) <= 9)
        & (yy >= 96)
        & (yy < 168)
        & (xx >= 48)
        & (xx < 192)
    )
    prob[diag] = 0.95  # diagonal grossa cruzando a sombra
    for _ in range(occlusion_mod._FILL_RECOVERY_PASSES + 1):
        guard.observe(dark, visible, ink_prob=prob, ink_support_threshold=0.7)
    assert guard._ink_streak.any()


def test_erase_and_redraw_lets_removal_finish_after_marks_hold_ages():
    """Apagamento + redesenho no mesmo tile: marcas novas em tile que
    segue provando papel caducam o veto por posição em _MARKS_HOLD_MAX
    — a remoção do traço antigo completa ao lado do substituto
    (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    visible = np.ones(shape, dtype=bool)
    stroke = (slice(30, 33), slice(50, 68))  # 3x18px no tile (1,2)
    page = paper.copy()
    page[stroke] = 60
    prob = np.zeros(shape, np.float32)
    prob[stroke] = 0.95
    for _ in range(3):
        guard.observe(page, visible, ink_prob=prob, ink_support_threshold=0.7)
    assert guard._texture_learned_clean[1, 2]

    redraw = (slice(40, 42), slice(50, 68))  # 2x18 no MESMO tile, fora da dilatação
    edited = paper.copy()  # traço antigo some; papel embaixo
    edited[redraw] = 60
    new_prob = np.zeros(shape, np.float32)
    new_prob[redraw] = 0.9
    # Enquanto o veto por posição segura, o traço antigo não expõe.
    for _ in range(occlusion_mod._MARKS_HOLD_MAX - 2):
        assessment = guard.observe(
            edited, visible, ink_prob=new_prob, ink_support_threshold=0.7
        )
    assert not np.asarray(assessment.paper_exposed)[stroke].any()
    # Passado o prazo, a edição é aceita e o papel reaparece.
    for _ in range(4):
        assessment = guard.observe(
            edited, visible, ink_prob=new_prob, ink_support_threshold=0.7
        )
    assert np.asarray(assessment.paper_exposed)[stroke].any()


def test_short_l_shaped_shadow_corner_cannot_seed_fill_recovery():
    """O L curto fecha a margem de densidade: 623 px são 27,0% do
    raster 48x48 mas 28,2% do retângulo de centros 47x47 — a fração
    sozinha não decide; a forma (aberto e fino, sem buraco nem
    núcleo espesso) é o que o separa do contorno e do traço
    (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    shape = (120, 168)  # grade exata de 24px
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    visible = np.ones(shape, dtype=bool)
    for _ in range(2):
        guard.observe(paper, visible)

    shadowed = paper.copy()
    shadowed[24:96, 24:96] = 150  # sombra 3x3 tiles
    bands = np.zeros(shape, np.float32)
    bands[24:31, 24:72] = 0.95  # faixa de 7x48 no topo
    bands[24:72, 24:31] = 0.95  # faixa de 7x48 na esquerda (L)
    # Os três tiles com faixas seguem qualificados; o L cobre 623 px.
    for _ in range(6):
        guard.observe(shadowed, visible, ink_prob=bands, ink_support_threshold=0.7)
    assert not guard._ink_streak.any()
    assert not guard._fill_adopted.any()


def test_stalled_removal_streak_returns_cadence_to_the_idle_probe():
    """Streak parcial sob sombra estável: inspecionado sem provar
    papel, a contagem segura em 2 para sempre — depois de
    _REMOVAL_PASSES passadas sem mover, o pendurado caduca e a
    cadência volta ao probe ocioso (Devin Review do PR #144)."""
    shape = (120, 160)
    prob = np.zeros(shape, np.float32)
    worker = rr.make_ink_worker(_MutablePageClean(prob), rr.SceneClock())
    acc = worker._accumulator
    stroke_a = np.zeros(shape, bool)
    stroke_a[40:44, 40:60] = True  # apagado, coberto pela sombra
    stroke_b = np.zeros(shape, bool)
    stroke_b[80:88, 40:80] = True  # testemunha: segue lida como tinta
    canvas = np.zeros(shape, np.uint8)
    canvas[stroke_a | stroke_b] = 255
    acc.restore_confirmed(canvas)
    acc._removal_streak = np.zeros(shape, np.uint8)
    acc._removal_streak[stroke_a] = 2
    prob[stroke_b] = 0.95
    visible = np.ones(shape, bool)
    exposed = ~stroke_a  # sombra estável: inspecionado, nunca prova papel

    assert worker._removal_pending()
    for index in range(registration_mod._REMOVAL_PASSES):
        acc.add(prob, visible, valid_mask=visible, paper_exposed=exposed)
        assert worker._removal_pending() == (
            index < registration_mod._REMOVAL_PASSES - 1
        )
    # Mão fora e confirmação gasta: o intervalo volta ao probe
    # ocioso, não ao 250 ms ativo nem ao descanso cego.
    worker.set_hand_presence(False)
    worker._confirm_once = False
    assert worker._pass_interval() == worker_mod._IDLE_PROBE_SECONDS
    # A sombra sai: a evidência volta a mover o streak e a cadência
    # urgente reabre — a espera caduca, o voto não.
    exposed_all = np.ones(shape, bool)
    acc.add(prob, visible, valid_mask=visible, paper_exposed=exposed_all)
    assert worker._removal_pending()


def test_pending_removal_uses_idle_probe_under_slow_ink_interval():
    """--ink-interval 5 não atrasa o voto de remoção pendente: a
    cadência urgente fica no teto do probe ocioso de 1 s (Devin
    Review do PR #144)."""
    shape = (120, 160)
    prob = np.zeros(shape, np.float32)
    worker = rr.make_ink_worker(
        _MutablePageClean(prob),
        rr.SceneClock(),
        config=InkConfig(pass_interval_s=5.0),
    )
    acc = worker._accumulator
    canvas = np.zeros(shape, np.uint8)
    canvas[40:44, 40:60] = 255
    acc.restore_confirmed(canvas)
    acc._removal_streak = np.zeros(shape, np.uint8)
    acc._removal_streak[40:44, 40:60] = 2
    worker.set_hand_presence(False)
    worker._confirm_once = False
    assert worker._pass_interval() == worker_mod._IDLE_PROBE_SECONDS


def test_stalled_removal_still_publishes_after_occluder_leaves():
    """Voto caducado segue sendo trabalho: a trabalhadora não pode
    entrar em descanso total enquanto o streak existir — a sonda
    ociosa continua rodando e, quando a sombra sai sozinha (frames
    limpos, mesma mão e geometria), a contagem destrava e a remoção
    publica (Devin Review do PR #144)."""
    shape = (120, 160)
    prob = np.zeros(shape, np.float32)
    clock = rr.SceneClock()
    page_clean = _MutablePageClean(prob)
    worker = rr.make_ink_worker(page_clean, clock)
    acc = worker._accumulator
    generation = worker.new_generation()
    stroke_a = np.zeros(shape, bool)
    stroke_a[40:44, 40:60] = True  # apagado, coberto pela sombra
    stroke_b = np.zeros(shape, bool)
    stroke_b[80:88, 40:80] = True  # testemunha: segue lida como tinta
    canvas = np.zeros(shape, np.uint8)
    canvas[stroke_a | stroke_b] = 255
    acc.restore_confirmed(canvas)
    acc._removal_streak = np.zeros(shape, np.uint8)
    acc._removal_streak[stroke_a] = 2
    prob[stroke_b] = 0.95
    visible = np.ones(shape, bool)
    exposed = ~stroke_a  # sombra estável: inspecionado, nunca prova papel

    for _ in range(registration_mod._REMOVAL_PASSES):
        acc.add(prob, visible, valid_mask=visible, paper_exposed=exposed)
    assert not worker._removal_pending()  # a espera caducou...
    worker.set_hand_presence(False)
    worker._confirm_once = False
    worker._generation_passes = registration_mod._REMOVAL_PASSES
    worker._quiet_passes = worker_mod._REST_QUIET_PASSES
    # ...mas o voto existe: descanso total proibido, cadência é a sonda.
    assert not worker._at_rest()
    assert worker._pass_interval() == worker_mod._IDLE_PROBE_SECONDS

    # A sombra sai sozinha: frames limpos chegam pela sonda e a
    # remoção publica por pixel — stroke_a some, a testemunha fica.
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    paper[stroke_b] = 60
    corners = rr.base_quad(shape)
    for _ in range(registration_mod._REMOVAL_PASSES * 3):
        clock.now += 1
        worker.submit(WarpedPage(paper, generation, corners))
        worker.pump()
    assert not acc._confirmed[stroke_a].any()
    assert acc._confirmed[stroke_b].all()


def test_stalled_removal_under_still_hand_uses_idle_probe():
    """Voto caducado com a mão parada sobre a página: o descanso
    total segue proibido (o voto existe) e a cadência é a sonda
    ociosa, não o ativo de 250 ms (Devin Review do PR #144)."""
    shape = (120, 160)
    prob = np.zeros(shape, np.float32)
    worker = rr.make_ink_worker(_MutablePageClean(prob), rr.SceneClock())
    acc = worker._accumulator
    canvas = np.zeros(shape, np.uint8)
    canvas[40:44, 40:60] = 255
    acc.restore_confirmed(canvas)
    acc._removal_streak = np.zeros(shape, np.uint8)
    acc._removal_streak[40:44, 40:60] = 2
    acc._removal_stale_passes = registration_mod._REMOVAL_PASSES

    worker.set_hand_presence(True)
    worker._hand_motion_at = -worker_mod._HAND_REST_STILL_SECONDS
    worker._confirm_once = False
    worker._generation_passes = registration_mod._REMOVAL_PASSES
    worker._quiet_passes = worker_mod._REST_QUIET_PASSES

    assert not worker._at_rest()
    assert worker._pass_interval() == worker_mod._IDLE_PROBE_SECONDS


def test_open_zigzag_seeds_fill_recovery():
    """Zigzag aberto de três bandas ligadas nas pontas alternadas:
    passa em tamanho e densidade mas não é sólido nem fecha buraco
    (~37–46% do hull). Três bandas paralelas disjuntas são a
    evidência de traço multi-volta que a borda de sombra não
    produz — a semente aceita pela terceira via sem reabrir o L
    (Devin Review do PR #144)."""
    guard = occlusion_mod.PaperExposureGuard()
    shape = (120, 168)  # grade exata de 24px
    paper = np.full((*shape, 3), 220, dtype=np.uint8)
    visible = np.ones(shape, dtype=bool)
    for _ in range(2):
        guard.observe(paper, visible)

    shadowed = paper.copy()
    shadowed[24:96, 24:96] = 150  # sombra 3x3 tiles
    bands = np.zeros(shape, np.float32)
    bands[24:31, 24:96] = 0.95  # faixa 1, 7 px
    bands[48:55, 24:96] = 0.95  # faixa 2
    bands[72:79, 24:96] = 0.95  # faixa 3
    bands[24:55, 89:96] = 0.95  # liga 1->2 pela direita
    bands[48:79, 24:31] = 0.95  # liga 2->3 pela esquerda
    for _ in range(6):
        guard.observe(shadowed, visible, ink_prob=bands, ink_support_threshold=0.7)
    assert guard._ink_streak.any()
    assert guard._fill_adopted.any()


def test_wrong_shape_paper_exposed_blocks_votes_while_omitted_permits():
    """Máscara fotométrica fora da grade do warp é bug do chamador:
    congela para False em vez de reabrir o contrato sem prova. Com a
    testemunha viva, a malformada zera os votos de remoção; a omitida
    (None, a abstenção legada) deixa o dreno andar — mesmo eixo do
    teste do visible malformado acima (Devin Review do PR #144)."""
    prob = ink()
    erased = prob.copy()
    erased[65:75, 65:85] = 0  # o bloco pequeno some; o grande testemunha
    visible = np.ones(prob.shape, bool)
    bad = np.ones((10, 10), bool)

    frozen = accumulator_mod.LayeredInkAccumulator()
    original = confirmed(frozen, prob)
    removed_px = 0
    for _ in range(registration_mod._REMOVAL_PASSES + 4):
        result = frozen.add(erased, visible, paper_exposed=bad)
        removed_px += result["removed_confirmed_px"]
    assert removed_px == 0
    assert not frozen._removal_streak.any()
    np.testing.assert_array_equal(result["confirmed"], original)

    legacy = accumulator_mod.LayeredInkAccumulator()
    confirmed(legacy, prob)
    for _ in range(registration_mod._REMOVAL_PASSES - 1):
        result = legacy.add(erased, visible, paper_exposed=None)
        np.testing.assert_array_equal(result["confirmed"], original)
    result = legacy.add(erased, visible, paper_exposed=None)
    assert result["removed_confirmed_px"] == 200
