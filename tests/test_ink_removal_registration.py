import cv2
import numpy as np
import pytest

import rr
from draw.contracts import WarpedPage
from draw.ink import accumulator as accumulator_mod
from draw.ink import occlusion as occlusion_mod
from draw.ink import registration as registration_mod
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


def test_removal_off_is_byte_identical_and_monotone():
    acc = accumulator_mod.LayeredInkAccumulator(config=InkConfig(removal=False))
    original = confirmed(acc, ink())
    for _ in range(20):
        result = acc.add(np.zeros_like(ink()), np.ones(ink().shape, bool))
    np.testing.assert_array_equal(original, result["confirmed"])
    assert "removed_confirmed_px" not in result


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


def test_live_telemetry_records_arbitration_config(monkeypatch):
    prob = ink()
    worker = drive(
        [(prob, rr.base_quad(prob.shape))] * 8,
        config=InkConfig(anchor_arbitration=False),
    )
    record = worker.telemetry_snapshot()["records"][-1]
    assert record["anchor_arbitration_enabled"] is False
    assert record["sequence"] == 8 and record["pass_gap_s"] == 1
    assert record["registration_is_identity"]
    assert "new_confirmed_px" in record and "removed_confirmed_px" in record


def test_worker_publishes_erasure_through_the_real_path():
    prob = ink()
    page = np.full((*prob.shape, 3), 220, dtype=np.uint8)
    erased = prob.copy()
    erased[65:75, 65:85] = 0

    class _MutablePageClean:
        threshold = 0.7

        def __init__(self):
            self.prob = prob

        def ink_prob(self, _image):
            return self.prob

    clock = rr.SceneClock()
    page_clean = _MutablePageClean()
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

    assert guard.observe(baseline, visible).visible.all()
    assert guard.observe(baseline, visible).visible.all()
    shadow = baseline.copy()
    shadow[24:96, 24:96] = 75
    assessment = guard.observe(shadow, visible)  # visible=True simulates a miss

    assert not assessment.paper_exposed[50:65, 50:65].any()
    assert not assessment.visible[50:65, 50:65].any()
    assert assessment.paper_exposed[0:12, 0:12].all()
    assert guard.observe(baseline, visible).visible.all()

    first_frame_guard = occlusion_mod.PaperExposureGuard()
    first_frame_hand = baseline.copy()
    first_frame_hand[24:96, 24:96] = 75
    first_frame_exposed = first_frame_guard.observe(first_frame_hand, visible)
    assert not first_frame_exposed.paper_exposed[30:60, 30:60].any()
    assert not first_frame_exposed.visible[30:60, 30:60].any()

    color_guard = occlusion_mod.PaperExposureGuard()
    assert color_guard.observe(baseline, visible).visible.all()
    colored_object = baseline.copy()
    colored_object[24:96, 24:96] = (255, 200, 190)  # quase mesma luminância
    color_exposed = color_guard.observe(colored_object, visible)
    assert not color_exposed.visible[30:60, 30:60].any()


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

    class _MutablePageClean:
        threshold = 0.7

        def __init__(self):
            self.prob = old

        def ink_prob(self, _image):
            return self.prob

    clock = rr.SceneClock()
    page_clean = _MutablePageClean()
    worker = rr.make_ink_worker(page_clean, clock, config=InkConfig(removal=True))
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
