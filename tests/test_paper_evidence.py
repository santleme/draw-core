"""Guarda fotometrica de oclusao (issue #129)."""

import numpy as np

from draw.ink import accumulator as accumulator_mod
from draw.ink import paper_evidence
from draw.ink import registration as registration_mod
from draw.ink.config import InkConfig

H, W = 160, 240
PAPER = (200, 205, 210)  # BGR, quase neutro


def _page():
    image = np.empty((H, W, 3), np.uint8)
    image[:] = PAPER
    return image


def _ink_prob():
    prob = np.zeros((H, W), np.float32)
    prob[10:60, 10:120] = 0.95  # testemunha grande
    prob[100:130, 150:210] = 0.95  # regiao que a mao vai cobrir
    return prob


def _confirm(acc, prob, paper_ok=None):
    for _ in range(16):
        result = acc.add(prob, np.ones(prob.shape, bool), paper_ok=paper_ok)
    return result["confirmed"].copy()


def test_clean_paper_is_exposed():
    image = _page()
    ref = paper_evidence.paper_reference(image)
    assert ref is not None
    assert paper_evidence.paper_exposed(image, ref).all()


def test_thin_pen_strokes_stay_paper():
    image = _page()
    image[::6, :, :] = 40  # traco fino a cada 6 linhas
    ref = paper_evidence.paper_reference(_page())
    assert paper_evidence.paper_exposed(image, ref).mean() > 0.95


def test_dark_block_and_skin_are_not_paper():
    image = np.empty((256, 384, 3), np.uint8)
    image[:] = PAPER
    ref = paper_evidence.paper_reference(image)
    image[96:224, 192:352] = (60, 62, 64)  # sombra/mao escura neutra
    image[0:96, 0:128] = (110, 140, 190)  # pele clara: B/R bem baixo
    ok = paper_evidence.paper_exposed(image, ref)
    assert not ok[112:208, 208:336].any()
    assert not ok[8:88, 8:120].any()
    assert ok[0:64, 200:380].all()
    assert ok[160:250, 0:100].all()


def test_uncertain_reference_freezes_everything():
    image = _page()
    nothing_visible = np.zeros((H, W), bool)
    assert paper_evidence.paper_reference(image, nothing_visible) is None
    assert not paper_evidence.paper_exposed(image, None).any()


def test_hand_over_ink_never_erases_confirmed_stroke():
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = _ink_prob()
    original = _confirm(acc, prob)
    # Mao nao mascarada (visible True) sobre o traco 2: o modelo ve prob 0
    # ali, a testemunha (traco 1) segue forte — o caminho que apagava.
    covered = prob.copy()
    covered[95:135, 145:215] = 0
    paper_ok = np.ones((H, W), bool)
    paper_ok[95:135, 145:215] = False
    for _ in range(registration_mod._REMOVAL_PASSES * 3):
        result = acc.add(covered, np.ones((H, W), bool), paper_ok=paper_ok)
    np.testing.assert_array_equal(result["confirmed"], original)
    # Contraprova: sem a guarda o mesmo cenario apaga.
    bare = accumulator_mod.LayeredInkAccumulator()
    _confirm(bare, prob)
    for _ in range(registration_mod._REMOVAL_PASSES + 1):
        erased = bare.add(covered, np.ones((H, W), bool))
    assert erased["confirmed"][100:130, 150:210].sum() == 0


def test_hand_pixels_are_never_added_as_ink():
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = _ink_prob()
    original = _confirm(acc, prob)
    noisy = prob.copy()
    noisy[70:90, 20:80] = 0.95  # borda/textura da mao lida como tinta
    paper_ok = np.ones((H, W), bool)
    paper_ok[65:95, 15:85] = False
    for _ in range(20):
        result = acc.add(noisy, np.ones((H, W), bool), paper_ok=paper_ok)
    np.testing.assert_array_equal(result["confirmed"], original)
    assert not result["provisional"][70:90, 20:80].any()


def test_hold_covers_short_detection_gaps_then_releases():
    config = InkConfig(occlusion_hold_passes=3)
    acc = accumulator_mod.LayeredInkAccumulator(config=config)
    prob = _ink_prob()
    original = _confirm(acc, prob)
    covered = prob.copy()
    covered[95:135, 145:215] = 0
    ok_all = np.ones((H, W), bool)
    blocked = ok_all.copy()
    blocked[95:135, 145:215] = False
    acc.add(covered, ok_all, paper_ok=blocked)
    # Evidencia some (queda de deteccao) mas o prob segue zero: segura.
    for _ in range(3):
        result = acc.add(covered, ok_all, paper_ok=ok_all)
        np.testing.assert_array_equal(result["confirmed"], original)
    assert acc._hold is not None


def test_new_ink_confirms_after_hand_leaves_and_old_stroke_survives():
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = _ink_prob()
    original = _confirm(acc, prob)
    ok_all = np.ones((H, W), bool)
    blocked = ok_all.copy()
    blocked[95:135, 145:215] = False
    covered = prob.copy()
    covered[95:135, 145:215] = 0
    for _ in range(8):
        acc.add(covered, ok_all, paper_ok=blocked)
    after = prob.copy()
    after[140:150, 20:60] = 0.95  # tinta nova
    for _ in range(16):
        result = acc.add(after, ok_all, paper_ok=ok_all)
    assert (result["confirmed"][140:150, 20:60] == 255).all()
    assert (result["confirmed"][original > 0] == 255).all()


def test_exposed_paper_still_erases():
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = _ink_prob()
    _confirm(acc, prob)
    erased = prob.copy()
    erased[100:130, 150:210] = 0
    ok_all = np.ones((H, W), bool)
    for _ in range(registration_mod._REMOVAL_PASSES + 1):
        result = acc.add(erased, ok_all, paper_ok=ok_all)
    assert result["confirmed"][100:130, 150:210].sum() == 0
    assert (result["confirmed"][10:60, 10:120] == 255).all()


def test_paper_ok_none_is_byte_identical():
    prob = _ink_prob()
    a = accumulator_mod.LayeredInkAccumulator()
    b = accumulator_mod.LayeredInkAccumulator()
    for _ in range(20):
        ra = a.add(prob, np.ones((H, W), bool))
        rb = b.add(prob, np.ones((H, W), bool), paper_ok=None)
    np.testing.assert_array_equal(ra["confirmed"], rb["confirmed"])
    np.testing.assert_array_equal(ra["provisional"], rb["provisional"])


def test_guard_without_visibility_does_not_invent_negative_evidence():
    acc = accumulator_mod.LayeredInkAccumulator()
    prob = _ink_prob()
    original = _confirm(acc, prob)
    erased = prob.copy()
    erased[100:130, 150:210] = 0
    ok_all = np.ones((H, W), bool)
    for _ in range(20):
        result = acc.add(erased, None, paper_ok=ok_all)
    np.testing.assert_array_equal(result["confirmed"], original)


def test_thick_blue_pen_stroke_is_not_mistaken_for_occlusion():
    image = _page()
    ref = paper_evidence.paper_reference(image)
    image[40:46, 20:200] = (170, 60, 30)  # traco grosso azul (BGR)
    image[20:140, 100:106] = (40, 40, 40)
    ok = paper_evidence.paper_exposed(image, ref)
    assert ok[30:60, 30:90].mean() > 0.9
