import cv2
import numpy as np

from draw.ink import occlusion


def test_four_by_four_raster_blob_meets_four_pixel_seed_extent():
    region = np.zeros((12, 12), dtype=bool)
    region[4:8, 4:8] = True

    assert occlusion._has_dense_pixel_seed(region)


def test_seed_rejects_blob_smaller_than_four_pixel_extent():
    region = np.zeros((12, 12), dtype=bool)
    region[2:5, 2:10] = True

    assert not occlusion._has_dense_pixel_seed(region)


def test_seed_rejects_hollow_l_shape():
    region = np.zeros((40, 40), dtype=bool)
    region[4:8, 4:34] = True
    region[4:34, 4:8] = True

    assert not occlusion._has_dense_pixel_seed(region)


def test_seed_accepts_open_zigzag_parallel_bands():
    """O traço aberto do review: três bandas de 7 px ligadas nas
    pontas alternadas — nem sólido nem fechado, mas a sombra não
    faz faixas paralelas disjuntas (Devin Review do PR #144)."""
    region = np.zeros((72, 72), dtype=bool)
    for row in (0, 24, 48):
        region[row : row + 7, :] = True
    region[0:31, 65:72] = True  # liga 1->2 pela direita
    region[24:55, 0:7] = True  # liga 2->3 pela esquerda

    assert occlusion._has_dense_pixel_seed(region)


def test_seed_rejects_thin_open_arc():
    """O arco fino aberto segue ambíguo com a borda curva de uma
    sombra: não é sólido, não fecha buraco e não tem banda
    paralela irmã — continua vetado."""
    region = np.zeros((48, 48), dtype=np.uint8)
    cv2.ellipse(region, (24, 24), (20, 20), 0, -90, 45, 1, thickness=3)

    assert not occlusion._has_dense_pixel_seed(region.astype(bool))


def test_seed_accepts_rotated_solid_stroke():
    region = np.zeros((32, 32), dtype=np.uint8)
    cv2.line(region, (4, 27), (27, 4), 1, thickness=5)

    assert occlusion._has_dense_pixel_seed(region.astype(bool))


def test_guard_recovers_rectangular_outline_inside_stable_shadow():
    paper = np.full((120, 120, 3), 200, dtype=np.uint8)
    shadow = paper.copy()
    shadow[24:96, 24:96] = 40  # stable 3x3 tile shadow
    ink_prob = np.zeros((120, 120), dtype=np.float32)
    ink_prob[24:72, 24:72] = 0.95
    ink_prob[27:69, 28:68] = 0.0  # centered 40x42 hole in a 48x48 outline

    guard = occlusion.PaperExposureGuard()
    guard.observe(paper)
    assessment = None
    for _ in range(occlusion._FILL_RECOVERY_PASSES):
        assessment = guard.observe(shadow, ink_prob=ink_prob)

    assert assessment is not None and assessment.visible is not None
    assert np.all(assessment.visible[ink_prob >= 0.95])
    assert not np.any(assessment.paper_exposed[24:72, 24:72])
