"""Testes de orientacao automatica (180°, retrato/paisagem) e caderno de 2 paginas."""

import numpy as np

from draw import geometry
from draw.auto_orient import (
    AutoPageOrientation,
    AutoRotationDetector,
    hand_mask_orientation_vote,
    quad_aspect_ratio,
)
from draw.camera import LatestFrameCapture


def test_fit_quad_unites_two_page_open_notebook_split_by_spine():
    """Caderno aberto com lombada dividindo a mascara em 2 paginas adjacentes
    deve produzir um unico quadrilatero cobrindo as duas paginas."""
    mask = np.zeros((400, 600), dtype=np.uint8)
    # Pagina esquerda e pagina direita separadas por uma lombada escura de 16 px.
    mask[80:320, 70:292] = 255
    mask[80:320, 308:530] = 255

    quad = geometry.fit_quad(mask)
    assert quad is not None
    xs = quad[:, 0]
    ys = quad[:, 1]
    assert xs.min() <= 80.0
    assert xs.max() >= 520.0
    assert ys.min() <= 90.0
    assert ys.max() >= 310.0


def test_fit_quad_ignores_small_stray_object_next_to_single_page():
    """Um objeto claro pequeno ao lado de uma folha unica NAO pode ser unido."""
    mask = np.zeros((400, 600), dtype=np.uint8)
    mask[60:340, 120:320] = 255  # folha principal (280 x 200 = 56000 px)
    mask[140:200, 340:390] = 255  # post-it pequeno (60 x 50 = 3000 px)

    quad = geometry.fit_quad(mask)
    assert quad is not None
    assert quad[:, 0].max() <= 330.0


def test_hand_mask_orientation_vote_detects_top_vs_bottom_entry():
    top_entry = np.zeros((200, 300), dtype=np.uint8)
    top_entry[0:90, 110:190] = 255
    assert hand_mask_orientation_vote(top_entry) == "inverted"

    bottom_entry = np.zeros((200, 300), dtype=np.uint8)
    bottom_entry[110:200, 110:190] = 255
    assert hand_mask_orientation_vote(bottom_entry) == "upright"

    floating_center = np.zeros((200, 300), dtype=np.uint8)
    floating_center[70:130, 110:190] = 255
    assert hand_mask_orientation_vote(floating_center) is None


def test_auto_rotation_detector_confirms_once_and_does_not_oscillate():
    det = AutoRotationDetector(confirm_votes=3)
    assert det.observe_vote("inverted") is False
    assert det.observe_vote("inverted") is False
    assert det.observe_vote("upright") is False  # reseta sequencia
    assert det.observe_vote("inverted") is False
    assert det.observe_vote(None) is False  # ambiguo nao reseta nem conta
    assert det.observe_vote("inverted") is False
    assert det.observe_vote("inverted") is True  # 3 votos seguidos -> flip!
    assert det.flipped is True
    # Nao oscila mais depois de virar.
    for _ in range(5):
        assert det.observe_vote("inverted") is False


def test_auto_page_orientation_hysteresis_portrait_and_landscape():
    orient = AutoPageOrientation(initial="retrato", confirm_frames=3)
    wide = np.array([[0, 0], [400, 0], [400, 250], [0, 250]], dtype=np.float32)
    tall = np.array([[0, 0], [250, 0], [250, 400], [0, 400]], dtype=np.float32)
    square = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float32)

    assert quad_aspect_ratio(wide) > 1.5
    assert orient.observe(square) is None
    assert orient.observe(wide) is None
    assert orient.observe(wide) is None
    assert orient.observe(wide) == "paisagem"
    assert orient.current == "paisagem"

    # Zona morta nao volta para retrato.
    for _ in range(5):
        assert orient.observe(square) is None
    assert orient.current == "paisagem"

    # Folha em pe sustentada volta para retrato.
    assert orient.observe(tall) is None
    assert orient.observe(tall) is None
    assert orient.observe(tall) == "retrato"
    assert orient.current == "retrato"


def test_latest_frame_capture_set_rotation_flips_live():
    class _FixedCam:
        def __init__(self):
            self.frame = np.full((40, 60, 3), 240, dtype=np.uint8)
            self.frame[2:8, 2:8] = 10  # canto superior esquerdo

        def read(self):
            return True, self.frame.copy()

        def release(self):
            pass

    src = LatestFrameCapture(_FixedCam(), rotation=0).start()
    try:
        p1 = src.wait_next(0, timeout=2.0)
        assert p1 is not None
        assert p1.image[2:8, 2:8].mean() < 50
        src.set_rotation(180)
        assert src.rotation == 180
        # Pega um quadro posterior a troca de rotacao.
        seq = p1.sequence
        flipped = False
        for _ in range(20):
            p = src.wait_next(seq, timeout=1.0)
            assert p is not None
            seq = p.sequence
            if p.image[32:38, 52:58].mean() < 50:
                flipped = True
                break
        assert flipped
    finally:
        src.stop()
