"""Rotacao de entrada (webcam invertida na mesa): gira na captura."""

import sys

import numpy as np
import pytest

from draw import app as app_module
from draw.camera import INPUT_ROTATIONS, LatestFrameCapture


class _OneFrame:
    """Fonte que entrega sempre o mesmo quadro com uma marca no canto."""

    def __init__(self):
        self.frame = np.full((48, 64, 3), 230, np.uint8)
        self.frame[2:8, 3:9] = 20  # marca no canto superior esquerdo

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


def _first_packet(rotation):
    source = LatestFrameCapture(_OneFrame(), rotation=rotation).start()
    try:
        packet = source.wait_next(0, timeout=2.0)
    finally:
        source.stop()
    assert packet is not None
    return packet.image


def test_rotate_180_moves_mark_to_opposite_corner():
    img = _first_packet(180)
    assert img.shape == (48, 64, 3)
    assert img[40:46, 55:61].mean() < 60  # canto inferior direito
    assert img[2:8, 3:9].mean() > 200


def test_rotation_zero_is_untouched():
    img = _first_packet(0)
    assert img[2:8, 3:9].mean() < 60
    assert img[40:46, 55:61].mean() > 200


@pytest.mark.parametrize("bad", [90, 270, -180, 1])
def test_unsupported_rotation_is_refused(bad):
    with pytest.raises(ValueError, match="rotacao"):
        LatestFrameCapture(_OneFrame(), rotation=bad)


def test_main_rotate_flag_reaches_run(monkeypatch):
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    for argv, expected in ((["draw"], 0), (["draw", "--rotate", "180"], 180)):
        monkeypatch.setattr(sys, "argv", argv)
        app_module.main()
        assert seen[-1]["rotation"] == expected
    assert INPUT_ROTATIONS == (0, 180)
