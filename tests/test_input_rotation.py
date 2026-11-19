"""Rotacao de entrada (webcam invertida na mesa): gira na captura."""

import sys
import time

import cv2
import numpy as np
import pytest

from draw import app as app_module
from draw import camera as camera_module
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


def test_captured_at_is_stamped_before_the_rotation(monkeypatch):
    """O HUD de latencia mede da leitura em diante: o tempo gasto
    girando o quadro entra na latencia, não some dela (review #49)."""
    real = cv2.rotate

    def slow_rotate(frame, code):
        time.sleep(0.2)
        return real(frame, code)

    monkeypatch.setattr(camera_module.cv2, "rotate", slow_rotate)
    source = LatestFrameCapture(_OneFrame(), rotation=180).start()
    try:
        packet = source.wait_next(0, timeout=3.0)
        received = time.monotonic()
    finally:
        source.stop()
    assert packet is not None
    assert received - packet.captured_at >= 0.15


def test_camera_switch_keeps_the_rotation(monkeypatch, tmp_path):
    """A troca de câmera abre a nova com a mesma rotação: sem isso, o
    quadro da câmera trocada voltaria de cabeça para baixo (review #49).
    Exercita o start_switch real dentro do run() headless."""
    from tests.test_app import _frame, _PacedStallFeed, _patch_marker_run

    calls = []

    def fake_open(target, helper_token_file=None, ready_seconds=None, rotation=0):
        calls.append((target, rotation))
        raise RuntimeError("fim do teste")

    feed = _PacedStallFeed([_frame(), _frame()])
    _patch_marker_run(monkeypatch, feed, [ord("q")])
    monkeypatch.setattr(app_module, "_open_ready_source", fake_open)
    monkeypatch.setitem(app_module._camera_menu, "request", None)
    base_wait = app_module.cv2.waitKey
    state = {"n": 0}

    def wait(ms):
        state["n"] += 1
        if state["n"] == 1:
            app_module._camera_menu["request"] = 1
        elif state["n"] == 3:
            deadline = time.monotonic() + 3.0
            while not calls and time.monotonic() < deadline:
                time.sleep(0.01)
        return base_wait(ms)

    monkeypatch.setattr(app_module.cv2, "waitKey", wait)
    app_module.run(source=0, save_dir=str(tmp_path), markers_mode=True, rotation=180)
    assert calls == [(1, 180)]
