"""Configuração da captura local: formato, fila mínima e fps.

A fila do driver mediu profundidade 4 na mesa do Lucas
(2026-10-31): até ~200-300 ms de quadro velho quando o consumo
atrasa. `_configure_local_capture` pede buffer 1 e 30 fps ao
driver, além do MJPG 720p de sempre. Backend que não suporta as
propriedades ignora o pedido — o teste pina os pedidos, não o
efeito, que só a câmera dele mede.
"""

import cv2

from draw import camera as camera_module


class _FakeCapture:
    def __init__(self):
        self.settings = {}

    def set(self, prop, value):
        self.settings[prop] = value
        return True


def test_local_capture_requests_format_minimal_buffer_and_fps():
    capture = _FakeCapture()
    camera_module._configure_local_capture(capture)
    settings = capture.settings
    assert settings[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    assert settings[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert settings[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert settings[cv2.CAP_PROP_BUFFERSIZE] == 1
    assert settings[cv2.CAP_PROP_FPS] == 30


class _SetRaisesCapture:
    """Driver que levanta cv2.error no set() — os pedidos de formato
    nao podem derrubar a abertura nem vazar a captura emprestada."""

    def __init__(self):
        self.released = False

    def set(self, *_args):
        raise cv2.error("v4l2: VIDIOC_S_CTRL falhou no dispositivo")

    def release(self):
        self.released = True


def test_configure_local_capture_cv2_error_nao_propaga():
    capture = _SetRaisesCapture()
    camera_module._configure_local_capture(capture)  # nao levanta
    capture.release()
    assert capture.released
