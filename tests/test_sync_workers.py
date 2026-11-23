"""Testes dos adaptadores síncronos de lab/sync_workers (iter_frames)."""

import cv2
import numpy as np
from lab.sync_workers import iter_frames


def test_iter_frames_ordena_diretorio_por_sufixo_numerico(tmp_path):
    """f2.png (zeros) chega antes de f10.png (255): a ordem de nome
    os invertia e o replay/simulador consumia o quadro 10 no lugar
    do 2."""
    cv2.imwrite(str(tmp_path / "f2.png"), np.zeros((4, 4, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "f10.png"), np.full((4, 4, 3), 255, np.uint8))

    frames = list(iter_frames(tmp_path, limit=10))

    assert [i for i, _frame in frames] == [0, 1]
    assert frames[0][1].max() == 0  # f2.png primeiro
    assert frames[1][1].min() == 255  # f10.png depois
