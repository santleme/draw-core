"""Memória de arestas: a folha nova não herda reta da folha anterior.

O caminho de armazenamento copiava a linha ANTIGA para a aresta que
não pôde confiar neste quadro; no quadro seguinte (mesma geometria,
IoU da máscara nova com ela mesma = 1) a linha velha virava
"histórico contínuo" e puxava os cantos da folha nova para a folha
anterior. Aresta sem medida confiável e sem herança válida grava
ausente.
"""

import numpy as np

from draw import perception
from draw.perception import _geometry


def _detector():
    detector = perception.NeuralSheetDetector.__new__(perception.NeuralSheetDetector)
    detector._edge_lines = None
    detector._memory_mask = None
    detector._memory_geometry = None
    return detector


def _mask_with_rect(left, right, shape=(80, 80)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:75, left:right] = 255
    return mask


def test_untrusted_edge_does_not_carry_the_previous_sheet_line(monkeypatch):
    detector = _detector()
    lines_a = [object(), object(), object(), object()]
    lines_b = [object(), object(), object(), object()]

    # Quadro 1 (folha A): as quatro arestas medidas com confiança.
    monkeypatch.setattr(
        _geometry,
        "_fit_quad_impl",
        lambda mask, frame, fallback, sx, sy, **kw: (None, lines_a, [True] * 4),
    )
    detector._fit(_mask_with_rect(5, 35))
    assert detector._edge_lines == lines_a

    # Quadro 2 (folha B, IoU com A < 70% → sem herança válida): uma
    # aresta não confiável NÃO pode gravar a linha de A.
    monkeypatch.setattr(
        _geometry,
        "_fit_quad_impl",
        lambda mask, frame, fallback, sx, sy, **kw: (
            None,
            lines_b,
            [True, True, True, False],
        ),
    )
    detector._fit(_mask_with_rect(45, 75))
    assert detector._edge_lines[0] is lines_b[0]
    assert detector._edge_lines[3] is not lines_a[3]
    assert detector._edge_lines[3] is None
