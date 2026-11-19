"""Grade de sessao, sugestao optica e escala dos formatos em retrato."""

import cv2
import numpy as np
import pytest

from draw import config, paper


@pytest.fixture(autouse=True)
def restore_a4():
    config.apply_format("a4")
    yield
    config.apply_format("a4")


@pytest.mark.parametrize(
    "name,width",
    [
        ("a4", 850),
        ("a5", 846),
        ("a3", 849),
        ("carta", 929),
        ("caderno", 873),
    ],
)
def test_paper_grid(name, width):
    assert config.paper_grid(name) == (width, 1200)
    w_mm, h_mm = config.FORMATS[name]
    error = abs(width / 1200 - w_mm / h_mm)
    isotropy = abs(config.mm_per_px(name) / (w_mm / width) - 1)
    if name == "a4":
        # A unica excecao e a grade legada, nao uma tolerancia geral maior.
        assert error == pytest.approx(abs(850 / 1200 - 210 / 297))
        assert isotropy == pytest.approx(0.001785714285714)
    else:
        assert width == round(1200 * w_mm / h_mm)
        assert error <= 0.001
        assert isotropy <= 0.001


def test_apply_format_a4_eh_incumbente():
    original = {0: (90, 90), 1: (760, 90), 2: (760, 1110), 3: (90, 1110)}
    config.apply_format("carta")
    assert config.apply_format("a4") == (850, 1200)
    assert config.MARKER_CENTERS == original
    assert config.PAPER_FORMAT == "a4"
    frame = np.arange(200 * 300 * 3, dtype=np.uint8).reshape(200, 300, 3)
    quad = np.float32([[10, 10], [290, 10], [290, 190], [10, 190]])
    matrix = cv2.getPerspectiveTransform(
        quad, np.float32([[0, 0], [850, 0], [850, 1200], [0, 1200]])
    )
    np.testing.assert_array_equal(
        paper.warp_sheet(frame, quad),
        cv2.warpPerspective(frame, matrix, (850, 1200)),
    )


def test_apply_format_recomputa_marcadores():
    config.apply_format("carta")
    assert config.MARKER_CENTERS[1] == (929 - 90, 90)
    assert config.MARKER_CENTERS[2] == (929 - 90, 1200 - 90)
    assert config.mm_per_px() == 279 / 1200
    with pytest.raises(ValueError):
        config.apply_format("paisagem")
    assert config.PAPER_FORMAT == "carta"


def projected_quad(name="carta", pitch=25, yaw=15, roll=0):
    w, h = config.FORMATS[name]
    points = np.array(
        [[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]]
    )
    x, y, z = np.deg2rad([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    camera = points @ (rz @ ry @ rx).T + [0, 0, 1000]
    return camera[:, :2] / camera[:, 2:] * (1.2 * 1280) + [640, 360]


@pytest.mark.parametrize(
    "name,label",
    [
        ("carta", "carta"),
        ("a4", "serie_a"),
        ("a3", "serie_a"),
        ("a5", "serie_a"),
        ("caderno", "caderno"),
    ],
)
@pytest.mark.parametrize("pitch,yaw", [(25, 15), (0, 0)])
def test_estimador_sugere_classe_certa(name, label, pitch, yaw):
    result = paper.estimate_format(projected_quad(name, pitch, yaw), (720, 1280))
    expected = config.FORMATS[name][0] / config.FORMATS[name][1]
    assert result["ratio"] == pytest.approx(expected, rel=0.02)
    assert result["suggestion"] == label
    assert result["reason"] is None


def test_estimador_se_abstem_com_um_vp_no_infinito():
    result = paper.estimate_format(projected_quad(pitch=25, yaw=0), (720, 1280))
    assert result["suggestion"] is None
    assert result["reason"] == "single_vp_at_infinity"


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[1, 1]] * 4,
        [[0, 0], [1, 0], [np.nan, 1], [0, 1]],
    ],
)
def test_estimador_se_abstem_com_quad_degenerado(points):
    result = paper.estimate_format(points, (720, 1280))
    assert result["suggestion"] is None
    assert result["reason"] == "degenerate_quad"


def test_estimador_se_abstem_com_focal_invalida():
    points = projected_quad()
    points[:, 0] = 1280 - points[:, 0]
    # A mesma camera observada com principal deslocado viola as hipoteses.
    result = paper.estimate_format(points + [5000, 2000], (720, 1280))
    assert result["suggestion"] is None
    assert result["reason"] == "invalid_focal_length"


def test_estimador_ratio_ambiguo_e_invariante_a_translacao_e_escala():
    ratio = np.sqrt((210 / 297) * (200 / 275))
    quad = np.array([[0, 0], [ratio, 0], [ratio, 1], [0, 1]]) * 300
    for scale, shift in [(1, [0, 0]), (0.5, [300, 200])]:
        result = paper.estimate_format(quad * scale + shift, (720, 1280))
        assert result["reason"] == "ambiguous_ratio"


def test_sugestao_se_abstem_para_centros_de_marcadores():
    suggester = paper.FormatSuggester()
    result = None
    for _ in range(15):
        result = suggester.observe(projected_quad(), (720, 1280), sheet_boundary=False)
    assert result["suggestion"] is None
    assert result["reason"] == "marker_centers_not_sheet_boundary"


def test_paisagem_troca_a_grade_e_mantem_mm_por_px():
    """Caderno aberto/folha deitada: a grade deita, o lado longo segue
    1200 px e a escala em mm não muda."""
    assert config.paper_grid("a4", "paisagem") == (1200, 850)
    assert (
        config.paper_grid("caderno", "paisagem") == config.paper_grid("caderno")[::-1]
    )
    unit = config.mm_per_px("a4")
    assert config.apply_format("a4", "paisagem") == (1200, 850)
    assert (config.PAPER_W, config.PAPER_H) == (1200, 850)
    assert config.PAPER_ORIENTATION == "paisagem"
    assert config.mm_per_px() == unit
    assert config.paper_mm() == (297.0, 210.0)
    assert config.MARKER_CENTERS[2] == (1200 - 90, 850 - 90)
    config.apply_format("a4")
    assert config.PAPER_ORIENTATION == "retrato"
    assert config.paper_mm() == (210.0, 297.0)


def test_orientacao_desconhecida_e_recusada():
    with pytest.raises(ValueError, match="orientacao"):
        config.paper_grid("a4", "diagonal")


def test_warp_sheet_em_paisagem_sai_deitado():
    config.apply_format("a4", "paisagem")
    try:
        frame = np.zeros((600, 900, 3), np.uint8)
        quad = np.float32([[50, 100], [850, 100], [850, 500], [50, 500]])
        assert paper.warp_sheet(frame, quad).shape == (850, 1200, 3)
    finally:
        config.apply_format("a4")


def test_paper_mm_valida_como_paper_grid():
    """Mesma validação nas duas funções (review #50): orientação ou
    formato desconhecido é ValueError, não dimensões inventadas."""
    with pytest.raises(ValueError, match="orientacao"):
        config.paper_mm("a4", "diagonal")
    with pytest.raises(ValueError, match="formato"):
        config.paper_mm("a9")
    assert config.paper_mm("carta", "paisagem") == (279.0, 216.0)
