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


def test_warp_sheet_uvmap_identity_map_reproduz_frame():
    frame = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
    yy, xx = np.mgrid[0:60, 0:80]
    uv = np.stack([xx / 79.0, yy / 59.0]).astype(np.float32)
    out = paper.warp_sheet_uvmap(
        frame, uv, destination=[(0, 0), (80, 0), (80, 60), (0, 60)]
    )
    # Mesmo contrato do warp_sheet: grade do papel, mapa no destino.
    assert out.shape == (config.PAPER_H, config.PAPER_W, 3)
    np.testing.assert_allclose(
        out[2:58, 2:78].astype(np.int32), frame[2:58, 2:78].astype(np.int32), atol=1
    )
    assert not out[60:, :].any() and not out[:, 80:].any()


def test_warp_sheet_uvmap_sem_mapa_cai_na_homografia():
    frame = np.arange(200 * 300 * 3, dtype=np.uint8).reshape(200, 300, 3)
    quad = np.float32([[10, 10], [290, 10], [290, 190], [10, 190]])
    np.testing.assert_array_equal(
        paper.warp_sheet_uvmap(frame, None, corners=quad),
        paper.warp_sheet(frame, quad),
    )


def test_warp_sheet_uvmap_mapa_quase_vazio_faz_fallback():
    frame = np.zeros((60, 80, 3), np.uint8)
    uv = np.full((2, 60, 80), np.nan, np.float32)
    uv[:, :10, :10] = 0.5  # 1,4% válido: abaixo do piso de 20%
    quad = np.float32([[0, 0], [79, 0], [79, 59], [0, 59]])
    np.testing.assert_array_equal(
        paper.warp_sheet_uvmap(
            frame,
            uv,
            corners=quad,
            destination=[(0, 0), (80, 0), (80, 60), (0, 60)],
        ),
        paper.warp_sheet(frame, quad, destination=[(0, 0), (80, 0), (80, 60), (0, 60)]),
    )


_QUAD = np.float32([[10, 10], [290, 10], [290, 190], [10, 190]])


def _frame_claro():
    return np.full((200, 300, 3), 255, np.uint8)


def _uv_identidade(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx / (w - 1.0), yy / (h - 1.0)]).astype(np.float32)


def test_warp_sheet_uvmap_mapa_fora_de_0_1_cai_na_homografia():
    """Coordenada finita fora de [0,1] não é ponto válido (review #46)."""
    frame = _frame_claro()
    uv = np.full((2, 60, 80), 1.7, np.float32)
    np.testing.assert_array_equal(
        paper.warp_sheet_uvmap(frame, uv, corners=_QUAD),
        paper.warp_sheet(frame, _QUAD),
    )


def test_warp_sheet_uvmap_mapa_de_area_zero_cai_na_homografia():
    frame = _frame_claro()
    for uv in (np.zeros((2, 0, 0), np.float32), np.zeros((2, 0, 5), np.float32)):
        np.testing.assert_array_equal(
            paper.warp_sheet_uvmap(frame, uv, corners=_QUAD),
            paper.warp_sheet(frame, _QUAD),
        )


def test_warp_sheet_uvmap_resize_nao_espalha_buraco():
    """Um NaN no mapa de meia resolução vira buraco local, não uma
    mancha: as amostras válidas vizinhas continuam válidas."""
    frame = _frame_claro()
    uv = _uv_identidade(30, 40)
    uv[:, 15, 20] = np.nan
    dest = [(0, 0), (80, 0), (80, 60), (0, 60)]
    out = paper.warp_sheet_uvmap(frame, uv, destination=dest)
    region = out[:60, :80].min(axis=2)
    black = int(np.count_nonzero(region == 0))
    assert black <= 4, black  # interpolar NaN direto apagava ~16 px
    assert region[30, 30] == 255 and region[28, 44] == 255


def test_warp_sheet_uvmap_cobertura_medida_na_grade_de_saida():
    """Mapa que só parece cheio na resolução de origem: medido depois do
    resize, cai abaixo do piso e vai para a homografia."""
    frame = _frame_claro()
    uv = _uv_identidade(4, 4)
    uv[:, ::2, :] = np.nan  # linhas alternadas: suporte < 50% quase todo
    uv[:, :, ::2] = np.nan
    dest = [(0, 0), (300, 0), (300, 200), (0, 200)]
    np.testing.assert_array_equal(
        paper.warp_sheet_uvmap(frame, uv, corners=_QUAD, destination=dest),
        paper.warp_sheet(frame, _QUAD, destination=dest),
    )


def test_warp_sheet_uvmap_cobertura_conta_so_o_que_cai_na_grade():
    """Review #46: destino que passa da grade — as amostras válidas
    descartadas pelo recorte não podem segurar o caminho denso (a
    saída sairia toda preta)."""
    frame = _frame_claro()
    w, h = config.PAPER_W, config.PAPER_H
    # Destino com o dobro da largura: a metade direita cai fora da grade.
    dest = [(0, 0), (2 * w, 0), (2 * w, h), (0, h)]
    uv = _uv_identidade(h, 2 * w)
    uv[:, :, :w] = np.nan  # válido só na metade que o recorte descarta
    np.testing.assert_array_equal(
        paper.warp_sheet_uvmap(frame, uv, corners=_QUAD, destination=dest),
        paper.warp_sheet(frame, _QUAD, destination=dest),
    )


def test_warp_sheet_uvmap_destino_todo_fora_da_grade_cai_na_homografia():
    frame = _frame_claro()
    w, h = config.PAPER_W, config.PAPER_H
    dest = [(w + 10, 0), (2 * w, 0), (2 * w, h), (w + 10, h)]
    uv = _uv_identidade(h, w - 10)
    np.testing.assert_array_equal(
        paper.warp_sheet_uvmap(frame, uv, corners=_QUAD, destination=dest),
        paper.warp_sheet(frame, _QUAD, destination=dest),
    )


def test_warp_sheet_uvmap_tamanho_igual_nos_dois_caminhos():
    """80x60 de destino: denso e fallback devolvem a mesma grade."""
    frame = _frame_claro()
    dest = [(0, 0), (80, 0), (80, 60), (0, 60)]
    dense = paper.warp_sheet_uvmap(frame, _uv_identidade(60, 80), destination=dest)
    fallback = paper.warp_sheet_uvmap(frame, None, corners=_QUAD, destination=dest)
    assert dense.shape == fallback.shape == paper.warp_sheet(frame, _QUAD).shape


def test_warp_sheet_uvmap_sem_mapa_nem_cantos_devolve_none():
    assert paper.warp_sheet_uvmap(np.zeros((4, 4, 3), np.uint8), None) is None


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
