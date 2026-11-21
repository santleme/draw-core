"""Configuração central do Draw.

A folha-modelo é um A4 com quatro marcadores ArUco, um em cada canto.
Todas as coordenadas aqui estão no espaço da folha retificada
(a "vista de cima" que o pipeline produz).
"""

import cv2

FORMATS = {
    "a4": (210.0, 297.0),
    "a5": (148.0, 210.0),
    "a3": (297.0, 420.0),
    "carta": (216.0, 279.0),
    "caderno": (200.0, 275.0),
}
PAPER_FORMAT = "a4"
# Orientação da grade: "retrato" (padrão) ou "paisagem" (caderno aberto
# ou folha deitada). Paisagem troca largura e altura da grade; o lado
# longo segue com 1200 px, então mm_per_px não muda.
ORIENTATIONS = ("retrato", "paisagem")
PAPER_ORIENTATION = "retrato"

# Grade A4 legada: nao derivar os 850 px da proporcao nominal.
PAPER_W = 850
PAPER_H = 1200

# Marcadores: id -> nome do canto.
MARKER_IDS = (0, 1, 2, 3)  # 0=sup. esq., 1=sup. dir., 2=inf. dir., 3=inf. esq.
MARKER_SIZE_PX = 120

# Centros dos marcadores no espaço da folha retificada.
MARKER_CENTERS = {
    0: (90, 90),
    1: (PAPER_W - 90, 90),
    2: (PAPER_W - 90, PAPER_H - 90),
    3: (90, PAPER_H - 90),
}

# Dicionario ArUco DICT_4X4_50 (constante 0 do OpenCV). Resolvido sem
# exigir `cv2.aruco` na importacao: um ambiente em que `opencv-python`
# comum (ou mistura de wheels) substituiu o pacote `-contrib` ainda
# sobe o caminho padrao (sem --markers) sem quebrar no import.
_DICT_4X4_50_FALLBACK = 0
ARUCO_DICTIONARY = (
    cv2.aruco.DICT_4X4_50
    if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_4X4_50")
    else _DICT_4X4_50_FALLBACK
)


def paper_grid(name, orientation="retrato"):
    """Grade com borda longa de 1200 px e A4 legado pinado.

    Retrato devolve (curto, 1200); paisagem, (1200, curto).

    Limitacao herdada da grade A4 legada (850 px pinados): 850 x 0,2475
    mm/px = 210,375 mm, 0,375 mm acima dos 210 nominais. A linha de
    centro da tinta nas duas ultimas colunas (retrato) ou linhas
    (paisagem) cai ate 0,13 mm fora da folha no export vetorial; como o
    SVG centra o traco nessa linha, o traco ultrapassa ainda mais, por
    meia largura de traco (da ordem de 0,6 mm para um traco de 5 px).
    E identico nas duas orientacoes (medido em 2026-11-19), nao vem da
    paisagem; mudar a grade A4 e decisao separada, pois mexe no mm/px de
    todo o historico.
    """
    if name not in FORMATS:
        raise ValueError(f"formato de folha desconhecido: {name}")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientacao desconhecida: {orientation}")
    width_mm, height_mm = FORMATS[name]
    short = 850 if name == "a4" else round(1200 * width_mm / height_mm)
    return (short, 1200) if orientation == "retrato" else (1200, short)


def paper_mm(name=None, orientation=None):
    """(largura_mm, altura_mm) da folha na orientação da grade."""
    name = PAPER_FORMAT if name is None else name
    orientation = PAPER_ORIENTATION if orientation is None else orientation
    if name not in FORMATS:
        raise ValueError(f"formato de folha desconhecido: {name}")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientacao desconhecida: {orientation}")
    width_mm, height_mm = FORMATS[name]
    return (width_mm, height_mm) if orientation == "retrato" else (height_mm, width_mm)


def mm_per_px(name=None):
    name = PAPER_FORMAT if name is None else name
    if name not in FORMATS:
        raise ValueError(f"formato de folha desconhecido: {name}")
    return FORMATS[name][1] / 1200


def apply_format(name, orientation="retrato"):
    """Configura a grade antes de construir os consumidores da sessao."""
    global PAPER_FORMAT, PAPER_ORIENTATION, PAPER_W, PAPER_H, MARKER_CENTERS
    width, height = paper_grid(name, orientation)
    PAPER_FORMAT, PAPER_ORIENTATION = name, orientation
    PAPER_W, PAPER_H = width, height
    MARKER_CENTERS = {
        0: (90, 90),
        1: (width - 90, 90),
        2: (width - 90, height - 90),
        3: (90, height - 90),
    }
    return width, height


def get_dictionary():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "getPredefinedDictionary"):
        raise RuntimeError(
            "O OpenCV instalado nao tem o modulo cv2.aruco (necessario para "
            "--markers). Instale a roda contrib: pip install "
            "'opencv-contrib-python>=4.8,<5.1'."
        )
    return aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
