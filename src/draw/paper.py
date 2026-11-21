"""Detecção da folha e correção de perspectiva (homografia).

A webcam vê a folha inclinada. O modo padrão procura o contorno da folha
na árvore de contornos da cena e usa os quatro cantos dela para montar a
vista de cima. Um portão de textura examina o interior do candidato
antes de aceitar a folha: formato sozinho não basta. O modo com
marcadores usa os centros dos quatro ArUco da folha-modelo. Os dois
caminhos entregam a folha retificada no mesmo
espaço definido em config.py.
"""

import cv2
import numpy as np

from . import config

# A folha nem sempre fecha como quadrilátero num epsilon só. Cantos
# arredondados e a lombada de um caderno pedem aproximações mais grossas.
_SHEET_EPSILON_LADDER = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
_SHEET_MAX_CANDIDATES = 8
_SHEET_MIN_AREA_RATIO = 0.12

# Portão de textura: a geometria sozinha aceitava qualquer quadrilátero
# grande. Na auditoria de textura, 60% das cenas sem papel ganharam uma
# folha. O portão mede o interior do candidato antes de aceitar: fração
# de branco, densidade de bordas internas e variância do Laplaciano.
#
# A regra original da auditoria foi calibrada em papel pautado, onde a
# densidade de bordas internas passa de 0,03 com folga. A folha com
# tinta esparsa, caso principal do produto e das cenas deste
# repositório, mede densidade de bordas perto de 0,006, abaixo desse
# limiar, mas variância do Laplaciano entre 523 e 783. Por isso o ramo
# extra: Laplaciano a partir de 200 também afirma papel. O pior falso
# positivo medido na auditoria ficou em 27,0, longe dessa faixa.
#
# Os limiares abaixo são calibrações de laboratório, feitas sobre cenas
# sintéticas e um frame real. Eles precisam ser refeitos com as
# capturas reais da mesa do usuário.
_TEXTURE_MIN_WHITE_FRACTION = 0.5
_TEXTURE_MIN_EDGE_DENSITY = 0.03
_TEXTURE_MIN_LAPLACIAN_VARIANCE = 30.0
_TEXTURE_HIGH_LAPLACIAN_VARIANCE = 200.0
_TEXTURE_INTERIOR_SHRINK = 0.86


def _make_detector():
    dictionary = config.get_dictionary()
    aruco = getattr(cv2, "aruco", None)
    if (
        aruco is None
        or not hasattr(aruco, "DetectorParameters")
        or not hasattr(aruco, "ArucoDetector")
    ):
        raise RuntimeError(
            "O OpenCV instalado nao tem o modulo cv2.aruco (necessario para "
            "--markers). Instale a roda contrib: pip install "
            "'opencv-contrib-python>=4.8,<5.1'."
        )
    params = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, params)


# O detector ArUco só é usado no modo com marcadores, mas era criado na
# importação do módulo. cv2.aruco.ArucoDetector só existe no OpenCV 4.7
# em diante: em 4.6 a importação inteira caía, inclusive para quem só
# queria o modo sem marcadores. A criação é preguiçosa, no primeiro uso.
_DETECTOR = None


def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = _make_detector()
    return _DETECTOR


def _order_corners(points):
    """Ordena quatro pontos como TL, TR, BR e BL."""
    ordered = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = ordered.mean(axis=0)
    angles = np.arctan2(ordered[:, 1] - center[1], ordered[:, 0] - center[0])
    ordered = ordered[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def _aspect_ratio(corners):
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    width = (top + bottom) / 2.0
    height = (left + right) / 2.0
    return width / max(height, 1.0)


def estimate_format(corners, image_shape):
    """Sugere proporcao: principal no centro, pixels quadrados e skew zero."""
    result = {"ratio": None, "suggestion": None, "reason": None}
    try:
        points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    except (TypeError, ValueError):
        return {**result, "reason": "degenerate_quad"}
    if not np.isfinite(points).all() or not cv2.isContourConvex(
        points.astype(np.float32)
    ):
        return {**result, "reason": "degenerate_quad"}
    height, width = image_shape[:2]
    if width <= 0 or height <= 0:
        return {**result, "reason": "degenerate_quad"}
    centered = points - [width / 2, height / 2]
    homogeneous = np.column_stack((centered, np.ones(4)))
    tl, tr, br, bl = homogeneous
    vp_w = np.cross(np.cross(tl, tr), np.cross(bl, br))
    vp_h = np.cross(np.cross(tl, bl), np.cross(tr, br))

    def at_infinity(vp):
        length = np.hypot(vp[0], vp[1])
        return length > 0 and abs(vp[2]) / length < 1 / (10 * width)

    infinite_w, infinite_h = at_infinity(vp_w), at_infinity(vp_h)
    if infinite_w and infinite_h:
        edge_w = np.linalg.norm(points[[1, 2]] - points[[0, 3]], axis=1)
        edge_h = np.linalg.norm(points[[3, 2]] - points[[0, 1]], axis=1)
        ratio = float(edge_w.mean() / edge_h.mean())
        result["regime"] = "near_frontal"
    elif infinite_w or infinite_h:
        return {**result, "reason": "single_vp_at_infinity"}
    else:
        if vp_w[2] == 0 or vp_h[2] == 0:
            return {**result, "reason": "degenerate_quad"}
        xy_w, xy_h = vp_w[:2] / vp_w[2], vp_h[:2] / vp_h[2]
        focal_sq = -float(xy_w @ xy_h)
        if (
            not np.isfinite(focal_sq)
            or focal_sq <= 0
            or not (0.5 * width) ** 2 <= focal_sq <= (3 * width) ** 2
        ):
            return {**result, "reason": "invalid_focal_length"}
        matrix = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
            centered.astype(np.float32),
        )
        omega = np.diag([1 / focal_sq, 1 / focal_sq, 1])
        h1, h2 = matrix[:, 0], matrix[:, 1]
        ratio = float(np.sqrt((h1 @ omega @ h1) / (h2 @ omega @ h2)))
        result["regime"] = "finite_vps"
    if not np.isfinite(ratio) or ratio <= 0:
        return {**result, "reason": "degenerate_quad"}
    result["ratio"] = ratio
    classes = {"serie_a": "a4", "caderno": "caderno", "carta": "carta"}
    distances = sorted(
        (
            abs(
                np.log(ratio)
                - np.log(config.FORMATS[name][0] / config.FORMATS[name][1])
            ),
            label,
        )
        for label, name in classes.items()
    )
    (d1, suggestion), (d2, _), _ = distances
    if d1 > np.log(1.06) or d2 < 1.5 * d1:
        return {**result, "reason": "ambiguous_ratio"}
    return {**result, "suggestion": suggestion}


class FormatSuggester:
    """Uma sugestao na partida, pela mediana dos primeiros 15 quads."""

    def __init__(self):
        self._quads = []
        self._done = False

    def observe(self, corners, image_shape, sheet_boundary=True):
        if self._done:
            return None
        self._quads.append(np.asarray(corners, dtype=np.float64).copy())
        if len(self._quads) < 15:
            return None
        self._done = True
        median = np.median(np.stack(self._quads), axis=0)
        self._quads.clear()
        if not sheet_boundary:
            return {
                "ratio": None,
                "suggestion": None,
                "reason": "marker_centers_not_sheet_boundary",
            }
        return estimate_format(median, image_shape)


def _sheet_candidate(contour, scale, small_frame_area):
    perimeter = cv2.arcLength(contour, True)
    for epsilon in _SHEET_EPSILON_LADDER:
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if cv2.contourArea(approx) < small_frame_area * _SHEET_MIN_AREA_RATIO:
            continue

        corners = _order_corners(approx.reshape(4, 2))
        if not 0.4 <= _aspect_ratio(corners) <= 2.5:
            continue
        return (corners / scale).astype(np.float32)
    return None


def _refine_corners(frame, corners):
    # O refino costumava ser pulado quando qualquer canto estava a
    # menos de 11 px da borda, porque a janela do cornerSubPix sairia
    # do frame. Mas canto colado na borda é o caso comum em uso real:
    # o caderno é cortado pela borda superior do quadro. Em vez de
    # pular, o frame ganha uma margem de replicação maior que a janela
    # (10 px): a janela de cada canto fica inteira sobre a imagem
    # acolchoada e o resultado é deslocado de volta. Cantos longe da
    # borda enxergam a mesma vizinhança de antes, então o refino deles
    # não muda.
    pad = 12
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    refined = (corners + pad).reshape(-1, 1, 2).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.01,
    )
    cv2.cornerSubPix(padded, refined, (10, 10), (-1, -1), criteria)
    return (refined.reshape(-1, 2) - pad).astype(np.float32)


_TEXTURE_ERODE_KERNEL = np.ones((9, 9), np.uint8)


class _LazyTextureMaps:
    """Mapas de textura calculados sob demanda e reusados no frame.

    Evita pagar Laplacian e Canny no frame inteiro quando os candidatos
    já caem no filtro barato de fração de branco (ex.: tapete de mesa,
    caixa de papelão, monitor escuro) ou quando a variância do
    Laplaciano já decide sozinha (< 30 ou >= 200).
    """

    def __init__(self, frame):
        self._frame = frame
        self._gray = None
        self._hsv = None
        self._cache = {}

    def _ensure_gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
        return self._gray

    def _ensure_hsv(self):
        if self._hsv is None:
            self._hsv = cv2.cvtColor(self._frame, cv2.COLOR_BGR2HSV)
        return self._hsv

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        if key == "saturation":
            val = self._ensure_hsv()[:, :, 1]
        elif key == "value":
            val = self._ensure_hsv()[:, :, 2]
        elif key == "laplacian":
            val = cv2.Laplacian(self._ensure_gray(), cv2.CV_32F)
        elif key == "edges":
            val = cv2.Canny(self._ensure_gray(), 30, 100)
        else:
            raise KeyError(key)
        self._cache[key] = val
        return val


def _candidate_interior_roi(frame_shape, corners):
    """Recorta e erode a máscara do interior apenas na bounding box."""
    fh, fw = frame_shape[:2]
    center = corners.mean(axis=0)
    shrunk = (center + (corners - center) * _TEXTURE_INTERIOR_SHRINK).astype(np.int32)
    pad = 6
    x0 = max(0, int(shrunk[:, 0].min()) - pad)
    y0 = max(0, int(shrunk[:, 1].min()) - pad)
    x1 = min(fw, int(shrunk[:, 0].max()) + pad + 1)
    y1 = min(fh, int(shrunk[:, 1].max()) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local_poly = shrunk - np.array([x0, y0], dtype=np.int32)
    cv2.fillPoly(mask, [local_poly], 255)
    eroded = cv2.erode(mask, _TEXTURE_ERODE_KERNEL)
    selected = eroded > 0
    if not selected.any():
        return None
    return slice(y0, y1), slice(x0, x1), selected


def _texture_verdict(frame, corners, maps=None):
    """Veredito do portão de textura com avaliação em curto-circuito.

    1) Rejeita direto quando white_fraction < 0,5 (sem calcular
       Laplaciano nem Canny).
    2) Decide "blank_maybe" (< 30) ou "paper" (>= 200) só com o
       Laplaciano (sem calcular Canny).
    3) Só mede Canny na faixa intermediária 30 <= Laplaciano < 200.
    Resultado 100% idêntico à avaliação completa.
    """
    if maps is None:
        maps = _LazyTextureMaps(frame)
    roi = _candidate_interior_roi(frame.shape, corners)
    if roi is None:
        return None
    ys, xs, selected = roi
    saturation = maps["saturation"][ys, xs][selected]
    value = maps["value"][ys, xs][selected]
    white_fraction = float(np.mean((saturation < 60) & (value > 180)))
    if white_fraction < _TEXTURE_MIN_WHITE_FRACTION:
        return None
    laplacian_variance = float(np.var(maps["laplacian"][ys, xs][selected]))
    if laplacian_variance < _TEXTURE_MIN_LAPLACIAN_VARIANCE:
        return "blank_maybe"
    if laplacian_variance >= _TEXTURE_HIGH_LAPLACIAN_VARIANCE:
        return "paper"
    edge_density = float(np.mean(maps["edges"][ys, xs][selected] > 0))
    return "paper" if edge_density >= _TEXTURE_MIN_EDGE_DENSITY else "blank_maybe"


def detect_sheet(frame):
    """Encontra os cantos de uma folha sem marcadores.

    Retorna um array float32 com os cantos TL, TR, BR e BL. Retorna None
    quando nenhum quadrilátero grande e convexo passa no portão de
    textura como papel. Candidato branco e liso demais não conta como
    folha aqui: ele aparece em detect_sheet_with_confidence com o
    veredito blank_maybe.
    """
    corners, verdict = detect_sheet_with_confidence(frame)
    return corners if verdict == "paper" else None


def detect_sheet_with_confidence(frame):
    """Encontra os cantos de uma folha sem marcadores, com veredito.

    Retorna (cantos, veredito). O veredito é "paper" quando o interior
    do candidato passa no portão de textura. É "blank_maybe" quando um
    candidato branco o bastante aparece, mas liso demais para afirmar
    papel. É None, com cantos None, quando não há candidato.
    """
    if frame is None or frame.size == 0:
        return None, None

    frame_h, frame_w = frame.shape[:2]
    scale = min(1.0, 960.0 / max(frame_w, frame_h))
    if scale < 1.0:
        small = cv2.resize(
            frame,
            (int(frame_w * scale), int(frame_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = frame

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # RETR_TREE: o contorno da folha pode ser interno na hierarquia, como
    # a volta de um caderno dentro de um borrão maior fundido na cena.
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    small_frame_area = float(small.shape[0] * small.shape[1])
    min_area = small_frame_area * _SHEET_MIN_AREA_RATIO
    candidates = [
        contour
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)
        if cv2.contourArea(contour) >= min_area
    ][:_SHEET_MAX_CANDIDATES]
    # Um candidato que falha no portão não encerra a busca: um contorno
    # seguinte ainda pode ser a folha.
    blank_corners = None
    texture_maps = None
    for contour in candidates:
        corners = _sheet_candidate(contour, scale, small_frame_area)
        if corners is None:
            continue
        if texture_maps is None:
            texture_maps = _LazyTextureMaps(frame)
        verdict = _texture_verdict(frame, corners, texture_maps)
        if verdict == "paper":
            return _refine_corners(frame, corners), "paper"
        if verdict == "blank_maybe" and blank_corners is None:
            blank_corners = corners
    if blank_corners is not None:
        return _refine_corners(frame, blank_corners), "blank_maybe"
    return None, None


def detect_markers(frame):
    """Retorna {id: centro (x, y)} dos marcadores encontrados no frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _get_detector().detectMarkers(gray)
    found = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True):
            marker_id = int(marker_id)
            if marker_id in config.MARKER_IDS:
                center = marker_corners[0].mean(axis=0)
                found[marker_id] = (float(center[0]), float(center[1]))
    return found


def perspective_matrix(corners, destination=None):
    """Matriz de homografia 3x3 dos cantos da cena para a folha retificada."""
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if destination is None:
        destination = [
            (0, 0),
            (config.PAPER_W, 0),
            (config.PAPER_W, config.PAPER_H),
            (0, config.PAPER_H),
        ]
    dst = np.asarray(destination, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src, dst)


def warp_points(points, corners, destination=None):
    """Projeta pontos Nx2 do frame da câmera para o espaço retificado."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    matrix = perspective_matrix(corners, destination=destination)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def warp_sheet(frame, corners, destination=None):
    """Retifica a folha a partir de quatro cantos explícitos."""
    matrix = perspective_matrix(corners, destination=destination)
    return cv2.warpPerspective(frame, matrix, (config.PAPER_W, config.PAPER_H))


# Piso de cobertura válida do mapa denso: abaixo disso o remap
# teria buraco demais e a homografia de 4 cantos é resposta melhor.
_UVMAP_MIN_VALID = 0.2
# Fração mínima de amostras válidas na vizinhança para um ponto do
# mapa redimensionado continuar válido (convolução normalizada).
_UVMAP_RESIZE_SUPPORT = 0.5


def _uv_valid(uv):
    """Ponto utilizável: as duas coordenadas finitas e dentro de [0, 1]."""
    with np.errstate(invalid="ignore"):
        return (np.isfinite(uv) & (uv >= 0.0) & (uv <= 1.0)).all(axis=0)


def _resize_uv(uv, valid, w, h):
    """Redimensiona o mapa sem espalhar buracos (convolução normalizada).

    Interpolar NaN direto contamina os vizinhos válidos. Aqui cada
    canal é reamostrado com os inválidos zerados e dividido pela
    reamostragem do próprio peso de validade; só fica inválido o
    ponto cuja vizinhança é majoritariamente buraco.
    """
    weight = valid.astype(np.float32)
    support = cv2.resize(weight, (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.empty((2, h, w), np.float32)
    for c in range(2):
        filled = np.where(valid, uv[c], 0.0).astype(np.float32)
        acc = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = acc / support
    out[:, support < _UVMAP_RESIZE_SUPPORT] = np.nan
    return out


def warp_sheet_uvmap(frame, uvmap, corners=None, destination=None):
    """Retifica a folha por remap denso (folha curva/dobrada).

    uvmap: mapa backward (2,H,W) normalizado em [0,1] — para cada
    posição da grade de saída, a coordenada de entrada a amostrar
    (a convenção do modelo de folha v4). Ponto NaN ou fora de [0,1]
    é inválido e vira borda preta.

    A saída tem o mesmo contrato de `warp_sheet`: sempre a grade do
    papel (PAPER_W x PAPER_H), com o mapa posicionado no retângulo de
    `destination` — trocar de caminho nunca muda o tamanho da imagem.

    Sem mapa, com mapa de área zero, com destino fora da grade ou com
    poucos pontos válidos (<20%, medidos no trecho que cai DENTRO da
    grade de saída), cai na homografia de
    `corners`. Devolve None se não houver nem mapa útil nem cantos.
    """
    if destination is None:
        destination = [
            (0, 0),
            (config.PAPER_W, 0),
            (config.PAPER_W, config.PAPER_H),
            (0, config.PAPER_H),
        ]
    dst = np.asarray(destination, dtype=np.float32).reshape(4, 2)
    x0 = int(round(min(dst[0, 0], dst[3, 0])))
    y0 = int(round(min(dst[0, 1], dst[1, 1])))
    w = int(round(max(dst[1, 0] - dst[0, 0], dst[2, 0] - dst[3, 0])))
    h = int(round(max(dst[3, 1] - dst[0, 1], dst[2, 1] - dst[1, 1])))
    grid_h, grid_w = config.PAPER_H, config.PAPER_W
    # Recorte do retângulo de destino que cai dentro da grade: só ele
    # chega à saída, então só ele conta para a cobertura.
    gx0, gy0 = max(x0, 0), max(y0, 0)
    gx1, gy1 = min(x0 + w, grid_w), min(y0 + h, grid_h)
    sub = ok = None
    if uvmap is not None and w > 0 and h > 0 and gx1 > gx0 and gy1 > gy0:
        candidate = np.asarray(uvmap, dtype=np.float32)
        if candidate.ndim == 3 and candidate.shape[0] == 2 and candidate[0].size:
            if candidate.shape[1:] != (h, w):
                candidate = _resize_uv(candidate, _uv_valid(candidate), w, h)
            sub = candidate[:, gy0 - y0 : gy1 - y0, gx0 - x0 : gx1 - x0]
            ok = _uv_valid(sub)
            if float(ok.mean()) < _UVMAP_MIN_VALID:
                sub = None
    if sub is None:
        if corners is None:
            return None
        return warp_sheet(frame, corners, destination=destination)
    map_x = np.full((grid_h, grid_w), -1.0, np.float32)
    map_y = np.full((grid_h, grid_w), -1.0, np.float32)
    map_x[gy0:gy1, gx0:gx1] = np.where(ok, sub[0] * (frame.shape[1] - 1), -1.0)
    map_y[gy0:gy1, gx0:gx1] = np.where(ok, sub[1] * (frame.shape[0] - 1), -1.0)
    return cv2.remap(
        frame,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def warp_paper(frame):
    """Retifica a folha-modelo a partir dos marcadores.

    Retorna (imagem_retificada, marcadores) ou (None, marcadores) se os
    quatro cantos não estiverem visíveis.
    """
    markers = detect_markers(frame)
    if len(markers) < len(config.MARKER_IDS):
        return None, markers

    src = np.array([markers[i] for i in config.MARKER_IDS], dtype=np.float32)
    dst = np.array(
        [config.MARKER_CENTERS[i] for i in config.MARKER_IDS], dtype=np.float32
    )
    return warp_sheet(frame, src, dst), markers
