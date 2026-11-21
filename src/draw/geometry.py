"""Geometria pura do ajuste do quadrilátero da folha.

Funções sem estado extraídas do detector neural: contorno, retas de
aresta e o ajuste completo da folha (fit_quad). A memória de arestas
entre chamadas mora no detector (draw.perception), não aqui.
"""

import cv2
import numpy as np

from . import paper as _paper

_MIN_CONTOUR_AREA_RATIO = 0.01
_FIT_MAX_DISPLACE_PX = 100.0
_FIT_MIN_EDGE_COVERAGE = 0.5
_FIT_MAX_INLIER_RESIDUAL_PX = 6.0
_FIT_MIN_INLIER_RATIO = 0.50
_FIT_MAX_INLIER_RMS_PX = 4.0
_MEMORY_STATIONARY_ANGLE_DEG = 3.0
_MEMORY_STATIONARY_SHIFT_PX = 8.0
_MEMORY_SINGLE_OUTLIER_ANGLE_DEG = 6.0
_MEMORY_SINGLE_OUTLIER_SHIFT_PX = 18.0


def fill_interior_holes(mask):
    """Fecha os buracos internos de uma máscara binária 0/1.

    Fundo encerrado por folha — mão, objeto ou traço que o modelo
    não reconheceu dentro da página — volta para a cobertura; fundo
    ligado à borda da cena (a mesa de verdade) segue fora. É a
    diferença entre "o detector não viu papel ali" e "não é folha".
    """
    outside = (mask == 0).astype(np.uint8)
    # A margem de 1 px garante um fundo contíguo para a semente.
    flood = np.pad(outside, 1, constant_values=1)
    cv2.floodFill(
        flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 0
    )
    holes = flood[1:-1, 1:-1]
    return mask | (holes != 0).astype(np.uint8)


# Guarda de plausibilidade do ajuste (medida em 2026-10-30 no
# laboratório, draw-lab/noite-2026-10-30/block6/): o ajuste neural cru
# é biestável sob oclusão. Nas 1.132 detecções gravadas dos dois
# vídeos de replay, uma família de ajustes colapsa uma aresta (a
# direita com 213–269 px contra a esquerda com 980–1.025 px: razão
# entre arestas opostas de 3,81 a 4,67), enquanto tudo o que o app
# exibiu em trechos quietos fica em 2,42 ou menos e o máximo do outro
# vídeo é 3,00. A guarda recusa o quadrilátero cuja maior razão entre
# arestas opostas passa de _FIT_OPPOSITE_RATIO_MAX. Recusa também o
# que extrapola um canto para além de _FIT_EXCURSION_DIAG_FRAC da
# diagonal do quadro: nas gravações a folha sai do recorte de tela de
# verdade até 9,9% da diagonal, mas o modo fora do quadro medido no
# frame_real.png (1289x710) chegou a 21,4%. Ajuste recusado vale
# "sem quadrilátero" e não alimenta a memória de arestas: quem chamou
# decide (no app, o rastreador segura o quadrilátero anterior). Nas
# gravações, a guarda recusa 26 de 1.132 ajustes (25 da família
# colapsada de um vídeo, 1 do outro). Ressalva: as gravações são de
# tela, com overlay queimado; a guarda precisa de reconfirmação em
# captura crua.
_FIT_OPPOSITE_RATIO_MAX = 3.0
# Aresta "na borda do quadro": os DOIS vértices a ~2% do lado menor da
# mesma borda (um corte de frame corre ao longo dela). A aresta curta
# interior é a assinatura do colapso por oclusão; na borda ela pode ser o
# próprio corte do clip, e a razão não decide. ~2% cobre a quantização da
# máscara do detector redimensionada.
_FIT_EDGE_BORDER_GAP_FRAC = 0.02
_FIT_EXCURSION_DIAG_FRAC = 0.15

# Guarda de suporte do ajuste na máscara (medida em 2026-11-21 nas
# capturas draw-data/captures s1-s4): o ajuste precisa ficar sobre a
# região não-fundo que o modelo marcou. Num cenário fora da mesa de
# referência (s3_geometria, câmera oblíqua 640x480 com piso claro à
# vista), o modelo rotula região de não-papel como folha e o ajuste
# explode para um quadrilátero que cobre braço, chão e o frame inteiro
# — razão de arestas ~1 e excursão dentro da margem, então a guarda de
# plausibilidade deixava passar. Nessas capturas, ajustes sãos cobrem a
# máscara em >= 0,84 (p5 0,91 na sessão de escrita da mesa de
# referência), os quads explodidos ficam entre 0,56 e 0,80 e os
# fantasmas de mesa vazia (s1) chegam a 0,43. A região de suporte é a
# cobertura (tudo que não é fundo) quando o detector a fornece — mão
# sobre a folha não pode secar a métrica — e a própria máscara fora
# dele. Ressalva igual à guarda de plausibilidade: limiar calibrado nas
# capturas da mesa de referência e de um cenário oblíquo; reconfirmar
# em capturas de outras mesas.
_FIT_MIN_MASK_FILL = 0.82


def _quad_supported(quad, support_mask):
    """True se a região não-fundo cobre a maior parte do quadrilátero.

    O quadrilátero é desenhado na grade de `support_mask`; a fração
    coberta por pixels não-fundo precisa chegar a _FIT_MIN_MASK_FILL.
    É a medida que separa o ajuste apoiado na folha do que escorregou
    para cima de fundo mal classificado — o piso claro rotulado como
    papel que produzia quad do tamanho do frame inteiro.
    """
    filled = np.zeros(support_mask.shape, np.uint8)
    cv2.fillConvexPoly(filled, np.round(quad).astype(np.int32), 255)
    area = int(np.count_nonzero(filled))
    if area == 0:
        return False
    inside = int(np.count_nonzero((filled != 0) & (support_mask != 0)))
    return inside / area >= _FIT_MIN_MASK_FILL


def _edge_border_sides(p, q, border_gap, width, height):
    """Bordas do quadro ao longo das quais a aresta p→q pode correr."""
    sides = []
    if max(p[0], q[0]) <= border_gap:
        sides.append("left")
    if max(p[1], q[1]) <= border_gap:
        sides.append("top")
    if width - min(p[0], q[0]) <= border_gap:
        sides.append("right")
    if height - min(p[1], q[1]) <= border_gap:
        sides.append("bottom")
    return sides


def _mask_supports_cut(evidence, observable, side, p, q, scale_x, scale_y):
    """O recorte declarado em `side` tem suporte no contorno observado?"""
    mh, mw = evidence.shape
    if side in ("left", "right"):
        i = 0 if side == "left" else mw - 1
        a = int(min(p[1], q[1]) // scale_y)
        b = int(max(p[1], q[1]) // scale_y)
        st_e = evidence[max(0, a) : min(mh, b + 1), i]
        st_o = observable[max(0, a) : min(mh, b + 1), i]
    else:
        i = 0 if side == "top" else mh - 1
        a = int(min(p[0], q[0]) // scale_x)
        b = int(max(p[0], q[0]) // scale_x)
        st_e = evidence[i, max(0, a) : min(mw, b + 1)]
        st_o = observable[i, max(0, a) : min(mw, b + 1)]
    obs = int(np.count_nonzero(st_o))
    return obs > 0 and int(np.count_nonzero(st_e)) * 2 >= obs


def _frame_cut(
    quad, u, v, border_gap, width, height, evidence, observable, scale_x, scale_y
):
    """A aresta quad[u]→quad[v] e corte de frame?

    A aresta precisa correr ao longo de uma borda do quadro e, quando a
    mascara nativa existe, ter suporte do contorno nela — perto de um
    canto a aresta pode caber em duas bordas e basta uma ser suportada.
    """
    p, q = quad[u % 4], quad[v % 4]
    sides = _edge_border_sides(p, q, border_gap, width, height)
    return bool(sides) and (
        evidence is None
        or any(
            _mask_supports_cut(evidence, observable, side, p, q, scale_x, scale_y)
            for side in sides
        )
    )


def fit_plausible(
    corners,
    frame_shape,
    mask=None,
    scale_x=1.0,
    scale_y=1.0,
    contour=None,
    hand=None,
):
    """True se o quadrilátero passa na guarda de plausibilidade.

    A primeira regra é finitude. Toda comparação com NaN é falsa, e
    sem esta linha um quadrilátero com NaN atravessava as outras
    regras sem disparar nenhuma: a guarda aprovava justamente o
    ajuste mais inválido possível. Infinito já era recusado pela
    comparação com a margem do quadro, mas a regra explícita cobre
    os dois casos com a mesma mensagem.

    `mask`, `scale_x` e `scale_y` trazem a máscara nativa e a escala
    frame→máscara para validar os cortes de frame que isentam a razão
    entre arestas opostas; `contour` restringe a evidência aos pixels
    do componente ajustado e `hand` marca pixels de mão como não
    observáveis. Sem `mask` a isenção é só geométrica (vértices).
    """
    quad = np.asarray(corners, dtype=np.float64)
    if quad.shape != (4, 2):
        return False
    if not np.all(np.isfinite(quad)):
        return False
    edges = np.array([np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)])
    if edges.min() <= 0:
        return False
    height, width = frame_shape[:2]
    border_gap = _FIT_EDGE_BORDER_GAP_FRAC * min(width, height)
    evidence = observable = None
    if mask is not None:
        # Evidencia do corte = pixels reais da mascara dentro do contorno
        # escolhido para o ajuste: foreground destacado ou preenchimento
        # de casco nao conta como contato com a borda. O denominador e o
        # trecho INTEIRO observavel — fundo visivel fora do contorno
        # conta como "nao papel"; so pixels de mao saem como nao
        # observaveis (e nunca podem ser evidencia).
        if contour is None:
            region = np.ones(mask.shape, bool)
        else:
            region = np.zeros(mask.shape, np.uint8)
            cv2.fillPoly(region, [contour], 255)
            region = region.astype(bool)
        evidence = (mask > 0) & region
        observable = np.ones(mask.shape, bool)
        if hand is not None and hand.shape == mask.shape:
            evidence &= hand == 0
            observable = hand == 0
    for a, b in ((0, 2), (3, 1)):
        ratio = max(edges[a], edges[b]) / min(edges[a], edges[b])
        if ratio <= _FIT_OPPOSITE_RATIO_MAX:
            continue
        short = a if edges[a] <= edges[b] else b
        other = b if short == a else a
        # Isenta o par quando uma aresta dele e corte de frame — o
        # comprimento e decidido pela borda da imagem, nao pela pagina:
        # a curta pode ser o corte, a longa pode ser o corte (a curta e
        # entao a aresta distante real, interior), ou um vizinho da
        # curta e o corte que comeu o canto dela. Com mascara disponivel
        # o corte precisa de suporte no contorno nativo: aresta de papel
        # passando perto da borda nao isenta um colapsado.
        if not any(
            _frame_cut(
                quad,
                u,
                v,
                border_gap,
                width,
                height,
                evidence,
                observable,
                scale_x,
                scale_y,
            )
            for u, v in (
                (short, short + 1),
                (other, other + 1),
                (short - 1, short),
                (short + 1, short + 2),
            )
        ):
            return False
    margin = _FIT_EXCURSION_DIAG_FRAC * float(np.hypot(width, height))
    return not (
        quad[:, 0].min() < -margin
        or quad[:, 0].max() > width + margin
        or quad[:, 1].min() < -margin
        or quad[:, 1].max() > height + margin
    )


def order_corners(points):
    return _paper._order_corners(points)


def _largest_contour(mask_bin):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# Calha maxima entre as paginas (fracao da extensao maior). Folhas soltas lado a
# lado ficam bem alem disso; so um caderno aberto tem calha estreita.
_SPREAD_MAX_GUTTER = 0.10
_SPREAD_MODES = ("auto", "always", "never")


def _validate_spread_mode(mode):
    if mode not in _SPREAD_MODES:
        choices = ", ".join(_SPREAD_MODES)
        raise ValueError(f"spread_mode deve ser um de: {choices}")


def _is_two_page_spread(contour_a, contour_b, *, mode="auto"):
    """True quando dois contornos grandes formam um caderno aberto (2 paginas).

    Projeta os dois contornos no referencial proprio da pagina maior
    (eixos do `minAreaRect`) para reconhecer cadernos girados na mesa,
    onde caixas alinhadas aos eixos da camera se sobrepoem falsamente. `auto`
    exige calha de ate 10%; `always` ignora apenas esse limite; `never`
    desativa a uniao.
    """
    _validate_spread_mode(mode)
    if mode == "never":
        return False

    area_a = float(cv2.contourArea(contour_a))
    area_b = float(cv2.contourArea(contour_b))
    if area_a <= 0 or area_b < 0.35 * area_a:
        return False

    (_cx, _cy), (rw, rh), angle_deg = cv2.minAreaRect(contour_a)
    if min(rw, rh) <= 0:
        return False
    rad = np.deg2rad(angle_deg)
    u = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
    v = np.array([-np.sin(rad), np.cos(rad)], dtype=np.float64)

    pts_a = contour_a.reshape(-1, 2).astype(np.float64)
    pts_b = contour_b.reshape(-1, 2).astype(np.float64)
    ua, va = pts_a @ u, pts_a @ v
    ub, vb = pts_b @ u, pts_b @ v

    ua_min, ua_max = float(ua.min()), float(ua.max())
    va_min, va_max = float(va.min()), float(va.max())
    ub_min, ub_max = float(ub.min()), float(ub.max())
    vb_min, vb_max = float(vb.min()), float(vb.max())

    wa, ha = ua_max - ua_min, va_max - va_min
    wb, hb = ub_max - ub_min, vb_max - vb_min
    if min(wa, ha, wb, hb) <= 0:
        return False

    # Separadas ao longo de u, alinhadas ao longo de v.
    v_overlap = min(va_max, vb_max) - max(va_min, vb_min)
    u_gap = max(ua_min, ub_min) - min(ua_max, ub_max)
    if (
        v_overlap >= 0.55 * min(ha, hb)
        and min(ha, hb) >= 0.65 * max(ha, hb)
        and -0.15 * min(wa, wb) <= u_gap
        and (mode == "always" or u_gap <= _SPREAD_MAX_GUTTER * max(wa, wb))
    ):
        return True

    # Separadas ao longo de v, alinhadas ao longo de u.
    u_overlap = min(ua_max, ub_max) - max(ua_min, ub_min)
    v_gap = max(va_min, vb_min) - min(va_max, vb_max)
    return (
        u_overlap >= 0.55 * min(wa, wb)
        and min(wa, wb) >= 0.65 * max(wa, wb)
        and -0.15 * min(ha, hb) <= v_gap
        and (mode == "always" or v_gap <= _SPREAD_MAX_GUTTER * max(ha, hb))
    )


def _sheet_contour_with_spread(mask_bin, *, spread_mode="auto"):
    """Devolve `(contorno, is_spread, contour_area)` para distinguir folha unica
    de caderno aberto sem duplicar a extracao de contorno no detector."""
    _validate_spread_mode(spread_mode)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, False, 0.0
    if len(contours) == 1:
        return contours[0], False, float(cv2.contourArea(contours[0]))
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)
    if _is_two_page_spread(ranked[0], ranked[1], mode=spread_mode):
        hull = cv2.convexHull(np.vstack([ranked[0], ranked[1]]))
        # Densifica o contorno do casco desenhando-o preenchido numa mascara:
        # cv2.convexHull devolve apenas ~4-8 vertices e faria _refit_corners
        # desistir (<8 pontos/aresta), perdendo o refino e a memoria de arestas.
        merged_mask = np.zeros(mask_bin.shape[:2], dtype=np.uint8)
        cv2.drawContours(merged_mask, [hull], -1, 255, cv2.FILLED)
        dense, _ = cv2.findContours(
            merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        merged = max(dense, key=cv2.contourArea) if dense else hull
        return merged, True, float(cv2.contourArea(merged))
    return ranked[0], False, float(cv2.contourArea(ranked[0]))


def _sheet_contour(mask_bin):
    """Contorno da folha: maior contorno ou uniao densa das 2 paginas do caderno."""
    contour, _is_spread, _area = _sheet_contour_with_spread(mask_bin)
    return contour


def _approx_quad(contour):
    perimeter = cv2.arcLength(contour, True)
    for epsilon in _paper._SHEET_EPSILON_LADDER:
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx.reshape(4, 2))
    return None


def _min_area_quad(contour):
    """Reserva do ajuste: o retângulo de área mínima do contorno.

    Só é chamado quando a escada de epsilon não fecha. Os cantos
    saem na mesma convenção TL/TR/BR/BL do caminho principal, e o
    retângulo volta cru, sem o reajuste de arestas e sem o refino
    subpixel. Medido: nos frames 24 e 26 das capturas reais, os
    únicos em que a escada falha com o modelo v2, o IoU do
    quadrilátero preenchido do frame 24 contra a máscara anotada
    dele é 0,895 com o retângulo cru, 0,850 com o reajuste, 0,843
    com reajuste e refino e 0,887 só com o refino. O reajuste foi
    desenhado para os vértices da aproximação, que moram perto dos
    cantos da folha; os vértices do retângulo mínimo moram fora
    dela, e as retas ajustadas passam a seguir a borda visível da
    máscara, que nesses frames é oclusão, não a aresta da página.
    """
    return order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))


def _fit_line(points):
    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
    ).ravel()
    direction = np.array([vx, vy], dtype=np.float64)
    point = np.array([x0, y0], dtype=np.float64)
    return direction, point


def _intersect(line_a, line_b):
    dir_a, point_a = line_a
    dir_b, point_b = line_b
    matrix = np.column_stack([dir_a, -dir_b])
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    t = np.linalg.solve(matrix, point_b - point_a)[0]
    return point_a + t * dir_a


def _segment_distances(points, start, end):
    segment = end - start
    t = np.clip(
        ((points - start) @ segment) / max(float(segment @ segment), 1e-9),
        0.0,
        1.0,
    )
    return np.linalg.norm(points - (start + np.outer(t, segment)), axis=1)


def _line_residuals(points, line):
    dir_fit, point_fit = line
    normal = np.array([-dir_fit[1], dir_fit[0]], dtype=np.float64)
    return np.abs((points - point_fit) @ normal)


def _line_angle_diff_deg(line_a, line_b):
    dir_a = line_a[0] / max(float(np.linalg.norm(line_a[0])), 1e-9)
    dir_b = line_b[0] / max(float(np.linalg.norm(line_b[0])), 1e-9)
    cos_ang = min(1.0, abs(float(dir_a @ dir_b)))
    return float(np.degrees(np.arccos(cos_ang)))


def _line_normal_shift_at(point, line_ref):
    dir_ref, pt_ref = line_ref
    normal = np.array([-dir_ref[1], dir_ref[0]], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    return abs(float((point - pt_ref) @ normal))


_RECTIFIED_EDGE_UNCERTAINTY_PX = 2.5
# Inclinacao maxima (|across|/|along|) de uma aresta no plano retificado:
# tan(6 graus) com folga para o aspecto nao isotropico do quadrado unitario.
_RECTIFIED_PARALLEL_SLOPE = 0.16


def _rectified_translation_consistent(
    fallback_lines,
    out_i,
    out_mid,
    out_norm,
    in_i,
    in_mid,
    in_norm,
    new_lines=None,
):
    """Confere se dois saltos em arestas opostas sao iguais e opostos no plano
    retificado da pagina anterior (invariante a perspectiva de camera inclinada).

    A tolerancia absoluta projeta a incerteza por aresta em pixels nativos da
    mascara (_RECTIFIED_EDGE_UNCERTAINTY_PX) atraves da homografia nos pontos
    medidos, sem encolher com a resolucao da camera. Retorna True/False quando
    o quadrilatero anterior e valido, ou None quando faltam retas de fallback.
    """
    if any(line is None for line in fallback_lines):
        return None
    prev_corners = []
    for k in range(4):
        corner = _intersect(fallback_lines[k - 1], fallback_lines[k])
        if corner is None:
            return None
        prev_corners.append(corner)
    prev_quad = np.asarray(prev_corners, dtype=np.float32)
    if not cv2.isContourConvex(prev_quad.reshape(4, 1, 2)):
        return None
    unit_sq = np.float32([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    homography = cv2.getPerspectiveTransform(prev_quad, unit_sq)

    def _outward_offset(edge_idx, point):
        p = homography @ np.array([point[0], point[1], 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-9:
            return None
        u, v = float(p[0] / p[2]), float(p[1] / p[2])
        if edge_idx == 0:
            return -v
        if edge_idx == 1:
            return u - 1.0
        if edge_idx == 2:
            return v - 1.0
        return -u

    out_off = _outward_offset(out_i, out_mid)
    in_off = _outward_offset(in_i, in_mid)
    if out_off is None or in_off is None:
        return None
    if out_off <= 0.0 or in_off >= 0.0:
        return False
    if new_lines is not None:
        # Translacao no plano da pagina mantem cada aresta paralela a si mesma
        # no referencial retificado, mesmo que o angulo na imagem mude muito
        # (trapezio). Pivo de oclusao gira a aresta tambem la.
        for edge_idx, line in ((out_i, new_lines[0]), (in_i, new_lines[1])):
            if line is None:
                return None
            direction, point = line
            a = homography @ np.array([point[0], point[1], 1.0], dtype=np.float64)
            probe = point + 50.0 * np.asarray(direction, dtype=np.float64)
            b = homography @ np.array([probe[0], probe[1], 1.0], dtype=np.float64)
            if abs(a[2]) < 1e-9 or abs(b[2]) < 1e-9:
                return None
            du = float(b[0] / b[2] - a[0] / a[2])
            dv = float(b[1] / b[2] - a[1] / a[2])
            along, across = (du, dv) if edge_idx in (0, 2) else (dv, du)
            if abs(across) > _RECTIFIED_PARALLEL_SLOPE * abs(along):
                return False
    err_px = _RECTIFIED_EDGE_UNCERTAINTY_PX
    out_pert = _outward_offset(out_i, out_mid + err_px * out_norm)
    in_pert = _outward_offset(in_i, in_mid + err_px * in_norm)
    if out_pert is None or in_pert is None:
        return None
    abs_tol = abs(out_pert - out_off) + abs(in_pert - in_off)
    mag = max(abs(out_off), abs(in_off))
    return abs(out_off + in_off) <= max(abs_tol, 0.35 * mag)


def _is_interior_pivot(edge_idx, new_line, new_start, new_end, fallback_lines):
    """Retorna True se a reta nova cruza a reta anterior no INTERIOR do trecho
    compartilhado entre a aresta velha e a nova (pivo real sob oclusao).

    Quando a intersecao ocorre na extremidade do trecho compartilhado (ex.: uma
    aresta previamente cortada em diagonal desentorta ao ser revelada), retorna
    False para nao travar a borda revelada. Retorna None se faltam vizinhos.
    """
    prev_line = fallback_lines[edge_idx]
    prev_left = fallback_lines[(edge_idx - 1) % 4]
    prev_right = fallback_lines[(edge_idx + 1) % 4]
    if prev_line is None or prev_left is None or prev_right is None:
        return None
    old_start = _intersect(prev_left, prev_line)
    old_end = _intersect(prev_line, prev_right)
    pivot = _intersect(new_line, prev_line)
    if old_start is None or old_end is None or pivot is None:
        return False
    dir_prev = prev_line[0] / max(float(np.linalg.norm(prev_line[0])), 1e-9)
    old_a = float(old_start @ dir_prev)
    old_b = float(old_end @ dir_prev)
    new_a = float(np.asarray(new_start, dtype=np.float64) @ dir_prev)
    new_b = float(np.asarray(new_end, dtype=np.float64) @ dir_prev)
    lo = max(min(old_a, old_b), min(new_a, new_b))
    hi = min(max(old_a, old_b), max(new_a, new_b))
    span = hi - lo
    if span <= 1e-6:
        return False
    t = (float(pivot @ dir_prev) - lo) / span
    return 0.15 < t < 0.85


def _best_inlier_seed_line(side, along, pixel_scale=1.0):
    """Escolhe semente linear robusta a entalhe de braço no meio da aresta."""
    candidates = [_fit_line(side)]
    order = np.argsort(along)
    n = len(order)
    if n >= 16:
        for q0, q1 in ((0.06, 0.94), (0.06, 0.38), (0.62, 0.94), (0.18, 0.82)):
            p0 = side[order[int(q0 * (n - 1))]]
            p1 = side[order[int(q1 * (n - 1))]]
            chord = p1 - p0
            chord_len = float(np.linalg.norm(chord))
            if chord_len >= 12.0 * pixel_scale:
                candidates.append((chord / chord_len, p0))

    best_line = candidates[0]
    best_score = (-1, -1e9)
    for cand in candidates:
        res = _line_residuals(side, cand)
        inliers = res <= _FIT_MAX_INLIER_RESIDUAL_PX * pixel_scale
        count = int(inliers.sum())
        neg_med = -float(np.median(res[inliers])) if count > 0 else -1e9
        score = (count, neg_med)
        if score > best_score:
            best_score = score
            best_line = cand
    return best_line


def _edge_line(side, start, end, centroid, fallback=None, pixel_scale=1.0):
    """A reta de uma aresta: ajustada, herdada ou travada na direção.

    Com cobertura boa E colinearidade real dentro de resíduo absoluto
    (_FIT_MAX_INLIER_RESIDUAL_PX), vale o ajuste de Huber podado aos
    inliers retos e a reta é confiável. Quando um braço/mão curvo invade
    a máscara mantendo cobertura >= 0,5, o portão absoluto de retidão
    reprova a aresta: ela não contamina a memória (_edge_lines) e herda
    `fallback` quando disponível.
    """
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length < 1e-6:
        return _fit_line(side), False, False
    direction = segment / length
    along = (side - start) @ direction
    coverage = float(along.max() - along.min()) / length
    if coverage >= _FIT_MIN_EDGE_COVERAGE:
        seed = _best_inlier_seed_line(side, along, pixel_scale)
        residuals = _line_residuals(side, seed)
        tol = min(
            _FIT_MAX_INLIER_RESIDUAL_PX * pixel_scale,
            max(3.0 * pixel_scale, float(np.percentile(residuals, 70))),
        )
        keep = residuals <= tol
        if int(keep.sum()) >= 8:
            line = _fit_line(side[keep])
            residuals = _line_residuals(side, line)
            inliers = residuals <= _FIT_MAX_INLIER_RESIDUAL_PX * pixel_scale
            if int(inliers.sum()) >= 8:
                line = _fit_line(side[inliers])
                residuals = _line_residuals(side, line)
                inliers = residuals <= _FIT_MAX_INLIER_RESIDUAL_PX * pixel_scale
        else:
            line = seed
            inliers = residuals <= _FIT_MAX_INLIER_RESIDUAL_PX * pixel_scale

        n_inliers = int(inliers.sum())
        if n_inliers >= 8:
            inlier_along = along[inliers]
            inlier_cov = float(inlier_along.max() - inlier_along.min()) / length
            inlier_ratio = n_inliers / float(len(side))
            inlier_rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
            # Aceita também entalhe localizado no meio quando sobram >= 24
            # inliers colineares cobrindo ambos os lados da aresta.
            ratio_ok = (inlier_ratio >= _FIT_MIN_INLIER_RATIO) or (
                n_inliers >= 24 and inlier_cov >= 0.70
            )
            if (
                inlier_cov >= _FIT_MIN_EDGE_COVERAGE
                and ratio_ok
                and inlier_rms <= _FIT_MAX_INLIER_RMS_PX * pixel_scale
            ):
                return line, False, True

    if fallback is not None:
        return fallback, True, False
    normal = np.array([-direction[1], direction[0]])
    if float((centroid - start) @ normal) < 0:
        normal = -normal
    offset = float(np.median((side - start) @ normal))
    return (direction, start + offset * normal), True, False


def _refit_corners(contour, quad, frame_shape, fallback_lines=None, scale=(1.0, 1.0)):
    """Reajusta os cantos como interseções das retas das arestas.

    Cada ponto do contorno é atribuído à aresta mais próxima. Pontos
    colados nos vértices da aproximação ficam fora (são a virada de
    uma aresta para outra). Pontos sobre a borda do frame também
    ficam fora quando sobra ponto bastante sem eles: a máscara ali
    foi cortada pelo quadro. A reta de cada aresta sai de _edge_line
    (ajustada; sem cobertura, herdada de fallback_lines ou travada
    na direção do segmento). Um canto reajustado só substitui o
    vértice se ficar a no máximo _FIT_MAX_DISPLACE_PX dele, exceto
    quando uma das arestas do canto é travada: ali a projeção já
    nasce limitada pela trava e o vértice de referência é justamente
    o contaminado. O quadrilátero final precisa continuar convexo.

    Retorna (quadrilatero, retas usadas, confiavel por aresta); as
    retas voltam None quando alguma aresta tem ponto de menos e o
    reajuste inteiro é abandonado.
    """
    frame_h, frame_w = frame_shape
    # Resíduos usam limites conservadores; deslocamentos dos quatro cantos
    # usam a métrica exata do frame, inclusive quando scale_x != scale_y.
    pixel_scale = 1.0 / max(scale)
    points = contour.reshape(-1, 2).astype(np.float64)
    distances = np.stack(
        [_segment_distances(points, quad[i], quad[(i + 1) % 4]) for i in range(4)],
        axis=1,
    )
    assignment = np.argmin(distances, axis=1)
    centroid = quad.astype(np.float64).mean(axis=0)
    lines = []
    locked = []
    trusted = []
    for i in range(4):
        start = quad[i].astype(np.float64)
        end = quad[(i + 1) % 4].astype(np.float64)
        side = points[assignment == i]
        if len(side) < 8:
            return quad, None, None
        near_vertex = (np.linalg.norm(side - start, axis=1) < 25 * pixel_scale) | (
            np.linalg.norm(side - end, axis=1) < 25 * pixel_scale
        )
        side = side[~near_vertex]
        on_border = (
            (side[:, 0] <= 0.5)
            | (side[:, 0] >= frame_w - 1.5)
            | (side[:, 1] <= 0.5)
            | (side[:, 1] >= frame_h - 1.5)
        )
        off_border = side[~on_border]
        if len(off_border) >= 8:
            side = off_border
        if len(side) < 8:
            return quad, None, None
        fallback = None
        if fallback_lines is not None:
            fallback = fallback_lines[i]
        line, is_locked, is_trusted = _edge_line(
            side, start, end, centroid, fallback, pixel_scale
        )
        lines.append(line)
        locked.append(is_locked)
        trusted.append(is_trusted)

    if fallback_lines is not None:
        # Guarda de corpo rígido: se >= 2 outras arestas confirmam que a
        # folha continua parada, uma única aresta que salte em ângulo ou
        # posição (ex.: manga/antebraço reto cobrindo metade de um lado)
        # não pode girar sozinha nem contaminar a memória.
        steady = []
        jumped = []
        for i in range(4):
            prev = fallback_lines[i]
            if prev is None or not trusted[i]:
                continue
            mid = 0.5 * (
                quad[i].astype(np.float64) + quad[(i + 1) % 4].astype(np.float64)
            )
            # Mede deslocamento no ponto médio da aresta nova contra a reta anterior.
            dir_cur, pt_cur = lines[i]
            proj_mid = pt_cur + float((mid - pt_cur) @ dir_cur) * dir_cur
            ang = _line_angle_diff_deg(lines[i], prev)
            shf = _line_normal_shift_at(proj_mid, prev)
            if (
                ang <= _MEMORY_STATIONARY_ANGLE_DEG
                and shf <= _MEMORY_STATIONARY_SHIFT_PX * pixel_scale
            ):
                steady.append(i)
            elif (
                ang > _MEMORY_SINGLE_OUTLIER_ANGLE_DEG
                or shf > _MEMORY_SINGLE_OUTLIER_SHIFT_PX * pixel_scale
            ):
                dir_prev, pt_prev = prev
                norm_out = np.array([-dir_prev[1], dir_prev[0]], dtype=np.float64)
                norm_out /= max(float(np.linalg.norm(norm_out)), 1e-9)
                if float((centroid - pt_prev) @ norm_out) > 0:
                    norm_out = -norm_out
                shift_outward = float((proj_mid - pt_prev) @ norm_out)
                jumped.append((i, shift_outward, ang, proj_mid, norm_out))
        if len(steady) >= 2 and len(jumped) == 1:
            bad, shift_outward, ang, _proj_mid, _norm_out = jumped[0]
            # Desobstrucao/expansao genuina move a aresta para fora do centroide
            # (sem giro extremo); oclusao move para dentro ou gira no lugar.
            is_outward_reveal = (
                shift_outward > 2.0 * pixel_scale
                and ang <= _MEMORY_SINGLE_OUTLIER_ANGLE_DEG
            )
            if not is_outward_reveal:
                lines[bad] = fallback_lines[bad]
                locked[bad] = True
                trusted[bad] = False
        elif len(steady) >= 2 and len(jumped) == 2:
            # Classifica deslocamento para fora, para dentro e pivo puramente
            # angular no interior do trecho compartilhado (nao na extremidade,
            # onde uma borda revelada pode desentortar passando pelo canto velho).
            outward_edges = [entry for entry in jumped if entry[1] > 2.0 * pixel_scale]
            inward_edges = [entry for entry in jumped if entry[1] < -2.0 * pixel_scale]
            pivot_edges = []
            for entry in jumped:
                e_i, e_shift, e_ang, _e_mid, _e_norm = entry
                if (
                    abs(e_shift) <= 2.0 * pixel_scale
                    and e_ang > _MEMORY_SINGLE_OUTLIER_ANGLE_DEG
                ):
                    interior = _is_interior_pivot(
                        e_i,
                        lines[e_i],
                        quad[e_i],
                        quad[(e_i + 1) % 4],
                        fallback_lines,
                    )
                    if interior is not False:
                        pivot_edges.append(entry)
            if len(outward_edges) == 1 and len(pivot_edges) == 1:
                piv_i = pivot_edges[0][0]
                lines[piv_i] = fallback_lines[piv_i]
                locked[piv_i] = True
                trusted[piv_i] = False
            elif len(outward_edges) == 1 and len(inward_edges) == 1:
                out_i, out_shift, out_ang, out_mid, out_norm = outward_edges[0]
                in_i, in_shift, in_ang, in_mid, in_norm = inward_edges[0]
                opposite = (out_i + 2) % 4 == in_i
                angles_ok = (
                    out_ang <= _MEMORY_SINGLE_OUTLIER_ANGLE_DEG
                    and in_ang <= _MEMORY_SINGLE_OUTLIER_ANGLE_DEG
                )
                consistent_translation = False
                if opposite:
                    rect_ok = _rectified_translation_consistent(
                        fallback_lines,
                        out_i,
                        out_mid,
                        out_norm,
                        in_i,
                        in_mid,
                        in_norm,
                        new_lines=(lines[out_i], lines[in_i]),
                    )
                    if rect_ok is not None:
                        consistent_translation = rect_ok
                    elif angles_ok:
                        mag = max(abs(out_shift), abs(in_shift))
                        consistent_translation = abs(out_shift + in_shift) <= max(
                            12.0 * pixel_scale, 0.35 * mag
                        )
                if not consistent_translation:
                    lines[in_i] = fallback_lines[in_i]
                    locked[in_i] = True
                    trusted[in_i] = False

    refined = quad.copy()
    for i in range(4):
        corner = _intersect(lines[i - 1], lines[i])
        if corner is None:
            continue
        if locked[i - 1] or locked[i]:
            refined[i] = corner
        elif np.linalg.norm((corner - quad[i]) * scale) <= _FIT_MAX_DISPLACE_PX:
            refined[i] = corner
    refined = order_corners(refined)
    if not cv2.isContourConvex(refined.reshape(4, 1, 2).astype(np.float32)):
        return quad, lines, trusted
    return refined, lines, trusted


def _refine_inside(frame, corners):
    """Refino subpixel só nos cantos que estão dentro do quadro.

    Canto projetado para fora do quadro (página cortada) não tem
    vizinhança de imagem para refinar; ele fica como a reta o pôs.
    Cantos dentro usam o mesmo acolchoamento do detector clássico.
    """
    frame_h, frame_w = frame.shape[:2]
    inside = [
        i
        for i, (x, y) in enumerate(corners)
        if 0.0 <= x <= frame_w - 1 and 0.0 <= y <= frame_h - 1
    ]
    if not inside:
        return corners
    refined = corners.copy()
    refined[inside] = _paper._refine_corners(frame, corners[inside])
    return refined.astype(np.float32)


def _fit_quad_impl(
    mask_bin,
    frame,
    fallback_lines,
    scale_x=1.0,
    scale_y=1.0,
    *,
    contour=None,
    support_mask=None,
):
    """O ajuste completo sobre a máscara, com herança opcional.

    Retorna (quadrilatero ou None, retas usadas ou None, confiavel
    por aresta ou None). fallback_lines são as retas confiáveis da
    chamada anterior, oferecidas às arestas sem cobertura.
    support_mask é a região não-fundo que precisa cobrir o quad
    ajustado (cobertura do detector quando existe; a própria máscara
    caso contrário): ajuste que escapa para cima de fundo não sai.

    Contrato dos retornos vazios, lido pelo rastro de aquisição do
    detector (issue #105): passado o piso de área, (None, None,
    None) é exclusivamente veto de suporte. Um motivo novo de
    rejeição não pode sair assim — o detector o rotularia
    "unsupported" e precisa de sinalização própria.
    """
    if support_mask is None or support_mask.shape != mask_bin.shape:
        support_mask = mask_bin
    if contour is None:
        contour = _sheet_contour(mask_bin)
    if contour is None:
        return None, None, None
    frame_area = float(mask_bin.shape[0] * mask_bin.shape[1])
    if cv2.contourArea(contour) < frame_area * _MIN_CONTOUR_AREA_RATIO:
        return None, None, None
    scale = np.array([scale_x, scale_y], dtype=np.float32)
    quad = _approx_quad(contour)
    if quad is None:
        # Reserva de área mínima: sai sem reajuste nem refino, mas não
        # sai sem apoio — todas as explosões medidas nas capturas
        # oblíquas entraram por aqui, com o retângulo mínimo cobrindo
        # braço e chão além da máscara.
        quad = _min_area_quad(contour)
        if not _quad_supported(quad, support_mask):
            return None, None, None
        return quad * scale, None, None
    quad, lines, trusted = _refit_corners(
        contour, quad, mask_bin.shape[:2], fallback_lines, scale
    )
    if lines is None and not _quad_supported(quad, support_mask):
        # Refit abandonado por falta de pontos: o approx que sobrou
        # também não teve guarda de aresta; a mesma regra vale.
        return None, None, None
    quad = quad * scale
    if frame is not None:
        quad = _refine_inside(frame, quad)
    return quad.astype(np.float32), lines, trusted


def fit_quad(mask_bin, frame=None, *, scale_x=1.0, scale_y=1.0):
    """Ajusta o quadrilátero da folha sobre a máscara binária.

    Contornos e retas ficam na grade da máscara; scale_x/scale_y
    levam apenas os quatro cantos ao espaço do frame antes do refino.
    Retorna float32[4,2] TL/TR/BR/BL ou None quando a máscara não
    tem contorno ou é pequena demais para sustentar um quadrilátero.
    Se a escada de epsilon não fecha, vale a reserva de
    _min_area_quad, sem reajuste nem refino. Com o frame, os cantos
    do caminho principal que estão dentro do quadro ganham refino
    subpixel sobre a imagem. Esta função não tem estado: a memória
    de arestas entre chamadas mora no NeuralSheetDetector.
    """
    quad, _lines, _trusted = _fit_quad_impl(mask_bin, frame, None, scale_x, scale_y)
    return quad
