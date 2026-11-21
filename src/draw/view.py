"""Composicao da janela do app: camera e canvas lado a lado em estilo Studio / Glass.

Monta a interface com:
- Barra superior Studio (logo DRAW, indicador de captura ao vivo e titulos editoriais).
- Barra vertical compacta de atalhos a esquerda da camera ([S], [C], [F], [Q]).
- Painel central com viewport dedicado para a visao da camera.
- Painel direito com viewport dedicado e paspatur para a folha digital, no mesmo padrao da camera.
- Rodape sutil com informacoes de formato e identidade.
- Quadros de aviso e erro em estilo card de vidro translucido.
"""

import math

import cv2
import numpy as np

from . import config
from .ink import render as ink_render

# Cores do sistema de design (BGR para OpenCV)
# Baseadas em design.md:
# Light canvas: warm off-white #f7f7f4 -> (244, 247, 247)
# Ink: soft near-black #22231f -> (31, 35, 34)
# Muted text: #777870 -> (112, 120, 119)
# Accent: pale paper yellow #e7dfa1 -> (161, 223, 231)
# Studio Dark: #171815 -> (21, 24, 23)
# Header / Footer surface: #1f201d -> (29, 32, 31)
# Subtle borders: #32342e -> (46, 52, 50)
COLOR_BG_STUDIO = (21, 24, 23)
COLOR_SURFACE_BAR = (29, 32, 31)
COLOR_BORDER_SUBTLE = (46, 52, 50)
COLOR_BORDER_CARD = (60, 66, 64)
COLOR_TEXT_BRIGHT = (244, 247, 247)
COLOR_TEXT_SOFT = (210, 215, 212)
COLOR_TEXT_MUTED = (112, 120, 119)
COLOR_ACCENT_YELLOW = (161, 223, 231)
COLOR_ACCENT_GREEN = (120, 210, 140)
COLOR_ACCENT_RED = (90, 100, 230)
COLOR_REC_ON = (50, 50, 235)
COLOR_REC_OFF = (30, 30, 80)

# Cores do painel direito: o mesmo tom escuro do estudio, sem moldura clara
COLOR_BG_CANVAS_SLOT = COLOR_BG_STUDIO
COLOR_CANVAS_SLOT_CONTAINER = (26, 28, 26)
COLOR_CANVAS_BORDER = COLOR_BORDER_SUBTLE
COLOR_PAPER_BAR = COLOR_SURFACE_BAR
COLOR_PAPER_BORDER = COLOR_BORDER_SUBTLE

_BLANK_SHEET = None


def fit(image, max_h=700):
    """Ajusta a altura de um quadro sem copiar quando ele ja cabe."""
    h, w = image.shape[:2]
    if h <= max_h:
        return image
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h))


def blank_sheet():
    global _BLANK_SHEET
    shape = (config.PAPER_H, config.PAPER_W)
    if _BLANK_SHEET is None or _BLANK_SHEET.shape[:2] != shape:
        _BLANK_SHEET = ink_render.render_canvas(np.zeros(shape, dtype=np.uint8))
    return _BLANK_SHEET


def _draw_pill(img, x, y, w, h, bg_color, border_color=None, radius=6, alpha=0.85):
    """Desenha um pill / retangulo arredondado translucido."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    ih, iw = img.shape[:2]
    if x < 0 or y < 0 or x + w > iw or y + h > ih or w <= 0 or h <= 0:
        return
    radius = min(radius, w // 2, h // 2)

    sub = img[y : y + h, x : x + w]
    overlay = sub.copy()

    cv2.rectangle(overlay, (radius, 0), (w - radius, h), bg_color, -1)
    cv2.rectangle(overlay, (0, radius), (w, h - radius), bg_color, -1)
    for cx, cy in [
        (radius, radius),
        (w - radius, radius),
        (radius, h - radius),
        (w - radius, h - radius),
    ]:
        cv2.circle(overlay, (cx, cy), radius, bg_color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)

    if border_color is not None:
        cv2.line(sub, (radius, 0), (w - radius, 0), border_color, 1, cv2.LINE_AA)
        cv2.line(
            sub, (radius, h - 1), (w - radius, h - 1), border_color, 1, cv2.LINE_AA
        )
        cv2.line(sub, (0, radius), (0, h - radius), border_color, 1, cv2.LINE_AA)
        cv2.line(
            sub, (w - 1, radius), (w - 1, h - radius), border_color, 1, cv2.LINE_AA
        )


_STUDIO_TEMPLATES = {}


_BASE_W, _BASE_H = 1600, 640


def _studio_template(cam_shape, canvas_shape, show_base, size=(_BASE_W, _BASE_H)):
    """Cromagem estatica da janela e geometria dos dois viewports.

    Header, rodape, botoes e molduras nao mudam entre quadros: sao
    desenhados uma vez por combinacao de (forma da camera, forma do
    canvas, show_base, tamanho) e copiados a cada quadro. Sem o cache,
    compor a vista redesenhava textos e pilulas a cada quadro do laco.
    O layout nasce no tamanho pedido: medidas, fontes e espessuras
    escalam com min(w/1600, h/640), sem esticar uma imagem pronta.
    """
    key = (cam_shape, canvas_shape, show_base, size)
    cached = _STUDIO_TEMPLATES.get(key)
    if cached is not None:
        return cached

    target_w, target_h = size
    sc = min(target_w / _BASE_W, target_h / _BASE_H)

    def k(v):
        return max(1, int(round(v * sc)))

    header_h = k(44)
    footer_h = k(36)
    panel_h = target_h - header_h - footer_h - k(8)
    panel_y = header_h + k(4)
    split_x = int(target_w * 0.7)

    # Geometria dos viewports (so depende das formas de entrada)
    cam_area_x = k(12)
    cam_area_w = split_x - cam_area_x - k(12)
    cam_h, cam_w = cam_shape[:2]
    aspect_cam = cam_w / max(1, cam_h)
    inner_cam_h = panel_h - k(8)
    inner_cam_w = int(inner_cam_h * aspect_cam)
    if inner_cam_w > cam_area_w - k(8):
        inner_cam_w = cam_area_w - k(8)
        inner_cam_h = int(inner_cam_w / aspect_cam)
    cam_x = cam_area_x + (cam_area_w - inner_cam_w) // 2
    cam_y = panel_y + (panel_h - inner_cam_h) // 2

    canvas_area_x = split_x + k(12)
    canvas_area_w = target_w - canvas_area_x - k(12)
    canvas_h, canvas_w = canvas_shape[:2]
    aspect_sheet = canvas_w / max(1, canvas_h)
    inner_sheet_h = panel_h - k(8)
    inner_sheet_w = int(inner_sheet_h * aspect_sheet)
    if inner_sheet_w > canvas_area_w - k(16):
        inner_sheet_w = canvas_area_w - k(16)
        inner_sheet_h = int(inner_sheet_w / aspect_sheet)
    sheet_x = canvas_area_x + (canvas_area_w - inner_sheet_w) // 2
    sheet_y = panel_y + (panel_h - inner_sheet_h) // 2

    geometry = {
        "panel_y": panel_y,
        "panel_h": panel_h,
        "split_x": split_x,
        "cam_area_x": cam_area_x,
        "cam_area_w": cam_area_w,
        "cam_x": cam_x,
        "cam_y": cam_y,
        "inner_cam_w": inner_cam_w,
        "inner_cam_h": inner_cam_h,
        "canvas_area_x": canvas_area_x,
        "canvas_area_w": canvas_area_w,
        "sheet_x": sheet_x,
        "sheet_y": sheet_y,
        "inner_sheet_w": inner_sheet_w,
        "inner_sheet_h": inner_sheet_h,
        "scale": sc,
    }

    studio = np.empty((target_h, target_w, 3), dtype=np.uint8)
    # Lado esquerdo: dark studio
    studio[:, :split_x] = COLOR_BG_STUDIO
    # Lado direito: mesmo fundo escuro, a folha digital destaca sozinha
    studio[:, split_x:] = COLOR_BG_CANVAS_SLOT

    # 1. Barra Superior (Header Studio)
    cv2.rectangle(studio, (0, 0), (split_x, header_h), COLOR_SURFACE_BAR, -1)
    cv2.line(
        studio, (0, header_h), (split_x, header_h), COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA
    )
    cv2.rectangle(studio, (split_x, 0), (target_w, header_h), COLOR_PAPER_BAR, -1)
    cv2.line(
        studio,
        (split_x, header_h),
        (target_w, header_h),
        COLOR_PAPER_BORDER,
        1,
        cv2.LINE_AA,
    )
    # Divisor vertical entre camera e canvas
    cv2.line(
        studio,
        (split_x, header_h),
        (split_x, target_h),
        COLOR_BORDER_SUBTLE,
        1,
        cv2.LINE_AA,
    )
    # Logo DRAW e badge STUDIO. Sem indicador de "ao vivo": o estado da
    # captura nao e conhecido aqui, e um selo estatico seguiria dizendo
    # que esta ao vivo sobre os quadros de erro/fonte morta.
    # 2. Barra Inferior (Footer limpo)
    cv2.rectangle(
        studio, (0, target_h - footer_h), (split_x, target_h), COLOR_SURFACE_BAR, -1
    )
    cv2.line(
        studio,
        (0, target_h - footer_h),
        (split_x, target_h - footer_h),
        COLOR_BORDER_SUBTLE,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        studio,
        (split_x, target_h - footer_h),
        (target_w, target_h),
        COLOR_PAPER_BAR,
        -1,
    )
    cv2.line(
        studio,
        (split_x, target_h - footer_h),
        (target_w, target_h - footer_h),
        COLOR_PAPER_BORDER,
        1,
        cv2.LINE_AA,
    )

    # 4. Molduras dos viewports e sombra da folha
    cv2.rectangle(
        studio,
        (cam_area_x, panel_y),
        (cam_area_x + cam_area_w, panel_y + panel_h),
        (26, 28, 26),
        -1,
    )
    cv2.rectangle(
        studio,
        (cam_area_x, panel_y),
        (cam_area_x + cam_area_w, panel_y + panel_h),
        COLOR_BORDER_SUBTLE,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        studio,
        (canvas_area_x, panel_y),
        (canvas_area_x + canvas_area_w, panel_y + panel_h),
        COLOR_CANVAS_SLOT_CONTAINER,
        -1,
    )
    cv2.rectangle(
        studio,
        (canvas_area_x, panel_y),
        (canvas_area_x + canvas_area_w, panel_y + panel_h),
        COLOR_CANVAS_BORDER,
        1,
        cv2.LINE_AA,
    )

    if len(_STUDIO_TEMPLATES) >= 4:
        _STUDIO_TEMPLATES.clear()
    _STUDIO_TEMPLATES[key] = (studio, geometry)
    return studio, geometry


FOOTER_BUTTONS = (("[S]", "Salvar", "s"), ("[F]", "Frame", "f"), ("[Q]", "Sair", "q"))
_BASE_BUTTON = ("[C]", "Base", "c")


def footer_buttons(show_base=False):
    """Botoes do rodape: (rotulo da tecla, texto, tecla que o botao aciona)."""
    if not show_base:
        return FOOTER_BUTTONS
    return (FOOTER_BUTTONS[0], _BASE_BUTTON) + FOOTER_BUTTONS[1:]


def button_rects(size, show_base=False):
    """Retangulos (x, y, w, h) dos botoes do rodape no tamanho dado."""
    target_w, target_h = size
    sc = min(target_w / _BASE_W, target_h / _BASE_H)

    def k(v):
        return max(1, int(round(v * sc)))

    footer_h, btn_h, bw = k(36), k(22), k(94)
    y = target_h - footer_h + (footer_h - btn_h) // 2
    return [
        (k(12) + i * (bw + k(8)), y, bw, btn_h)
        for i in range(len(footer_buttons(show_base)))
    ]


def _round_mask(shape, x, y, w, h, radius):
    mask = np.zeros(shape, dtype=np.uint8)
    radius = max(0, min(radius, w // 2, h // 2))
    cv2.rectangle(mask, (x + radius, y), (x + w - radius, y + h), 255, -1)
    cv2.rectangle(mask, (x, y + radius), (x + w, y + h - radius), 255, -1)
    for cx, cy in (
        (x + radius, y + radius),
        (x + w - radius, y + radius),
        (x + radius, y + h - radius),
        (x + w - radius, y + h - radius),
    ):
        cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    return mask


def _mix(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b, strict=True))


def _draw_footer_button(studio, rect, key_label, label, p, pressed, sc):
    """Botao do rodape com hover: sobe, ganha sombra e um brilho amarelo.

    p (0..1) e o progresso do hover. O brilho nasce no centro e cresce ate
    as bordas; com o botao pressionado ele desce de volta e acende mais.
    """

    def k(v):
        return max(1, int(round(v * sc)))

    x, y, w, h = rect
    th = max(1, int(round(sc)))
    ih, iw = studio.shape[:2]
    lift = 0 if pressed else int(round(2 * sc * p))
    top = y - lift
    radius = k(5)

    if p > 0.01:
        margin = k(8)
        x0, x1 = max(0, x - margin), min(iw, x + w + margin)
        y0, y1 = max(0, top - margin), min(ih, y + h + margin)
        roi = studio[y0:y1, x0:x1]
        shadow = _round_mask(roi.shape[:2], x - x0, y - y0 + k(2), w, h, radius)
        size = k(4) * 2 + 1
        shadow = cv2.GaussianBlur(shadow, (size, size), 0)
        alpha = (shadow.astype(np.float32) / 255.0 * 0.5 * p)[..., None]
        roi[:] = (roi * (1.0 - alpha)).astype(np.uint8)

    body = _mix((35, 38, 36), (44, 47, 44), p)
    border = _mix((58, 64, 61), COLOR_ACCENT_YELLOW, p * 0.55)
    _draw_pill(studio, x, top, w, h, body, border, radius=radius, alpha=1.0)

    if p > 0.01:
        mask = _round_mask((h, w), 0, 0, w, h, radius).astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
        reach = 0.2 + 1.4 * p
        strength = 0.3 * min(1.0, p * 2.0) * (1.6 if pressed else 1.0)
        glow = np.clip(1.0 - d / reach, 0.0, 1.0) ** 1.5 * strength * mask
        sub = studio[top : top + h, x : x + w].astype(np.float32)
        accent = np.array(COLOR_ACCENT_YELLOW, dtype=np.float32)
        sub = sub * (1.0 - glow[..., None]) + accent * glow[..., None]
        studio[top : top + h, x : x + w] = sub.astype(np.uint8)

    cv2.putText(
        studio,
        key_label,
        (x + k(9), top + k(15)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.38 * sc,
        COLOR_ACCENT_YELLOW,
        th,
        cv2.LINE_AA,
    )
    cv2.putText(
        studio,
        label,
        (x + k(40), top + k(15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38 * sc,
        _mix(COLOR_TEXT_SOFT, COLOR_TEXT_BRIGHT, p),
        th,
        cv2.LINE_AA,
    )


def _wordmark_geometry(size):
    """Posicao do nome DRAW no cabecalho: (x, base, largura, topo, escala)."""
    target_w, target_h = size
    sc = min(target_w / _BASE_W, target_h / _BASE_H)
    th = max(1, int(round(sc)))
    header_h = max(1, int(round(44 * sc)))
    (w, h), _ = cv2.getTextSize("DRAW", cv2.FONT_HERSHEY_DUPLEX, 0.95 * sc, th)
    x = target_w // 2 - w // 2
    base = (header_h + h) // 2 - max(1, int(round(2 * sc)))
    return x, base, w, base - h, sc


def wordmark_rect(size):
    """Retangulo (x, y, w, h) justo do nome DRAW e do fio sob ele."""
    x, base, w, top, sc = _wordmark_geometry(size)
    pad = max(1, int(round(4 * sc)))
    y0 = max(0, top - pad)
    return x - pad, y0, w + 2 * pad, base + max(1, int(round(6 * sc))) + pad - y0


def _draw_wordmark(studio, size, p, origin):
    """Nome DRAW centralizado, com fio por baixo e amarelo no hover.

    p (0..1) e o progresso do hover; o amarelo nasce em origin (onde o
    mouse tocou) e se espalha por dentro das letras e do fio.
    """
    x, base, w, top, sc = _wordmark_geometry(size)
    th = max(1, int(round(sc)))
    pad = max(2, int(round(3 * sc)))
    line_y = base + max(1, int(round(6 * sc)))
    x0, x1 = max(0, x - pad), min(studio.shape[1], x + w + pad)
    y0, y1 = max(0, top - pad), line_y + pad

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.putText(
        mask,
        "DRAW",
        (x - x0, base - y0),
        cv2.FONT_HERSHEY_DUPLEX,
        0.95 * sc,
        255,
        th,
        cv2.LINE_AA,
    )
    cv2.line(
        mask, (x - x0, line_y - y0), (x + w - x0, line_y - y0), 120, 1, cv2.LINE_AA
    )
    alpha = (mask.astype(np.float32) / 255.0)[..., None]

    white = np.array(COLOR_TEXT_BRIGHT, dtype=np.float32)
    accent = np.array(COLOR_ACCENT_YELLOW, dtype=np.float32)
    if p > 0.01:
        ox, oy = origin if origin is not None else (x + w // 2, top)
        reach = (max(abs(ox - x), abs(x + w - ox)) + 1) * 1.1 * p
        soft = max(1.0, 36 * sc)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dist = np.hypot(xx - ox, (yy - oy) * 1.4)
        cover = np.clip((reach - dist) / soft, 0.0, 1.0)[..., None]
        color = white * (1.0 - cover) + accent * cover
    else:
        color = white
    sub = studio[y0:y1, x0:x1].astype(np.float32)
    studio[y0:y1, x0:x1] = (sub * (1.0 - alpha) + color * alpha).astype(np.uint8)


def camera_menu_layout(size, cam_shape, canvas_shape, count, first=0):
    """Geometria da bolinha de cameras e do menu que sobe dela.

    A bolinha fica no canto inferior direito da area escura da camera,
    fora da imagem; o menu abre acima dela, alinhado a direita. O menu
    mostra so as linhas que cabem acima da bolinha (visible); first e a
    primeira linha visivel, para rolar a lista. Devolve retangulos
    (x, y, w, h).
    """
    geo = _studio_template(cam_shape, canvas_shape, False, tuple(size))[1]
    sc = geo["scale"]

    def k(v):
        return max(1, int(round(v * sc)))

    r = k(8)
    cx = geo["cam_area_x"] + geo["cam_area_w"] - k(20)
    cy = geo["panel_y"] + geo["panel_h"] - k(20)
    width, row_h, pad = k(200), k(28), k(6)
    bottom = cy - r - k(10)
    fit = max(1, (bottom - (geo["panel_y"] + k(6)) - 2 * pad) // row_h)
    count = max(1, count)
    visible = min(count, fit)
    first = max(0, min(first, count - visible))
    height = visible * row_h + 2 * pad
    left = cx + r - width
    top = bottom - height
    rows = [
        (left + pad, top + pad + i * row_h, width - 2 * pad, row_h)
        for i in range(visible)
    ]
    scroll_up = scroll_down = None
    if count > visible:
        arrow_w = k(22)
        ax = left + width - pad - arrow_w
        scroll_up = (ax, top + pad, arrow_w, row_h)
        scroll_down = (ax, top + pad + (visible - 1) * row_h, arrow_w, row_h)
    return {
        "dot": (cx, cy, r),
        "card": (left, top, width, height),
        "rows": rows,
        "scale": sc,
        "first": first,
        "visible": visible,
        "count": count,
        "scroll_up": scroll_up,
        "scroll_down": scroll_down,
    }


def point_in_dot(layout, pos):
    cx, cy, r = layout["dot"]
    reach = r + max(1, int(round(4 * layout["scale"])))
    return (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2 <= reach**2


def point_in_gear(layout, pos):
    cx, cy, r = layout["gear"]
    reach = r + max(1, int(round(4 * layout["scale"])))
    return (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2 <= reach**2


def point_in_card(layout, pos):
    x, y, w, h = layout["card"]
    return x <= pos[0] < x + w and y <= pos[1] < y + h


def scroll_hit(layout, pos):
    """-1 na seta de cima, +1 na de baixo, None fora das setas."""
    for key, step in (("scroll_up", -1), ("scroll_down", 1)):
        rect = layout.get(key)
        if rect and rect[0] <= pos[0] < rect[0] + rect[2]:
            if rect[1] <= pos[1] < rect[1] + rect[3]:
                return step
    return None


def row_at(layout, pos):
    if scroll_hit(layout, pos) is not None:
        return None
    for i, (x, y, w, h) in enumerate(layout["rows"]):
        if x <= pos[0] < x + w and y <= pos[1] < y + h:
            return i
    return None


def _draw_camera_dot(studio, layout, hover, opened):
    """Bolinha minimalista de cameras; o centro acende em amarelo no hover."""
    cx, cy, r = layout["dot"]
    sc = layout["scale"]
    th = max(1, int(round(sc)))
    x0, y0 = cx - r - 2, cy - r - 2
    size = 2 * r + 5
    if x0 < 0 or y0 < 0 or x0 + size > studio.shape[1] or y0 + size > studio.shape[0]:
        return
    roi = studio[y0 : y0 + size, x0 : x0 + size]
    overlay = roi.copy()
    c = (cx - x0, cy - y0)
    cv2.circle(overlay, c, r, (20, 22, 21), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, roi, 0.15, 0, roi)
    strength = max(hover, 1.0 if opened else 0.0)
    ring = _mix((95, 102, 100), COLOR_ACCENT_YELLOW, strength * 0.8)
    cv2.circle(roi, c, r, ring, th, cv2.LINE_AA)
    inner = max(2, int(round(r * (0.34 + 0.12 * strength))))
    cv2.circle(
        roi,
        c,
        inner,
        _mix((190, 196, 194), COLOR_ACCENT_YELLOW, strength),
        -1,
        cv2.LINE_AA,
    )


def _menu_card(studio, card, sc, progress):
    """Cartao escuro de menu com sombra e fio; devolve o deslocamento vertical."""

    def k(v):
        return max(1, int(round(v * sc)))

    ih, iw = studio.shape[:2]
    left, top, width, height = card
    dy = int(round(k(6) * (1.0 - progress)))
    margin = k(10)
    x0, y0 = max(0, left - margin), max(0, top + dy - margin)
    x1, y1 = min(iw, left + width + margin), min(ih, top + dy + height + margin)
    if x1 <= x0 or y1 <= y0:
        return dy
    roi = studio[y0:y1, x0:x1]
    shape = roi.shape[:2]
    radius = k(8)
    body = _round_mask(shape, left - x0, top + dy - y0, width, height, radius)
    ksize = k(5) * 2 + 1
    shadow = _round_mask(shape, left - x0, top + dy - y0 + k(3), width, height, radius)
    shadow = cv2.GaussianBlur(shadow, (ksize, ksize), 0)
    a_shadow = (shadow.astype(np.float32) / 255.0 * 0.45 * progress)[..., None]
    out = roi.astype(np.float32) * (1.0 - a_shadow)
    a_body = (body.astype(np.float32) / 255.0 * 0.97 * progress)[..., None]
    out = out * (1.0 - a_body) + np.array((26, 28, 27), np.float32) * a_body
    edge = cv2.subtract(body, cv2.erode(body, np.ones((3, 3), np.uint8)))
    a_edge = (edge.astype(np.float32) / 255.0 * progress)[..., None]
    out = out * (1.0 - a_edge) + np.array(COLOR_BORDER_CARD, np.float32) * a_edge
    roi[:] = out.astype(np.uint8)
    return dy


def _row_glow(studio, rect, hover, progress):
    """Brilho amarelo leve que nasce no centro da linha sob o mouse."""
    rx, ry, rw, rh = rect
    if hover <= 0.01:
        return
    x0, y0 = max(0, rx), max(0, ry)
    x1, y1 = min(studio.shape[1], rx + rw), min(studio.shape[0], ry + rh)
    if x1 <= x0 or y1 <= y0:
        return
    sub = studio[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.hypot((xx - (rx + rw / 2)) / (rw / 2), (yy - (ry + rh / 2)) / (rh / 2))
    glow = np.clip(1.0 - d / (0.3 + 1.2 * hover), 0.0, 1.0) ** 1.5
    glow = (glow * 0.26 * min(1.0, hover * 2.0) * progress)[..., None]
    accent = np.array(COLOR_ACCENT_YELLOW, np.float32)
    studio[y0:y1, x0:x1] = (sub * (1.0 - glow) + accent * glow).astype(np.uint8)


def _draw_camera_menu(studio, layout, items, progress):
    """Menu minimalista com as cameras: sobe e aparece com o progresso 0..1."""
    if progress <= 0.01 or not items:
        return
    sc = layout["scale"]

    def k(v):
        return max(1, int(round(v * sc)))

    dy = _menu_card(studio, layout["card"], sc, progress)
    th = max(1, int(round(sc)))
    for (rx, ry, rw, rh), item in zip(layout["rows"], items, strict=False):
        ry += dy
        hp = item["hover"]
        _row_glow(studio, (rx, ry, rw, rh), hp, progress)
        color = _mix(COLOR_TEXT_SOFT, COLOR_TEXT_BRIGHT, hp)
        color = _mix((26, 28, 27), color, progress)
        if item["selected"]:
            cv2.circle(
                studio,
                (rx + k(10), ry + rh // 2),
                k(3),
                _mix((26, 28, 27), COLOR_ACCENT_YELLOW, progress),
                -1,
                cv2.LINE_AA,
            )
        cv2.putText(
            studio,
            item["label"],
            (rx + k(22), ry + rh // 2 + k(4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36 * sc,
            color,
            th,
            cv2.LINE_AA,
        )
    _draw_scroll_arrows(studio, layout, dy, progress)


def _draw_scroll_arrows(studio, layout, dy, progress):
    """Setas de rolagem da lista; apagadas quando nao ha mais para onde ir."""
    bg = (26, 28, 27)
    half = max(2, int(round(4 * layout["scale"])))
    first, visible, count = layout["first"], layout["visible"], layout["count"]
    for key, direction in (("scroll_up", -1), ("scroll_down", 1)):
        rect = layout.get(key)
        if not rect:
            continue
        x, y, w, h = rect
        at_end = first == 0 if direction < 0 else first + visible >= count
        color = _mix(bg, (70, 75, 73) if at_end else COLOR_ACCENT_YELLOW, progress)
        cx, cy = x + w // 2, y + dy + h // 2
        tip_y, base_y = (
            (cy - half, cy + half) if direction < 0 else (cy + half, cy - half)
        )
        pts = np.array(
            [(cx, tip_y), (cx - half, base_y), (cx + half, base_y)], np.int32
        )
        cv2.fillPoly(studio, [pts], color, cv2.LINE_AA)


def settings_menu_layout(size, count):
    """Engrenagem no canto inferior direito do rodape e o menu acima dela."""
    target_w, target_h = size
    sc = min(target_w / _BASE_W, target_h / _BASE_H)

    def k(v):
        return max(1, int(round(v * sc)))

    footer_h = k(36)
    r = k(9)
    cx = target_w - k(10) - r
    cy = target_h - footer_h // 2
    width, row_h, pad = k(210), k(28), k(6)
    height = max(1, count) * row_h + 2 * pad
    left = cx + r - width
    top = target_h - footer_h - k(8) - height
    rows = [
        (left + pad, top + pad + i * row_h, width - 2 * pad, row_h)
        for i in range(max(1, count))
    ]
    return {
        "gear": (cx, cy, r),
        "card": (left, top, width, height),
        "rows": rows,
        "scale": sc,
    }


def _draw_gear(studio, layout, hover, opened):
    """Engrenagem minimalista; o amarelo nasce no centro e fica so dentro dela."""
    cx, cy, r = layout["gear"]
    strength = max(hover, 1.0 if opened else 0.0)
    half = r + 2
    x0, y0 = cx - half, cy - half
    size = 2 * half + 1
    if x0 < 0 or y0 < 0 or x0 + size > studio.shape[1] or y0 + size > studio.shape[0]:
        return
    shift = 4
    pts = []
    for tooth in range(8):
        base = 2 * math.pi * tooth / 8
        for da, radius in ((-0.27, 0.72), (-0.15, 1.0), (0.15, 1.0), (0.27, 0.72)):
            a = base + da
            pts.append(
                (
                    int(round((half + r * radius * math.cos(a)) * (1 << shift))),
                    int(round((half + r * radius * math.sin(a)) * (1 << shift))),
                )
            )
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255, cv2.LINE_AA, shift)
    hole = max(1, int(round(r * 0.38)))
    cv2.circle(mask, (half, half), hole, 0, -1, cv2.LINE_AA)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.hypot(xx - half, yy - half) / max(1.0, float(r))
    cover = np.clip((0.15 + 1.35 * strength - dist) / 0.45, 0.0, 1.0)[..., None]
    gray = np.array((150, 156, 154), np.float32)
    accent = np.array(COLOR_ACCENT_YELLOW, np.float32)
    color = gray * (1.0 - cover) + accent * cover
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    roi = studio[y0 : y0 + size, x0 : x0 + size]
    roi[:] = (roi.astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)


def _smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _draw_label(studio, text, org, scale, th, color, wave=None):
    """Texto com uma onda amarela opcional que corre por dentro das letras.

    wave = (centro 0..1 ao longo do texto, forca 0..1).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, tht), baseline = cv2.getTextSize(text, font, scale, th)
    if wave is None or wave[1] <= 0.01:
        cv2.putText(studio, text, org, font, scale, color, th, cv2.LINE_AA)
        return
    pad = 2
    x0, y0 = org[0] - pad, org[1] - tht - pad
    x1, y1 = org[0] + tw + pad, org[1] + baseline + pad
    if x0 < 0 or y0 < 0 or x1 > studio.shape[1] or y1 > studio.shape[0]:
        cv2.putText(studio, text, org, font, scale, color, th, cv2.LINE_AA)
        return
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.putText(mask, text, (pad, tht + pad), font, scale, 255, th, cv2.LINE_AA)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    u = (np.arange(x1 - x0, dtype=np.float32) - pad) / max(1, tw)
    cover = np.clip(1.0 - np.abs(u - wave[0]) / 0.3, 0.0, 1.0) * wave[1]
    base = np.array(color, np.float32)
    accent = np.array(COLOR_ACCENT_YELLOW, np.float32)
    tint = base[None, :] * (1.0 - cover[:, None]) + accent[None, :] * cover[:, None]
    sub = studio[y0:y1, x0:x1].astype(np.float32)
    studio[y0:y1, x0:x1] = (sub * (1.0 - alpha) + tint[None, :, :] * alpha).astype(
        np.uint8
    )


def _draw_toggle(studio, rect, p, direction, progress, bg):
    """Interruptor: o botao desliza suave e deixa um rastro amarelo atras."""
    tx, ty, tw, tg = rect
    radius = tg // 2
    ex = _smooth(p)
    span = tw - 2 * radius
    knob_x = tx + radius + ex * span
    sub = studio[ty : ty + tg + 1, tx : tx + tw + 1]
    mask = _round_mask(sub.shape[:2], 0, 0, tw, tg, radius).astype(np.float32) / 255.0
    cols = np.arange(tw + 1, dtype=np.float32) + tx
    ahead = np.clip((knob_x - cols) / max(1.0, tw * 0.9), 0.0, 1.0)
    fill = ex * (0.5 + 0.5 * (1.0 - ahead)) * (cols <= knob_x)
    if direction < 0 and 0.01 < ex < 0.99:
        tail = np.clip(1.0 - (cols - knob_x) / (tw * 0.7), 0.0, 1.0) * (1.0 - ex)
        fill = np.maximum(fill, tail * (cols > knob_x))
    off = np.array((52, 57, 55), np.float32)
    accent = np.array(COLOR_ACCENT_YELLOW, np.float32)
    track = (
        off[None, None, :] * (1.0 - fill[None, :, None])
        + accent[None, None, :] * fill[None, :, None]
    )
    track = track * progress + np.array(bg, np.float32) * (1.0 - progress)
    a = (mask * progress)[..., None]
    sub[:] = (sub.astype(np.float32) * (1.0 - a) + track * a).astype(np.uint8)
    knob = _mix((170, 176, 174), (30, 32, 31), ex)
    cv2.circle(
        studio,
        (int(round(knob_x)), ty + radius),
        max(1, radius - max(1, tg // 7)),
        _mix(bg, knob, progress),
        -1,
        cv2.LINE_AA,
    )


def _draw_settings_menu(studio, layout, items, progress):
    """Menu de opcoes: cada linha tem um interruptor minimalista."""
    if progress <= 0.01 or not items:
        return
    sc = layout["scale"]

    def k(v):
        return max(1, int(round(v * sc)))

    dy = _menu_card(studio, layout["card"], sc, progress)
    th = max(1, int(round(sc)))
    bg = (26, 28, 27)
    for (rx, ry, rw, rh), item in zip(layout["rows"], items, strict=False):
        ry += dy
        hp = item["hover"]
        _row_glow(studio, (rx, ry, rw, rh), hp, progress)
        color = _mix(bg, _mix(COLOR_TEXT_SOFT, COLOR_TEXT_BRIGHT, hp), progress)
        react = item.get("react", 1.0)
        direction = item.get("dir", 1)
        wave = None
        if react < 1.0:
            center = react if direction > 0 else 1.0 - react
            wave = (center, math.sin(math.pi * react) * progress)
        _draw_label(
            studio,
            item["label"],
            (rx + k(12), ry + rh // 2 + k(4)),
            0.36 * sc,
            th,
            color,
            wave,
        )
        tw, tg = k(26), k(14)
        tx, ty = rx + rw - k(12) - tw, ry + (rh - tg) // 2
        on_p = item.get("on_p", 1.0 if item["on"] else 0.0)
        _draw_toggle(studio, (tx, ty, tw, tg), on_p, direction, progress, bg)


def _draw_hud_footer(studio, geo, sc, lines, size):
    """Painel numa linha so no rodape, quando nao ha margem ao lado da camera."""

    def k(v):
        return max(1, int(round(v * sc)))

    font, th = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(sc)))
    left = button_rects(size)[-1]
    left = left[0] + left[2] + k(24)
    right = settings_menu_layout(size, 1)["gear"]
    right = right[0] - right[2] - k(20)
    segments = []
    for line in lines:
        segments.extend(line)
        segments.append(("    ", COLOR_TEXT_MUTED))
    base_fs = 0.38 * sc

    def width(fs):
        return sum(cv2.getTextSize(t, font, fs, th)[0][0] for t, _ in segments)

    needed = width(base_fs)
    factor = min(1.0, (right - left) / needed) if needed > 0 else 1.0
    if factor < 0.6:
        return
    fs = base_fs * factor
    tx, ty = left, size[1] - k(36) // 2 + k(4)
    for text, color in segments:
        cv2.putText(studio, text, (tx, ty), font, fs, color, th, cv2.LINE_AA)
        tx += cv2.getTextSize(text, font, fs, th)[0][0]


def _draw_hud(studio, geo, sc, lines, size):
    """Informacoes no canto superior esquerdo da area escura da camera.

    O painel nunca passa da margem escura a esquerda da imagem: se o texto
    nao couber, a fonte diminui; se a margem for pequena demais (camera
    larga, tela cheia), o painel vai para o rodape numa linha so.
    """
    if not lines:
        return

    def k(v):
        return max(1, int(round(v * sc)))

    font, th = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(sc)))
    x = geo["cam_area_x"] + k(10)
    y = geo["panel_y"] + k(10)
    pad = k(8)
    avail = geo["cam_x"] - k(8) - x - 2 * pad
    base_fs = 0.38 * sc

    def width(fs):
        return max(
            sum(cv2.getTextSize(t, font, fs, th)[0][0] for t, _ in line)
            for line in lines
        )

    needed = width(base_fs)
    factor = min(1.0, avail / needed) if needed > 0 else 1.0
    if factor < 0.6:
        _draw_hud_footer(studio, geo, sc, lines, size)
        return
    fs = base_fs * factor
    line_h = max(1, int(round(18 * sc * factor)))
    _draw_pill(
        studio,
        x,
        y,
        width(fs) + 2 * pad,
        len(lines) * line_h + pad,
        (20, 22, 21),
        (46, 52, 50),
        radius=k(6),
        alpha=0.82,
    )
    for i, line in enumerate(lines):
        tx = x + pad
        ty = y + pad // 2 + (i + 1) * line_h - k(5)
        for text, color in line:
            cv2.putText(studio, text, (tx, ty), font, fs, color, th, cv2.LINE_AA)
            tx += cv2.getTextSize(text, font, fs, th)[0][0]


def _hud_lines(studio, hud):
    """Linhas do painel a partir dos valores ligados em hud."""
    muted, bright = COLOR_TEXT_MUTED, COLOR_TEXT_BRIGHT
    lines = []
    probe = hud.get("probe")
    if probe is not None:
        x, y = probe
        h, w = studio.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            b, g, r = (int(v) for v in studio[y, x])
            lines.append([(f"x={x} y={y}", muted)])
            lines.append(
                [
                    (f"R:{r} ", (90, 100, 230)),
                    (f"G:{g} ", (120, 210, 140)),
                    (f"B:{b}", (230, 150, 90)),
                ]
            )
        else:
            lines.append([("x - y -", muted)])
            lines.append([("R - G - B -", muted)])
    if hud.get("latency") is not None:
        lines.append([("Latencia  ", muted), (f"{hud['latency']:.0f} ms", bright)])
    if hud.get("fps") is not None:
        lines.append([("FPS  ", muted), (f"{hud['fps']:.1f}", bright)])
    return lines


def _resize(image, width, height):
    """Resize de qualidade: AREA ao reduzir, CUBIC ao ampliar."""
    h, w = image.shape[:2]
    shrink = width * height < w * h
    interp = cv2.INTER_AREA if shrink else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interp)


# Cache de uma entrada do resize do canvas do painel. O canvas so
# muda quando a trabalhadora publica (~4/s) enquanto compose_view
# roda a cada quadro de UI (~12-30/s): chaveado por
# (versao, destino, forma) os quadros sem publicacao reutilizam o
# ultimo resize. Uma entrada basta — a versao e monotonica e so o
# canvas mais novo e exibido.
_CANVAS_RESIZE = {}


def _resize_canvas(image, width, height, version):
    """Resize do canvas do painel, reutilizado entre quadros sem publicacao.

    version e o token de conteudo do canvas (ex.: (geracao, versao)
    da publicacao): a mesma versao garante os mesmos pixels. None
    desliga o cache — conteudo sem identidade propria (warp de
    marcadores, folha em branco) re-redimensiona a cada quadro.
    """
    if version is None:
        return _resize(image, width, height)
    key = (version, width, height, image.shape)
    if _CANVAS_RESIZE.get("key") != key:
        _CANVAS_RESIZE["key"] = key
        _CANVAS_RESIZE["img"] = _resize(image, width, height)
    return _CANVAS_RESIZE["img"]


def compose_view(
    camera_img,
    canvas_img,
    show_base=False,
    size=(_BASE_W, _BASE_H),
    rec_on=True,
    buttons=None,
    pressed=None,
    wordmark=(0.0, None),
    camera_menu=None,
    settings_menu=None,
    hud=None,
    canvas_version=None,
):
    """Janela unica: camera e folha digital em viewports dedicados simetricos.

    Tamanho padrao 1600x640; com size=(w, h) o layout inteiro e redesenhado
    nessa resolucao (janela maximizada ou em tela cheia), garantindo
    contraste, legibilidade, enquadramento simetrico da folha no mesmo
    padrao da camera e sem mutacao.
    rec_on acende ou apaga o ponto vermelho de gravacao da pilula CAMERA;
    o laco o alterna para o ponto piscar.
    buttons e o progresso de hover (0..1) de cada botao do rodape, na
    ordem de footer_buttons; wordmark e (progresso, origem) do hover no
    nome DRAW; camera_menu e o estado da bolinha de cameras (dot, opened,
    progress, items); pressed e o indice do botao sob o clique.
    show_base liga o botao [C] Base do rodape: a tecla so tem efeito no
    modo baseline (session_ink), entao fora dele o botao nao aparece.
    canvas_version e o token de conteudo de canvas_img para o cache do
    resize do painel (a mesma versao garante os mesmos pixels); None
    re-redimensiona a cada quadro.
    A cromagem estatica vem do cache de _studio_template; por quadro
    restam os dois resizes, as colagens e as bordas sobre as imagens.
    """
    template, geo = _studio_template(
        camera_img.shape, canvas_img.shape, show_base, tuple(size)
    )
    studio = template.copy()
    sc = geo["scale"]

    def k(v):
        return max(1, int(round(v * sc)))

    # Preview da camera em INTER_LINEAR: ~10x mais barato que AREA
    # no downscale e a diferenca e invisivel numa vista ao vivo
    # (nada daqui alimenta modelo ou tinta — issue #99).
    cam_resized = cv2.resize(
        camera_img,
        (geo["inner_cam_w"], geo["inner_cam_h"]),
        interpolation=cv2.INTER_LINEAR,
    )
    cam_x, cam_y = geo["cam_x"], geo["cam_y"]
    studio[cam_y : cam_y + geo["inner_cam_h"], cam_x : cam_x + geo["inner_cam_w"]] = (
        cam_resized
    )
    cv2.rectangle(
        studio,
        (cam_x, cam_y),
        (cam_x + geo["inner_cam_w"], cam_y + geo["inner_cam_h"]),
        (50, 56, 54),
        1,
        cv2.LINE_AA,
    )

    # Pilula CAMERA no canto superior direito da imagem: o canto esquerdo
    # e onde o app desenha o texto de status ("Folha detectada", "Mostre
    # uma folha...") sobre o quadro, e a pilula ali cobria o comeco dele.
    pill_x = cam_x + geo["inner_cam_w"] - k(102)
    pill_y = cam_y + k(12)
    _draw_pill(
        studio,
        pill_x,
        pill_y,
        k(90),
        k(22),
        (20, 22, 21),
        (60, 66, 64),
        radius=k(11),
        alpha=0.9,
    )
    cv2.circle(
        studio,
        (pill_x + k(10), pill_y + k(11)),
        k(3),
        COLOR_REC_ON if rec_on else COLOR_REC_OFF,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        studio,
        "CAMERA",
        (pill_x + k(20), pill_y + k(15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32 * sc,
        COLOR_TEXT_BRIGHT,
        max(1, int(round(sc))),
        cv2.LINE_AA,
    )

    sheet_resized = _resize_canvas(
        canvas_img,
        geo["inner_sheet_w"],
        geo["inner_sheet_h"],
        canvas_version,
    )
    sheet_x, sheet_y = geo["sheet_x"], geo["sheet_y"]
    studio[
        sheet_y : sheet_y + geo["inner_sheet_h"],
        sheet_x : sheet_x + geo["inner_sheet_w"],
    ] = sheet_resized
    cv2.rectangle(
        studio,
        (sheet_x, sheet_y),
        (sheet_x + geo["inner_sheet_w"], sheet_y + geo["inner_sheet_h"]),
        (50, 56, 54),
        1,
        cv2.LINE_AA,
    )

    _draw_wordmark(studio, size, *wordmark)

    menu = camera_menu or {}
    layout = camera_menu_layout(
        size,
        camera_img.shape,
        canvas_img.shape,
        menu.get("count", len(menu.get("items", ()))),
        menu.get("first", 0),
    )
    _draw_camera_dot(studio, layout, menu.get("dot", 0.0), menu.get("opened", False))
    _draw_camera_menu(studio, layout, menu.get("items", ()), menu.get("progress", 0.0))

    _draw_hud(studio, geo, sc, _hud_lines(studio, hud or {}), size)

    settings = settings_menu or {}
    gear = settings_menu_layout(size, len(settings.get("items", ())))
    _draw_gear(studio, gear, settings.get("gear", 0.0), settings.get("opened", False))
    _draw_settings_menu(
        studio, gear, settings.get("items", ()), settings.get("progress", 0.0)
    )

    for i, (rect, (key_label, label, _)) in enumerate(
        zip(button_rects(size, show_base), footer_buttons(show_base), strict=True)
    ):
        p = buttons[i] if buttons is not None and i < len(buttons) else 0.0
        _draw_footer_button(studio, rect, key_label, label, p, pressed == i, sc)

    return studio


def message_view(last_view, title, lines):
    """Aviso desenhado em card de vidro sobre a ultima vista ou fundo solido."""
    if last_view is not None:
        view = last_view.copy()
    else:
        view = np.full((640, 1280, 3), COLOR_BG_STUDIO, dtype=np.uint8)

    vh, vw = view.shape[:2]
    avail_w = 1120 if vw >= 1600 else vw
    card_w = min(680, avail_w - 60)
    card_h = min(280, vh - 60)
    cx = (avail_w - card_w) // 2
    cy = (vh - card_h) // 2

    # Card de vidro escuro
    _draw_pill(
        view,
        cx,
        cy,
        card_w,
        card_h,
        (28, 30, 29),
        COLOR_BORDER_CARD,
        radius=14,
        alpha=0.95,
    )

    # Badge de Aviso
    _draw_pill(
        view,
        cx + 36,
        cy + 28,
        100,
        24,
        (38, 42, 40),
        (80, 88, 85),
        radius=12,
        alpha=1.0,
    )
    cv2.circle(view, (cx + 48, cy + 40), 4, COLOR_ACCENT_RED, -1, cv2.LINE_AA)
    cv2.putText(
        view,
        "AVISO",
        (cx + 58, cy + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_BRIGHT,
        1,
        cv2.LINE_AA,
    )

    # Titulo
    cv2.putText(
        view,
        title,
        (cx + 36, cy + 86),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        COLOR_TEXT_BRIGHT,
        1,
        cv2.LINE_AA,
    )

    # Divisor sutil
    cv2.line(
        view,
        (cx + 36, cy + 104),
        (cx + card_w - 36, cy + 104),
        COLOR_BORDER_SUBTLE,
        1,
        cv2.LINE_AA,
    )

    # Linhas explicativas
    for idx, line in enumerate(lines):
        cv2.putText(
            view,
            line,
            (cx + 36, cy + 138 + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_TEXT_SOFT,
            1,
            cv2.LINE_AA,
        )

    # Instrucao de saida
    _draw_pill(
        view,
        cx + card_w - 180,
        cy + card_h - 44,
        144,
        24,
        (36, 40, 38),
        (65, 72, 70),
        radius=6,
        alpha=1.0,
    )
    cv2.putText(
        view,
        "[Q / ESC] Fechar",
        (cx + card_w - 168, cy + card_h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        COLOR_ACCENT_YELLOW,
        1,
        cv2.LINE_AA,
    )

    return view


def dead_source_view(last_view):
    """Quadro de aviso da fonte morta, desenhado sobre a ultima vista."""
    return message_view(
        last_view,
        "A fonte de video morreu",
        [
            "A camera parou de entregar quadros.",
            "Feche outros programas usando a camera e tente de novo.",
        ],
    )


def error_view(last_view, title, lines):
    """Quadro de aviso de falha do pipeline, no formato da vista."""
    return message_view(last_view, title, lines)
