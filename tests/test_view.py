"""Contrato de compose_view e dos quadros de aviso (draw/view.py).

compose_view é a única função de desenho que recebe o canvas do
chamador. O putText do rótulo "folha ao vivo" escrevia no array
recebido quando ele já estava na altura alvo (o fit devolve o
próprio array nesse caso): com um canvas de 640 de altura, compor a
vista pintava o canvas do chamador e o cache de módulo _BLANK_SHEET.
"""

import numpy as np

from draw import config
from draw import view as view_module


def _camera(height=720, width=1280):
    return np.full((height, width, 3), 120, dtype=np.uint8)


def test_view_has_the_fixed_size():
    view = view_module.compose_view(_camera(), np.full((1200, 850, 3), 255, np.uint8))
    assert view.shape == (640, 1600, 3)
    assert view.dtype == np.uint8


def test_canvas_at_target_height_is_not_mutated():
    canvas = np.full((640, 850, 3), 255, dtype=np.uint8)
    before = canvas.copy()
    view_module.compose_view(_camera(), canvas)
    assert np.array_equal(canvas, before), "a vista escreveu no canvas do chamador"


def test_canvas_smaller_than_the_target_is_not_mutated_either():
    canvas = np.full((300, 200, 3), 255, dtype=np.uint8)
    before = canvas.copy()
    view_module.compose_view(_camera(), canvas)
    assert np.array_equal(canvas, before)


def test_blank_sheet_cache_survives_composition():
    view_module._BLANK_SHEET = None
    blank = view_module.blank_sheet()
    view_module._BLANK_SHEET = blank
    view_module.compose_view(_camera(), blank)
    assert np.array_equal(view_module._BLANK_SHEET, view_module.blank_sheet())
    assert view_module.blank_sheet().shape == (
        config.current().height,
        config.current().width,
        3,
    )


def test_message_view_paints_over_a_copy():
    last = np.full((640, 1600, 3), 200, dtype=np.uint8)
    before = last.copy()
    view = view_module.message_view(last, "falhou", ["linha"])
    assert not np.array_equal(view, before)
    assert np.array_equal(last, before)


def test_message_view_without_a_last_frame_uses_the_view_size():
    view = view_module.message_view(None, "falhou", ["linha"])
    assert view.shape == (640, 1280, 3)


def test_base_button_only_appears_in_baseline_mode():
    cam = _camera()
    canvas = np.full((1200, 850, 3), 255, np.uint8)
    default = view_module.compose_view(cam, canvas)
    baseline = view_module.compose_view(cam, canvas, show_base=True)
    footer = slice(600, 640)
    assert not np.array_equal(default[footer], baseline[footer])
    # E o modo padrao continua estavel depois de o baseline ter rodado:
    # os templates em cache nao se contaminam entre modos.
    again = view_module.compose_view(cam, canvas)
    assert np.array_equal(default, again)


def test_composed_view_does_not_share_the_cached_template():
    cam = _camera()
    canvas = np.full((1200, 850, 3), 255, np.uint8)
    first = view_module.compose_view(cam, canvas)
    snapshot = first.copy()
    first[:] = 0  # o chamador pode pintar sobre a vista devolvida
    second = view_module.compose_view(cam, canvas)
    assert np.array_equal(second, snapshot)


def test_camera_resize_uses_linear_for_the_live_preview(monkeypatch):
    """Issue #99: o preview da câmera não paga INTER_AREA — LINEAR é
    ~10x mais barato no downscale e a diferença é invisível na vista
    ao vivo (o canvas segue com a interpolação de qualidade)."""
    import cv2

    interps = []
    real_resize = cv2.resize

    def spy(image, dsize, *args, **kwargs):
        interps.append(kwargs.get("interpolation"))
        return real_resize(image, dsize, *args, **kwargs)

    monkeypatch.setattr(cv2, "resize", spy)
    view_module.compose_view(_camera(), np.full((1200, 850, 3), 255, np.uint8))
    assert interps[0] == cv2.INTER_LINEAR  # câmera
    assert interps[1] == cv2.INTER_AREA  # canvas segue AREA no downscale


def test_canvas_resize_is_cached_per_publication_version(monkeypatch):
    """Issue #99: o canvas só muda por publicação; a mesma versão
    reusa o resize do quadro anterior e versão None sempre
    redimensiona."""
    calls = []
    real_resize = view_module._resize

    def spy(image, w, h):
        calls.append(1)
        return real_resize(image, w, h)

    monkeypatch.setattr(view_module, "_resize", spy)
    cam = _camera()
    canvas = np.full((1200, 850, 3), 255, np.uint8)
    view_module.compose_view(cam, canvas, canvas_version=(0, 3))
    view_module.compose_view(cam, canvas, canvas_version=(0, 3))
    assert len(calls) == 1  # o segundo quadro bebe do cache
    view_module.compose_view(cam, canvas, canvas_version=(0, 4))
    assert len(calls) == 2
    view_module.compose_view(cam, canvas)
    assert len(calls) == 3  # sem versão não há cache


def test_dead_source_view_uses_the_message_card():
    last = np.full((640, 1600, 3), 200, dtype=np.uint8)
    view = view_module.dead_source_view(last)
    assert view.shape == last.shape
    assert not np.array_equal(view, last)
    assert np.array_equal(last, np.full((640, 1600, 3), 200, dtype=np.uint8))


def test_error_view_paints_title_and_lines():
    view = view_module.error_view(None, "falhou", ["linha 1", "linha 2"])
    assert view.shape == (640, 1280, 3)


def test_camera_menu_layout_sits_inside_the_camera_corner():
    cam, canvas = _camera(), np.full((1200, 850, 3), 255, np.uint8)
    layout = view_module.camera_menu_layout((1600, 640), cam.shape, canvas.shape, 3)
    cx, cy, r = layout["dot"]
    x, y, w, h = layout["card"]
    assert 0 < cx < 1120 and 0 < cy < 640
    assert y + h < cy - r and x + w == cx + r
    assert len(layout["rows"]) == 3
    assert view_module.point_in_dot(layout, (cx, cy))
    assert not view_module.point_in_dot(layout, (cx - 60, cy))
    rx, ry, rw, rh = layout["rows"][1]
    assert view_module.row_at(layout, (rx + 2, ry + 2)) == 1
    assert view_module.row_at(layout, (cx, cy)) is None


def test_camera_menu_draws_without_mutating_inputs():
    cam, canvas = _camera(), np.full((1200, 850, 3), 255, np.uint8)
    closed = view_module.compose_view(cam, canvas)
    items = [{"label": "Camera 0", "selected": True, "hover": 0.5}]
    opened = view_module.compose_view(
        cam,
        canvas,
        camera_menu={"dot": 1.0, "opened": True, "progress": 1.0, "items": items},
    )
    assert closed.shape == opened.shape and (closed != opened).any()
    assert (cam == 120).all() and (canvas == 255).all()


def test_settings_menu_layout_and_hud_render():
    layout = view_module.settings_menu_layout((1600, 640), 3)
    cx, cy, r = layout["gear"]
    assert cx + r < 1600 and cy > 640 - 36
    assert view_module.point_in_gear(layout, (cx, cy))
    assert len(layout["rows"]) == 3 and layout["card"][1] + layout["card"][3] < 640 - 36
    cam, canvas = _camera(), np.full((1200, 850, 3), 255, np.uint8)
    base = view_module.compose_view(cam, canvas)
    items = [{"label": "FPS", "on": True, "hover": 0.0}] * 3
    with_ui = view_module.compose_view(
        cam,
        canvas,
        settings_menu={"gear": 1.0, "opened": True, "progress": 1.0, "items": items},
        hud={"probe": (300, 300), "latency": 80.0, "fps": 14.0},
    )
    assert base.shape == with_ui.shape and (base != with_ui).any()


def test_camera_menu_layout_limits_rows_to_the_panel_and_scrolls():
    cam, canvas = _camera(), np.full((1200, 850, 3), 255, np.uint8)
    layout = view_module.camera_menu_layout((1600, 640), cam.shape, canvas.shape, 30)
    assert 1 <= layout["visible"] < 30 and len(layout["rows"]) == layout["visible"]
    x, y, w, h = layout["card"]
    assert y >= 0 and all(ry >= 0 for _, ry, _, _ in layout["rows"])
    scrolled = view_module.camera_menu_layout(
        (1600, 640), cam.shape, canvas.shape, 30, first=99
    )
    assert scrolled["first"] == 30 - scrolled["visible"]


def test_row_glow_clips_rows_that_leave_the_image():
    studio = np.zeros((100, 200, 3), np.uint8)
    view_module._row_glow(studio, (10, -14, 100, 28), 1.0, 1.0)
    assert studio.any()


def test_hud_moves_to_the_footer_when_the_camera_leaves_no_side_margin():
    canvas = np.full((1200, 850, 3), 255, np.uint8)
    cam = np.full((1080, 1920, 3), 120, np.uint8)
    size = (1920, 1080)
    base = view_module.compose_view(cam, canvas, size=size)
    shown = view_module.compose_view(
        cam, canvas, size=size, hud={"fps": 14.0, "latency": 50.0}
    )
    ys, _xs = np.nonzero((base != shown).any(axis=2))
    assert ys.size and ys.min() > 1080 - 62


def test_hud_footer_starts_after_the_last_button_with_show_base():
    """Com show_base o rodape tem um botao a mais e o [Q] anda para a
    direita: o HUD do rodape tinha que recomecar depois dele, mas media
    o limite esquerdo pela fileira de 3 botoes e pintava por cima."""
    canvas = np.full((1200, 850, 3), 255, np.uint8)
    cam = np.full((1080, 1920, 3), 120, np.uint8)
    size = (1920, 1080)
    base = view_module.compose_view(cam, canvas, size=size, show_base=True)
    shown = view_module.compose_view(
        cam, canvas, size=size, show_base=True, hud={"fps": 14.0, "latency": 50.0}
    )
    ys, xs = np.nonzero((base != shown).any(axis=2))
    assert ys.size and ys.min() > 1080 - 62
    # Os botoes do rodape sao repintados depois do HUD: o texto que
    # comeca sob o [Q] deslocado some atras dele e sobra truncado.
    last = view_module.button_rects(size, show_base=True)[-1]
    sc = min(size[0] / view_module._BASE_W, size[1] / view_module._BASE_H)
    pad = view_module._scale(sc)(24)
    assert xs.min() >= last[0] + last[2] + pad


def test_camera_menu_scroll_arrows_exist_only_when_the_list_overflows():
    cam, canvas = _camera(), np.full((1200, 850, 3), 255, np.uint8)
    short = view_module.camera_menu_layout((1600, 640), cam.shape, canvas.shape, 3)
    assert short["scroll_up"] is None and short["scroll_down"] is None
    layout = view_module.camera_menu_layout((1600, 640), cam.shape, canvas.shape, 30)
    ux, uy, uw, uh = layout["scroll_up"]
    dx, dy, dw, dh = layout["scroll_down"]
    assert view_module.scroll_hit(layout, (ux + 2, uy + 2)) == -1
    assert view_module.scroll_hit(layout, (dx + 2, dy + 2)) == 1
    assert view_module.row_at(layout, (ux + 2, uy + 2)) is None
    rx, ry, _, _ = layout["rows"][0]
    assert view_module.scroll_hit(layout, (rx + 2, ry + 2)) is None
    assert view_module.row_at(layout, (rx + 2, ry + 2)) == 0
    items = [
        {"label": f"Camera {i}", "selected": False, "hover": 0.0} for i in range(30)
    ]
    drawn = view_module.compose_view(
        cam,
        canvas,
        camera_menu={
            "dot": 0.0,
            "opened": True,
            "progress": 1.0,
            "items": items[: layout["visible"]],
            "count": 30,
            "first": 0,
        },
    )
    assert drawn.shape == (640, 1600, 3)
