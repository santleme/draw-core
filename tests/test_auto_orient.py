"""Testes de orientacao automatica (180°, retrato/paisagem) e caderno de 2 paginas."""

import os

import cv2
import numpy as np
import pytest

from draw import config, contracts, geometry
from draw.auto_orient import (
    AutoPageOrientation,
    AutoRotationDetector,
    hand_mask_orientation_vote,
    quad_aspect_ratio,
)
from draw.camera import LatestFrameCapture
from draw.stream import KIND_FULL, InkCanvas, InkStreamPublisher, records_from_file


def test_fit_quad_unites_two_page_open_notebook_split_by_spine_and_keeps_refit():
    """Caderno aberto com lombada dividindo a mascara em 2 paginas adjacentes
    deve produzir um unico quadrilatero cobrindo as duas paginas E preservar
    retas ajustadas (para refino e memoria de arestas)."""
    mask = np.zeros((400, 600), dtype=np.uint8)
    mask[80:320, 70:292] = 255
    mask[80:320, 308:530] = 255

    quad, lines, trusted = geometry._fit_quad_impl(mask, None, None)
    assert quad is not None
    assert lines is not None
    assert trusted is not None and all(trusted)
    xs = quad[:, 0]
    ys = quad[:, 1]
    assert xs.min() <= 80.0
    assert xs.max() >= 520.0
    assert ys.min() <= 90.0
    assert ys.max() >= 310.0


def test_fit_quad_unites_angled_two_page_open_notebook():
    """Caderno aberto girado 15° na mesa ainda precisa unir as duas paginas."""
    mask = np.zeros((500, 700), dtype=np.uint8)
    mask[120:380, 120:340] = 255
    mask[120:380, 354:574] = 255
    rot = cv2.getRotationMatrix2D((350.0, 250.0), 15.0, 1.0)
    rotated = cv2.warpAffine(mask, rot, (700, 500), flags=cv2.INTER_NEAREST)

    quad = geometry.fit_quad(rotated)
    assert quad is not None
    # Largura combinada ~454 px; uma unica pagina teria ~220 px.
    width = 0.5 * (
        float(np.linalg.norm(quad[1] - quad[0]))
        + float(np.linalg.norm(quad[2] - quad[3]))
    )
    assert width >= 400.0


def test_fit_quad_ignores_small_stray_object_next_to_single_page():
    """Um objeto claro pequeno ao lado de uma folha unica NAO pode ser unido."""
    mask = np.zeros((400, 600), dtype=np.uint8)
    mask[60:340, 120:320] = 255  # folha principal (280 x 200 = 56000 px)
    mask[140:200, 340:390] = 255  # post-it pequeno (60 x 50 = 3000 px)

    quad = geometry.fit_quad(mask)
    assert quad is not None
    assert quad[:, 0].max() <= 330.0


def test_hand_mask_orientation_vote_requires_top_or_bottom_contact():
    top_entry = np.zeros((200, 300), dtype=np.uint8)
    top_entry[0:90, 110:190] = 255
    assert hand_mask_orientation_vote(top_entry) == "inverted"

    bottom_entry = np.zeros((200, 300), dtype=np.uint8)
    bottom_entry[110:200, 110:190] = 255
    assert hand_mask_orientation_vote(bottom_entry) == "upright"

    # Mao entrando pela borda ESQUERDA perto do topo nao toca a borda superior.
    side_near_top = np.zeros((720, 1280), dtype=np.uint8)
    side_near_top[55:195, 0:180] = 255
    assert hand_mask_orientation_vote(side_near_top) is None

    floating_center = np.zeros((200, 300), dtype=np.uint8)
    floating_center[70:130, 110:190] = 255
    assert hand_mask_orientation_vote(floating_center) is None


def test_auto_rotation_detector_confirms_once_and_resets_on_camera_switch():
    det = AutoRotationDetector(confirm_votes=3)
    assert det.observe_vote("inverted") is False
    assert det.observe_vote("inverted") is False
    assert det.observe_vote("upright") is False  # reseta sequencia
    assert det.observe_vote("inverted") is False
    assert det.observe_vote(None) is False
    assert det.observe_vote("inverted") is False
    assert det.observe_vote("inverted") is True
    assert det.flipped is True
    for _ in range(5):
        assert det.observe_vote("inverted") is False

    det.reset()
    assert det.flipped is False
    assert det.observe_vote("inverted") is False


def test_auto_page_orientation_hysteresis_and_oblique_portrait_guard():
    orient = AutoPageOrientation(initial="retrato", confirm_frames=3)
    frame_shape = (720, 1280, 3)

    # Folha A4 em pe vista sob camera obliqua: razao projetada em pixels e ~1,23,
    # mas a razao corrigida de perspectiva continua retrato (~0,71).
    oblique_portrait = np.array(
        [[451, 234], [782, 228], [781, 454], [542, 451]], dtype=np.float32
    )
    for _ in range(6):
        assert orient.observe(oblique_portrait, frame_shape) is None
    assert orient.current == "retrato"

    # Caderno aberto / folha paisagem frontal genuina -> troca para paisagem.
    wide = np.array([[240, 160], [1040, 160], [1040, 560], [240, 560]], np.float32)
    tall = np.array([[440, 60], [840, 60], [840, 660], [440, 660]], np.float32)
    square = np.array([[440, 160], [840, 160], [840, 560], [440, 560]], np.float32)

    # Folha paisagem centralizada com inclinacao em um unico eixo (pitch 12°):
    # arestas horizontais paralelas (um ponto de fuga no infinito) -> deve
    # confirmar paisagem em vez de ficar presa em retrato.
    tilted_landscape = np.array(
        [[288, 117], [992, 117], [1033, 632], [247, 632]], dtype=np.float32
    )
    orient_tilted = AutoPageOrientation(initial="retrato", confirm_frames=3)
    assert orient_tilted.observe(tilted_landscape, frame_shape) is None
    assert orient_tilted.observe(tilted_landscape, frame_shape) is None
    assert orient_tilted.observe(tilted_landscape, frame_shape) == "paisagem"

    # Pagina quase quadrada (razao fisica 1,16, dentro da zona morta) sob pitch
    # de 14°: apenas um extremo da faixa focal passa de 1,18 -> deve abster-se.
    near_square_tilted = np.array(
        [[337.31, 106.81], [942.69, 106.81], [924.01, 597.57], [355.99, 597.57]],
        dtype=np.float32,
    )
    orient_near_sq = AutoPageOrientation(initial="retrato", confirm_frames=3)
    for _ in range(6):
        assert orient_near_sq.observe(near_square_tilted, frame_shape) is None
    assert orient_near_sq.current == "retrato"

    # A4 paisagem sob pitch de 18° e f = largura do frame (1280 px):
    # deve confirmar a troca para paisagem.
    a4_tilted_18deg = np.array(
        [[233.5, 86.6], [1046.5, 86.6], [997.0, 600.1], [283.0, 600.1]],
        dtype=np.float32,
    )
    orient_a4 = AutoPageOrientation(initial="retrato", confirm_frames=3)
    assert orient_a4.observe(a4_tilted_18deg, frame_shape) is None
    assert orient_a4.observe(a4_tilted_18deg, frame_shape) is None
    assert orient_a4.observe(a4_tilted_18deg, frame_shape) == "paisagem"

    assert quad_aspect_ratio(wide, frame_shape) > 1.5
    assert orient.observe(square, frame_shape) is None
    assert orient.observe(wide, frame_shape) is None
    assert orient.observe(wide, frame_shape) is None
    assert orient.observe(wide, frame_shape) == "paisagem"
    assert orient.current == "paisagem"

    # Zona morta nao volta para retrato.
    for _ in range(5):
        assert orient.observe(square, frame_shape) is None
    assert orient.current == "paisagem"

    # Folha em pe sustentada volta para retrato.
    assert orient.observe(tall, frame_shape) is None
    assert orient.observe(tall, frame_shape) is None
    assert orient.observe(tall, frame_shape) == "retrato"
    assert orient.current == "retrato"


def test_latest_frame_capture_set_rotation_guarantees_next_packet_flipped():
    class _FixedCam:
        def __init__(self):
            self.frame = np.full((40, 60, 3), 240, dtype=np.uint8)
            self.frame[2:8, 2:8] = 10  # canto superior esquerdo

        def read(self):
            return True, self.frame.copy()

        def release(self):
            pass

    src = LatestFrameCapture(_FixedCam(), rotation=0).start()
    try:
        p1 = src.wait_next(0, timeout=2.0)
        assert p1 is not None
        assert p1.image[2:8, 2:8].mean() < 50
        src.set_rotation(180)
        assert src.rotation == 180
        # O PRIMEIRO pacote retornado apos set_rotation(180) ja deve vir girado.
        p2 = src.wait_next(p1.sequence, timeout=2.0)
        assert p2 is not None
        assert p2.image[32:38, 52:58].mean() < 50
    finally:
        src.stop()


def test_ink_stream_publisher_updates_paper_mm_on_orientation_change(tmp_path):
    import time

    rec = tmp_path / "orient.dral"
    pub = InkStreamPublisher(
        port=None,
        paper_mm=config.paper_mm("a4", "retrato"),
        record_path=str(rec),
    )
    try:
        portrait_mask = np.zeros((120, 85), dtype=np.uint8)
        portrait_mask[10:20, 10:20] = 255
        pub.publish(contracts.InkPublication(1, 1, portrait_mask))
        # Aguarda o registro retrato ser gravado antes de publicar a proxima geracao.
        deadline = time.monotonic() + 2.0
        while rec.stat().st_size == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        # Simula transicao para paisagem antes da nova geracao.
        pub.set_paper_mm(config.paper_mm("a4", "paisagem"))
        landscape_mask = np.zeros((85, 120), dtype=np.uint8)
        landscape_mask[10:20, 10:20] = 255
        pub.publish(contracts.InkPublication(2, 1, landscape_mask))
    finally:
        pub.close()

    records = list(records_from_file(rec))
    fulls = [r for r in records if r.kind == KIND_FULL]
    assert len(fulls) >= 2
    canvas = InkCanvas()
    canvas.apply(fulls[0])
    assert (fulls[0].width, fulls[0].height) == (85, 120)
    assert canvas.paper_mm == (210.0, 297.0)
    canvas.apply(fulls[-1])
    assert (fulls[-1].width, fulls[-1].height) == (120, 85)
    assert canvas.paper_mm == (297.0, 210.0)


def test_neural_detector_expands_from_single_page_to_gradually_uncovered_spread():
    """Quando a segunda pagina aparece gradualmente, a memoria de arestas da
    pagina unica NAO pode prender a borda direita na lombada."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    # 1. Comeca enxergando apenas a pagina esquerda (222x240).
    mask = np.zeros((400, 600), dtype=np.uint8)
    mask[80:320, 70:292] = 255
    q1 = det._fit(mask)
    assert q1 is not None
    assert q1[:, 0].max() <= 300.0

    # 2. Segunda pagina entra gradualmente (86 px, +30, +30, +30, +46 px)
    #    com lombada de 16 px (x=292..308).
    for right_w in (86, 116, 146, 176, 222):
        step_mask = np.zeros((400, 600), dtype=np.uint8)
        step_mask[80:320, 70:292] = 255
        step_mask[80:320, 308 : 308 + right_w] = 255
        q = det._fit(step_mask)
        assert q is not None
        assert q[:, 0].max() >= 308 + right_w - 12.0


def test_neural_detector_keeps_spread_edge_under_hand_occlusion():
    """Depois de estabelecer um caderno aberto completo, uma oclusao de mao
    cobrindo a faixa externa da pagina direita NAO pode descartar a memoria
    de arestas nem encolher o quadrilatero."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    full_spread = np.zeros((500, 600), dtype=np.uint8)
    full_spread[60:444, 72:296] = 255
    full_spread[60:444, 304:528] = 255

    q_full = det._fit(full_spread)
    assert q_full is not None
    assert q_full[:, 0].max() >= 518.0

    # Mao cobre os 132 px externos da pagina direita (overlap ~0,705).
    occluded = full_spread.copy()
    occluded[:, 396:] = 0
    for _ in range(3):
        q_occ = det._fit(occluded)
        assert q_occ is not None
        assert q_occ[:, 0].max() >= 510.0


def test_neural_detector_uncovers_final_strip_of_spread_under_5_percent_area():
    """Quando o ultimo pedaco de uma pagina aparece (ex.: 10 colunas em mascara
    384x384 sob escala 1280x720 -> 33 px de deslocamento, > 18 px de limite da
    guarda, mas < 5% de area do caderno), a aresta desobstruida NAO pode ficar
    cortada na memoria."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    scale_x, scale_y = 1280.0 / 384.0, 720.0 / 384.0

    # Caderno aberto: duas paginas de 134 px de largura com lombada de 8 px (x=184..192).
    # Pagina direita inicialmente exposta com 124 px (x=192..316), faltando 10 colunas.
    mask_initial = np.zeros((384, 384), dtype=np.uint8)
    mask_initial[60:320, 50:184] = 255
    mask_initial[60:320, 192:316] = 255

    q1 = det._fit(mask_initial, scale_x=scale_x, scale_y=scale_y)
    assert q1 is not None
    # Borda direita inicial em torno de 316 * scale_x (~1053 px)
    assert q1[:, 0].max() <= 318.0 * scale_x

    # Desobstrui as ultimas 10 colunas (x=316..326): +33 px em coordenadas de frame,
    # que excede os 18 px da guarda de deslocamento, mas acrescenta apenas ~3,8% de area.
    mask_full = mask_initial.copy()
    mask_full[60:320, 316:326] = 255

    for _ in range(3):
        q2 = det._fit(mask_full, scale_x=scale_x, scale_y=scale_y)
        assert q2 is not None
        # A borda direita deve expandir para o novo limite (~326 * scale_x = ~1086 px)
        assert q2[:, 0].max() >= 324.0 * scale_x - 5.0


def test_neural_detector_sideways_slide_does_not_leave_stale_trailing_edge():
    """Uma translacao lateral da folha (40 px para a direita em 1280x720) move
    as duas arestas verticais juntas; a guarda de outlier unico NAO pode
    contar apenas a aresta traseira e manter a folha esticada."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    m1 = np.zeros((720, 1280), dtype=np.uint8)
    m1[120:600, 250:910] = 255
    q1 = det._fit(m1)
    assert q1 is not None
    assert abs(q1[:, 0].min() - 250.0) <= 5.0
    assert abs(q1[:, 0].max() - 910.0) <= 5.0

    # Desliza 40 px para a direita (x=290..950, alta sobreposicao com m1).
    m2 = np.zeros((720, 1280), dtype=np.uint8)
    m2[120:600, 290:950] = 255
    for _ in range(3):
        q2 = det._fit(m2)
        assert q2 is not None
        assert abs(q2[:, 0].min() - 290.0) <= 6.0
        assert abs(q2[:, 0].max() - 950.0) <= 6.0


def test_neural_detector_updates_spread_peak_when_expanded_with_untrusted_edge(
    monkeypatch,
):
    """Se o caderno aberto expande > 5% enquanto uma aresta esta nao-confiavel
    (ex.: antebraco na borda inferior), fallback e None nesse quadro e o pico
    de area precisa acompanhar o novo tamanho para nao desabilitar a memoria
    de arestas nos quadros seguintes."""
    from draw import geometry as _geometry
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    m1 = np.zeros((400, 600), dtype=np.uint8)
    m1[80:320, 70:260] = 255
    m1[80:320, 270:460] = 255
    assert det._fit(m1) is not None
    initial_peak = det._memory_spread_area
    assert initial_peak > 0

    # Expande > 5% (pagina direita cresce de 190 para 230 px) enquanto a
    # aresta inferior volta nao-confiavel.
    m2 = np.zeros((400, 600), dtype=np.uint8)
    m2[80:320, 70:260] = 255
    m2[80:320, 270:500] = 255

    orig_fit = _geometry._fit_quad_impl
    seen_fallbacks = []

    def _fit_with_untrusted_bottom(mask, frame, fallback, sx, sy, **kwargs):
        seen_fallbacks.append(fallback)
        corners, lines, trusted = orig_fit(mask, frame, fallback, sx, sy, **kwargs)
        if trusted is not None:
            trusted = [trusted[0], trusted[1], False, trusted[3]]
        return corners, lines, trusted

    monkeypatch.setattr(_geometry, "_fit_quad_impl", _fit_with_untrusted_bottom)

    # Quadro da expansao: fallback e None e o pico deve subir.
    assert det._fit(m2) is not None
    assert seen_fallbacks[-1] is None
    assert det._memory_spread_area > initial_peak * 1.05

    # Quadro seguinte com a mesma mascara: memoria de arestas volta ativa.
    assert det._fit(m2) is not None
    assert seen_fallbacks[-1] is not None


def test_neural_detector_shifting_occlusion_recovers_exposed_edge_and_guards_new_occlusion():
    """Quando o braco sai da borda esquerda (revelando x=250) e passa a cobrir
    a borda superior (empurrando y=120 -> 160), nao e translacao rigida: a
    borda esquerda deve recuperar x=250 e a superior deve continuar em y=120."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)

    # Quadro 1: folha x=250..909, y=120..599 com 40 px da esquerda cobertos.
    m1 = np.zeros((720, 1280), dtype=np.uint8)
    m1[120:600, 290:910] = 255
    q1 = det._fit(m1)
    assert q1 is not None
    assert abs(q1[:, 1].min() - 120.0) <= 5.0

    # Quadro 2+: esquerda livre (x=250), mas 40 px superiores agora cobertos (y=160).
    m2 = np.zeros((720, 1280), dtype=np.uint8)
    m2[160:600, 250:910] = 255
    for _ in range(3):
        q2 = det._fit(m2)
        assert q2 is not None
        assert abs(q2[:, 0].min() - 250.0) <= 6.0
        assert abs(q2[:, 1].min() - 120.0) <= 6.0


def test_neural_detector_expansion_without_fresh_lines_clears_stale_edges(
    monkeypatch,
):
    """Se _approx_quad falha e _fit_quad_impl cai no caminho real de
    _min_area_quad (lines=None) em uma expansao de caderno, o detector nao
    pode elevar o pico nem parear a nova mascara com as linhas pre-expansao."""
    from draw import geometry as _geometry
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    m1 = np.zeros((400, 600), dtype=np.uint8)
    m1[80:320, 70:260] = 255
    m1[80:320, 270:460] = 255
    assert det._fit(m1) is not None
    initial_peak = det._memory_spread_area
    assert det._edge_lines is not None

    m2 = np.zeros((400, 600), dtype=np.uint8)
    m2[80:320, 70:260] = 255
    m2[80:320, 270:500] = 255

    orig_approx = _geometry._approx_quad
    orig_min_area = _geometry._min_area_quad
    orig_fit = _geometry._fit_quad_impl
    approx_calls = {"n": 0}
    min_area_calls = {"n": 0}
    seen_fallbacks = []

    def _approx_none_once(contour):
        approx_calls["n"] += 1
        if approx_calls["n"] == 1:
            return None
        return orig_approx(contour)

    def _spy_min_area(contour):
        min_area_calls["n"] += 1
        return orig_min_area(contour)

    def _spy_fit(mask, frame, fallback, sx, sy, **kwargs):
        seen_fallbacks.append(fallback)
        return orig_fit(mask, frame, fallback, sx, sy, **kwargs)

    monkeypatch.setattr(_geometry, "_approx_quad", _approx_none_once)
    monkeypatch.setattr(_geometry, "_min_area_quad", _spy_min_area)
    monkeypatch.setattr(_geometry, "_fit_quad_impl", _spy_fit)

    # Quadro da expansao exercitando o _min_area_quad real: pico nao sobe e
    # retas pre-expansao sao limpas.
    assert det._fit(m2) is not None
    assert min_area_calls["n"] == 1
    assert det._edge_lines is None
    assert det._memory_spread_area == initial_peak

    # Quadro seguinte: fallback continua None (sem parear linhas velhas) e pico sobe.
    assert det._fit(m2) is not None
    assert seen_fallbacks[-1] is None
    assert det._memory_spread_area > initial_peak * 1.05


def test_neural_detector_trapezoidal_perspective_slide_does_not_stretch():
    """Deslize rigido de um trapezio sob camera inclinada (onde a borda proxima
    anda mais pixels que a distante na imagem, mas deslocamentos sao iguais e
    opostos no plano retificado) nao pode travar a borda traseira."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    scale_x, scale_y = 1280.0 / 384.0, 720.0 / 384.0

    # Trapezio inicial em mascara 384x384: topo y=100 (x=97..287), base y=285 (x=57..327).
    trap1 = np.array([[97, 100], [287, 100], [327, 285], [57, 285]], dtype=np.int32)
    m1 = np.zeros((384, 384), dtype=np.uint8)
    cv2.fillPoly(m1, [trap1], 255)
    q1 = det._fit(m1, scale_x=scale_x, scale_y=scale_y)
    assert q1 is not None
    assert abs(q1[:, 1].max() - 285.0 * scale_y) <= 8.0

    # Desliza para cima no plano da mesa mantendo as retas laterais fixas:
    # topo sobe para y=91, base sobe para y=266.
    def _side_x(y, x0, y0, x1, y1):
        return int(round(x0 + (y - y0) * (x1 - x0) / (y1 - y0)))

    trap2 = np.array(
        [
            [_side_x(91, 97, 100, 57, 285), 91],
            [_side_x(91, 287, 100, 327, 285), 91],
            [_side_x(266, 287, 100, 327, 285), 266],
            [_side_x(266, 97, 100, 57, 285), 266],
        ],
        dtype=np.int32,
    )
    m2 = np.zeros((384, 384), dtype=np.uint8)
    cv2.fillPoly(m2, [trap2], 255)

    for _ in range(3):
        q2 = det._fit(m2, scale_x=scale_x, scale_y=scale_y)
        assert q2 is not None
        assert abs(q2[:, 1].min() - 91.0 * scale_y) <= 8.0
        assert abs(q2[:, 1].max() - 266.0 * scale_y) <= 10.0


def test_neural_detector_uncovering_two_edges_with_angle_change_recovers_both():
    """Quando duas bordas reaparecem juntas para fora e uma delas desentorta
    > 4,5 graus (ex.: ~5,6 graus), ambas devem ser aceitas e permanecer
    recuperadas em quadros repetidos."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)

    # Quadro 1 (384x384): topo em y=85, borda esquerda cortada em diagonal
    # de (87, 85) ate (110, 320), direita em x=310, base em y=320.
    poly1 = np.array([[87, 85], [310, 85], [310, 320], [110, 320]], dtype=np.int32)
    m1 = np.zeros((384, 384), dtype=np.uint8)
    cv2.fillPoly(m1, [poly1], 255)
    q1 = det._fit(m1)
    assert q1 is not None

    # Quadro 2+: ambas as oclusoes saem -> topo volta para y=64 e esquerda
    # volta reta para x=75 (ambas para fora, esquerda desentortando ~5,6 graus).
    m2 = np.zeros((384, 384), dtype=np.uint8)
    m2[64:320, 75:310] = 255
    for _ in range(3):
        q2 = det._fit(m2)
        assert q2 is not None
        assert abs(q2[:, 1].min() - 64.0) <= 5.0
        assert abs(q2[:, 0].min() - 75.0) <= 5.0


def test_neural_detector_pivoting_occlusion_during_edge_reveal_is_guarded():
    """Quando a borda esquerda reaparece (x=80 -> x=70) enquanto a borda
    superior gira em torno do proprio ponto medio (y=100 -> diagonal (70,80)..(310,120),
    ~9,5 graus), a guarda deve proteger a borda superior em vez de salvar a diagonal."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    scale_x, scale_y = 1280.0 / 384.0, 720.0 / 384.0

    # Quadro 1: x=80..310, y=100..310.
    m1 = np.zeros((384, 384), dtype=np.uint8)
    m1[100:310, 80:310] = 255
    q1 = det._fit(m1, scale_x=scale_x, scale_y=scale_y)
    assert q1 is not None

    # Quadro 2+: esquerda revela x=70, topo gira ~9,5 graus ao redor de y=100.
    poly2 = np.array([[70, 80], [310, 120], [310, 310], [70, 310]], dtype=np.int32)
    m2 = np.zeros((384, 384), dtype=np.uint8)
    cv2.fillPoly(m2, [poly2], 255)
    for _ in range(3):
        q2 = det._fit(m2, scale_x=scale_x, scale_y=scale_y)
        assert q2 is not None
        # Esquerda recupera para ~70 * scale_x; topo permanece horizontal perto de 100 * scale_y.
        assert abs(q2[:, 0].min() - 70.0 * scale_x) <= 10.0
        assert abs(q2[0, 1] - 100.0 * scale_y) <= 10.0
        assert abs(q2[1, 1] - 100.0 * scale_y) <= 10.0


def test_neural_detector_unequal_opposite_edge_shifts_not_mistaken_for_slide():
    """Em pagina frontal 384x384 (x=80..330), expor 7 colunas na direita
    enquanto cobre 16 na esquerda (diferenca de 9 px > 35% de 16 px) NAO pode
    ser aceito como translacao: a esquerda deve permanecer em x=80."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    scale_x, scale_y = 1280.0 / 384.0, 720.0 / 384.0

    m1 = np.zeros((384, 384), dtype=np.uint8)
    m1[80:310, 80:330] = 255
    q1 = det._fit(m1, scale_x=scale_x, scale_y=scale_y)
    assert q1 is not None
    assert abs(q1[:, 0].min() - 80.0 * scale_x) <= 8.0

    # Direita expoe +7 colunas (330 -> 337); esquerda cobre 16 colunas (80 -> 96).
    m2 = np.zeros((384, 384), dtype=np.uint8)
    m2[80:310, 96:337] = 255
    for _ in range(3):
        q2 = det._fit(m2, scale_x=scale_x, scale_y=scale_y)
        assert q2 is not None
        assert abs(q2[:, 0].min() - 80.0 * scale_x) <= 10.0
        assert abs(q2[:, 0].max() - 337.0 * scale_x) <= 10.0


def test_neural_detector_boundary_pivot_reveal_does_not_keep_skewed_edge():
    """Quando duas bordas reaparecem e o novo ponto medio da borda superior
    cai sobre o antigo canto superior direito (extremidade do trecho
    compartilhado, nao pivo interior), a borda superior reta deve ser aceita."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)

    # Quadro 1 (384x384): topo cortado em diagonal (88,96)->(190,80).
    poly1 = np.array([[88, 96], [190, 80], [330, 310], [60, 310]], dtype=np.int32)
    m1 = np.zeros((384, 384), dtype=np.uint8)
    cv2.fillPoly(m1, [poly1], 255)
    assert det._fit(m1) is not None

    # Quadro 2+: obstrucao removida -> topo reto (90,80)..(290,80), ponto medio
    # (190,80) coincide com o canto antigo na fronteira do trecho compartilhado.
    poly2 = np.array([[90, 80], [290, 80], [330, 310], [60, 310]], dtype=np.int32)
    m2 = np.zeros((384, 384), dtype=np.uint8)
    cv2.fillPoly(m2, [poly2], 255)
    for _ in range(3):
        q2 = det._fit(m2)
        assert q2 is not None
        assert abs(q2[0, 1] - 80.0) <= 5.0
        assert abs(q2[1, 1] - 80.0) <= 5.0
        assert abs(q2[1, 0] - 290.0) <= 6.0


def test_neural_detector_noisy_native_scale_slide_does_not_lock_trailing_edge():
    """Em mascara nativa 384x384 para frame 1280x720, um deslize rigido de ~9 px
    com ruido de discretizacao (+7 px na lider, +11 px na traseira, diferenca
    de 4 px) deve ser aceito como deslize sem travar a borda traseira."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    scale_x, scale_y = 1280.0 / 384.0, 720.0 / 384.0

    m1 = np.zeros((384, 384), dtype=np.uint8)
    m1[80:310, 80:330] = 255
    assert det._fit(m1, scale_x=scale_x, scale_y=scale_y) is not None

    # Quadro 2 (ruidoso): lider +7 px (337), traseira +11 px (91).
    m2 = np.zeros((384, 384), dtype=np.uint8)
    m2[80:310, 91:337] = 255
    q2 = det._fit(m2, scale_x=scale_x, scale_y=scale_y)
    assert q2 is not None
    assert abs(q2[:, 0].min() - 91.0 * scale_x) <= 10.0

    # Quadros 3+ (estabilizados em +9 px: 89..339): traseira acompanha normalmente.
    m3 = np.zeros((384, 384), dtype=np.uint8)
    m3[80:310, 89:339] = 255
    for _ in range(2):
        q3 = det._fit(m3, scale_x=scale_x, scale_y=scale_y)
        assert q3 is not None
        assert abs(q3[:, 0].min() - 89.0 * scale_x) <= 10.0
        assert abs(q3[:, 0].max() - 339.0 * scale_x) <= 10.0


def test_neural_detector_lateral_trapezoid_slide_with_large_angle_change():
    """Slide lateral de trapezio: os lados mudam >6 graus na imagem, mas no plano
    retificado e uma translacao pura. A borda traseira nao pode ficar presa."""
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector("unused.onnx", verify=None)
    t1 = np.array([[109, 100], [241, 100], [325, 260], [25, 260]], dtype=np.int32)
    m1 = np.zeros((384, 384), dtype=np.uint8)
    cv2.fillPoly(m1, [t1], 255)
    assert det._fit(m1) is not None

    # Topo desliza 19 px, base 43 px (proporcional as larguras projetadas).
    t2 = np.array([[128, 100], [260, 100], [368, 260], [68, 260]], dtype=np.int32)
    m2 = np.zeros((384, 384), dtype=np.uint8)
    cv2.fillPoly(m2, [t2], 255)
    for _ in range(3):
        q = det._fit(m2)
        assert q is not None
        assert abs(q[:, 0].min() - 68.0) <= 6.0
        assert abs(q[:, 0].max() - 368.0) <= 6.0
        top = q[np.argsort(q[:, 1])[:2]]
        assert abs(top[:, 0].min() - 128.0) <= 6.0


def test_hand_vote_abstains_on_side_entry_inside_top_band():
    from draw.auto_orient import hand_mask_orientation_vote

    mask = np.zeros((1280, 720), dtype=np.uint8)
    mask[8:75, 0:220] = 255  # entra pela esquerda, sem tocar a borda superior
    assert hand_mask_orientation_vote(mask) is None
    side_and_corner = mask.copy()
    side_and_corner[0:75, 0:60] = 255  # toca o topo, mas o contato lateral domina
    assert hand_mask_orientation_vote(side_and_corner) is None


def test_hand_vote_still_detects_real_top_entry():
    from draw.auto_orient import hand_mask_orientation_vote

    mask = np.zeros((1280, 720), dtype=np.uint8)
    mask[0:300, 300:420] = 255
    assert hand_mask_orientation_vote(mask) == "inverted"


def _contour_of(x0, y0, x1, y1):
    m = np.zeros((400, 800), dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours[0]


def test_loose_sheets_with_desk_gap_are_not_a_spread():
    from draw.geometry import _is_two_page_spread

    # Duas A4 lado a lado com ~19% de largura de intervalo: folhas soltas.
    a = _contour_of(50, 50, 300, 350)
    b = _contour_of(347, 50, 597, 350)
    assert _is_two_page_spread(a, b) is False


def test_notebook_with_narrow_gutter_is_still_a_spread():
    from draw.geometry import _is_two_page_spread

    a = _contour_of(50, 50, 300, 350)
    b = _contour_of(310, 50, 560, 350)
    assert _is_two_page_spread(a, b) is True


def test_fit_does_not_swallow_typeerror_from_geometry(monkeypatch):
    from draw import geometry as geometry_module
    from draw.perception import NeuralSheetDetector

    def boom(mask, frame, fallback, sx, sy, contour=None):
        raise TypeError("erro real do ajuste")

    monkeypatch.setattr(geometry_module, "_fit_quad_impl", boom)
    det = NeuralSheetDetector("unused.onnx", verify=None)
    m = np.zeros((384, 384), dtype=np.uint8)
    m[80:300, 80:300] = 255
    with pytest.raises(TypeError, match="erro real"):
        det._fit(m)


@pytest.mark.parametrize("h", [480, 720, 1280])
def test_hand_vote_tolerates_narrow_segmentation_gap_at_top_edge(h):
    from draw.auto_orient import hand_mask_orientation_vote

    mask = np.zeros((h, 720), dtype=np.uint8)
    mask[2 : h // 4, 300:420] = 255  # segmentacao deixou 2 linhas vazias no topo
    assert hand_mask_orientation_vote(mask) == "inverted"


@pytest.mark.parametrize("h", [480, 720, 1280])
def test_wide_lateral_hand_inside_top_slack_band_abstains(h):
    from draw.auto_orient import hand_mask_orientation_vote

    mask = np.zeros((h, 720), dtype=np.uint8)
    mask[2:72, 0:240] = 255  # toca so a esquerda; sobrepoe a faixa de folga do topo
    assert hand_mask_orientation_vote(mask) is None


@pytest.mark.skipif(os.name != "posix", reason="modos POSIX")
def test_png_writer_and_private_dir_are_owner_only(tmp_path):
    from draw.camera import PngWriter
    from draw.infra.safe_io import ensure_private_dir

    old = os.umask(0o022)
    try:
        folder = ensure_private_dir(tmp_path / "capturas")
        writer = PngWriter()
        out = folder / "x.png"
        writer.save(out, np.zeros((4, 4, 3), np.uint8))
        writer.close()
    finally:
        os.umask(old)
    assert out.exists()
    assert (folder.stat().st_mode & 0o077) == 0
    assert (out.stat().st_mode & 0o077) == 0
