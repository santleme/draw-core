"""Testes das guardas geométricas e pré/pós-processamento de draw/perception.py + draw/geometry.py."""

import cv2
import numpy as np

from draw import geometry, perception


class _DummyInput:
    name = "image"
    shape = [1, 3, "H", "W"]


class _DummyOutput:
    def __init__(self, name):
        self.name = name


class _DummySession:
    def __init__(self, logits, presence=10.0):
        self._logits = logits
        self._presence = np.array([presence], dtype=np.float32)
        self.last_tensor = None

    def get_inputs(self):
        return [_DummyInput()]

    def get_outputs(self):
        # A main pos-PR #9 le o contrato da sessao (nomes das saidas);
        # a sessao falsa declara as duas saidas que devolve em run().
        return [_DummyOutput("mask_logits"), _DummyOutput("presence_logit")]

    def run(self, _outputs, feed):
        self.last_tensor = feed["image"]
        return [self._logits, self._presence]


class _DummySegSession:
    """Sessão só-segmentação: uma saída 'logits' [1,4,H,W], sem cabeça de presença."""

    def __init__(self, logits):
        self._logits = logits

    def get_inputs(self):
        return [_DummyInput()]

    def get_outputs(self):
        return [_DummyOutput("logits")]

    def run(self, _outputs, feed):
        return [self._logits]


def _poly_mask(shape, pts):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


def _max_corner_err(got, want):
    assert got is not None, "esperava quadrilatero valido, obteve None"
    return float(np.linalg.norm(got - want, axis=1).max())


def test_predict_matches_legacy_softmax_and_normalization():
    rng = np.random.default_rng(19)
    frame = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    size = 384
    logits = rng.standard_normal((1, 2, size, size)).astype(np.float32)

    det = perception.NeuralSheetDetector("dummy.onnx", verify=None)
    sess = _DummySession(logits, presence=9.25)
    det._session = sess

    pres, mask_fast = det.predict(frame)
    assert abs(pres - 9.25) < 1e-5

    # Caminho legado de referência:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(
        np.float32
    )
    want_tensor = ((resized / 255.0 - perception._MEAN) / perception._STD).transpose(
        2, 0, 1
    )[None]
    assert float(np.abs(sess.last_tensor - want_tensor).max()) < 1e-5

    c_bg, c_p = logits[0, 0], logits[0, 1]
    peak = np.maximum(c_bg, c_p)
    prob = np.exp(c_p - peak) / (np.exp(c_bg - peak) + np.exp(c_p - peak))
    want_mask = (
        cv2.resize(
            (prob > 0.5).astype(np.uint8),
            (1920, 1080),
            interpolation=cv2.INTER_NEAREST,
        )
        * 255
    )
    assert np.array_equal(mask_fast, want_mask)


def test_middle_arm_notch_recovers_straight_chord_without_memory():
    shape = (720, 1280)
    true_quad = np.array(
        [[280.0, 90.0], [980.0, 105.0], [1010.0, 630.0], [255.0, 615.0]],
        dtype=np.float32,
    )
    mask = _poly_mask(shape, true_quad)
    # Entalhe curvo de antebraço (50 px de profundidade) no meio da borda inferior,
    # deixando trechos retos nos dois lados (coverage = 1.0).
    cv2.circle(mask, (630, 625), 75, 0, -1)

    got = geometry.fit_quad(mask)
    err = _max_corner_err(got, true_quad)
    assert err <= 4.0, f"entalhe central distorceu canto em {err:.2f} px"


def test_curved_arm_with_high_coverage_does_not_poison_memory():
    shape = (720, 1280)
    true_quad = np.array(
        [[280.0, 90.0], [980.0, 105.0], [1010.0, 630.0], [255.0, 615.0]],
        dtype=np.float32,
    )
    clean_mask = _poly_mask(shape, true_quad)

    det = perception.NeuralSheetDetector("dummy.onnx", verify=None)
    c0 = det._fit(clean_mask, None)
    assert _max_corner_err(c0, true_quad) <= 2.5
    saved_bottom_dir = det._edge_lines[2][0].copy()

    # Quadro seguinte: mão/braço curvo esculpe um arco suave ao longo de ~75%
    # da aresta inferior (coverage >= 0.5, mas contorno curvo não-colinear).
    curved_pts = [true_quad[0], true_quad[1], [1006.0, 595.0]]
    for t in np.linspace(0.88, 0.12, 18):
        x = (1.0 - t) * 255.0 + t * 1010.0
        y_base = (1.0 - t) * 615.0 + t * 630.0
        arch = 52.0 * np.sin(np.pi * (t - 0.12) / 0.76)
        curved_pts.append([x, y_base - arch])
    curved_pts.append([258.0, 585.0])
    occluded_mask = _poly_mask(shape, np.array(curved_pts, dtype=np.float32))

    c1 = det._fit(occluded_mask, None)
    err = _max_corner_err(c1, true_quad)
    assert err <= 4.0, f"arco curvo deslocou quadrilatero em {err:.2f} px"
    # Memória da aresta inferior deve permanecer intacta.
    cos_sim = abs(float(det._edge_lines[2][0] @ saved_bottom_dir))
    assert cos_sim > 0.9995, "aresta curva contaminou self._edge_lines"


def test_rigid_guard_vetoes_single_edge_sleeve_but_allows_real_sheet_rotation():
    shape = (720, 1280)
    true_quad = np.array(
        [[300.0, 100.0], [960.0, 100.0], [960.0, 620.0], [300.0, 620.0]],
        dtype=np.float32,
    )
    clean_mask = _poly_mask(shape, true_quad)
    det = perception.NeuralSheetDetector("dummy.onnx", verify=None)
    assert _max_corner_err(det._fit(clean_mask, None), true_quad) <= 2.0

    # Caso 1: 3 arestas imóveis, mas uma manga reta entra a ~13 graus na
    # aresta direita cortando o canto inferior direito.
    sleeve_quad = np.array(
        [[300.0, 100.0], [960.0, 100.0], [840.0, 620.0], [300.0, 620.0]],
        dtype=np.float32,
    )
    sleeve_mask = _poly_mask(shape, sleeve_quad)
    c_sleeve = det._fit(sleeve_mask, None)
    err_sleeve = _max_corner_err(c_sleeve, true_quad)
    assert err_sleeve <= 3.0, (
        f"manga reta numa unica aresta enganou guarda rigida: {err_sleeve:.2f} px"
    )

    # Caso 2: o usuario gira a folha inteira em +7 graus (todas as 4 arestas
    # giram juntas; IoU da mascara > 0,70). A guarda NAO pode travar!
    center = true_quad.mean(axis=0)
    ang = np.radians(7.0)
    rot = np.array(
        [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]],
        dtype=np.float32,
    )
    rot_quad = (true_quad - center) @ rot.T + center
    rot_mask = _poly_mask(shape, rot_quad)
    c_rot = det._fit(rot_mask, None)
    err_rot = _max_corner_err(c_rot, rot_quad)
    assert err_rot <= 3.5, (
        f"guarda rigida travou rotacao real da folha: erro {err_rot:.2f} px"
    )


def test_fit_sem_contorno_reporta_no_contour():
    det = perception.NeuralSheetDetector("dummy.onnx", verify=None)
    mask = np.zeros((720, 1280), dtype=np.uint8)
    assert det._fit(mask, None) is None
    assert det.last_fit_reject == {"reason": "no_contour", "area_ratio": 0.0}


def test_fit_contorno_minusculo_reporta_min_area():
    det = perception.NeuralSheetDetector("dummy.onnx", verify=None)
    mask = np.zeros((720, 1280), dtype=np.uint8)
    # 20x20 px: ~0,04% da mascara, abaixo do piso de 1% do ajuste.
    cv2.rectangle(mask, (100, 100), (119, 119), 255, -1)
    assert det._fit(mask, None) is None
    reject = det.last_fit_reject
    assert reject["reason"] == "min_area"
    _, _, contour_area = geometry._sheet_contour_with_spread(mask)
    expected = contour_area / float(mask.shape[0] * mask.shape[1])
    assert reject["area_ratio"] == expected
    assert 0.0 < reject["area_ratio"] < geometry._MIN_CONTOUR_AREA_RATIO


def test_fit_com_folha_fecha_e_limpa_o_reject():
    shape = (720, 1280)
    true_quad = np.array(
        [[280.0, 90.0], [980.0, 105.0], [1010.0, 630.0], [255.0, 615.0]],
        dtype=np.float32,
    )
    det = perception.NeuralSheetDetector("dummy.onnx", verify=None)
    # Rejeicao anterior nao pode vazar para o quadro que fecha.
    empty = np.zeros(shape, dtype=np.uint8)
    assert det._fit(empty, None) is None
    assert det.last_fit_reject is not None
    mask = _poly_mask(shape, true_quad)
    corners = det._fit(mask, None)
    assert corners is not None
    assert det.last_fit_reject is None


def test_fit_implausivel_reporta_implausible():
    # Trapezio com arestas opostas 9:1 — fecha na escada de epsilon
    # mas a guarda de plausibilidade recusa.
    shape = (720, 1280)
    pts = np.array(
        [[100.0, 100.0], [1180.0, 100.0], [700.0, 400.0], [580.0, 400.0]],
        dtype=np.float32,
    )
    mask = _poly_mask(shape, pts)
    det = perception.NeuralSheetDetector("dummy.onnx", verify=None)
    assert det._fit(mask, None) is None
    reject = det.last_fit_reject
    assert reject["reason"] == "implausible"
    assert reject["area_ratio"] > geometry._MIN_CONTOUR_AREA_RATIO


def test_rejeicao_de_suporte_reporta_unsupported():
    """A guarda de fill (#107) recusa quad sem apoio da máscara —
    o veto sobe como motivo próprio no rastro (issue #105)."""
    det = perception.NeuralSheetDetector("dummy.onnx", verify=None)
    # Folha na metade inferior + mancha conectada de "falso papel"
    # acima (o padrão s3_geometria das capturas oblíquas).
    mask = np.zeros((480, 640), np.uint8)
    mask[240:479, 60:600] = 255
    cv2.fillConvexPoly(
        mask,
        np.array([[30, 20], [300, 90], [380, 200], [200, 260], [20, 180]]),
        255,
    )
    assert det._fit(mask, None) is None
    reject = det.last_fit_reject
    assert reject["reason"] == "unsupported"
    assert reject["area_ratio"] >= geometry._MIN_CONTOUR_AREA_RATIO


def test_presenca_baixa_registra_area_da_mascara():
    """No modelo só-segmentação o logit ausente é fixo: a razão de
    área é o que separa folha pequena sob o piso de quadro vazio."""
    size = 100
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    det = perception.NeuralSheetDetector("dummy.onnx", verify=None)

    # Folha minúscula: 8x8 px de papel numa máscara de 100x100
    # (0,64% < 1% do piso de presença) — o logit ausente é fixo.
    logits = np.zeros((1, 4, size, size), dtype=np.float32)
    logits[0, 0] = 1.0  # fundo vence em todo lugar
    logits[0, 1, 40:48, 40:48] = 2.0  # papel vence num miolo de 8x8
    det._session = _DummySegSession(logits)
    corners, _, _ = det(frame)
    assert corners is None
    assert det.last_presence_logit == perception._SEGMENTATION_ABSENT_LOGIT
    reject = det.last_fit_reject
    assert reject["reason"] == "low_presence"
    assert reject["area_ratio"] == 64 / (size * size)

    # Quadro vazio de verdade: razão zero, distinguível da folha.
    logits_empty = np.zeros((1, 4, size, size), dtype=np.float32)
    logits_empty[0, 0] = 1.0
    det._session = _DummySegSession(logits_empty)
    corners, _, _ = det(frame)
    assert corners is None
    reject = det.last_fit_reject
    assert reject["reason"] == "low_presence"
    assert reject["area_ratio"] == 0.0


def main():
    test_predict_matches_legacy_softmax_and_normalization()
    test_middle_arm_notch_recovers_straight_chord_without_memory()
    test_curved_arm_with_high_coverage_does_not_poison_memory()
    test_rigid_guard_vetoes_single_edge_sleeve_but_allows_real_sheet_rotation()
    test_fit_sem_contorno_reporta_no_contour()
    test_fit_contorno_minusculo_reporta_min_area()
    test_fit_com_folha_fecha_e_limpa_o_reject()
    test_fit_implausivel_reporta_implausible()
    test_rejeicao_de_suporte_reporta_unsupported()
    test_presenca_baixa_registra_area_da_mascara()
    print("TESTES DE GUARDAS NEURAIS OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
