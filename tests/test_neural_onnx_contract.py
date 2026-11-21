"""Contratos ONNX do detector, com sessões falsas e com ONNX de verdade.

A primeira metade usa sessões falsas em Python (sem onnxruntime, sem
webcam) e fixa o contrato do adaptador: qual canal é papel, o que o
veredito de presença significa, o que acontece com saídas
inesperadas. A segunda metade gera arquivos ONNX minúsculos e passa
pelo ONNX Runtime de verdade — é a única forma de exercitar o que a
sessão falsa não vê: o nome da entrada no arquivo, a forma declarada
e o lote simbólico. Sem o pacote `onnx`, esses testes saem como skip.
"""

import cv2
import numpy as np
import pytest

from draw.geometry import fit_plausible
from draw.perception import NeuralSheetDetector


class _Output:
    def __init__(self, name):
        self.name = name


class _Input:
    def __init__(self, name="image", shape=None):
        self.name = name
        self.shape = shape or [1, 3, 8, 8]


class _Session:
    """Sessão falsa que confere o feed contra o próprio contrato.

    O nome da entrada é conferido de propósito: o adaptador antigo
    mandava o tensor sempre como "image" e uma sessão que aceitasse
    qualquer chave nunca pegaria o erro.
    """

    def __init__(self, outputs, input_name="image", input_shape=None):
        self._outputs = outputs
        self._input = _Input(input_name, input_shape)

    def get_outputs(self):
        return [_Output(name) for name in self._outputs]

    def get_inputs(self):
        return [self._input]

    def run(self, output_names, feed):
        assert list(feed) == [self._input.name], (
            f"feed com {list(feed)} numa entrada {self._input.name!r}"
        )
        assert output_names == list(self._outputs)
        return [self._outputs[name] for name in output_names]


def _logits_for_classes(classes, class_count=4):
    logits = np.full((1, class_count, *classes.shape), -4.0, dtype=np.float32)
    for class_id in range(class_count):
        logits[0, class_id][classes == class_id] = 4.0
    return logits


def test_four_class_logits_use_paper_and_ink_but_exclude_hand():
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[1:7, 1:7] = 1
    classes[2:4, 2:4] = 2
    classes[4:6, 4:6] = 3
    session = _Session({"logits": _logits_for_classes(classes)})
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    presence, mask = detector.predict(np.zeros((32, 48, 3), dtype=np.uint8))

    assert presence == 1.0
    assert mask.shape == (32, 48)
    assert np.all(mask[8:16, 12:24] == 0)  # mão não vira papel
    assert np.all(mask[16:24, 24:36] == 255)  # tinta integra a folha
    assert mask[0, 0] == 0


def test_four_class_logits_without_page_are_a_strong_negative():
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[1:3, 1:3] = 2
    session = _Session({"logits": _logits_for_classes(classes)})
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    presence, mask = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert presence == -10.0
    assert np.count_nonzero(mask) == 0


def test_legacy_mask_and_presence_outputs_keep_their_contract():
    logits = np.full((1, 2, 8, 8), -4.0, dtype=np.float32)
    logits[0, 1, 1:7, 1:7] = 4.0
    logits[0, 0, 1:7, 1:7] = -4.0
    session = _Session(
        {
            "mask_logits": logits,
            "presence_logit": np.array([2.5], dtype=np.float32),
        }
    )
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    presence, mask = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert presence == 2.5
    assert mask[4, 4] == 255
    assert mask[0, 0] == 0


def test_four_class_logits_expose_hand_mask():
    """A classe mão sai em last_hand_mask (segunda fonte de oclusão)."""
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[1:7, 1:7] = 1
    classes[2:4, 2:4] = 2
    session = _Session({"logits": _logits_for_classes(classes)})
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    hand = detector.last_hand_mask
    assert hand is not None and hand.shape == (8, 8)
    assert np.all(hand[2:4, 2:4] == 255)
    assert hand.sum() / 255 == 4
    assert np.all(hand[1:7, 1:7][classes[1:7, 1:7] != 2] == 0)


def test_two_channel_contract_has_no_hand_mask():
    """O contrato legado (2 canais) não sabe o que é mão: None."""
    logits = np.full((1, 2, 8, 8), -4.0, dtype=np.float32)
    logits[0, 1, 1:7, 1:7] = 4.0
    session = _Session(
        {
            "mask_logits": logits,
            "presence_logit": np.array([2.5], dtype=np.float32),
        }
    )
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert detector.last_hand_mask is None


def test_input_name_comes_from_the_file_not_from_a_literal():
    """A entrada pode se chamar "input": o feed tem de seguir o arquivo."""
    logits = np.zeros((1, 2, 8, 8), dtype=np.float32)
    session = _Session(
        {"mask_logits": logits, "presence_logit": np.array([1.0], np.float32)},
        input_name="input",
    )
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))
    assert detector.contract().input_name == "input"


def test_rectangular_input_is_refused_instead_of_squeezed():
    session = _Session(
        {"logits": _logits_for_classes(np.zeros((8, 8), np.uint8))},
        input_shape=[1, 3, 384, 640],
    )
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    with pytest.raises(RuntimeError, match="retangular"):
        detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))


def test_dynamic_input_falls_back_to_the_measured_resolution():
    session = _Session(
        {"logits": _logits_for_classes(np.zeros((8, 8), np.uint8))},
        input_shape=["N", 3, "H", "W"],
    )
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    assert detector.contract().size == (384, 384)


def test_four_channel_mask_logits_with_presence_use_paper_and_ink():
    """Export híbrido: 4 canais + presença treinada (não ler como fundo x 1)."""
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[1:7, 1:7] = 1
    classes[2:4, 2:4] = 2  # mão
    session = _Session(
        {
            "mask_logits": _logits_for_classes(classes),
            "presence_logit": np.array([6.0], np.float32),
        }
    )
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    presence, mask = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert presence == 6.0
    # a máscara de 8x8 sobe para 16x16 por vizinho mais próximo:
    # cada célula de classe vira 2x2 px
    assert np.all(mask[10:14, 10:14] == 255)  # papel
    assert np.all(mask[4:7, 4:7] == 0)  # região de mão fica fora


def test_mask_logits_with_three_channels_is_a_named_error():
    session = _Session(
        {
            "mask_logits": np.zeros((1, 3, 8, 8), np.float32),
            "presence_logit": np.array([1.0], np.float32),
        }
    )
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    with pytest.raises(ValueError, match="2 canais"):
        detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))


def test_extra_auxiliary_output_does_not_break_the_four_class_branch():
    classes = np.zeros((8, 8), dtype=np.uint8)
    classes[2:6, 2:6] = 1
    logits = _logits_for_classes(classes)
    session = _Session({"logits": logits, "aux": logits})
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    presence, mask = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert presence == 1.0
    assert np.count_nonzero(mask) > 0


def test_incompatible_outputs_are_a_named_error():
    session = _Session({"mystery": np.zeros((1, 2, 8, 8), np.float32)})
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session

    with pytest.raises(ValueError, match="saídas ONNX incompatíveis"):
        detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))


# --------------------------------------------------------------- guarda
def test_plausibility_guard_refuses_non_finite_quads():
    """NaN atravessava a guarda: toda comparação com NaN é falsa."""
    nan_quad = np.array(
        [[np.nan, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    inf_quad = np.array(
        [[np.inf, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    assert fit_plausible(nan_quad, (720, 1280)) is False
    assert fit_plausible(inf_quad, (720, 1280)) is False


def test_plausibility_guard_still_accepts_a_normal_sheet():
    quad = np.array(
        [[300.0, 200.0], [1000.0, 180.0], [980.0, 650.0], [320.0, 660.0]],
        dtype=np.float32,
    )
    assert fit_plausible(quad, (720, 1280)) is True


def test_plausibility_guard_accepts_clipped_sheet_with_border_short_edge():
    # Issue #104: folha cortada pelo frame tem a aresta curta alinhada a
    # borda — e o recorte do frame, nao um colapso. A razao entre opostas
    # (aqui ~9x) nao pode rejeitar. A aresta a 10 px da borda segue dentro
    # da tolerancia de ~2% que cobre a quantizacao da mascara nativa.
    quad = np.array(
        [[10.0, 200.0], [10.0, 260.0], [800.0, 560.0], [450.0, 120.0]],
        dtype=np.float32,
    )
    assert fit_plausible(quad, (720, 1280)) is True


def test_plausibility_guard_accepts_clip_that_truncates_short_edge_corner():
    # O corte pode tomar um lado inteiro do poligono visivel e encurtar a
    # aresta adjacente: ela fica com um vertice na borda e outro no papel.
    # O corte vale porque o contorno da mascara nativa alcanca a borda.
    quad = np.array(
        [[0.0, 100.0], [180.0, 130.0], [850.0, 650.0], [0.0, 680.0]],
        dtype=np.float32,
    )
    mask = np.zeros((72, 128), np.uint8)
    mask[10:69, :85] = 255
    assert fit_plausible(quad, (720, 1280), mask, 10.0, 10.0) is True


def test_plausibility_guard_tries_all_borders_for_corner_cut():
    # Perto do canto, o corte vizinho cabe na tolerancia de duas bordas
    # (aqui topo e direita). A mascara so suporta a segunda: a linha 0 e
    # vazia mas a coluna 127 tem papel. Checar so a primeira rejeitaria
    # uma folha clipada valida.
    quad = np.array(
        [[1267.0, 7.0], [1274.0, 13.0], [1150.0, 60.0], [700.0, 90.0]],
        dtype=np.float32,
    )
    mask = np.zeros((72, 128), np.uint8)
    mask[1:60, 64:] = 255
    assert fit_plausible(quad, (720, 1280), mask, 10.0, 10.0) is True
    mask[:, 127] = 0  # sem suporte na borda direita, nada isenta
    assert fit_plausible(quad, (720, 1280), mask, 10.0, 10.0) is False


def test_plausibility_guard_accepts_sheet_clipped_along_long_edge():
    # A familia medida na issue #104: a aresta-corte e a LONGA do par —
    # o comprimento e decidido pela borda da imagem; a curta e a aresta
    # distante real, interior. Sem suporte de mascara na borda, rejeita.
    quad = np.array(
        [[100.0, 50.0], [400.0, 70.0], [1200.0, 714.0], [80.0, 718.0]],
        dtype=np.float32,
    )
    mask = np.zeros((72, 128), np.uint8)
    mask[5:72, 8:] = 255
    assert fit_plausible(quad, (720, 1280), mask, 10.0, 10.0) is True
    mask[71, :] = 0  # papel para uma linha antes da borda: nao e corte
    assert fit_plausible(quad, (720, 1280), mask, 10.0, 10.0) is False


def test_plausibility_guard_ignores_detached_strip_on_border():
    # Uma faixa de papel destacada tocando a borda no mesmo trecho nao
    # prova que a FOLHA ajustada alcanca a borda: a evidencia do corte
    # e restrita aos pixels dentro do contorno selecionado para o fit.
    quad = np.array(
        [[80.0, 40.0], [340.0, 60.0], [1180.0, 9.0], [100.0, 9.0]],
        dtype=np.float32,
    )
    mask = np.zeros((72, 128), np.uint8)
    mask[0, 10:118] = 255  # faixa destacada na linha do topo
    mask[2:61, 8:118] = 255  # a folha ajustada, um componente separado
    contour = np.array([[[8, 2]], [[117, 2]], [[117, 60]], [[8, 60]]], dtype=np.int32)
    # Sem o contorno a faixa fornecia suporte falso; com ele, rejeita.
    assert fit_plausible(quad, (720, 1280), mask, 10.0, 10.0) is True
    assert fit_plausible(quad, (720, 1280), mask, 10.0, 10.0, contour=contour) is False


def test_plausibility_guard_treats_hand_on_border_as_unobserved():
    # Mao cobrindo o meio do trecho de borda: os pixels de mao sao
    # "nao observaveis" — o corte segue valido se a metade observada
    # e papel; mas mao sozinha nao estabelece corte nenhum.
    quad = np.array(
        [[100.0, 9.0], [1180.0, 9.0], [400.0, 300.0], [80.0, 280.0]],
        dtype=np.float32,
    )
    mask = np.zeros((72, 128), np.uint8)
    mask[0:30, 10:118] = 255  # folha clipada pelo topo
    mask[0:6, 40:100] = 0  # a mao apaga o meio do trecho na borda
    hand = np.zeros((72, 128), np.uint8)
    hand[0:6, 40:100] = 255
    contour = np.array([[[10, 0]], [[117, 0]], [[117, 29]], [[10, 29]]], dtype=np.int32)
    kwargs = dict(mask=mask, scale_x=10.0, scale_y=10.0, contour=contour, hand=hand)
    assert fit_plausible(quad, (720, 1280), **kwargs) is True
    mask[0:6, :] = 0  # sem papel observado na borda, mao nao basta
    assert fit_plausible(quad, (720, 1280), **kwargs) is False


def test_detector_rejects_fit_grazing_border_at_one_corner():
    # Pelo caminho do detector: trapezio que encosta no topo so num
    # canto — a linha do topo e quase toda fundo visivel, entao a
    # "aresta no topo" nao e corte de frame e a razao rejeita. Ja uma
    # folha realmente clipada (linha 0 toda papel) segue aceita.
    det = NeuralSheetDetector("unused.onnx", verify=None)
    grazing = np.zeros((480, 640), np.uint8)
    cv2.fillConvexPoly(
        grazing,
        np.array([[40, 0], [450, 8], [280, 280], [200, 280]], np.int32),
        255,
    )
    assert det._fit(grazing) is None
    clipped = np.zeros((480, 640), np.uint8)
    cv2.fillConvexPoly(
        clipped,
        np.array([[40, 0], [450, 0], [300, 260], [250, 260]], np.int32),
        255,
    )
    assert det._fit(clipped) is not None


def test_plausibility_guard_refuses_paper_edge_passing_near_border():
    # Uma aresta de papel a ~12 px da borda NAO e corte de frame: a
    # mascara nativa nao alcanca a borda ali, entao a aresta vizinha
    # colapsada continua rejeitada.
    quad = np.array(
        [[12.0, 100.0], [180.0, 130.0], [850.0, 650.0], [12.0, 680.0]],
        dtype=np.float32,
    )
    mask = np.zeros((72, 128), np.uint8)
    mask[10:69, 1:85] = 255  # papel para na coluna 1: borda sem suporte
    assert fit_plausible(quad, (720, 1280), mask, 10.0, 10.0) is False


def test_plausibility_guard_refuses_short_edge_with_one_vertex_near_border():
    # Um vertice perto da borda nao basta: o corte de frame cola a aresta
    # inteira numa borda so. Aresta curta que mergulha no quadro e colapso.
    quad = np.array(
        [[600.0, 5.0], [650.0, 250.0], [900.0, 700.0], [140.0, 690.0]],
        dtype=np.float32,
    )
    assert fit_plausible(quad, (720, 1280)) is False


def test_plausibility_guard_still_refuses_collapsed_interior_short_edge():
    # A assinatura do colapso por oclusao e a aresta curta interior ao
    # quadro: a razao entre opostas continua rejeitando.
    quad = np.array(
        [[300.0, 300.0], [340.0, 310.0], [900.0, 700.0], [280.0, 690.0]],
        dtype=np.float32,
    )
    assert fit_plausible(quad, (720, 1280)) is False


# ------------------------------------------------------- ONNX de verdade
onnx_fixtures = pytest.importorskip("onnx_fixtures")


def _detector_for(path):
    detector = NeuralSheetDetector(path, verify=None)
    detector.load()
    return detector


def test_real_onnx_with_input_named_input(tmp_path):
    path = onnx_fixtures.sheet_model(
        tmp_path / "legacy.onnx", input_name="input", size=64, channels=2
    )
    detector = _detector_for(path)

    assert detector.contract().input_name == "input"
    presence, mask = detector.predict(np.full((120, 160, 3), 200, dtype=np.uint8))

    assert presence > 0
    assert mask.shape == (120, 160)


def test_real_onnx_four_class_with_auxiliary_output(tmp_path):
    path = onnx_fixtures.four_class_model(
        tmp_path / "four.onnx", size=64, extra_output=True
    )
    detector = _detector_for(path)

    assert detector.contract().outputs == ("logits", "aux")
    presence, mask = detector.predict(np.full((64, 64, 3), 230, dtype=np.uint8))
    assert presence in (1.0, -10.0)
    assert mask.shape == (64, 64)


def test_verify_is_a_required_keyword_argument():
    """verify sai do default: esquecer de passá-lo falha na assinatura."""
    import pytest

    with pytest.raises(TypeError):
        NeuralSheetDetector("unused.onnx")
