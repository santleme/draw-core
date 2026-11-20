"""Contrato e equivalência do mosaico da tinta página->limpa.

O caminho padrão do app manda os 20 blocos de 256 da página em um
lote só quando o arquivo ONNX declara lote dinâmico. Estes testes
provam três coisas com um ONNX de verdade (gerado pelo fixture):

1. o resultado do caminho em lote é idêntico ao do caminho de um
   bloco por chamada;
2. um arquivo com lote fixo continua funcionando (cai no caminho
   de um bloco por chamada);
3. um export com resolução de saída diferente do bloco falha na
   carga com erro nomeado, em vez de estourar de broadcast dentro
   da thread de tinta.

Sem o pacote `onnx` (só para gerar o arquivo), os testes saem como
skip; o onnxruntime continua sendo requisito de execução.
"""

import numpy as np
import pytest

from draw import page_clean

onnx_fixtures = pytest.importorskip("onnx_fixtures")


def _page(seed=3):
    rng = np.random.default_rng(seed)
    page = rng.integers(120, 240, (300, 220, 3), dtype=np.uint8)
    page[60:140, 40:160] = 20
    return page


def _position_sensitive_model(path):
    """ONNX [N,3,256,256] -> [N,1,256,256]: logit = (luminancia-0.5)*100.

    A saída DEPENDE do conteúdo de cada bloco: região escura vira
    logit fortemente negativo (prob≈0, fora da máscara) e região
    clara vira fortemente positivo — um bloco repetido ou trocado
    de lugar muda a máscara na posição, coisa que o fixture
    uniforme (logit positivo em todo pixel) não detectava.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["mean", "half"], ["centered"]),
        helper.make_node("Mul", ["centered", "gain"], ["logits"]),
    ]
    initializers = [
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half"),
        numpy_helper.from_array(np.array(100.0, dtype=np.float32), "gain"),
    ]
    graph = helper.make_graph(
        nodes,
        "draw-test-position-sensitive",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, 256, 256])],
        [
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, ["N", 1, 256, 256]
            )
        ],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def test_batched_mosaic_matches_one_tile_per_call(tmp_path):
    path = _position_sensitive_model(tmp_path / "pc.onnx")
    shared = dict(model_path=path, threshold=0.5, verify=None)

    batched = page_clean.PageCleanInk(**shared)
    per_tile = page_clean.PageCleanInk(batch_size=1, **shared)
    batched.load()
    per_tile.load()

    page = _page()
    a = batched.ink_mask(page)
    b = per_tile.ink_mask(page)

    assert a.shape == page.shape[:2]
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)
    # A máscara não é constante: a mancha escura de _page()
    # ([60:140, 40:160]) sai zerada e o fundo claro sai marcado —
    # um bloco repetido ou reordenado moveria a mancha.
    assert a[100, 100] == 0
    assert a[10, 10] == 255


def test_fixed_batch_file_uses_the_per_tile_path(tmp_path):
    path = onnx_fixtures.page_clean_model(
        tmp_path / "pc_fixed.onnx", patch=256, dynamic_batch=False
    )
    ink = page_clean.PageCleanInk(path, threshold=0.5, verify=None)
    ink.load()

    mask = ink.ink_mask(_page())

    assert mask.shape == (300, 220)
    assert np.count_nonzero(mask) > 0


def test_output_resolution_must_match_the_patch(tmp_path):
    path = onnx_fixtures.page_clean_model(
        tmp_path / "pc_bad.onnx", patch=256, output_scale=2
    )
    ink = page_clean.PageCleanInk(path, verify=None)

    with pytest.raises(RuntimeError, match="256x256"):
        ink.load()


def test_batch_size_helper_respects_dynamic_and_fixed_shapes():
    assert page_clean._batch_size(None, 20, True) == 20
    assert page_clean._batch_size(None, 20, False) == 1
    assert page_clean._batch_size(4, 20, True) == 4
    assert page_clean._batch_size(0, 20, True) == 20
    assert page_clean._batch_size(99, 20, True) == 20


def test_default_threads_never_exceed_the_host(monkeypatch):
    """O padrão de threads intra-op é o teto combinado (4)
    limitado às CPUs do host: pedir threads que o host não tem
    degrada a passada (auditoria de 2026-11-02: 2.150 ms com 8
    threads num host de 4 vCPUs contra 721 ms com 4). O argumento
    explícito manda no padrão."""
    monkeypatch.setattr(page_clean.os, "cpu_count", lambda: 4)
    assert page_clean._page_clean_threads() == 4
    assert page_clean.PageCleanInk("inexistente.onnx", verify=None).threads == 4
    monkeypatch.setattr(page_clean.os, "cpu_count", lambda: 16)
    assert page_clean._page_clean_threads() == page_clean.DEFAULT_THREADS
    monkeypatch.setattr(page_clean.os, "cpu_count", lambda: None)
    assert page_clean._page_clean_threads() == page_clean.DEFAULT_THREADS
    monkeypatch.setattr(page_clean.os, "cpu_count", lambda: 4)
    assert page_clean._page_clean_threads(2) == 2


def test_batch_size_comes_from_the_constructor():
    ink = page_clean.PageCleanInk("inexistente.onnx", batch_size=1, verify=None)
    assert ink.batch_size == 1
    assert page_clean.PageCleanInk("inexistente.onnx", verify=None).batch_size is None


def test_mosaic_covers_the_page_edges_without_seams(tmp_path):
    """Blocos de borda incompletos: a máscara cobre a página inteira."""
    path = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    ink = page_clean.PageCleanInk(path, threshold=0.5, verify=None)
    ink.load()

    # página com altura e largura que não são múltiplas de 256
    page = np.full((300, 220, 3), 10, dtype=np.uint8)
    mask = ink.ink_mask(page)

    assert mask.shape == (300, 220)
    assert np.all(mask == 255)


def test_verify_is_a_required_keyword_argument():
    """verify sai do default: esquecer de passá-lo falha na assinatura."""
    with pytest.raises(TypeError):
        page_clean.PageCleanInk("inexistente.onnx")


def test_threshold_comes_from_manifest_entry(tmp_path):
    path = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    ink = page_clean.PageCleanInk(path, verify=lambda _p: {"runtime_threshold": 0.79})
    ink.load()
    assert ink.threshold == 0.79


def test_threshold_falls_back_when_manifest_lacks_field(tmp_path):
    path = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    ink = page_clean.PageCleanInk(path, verify=lambda _p: {"sha256": "abc"})
    ink.load()
    assert ink.threshold == page_clean.DEFAULT_THRESHOLD


def test_explicit_threshold_beats_manifest(tmp_path):
    path = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    ink = page_clean.PageCleanInk(
        path, threshold=0.42, verify=lambda _p: {"runtime_threshold": 0.79}
    )
    ink.load()
    assert ink.threshold == 0.42


def test_verify_none_keeps_default_threshold(tmp_path):
    path = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    ink = page_clean.PageCleanInk(path, verify=None)
    ink.load()
    assert ink.threshold == page_clean.DEFAULT_THRESHOLD


@pytest.mark.parametrize("bad", [1.5, 0.0, -0.2, "alto", True, None])
def test_invalid_manifest_threshold_is_refused(tmp_path, bad):
    path = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    ink = page_clean.PageCleanInk(path, verify=lambda _p: {"runtime_threshold": bad})
    with pytest.raises(RuntimeError, match="runtime_threshold"):
        ink.load()


def test_resolve_threshold_absent_field_falls_back_but_null_is_refused():
    """Ausência = manifesto antigo (fallback); null = calibração quebrada."""
    assert page_clean.resolve_threshold({}) == page_clean.DEFAULT_THRESHOLD
    assert page_clean.resolve_threshold(None) == page_clean.DEFAULT_THRESHOLD
    with pytest.raises(RuntimeError, match="runtime_threshold"):
        page_clean.resolve_threshold({"runtime_threshold": None})


def test_live_vote_threshold_is_decoupled_from_mask_threshold():
    """O limiar calibrado da máscara não muda o voto ao vivo.

    Identidade de página, âncora da fronteira e camada provisória
    seguem InkConfig.seen_threshold (0,7 legado) mesmo com a máscara
    em 0,85 — review do PR #43.
    """
    from draw.ink import worker as worker_mod
    from draw.ink.config import InkConfig

    class _Ink:
        threshold = 0.85

    worker = worker_mod.PageCleanWorker(_Ink())
    assert worker._accumulator.threshold == InkConfig().seen_threshold == 0.7
    tuned = worker_mod.PageCleanWorker(_Ink(), config=InkConfig(seen_threshold=0.6))
    assert tuned._accumulator.threshold == 0.6


# --- pagina inteira (arquivo de lado dinamico) ------------------------------


def _textured(shape, seed=11):
    """Pagina com conteudo por pixel: desalinhamento de 1 px muda a saida."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (*shape, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "shape",
    [(1200, 850), (850, 1200), (853, 1201), (8, 8)],
    ids=["retrato", "paisagem", "lados-impares", "minima"],
)
def test_full_page_equals_mosaic_in_both_orientations(tmp_path, shape):
    # Modelo por pixel: a pagina inteira tem de reproduzir o mosaico
    # pixel a pixel. Padding no lugar errado ou recorte deslocado
    # mudaria a saida (o conteudo e ruido por pixel).
    full = page_clean.PageCleanInk(
        onnx_fixtures.page_clean_model(tmp_path / "full.onnx", dynamic_side=True),
        verify=None,
    )
    tiled = page_clean.PageCleanInk(
        onnx_fixtures.page_clean_model(tmp_path / "tiled.onnx", patch=256),
        verify=None,
    )
    full.load()
    tiled.load()
    assert full._layout == "page" and tiled._layout == "tiles"
    page = _textured(shape)
    a = full.ink_prob(page)
    assert a.shape == shape
    assert np.array_equal(a, tiled.ink_prob(page))


@pytest.mark.parametrize(
    ("shape", "padded"),
    [
        ((1200, 850), (1200, 856)),
        ((850, 1200), (856, 1200)),
        ((1201, 853), (1208, 856)),
    ],
)
def test_full_page_pads_only_to_the_stride_multiple(tmp_path, shape, padded):
    ink = page_clean.PageCleanInk(
        onnx_fixtures.page_clean_model(tmp_path / "full.onnx", dynamic_side=True),
        verify=None,
    )
    ink.load()
    seen = []
    real_run = ink._session.run

    class _Spy:
        def run(self, names, feeds):
            seen.append(next(iter(feeds.values())).shape)
            return real_run(names, feeds)

    ink._session = _Spy()
    ink.ink_prob(_textured(shape))
    assert seen == [(1, 3, *padded)], "uma chamada so, padding minimo embaixo/direita"


def test_full_page_output_must_keep_the_input_size(tmp_path):
    path = onnx_fixtures.page_clean_model(
        tmp_path / "half.onnx", patch=256, output_scale=2, dynamic_side=True
    )
    ink = page_clean.PageCleanInk(path, verify=None)
    ink.load()
    with pytest.raises(RuntimeError, match="tamanho da entrada"):
        ink.ink_prob(_textured((120, 80)))


def test_dynamic_input_with_fixed_output_is_refused_at_load(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["image"], ["logits"], axes=[1], keepdims=1)],
        "draw-test-mixed",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, "H", "W"])],
        [
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, ["N", 1, 256, 256]
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(tmp_path / "mixed.onnx"))
    ink = page_clean.PageCleanInk(tmp_path / "mixed.onnx", verify=None)
    with pytest.raises(RuntimeError, match="lado dinamico"):
        ink.load()
