"""Receita do page_clean INT8 (lab/quantize_page_clean.py): o que o CI alcanca."""

import pytest

onnx = pytest.importorskip("onnx")
onnx_fixtures = pytest.importorskip("onnx_fixtures")

from lab import quantize_page_clean as qpc  # noqa: E402


def test_dynamic_side_drops_unused_domains_and_frees_height_width(tmp_path):
    # O quant_pre_process declara dominios que nenhum no usa; o
    # ai.onnx.ml 5 fazia o onnxruntime 1.17 (piso do pyproject) recusar
    # o arquivo publicado (review do PR #66).
    src = onnx_fixtures.page_clean_model(tmp_path / "fixed.onnx", patch=256)
    model = onnx.load(str(src))
    model.opset_import.extend(
        [
            onnx.helper.make_opsetid("ai.onnx.ml", 5),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ]
    )
    onnx.save(model, str(src))

    qpc.dynamic_side(src, tmp_path / "dyn.onnx")
    out = onnx.load(str(tmp_path / "dyn.onnx"))
    assert {op.domain for op in out.opset_import} == {""}
    for value in (out.graph.input[0], out.graph.output[0]):
        dims = value.type.tensor_type.shape.dim
        assert [d.dim_param for d in dims[2:]] == ["height", "width"]


def test_load_pages_reads_only_the_given_archives(tmp_path):
    import numpy as np

    keep, stale = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez_compressed(keep, np.zeros((4, 4, 3), np.uint8))
    np.savez_compressed(
        stale, np.ones((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)
    )
    assert len(qpc.load_pages([keep])) == 1  # so o que esta captura escreveu
    assert len(qpc.load_pages(tmp_path)) == 3  # pasta explicita (--pages): tudo


def test_calibration_never_uses_the_lossy_intermediate_output_cap(
    tmp_path, monkeypatch
):
    # No onnxruntime 1.23, CalibMaxIntermediateOutputs limpa o buffer ao
    # encher SEM calcular os limites: so o ultimo lote calibrava (review
    # do PR #66). A receita tem de ver todos os blocos.
    import shutil

    import numpy as np
    from onnxruntime import quantization

    seen = {}

    def fake_quantize_static(src, dst, reader, **kwargs):
        seen.update(kwargs)
        seen["tiles"] = 0
        while reader.get_next() is not None:
            seen["tiles"] += 1
        shutil.copy(src, dst)

    monkeypatch.setattr(quantization, "quantize_static", fake_quantize_static)
    src = onnx_fixtures.page_clean_model(tmp_path / "fixed.onnx", patch=256)
    pages = [np.zeros((300, 520, 3), np.uint8)] * 4
    used = qpc.quantize(src, pages, tmp_path / "out.onnx")
    assert used == 2
    assert "CalibMaxIntermediateOutputs" not in (seen.get("extra_options") or {})
    assert seen["tiles"] == 2 * 2 * 3  # 2 paginas de calibracao x 6 blocos
