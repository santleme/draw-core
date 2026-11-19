"""Driver lab/export_vectors: orquestrador e CLI de ponta a ponta
(--mask otsu não precisa de ONNX; o caminho --mask model é o
PageCleanInk já coberto por tests/test_page_clean.py)."""

import json

import cv2
import numpy as np
import pytest
from lab.export_vectors import export_vectors, main


def _page(path):
    img = np.full((120, 85, 3), 245, np.uint8)
    cv2.line(img, (10, 10), (75, 100), 30, 6)
    cv2.line(img, (10, 60), (70, 60), 30, 5)
    cv2.imwrite(str(path), img)
    return img


def test_export_vectors_otsu():
    img = np.full((120, 85, 3), 245, np.uint8)
    cv2.line(img, (10, 10), (75, 100), 30, 6)
    result = export_vectors(img, mask_source="otsu")
    assert result["ink"].sum() > 0
    assert result["polylines"]
    sx, sy = result["scale_mm_per_px"]
    # Fora da grade canônica A4 (850x1200): escala por eixo das
    # dimensões do formato, não o mm/px uniforme do formato.
    assert sx == pytest.approx(210 / 85)
    assert sy == pytest.approx(297 / 120)


def test_export_vectors_model_missing(tmp_path):
    img = np.full((20, 20, 3), 245, np.uint8)
    with pytest.raises(FileNotFoundError):
        export_vectors(img, mask_source="model", onnx_path=tmp_path / "x.onnx")


def test_main_end_to_end_otsu(tmp_path):
    page = tmp_path / "pagina.png"
    _page(page)
    out = tmp_path / "out"
    assert main([str(page), "--out-dir", str(out), "--mask", "otsu"]) == 0
    proof = json.loads((out / "pagina_proof.json").read_text())
    for artifact in (
        "pagina.svg",
        "pagina.dxf",
        "pagina_overlay.png",
        "pagina_mask.png",
    ):
        assert (out / artifact).exists(), artifact
    assert proof["polylines"] >= 1
    assert proof["iou_svg_roundtrip"] > 0.5
    assert proof["mask_source"].startswith("otsu")


def test_main_input_missing(tmp_path, capsys):
    assert main([str(tmp_path / "nada.png"), "--out-dir", str(tmp_path)]) == 1
    assert "não foi possível ler" in capsys.readouterr().err


def test_main_model_proof_names_mask_threshold_not_live_vote(tmp_path):
    """A prova registra o limiar efetivo da máscara e não o vende como o voto ao vivo."""
    onnx_fixtures = pytest.importorskip("onnx_fixtures")
    from draw.page_clean import DEFAULT_THRESHOLD

    model = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    page = tmp_path / "pagina.png"
    _page(page)
    out = tmp_path / "out"
    argv = [str(page), "--out-dir", str(out), "--mask", "model"]
    argv += ["--onnx", str(model), "--allow-unverified"]
    assert main(argv) == 0
    source = json.loads((out / "pagina_proof.json").read_text())["mask_source"]
    assert f"quadro unico {DEFAULT_THRESHOLD}" in source
    assert "seen_threshold" in source
