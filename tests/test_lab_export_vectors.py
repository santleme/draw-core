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


def test_main_model_proof_records_calibrated_mask_threshold(tmp_path, monkeypatch):
    """A prova registra o runtime_threshold do manifesto (não o fallback)
    e não o vende como o limiar do voto ao vivo."""
    onnx_fixtures = pytest.importorskip("onnx_fixtures")
    from draw.infra import manifest
    from draw.page_clean import DEFAULT_THRESHOLD

    calibrated = 0.79
    assert calibrated != DEFAULT_THRESHOLD
    model = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    seen = []

    def fake_verify(path):
        seen.append(path)
        return {"runtime_threshold": calibrated}

    # Sem --allow-unverified: o caminho "manifest" injeta verify_model.
    monkeypatch.setattr(manifest, "verify_model", fake_verify)
    page = tmp_path / "pagina.png"
    _page(page)
    out = tmp_path / "out"
    argv = [str(page), "--out-dir", str(out), "--mask", "model"]
    assert main(argv + ["--onnx", str(model)]) == 0
    assert seen, "o verificador do manifesto não foi chamado"
    source = json.loads((out / "pagina_proof.json").read_text())["mask_source"]
    assert f"quadro unico {calibrated};" in source
    assert "seen_threshold" in source


def test_main_allow_unverified_skips_verifier_and_records_fallback(
    tmp_path, monkeypatch
):
    """--allow-unverified: verify=None — o verificador não roda e a prova
    registra DEFAULT_THRESHOLD, nunca uma calibração de manifesto."""
    onnx_fixtures = pytest.importorskip("onnx_fixtures")
    from draw.infra import manifest
    from draw.page_clean import DEFAULT_THRESHOLD

    def must_not_run(path):
        raise AssertionError(f"verify_model chamado com --allow-unverified: {path}")

    monkeypatch.setattr(manifest, "verify_model", must_not_run)
    model = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    page = tmp_path / "pagina.png"
    _page(page)
    out = tmp_path / "out"
    argv = [str(page), "--out-dir", str(out), "--mask", "model"]
    assert main(argv + ["--onnx", str(model), "--allow-unverified"]) == 0
    source = json.loads((out / "pagina_proof.json").read_text())["mask_source"]
    assert f"quadro unico {DEFAULT_THRESHOLD};" in source


def test_main_threshold_override_for_older_model(tmp_path):
    """Modelo antigo sob --allow-unverified não fica preso no
    DEFAULT_THRESHOLD do modelo vigente: --threshold vence e chega à
    máscara exportada, não só ao relatório (review #47).

    O fixture devolve sigmoid(luminância): papel ~0,72, traço ~0,53,
    nada acima de 0,73. Em 0,6 o papel entra na máscara; em 0,85 (o
    round-v1) e no default 0,97 a máscara sai vazia.
    """
    onnx_fixtures = pytest.importorskip("onnx_fixtures")
    from draw.page_clean import DEFAULT_THRESHOLD

    assert DEFAULT_THRESHOLD != 0.85
    model = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    page = tmp_path / "pagina.png"
    _page(page)

    def run(*extra):
        out = tmp_path / ("out" + "_".join(extra))
        argv = [str(page), "--out-dir", str(out), "--mask", "model"]
        assert main(argv + ["--onnx", str(model), "--allow-unverified", *extra]) == 0
        proof = json.loads((out / "pagina_proof.json").read_text())
        mask = cv2.imread(str(out / "pagina_mask.png"), cv2.IMREAD_GRAYSCALE)
        # imread devolve None se o PNG nao abre, e count_nonzero(None) e 0:
        # sem esta checagem o caso "mascara vazia" passaria sem arquivo.
        assert mask is not None and mask.shape == (120, 85), "mascara ilegivel"
        return proof["mask_source"], int(np.count_nonzero(mask))

    source, px_085 = run("--threshold", "0.85")
    assert "quadro unico 0.85;" in source
    _source, px_060 = run("--threshold", "0.6")
    _source, px_default = run()
    assert px_060 > 1000  # o limiar chegou à máscara: papel acima de 0,6
    assert px_085 == px_default == 0


@pytest.mark.parametrize("bad", ["0", "1", "1.5", "-0.1"])
def test_main_threshold_outside_open_interval_is_refused(tmp_path, bad):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "x.png"), "--threshold", bad])
