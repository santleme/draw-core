"""Driver lab/export_vectors: orquestrador e CLI de ponta a ponta
(--mask otsu não precisa de ONNX; o caminho --mask model é o
PageCleanInk já coberto por tests/test_page_clean.py)."""

import json

import cv2
import numpy as np
import pytest
from lab.export_vectors import export_vectors, main

from draw import config


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


def _canvas(w, h):
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.line(img, (w // 12, h // 8), (w - w // 12, h // 8), 0, 5)
    cv2.line(img, (w // 12, h // 8), (w // 12, h - h // 8), 0, 5)
    return img


def test_landscape_canvas_exports_with_uniform_scale():
    """Canvas salvo em paisagem (1200x850): escala igual nos dois eixos,
    folha 297x210. Antes saía 0,175 x 0,349 mm/px: 1000 px viravam
    175 mm em vez de 247,5 (review #50)."""
    result = export_vectors(_canvas(1200, 850), mask_source="otsu")
    assert result["orientation"] == "paisagem"
    assert result["paper_mm"] == (297.0, 210.0)
    assert result["scale_mm_per_px"] == pytest.approx((0.2475, 0.2475))


def test_portrait_canvas_is_unchanged():
    result = export_vectors(_canvas(850, 1200), mask_source="otsu")
    assert result["orientation"] == "retrato"
    assert result["paper_mm"] == (210.0, 297.0)
    assert result["scale_mm_per_px"] == pytest.approx((0.2475, 0.2475))


@pytest.mark.parametrize("name", list(config.FORMATS))
@pytest.mark.parametrize("orientation", config.ORIENTATIONS)
def test_every_format_exports_uniform_scale_in_both_orientations(name, orientation):
    """A grade canônica de qualquer formato, nas duas orientações, tem
    a escala uniforme mm_per_px — nada de eixo esticado."""
    w, h = config.paper_grid(name, orientation)
    result = export_vectors(
        _canvas(w, h), mask_source="otsu", paper_format=name, orientation=orientation
    )
    unit = config.mm_per_px(name)
    assert result["scale_mm_per_px"] == pytest.approx((unit, unit))
    assert result["paper_mm"] == config.paper_mm(name, orientation)


def test_orientation_can_be_forced_and_validated():
    img = _canvas(1200, 850)
    assert (
        export_vectors(img, mask_source="otsu", orientation="retrato")["orientation"]
        == "retrato"
    )
    with pytest.raises(ValueError, match="orientacao"):
        export_vectors(img, mask_source="otsu", orientation="diagonal")


def test_main_landscape_png_writes_landscape_svg(tmp_path):
    """Caminho completo do CLI com o PNG que a tecla s salva em paisagem."""
    page = tmp_path / "canvas.png"
    cv2.imwrite(str(page), _canvas(1200, 850))
    out = tmp_path / "out"
    assert main([str(page), "--out-dir", str(out), "--mask", "otsu"]) == 0
    svg = (out / "canvas.svg").read_text()
    assert 'width="297.0mm" height="210.0mm"' in svg
    proof = json.loads((out / "canvas_proof.json").read_text())
    assert proof["orientation"] == "paisagem"
    assert proof["paper_mm"] == [297.0, 210.0]
    assert proof["iou_svg_roundtrip"] > 0.5
    assert proof["iou_dxf_roundtrip"] > 0.5


# ------------------------------------------- trace O(H*W) e afinamento nativo


def test_index_node_components_matches_the_naive_per_node_scan():
    import cv2

    from draw.exporters import export_vectors as ev

    rng = np.random.default_rng(3)
    mask = (rng.random((90, 130)) > 0.93).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    anchors, pixels = ev._index_node_components(labels, n)
    width = labels.shape[1]
    for nid in range(1, n):
        ys, xs = np.nonzero(labels == nid)
        assert anchors[nid] == (float(xs.mean()), float(ys.mean()))
        assert pixels[nid] == [
            int(y) * width + int(x) for y, x in zip(ys, xs, strict=True)
        ]
    assert set(anchors) == set(range(1, n))
    assert ev._index_node_components(np.zeros((5, 5), np.int32), 1) == ({}, {})


@pytest.mark.slow
def test_trace_polylines_scales_with_nodes_not_nodes_times_page():
    import time

    from draw.exporters import export_vectors as ev

    skel = np.zeros((1200, 850), bool)
    for k in range(
        3000
    ):  # 3000 pontas/nos dispersos: o laco antigo varria 1 M px por no
        y, x = (k // 50) * 20 + 3, (k % 50) * 16 + 3
        skel[y, x : x + 3] = True
    start = time.perf_counter()
    lines = ev.trace_polylines(skel)
    assert time.perf_counter() - start < 2.0
    assert len(lines) >= 3000


def test_native_thinning_is_opt_in_valid_and_close_to_numpy():
    import cv2

    from draw.exporters import export_vectors as ev

    if not hasattr(cv2, "ximgproc"):
        pytest.skip("cv2.ximgproc indisponivel")
    ink = np.zeros((160, 220), np.uint8)
    cv2.line(ink, (10, 20), (200, 120), 1, 5)
    cv2.circle(ink, (110, 80), 40, 1, 4)
    ink[150:152, 5:7] = 1  # componente minimo: precisa manter 1 px
    default = ev.zhang_suen_thin(ink > 0)
    native = ev.zhang_suen_thin(ink > 0, native=True)
    assert not (native & ~(ink > 0)).any()
    assert native[150:152, 5:7].any()
    assert (native ^ default).sum() <= 0.05 * default.sum()
