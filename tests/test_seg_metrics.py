import json
import time

import numpy as np
import pytest
from lab import seg_metrics


def _mask_with_all_classes():
    """Máscara 10x10 com as 4 classes presentes, para métricas exatas."""
    gt = np.zeros((10, 10), dtype=np.int64)
    gt[0:5, 0:5] = 1  # paper
    gt[0:5, 5:10] = 2  # hand
    gt[5:10, 0:5] = 3  # ink
    return gt


def test_perfect_prediction_scores_one():
    gt = _mask_with_all_classes()
    metrics = seg_metrics.evaluate(gt, gt)
    for cls in seg_metrics.CLASSES:
        assert metrics["iou"][cls] == 1.0
        assert metrics["dice"][cls] == 1.0
        assert metrics["precision"][cls] == 1.0
        assert metrics["recall"][cls] == 1.0
    assert metrics["mean_iou"] == 1.0
    assert metrics["accuracy"] == 1.0
    tol = seg_metrics.ink_tolerance_metrics(gt, gt)
    assert tol["precision"] == 1.0 and tol["recall"] == 1.0 and tol["f1"] == 1.0


def test_empty_prediction_scores_zero_for_present_classes():
    gt = _mask_with_all_classes()
    pred = np.zeros_like(gt)  # tudo background
    metrics = seg_metrics.evaluate(pred, gt)
    assert metrics["iou"]["paper"] == 0.0
    assert metrics["iou"]["hand"] == 0.0
    assert metrics["iou"]["ink"] == 0.0
    assert metrics["dice"]["ink"] == 0.0
    assert metrics["recall"]["ink"] == 0.0
    # Background não é zero: parte do ground truth já é background.
    assert 0.0 < metrics["iou"]["background"] < 1.0
    tol = seg_metrics.ink_tolerance_metrics(pred, gt)
    assert tol["precision"] == 0.0 and tol["recall"] == 0.0


def test_ink_shifted_two_pixels_passes_tolerance_but_not_exact_iou():
    gt = np.zeros((20, 20), dtype=np.int64)
    gt[10, 3:17] = 3  # linha fina de ink
    pred = np.zeros((20, 20), dtype=np.int64)
    pred[12, 3:17] = 3  # mesma linha, deslocada 2 px para baixo
    exact = seg_metrics.evaluate(pred, gt)
    assert exact["iou"]["ink"] == 0.0  # sem sobreposição exata
    tol = seg_metrics.ink_tolerance_metrics(pred, gt, tolerance_px=3)
    assert tol["precision"] == 1.0
    assert tol["recall"] == 1.0
    # Com tolerância 1, o deslocamento de 2 px NÃO deve passar.
    tol1 = seg_metrics.ink_tolerance_metrics(pred, gt, tolerance_px=1)
    assert tol1["precision"] == 0.0 and tol1["recall"] == 0.0


def test_ink_tolerance_empty_cases():
    empty = np.zeros((5, 5), dtype=np.int64)
    tol = seg_metrics.ink_tolerance_metrics(empty, empty)
    assert tol["precision"] == 1.0 and tol["recall"] == 1.0
    gt = empty.copy()
    gt[2, 2] = 3
    tol = seg_metrics.ink_tolerance_metrics(empty, gt)
    assert tol["precision"] == 0.0 and tol["recall"] == 0.0


def test_invalid_masks_raise():
    gt = _mask_with_all_classes()
    with pytest.raises(ValueError):
        seg_metrics.evaluate(np.zeros((5, 5), dtype=np.int64), gt)
    bad = gt.copy()
    bad[0, 0] = 4  # classe inexistente
    with pytest.raises(ValueError):
        seg_metrics.evaluate(bad, gt)


def test_ignore_index_255_is_excluded_from_metrics_and_ink_tolerance():
    gt = _mask_with_all_classes()
    gt[5:7, 5:7] = seg_metrics.IGNORE_INDEX
    perfect = gt.copy()
    wrong_in_ignored = gt.copy()
    perfect[5:7, 5:7] = seg_metrics.BACKGROUND
    wrong_in_ignored[5:7, 5:7] = seg_metrics.INK

    baseline = seg_metrics.evaluate(perfect, gt)
    changed = seg_metrics.evaluate(wrong_in_ignored, gt)
    assert changed == baseline
    assert baseline["ignored_px"] == 4
    assert baseline["supervised_px"] == gt.size - 4
    assert baseline["support_px"]["ink"] == 25

    tol = seg_metrics.ink_tolerance_metrics(wrong_in_ignored, gt, tolerance_px=1)
    assert tol["pred_ink"] == 25
    assert tol["matched_pred"] == 25
    assert tol["matched_gt"] == 25


def test_all_ignored_ground_truth_is_rejected():
    ignored = np.full((4, 4), seg_metrics.IGNORE_INDEX, dtype=np.uint8)
    with pytest.raises(ValueError, match="sem pixels supervisionados"):
        seg_metrics.evaluate(np.zeros_like(ignored), ignored)


def test_streaming_confusion_metrics_match_full_evaluation():
    gt = _mask_with_all_classes()
    gt[:2, :2] = seg_metrics.IGNORE_INDEX
    pred = gt.copy()
    pred[:2, :2] = seg_metrics.BACKGROUND
    pred[7, 7] = 0
    assert seg_metrics.evaluate_confusion_matrix(
        seg_metrics.confusion_matrix(pred, gt), 4
    ) == (seg_metrics.evaluate(pred, gt))


def test_scenario_battery_has_exactly_fifteen():
    assert len(seg_metrics.SCENARIOS) == 15
    ids = [s["id"] for s in seg_metrics.SCENARIOS]
    assert len(set(ids)) == len(ids)  # ids únicos
    names = [s["name"] for s in seg_metrics.SCENARIOS]
    for required in (
        "folha limpa",
        "folha com desenho",
        "folha muito desenhada",
        "mão cobrindo canto",
        "mão cobrindo centro",
        "mão cobrindo traço",
        "mão parcialmente fora da imagem",
        "fundo claro",
        "fundo escuro",
        "sombra forte",
        "folha parcialmente fora da imagem",
        "mão sem folha",
        "objeto branco sem folha",
        "traço fino de lápis",
        "traço grosso de caneta",
    ):
        assert required in names


def test_save_predictions_writes_png_and_manifest(tmp_path):
    pred = _mask_with_all_classes()
    predictions = {"01_clean_sheet": pred}
    manifest_path = seg_metrics.save_predictions(
        tmp_path, predictions, model_name="dummy-v1"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_name"] == "dummy-v1"
    assert "01_clean_sheet" in manifest["scenarios"]
    png_path = tmp_path / "01_clean_sheet_pred.png"
    assert png_path.exists() and png_path.stat().st_size > 0


def test_evaluate_scenarios_aggregates(tmp_path):
    gt = _mask_with_all_classes()
    result = seg_metrics.evaluate_scenarios(
        {"01_clean_sheet": gt}, {"01_clean_sheet": gt}
    )
    assert result["per_scenario"]["01_clean_sheet"]["metrics"]["mean_iou"] == 1.0
    assert result["aggregate"]["n_evaluated"] == 1
    assert result["aggregate"]["mean_iou"] == 1.0


def _stroke(shape=(40, 60), rows=slice(18, 21), cols=slice(5, 55)):
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    return mask


def test_skeletonize_thins_bar_to_one_pixel_line():
    skel = seg_metrics.skeletonize(_stroke())
    assert skel.sum() > 30
    assert skel.sum(axis=0).max() == 1


def test_skeletonize_matches_opencv_thinning_away_from_border():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "ximgproc"):
        pytest.skip("opencv-contrib ausente")
    rng = np.random.default_rng(1)
    for _ in range(5):
        m = np.zeros((100, 110), np.uint8)
        for _ in range(5):
            p0 = tuple(int(v) + 10 for v in rng.integers(0, 80, 2))
            p1 = tuple(int(v) + 10 for v in rng.integers(0, 80, 2))
            cv2.line(m, p0, p1, 1, int(rng.integers(1, 6)))
        ref = cv2.ximgproc.thinning(m * 255, thinningType=0) > 0
        assert (seg_metrics.skeletonize(m) == ref).all()


def test_cldice_perfect_empty_and_one_sided():
    s = _stroke()
    assert seg_metrics.cldice_binary(s, s)["cldice"] == 1.0
    empty = np.zeros_like(s)
    assert seg_metrics.cldice_binary(empty, empty)["cldice"] == 1.0
    assert seg_metrics.cldice_binary(empty, s)["cldice"] == 0.0
    assert seg_metrics.cldice_binary(s, empty)["cldice"] == 0.0
    with pytest.raises(ValueError):
        seg_metrics.cldice_binary(s, s[:10])


def test_cldice_punishes_broken_stroke_more_than_iou():
    gt = _stroke()
    broken = gt.copy()
    broken[:, 20:40] = False  # traço partido: 40% do comprimento some
    m = seg_metrics.cldice_binary(broken, gt)
    assert m["tprec"] == 1.0 and m["tsens"] < 0.7
    thick = _stroke(rows=slice(16, 23))  # borrão mais grosso, mesmo traço
    t = seg_metrics.cldice_binary(thick, gt)
    assert t["tsens"] == 1.0 and t["tprec"] == 1.0


def test_cldice_tolerance_forgives_small_shift():
    gt = _stroke()
    shifted = _stroke(rows=slice(21, 24))
    assert seg_metrics.cldice_binary(shifted, gt)["cldice"] < 0.5
    assert seg_metrics.cldice_binary(shifted, gt, tolerance_px=2)["cldice"] == 1.0


def test_ink_cldice_ignores_255():
    gt = np.full((40, 60), seg_metrics.PAPER, np.uint8)
    gt[_stroke()] = seg_metrics.INK
    pred = gt.copy()
    pred[:, 40:] = seg_metrics.PAPER
    gt_ignored = gt.copy()
    gt_ignored[:, 40:] = seg_metrics.IGNORE_INDEX
    assert seg_metrics.ink_cldice(pred, gt)["cldice"] < 1.0
    assert seg_metrics.ink_cldice(pred, gt_ignored)["cldice"] == pytest.approx(
        1.0, abs=0.05
    )


def test_evaluate_in_roi_only_counts_inside():
    gt = np.full((20, 20), seg_metrics.PAPER, np.uint8)
    gt[5:8, 2:18] = seg_metrics.INK
    pred = gt.copy()
    pred[15:18, 2:18] = seg_metrics.INK  # fantasma longe do traço real
    roi = seg_metrics.interior_roi(gt.shape, 0.0)
    roi[12:, :] = False
    inside = seg_metrics.evaluate_in_roi(pred, gt, roi)
    assert inside["iou"]["ink"] == 1.0
    assert seg_metrics.evaluate(pred, gt)["iou"]["ink"] < 1.0
    stroke = seg_metrics.stroke_roi(gt == seg_metrics.INK, pred == seg_metrics.INK, 1)
    assert stroke[16, 10] and not stroke[11, 10]
    assert seg_metrics.evaluate_in_roi(pred, gt, stroke)["iou"]["ink"] < 1
    with pytest.raises(ValueError):
        seg_metrics.evaluate_in_roi(pred, gt, roi[:5])


def test_interior_roi_cuts_margin():
    roi = seg_metrics.interior_roi((100, 200), 0.1)
    assert roi.sum() == 80 * 160 and not roi[0, 0] and roi[50, 100]


def test_cldice_empty_prediction_against_point_scores_zero():
    """Esqueleto vazio de mascara NAO vazia (um ponto 2x2 some na
    esqueletizacao) nao pode valer "vazio dos dois lados": predicao
    vazia contra GT real vale 0.0."""
    pred = np.zeros((2, 2), dtype=bool)
    gt = np.ones((2, 2), dtype=bool)
    r = seg_metrics.cldice_binary(pred, gt)
    assert r["cldice"] == 0.0


def test_dilate_disk_and_tolerance_on_tiny_mask():
    """Mascara menor que a tolerancia (2x2, raio 3): deslocamentos
    maiores que a dimensao produziam fatias incompativeis e
    quebravam a dilatacao."""
    m = np.ones((2, 2), dtype=bool)
    assert seg_metrics._dilate_disk(m, 3).all()
    full_ink = np.full((2, 2), seg_metrics.INK)
    r = seg_metrics.ink_tolerance_metrics(full_ink, full_ink, 3)
    assert r["f1"] == 1.0


def test_cldice_dot_beside_stroke_counts_in_gt():
    # Um ponto 2x2 ao lado de um traco: a esqueletizacao apaga o ponto,
    # mas o esqueleto do traco sobrevive — o fallback por mascara
    # inteira nao disparava e o ponto sumia da conta (score perfeito
    # para uma predicao sem o ponto).
    gt = np.zeros((60, 60), dtype=bool)
    gt[30:35, 5:55] = True
    gt[10:12, 20:22] = True
    pred = np.zeros((60, 60), dtype=bool)
    pred[30:35, 5:55] = True
    r = seg_metrics.cldice_binary(pred, gt)
    assert r["cldice"] < 1.0
    assert r["tsens"] < 1.0  # o esqueleto do ponto do GT nao foi coberto


def test_cldice_extra_predicted_dot_is_punished():
    gt = np.zeros((60, 60), dtype=bool)
    gt[30:35, 5:55] = True
    pred = gt.copy()
    pred[10:12, 20:22] = True  # ponto previsto a mais
    r = seg_metrics.cldice_binary(pred, gt)
    assert r["cldice"] < 1.0
    assert r["tprec"] < 1.0


def test_cldice_identical_dot_still_perfect():
    m = np.zeros((60, 60), dtype=bool)
    m[10:12, 20:22] = True
    assert seg_metrics.cldice_binary(m, m)["cldice"] == 1.0


def test_validate_masks_rejects_fractional_labels():
    pred = np.zeros((10, 10), dtype=np.float64)
    pred[2:4, 2:4] = 1.9
    gt = np.zeros((10, 10), dtype=np.int64)
    with pytest.raises(ValueError, match="inteiros"):
        seg_metrics.iou_per_class(pred, gt)
    pred_int = np.zeros((10, 10), dtype=np.int64)
    gt_frac = np.zeros((10, 10), dtype=np.float64)
    gt_frac[2:4, 2:4] = 1.9
    with pytest.raises(ValueError, match="inteiros"):
        seg_metrics.iou_per_class(pred_int, gt_frac)
    # floats com valores inteiros exatos continuam aceitos
    pred_ok = np.zeros((10, 10), dtype=np.float64)
    pred_ok[2:4, 2:4] = 2.0
    seg_metrics.iou_per_class(pred_ok, gt)


def test_cldice_overlapping_dots_score_half():
    # Dois pontos 2x2 deslocados 1 px: o fallback por componente inteiro
    # preserva a sobreposicao real de 2 px (com o fallback de 1 px,
    # os pixels "mais centrais" divergiam e a interseccao ia a zero).
    gt = np.zeros((60, 60), dtype=bool)
    gt[2:4, 2:4] = True
    pred = np.zeros((60, 60), dtype=bool)
    pred[2:4, 3:5] = True
    r = seg_metrics.cldice_binary(pred, gt)
    # S(pred) tem 4 px, 2 dentro de V(gt): tprec = 0.5; simetrico tsens.
    assert r["tprec"] == pytest.approx(0.5)
    assert r["tsens"] == pytest.approx(0.5)
    assert r["cldice"] == pytest.approx(0.5)


def test_cldice_dot_fully_covered_scores_one():
    gt = np.zeros((60, 60), dtype=bool)
    gt[2:4, 2:4] = True
    assert seg_metrics.cldice_binary(gt.copy(), gt)["cldice"] == 1.0
    pred = gt.copy()
    pred[30:35, 5:55] = True  # cobre o ponto e adiciona um traco a mais
    r = seg_metrics.cldice_binary(pred, gt)
    assert r["tsens"] == 1.0  # todo o ponto do GT coberto
    assert r["tprec"] < 1.0  # o traco extra pesa na precisao


def test_cldice_dot_partially_covered():
    gt = np.zeros((60, 60), dtype=bool)
    gt[2:4, 2:4] = True
    pred = np.zeros((60, 60), dtype=bool)
    pred[2:4, 2:3] = True  # barra 2x1 cobrindo metade do ponto
    r = seg_metrics.cldice_binary(pred, gt)
    assert r["tprec"] == 1.0  # a barra sobrevive e esta toda dentro
    assert r["tsens"] == pytest.approx(0.5)
    assert r["cldice"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)


@pytest.mark.slow
def test_skeleton_fallback_scales_and_matches_naive_reference():
    rng = np.random.default_rng(0)
    mask = rng.random((1024, 1024)) < 0.005  # ~5k graos isolados
    times = []
    for _ in range(2):
        t0 = time.perf_counter()
        seg_metrics._skeleton_or_component_pixels(mask)
        times.append(time.perf_counter() - t0)
    # Limite folgado: a varredura por componente custava ~0.7 s aqui
    # (2.4 s na medicao do review); a versao indexada roda em dezenas
    # de ms. Mede-se duas vezes e pega-se a menor para nao brigar com
    # jitter de CI.
    assert min(times) < 0.5

    # Propriedade: resultado == esqueleto uniao componentes sumidos,
    # conferido contra uma referencia ingenua por componente.
    small = rng.random((60, 80)) < 0.08
    skel = seg_metrics.skeletonize(small)
    labels, n = seg_metrics._label_components(small)
    ref = skel.copy()
    for i in range(1, n + 1):
        comp = labels == i
        if not (comp & skel).any():
            ref |= comp
    got = seg_metrics._skeleton_or_component_pixels(small)
    assert np.array_equal(got, ref)
