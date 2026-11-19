"""Régua do motor de tinta (lab/bench_ink.py)."""

import json
import time

import cv2
import numpy as np
import pytest
from lab import bench_ink as bench

from draw import config


@pytest.fixture(autouse=True)
def _restore_format():
    yield
    config.apply_format("a4")


@pytest.fixture(scope="module")
def page():
    return bench.make_synthetic_page(seed=0)


def _oracle(prob_page):
    """Filtro perfeito: só o traço sobrevive."""
    return lambda prob: np.where(prob_page.strokes, prob, 0.0).astype(np.float32)


def _naive_horizontal_opening(length=41):
    """O filtro ingênuo que o requisito (c) existe para pegar: tira toda
    estrutura horizontal comprida — pauta, mas também os braços de E e T."""

    def apply(prob):
        mask = (prob > 0.5).astype(np.uint8)
        kernel = np.ones((1, length), np.uint8)
        horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) > 0
        return np.where(horizontal, 0.0, prob).astype(np.float32)

    return apply


# ------------------------------------------------------------- gabarito


def test_synthetic_page_is_deterministic():
    a = bench.make_synthetic_page(seed=3)
    b = bench.make_synthetic_page(seed=3)
    c = bench.make_synthetic_page(seed=4)
    np.testing.assert_array_equal(a.prob, b.prob)
    assert not np.array_equal(a.prob, c.prob)  # o ruído depende da semente
    np.testing.assert_array_equal(a.strokes, c.strokes)  # o gabarito não


def test_synthetic_ground_truth_is_consistent(page):
    assert page.prob.shape == (850, 1200) and page.prob.dtype == np.float32
    assert page.ruling_only.sum() > 50_000
    assert page.strokes.sum() > 5_000
    assert page.crossing_strokes.sum() > 500  # há cruzamentos para medir
    assert page.ticks_zone.sum() > 500  # e tracinhos possíveis
    assert not (page.ruling_only & page.strokes).any()
    assert (page.crossing_strokes <= page.strokes).all()
    for name in (
        "E",
        "T",
        "T2",
        "S",
        "H",
        "diag",
        "thin",
        "E_on_line",
        "pencil_on_line",
    ):
        assert page.glyphs[name].sum() > 300, name
    # a pauta é de fato curva: a ordenada da mesma linha varia com x
    geo = dict(y0=0.0, period=34.0, slope=0.03, bend_amp=22.0, bend_x=600, bend_w=144)
    ys = bench.ruling_y(5, np.array([0, 600, 1200]), **geo)
    assert abs(ys[1] - (ys[0] + ys[2]) / 2) > 10


def test_model_like_prob_sees_ruling_as_ink(page):
    """O v2 enxerga a pauta como tinta (medido ao vivo): a identidade deixa
    quase toda a pauta na saída."""
    assert page.prob[page.ruling_only].mean() > 0.78
    assert page.prob[~page.ruling & ~page.strokes].mean() < 0.1


# -------------------------------------------------------------- métricas


def test_identity_filter_keeps_everything_including_the_ruling(page):
    r = bench.score_filter(page)
    assert r["ruling_removed"] < 0.01
    assert r["crossing_preserved"] > 0.99
    assert r["short_stroke_recall"] > 0.99
    assert r["on_line_recall"] > 0.99
    assert r["pencil_on_line_recall"] > 0.99
    assert r["ticks_residual"] > 500


def test_oracle_filter_scores_perfectly(page):
    r = bench.score_filter(page, _oracle(page))
    assert r["ruling_removed"] == pytest.approx(1.0, abs=0.005)
    assert r["crossing_preserved"] > 0.99
    assert r["ticks_residual"] == 0
    assert r["short_stroke_recall"] > 0.99
    assert r["on_line_recall"] > 0.99


def test_kill_everything_filter_scores_the_trade_off_floor(page):
    r = bench.score_filter(page, lambda prob: np.zeros_like(prob))
    assert r["ruling_removed"] == 1.0
    assert r["crossing_preserved"] == 0.0
    assert r["short_stroke_recall"] == 0.0
    assert r["ticks_residual"] == 0


def test_naive_morphology_is_caught_by_the_short_stroke_metric(page):
    """O filtro ingênuo remove a pauta mas mutila E/T/S/H: é o que a
    métrica (c) tem de denunciar."""
    r = bench.score_filter(page, _naive_horizontal_opening(41))
    # "funciona" na pauta reta, mas não pega os trechos curvos (lombada)...
    assert 0.5 < r["ruling_removed"] < 0.95
    # ...e ainda picota os braços das letras
    assert r["short_stroke_recall"] < 0.85
    assert (
        r["short_stroke_recall"]
        < bench.score_filter(page, _oracle(page))["short_stroke_recall"] - 0.15
    )


def test_glyph_recall_names_every_glyph(page):
    recalls = bench.glyph_recall(page.prob > 0.5, page)
    assert set(recalls) == {
        "E",
        "T",
        "T2",
        "S",
        "H",
        "diag",
        "thin",
        "E_on_line",
        "pencil_on_line",
    }
    assert set(bench.SHORT_STROKE_GLYPHS) <= set(recalls)


def test_short_stroke_metric_covers_all_four_letters():
    """O requisito (c) é sobre E, T, S e H: reduzir o conjunto esconderia
    uma regressão num deles."""
    assert bench.SHORT_STROKE_GLYPHS == ("E", "T", "T2", "S", "H")


def test_short_stroke_recall_drops_when_any_one_letter_is_lost(page):
    full = page.prob > 0.5
    for name in bench.SHORT_STROKE_GLYPHS:
        pred = full & ~page.glyphs[name]
        assert bench.short_stroke_recall(pred, page) < 0.9, name


def test_empty_denominators_are_nan_not_zero_division():
    assert np.isnan(bench._frac(0, 0))
    assert bench._frac(1, 4) == 0.25


# ------------------------------------------------------------- orçamento


def test_time_ms_reports_p50_and_p90_and_copies_the_input():
    data = np.zeros((4, 4), np.float32)

    def slow(arr):
        arr += 1  # não pode contaminar as próximas medidas
        time.sleep(0.002)

    out = bench.time_ms(slow, data, runs=8, warmup=1)
    assert out["p50_ms"] >= 1.5
    assert out["p90_ms"] >= out["p50_ms"]
    assert data.sum() == 0  # o original ficou intacto


# ----------------------------------------------------------- clipes reais


class _Detector:
    last_presence_logit = None

    def __call__(self, frame, return_mask=False):
        quad = np.float32([[2, 2], [60, 2], [60, 60], [2, 60]])
        return quad, "paper", np.ones(frame.shape[:2], np.uint8)


class _Ink:
    threshold = 0.5

    def ink_prob(self, warped):
        prob = np.zeros(warped.shape[:2], np.float32)
        prob[100:200, 100:200] = 0.97  # um bloco de "tinta" estável
        return prob

    def ink_mask(self, warped):
        return (self.ink_prob(warped) > self.threshold).astype(np.uint8)


def _fake_clip(root, with_regions=True):
    clip = root / "clipe-teste"
    clip.mkdir(parents=True)
    for i in range(10):
        cv2.imwrite(str(clip / f"f{i:04d}.png"), np.full((64, 64, 3), 200, np.uint8))
    (clip / "clip.json").write_text(
        json.dumps({"scene_fps": 4.0, "orientation": "retrato", "frames": 10})
    )
    if with_regions:
        (clip / "regions.json").write_text(
            json.dumps(
                {
                    "writing": [[90, 90, 210, 210]],
                    "ruling_only": [[400, 400, 600, 600]],
                }
            )
        )
    return clip


def test_run_clip_measures_canvas_regions_and_cpu(tmp_path):
    clip = _fake_clip(tmp_path)
    out = bench.run_clip(clip, [], detector=_Detector(), page_clean=_Ink())
    assert out["clip"] == "clipe-teste" and out["frames"] == 10
    assert out["canvas_px"] > 0
    assert out["writing_px"] == out["canvas_px"]  # tudo caiu na região de escrita
    assert out["ruling_px"] == 0
    assert out["ruling_over_writing"] == 0.0
    assert np.isfinite(out["cpu_s_per_scene_s"]) and out["cpu_s_per_scene_s"] > 0
    assert out["passes"] > 0
    assert out["_canvas"].shape == (config.PAPER_H, config.PAPER_W)
    assert not list(clip.glob(".bench_out"))  # nada escrito na pasta do clipe


def test_run_clip_without_regions_skips_region_metrics(tmp_path):
    clip = _fake_clip(tmp_path, with_regions=False)
    out = bench.run_clip(clip, [], detector=_Detector(), page_clean=_Ink())
    assert "writing_px" not in out and out["canvas_px"] > 0


def test_run_clip_honors_the_orientation_of_the_clip(tmp_path):
    clip = _fake_clip(tmp_path)
    meta = json.loads((clip / "clip.json").read_text())
    meta["orientation"] = "paisagem"
    (clip / "clip.json").write_text(json.dumps(meta))
    out = bench.run_clip(clip, [], detector=_Detector(), page_clean=_Ink())
    assert out["_canvas"].shape == (850, 1200)


# ----------------------------------------------------------------- presets


def test_presets_resolve_to_run_replay_flags():
    assert bench.preset_flags("legado") == []
    assert bench.preset_flags("ganho") == ["ink_auto_gain=1"]
    flags = bench.preset_flags("ganho+voto0.85")
    assert "ink_auto_gain=1" in flags
    assert {f.split("=")[0] for f in flags} >= {
        "ink.vote_hit",
        "ink.vote_score",
        "ink.vote_score_strong",
    }
    assert not any(f.startswith("@") for f in flags)  # o @strict foi resolvido
    strong = float(dict(f.split("=") for f in flags)["ink.vote_score_strong"])
    assert strong > 2.4  # 0,85 é mais estrito que o legado (1,9)


# --------------------------------------------------------------------- CLI


def test_cli_synthetic_prints_the_table(capsys):
    assert bench.main(["--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "(a) pauta" in out and "sem filtro" in out and "sobre linha" in out


def test_cli_writes_json(tmp_path):
    target = tmp_path / "r.json"
    assert bench.main(["--synthetic", "--json", str(target)]) == 0
    data = json.loads(target.read_text())
    assert "sem filtro (identidade)" in data["synthetic"]
    assert "recalls" not in data["synthetic"]["sem filtro (identidade)"]


def test_cli_requires_a_mode_and_rejects_unknown_presets(tmp_path):
    with pytest.raises(SystemExit) as error:
        bench.main([])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        bench.main(["--synthetic", "--presets", "inventado"])
    assert error.value.code == 2


def test_pencil_on_line_has_the_ruling_thickness_and_is_not_a_short_stroke(page):
    """O lápis fino sobre a linha tem a espessura da pauta (é o caso
    ambíguo) e fica de fora da meta (c), reportado à parte."""
    assert "pencil_on_line" not in bench.SHORT_STROKE_GLYPHS
    mask = page.glyphs["pencil_on_line"]
    col = mask[:, 730]  # no braço horizontal
    assert 1 <= int(col.sum()) <= page.h_pauta + 1
    assert (mask & page.crossing_strokes).any()  # a haste cruza linhas
