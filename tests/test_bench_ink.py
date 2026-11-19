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
        pred = full & ~page.arms[name]
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
    assert "pagina torta" in out
    assert "nan" not in out  # sem filtro não há tempo: imprime "-"


def test_cli_writes_json(tmp_path):
    target = tmp_path / "r.json"
    assert bench.main(["--synthetic", "--json", str(target)]) == 0
    data = json.loads(target.read_text())
    # a chave da linha de base reta não mudou (scripts que a leem seguem valendo)
    assert "sem filtro (identidade)" in data["synthetic"]
    assert "sem filtro (identidade), pagina torta (V 0,35)" in data["synthetic"]
    assert "recalls" not in data["synthetic"]["sem filtro (identidade)"]
    assert "arm_recalls" not in data["synthetic"]["sem filtro (identidade)"]


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
    assert (mask & page.ruling).any()  # a haste cruza linhas e o braço está sobre uma


def test_tilted_page_opens_the_two_halves_in_a_v():
    """tilt = caderno aberto torto: inclinações opostas de cada lado da lombada."""
    straight = bench.make_synthetic_page(seed=0)
    tilted = bench.make_synthetic_page(seed=0, tilt=0.35)
    geo = dict(y0=0.0, period=34.0, slope=0.03, bend_amp=22.0, bend_x=600, bend_w=144)
    left = bench.ruling_y(10, np.array([0.0, 300.0]), tilt=0.35, **geo)
    right = bench.ruling_y(10, np.array([900.0, 1200.0]), tilt=0.35, **geo)
    assert (left[1] - left[0]) / 300 < -0.25  # sobe para a lombada à esquerda
    assert (right[1] - right[0]) / 300 > 0.30  # desce à direita
    assert tilted.ruling_only.sum() > 40_000
    assert tilted.crossing_strokes.sum() > 300
    # o gabarito de traço não depende da inclinação, só a pauta
    for name in ("E", "T", "S", "H"):
        np.testing.assert_array_equal(tilted.glyphs[name], straight.glyphs[name])
    assert not np.array_equal(tilted.ruling, straight.ruling)


def test_crossing_strokes_exclude_the_on_line_glyphs(page):
    """E_on_line e pencil_on_line têm métrica própria."""
    assert not (page.crossing_strokes & page.glyphs["pencil_on_line"]).any()
    assert not (page.crossing_strokes & page.glyphs["E_on_line"]).any()


# --------------------------------------------- braços (sem as hastes) — #54


def test_stems_do_not_inflate_the_short_stroke_recall(page):
    """Filtro que apaga os braços mas preserva as hastes: o recall de (c)
    tem de despencar (a haste intacta não conta)."""
    full = page.prob > 0.5
    arms = np.zeros_like(full)
    for name in bench.SHORT_STROKE_GLYPHS:
        arms |= page.arms[name]
    pred = full & ~arms
    assert bench.short_stroke_recall(pred, page) < 0.15
    # sem a máscara de braços, o recall do glifo inteiro seguiria alto
    whole = bench.glyph_recall(pred, page)
    assert whole["E"] > 0.3 and whole["H"] > 0.3


def test_on_line_recall_only_counts_the_arm_on_the_ruling(page):
    full = page.prob > 0.5
    pred = full & ~page.arms["E_on_line"]
    r = bench.score_filter(page, lambda prob: np.where(pred, prob, 0.0))
    assert r["on_line_recall"] < 0.05  # E de pé, braço da linha apagado
    assert r["recalls"]["E_on_line"] > 0.5  # o glifo inteiro mascararia isso
    pred = full & ~page.arms["pencil_on_line"]
    r = bench.score_filter(page, lambda prob: np.where(pred, prob, 0.0))
    assert r["pencil_on_line_recall"] < 0.05


@pytest.mark.parametrize("tilt", [0.0, 0.35])
def test_on_line_arms_sit_on_the_ruling_even_on_the_tilted_page(tilt):
    """Na página em V o braço começava na altura do ponto médio e perdia a
    linha na ponta esquerda (review do PR #54): agora segue a polilinha."""
    pg = bench.make_synthetic_page(seed=0, tilt=tilt)
    for name in ("E_on_line", "pencil_on_line"):
        arm = pg.arms[name]
        cols = np.nonzero(arm.any(axis=0))[0]
        assert cols.size > 40
        # nas colunas do braço, a pauta mais próxima do braço está dentro dele
        covered = 0
        for x in cols:
            ys = np.nonzero(arm[:, x])[0]
            near = pg.ruling[ys.min() - 1 : ys.max() + 2, x]
            covered += bool(near.any() and (arm[:, x] & pg.ruling[:, x]).any())
        assert covered / cols.size > 0.98, (name, tilt)


def test_arms_cover_the_letters_and_exclude_the_stems(page):
    for name in bench.SHORT_STROKE_GLYPHS:
        arm, glyph = page.arms[name], page.glyphs[name]
        assert (arm <= glyph).all() and arm.sum() > 100, name
    # E, T e H têm haste: o braço é estritamente menor que o glifo
    for name in ("E", "T", "T2", "H"):
        assert page.arms[name].sum() < 0.8 * page.glyphs[name].sum(), name
    np.testing.assert_array_equal(page.arms["S"], page.glyphs["S"])


# ------------------------------------------------- JSON, tempo, launch — #54


def test_json_report_is_strict_even_with_empty_regions(tmp_path):
    out = bench._json_safe(
        {"a": float("nan"), "b": [float("inf"), 1.5], "c": np.float32("nan")}
    )
    assert out == {"a": None, "b": [None, 1.5], "c": None}
    json.dumps(out, allow_nan=False)  # não levanta


def test_cli_filter_reports_timings_in_table_and_json(tmp_path, capsys):
    target = tmp_path / "r.json"
    argv = ["--synthetic", "--filter", "lab.bench_ink:identity_filter"]
    assert bench.main([*argv, "--json", str(target)]) == 0
    out = capsys.readouterr().out
    assert "lab.bench_ink:identity_filter" in out
    data = json.loads(target.read_text())  # JSON estrito
    row = data["synthetic"]["lab.bench_ink:identity_filter"]
    assert (
        row["timing"]["p50_ms"] >= 0
        and row["timing"]["p90_ms"] >= row["timing"]["p50_ms"]
    )
    assert "timing" not in data["synthetic"]["sem filtro (identidade)"]


def test_cli_rejects_a_bad_filter_spec():
    for bad in ("sem_dois_pontos", "modulo.que.nao.existe:f", "lab.bench_ink:nada"):
        with pytest.raises(SystemExit) as error:
            bench.main(["--synthetic", "--filter", bad])
        assert error.value.code == 2, bad


def test_script_launch_from_another_cwd_can_import_the_lab(tmp_path):
    """`python lab/bench_ink.py` põe só lab/ no path; run_clip precisa de
    `lab.replay_lab` (review do PR #54: o pytest esconde isso)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = Path(bench.__file__)
    code = (
        "import runpy, sys\n"
        f"sys.path[:] = [{str(script.parent)!r}] + [p for p in sys.path if p]\n"
        f"runpy.run_path({str(script)!r}, run_name='bench_as_script')\n"
        "import lab.replay_lab\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


# ----------------------------------------------- run_clip: medidas — #54


def test_run_clip_final_provisional_comes_from_the_final_state(tmp_path, monkeypatch):
    """A amostra por passo antecede a publicação do último quadro; o
    `provisional_last_px` tem de refletir TODAS as publicações (pump)."""
    clip = _fake_clip(tmp_path)
    from draw.ink import worker as worker_mod

    pumps = {"n": 0}
    real_pump = worker_mod.PageCleanWorker.pump
    real_stats = worker_mod.PageCleanWorker.stats

    def pump(self):
        out = real_pump(self)
        pumps["n"] += 1  # cada pump publica; o contador imita o provisório
        return out

    def stats(self):
        out = dict(real_stats(self))
        out["provisional_px"] = pumps["n"]
        return out

    monkeypatch.setattr(worker_mod.PageCleanWorker, "pump", pump)
    monkeypatch.setattr(worker_mod.PageCleanWorker, "stats", stats)
    out = bench.run_clip(clip, [], detector=_Detector(), page_clean=_Ink())
    assert pumps["n"] > 0
    assert out["provisional_last_px"] == pumps["n"]  # não pumps["n"] - 1


def test_score_filter_rejects_a_filter_that_changes_the_shape(page):
    stacked = lambda prob: np.stack([prob, prob])  # noqa: E731
    with pytest.raises(ValueError, match=r"\(2, 850, 1200\).*\(850, 1200\)"):
        bench.score_filter(page, stacked)
    with pytest.raises(ValueError):
        bench.score_filter(page, lambda prob: prob[:100])


def test_run_clip_sums_retractions_across_accumulator_resets(tmp_path, monkeypatch):
    """Ao trocar de folha o acumulador zera o contador; o total do clipe
    soma o que foi retratado antes do zero."""
    clip = _fake_clip(tmp_path)
    from draw.ink import worker as worker_mod

    real = worker_mod.PageCleanWorker.stats
    seq = iter([v for v in (0, 100, 250, 5, 40, 60) for _ in range(6)])
    last = {"v": 0}

    def stats(self):
        out = dict(real(self))
        last["v"] = next(seq, last["v"])
        out["retracted_px_total"] = last["v"]
        return out

    monkeypatch.setattr(worker_mod.PageCleanWorker, "stats", stats)
    out = bench.run_clip(clip, [], detector=_Detector(), page_clean=_Ink())
    assert out["retracted_px_total"] == 250 + 60  # 250 antes do zero + 60 depois


def test_main_builds_a_fresh_detector_for_every_clip_and_preset(tmp_path, monkeypatch):
    """O detector guarda bordas/máscara entre detecções: reuso faria o
    resultado depender da ordem de execução."""
    import draw.perception as perception

    built = []

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            built.append(self)

        def load(self):
            pass

    seen = []
    monkeypatch.setattr(perception, "NeuralSheetDetector", FakeDetector)
    monkeypatch.setattr(
        bench,
        "run_clip",
        lambda clip, flags, **kw: (
            seen.append(kw["detector"])
            or {
                "clip": "x",
                "canvas_px": 0,
                "provisional_median_px": 0.0,
                "retracted_over_canvas": float("nan"),
                "cpu_s_per_scene_s": 0.0,
                "_canvas": None,
            }
        ),
    )
    import draw.page_clean as page_clean_mod

    class FakeInk:
        def __init__(self, *a, **k):
            pass

        def load(self):
            pass

    monkeypatch.setattr(page_clean_mod, "PageCleanInk", FakeInk)
    _fake_clip(tmp_path)
    assert bench.main(["--clips", str(tmp_path), "--presets", "legado,ganho"]) == 0
    assert len(seen) == 2 and seen[0] is not seen[1]
