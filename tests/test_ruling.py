"""Remocao de pauta por DP (draw.ink.ruling), medida na regua lab.bench_ink."""

import dataclasses
import sys

import numpy as np
import pytest
from lab import bench_ink as bench

from draw import app as app_module
from draw.contracts import WarpedPage
from draw.ink import ruling
from draw.ink.config import InkConfig
from draw.ink.worker import PageCleanWorker


def _score(page):
    return bench.score_filter(page, lambda prob: ruling.RulingFilter().apply(prob))


@pytest.fixture(scope="module")
def straight():
    return bench.make_synthetic_page(seed=0)


@pytest.fixture(scope="module")
def tilted():
    return bench.make_synthetic_page(seed=0, tilt=0.35)


def test_removes_the_ruling_and_keeps_crossings_and_short_strokes(straight):
    s = _score(straight)
    assert s["ruling_removed"] > 0.95
    assert s["crossing_preserved"] >= 0.98
    assert s["short_stroke_recall"] >= 0.97
    assert s["on_line_recall"] >= 0.97  # braco do E sobre a linha


def test_open_notebook_with_opposite_slopes_in_each_half(tilted):
    """Caderno aberto torto: as duas metades sao rastreadas por separado.
    Sem isso (uma unica inclinacao) a remocao cai para ~79%."""
    s = _score(tilted)
    assert s["ruling_removed"] > 0.90
    assert s["crossing_preserved"] >= 0.98
    assert s["short_stroke_recall"] >= 0.97
    model = ruling.estimate(tilted.prob.copy())
    assert len(model.segments) == 2
    assert {seg.x0 < 600 for seg in model.segments} == {True, False}


def test_page_without_ruling_is_a_noop():
    rng = np.random.default_rng(0)
    prob = (rng.random((850, 1200)) * 0.05).astype(np.float32)
    before = prob.copy()
    f = ruling.RulingFilter()
    out = f.apply(prob)
    np.testing.assert_array_equal(out, before)
    assert f.last_info["lines"] == 0
    assert ruling.estimate(before) is None


def test_isolated_pen_strokes_without_ruling_are_untouched():
    prob = np.zeros((850, 1200), np.float32)
    prob[300:306, 100:700] = 0.95  # um traco grosso horizontal
    prob[100:800, 400:405] = 0.95  # e uma haste
    before = prob.copy()
    np.testing.assert_array_equal(ruling.RulingFilter().apply(prob), before)


def test_filter_caches_the_model_between_passes(monkeypatch, straight):
    calls = []
    real = ruling.estimate
    monkeypatch.setattr(
        ruling, "estimate", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    f = ruling.RulingFilter(reestimate_every=3)
    for _ in range(7):
        f.apply(straight.prob.copy())
    assert len(calls) == 3  # passadas 1, 4 e 7


def test_reset_and_shape_change_force_a_new_estimate(monkeypatch, straight):
    calls = []
    real = ruling.estimate
    monkeypatch.setattr(
        ruling, "estimate", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    f = ruling.RulingFilter()
    f.apply(straight.prob.copy())
    f.apply(straight.prob.copy())
    assert len(calls) == 1
    f.reset()
    f.apply(straight.prob.copy())
    assert len(calls) == 2
    f.apply(straight.prob[:700, :1000].copy())
    assert len(calls) == 3


def test_a_single_thick_column_on_the_ruling_is_not_an_anchor(straight):
    """Um ponto isolado de espessura maior (flutuacao da propria pauta) nao
    protege colunas em volta: so cruzamento de 2+ colunas seguidas."""
    prob = straight.prob.copy()
    model = ruling.estimate(prob.copy())
    # coluna longe de qualquer glifo: uma linha so de pauta
    glyphs = np.zeros(prob.shape, bool)
    for mask in straight.glyphs.values():
        glyphs |= mask
    seg = model.segments[0]
    for x in range(seg.x0 + 20, seg.x1 - 60):
        y = int(round(seg.paths[5, x - seg.x0]))
        if not glyphs[y - 8 : y + 9, x - 4 : x + 41].any():
            break
    else:
        pytest.fail("sem trecho de pauta limpa")
    base = ruling.RulingFilter().apply(prob.copy())
    prob[y - 5 : y + 6, x] = 0.95  # coluna unica engrossada
    out = ruling.RulingFilter().apply(prob.copy())
    # fora da coluna e da folga de 2 colunas, o resultado e o do caso sem o ponto
    window = (slice(y - 8, y + 9), slice(x + 1, x + 40))
    np.testing.assert_array_equal(out[window], base[window])
    assert (out[window] > 0.5).sum() == 0


@pytest.mark.slow
def test_cost_per_pass_stays_inside_the_budget(straight):
    f = ruling.RulingFilter()
    f.apply(straight.prob.copy())  # estima
    t = bench.time_ms(lambda p: f.apply(p), straight.prob, runs=15)
    assert t["p50_ms"] < 8.0  # meta 3 ms; folga p/ CI lento, o numero real vai na regua


def test_estimate_has_no_python_loop_over_y():
    """A DP e vetorizada em y: o laco e sobre colunas reduzidas."""
    import inspect

    src = inspect.getsource(ruling._comb_track) + inspect.getsource(ruling._refine)
    assert "for y" not in src and "for row" not in src


def test_default_config_is_off_and_the_worker_builds_the_filter_on_demand():
    class _Ink:
        threshold = 0.5

    assert InkConfig().ruling_removal is False
    assert PageCleanWorker(_Ink())._ruling is None
    on = PageCleanWorker(_Ink(), config=InkConfig(ruling_removal=True))
    assert isinstance(on._ruling, ruling.RulingFilter)


def test_main_default_is_on_and_no_ink_ruling_turns_it_off(monkeypatch):
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(sys, "argv", ["draw"])
    app_module.main()
    ink = seen[-1]["pipeline_config"].ink
    assert ink.ruling_removal is True
    assert ink.vote_score_strong > 1.9  # coexiste com o voto estrito
    monkeypatch.setattr(sys, "argv", ["draw", "--no-ink-ruling"])
    app_module.main()
    assert seen[-1]["pipeline_config"].ink.ruling_removal is False


def _ruled_page(thick_cols=(), thick=4, stroke_cols=None):
    """Pagina so de pauta (2 px), com trechos mais grossos e/ou uma haste."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        for x in thick_cols:
            prob[y : y + thick, x] = 0.95
    if stroke_cols is not None:
        prob[60:560, stroke_cols] = 0.95
    return prob


def test_margin_above_h_pauta_still_counts_as_ruling():
    """A pauta real oscila (3-6 px no e5): um trecho ate h_pauta + 2 e
    pauta, nao cruzamento. Com margem 0 ele virava ancora e ficava."""
    cols = list(range(300, 360))
    prob = _ruled_page(thick_cols=cols, thick=4)
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out > 0.5).sum() == 0


def test_a_stroke_crossing_the_ruling_is_kept_with_two_columns_of_margin():
    stroke = slice(700, 706)
    prob = _ruled_page(stroke_cols=stroke)
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[60:560, stroke] > 0.5).all()  # o traco inteiro
    row = 80 + 2 * 40  # uma linha da pauta
    assert (out[row, 698:700] > 0.5).all()  # folga de 2 colunas a esquerda
    assert (out[row, 706:708] > 0.5).all()  # e a direita
    assert (out[row, 690:697] > 0.5).sum() == 0  # alem dela a pauta sai
    assert (out[row, 709:720] > 0.5).sum() == 0


class _RuledInk:
    threshold = 0.7

    def __init__(self, prob):
        self._prob = prob

    def ink_prob(self, _warped):
        return self._prob.copy()


def _worker(prob, enabled):
    now = [1000.0]
    worker = PageCleanWorker(
        _RuledInk(prob),
        clock=lambda: now[0],
        config=InkConfig(ruling_removal=enabled),
    )
    return worker, now


def _run_one_pass(worker, now, shape, generation):
    now[0] += 10.0  # vence qualquer cadencia
    worker.submit(WarpedPage(np.zeros((*shape, 3), np.uint8), generation))
    assert worker.pump() is True


def test_worker_filters_the_probability_before_the_vote_only_when_enabled():
    prob = _ruled_page()
    for enabled in (False, True):
        worker, now = _worker(prob, enabled)
        gen = worker.new_generation()
        for _ in range(3):
            _run_one_pass(worker, now, prob.shape, gen)
        shown = worker.poll_display(gen, -1)
        confirmed = np.count_nonzero(shown.confirmed) if shown else 0
        if enabled:
            assert not confirmed  # pauta pura: nada entra no canvas
        else:
            assert confirmed  # sem a flag, a pauta entra (comportamento atual)


def test_worker_drops_the_cached_model_when_the_generation_changes(monkeypatch):
    prob = _ruled_page()
    worker, now = _worker(prob, True)
    resets = []
    real = worker._ruling.reset
    monkeypatch.setattr(worker._ruling, "reset", lambda: resets.append(1) or real())
    gen = worker.new_generation()
    _run_one_pass(worker, now, prob.shape, gen)
    n = len(resets)
    _run_one_pass(worker, now, prob.shape, gen)
    assert len(resets) == n  # mesma folha: reaproveita
    gen2 = worker.new_generation()
    _run_one_pass(worker, now, prob.shape, gen2)
    assert len(resets) == n + 1  # outra folha: reestima


# ------------------------------------------------ achados do Devin (#56)


@pytest.mark.parametrize(("x0", "x1"), [(300, 440), (80, 219)])
def test_a_local_hatch_of_pen_strokes_is_not_mistaken_for_ruling(x0, x1):
    """8 tracos de caneta igualmente espacados numa faixa so, em papel liso:
    periodicos, mas sem suporte ao longo da folha. (300-440) atravessa a
    divisoria das metades; (80-219) fica inteiro na metade esquerda (review
    #56)."""
    prob = np.zeros((850, 850), np.float32)
    for i in range(8):
        prob[200 + 40 * i : 202 + 40 * i, x0:x1] = 0.95
    before = prob.copy()
    assert ruling.estimate(before) is None
    f = ruling.RulingFilter()
    np.testing.assert_array_equal(f.apply(prob), before)
    assert f.last_info["lines"] == 0


def test_ruling_that_reaches_the_right_edge_of_an_850_wide_page_is_erased():
    """850 // 6 = 141 colunas: o passo real nao e 6. As ultimas 4 colunas
    ficavam sem caminho e a pauta ali sobrevivia ao voto (review #56)."""
    prob = np.zeros((1200, 850), np.float32)
    for k in range(18):
        prob[100 + 50 * k : 102 + 50 * k, :] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out > 0.5).sum() == 0
    assert (out[:, 840:] > 0.5).sum() == 0  # inclusive as colunas 846-849


def test_soft_fringe_of_the_thickest_erasable_run_is_erased_too():
    """Run de h_pauta + 2 linhas: a franja suave de baixo ficava (o laco
    parava uma linha antes)."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 4, 300:360] = 0.95  # trecho de h_pauta + 2
        prob[y + 4, 300:360] = 0.3  # franja suave logo abaixo
        prob[y - 1, 300:360] = 0.3  # e acima
    out = ruling.RulingFilter().apply(prob.copy())
    assert out[:, 300:360].max() == 0.0


def _count_estimates(monkeypatch):
    calls = []
    real = ruling.estimate
    monkeypatch.setattr(
        ruling, "estimate", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    return calls


def test_a_missed_estimate_under_a_hand_is_retried_as_soon_as_the_hand_leaves(
    monkeypatch,
):
    """Sem pauta achada com a mao tapando a folha, o negativo nao pode ficar
    em cache por 24 passadas: a pauta seria confirmada como tinta (review
    #56)."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        prob[80 + 40 * k : 82 + 40 * k, :] = 0.95
    covered = np.zeros(prob.shape, bool)  # `visible` = sem mao: tudo tapado
    f = ruling.RulingFilter()
    f.apply(prob.copy(), covered)
    assert f.last_info["lines"] == 0
    out = f.apply(prob.copy(), None)  # a mao saiu: reestima JA nesta passada
    assert f.last_info["lines"] > 0 and (out > 0.5).sum() == 0


def test_blank_page_with_a_small_hand_does_not_estimate_every_pass(monkeypatch):
    """Em papel liso a mao esta quase sempre no quadro: so uma oclusao grande
    justifica reestimar cedo, e mesmo assim com recuo (review #56)."""
    calls = _count_estimates(monkeypatch)
    blank = np.zeros((600, 1200), np.float32)
    small_hand = np.ones(blank.shape, bool)
    small_hand[:, :120] = False  # 10% tapado
    f = ruling.RulingFilter()
    for _ in range(10):
        f.apply(blank.copy(), small_hand)
    assert len(calls) == 1
    big_hand = np.zeros(blank.shape, bool)  # tudo tapado: tenta de novo, com recuo
    calls.clear()
    f = ruling.RulingFilter()
    for _ in range(10):
        f.apply(blank.copy(), big_hand)
    assert 2 <= len(calls) <= 4  # a cada _RETRY_PASSES, nao a cada passada
    # e sem mao, um negativo continua em cache
    calls.clear()
    f = ruling.RulingFilter()
    for _ in range(10):
        f.apply(blank.copy(), None)
    assert len(calls) == 1


def test_paths_use_pixel_index_coordinates_and_do_not_grab_a_neighbour_stroke():
    """O centro de um bloco reduzido 2:1 que cobre as linhas y e y+1 e y+0,5
    (coordenadas de indice), nao y+1: o erro de meio pixel fazia o arredondamento
    escolher um traco de caneta vizinho (review #56)."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        prob[80 + 40 * k : 82 + 40 * k, :] = 0.95
    model = ruling.estimate(prob.copy())
    first = model.segments[0].paths[0]
    assert abs(float(np.median(first)) - 80.5) < 0.3
    # traco de caneta (2 px) colado a pauta, separado por UMA linha em branco
    pen = np.zeros_like(prob)
    pen[80 + 40 * 3 + 3 : 80 + 40 * 3 + 5, 400:520] = 0.95
    merged = np.maximum(prob, pen)
    out = ruling.RulingFilter().apply(merged.copy())
    assert (out[80 + 40 * 3 + 3 : 80 + 40 * 3 + 5, 400:520] > 0.5).all()  # ficou
    assert (out[80 + 40 * 3 : 80 + 40 * 3 + 2, 540:700] > 0.5).sum() == 0  # pauta saiu


@pytest.mark.parametrize("x0", [0, 7, 19, 30, 41, 53])
def test_span_cutoff_does_not_depend_on_window_alignment(x0):
    """O mesmo tracejado deslocado dentro da metade tem de dar o mesmo
    veredito (review #56): a extensao e medida em colunas, nao em janelas."""
    prob = np.zeros((850, 850), np.float32)
    for i in range(8):
        prob[200 + 40 * i : 202 + 40 * i, 80 + x0 : 80 + x0 + 146] = 0.95
    assert ruling.estimate(prob) is None


def _confirmed_after_votes(prob, config, passes=4):
    from draw.ink.worker import PageCleanWorker

    now = [1000.0]
    worker = PageCleanWorker(_RuledInk(prob), clock=lambda: now[0], config=config)
    gen = worker.new_generation()
    for _ in range(passes):
        _run_one_pass(worker, now, prob.shape, gen)
    shown = worker.poll_display(gen, -1)
    return np.asarray(shown.confirmed) > 0


@pytest.mark.parametrize("level", [None, 0.85])
def test_filter_holds_up_through_the_live_vote(straight, level):
    """A regua mede o mapa; aqui o mapa filtrado passa pelo voto REAL do
    worker (padrao e estrito) e o canvas confirmado e medido (review #56)."""
    cfg = InkConfig(ruling_removal=True)
    if level is not None:
        cfg = InkConfig.strict_vote(level, ruling_removal=True)
    confirmed = _confirmed_after_votes(straight.prob, cfg)
    # mesma regra de voto, sem o filtro: separa o que o voto rejeita do que a
    # remocao de pauta apaga
    plain = _confirmed_after_votes(
        straight.prob, dataclasses.replace(cfg, ruling_removal=False)
    )
    iso = straight.ruling_only
    assert confirmed[iso].sum() < 0.1 * plain[iso].sum()  # pauta: >= 90% fora do canvas
    arms = np.zeros_like(confirmed)
    for name in bench.SHORT_STROKE_GLYPHS:
        arms |= straight.arms[name]
    assert (confirmed & arms).sum() >= 0.97 * (plain & arms).sum()  # braços ficam
