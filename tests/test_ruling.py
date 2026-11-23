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


def test_worker_erases_the_fold_ruling_from_the_confirmed_canvas():
    """Integração pelo worker: vinco sombreado puro não pode entrar no canvas
    confirmado (32.640 px de pauta sem o filtro)."""
    prob = _fold_page(shade_p=0.9)
    for enabled in (True, False):
        worker, now = _worker(prob, enabled)
        gen = worker.new_generation()
        for _ in range(4):
            _run_one_pass(worker, now, prob.shape, gen)
        shown = worker.poll_display(gen, -1)
        confirmed = np.count_nonzero(shown.confirmed) if shown else 0
        if enabled:
            assert confirmed == 0
        else:
            assert confirmed > 30000


def test_worker_keeps_a_bend_stroke_across_the_seam_on_the_canvas():
    """Integração pelo worker: o traço dobrando na costura (par esparso) entra
    inteiro no canvas confirmado — a remoção não pode comê-lo."""
    prob = _bend_across_offset_seam()
    worker, now = _worker(prob, True)
    gen = worker.new_generation()
    for _ in range(4):
        _run_one_pass(worker, now, prob.shape, gen)
    shown = worker.poll_display(gen, -1)
    box = (slice(280, 294), slice(562, 638))
    expected = (prob > 0.5)[box]
    kept = (shown.confirmed[box] & expected).sum() if shown else 0
    assert kept == expected.sum()


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
    paths = model.segments[0].paths
    first = paths[int(np.argmin(np.abs(np.median(paths, axis=1) - 80.5)))]
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


# ------------------------------------------------ caderno aberto com vinco (#132)


def _shaded_notebook(tilt, house=None):
    """Duas paginas curvas, pauta com lacunas e vinco sombreado: na faixa
    central TODAS as linhas engrossam (sombra), com probabilidade menor."""
    page = bench.make_synthetic_page(seed=0, tilt=tilt)
    prob = page.prob.copy()
    h, w = prob.shape
    mid = w // 2
    rng = np.random.default_rng(1)
    band = (slice(None), slice(mid - 40, mid + 40))
    shade = (
        np.asarray(
            __import__("cv2").dilate(
                page.ruling[band].astype(np.uint8), np.ones((5, 1), np.uint8)
            )
        )
        > 0
    )
    sub = prob[band]
    sub[shade] = np.maximum(sub[shade], rng.uniform(0.6, 0.9, int(shade.sum())))
    for x in range(30, w, 53):  # lacunas na pauta, longe do vinco
        if abs(x - mid) > 60:
            prob[page.ruling_only[:, x : x + 7].any(axis=1), x : x + 7] *= 0.4
    if house is not None:
        import cv2

        x0, y0, size = house
        mask = np.zeros(prob.shape, np.uint8)
        pts = [
            (x0, y0 + size),
            (x0, y0),
            (x0 + size, y0),
            (x0 + size, y0 + size),
            (x0, y0 + size),
        ]
        roof = [(x0, y0), (x0 + size // 2, y0 - size // 2), (x0 + size, y0)]
        for line in (pts, roof):
            cv2.polylines(mask, [np.array(line, np.int32)], False, 255, 5)
        prob[mask > 0] = 0.97
    return page, prob


def _leftover_in_fold(page, prob, out):
    """Pixels que sobram no vinco (rows sem a borda externa), incluindo a franja
    de sombra que `ruling_only` nao cobre, fora dos glifos do desenho."""
    cv2 = __import__("cv2")
    mid = prob.shape[1] // 2
    zone = np.zeros(prob.shape, bool)
    zone[40:-40, mid - 50 : mid + 50] = True
    pen = cv2.dilate(page.strokes.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ink = (prob > 0.5) & zone & ~pen
    return int(((out > 0.5) & ink).sum()), int(ink.sum())


@pytest.mark.parametrize("tilt", [0.0, 0.35])
def test_shaded_gutter_does_not_leave_a_staircase_of_ruling(tilt):
    page, prob = _shaded_notebook(tilt)
    out = ruling.RulingFilter().apply(prob.copy())
    left, total = _leftover_in_fold(page, prob, out)
    assert left < 0.01 * total  # inclui a franja de sombra, nao so a pauta


def _house_parts(x0, y0=300, size=150):
    import cv2

    shapes = {
        "left": [(x0, y0 + size), (x0, y0)],
        "top": [(x0, y0), (x0 + size, y0)],
        "right": [(x0 + size, y0), (x0 + size, y0 + size)],
        "bottom": [(x0 + size, y0 + size), (x0, y0 + size)],
        "roof": [(x0, y0), (x0 + size // 2, y0 - size // 2), (x0 + size, y0)],
    }
    parts = {}
    for name, line in shapes.items():
        mask = np.zeros((850, 1200), np.uint8)
        cv2.polylines(mask, [np.array(line, np.int32)], False, 255, 5)
        parts[name] = mask > 0
    return parts


@pytest.mark.parametrize("x0", [150, 520])  # 520: o topo cruza o vinco
def test_a_house_drawn_over_the_ruling_keeps_each_stroke(x0):
    page, prob = _shaded_notebook(0.35, house=(x0, 300, 150))
    out = ruling.RulingFilter().apply(prob.copy())
    for name, part in _house_parts(x0).items():
        kept = ((out > 0.5) & part).sum() / part.sum()
        assert kept >= 0.97, name  # cada lado, nao a media do desenho


def test_a_short_pen_dash_entirely_inside_the_fold_is_kept():
    """Traco curto sobre uma linha sombreada, sem continuacao fora do vinco.
    Aqui a sombra tem ~7 px, entao o traco de marcador tem 9."""
    page, prob = _shaded_notebook(0.0)
    mid = prob.shape[1] // 2
    rows = np.flatnonzero(page.ruling[:, mid])
    y = int(rows[len(rows) // 2])
    dash = np.zeros(prob.shape, bool)
    dash[y - 4 : y + 5, mid - 10 : mid + 4] = True  # 9 px x 14 colunas
    prob[dash] = 0.97
    out = ruling.RulingFilter().apply(prob.copy())
    assert ((out > 0.5) & dash).sum() >= 0.97 * dash.sum()


def _fold_page(
    fa=560,
    fb=640,
    shade_p=0.95,
    left_sh=0,
    right_sh=0,
    split_x=None,
    offset_r=0,
    x_end=1200,
):
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        yl = 80 + 40 * k
        yr = yl + offset_r
        if split_x is None:
            prob[yl : yl + 2, :x_end] = 0.95
            prob[yl : yl + 6, fa:fb] = shade_p
            if left_sh and k % 2 == 0:
                prob[yl : yl + 6, fa - left_sh : fa] = 0.95
            if right_sh and k % 2 == 0:
                prob[yl : yl + 6, fb : fb + right_sh] = 0.95
        else:
            prob[yl : yl + 2, :split_x] = 0.95
            if fa < split_x:
                prob[yl : yl + 6, fa : min(fb, split_x)] = shade_p
            if x_end > split_x:
                prob[yr : yr + 2, split_x:x_end] = 0.95
                if fb > split_x:
                    prob[yr : yr + 6, max(fa, split_x) : fb] = shade_p
            if left_sh and k % 2 == 0:
                prob[yl : yl + 6, fa - left_sh : fa] = 0.95
            if right_sh and k % 2 == 0:
                y_rsh = yr if fb >= split_x else yl
                prob[y_rsh : y_rsh + 6, fb : fb + right_sh] = 0.95
    return prob


def _assert_only_stroke_kept(out, stroke_rows, fa, fb):
    rows = [stroke_rows] if isinstance(stroke_rows, slice) else stroke_rows
    for r in rows:
        assert (out[r, fa:fb] > 0.5).all()
    others = out[:, fa:fb] > 0.5
    for r in rows:
        others[r, :] = False
    assert others.sum() == 0


def test_a_7px_dash_on_a_6px_shaded_rule_is_kept_and_the_rest_is_erased():
    """Caso do review: pauta de 2 px engrossa para 6 no vinco; um traco de 7 px
    e 14 colunas numa so linha (prob 0,97) fica, a sombra das outras sai."""
    prob = _fold_page(shade_p=0.8)
    dash = np.zeros(prob.shape, bool)
    dash[80 + 40 * 5 - 1 : 80 + 40 * 5 + 6, 590:604] = True
    prob[dash] = 0.97
    out = ruling.RulingFilter().apply(prob.copy())
    assert ((out > 0.5) & dash).sum() >= 0.97 * dash.sum()
    assert ((out > 0.5) & ~dash)[:, 560:640].sum() == 0


def test_pen_mark_on_a_predicted_row_of_an_unruled_section_is_kept():
    """Pauta so de y=100 para baixo; uma marca de 40 colunas na linha prevista
    acima dela nao pode validar a propria linha extrapolada."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        prob[100 + 40 * k : 102 + 40 * k, :] = 0.95
    prob[60:62, 300:340] = 0.95  # fora da pauta, no passo dela
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[60:62, 300:340] > 0.5).all()
    assert (out[100:580] > 0.5).sum() == 0  # e a pauta de verdade sai


def test_ruling_only_in_the_middle_of_the_page_is_still_detected():
    """Areas em branco acima e abaixo nao podem derrubar o segmento."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(7):
        prob[160 + 40 * k : 162 + 40 * k, :] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out > 0.5).sum() == 0


def test_a_shadow_centered_on_the_page_split_is_erased_on_both_sides():
    """Faixa de 16 colunas repartida 8 + 8 entre as metades (divisa em 600)."""
    prob = _ruled_page(thick_cols=range(592, 608), thick=6)
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 584:616] > 0.5).sum() == 0


def test_uneven_shadow_shoulders_do_not_exempt_the_whole_fold():
    """Algumas linhas ficam grossas 2 colunas alem da faixa (sombra que afina
    irregular), sem caneta: o interior da faixa sai por inteiro."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:640] = 0.95  # a faixa
        if k % 2 == 0:
            prob[y : y + 6, 558:560] = 0.95  # ombros
            prob[y : y + 6, 640:642] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 566:634] > 0.5).sum() == 0


def test_staggered_shadow_shoulders_longer_than_a_stroke_do_not_exempt_the_fold():
    """Ombros de 13 colunas (acima do minimo de continuacao) em linhas
    alternadas: a sombra que afina de forma escalonada nao e um traco.
    Continuacao em muitas linhas nao isenta a faixa (review #139)."""
    prob = _fold_page(left_sh=13, right_sh=13)
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 566:634] > 0.5).sum() == 0
    # Nos ombros (< 60% das linhas), as linhas finas (impares) saem e as
    # marcas dos ombros (pares) ficam fora de _gutter_mask
    odd_rows = np.concatenate(
        [np.arange(80 + 40 * k, 82 + 40 * k) for k in range(1, 12, 2)]
    )
    assert (out[odd_rows, 547:560] > 0.5).sum() == 0
    assert (out[odd_rows, 640:653] > 0.5).sum() == 0
    even_rows = np.concatenate(
        [np.arange(80 + 40 * k, 86 + 40 * k) for k in range(0, 12, 2)]
    )
    assert (out[even_rows, 547:560] > 0.5).all()
    assert (out[even_rows, 640:653] > 0.5).all()


def test_shadow_shoulders_covering_most_rules_are_erased_with_the_fold():
    """Ombros em >= 60% das linhas (8 de 12) entram em _gutter_mask e saem."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:640] = 0.95
        if k % 3 != 0:  # 8 de 12 linhas (66,7% >= 60%)
            prob[y : y + 6, 547:560] = 0.95
            prob[y : y + 6, 640:653] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 550:650] > 0.5).sum() == 0


def test_a_stroke_crossing_the_fold_through_staggered_shoulders_is_kept():
    """Caso do review: ombros de 13 colunas em linhas alternadas E um traco
    de 6 px na linha 7, de 540 a 660. O padrao comum e sombra (sai), mas o
    traco continua ALEM do comprimento comum e o miolo dele fica."""
    prob = _fold_page(left_sh=13, right_sh=13)
    y = 80 + 40 * 7
    prob[y : y + 6, 540:660] = 0.95  # mesma espessura da sombra
    out = ruling.RulingFilter().apply(prob.copy())
    _assert_only_stroke_kept(out, slice(y, y + 6), 566, 634)


def test_shadow_shoulders_that_grow_down_the_page_do_not_exempt_the_fold():
    """Ombros de largura crescente (12 a 22 colunas) em linhas alternadas:
    o corte nao pode ser so uma margem sobre a mediana — exige um gap claro
    ate o ombro mais longo das outras linhas (review #139)."""
    prob = _fold_page()
    for i, k in enumerate(range(0, 12, 2)):
        y = 80 + 40 * k
        sh = 12 + 2 * i  # 12, 14, 16, 18, 20, 22
        prob[y : y + 6, 560 - sh : 560] = 0.95
        prob[y : y + 6, 640 : 640 + sh] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 566:634] > 0.5).sum() == 0


def test_shadow_shoulders_with_a_gap_at_the_fold_edge_do_not_exempt_it():
    """Ombros de 13 colunas em linhas alternadas, 4 delas finas so na coluna
    vizinha da faixa (o voto tolera 11/12). A medida da extensao honra a
    mesma tolerancia: nenhuma isenta (review #139)."""
    prob = _fold_page(left_sh=13, right_sh=13)
    for k in (0, 4, 6, 8):  # gap na coluna vizinha da faixa
        y = 80 + 40 * k
        prob[y : y + 6, 559] = 0.0
        prob[y : y + 6, 640] = 0.0
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 566:634] > 0.5).sum() == 0


def test_two_equal_strokes_crossing_the_fold_are_both_kept():
    """Caso do review: dois tracos de 6 px com a MESMA extensao (20 colunas
    de cada lado) entre quatro ombros de 13. Antes, cada traco era o "ombro
    mais longo" do outro e os dois perdiam o miolo; agrupando por gap, os
    dois atravessam ALEM do grupo dos ombros (review #139)."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:640] = 0.95
        if k % 3 == 0:  # k in {0, 3, 6, 9}: quatro linhas com ombros
            prob[y : y + 6, 547:560] = 0.95
            prob[y : y + 6, 640:653] = 0.95
    strokes = (80 + 40 * 1, 80 + 40 * 7)  # fora do grupo dos ombros
    for y in strokes:
        prob[y : y + 6, 540:660] = 0.95  # 20 colunas de cada lado
    out = ruling.RulingFilter().apply(prob.copy())
    for y in strokes:
        assert (out[y : y + 6, 566:634] > 0.5).all()  # o miolo de cada traco
    others = out[:, 566:634] > 0.5
    for y in strokes:
        others[y : y + 6, :] = False
    assert others.sum() == 0  # e so eles


def test_a_stroke_crossing_a_split_fold_gets_one_verdict_on_both_halves():
    """Ombros assimetricos: 13 colunas a esquerda e 20 a direita do vinco
    [560,700), que a divisa das metades parte em 600. Um traco com 18
    colunas de continuacao de cada lado era medido por um ombro diferente
    em cada trilha (passava na esquerda, nao na direita) e saia pela
    metade; julgado pelo MESMO ombro comum (o maior dos dois lados), o
    veredito e unico: nao passa em nenhuma (review #139)."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:700] = 0.95  # a faixa cruza a divisa em 600
        if k % 2 == 0:
            prob[y : y + 6, 547:560] = 0.95  # ombros de 13 a esquerda
            prob[y : y + 6, 700:720] = 0.95  # ombros de 20 a direita
    y = 80 + 40 * 7
    prob[y : y + 6, 542:718] = 0.95  # traco: 18 colunas de cada lado
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 562:698] > 0.5).sum() == 0  # o vinco inteiro sai


def test_a_clear_stroke_crossing_a_split_fold_is_kept_on_both_halves():
    """Mesma geometria (ombros 13/20, vinco na divisa), mas o traco vai 30
    colunas alem de cada ponta — claramente alem do ombro comum (20 + 3):
    o miolo fica INTEIRO nas duas metades (review #139)."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:700] = 0.95
        if k % 2 == 0:
            prob[y : y + 6, 547:560] = 0.95
            prob[y : y + 6, 700:720] = 0.95
    y = 80 + 40 * 7
    prob[y : y + 6, 530:730] = 0.95  # traco: 30 colunas de cada lado
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 560:600] > 0.5).all()  # miolo a esquerda da divisa
    assert (out[y : y + 6, 600:698] > 0.5).all()  # e a direita
    others = out[:, 562:698] > 0.5
    others[y : y + 6, :] = False
    assert others.sum() == 0  # e so ele


def test_a_long_continuation_on_a_single_line_is_still_a_stroke():
    """Ombros de 20 colunas dos dois lados, mas numa linha so: continuacao
    rara = traco de caneta. A faixa dessa linha fica; a das outras sai
    (review #139)."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:640] = 0.95
    y = 80 + 40 * 5
    prob[y : y + 6, 540:560] = 0.95
    prob[y : y + 6, 640:660] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 566:634] > 0.5).all()  # a linha do traco
    others = out[:, 566:634] > 0.5
    others[y : y + 6, :] = False
    assert others.sum() == 0  # e so ela


def test_an_8_column_dash_inside_the_fold_is_kept():
    """Traco mais curto que 12 colunas inteiro dentro do vinco: na sombra
    uniforme, o desvio de uma so linha ja o separa das vizinhas (review
    #139)."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:640] = 0.8
    dash = np.zeros(prob.shape, bool)
    dash[80 + 40 * 5 - 1 : 80 + 40 * 5 + 6, 580:588] = True  # 7 px x 8 colunas
    prob[dash] = 0.97
    out = ruling.RulingFilter().apply(prob.copy())
    assert ((out > 0.5) & dash).sum() >= 0.97 * dash.sum()
    assert ((out > 0.5) & ~dash)[:, 560:640].sum() == 0


def test_a_shadow_ripple_shared_by_many_lines_is_erased():
    """Ondulacao da sombra: seis linhas engrossam 1 px a mais por 8 colunas
    dentro do vinco. Desvio compartilhado = sombra, nao seis tracos curtos;
    so o desvio isolado vale como caneta (review #139)."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:640] = 0.8
        if k % 2 == 0:
            prob[y : y + 7, 580:588] = 0.8  # a ondulacao
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 566:634] > 0.5).sum() == 0


def test_a_dash_thicker_than_a_shared_ripple_is_kept():
    """Ondulacao de 7 px em quatro linhas e, nas mesmas 8 colunas, um traco
    de 8 px numa quinta linha: o nivel compartilhado sai, quem vai acima
    dele fica (review #139)."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:640] = 0.8
        if k % 3 == 0:  # quatro linhas: 0, 3, 6, 9
            prob[y : y + 7, 580:588] = 0.8
    dash = np.zeros(prob.shape, bool)
    y = 80 + 40 * 5  # fora da ondulacao
    dash[y - 1 : y + 7, 580:588] = True  # 8 px
    prob[dash] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert ((out > 0.5) & dash).sum() >= 0.97 * dash.sum()
    assert ((out > 0.5) & ~dash)[:, 560:640].sum() == 0


def test_a_fold_ending_a_vote_window_before_the_split_is_erased():
    """A faixa termina a exatas 12 colunas da divisa das metades: a janela
    de voto ainda existe, mas a cauda alem dela e vazia. A medida da
    extensao nao pode quebrar nesse caso (argmax num eixo vazio) e o
    interior da faixa sai (review #139)."""
    prob = _fold_page(fa=560, fb=588, left_sh=13, right_sh=12)
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 566:582] > 0.5).sum() == 0


def _two_line_segment(n, second):
    paths = np.stack([np.full(n, 100.0), second]).astype(np.float32)
    return ruling.Segment(0, n, paths, 40.0, np.array([False, True]))


def test_a_dash_on_a_predicted_line_clipped_to_the_last_columns_is_not_ruling():
    """Linha prevista que so aparece nas ultimas 30 colunas, com um traco de
    27 sobre ela: sozinho ele nao autoriza apagar (review #139)."""
    x = np.arange(600)
    seg = _two_line_segment(600, 599.0 - (x - 570.0))
    prob = np.zeros((600, 600), np.float32)
    prob[99:101, :] = 0.95
    for c in range(572, 600):
        r = int(round(seg.paths[1, c]))
        prob[r : r + 2, c] = 0.95
    assert ruling._drop_unsupported(prob, seg).paths.shape[0] == 1


def test_a_blank_partial_window_does_not_dilute_the_support_of_a_gapped_rule():
    """Metade de 422 px (8 janelas + 38 vazios): tinta em 3 das 8 janelas
    inteiras (37,5%) continua valendo (review #139)."""
    seg = _two_line_segment(422, np.full(422, 200.0))
    prob = np.zeros((600, 422), np.float32)
    prob[99:101, :] = 0.95
    for w in (0, 4, 7):
        prob[200:202, w * 48 : w * 48 + 48] = 0.95
    assert ruling._drop_unsupported(prob, seg).paths.shape[0] == 2


def test_the_partial_tail_window_counts_only_when_it_has_ink():
    """Linha que entra a meio da trilha (120 colunas visiveis: duas janelas
    inteiras e um resto de 24). Tinta so na janela inteira final: 1/2. Com
    tinta tambem no resto: 2/3. Resto vazio nao dilui (review #139)."""
    inside = np.zeros((1, 600), bool)
    inside[0, 480:] = True
    base = np.zeros((1, 600), bool)
    base[0, 528:576] = True  # janela inteira final
    frac_blank, _ = ruling._window_support(base, inside)
    tail = base.copy()
    tail[0, 576:] = True  # e o resto parcial
    frac_tail, _ = ruling._window_support(tail, inside)
    assert frac_blank[0] == pytest.approx(1 / 2)
    assert frac_tail[0] == pytest.approx(2 / 3)
    # e o resto de 24 colunas tem de existir: sem ele (janelas inteiras so) a
    # tinta do resto seria ignorada e frac_tail seria 1/2
    assert frac_tail[0] > frac_blank[0]


def test_the_inked_partial_tail_is_what_keeps_a_tilted_extrapolated_line():
    """O mesmo cenario, ponta a ponta pelo filtro de linhas: a prevista so
    entra na vista nas ultimas 168 colunas (3 janelas inteiras + resto de
    24). Tinta na primeira janela inteira: 1/3 < 35% e ela cai. Com tinta
    tambem no resto: 2/4 e ela fica — so a cauda parcial a salva (review
    #139)."""
    x = np.arange(600)
    seg = _two_line_segment(600, 599.0 + (432.0 - x) * 0.5)
    prob = np.zeros((600, 600), np.float32)
    prob[99:101, :] = 0.95
    for c in range(432, 480):  # primeira janela inteira visivel
        r = int(round(seg.paths[1, c]))
        prob[r : r + 2, c] = 0.95
    assert ruling._drop_unsupported(prob, seg).paths.shape[0] == 1
    with_tail = prob.copy()
    for c in range(576, 600):  # e o resto parcial
        r = int(round(seg.paths[1, c]))
        with_tail[r : r + 2, c] = 0.95
    assert ruling._drop_unsupported(with_tail, seg).paths.shape[0] == 2


def test_a_stroke_with_unequal_valid_extensions_across_a_split_fold_is_kept_on_both_halves():
    """Ombros 13/20 e vinco [560,700) na divisa 600; traco com 18 col a esquerda
    e 30 a direita vence os dois ombros locais e fica inteiro nos dois lados."""
    prob = _fold_page(fa=560, fb=700, left_sh=13, right_sh=20)
    y = 80 + 40 * 7
    prob[y : y + 6, 542:730] = 0.95  # 18 a esquerda, 30 a direita
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 560:600] > 0.5).all()
    _assert_only_stroke_kept(out, slice(y, y + 6), 562, 698)


def test_a_fold_ending_8_columns_before_the_split_uses_adjacent_track_for_vote():
    """Faixa [560, 592) termina 8 colunas antes da divisa em 600: a janela de
    voto (12 col) cruza a divisa e usa amostras da trilha direita."""
    prob = _fold_page(fa=560, fb=592, left_sh=13, right_sh=13)
    out_plain = ruling.RulingFilter().apply(prob.copy())
    assert (out_plain[:, 566:586] > 0.5).sum() == 0

    y = 80 + 40 * 7
    prob[y : y + 6, 540:612] = 0.95  # 20 col a esquerda, 20 a direita
    out_stroke = ruling.RulingFilter().apply(prob.copy())
    assert (out_stroke[y : y + 6, 562:590] > 0.5).all()
    _assert_only_stroke_kept(out_stroke, slice(y, y + 6), 566, 586)


def test_a_narrow_vertical_crossing_inside_the_fold_is_kept():
    """Cruzamento vertical estreito (8 px x 3 col) no meio da sombra (6 px)
    fica protegido mesmo abaixo do teto elevado da faixa."""
    prob = _fold_page(shade_p=0.8)
    dash = np.zeros(prob.shape, bool)
    y = 80 + 40 * 5
    dash[y - 1 : y + 7, 575:578] = True  # 8 px x 3 col = 24 px
    prob[dash] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y - 1 : y + 7, 575:578] > 0.5).all()
    assert ((out > 0.5) & ~dash)[:, 560:640].sum() == 0


def test_off_frame_extrapolated_rule_does_not_erase_border_dash():
    """Linha extrapolada inclinada (0,15 px/col) que entra na imagem no fim da
    metade direita nao pode apagar um traco desconectado na borda onde ela
    ainda esta fora da imagem, e deve apagar o trecho que entra."""
    prob = np.zeros((600, 1200), np.float32)
    xs = np.arange(1200)
    for k in range(18):
        ys = np.round(-160.0 + 40.0 * k + 0.15 * xs).astype(np.int32)
        for dy in (0, 1):
            r = ys + dy
            ok = (r >= 0) & (r < 600)
            prob[r[ok], xs[ok]] = 0.95
    # Entre 952 e 975, a linha k=0 esta em y <= -13,75 (fora da imagem)
    prob[5:7, 952:976] = 0.95  # traco 2x24 = 48 px na borda superior
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[5:7, 952:976] > 0.5).all()
    # O trecho da linha k=0 que entra na imagem (x >= 1075) e apagado
    enter_x = np.arange(1075, 1200)
    enter_y = np.round(-160.0 + 0.15 * enter_x).astype(np.int32)
    assert (out[enter_y, enter_x] > 0.5).sum() == 0
    assert (out[enter_y + 1, enter_x] > 0.5).sum() == 0


def test_staggered_shoulders_on_different_rows_do_not_drop_crossing_strokes():
    """Ombros de 13 colunas em linhas diferentes de cada lado (1..4 a esquerda,
    4..7 a direita) e tres tracos de 20 colunas (linhas 8..10): cada ponta
    estima seu ombro usando os votos daquele lado e os 3 tracos ficam."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:700] = 0.95
    for k in range(1, 5):
        y = 80 + 40 * k
        prob[y : y + 6, 547:560] = 0.95
    for k in range(4, 8):
        y = 80 + 40 * k
        prob[y : y + 6, 700:713] = 0.95
    for k in range(8, 11):
        y = 80 + 40 * k
        prob[y : y + 6, 540:720] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    _assert_only_stroke_kept(
        out, [slice(80 + 40 * k, 80 + 40 * k + 6) for k in range(8, 11)], 562, 698
    )


def test_a_stroke_at_the_split_with_only_one_ruled_half_is_kept():
    """Pauta apenas na metade esquerda [0, 600), vinco em [560, 600) e traco
    de 6 px correndo de 540 a 630: sem trilha na direita, o voto da esquerda
    basta para preservar o traco dentro do vinco."""
    prob = _fold_page(fa=560, fb=600, x_end=600)
    y = 80 + 40 * 5
    prob[y : y + 6, 540:630] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 562:600] > 0.5).all()
    _assert_only_stroke_kept(out, slice(y, y + 6), 562, 598)


def test_a_1px_two_column_shadow_bump_inside_the_fold_is_erased():
    """Flutuacao de +1 px por apenas 2 colunas na sombra de 6 px nao pode
    sobreviver como se fosse cruzamento de caneta."""
    prob = _fold_page(shade_p=0.8)
    y = 80 + 40 * 5
    prob[y : y + 7, 575:577] = 0.8  # +1 px por 2 colunas = 14 px de ruido
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 566:634] > 0.5).sum() == 0


def test_one_sided_shoulders_do_not_erase_a_crossing_stroke():
    """Tres ombros de 13 colunas so a esquerda (sem ombros a direita) e um
    traco de 15 col a esquerda e 30 a direita: o traco cruza os dois lados
    e nao pode ser vetado pelos ombros unilaterais."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:700] = 0.95
    for k in range(3):
        y = 80 + 40 * k
        prob[y : y + 6, 547:560] = 0.95  # ombros so na esquerda
    y = 80 + 40 * 7
    prob[y : y + 6, 545:730] = 0.95  # 15 a esquerda, 30 a direita
    out = ruling.RulingFilter().apply(prob.copy())
    _assert_only_stroke_kept(out, slice(y, y + 6), 562, 698)


def test_a_4_column_1px_dash_straddling_the_split_is_kept():
    """Traco de 7 px (+1 px sobre a sombra de 6 px) por 4 colunas (598..601)
    cruzando a divisa das metades em 600: somando os dois lados atinge 4 col
    e sobrevive inteiro."""
    prob = _fold_page(shade_p=0.8)
    dash = np.zeros(prob.shape, bool)
    y = 80 + 40 * 5
    dash[y - 1 : y + 6, 598:602] = True  # 7 px x 4 col (2 de cada lado)
    prob[dash] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y - 1 : y + 6, 598:602] > 0.5).all()
    assert ((out > 0.5) & ~dash)[:, 560:640].sum() == 0


def test_staggered_shoulders_on_both_sides_without_stroke_are_erased():
    """Ombros escalonados de 13 col dos dois lados (4 linhas na esquerda,
    4 na direita, apenas 1 linha compartilhada) e SEM traço: nenhuma linha
    pode escapar do apagamento na faixa."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, 560:700] = 0.95
    for k in range(1, 5):  # linhas 1..4 com ombro a esquerda
        y = 80 + 40 * k
        prob[y : y + 6, 547:560] = 0.95
    for k in range(4, 8):  # linhas 4..7 com ombro a direita (so linha 4 nos dois)
        y = 80 + 40 * k
        prob[y : y + 6, 700:713] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 562:698] > 0.5).sum() == 0


def test_disconnected_2px_bumps_across_a_stepped_split_are_erased():
    """Degrau vertical na divisa (10 px) com calombos de +1 px por 2 colunas
    em cada lado: desconexos verticalmente, nenhum pode somar 4 colunas."""
    prob = _fold_page(shade_p=0.8, split_x=600, offset_r=10)
    yl = 80 + 40 * 5
    yr = yl + 10
    prob[yl : yl + 7, 598:600] = 0.8  # +1 px x 2 col na esquerda
    prob[yr : yr + 7, 600:602] = 0.8  # +1 px x 2 col na direita (degrau 10 px)
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 566:634] > 0.5).sum() == 0


def test_empty_fold_without_any_pen_strokes_is_completely_erased():
    """Invariante: faixa inteira sem tinta de caneta (com sombra na divisa)
    nao pode deixar nenhuma linha isenta."""
    prob = _fold_page(shade_p=0.9)
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 560:640] > 0.5).sum() == 0


def test_offset_halves_preserve_a_stroke_crossing_the_fold():
    """Pautas defasadas de meio periodo entre as metades (80/120/... a esquerda
    e 100/140/... a direita): nenhum par casa na divisa, mas o traco de 6 px
    sobre a pauta esquerda atravessa a sombra e sobrevive inteiro."""
    prob = _fold_page(split_x=600, offset_r=20)
    y = 80 + 40 * 5  # 280: cai no espaco em branco da metade direita (260 vs 300)
    prob[y : y + 6, 540:660] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 540:660] > 0.5).all()
    _assert_only_stroke_kept(out, slice(y, y + 6), 562, 638)


def test_single_ruled_half_fold_at_edge_with_shoulders_and_no_stroke_is_erased():
    """Pagina unica pautada [0, 600), sombra na borda [560, 600) com ombros
    de 13 col em metade das linhas e sem tinta: nada sobrevive na faixa."""
    prob = _fold_page(fa=560, fb=600, left_sh=13, x_end=600)
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 562:600] > 0.5).sum() == 0


@pytest.mark.parametrize("layout", ["paired_split", "offset_split", "blank_other"])
@pytest.mark.parametrize(
    "content",
    [
        "empty",
        "straight_stroke",
        "bending_stroke",
        "reverse_bending_stroke",
        "branched_stroke_shoulder",
        "shoulders_only",
        "vertical_wall",
    ],
)
def test_fold_junction_matrix(layout, content):
    """Matriz {pareada, defasada, sem pauta do outro lado} x {sem tinta,
    traço reto, dobra esq->dir, dobra dir->esq, ramificação traço+ombro,
    ombro puro, parede vertical}."""
    prob = np.zeros((600, 1200), np.float32)
    with_shoulders = content in ("shoulders_only", "branched_stroke_shoulder") or (
        layout == "paired_split"
    )
    if layout == "paired_split":
        fa, fb = 560, 640
        for k in range(12):
            y = 80 + 40 * k
            prob[y : y + 2, :] = 0.95
            prob[y : y + 6, fa:fb] = 0.95
            if with_shoulders and k % 2 == 0:
                prob[y : y + 6, fa - 13 : fa] = 0.95
                prob[y : y + 6, fb : fb + 13] = 0.95
    elif layout == "offset_split":
        fa, fb = 560, 640
        for k in range(12):
            yl, yr = 80 + 40 * k, 100 + 40 * k
            prob[yl : yl + 2, :600] = 0.95
            prob[yl : yl + 6, fa:600] = 0.95
            prob[yr : yr + 2, 600:] = 0.95
            prob[yr : yr + 6, 600:fb] = 0.95
            if with_shoulders and k % 2 == 0:
                prob[yl : yl + 6, fa - 13 : fa] = 0.95
                prob[yr : yr + 6, fb : fb + 13] = 0.95
    else:  # blank_other
        fa, fb = 560, 600
        for k in range(12):
            y = 80 + 40 * k
            prob[y : y + 2, :600] = 0.95
            prob[y : y + 6, fa:fb] = 0.95
            if with_shoulders and k % 2 == 0:
                prob[y : y + 6, fa - 13 : fa] = 0.95

    y_stroke = 80 + 40 * 5
    stroke_mask = np.zeros(prob.shape, bool)
    if content == "straight_stroke":
        stroke_mask[y_stroke : y_stroke + 6, fa - 20 : fb + 60] = True
    elif content == "bending_stroke":
        stroke_mask[y_stroke : y_stroke + 6, fa - 20 : 600] = True
        for x in range(600, 620):
            dy = x - 600
            stroke_mask[y_stroke + dy : y_stroke + dy + 6, x] = True
        stroke_mask[y_stroke + 20 : y_stroke + 26, 620 : fb + 40] = True
    elif content == "reverse_bending_stroke":
        y_r = 100 + 40 * 5 if layout == "offset_split" else y_stroke
        stroke_mask[y_r : y_r + 6, 590 : fb + 40] = True
        for x in range(570, 590):
            dy = 590 - x
            stroke_mask[y_r + dy : y_r + dy + 6, x] = True
        stroke_mask[y_r + 20 : y_r + 26, fa - 30 : 570] = True
    elif content == "branched_stroke_shoulder":
        # traço na pauta esquerda (280) saindo no branco a direita, mais ramo
        # vertical na costura que toca a pauta direita com ombro de 13 col
        stroke_mask[y_stroke : y_stroke + 6, fa - 20 : fb + 60] = True
        if layout != "blank_other":
            prob[y_stroke : y_stroke + 26, 599:601] = 0.95
    elif content == "vertical_wall":
        # parede vertical no meio da faixa + traço reto apenas na linha 5
        stroke_mask[y_stroke : y_stroke + 6, fa - 20 : fb + 60] = True
        prob[80:400, 580:584] = 0.95

    prob[stroke_mask] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    if content in ("empty", "shoulders_only"):
        assert (out[:, fa + 2 : fb - 2] > 0.5).sum() == 0
    else:
        inside = np.zeros(prob.shape, bool)
        inside[:, fa + 2 : fb - 2] = True
        kept = stroke_mask & inside
        assert (out[kept] > 0.5).all()
        if content in ("straight_stroke", "branched_stroke_shoulder", "vertical_wall"):
            # linhas distantes do traço devem ser apagadas mesmo com parede vertical
            assert (out[: y_stroke - 25, fa + 2 : 578] > 0.5).sum() == 0
            assert (out[y_stroke + 30 :, fa + 2 : 578] > 0.5).sum() == 0
        else:
            y_lo = min(y_stroke, 100 + 40 * 5) - 10
            y_hi = max(y_stroke, 100 + 40 * 5) + 50
            assert (out[:y_lo, fa + 2 : fb - 2] > 0.5).sum() == 0
            assert (out[y_hi:, fa + 2 : fb - 2] > 0.5).sum() == 0


def test_single_ruled_half_keeps_short_left_stroke_continuing_into_blank_half():
    """Metade esquerda pautada com ombros de 13 col, metade direita em branco;
    traço de 6 px com 16 col a esquerda (<= 13+3) e 60 col no branco sobrevive."""
    prob = _fold_page(fa=560, fb=600, left_sh=13, x_end=600)
    y = 80 + 40 * 5
    prob[y : y + 6, 544:660] = 0.95  # 16 col a esquerda, 60 col no branco
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 560:660] > 0.5).all()
    _assert_only_stroke_kept(out, slice(y, y + 6), 562, 600)


def test_offset_halves_preserve_a_stroke_that_bends_onto_a_destination_rule():
    """Pautas defasadas (80/120/... e 100/140/...); traço na pauta esquerda
    cruza a costura em 600 e dobra 20 px para pousar sobre a pauta direita em
    620..640: o segmento de chegada sobrevive."""
    prob = _fold_page(split_x=600, offset_r=20)
    y = 80 + 40 * 5  # 280 -> dobra para 300 (pauta direita k=5)
    stroke = np.zeros(prob.shape, bool)
    stroke[y : y + 6, 540:600] = True
    for x in range(600, 620):
        dy = x - 600
        stroke[y + dy : y + dy + 6, x] = True
    stroke[y + 20 : y + 26, 620:665] = True
    prob[stroke] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[stroke] > 0.5).all()
    assert (out[: y - 10, 562:638] > 0.5).sum() == 0
    assert (out[y + 50 :, 562:638] > 0.5).sum() == 0


def test_disconnected_shadow_fragment_before_gap_is_erased_while_stroke_survives():
    """Faixa [560, 640) em pautas defasadas com fragmento de sombra [547, 570)
    antes de um gap [570, 575) e traço depois [575, 660): o fragmento dentro
    da faixa [560, 570) sai, o traço fica."""
    prob = _fold_page(split_x=600, offset_r=20)
    y = 80 + 40 * 5
    prob[y : y + 6, 547:570] = 0.95  # fragmento/ombro desconectado
    prob[y : y + 6, 570:575] = 0.0  # gap desconectando [560, 570)
    prob[y : y + 6, 575:660] = 0.95  # traço cruzando a costura
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 562:570] > 0.5).sum() == 0
    assert (out[y : y + 6, 575:660] > 0.5).all()


def test_dark_seam_linking_to_destination_shoulder_is_erased_while_blank_crossing_survives():
    """Costura escura ligando pauta fonte (sem ombros) a pauta destino com
    ombros de 13 col e apagada; traço real cruzando para espaço em branco
    sobrevive."""
    prob = _fold_page(right_sh=13, split_x=600, offset_r=20)
    # costura de 1 col ligando pauta esquerda k=2 (160) a direita k=2 (180, com ombro)
    y_seam = 80 + 40 * 2
    prob[y_seam : y_seam + 26, 599:601] = 0.95
    # traço real na pauta esquerda k=5 (280) cruzando para espaço em branco
    y_stroke = 80 + 40 * 5
    prob[y_stroke : y_stroke + 6, 540:660] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y_stroke : y_stroke + 6, 540:660] > 0.5).all()
    assert (out[y_seam : y_seam + 6, 562:598] > 0.5).sum() == 0


def test_right_only_shoulders_with_offset_halves_erase_left_fold_ruling():
    """M1: ombros de 13 col só na direita + pautas defasadas (80/120 vs 100/140)
    sem traço: a pauta esquerda dentro do vinco é apagada."""
    prob = _fold_page(right_sh=13, split_x=600, offset_r=20)
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 562:638] > 0.5).sum() == 0


def test_single_half_fold_ending_6_cols_before_track_end_preserves_stroke():
    """M2: vinco [560, 594) terminando 6 colunas antes do fim da trilha [0, 600)
    com traço de 6 px em [540, 620): o traço dentro de [562, 594) sobrevive."""
    prob = _fold_page(fa=560, fb=594, x_end=600)
    y = 80 + 40 * 5
    prob[y : y + 6, 540:620] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    _assert_only_stroke_kept(out, slice(y, y + 6), 562, 594)


def test_8px_stepped_halves_preserve_straight_stroke_crossing_fold():
    """M3: degrau de 8 px entre metades [0, 600) e [600, 1200) com traço reto
    de 6 px cruzando o vinco em y=280: o traço sobrevive dentro do vinco."""
    prob = _fold_page(split_x=600, offset_r=8)
    y = 280
    prob[y : y + 6, 540:660] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 562:638] > 0.5).all()


@pytest.mark.parametrize("step", [0, 8, 20])
@pytest.mark.parametrize("tail", [0, 6, 20])
@pytest.mark.parametrize(
    "content",
    ["empty", "straight_stroke", "bending_stroke", "one_sided_shoulder"],
)
def test_step_tail_matrix(step, tail, content):
    """Matriz {degrau 0, 8, 20} x {cauda 0, 6, 20 antes de 600} x {vazio,
    traço reto, traço dobrado, ombro unilateral}."""
    fa = 560
    fb = 600 - tail
    left_sh = 13 if content == "one_sided_shoulder" else 0
    if step == 0:
        # paginas alinhadas: a janela de voto atravessa a divisa pela linha
        # pareada da trilha vizinha (gap apontado no review d64716a1)
        prob = _fold_page(
            fa=fa, fb=fb, left_sh=left_sh, split_x=600, offset_r=0, x_end=1200
        )
    else:
        prob = _fold_page(
            fa=fa, fb=fb, left_sh=left_sh, split_x=600, offset_r=step, x_end=1200
        )

    y_stroke = 80 + 40 * 5
    stroke_mask = np.zeros(prob.shape, bool)
    if content == "straight_stroke":
        stroke_mask[y_stroke : y_stroke + 6, fa - 20 : fb + 40] = True
    elif content == "bending_stroke":
        bend = step if step else 20
        stroke_mask[y_stroke : y_stroke + 6, fa - 20 : 600] = True
        for x in range(600, 600 + bend):
            dy = x - 600
            stroke_mask[y_stroke + dy : y_stroke + dy + 6, x] = True
        stroke_mask[y_stroke + bend : y_stroke + bend + 6, 600 + bend : 640] = True

    prob[stroke_mask] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    if content in ("empty", "one_sided_shoulder"):
        assert (out[:, fa + 2 : fb - 2] > 0.5).sum() == 0
    else:
        inside = np.zeros(prob.shape, bool)
        inside[:, fa + 2 : fb - 2] = True
        assert (out[stroke_mask & inside] > 0.5).all()


def test_a_stroke_ending_in_blank_space_past_the_fold_is_kept():
    """Review d64716a1: costura escura + traco na pauta esquerda terminando
    na col 653 (13 col em branco alem do vinco), com ombros de 13 nas linhas
    do destino. O veredito de ombro so vale para ramos que aterrissam em
    pauta do destino — o miolo do traco fica INTEIRO."""
    prob = _fold_page(right_sh=13, split_x=600, offset_r=20)
    y_seam = 80 + 40 * 2
    prob[y_seam : y_seam + 26, 599:601] = 0.95
    y = 80 + 40 * 5
    prob[y : y + 6, 540:653] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 562:653] > 0.5).all()
    # a costura escura (pauta do destino) segue apagada no vinco
    assert (out[y_seam : y_seam + 6, 562:598] > 0.5).sum() == 0


def test_a_long_stroke_on_one_side_does_not_save_a_shoulder_on_the_other():
    """Review d64716a1: numa trilha so, traco longo a esquerda (40 col,
    acima do ombro comum) e so ombro de 13 a direita: o filtro de ombro tem
    de valer mesmo quando origem e destino sao a mesma trilha — a faixa sai
    inteira."""
    prob = np.zeros((600, 1200), np.float32)
    fa, fb = 560, 640
    for k in range(12):
        y = 80 + 40 * k
        prob[y : y + 2, :] = 0.95
        prob[y : y + 6, fa:fb] = 0.95
        if k % 2 == 0:
            prob[y : y + 6, fa - 13 : fa] = 0.95
            prob[y : y + 6, fb : fb + 13] = 0.95
    y = 80 + 40 * 5
    prob[y : y + 6, 520 : fb + 13] = 0.95  # traco longo a esq, ombro comum a dir
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, fa + 2 : fb - 2] > 0.5).sum() == 0


def test_a_shadow_shoulder_on_a_short_tail_does_not_protect_the_fold():
    """Review d64716a1: vinco [560, 594) a 6 col do fim da trilha. Um ombro
    de 13 col numa linha nao isenta a faixa — cauda curta exige continuacao
    conectada para alem do limite do vinco."""
    prob = _fold_page(fa=560, fb=594, x_end=600)
    y = 80 + 40 * 5
    prob[y : y + 6, 547:560] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 562:594] > 0.5).sum() == 0


def test_a_stroke_on_a_short_tail_still_crosses_the_fold():
    """Espelho do anterior: mesma cauda curta de 6 col, mas com traco real
    continuando 26 col alem do vinco — continua inteiro."""
    prob = _fold_page(fa=560, fb=594, x_end=600)
    y = 80 + 40 * 5
    prob[y : y + 6, 540:620] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    _assert_only_stroke_kept(out, slice(y, y + 6), 562, 594)


def test_aligned_pages_split_the_vote_window_across_the_seam():
    """Duas metades alinhadas e o vinco termina 6 col antes da divisa: a
    janela de voto do lado direito atravessa a costura (6 col de cauda +
    colunas da trilha vizinha). Com traco, fica; sem traco, sai."""
    prob = _fold_page(fa=560, fb=594, split_x=600, offset_r=0)
    y = 80 + 40 * 5
    prob[y : y + 6, 540:612] = 0.95  # 6 na cauda + 6 na trilha vizinha + folga
    out = ruling.RulingFilter().apply(prob.copy())
    _assert_only_stroke_kept(out, slice(y, y + 6), 562, 594)


def test_unmatched_shadow_lines_do_not_veto_a_matched_stroke():
    """Pautas com periodos diferentes (40 vs 48 col): quase todas as linhas
    ficam sem par e levam ombros de 18 col. O traco de 16 col na linha
    pareada e julgado pelo conjunto pareado, nao pelo bulk dos sem-par."""
    prob = np.zeros((600, 1400), np.float32)
    for k in range(12):  # esquerda: periodo 40
        y = 80 + 40 * k
        prob[y : y + 2, :600] = 0.95
        prob[y : y + 6, 560:600] = 0.95
    for j in range(11):  # direita: periodo 48, so a linha 250 casa
        y = 90 + 48 * j
        prob[y : y + 2, 600:1400] = 0.95
        prob[y : y + 6, 600:640] = 0.95
        if j % 2 == 0:
            prob[y : y + 6, 640 : 640 + 18] = 0.95  # ombros nas linhas sem par
    y = 80 + 40 * 4  # 240 ~ pauta dir 250 -> par
    prob[y : y + 6, 544:656] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 562:598] > 0.5).all()
    assert (out[y : y + 6, 602:638] > 0.5).all()


def test_a_fold_reaching_the_image_edge_with_a_short_tail_is_erased():
    """Borda do dominio, nao borda de evidencia: faixa [4, 1200) cobre as
    duas metades ate a borda da imagem, e sobram so 4 col observaveis a
    esquerda. Continuacao de 4 col nao prova a janela de 12 — o veredito
    abstem e a sombra sai, mesmo com um traco chegando ate a coluna 0."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        yl, yr = 80 + 40 * k, 80 + 40 * k
        prob[yl : yl + 2, :600] = 0.95
        prob[yr : yr + 2, 600:] = 0.95
        prob[yl : yl + 6, 4:600] = 0.95
        prob[yr : yr + 6, 600:] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 6:1200] > 0.5).sum() == 0

    prob2 = prob.copy()
    y = 80 + 40 * 5
    prob2[y : y + 6, 0:1200] = 0.95  # traco real encostando na borda esquerda
    out = ruling.RulingFilter().apply(prob2.copy())
    assert (out[:, 6:1200] > 0.5).sum() == 0


def test_unmatched_rule_with_source_shoulder_length_keeps_blank_continuation():
    """Review: pautas defasadas (sem par na divisa) com ombros de 13 col na
    origem; um traco entra com a mesma extensao de ombro (13 col) e continua
    26 col em branco do lado do destino. O ombro da origem e provisorio —
    a continuacao qualificada protege o trecho na pauta."""
    prob = _fold_page(split_x=600, offset_r=20, left_sh=13)
    y = 80 + 40 * 5  # 280: linha esquerda sem par na direita
    prob[y : y + 6, 547:666] = 0.95  # 13 a esquerda, 26 em branco a direita
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 547:666] > 0.5).all()


def test_fork_past_the_fold_keeps_blank_sibling_when_shadow_sibling_ends_on_a_rule():
    """Review: traco segue a pauta pelo vinco, desce na faixa ate encostar na
    pauta vizinha (que tem ombro de 13 col a direita) e termina 13 col depois
    do vinco em espaco em branco. A ponta do irmao de sombra nao veta o ramo
    independente do traco."""
    prob = _fold_page(right_sh=13)
    y = 80 + 40 * 5  # 280
    stroke = np.zeros(prob.shape, bool)
    stroke[y : y + 6, 540:600] = True
    for x in range(600, 640):  # desce 1 px/col na faixa ate a pauta 320
        dy = x - 600
        stroke[y + dy : y + dy + 6, x] = True
    for x in range(640, 654):  # 13 col de saida descendo para o branco
        dy = 40 - (x - 640)
        stroke[y + dy : y + dy + 6, x] = True
    prob[stroke] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[stroke] > 0.5).all()


def test_full_width_thick_ruling_without_observable_side_is_erased():
    """Review: faixa cobrindo o crop inteiro ([0, w) - nenhum lado
    observavel) nao pode ser isenta por voto de borda: sem evidencia de
    continuacao, a pauta engrossada de ponta a ponta sai."""
    prob = np.zeros((600, 1200), np.float32)
    for k in range(12):
        y = 80 + 40 * k
        th = 7 if k % 3 != 0 else 2
        prob[y : y + th, :] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out > 0.5).sum() == 0


def test_rejoining_thick_stroke_and_thin_rule_keeps_the_stroke():
    """Review: um traco grosso e a pauta fina se separam apos o vinco e se
    reencontram 5 col depois; o reencontro nao pode enterrar a historia
    grossa do traco (a uniao das sequencias e a da tinta conectada)."""
    prob = _fold_page()
    y = 80 + 40 * 5  # 280
    prob[y : y + 6, 540:600] = 0.95
    for x in range(600, 640):  # dobra na faixa ate a ponta ~318-323
        dy = x - 600
        prob[y + dy : y + dy + 6, x] = 0.95
    prob[323:329, 640:645] = 0.95  # traco grosso, gap de 1 linha sob a pauta 320
    prob[320:329, 645] = 0.95  # ponte vertical: reencontro traco + pauta
    for x in range(646, 662):  # traco segue grosso descendo p/ o branco
        dy = x - 646
        prob[323 + dy : 329 + dy, x] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 562:600] > 0.5).all()


def test_thin_tip_off_the_rule_does_not_rescue_a_shadow_shoulder():
    """Review: traco alcanca a faixa e sai por um ombro de 13 col que vira a
    pauta fina + uma ponta fina de 2 px fora dela. A ponta fina nao prova
    nada: quem responde ao veto e o ultimo intervalo grosso (na pauta)."""
    prob = _fold_page(right_sh=13)
    y = 80 + 40 * 5  # 280
    prob[y : y + 6, 540:600] = 0.95
    for x in range(600, 640):  # dobra ate o ombro da pauta 320
        dy = x - 600
        prob[y + dy : y + dy + 6, x] = 0.95
    prob[325:328, 653:662] = 0.95  # ponta fina fora da pauta, colada ao ombro
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 562:600] > 0.5).sum() == 0
    assert (out[320:326, 600:630] > 0.5).sum() == 0


def test_repeated_reunions_do_not_duplicate_paths(monkeypatch):
    """Review: separa/reencontra a cada ciclo produzia estados identicos que
    dobravam por bifurcacao (16 ciclos -> 65.536 caminhos, >2 s por passada).
    Estados iguais sao o mesmo caminho: o traco conta cada fim uma vez."""
    calls = 0
    extent = ruling._extent

    def counting(arr, need):
        nonlocal calls
        calls += 1
        return extent(arr, need)

    monkeypatch.setattr(ruling, "_extent", counting)
    h, w = 90, 46
    ink = np.zeros((h, w), np.uint8)
    ink[40:46, :] = 1  # pauta grossa
    ink[48:54, :] = 1  # traco paralelo 2 px abaixo
    ink[46:48, ::2] = 1  # ponte vertical: reencontro a cada 2 col
    paths = ruling._trace_branch_out(ink, 0, 40, 53, w - 1, 1, 3, 12, 33)
    assert paths  # a continuação grossa existe e qualifica
    assert calls <= 16  # sem dedup seriam ~2^22 fins de caminho


def test_reunions_with_mixed_widths_stay_bounded(monkeypatch):
    """Review: o teste de dedup usava larguras iguais (estados idênticos se
    fundem). Com pauta fina + traço grosso as histórias diferem e o dedup
    não funde — mede o trabalho que SOBREVIVE: linear, não 2^n.
    `_extent` conta só quem passa no voto — `flatnonzero` roda uma vez por
    estado ativo por coluna e mede a expansão inteira, incluindo ramos que
    o voto rejeita depois."""
    calls = 0
    extent = ruling._extent

    def counting(arr, need):
        nonlocal calls
        calls += 1
        return extent(arr, need)

    scans = 0
    flatnonzero = ruling.np.flatnonzero

    def counting_scan(arr):
        nonlocal scans
        scans += 1
        return flatnonzero(arr)

    monkeypatch.setattr(ruling, "_extent", counting)
    monkeypatch.setattr(ruling.np, "flatnonzero", counting_scan)
    h, w = 90, 46
    ink = np.zeros((h, w), np.uint8)
    ink[40:42, :] = 1  # pauta fina (2 px)
    ink[44:50, :] = 1  # traço grosso (6 px) 2 px abaixo
    ink[42:44, ::2] = 1  # ponte vertical: reencontro a cada 2 col
    paths = ruling._trace_branch_out(ink, 0, 40, 49, w - 1, 1, 3, 12, 33)
    assert paths
    assert calls <= 128  # ~2 por coluna observado; 2^22 estaria fora da mesa
    # ~34 estados/coluna observados: a expansão pré-voto existe e é medida
    # aqui — o cap é tripwire de regressão exponencial, não bound justo
    assert scans <= 2048


def test_thin_branch_after_reunion_does_not_borrow_thickness():
    """Review: so a ponta fina sai da pauta depois do reencontro. Com
    historias por camada ela nao herda as colunas grossas do traco: a
    sombra nao tem continuacao qualificada e sai do vinco."""
    prob = _fold_page()
    y = 80 + 40 * 5  # 280
    prob[y : y + 6, 540:600] = 0.95
    for x in range(600, 640):  # dobra ate a pauta 320 na borda da faixa
        dy = x - 600
        prob[y + dy : y + dy + 6, x] = 0.95
    prob[320:329, 640:643] = 0.95  # reencontro: bolha grossa traco+pauta
    prob[323:325, 643:662] = 0.95  # so o esporro fino sai da pauta
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[320:326, 600:630] > 0.5).sum() == 0


def test_a_shadow_bridge_between_sparse_pairs_does_not_protect_the_fold():
    """Review: metades defasadas 8 px + ombros de 13 col — só um par emenda na
    costura via remendo escuro de 2 col. Com 1 par não há referência de ombro
    comum medida (min_bulk(1)=2), então o voto cru não decide: o par é provado
    pelo traço, que desemboca no ombro do destino e sai inteiro."""
    prob = _fold_page(offset_r=8, left_sh=13, right_sh=13, split_x=600)
    prob[80:90, 598:602] = 0.97  # remendo ligando o par (0, 0) na costura
    out = ruling.RulingFilter().apply(prob.copy())
    # a faixa sai; o próprio remendo na costura é tinta conectada e pode
    # ficar (desvio de espessura = assinatura de traço na sombra)
    assert (out[80:96, 562:596] > 0.5).sum() == 0
    assert (out[80:96, 606:638] > 0.5).sum() == 0


def test_a_real_crossing_between_sparse_pairs_is_kept():
    """Mesmo layout esparso, mas com traço real cruzando a faixa em cima da
    pauta esquerda k=2 e saindo para espaço em branco do outro lado — o
    traço conectado qualifica e o trecho no vinco fica."""
    prob = _fold_page(offset_r=8, left_sh=13, right_sh=13, split_x=600)
    prob[80:90, 598:602] = 0.97  # o remendo de sombra continua saindo
    y = 80 + 40 * 2  # 160
    prob[y : y + 6, 540:660] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[y : y + 6, 560:640] > 0.5).all()
    assert (out[80:96, 562:596] > 0.5).sum() == 0
    assert (out[80:96, 606:638] > 0.5).sum() == 0


def test_a_shoulder_ending_on_the_neighboring_track_does_not_protect_the_fold():
    """Review: vinco [560, 593) dentro da trilha esquerda (divisa em 600).
    O ombro comum de 13 col atravessa a divisa e termina na pauta da trilha
    vizinha — o veto tem de resolver a trilha do terminal por coluna:
    ombro é ombro mesmo fora da trilha da faixa."""
    prob = _fold_page(fa=560, fb=593, split_x=600, left_sh=13, right_sh=13)
    prob[160:166, 480:560] = 0.97  # traço grosso longo à esquerda da linha 2
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 562:591] > 0.5).sum() == 0


def test_a_shoulder_before_the_split_does_not_protect_the_right_fold():
    """Espelho: vinco [607, 640) dentro da trilha direita. O ombro de 13 col
    termina na pauta da trilha esquerda (594..606 cruzando a divisa)."""
    prob = _fold_page(fa=607, fb=640, split_x=600, left_sh=13, right_sh=13)
    prob[160:166, 640:720] = 0.97  # traço grosso longo à direita da linha 2
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 609:638] > 0.5).sum() == 0


def test_a_stroke_bending_onto_offset_ruling_is_not_a_shoulder():
    """Review: faixa [560,588) antes da divisa (600), pauta direita deslocada
    8 px, ombros do lado da costura terminando em 599. Um traço de 6 px segue
    a pauta esquerda e dobra para a pauta direita em 599-602: o veto de ombro
    não pode olhar só o ponto final — só as colunas de pouso ficam na pauta,
    o resto da saída é dobra, não ombro."""
    prob = _fold_page(fa=560, fb=588, left_sh=13, right_sh=12, split_x=600, offset_r=8)
    y = 280  # pauta esquerda k=5; a direita equivalente é y+8=288
    prob[y : y + 6, 530:599] = 0.95
    for i, x in enumerate(range(599, 604)):
        dy = min(8, 2 * i)
        prob[y + dy : y + dy + 6, x] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[280:286, 562:586] > 0.5).all()
    others = out[:, 562:586] > 0.5
    others[280:286, :] = False
    assert others.sum() == 0


def test_an_offset_bend_that_only_lands_is_still_a_shoulder():
    """Controle do anterior: um ombro comum de 13 col que SEGUE a pauta até
    depois da divisa continua vetado — a dobra libera o veto porque só
    termina na pauta, não porque termina nela."""
    prob = _fold_page(fa=560, fb=588, left_sh=13, right_sh=12, split_x=600, offset_r=0)
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[:, 562:586] > 0.5).sum() == 0


def test_a_sparse_pair_with_an_in_fold_gap_is_not_protected_by_foreign_shoulder():
    """Review: metades defasadas 8 px, só o par (0,0) emendado por remendo
    escuro na costura. Ombros de 13 col nas linhas alternadas à esquerda —
    o lado inteiro mede referência de fallback, mas o conjunto pareado não
    sustenta a própria. Com gap de 7 col dentro da faixa na pauta direita,
    o par está desconectado: o voto cru não decide, o traço encontra o gap."""
    prob = _fold_page(offset_r=8, left_sh=13, right_sh=0, split_x=600)
    prob[80:90, 598:602] = 0.97  # remendo ligando o par (0,0) na costura
    prob[88:94, 640:653] = 0.95  # ombro de 13 col só na pauta direita do par
    prob[88:94, 610:617] = 0  # gap de 7 col dentro da faixa na pauta direita
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[80:86, 562:596] > 0.5).sum() == 0
    assert (out[88:94, 606:638] > 0.5).sum() == 0


def test_a_gradual_bend_across_the_seam_survives_on_a_sparse_pair():
    """Review: metades defasadas 8 px + ombros de 13 col — um traço de 6 px
    numa pauta SEM ombro dobra gradualmente através da costura para a pauta
    direita, estendendo 13 col além das bordas da faixa. O traço cria o único
    par: sem referência própria, o ombro das pautas soltas não pode vetá-lo.
    A dobra distribui o desnível ao longo das colunas — diferente do remendo
    de sombra, que concentra o gap inteiro numa coluna."""
    prob = _fold_page(offset_r=8, left_sh=13, right_sh=13, split_x=600)
    y = 280  # pauta esquerda k=5 (ímpar, sem ombro); a direita é y+8=288
    prob[y : y + 6, 547:597] = 0.95
    for x in range(597, 604):
        prob[y + (x - 597) : y + (x - 597) + 6, x] = 0.95
    prob[288:294, 604:653] = 0.95
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[280:286, 562:596] > 0.5).all()
    assert (out[288:294, 604:638] > 0.5).all()
    # os ombros das outras pautas continuam saindo
    assert (out[80:86, 562:596] > 0.5).sum() == 0
    assert (out[88:94, 606:638] > 0.5).sum() == 0


def _bend_across_offset_seam(mark_shift=None):
    """Traço de 6 px que dobra 8 px na costura de metades defasadas (par
    esparso). `mark_shift` põe dois cruzamentos verticais de caneta, longe da
    costura, deslocados do nível da pauta em `mark_shift` linhas."""
    prob = _fold_page(split_x=600, offset_r=8)
    y = 280
    prob[y : y + 6, 530:599] = 0.95
    for i, x in enumerate(range(599, 604)):
        dy = min(8, 2 * i)
        prob[y + dy : y + dy + 6, x] = 0.95
    prob[y + 8 : y + 14, 603:660] = 0.95
    if mark_shift is not None:
        for x in (575, 620):
            prob[y - 10 + mark_shift : y + 16 + mark_shift, x : x + 2] = 0.95
    return prob


@pytest.mark.parametrize("mark_shift", [None, 0, 2, 4, 7])
def test_pen_crossings_away_from_the_seam_do_not_erase_a_genuine_bend(mark_shift):
    """Review #139 (_seam_bridge): um cruzamento alto longe da costura não liga
    as pautas defasadas, então não é evidência de remendo de sombra. TODO pixel
    da dobra dentro da faixa fica — conferido pela máscara do traço, não por
    total, que as marcas de caneta inflariam."""
    box = (slice(280, 294), slice(562, 638))
    expected = (_bend_across_offset_seam(None) > 0.5)[box]
    out = ruling.RulingFilter().apply(_bend_across_offset_seam(mark_shift).copy())
    kept = (out > 0.5)[box]
    marks = np.zeros_like(expected)
    marks[:, [575 - 562, 576 - 562, 620 - 562, 621 - 562]] = True  # as marcas
    assert expected.sum() == 468
    assert (kept & expected & ~marks).sum() == (expected & ~marks).sum()


def test_a_wide_shadow_patch_joining_offset_rules_does_not_pass_as_a_bend():
    """Review #139: remendo largo (40 col) de 12 px ligando pautas defasadas
    8 px. A largura sobe a mediana da espessura e escondia a ponte; a ponte
    se mede onde as pautas se ligam, não contra a mediana do trecho."""
    prob = _fold_page(offset_r=8, left_sh=13, right_sh=13, split_x=600)
    prob[80:92, 580:620] = 0.97
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[80:96, 562:578] > 0.5).sum() == 0
    assert (out[80:96, 622:638] > 0.5).sum() == 0


def test_a_staggered_shadow_patch_joining_offset_rules_does_not_pass_as_a_bend():
    """Review #139: o remendo liga as pautas por intervalos que só se
    sobrepõem em colunas vizinhas (80-91 à esquerda, 82-93 à direita), nenhum
    cobrindo o desnível inteiro sozinho. Duas colunas seguidas quase tão altas
    já são ponte."""
    prob = _fold_page(offset_r=8, left_sh=13, right_sh=13, split_x=600)
    prob[82:92, 598:602] = 0.97
    out = ruling.RulingFilter().apply(prob.copy())
    assert (out[80:96, 562:596] > 0.5).sum() == 0
    assert (out[80:96, 606:638] > 0.5).sum() == 0


def _chain(spans, x0):
    """Cadeia de `_Run` (x crescente) a partir de (top, bot) por coluna."""
    nodes = [ruling._Run(x0 + i, top, bot, i - 1) for i, (top, bot) in enumerate(spans)]
    return nodes, len(nodes) - 1


def test_seam_bridge_reads_both_levels_from_the_chain_when_the_back_trace_is_empty():
    """Review #139: na cauda curta o traço de volta não cobre coluna fora da
    janela (`src_tops` vazio). A cadeia cruza a divisa, então os dois níveis
    saem dela: o remendo (12 px de altura numa coluna só) é ponte; a dobra
    gradual, não."""
    split = 600
    flat_a = [(280, 285)] * 12  # pauta de origem, x 588..599
    flat_b = [(288, 293)] * 12  # pauta de destino, x 600..611 (+ janela)
    patch = flat_a[:10] + [(280, 293)] * 4 + flat_b[2:]  # remendo em 598..601
    bend = flat_a + [(282, 287), (284, 289), (286, 291)] + [(288, 293)] * 18
    nodes, tid = _chain(patch, 588)
    assert ruling._seam_bridge(nodes, tid, 640, 1, split, [])
    nodes, tid = _chain(bend, 588)
    assert not ruling._seam_bridge(nodes, tid, 640, 1, split, [])
    # sem nível de origem observável (nem na cadeia): abstém
    nodes, tid = _chain([(288, 293)] * 30, 600)
    assert not ruling._seam_bridge(nodes, tid, 640, 1, split, [])


@pytest.mark.parametrize(
    ("offset", "tops"),
    [(4, [281, 282, 282, 283]), (4, [281, 282, 283]), (5, [281, 282, 283, 284])],
)
def test_a_gradual_pen_bend_between_slightly_offset_rules_is_kept(offset, tops):
    """Review #139 (18127fb): com desnível pequeno, duas colunas do traço que
    engordam só por encostar na pauta sombreada (espessura + poucos px) não
    são remendo de sombra. A dobra suave de 1 linha por coluna fica inteira."""
    prob = _fold_page(split_x=600, offset_r=offset)
    y = 280
    prob[y : y + 6, 530:598] = 0.95
    for i, top in enumerate(tops):
        prob[top : top + 6, 598 + i] = 0.95
    prob[y + offset : y + offset + 6, 598 + len(tops) : 660] = 0.95
    box = (slice(278, 296), slice(562, 638))
    out = ruling.RulingFilter().apply(prob.copy())
    assert ((out > 0.5)[box]).sum() == ((prob > 0.5)[box]).sum()


def test_a_midpoint_stroke_between_small_offset_rules_is_kept_by_policy():
    """Review #139 (734ed7a): com desnivel de 4 px, um traço de 6 px parado na
    altura intermediária (topos 282) tem EXATAMENTE a tinta de um remendo de
    sombra escalonado de 8 px (intervalos 80-87 e 82-89). Nenhuma medida local
    de altura ou progressão separa os dois; a política é conservadora: com
    traço conectado entrando e saindo da faixa, preserva a tinta. Perder
    caneta custa mais que deixar sombra."""
    prob = _fold_page(split_x=600, offset_r=4)
    y = 280
    prob[y : y + 6, 530:598] = 0.95
    for x in range(598, 602):
        prob[y + 2 : y + 8, x] = 0.95  # traço parado 2 linhas abaixo da pauta
    prob[y + 4 : y + 10, 602:660] = 0.95
    box = (slice(278, 296), slice(562, 638))
    out = ruling.RulingFilter().apply(prob.copy())
    assert ((out > 0.5)[box]).sum() == ((prob > 0.5)[box]).sum()
    # o filtro RODOU: a sombra das outras pautas dentro da faixa sai, então a
    # preservação acima é proteção de traço e não remoção pulada
    others = (out > 0.5)[:, 562:638].copy()
    others[278:296, :] = False
    assert others.sum() == 0
    assert (prob > 0.5)[:, 562:638].sum() > (out > 0.5)[:, 562:638].sum()


def test_a_staggered_shadow_patch_is_identical_ink_to_a_midpoint_stroke():
    """Review #139 (remendo escalonado, desnível 4 px): a sombra com ombros
    de 13 col + intervalos 80-87 na origem e 82-89 no destino gera EXATAMENTE
    o mapa de tinta de um traço que dobra pelo meio do desnível na mesma
    pauta — array_equal dos dois mapas. Nenhuma regra sobre esse mapa separa
    as duas interpretações; a política conservadora fica fixada por este par
    (apagar aqui apagaria caneta). Separar de verdade exige evidência fora do
    mapa binário — a imagem original (iluminação do papel) — limitação
    conhecida, fora do escopo deste filtro."""
    patch = _fold_page(split_x=600, offset_r=4)
    patch[80:86, 547:560] = 0.95  # ombros da pauta sombreada
    patch[84:90, 640:653] = 0.95
    patch[80:88, 598:600] = 0.95  # remendo esq: intervalo 80-87
    patch[82:90, 600:602] = 0.95  # remendo dir: intervalo 82-89

    stroke = _fold_page(split_x=600, offset_r=4)
    stroke[80:86, 547:598] = 0.95  # traço grosso seguindo a pauta esquerda
    stroke[84:90, 640:653] = 0.95
    for x in range(598, 602):
        stroke[82:88, x] = 0.95  # dobra pelo meio do desnível
    stroke[84:90, 602:653] = 0.95  # aterrissa na pauta direita

    np.testing.assert_array_equal(patch, stroke)
    out_patch = ruling.RulingFilter().apply(patch.copy())
    out_stroke = ruling.RulingFilter().apply(stroke.copy())
    np.testing.assert_array_equal(out_patch, out_stroke)
    # política conservadora: a tinta ambígua fica (a sombra das outras pautas sai)
    kept = (out_patch > 0.5)[80:90, 540:660].sum()
    assert kept >= 0.97 * (patch > 0.5)[80:90, 540:660].sum()
    others = (out_patch > 0.5)[:, 562:638].copy()
    others[80:90, :] = False
    assert others.sum() == 0


def test_project_forest_skips_the_back_trace_when_no_exit_tip(monkeypatch):
    """Custo (análise #139): sem ponta de saída na floresta o veredito é
    vazio de qualquer forma — o traço de volta na origem (o ramo mais caro,
    ~78% do tempo no cenário de dobra) não pode rodar. `_trace_ink_run` é
    chamado uma única vez por projeção sem saída."""
    calls = []
    monkeypatch.setattr(
        ruling, "_trace_ink_run", lambda *a: calls.append(1) or ([], [])
    )

    class _T:
        x0, x1 = 0, 600
        paths = np.full((1, 600), 80.0, dtype=np.float32)

    t = _T()
    out = ruling._project_forest(
        np.zeros((600, 1200), np.uint8),
        t,
        0,
        np.ones(600, bool),
        599,
        640,
        560,
        1,
        6,
        12,
        [t],
        [np.ones((1, 600), bool)],
    )
    assert out == ([], [])
    assert len(calls) == 1
