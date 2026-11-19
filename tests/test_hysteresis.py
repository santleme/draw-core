"""Histerese espaco-temporal da tinta (InkConfig.hysteresis)."""

import sys

import numpy as np
import pytest
from lab import bench_ink as bench

from draw import app as app_module
from draw.ink import accumulator as accumulator_mod
from draw.ink import hysteresis
from draw.ink.config import InkConfig

H, W = 60, 80


def _mask(*boxes, shape=(H, W)):
    m = np.zeros(shape, bool)
    for y0, y1, x0, x1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


# ------------------------------------------------------------ keep_anchored


def test_weak_component_without_an_anchor_is_dropped():
    weak = _mask((10, 13, 10, 13))
    assert not hysteresis.keep_anchored(weak, np.zeros_like(weak)).any()


def test_weak_component_touching_an_anchor_is_kept_including_the_diagonal():
    weak = _mask((10, 13, 10, 13))
    side = _mask((10, 11, 13, 14))  # ancora colada pelo lado
    diag = _mask((13, 14, 13, 14))  # e outra so pela diagonal (8-conectividade)
    assert hysteresis.keep_anchored(weak, side).sum() == weak.sum()
    assert hysteresis.keep_anchored(weak, diag).sum() == weak.sum()


def test_anchor_two_pixels_away_does_not_count():
    weak = _mask((10, 13, 10, 13))
    far = _mask((10, 11, 15, 16))
    assert not hysteresis.keep_anchored(weak, far).any()


def test_a_chain_of_weak_pixels_reaches_the_anchor():
    weak = _mask((20, 21, 5, 40))  # fio de 35 px
    anchor = _mask((20, 22, 40, 42))
    assert hysteresis.keep_anchored(weak, anchor).sum() == weak.sum()


def test_only_the_anchored_component_survives_among_several():
    a, b = _mask((5, 8, 5, 8)), _mask((30, 33, 50, 53))
    out = hysteresis.keep_anchored(a | b, _mask((5, 6, 8, 9)))
    assert out[a].all() and not out[b].any()


def test_keep_anchored_never_returns_pixels_outside_weak_nor_mutates():
    weak = _mask((10, 13, 10, 13))
    anchors = _mask((10, 11, 13, 14))
    before_w, before_a = weak.copy(), anchors.copy()
    out = hysteresis.keep_anchored(weak, anchors)
    assert not (out & ~weak).any()
    np.testing.assert_array_equal(weak, before_w)
    np.testing.assert_array_equal(anchors, before_a)


def test_keep_anchored_works_at_the_image_border():
    weak = _mask((0, 3, 0, 3))
    anchors = _mask((0, 1, 3, 4))
    assert hysteresis.keep_anchored(weak, anchors).sum() == 9


# -------------------------------------------------------- filter_provisional


def _filter(prov, evidence=None, confirmed=None, min_px=12, t_high=0.9):
    ev = np.zeros(prov.shape, np.float32) if evidence is None else evidence
    conf = np.zeros(prov.shape, bool) if confirmed is None else confirmed
    return hysteresis.filter_provisional(prov, ev, conf, t_high=t_high, min_px=min_px)


def test_provisional_needs_the_minimum_size_and_an_anchor():
    big = _mask((10, 14, 10, 20))  # 40 px
    ev = np.zeros((H, W), np.float32)
    assert not _filter(big, ev).any()  # grande, mas sem ancora
    ev[11, 12] = 0.95  # uma ancora forte dentro
    assert _filter(big, ev).sum() == big.sum()
    tiny = _mask((30, 32, 30, 32))  # 4 px
    ev[30, 30] = 0.99
    assert not _filter(tiny, ev).any()  # ancorado, mas pequeno demais


def test_provisional_size_boundary_is_inclusive():
    ev = np.full((H, W), 0.95, np.float32)
    assert _filter(_mask((5, 6, 5, 17)), ev, min_px=12).any()  # 12 px
    assert not _filter(_mask((5, 6, 5, 16)), ev, min_px=12).any()  # 11 px


def test_provisional_touching_confirmed_ink_counts_as_anchored():
    prov = _mask((10, 14, 10, 20))
    conf = _mask((10, 14, 20, 24))  # tinta confirmada ao lado
    assert _filter(prov, None, conf).sum() == prov.sum()
    far = _mask((10, 14, 22, 26))
    assert not _filter(prov, None, far).any()


def test_provisional_filter_handles_empty_input():
    assert not _filter(np.zeros((H, W), bool)).any()


# ---------------------------------------------------------------- acumulador


def _acc(config):
    return accumulator_mod.LayeredInkAccumulator(config.seen_threshold, config)


def _run(config, maps, visible=None):
    acc = _acc(config)
    out = None
    shown = []
    for m in maps:
        out = acc.add(m.copy(), visible)
        shown.append(out["provisional"] > 0)
    return out, shown


def _frames(blob_value, n=4, stroke=0.97, blob_box=(40, 43, 60, 63)):
    stroke_box = _mask((10, 16, 5, 35))
    maps = []
    for _ in range(n):
        m = np.zeros((H, W), np.float32)
        m[stroke_box] = stroke
        y0, y1, x0, x1 = blob_box
        m[y0:y1, x0:x1] = blob_value
        maps.append(m)
    return maps, stroke_box, _mask(blob_box)


def test_default_config_is_identical_with_the_flag_off():
    """hysteresis=False nao muda um bit do acumulador."""
    rng = np.random.default_rng(1)
    maps = [(rng.random((H, W)) ** 3).astype(np.float32) for _ in range(6)]
    a, shown_a = _run(InkConfig(), maps)
    b, shown_b = _run(InkConfig(hysteresis=False, t_high=0.8, t_low=0.5), maps)
    np.testing.assert_array_equal(a["confirmed"], b["confirmed"])
    for x, y in zip(shown_a, shown_b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_isolated_weak_dust_confirms_without_the_flag_and_not_with_it():
    # media 0,75: confirma pelo voto padrao, mas e fraco (< t_high) e isolado
    maps, stroke, blob = _frames(0.75)
    plain, _ = _run(InkConfig(), maps)
    assert (plain["confirmed"][blob] > 0).any()
    out, _ = _run(InkConfig(hysteresis=True), maps)
    assert not (out["confirmed"][blob] > 0).any()
    assert (out["confirmed"][stroke] > 0).all()  # o traco forte fica inteiro


def test_isolated_strong_speckle_is_its_own_anchor():
    """Limite declarado: poeira persistente e forte (media >= t_high) e
    ancora de si mesma e confirma."""
    maps, _, blob = _frames(0.97)
    out, _ = _run(InkConfig(hysteresis=True), maps)
    assert (out["confirmed"][blob] > 0).all()


def test_weak_pixels_next_to_confirmed_ink_confirm_but_below_t_low_do_not():
    stroke_box = _mask((10, 16, 5, 35))
    halo = _mask((10, 16, 35, 37))  # colado ao traco
    maps = []
    for value in (0.75, 0.55):
        frames = []
        for _ in range(4):
            m = np.zeros((H, W), np.float32)
            m[stroke_box] = 0.97
            m[halo] = value
            frames.append(m)
        maps.append(frames)
    out, _ = _run(InkConfig(hysteresis=True), maps[0])
    assert (out["confirmed"][halo] > 0).all()  # fraco >= t_low, ancorado
    out, _ = _run(InkConfig(hysteresis=True), maps[1])
    assert not (out["confirmed"][halo] > 0).any()  # < t_low: nem ancorado entra


def _cycled_halo_frames(cycle, n=6):
    stroke_box, halo = _mask((10, 16, 5, 35)), _mask((10, 16, 35, 37))
    frames = []
    for k in range(n):
        m = np.zeros((H, W), np.float32)
        m[stroke_box] = 0.97
        m[halo] = cycle[k % len(cycle)]
        frames.append(m)
    return frames, halo


def test_t_low_floor_drops_a_weak_halo_that_the_plain_vote_would_confirm():
    """Halo que alterna 0,72/0,52/0,52: media 0,59 (< t_low), visto como
    tinta e 3 hits: o voto padrao confirma; com histerese, nem colado a uma
    ancora entra (piso t_low)."""
    frames, halo = _cycled_halo_frames((0.72, 0.52, 0.52))
    plain, _ = _run(InkConfig(), frames)
    assert (plain["confirmed"][halo] > 0).any()
    out, _ = _run(InkConfig(hysteresis=True), frames)
    assert not (out["confirmed"][halo] > 0).any()


def test_weak_pixels_anchor_to_ink_confirmed_earlier_even_if_it_is_no_longer_strong():
    stroke_box, halo = _mask((10, 16, 5, 35)), _mask((10, 16, 35, 37))
    frames = []
    for k in range(8):
        m = np.zeros((H, W), np.float32)
        m[stroke_box] = 0.97 if k < 4 else 0.6  # confirmada antes; depois fraca
        if k >= 4:
            m[halo] = 0.75  # fraco (>= t_low, < t_high), so a tinta confirmada ancora
        frames.append(m)
    out, _ = _run(InkConfig(hysteresis=True), frames)
    assert (out["confirmed"][stroke_box] > 0).all()
    assert (out["confirmed"][halo] > 0).all()


def test_provisional_layer_hides_flicker_but_shows_the_new_stroke_at_once():
    maps, stroke, blob = _frames(0.0)
    flicker = maps[0].copy()
    flicker[blob] = 0.95  # poeira de UMA passada
    maps = [flicker] + maps[1:]
    _, shown = _run(InkConfig(hysteresis=True), maps)
    assert not shown[0][blob].any()  # poeira nunca aparece
    assert shown[0][stroke].all()  # o traco novo aparece na 1a passada
    _, shown_plain = _run(InkConfig(), maps)
    assert shown_plain[0][blob].any()  # sem a flag, aparecia


def test_retraction_count_does_not_include_filtered_dust():
    maps, _, blob = _frames(0.0)
    maps[0] = maps[0].copy()
    maps[0][blob] = 0.95
    on, _ = _run(InkConfig(hysteresis=True), maps)
    off, _ = _run(InkConfig(), maps)
    assert on["retracted_px_total"] == 0
    assert off["retracted_px_total"] > 0  # sem a flag, a poeira era exibida e retratada


def test_provisional_under_the_hand_stays_frozen_with_hysteresis():
    maps, stroke, _ = _frames(0.0, n=2)
    acc = _acc(InkConfig(hysteresis=True))
    acc.add(maps[0].copy(), np.ones((H, W), bool))
    hand = np.ones((H, W), bool)
    hand[stroke] = False  # a mao cobre o traco
    out = acc.add(np.zeros((H, W), np.float32), hand)
    assert (out["provisional"][stroke] > 0).all()  # congelado, nao retratado
    assert out["retracted_px_total"] == 0


# -------------------------------------------------------------------- config


def test_config_validates_thresholds():
    InkConfig(hysteresis=True, t_low=0.5, t_high=0.8)
    for bad in (
        dict(t_low=0.9, t_high=0.9),
        dict(t_low=0.95, t_high=0.9),
        dict(t_low=0.0),
        dict(t_high=1.1),
        dict(provisional_min_px=0),
    ):
        with pytest.raises(ValueError):
            InkConfig(**bad)


def test_default_is_off_and_strict_vote_keeps_it_off():
    assert InkConfig().hysteresis is False
    assert InkConfig.strict_vote(0.85).hysteresis is False
    assert InkConfig.strict_vote(0.85, hysteresis=True).hysteresis is True
    assert InkConfig().seen_threshold == 0.7  # nao tocado


def test_main_flag_enables_it_combines_and_markers_mode_rejects(monkeypatch):
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(sys, "argv", ["draw"])
    app_module.main()
    assert seen[-1]["pipeline_config"].ink.hysteresis is False
    monkeypatch.setattr(
        sys, "argv", ["draw", "--ink-hysteresis", "--ink-ruling", "--ink-vote", "0.85"]
    )
    app_module.main()
    ink = seen[-1]["pipeline_config"].ink
    assert ink.hysteresis and ink.ruling_removal and ink.vote_score_strong > 1.9
    monkeypatch.setattr(sys, "argv", ["draw", "--markers", "--ink-hysteresis"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2


# --------------------------------------------------------------------- régua


@pytest.fixture(scope="module")
def page():
    return bench.make_synthetic_page(seed=0)


def test_dust_sequence_is_deterministic_and_away_from_strokes(page):
    a, dust_a = bench.make_dust_passes(page, 6, seed=2)
    b, dust_b = bench.make_dust_passes(page, 6, seed=2)
    np.testing.assert_array_equal(a[3], b[3])
    np.testing.assert_array_equal(dust_a, dust_b)
    assert dust_a.sum() > 100
    assert not (dust_a & (page.strokes | page.ruling)).any()
    assert len(a) == 6


def test_ruler_shows_hysteresis_cutting_dust_and_keeping_the_stroke(page):
    off = bench.score_accumulator(page, InkConfig())
    on = bench.score_accumulator(page, InkConfig(hysteresis=True))
    assert off["dust_provisional_median_px"] > 50  # a tela mostrava poeira
    assert on["dust_provisional_median_px"] == 0
    assert on["dust_confirmed_px"] < 0.6 * off["dust_confirmed_px"]
    assert on["stroke_recall"] >= 0.99 * off["stroke_recall"]
    assert on["stroke_first_pass_visible"] >= 0.99 * off["stroke_first_pass_visible"]


def test_cli_prints_the_temporal_table(capsys):
    assert bench.main(["--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "sintético temporal" in out and "voto0.85+histerese" in out


# ------------------------------------------------- revisao do PR #57 (Devin)


def test_two_pass_speck_at_startup_does_not_confirm():
    speck = _mask((50, 52, 60, 62))
    acc = _acc(InkConfig(hysteresis=True))
    out = None
    for v in (0.97, 0.97, 0.0):
        m = np.zeros((H, W), np.float32)
        m[speck] = v
        out = acc.add(m, np.ones((H, W), bool))
    assert not (out["confirmed"][speck] > 0).any()


def test_persistent_strong_dot_still_confirms_at_startup():
    dot = _mask((50, 53, 60, 63))
    acc = _acc(InkConfig(hysteresis=True))
    out = None
    for _ in range(3):
        m = np.zeros((H, W), np.float32)
        m[dot] = 0.99
        out = acc.add(m, np.ones((H, W), bool))
    assert (out["confirmed"][dot] > 0).all()


def test_new_anchor_on_visible_side_does_not_reveal_the_covered_side():
    stroke = _mask((10, 14, 5, 29))  # 24 px de largura
    left, right = _mask((10, 14, 5, 17)), _mask((10, 14, 17, 29))
    acc = _acc(InkConfig(hysteresis=True, provisional_min_px=1))
    m = np.zeros((H, W), np.float32)
    m[stroke] = 0.85  # sem ancora: filtrado
    out = acc.add(m, np.ones((H, W), bool))
    assert not (out["provisional"] > 0).any()
    hand = np.ones((H, W), bool)
    hand[left] = False
    m2 = np.zeros((H, W), np.float32)
    m2[right] = 0.95
    out = acc.add(m2, hand)
    assert (out["provisional"][right] > 0).all()
    assert not (out["provisional"][left] > 0).any()
