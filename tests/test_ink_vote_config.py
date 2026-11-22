"""Regra de voto configuravel (InkConfig.vote_*, strict_vote)."""

import dataclasses
import sys

import numpy as np
import pytest

from draw import app as app_module
from draw.ink import accumulator as accumulator_mod
from draw.ink import votes
from draw.ink.config import (
    LEGACY_VOTE_LEVEL,
    PRODUCTION_PASS_INTERVAL_S,
    TECHNICAL_VOTE_LEVEL,
    InkConfig,
)
from draw.ink.worker import PageCleanWorker

H, W = 40, 60
BLOCK = (slice(5, 15), slice(5, 15))


def _prob(value):
    prob = np.zeros((H, W), dtype=np.float32)
    prob[BLOCK] = value
    return prob


def _confirmed_px(config, value, passes=3):
    acc = accumulator_mod.LayeredInkAccumulator(config.seen_threshold, config)
    result = None
    for _ in range(passes):
        result = acc.add(_prob(value))
    return int(np.count_nonzero(result["confirmed"][BLOCK]))


def test_defaults_match_the_historical_vote_constants():
    """Os campos novos não podem divergir de draw.ink.votes (a
    implementação de referência de test_perf_impl)."""
    cfg = InkConfig()
    assert cfg.vote_hit == votes._INK_VOTE_HIT
    assert cfg.vote_score == votes._INK_VOTE_SCORE
    assert cfg.vote_score_strong == votes._INK_VOTE_SCORE_STRONG


def test_strict_vote_at_legacy_level_is_exactly_the_default():
    """O nível legado reproduz o InkConfig padrão campo a campo — não
    "quase" (1,89 x 1,9 confirmava mais tinta, review #53)."""
    assert InkConfig.strict_vote(LEGACY_VOTE_LEVEL) == InkConfig()


def test_legacy_level_matches_the_default_canvas_on_a_two_pass_flash():
    """0,947 em 2 passadas soma 1,894: o legado não confirma, e o nível
    legado do strict_vote também não (com 3 x 0,63 = 1,89 confirmava)."""

    def run(config):
        acc = accumulator_mod.LayeredInkAccumulator(config.seen_threshold, config)
        result = None
        for value in (0.947, 0.947, 0.0, 0.0):
            result = acc.add(_prob(value))
        return int(np.count_nonzero(result["confirmed"]))

    assert run(InkConfig()) == 0
    assert run(InkConfig.strict_vote(LEGACY_VOTE_LEVEL)) == 0


@pytest.mark.parametrize("bad", [0.0, 0.5, 0.63, 0.995, 1.0, -1.0])
def test_strict_vote_refuses_levels_outside_the_measured_range(bad):
    with pytest.raises(ValueError, match="nivel de voto"):
        InkConfig.strict_vote(bad)


def test_strict_vote_shifts_the_vote_fields_and_keeps_seen_threshold():
    cfg = InkConfig.strict_vote(0.92)
    shift = 0.92 - LEGACY_VOTE_LEVEL
    assert cfg.vote_hit == pytest.approx(0.5 + shift)
    assert cfg.vote_score == pytest.approx(1.4 + 3 * shift)
    assert cfg.vote_score_strong == pytest.approx(2.76)
    # O limiar de "visto como tinta" governa identidade, remoção e
    # âncora da fronteira: o voto estrito não o toca (review #53).
    assert cfg.seen_threshold == InkConfig().seen_threshold == 0.7
    assert InkConfig.strict_vote(0.92, removal=False).removal is False


def test_strict_vote_keeps_the_page_identity_threshold():
    """A identidade da página no reencontro lê o limiar do acumulador:
    com o voto estrito ele segue no 0,7, e a folha que volta não é
    tomada por nova (review #53)."""

    class _Ink:
        threshold = 0.97

    worker = PageCleanWorker(_Ink(), config=InkConfig.strict_vote(0.92))
    assert worker._accumulator.threshold == pytest.approx(0.7)


def test_strict_vote_does_not_admit_a_halo_next_to_a_new_core():
    """Núcleo 0,82 + halo 0,60 estáveis: o halo não confirma no estrito,
    como no legado (o limiar de âncora não subiu, review #53)."""
    halo = (slice(5, 15), slice(15, 16))
    for config in (InkConfig(), InkConfig.strict_vote(0.85)):
        acc = accumulator_mod.LayeredInkAccumulator(config.seen_threshold, config)
        result = None
        for _ in range(3):
            prob = _prob(0.82)
            prob[halo] = 0.60
            result = acc.add(prob)
        assert int(np.count_nonzero(result["confirmed"][halo])) == 0


def test_default_vote_confirms_mid_confidence_ink_strict_vote_does_not():
    """0,75 x 3 passadas (média 0,75 > 0,63): o legado confirma; o voto
    estrito em 0,92 não — é a pauta de confiança média do v2."""
    assert _confirmed_px(InkConfig(), 0.75) == 100
    assert _confirmed_px(InkConfig.strict_vote(0.92), 0.75) == 0


def test_strict_vote_still_confirms_strong_ink():
    assert _confirmed_px(InkConfig.strict_vote(0.92), 0.95) == 100


def test_strict_vote_drops_the_stable_weak_pencil_rule():
    """Lápis fraco estável (0,55 x 3) confirma pela regra (b) no legado
    e não deve confirmar no regime estrito."""
    assert _confirmed_px(InkConfig(), 0.55) == 100
    assert _confirmed_px(InkConfig.strict_vote(0.92), 0.55) == 0


def test_worker_hands_the_config_to_its_accumulator():
    cfg = InkConfig.strict_vote(0.9)

    class _Ink:
        threshold = 0.97

    worker = PageCleanWorker(_Ink(), config=cfg)
    assert worker._accumulator._config is cfg
    assert worker._accumulator.threshold == pytest.approx(0.7)  # seen não sobe
    assert worker._accumulator._config.vote_score_strong == cfg.vote_score_strong


def test_main_ink_vote_flag_builds_the_pipeline_config(monkeypatch):
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(sys, "argv", ["draw"])
    app_module.main()
    assert (
        seen[-1]["pipeline_config"].ink == InkConfig.production()
    )  # sem a flag: preset de produção
    monkeypatch.setattr(sys, "argv", ["draw", "--ink-vote", "0.92"])
    app_module.main()
    assert seen[-1]["pipeline_config"].ink.vote_score_strong == pytest.approx(2.76)


def test_main_ink_vote_out_of_range_exits_2(monkeypatch):
    monkeypatch.setattr(app_module, "run", lambda **kw: None)
    monkeypatch.setattr(sys, "argv", ["draw", "--ink-vote", "0.2"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2


def test_check_validates_the_vote_level_before_dispatching(monkeypatch):
    """--check com nível inválido saía com sucesso enquanto a sessão ao
    vivo não subia (review #53)."""
    calls = []
    monkeypatch.setattr(app_module, "run_check", lambda *a, **k: calls.append(a) or 0)
    monkeypatch.setattr(sys, "argv", ["draw", "--check", "--ink-vote", "0.2"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2
    assert calls == []  # o preflight da câmera nem rodou


def test_markers_mode_rejects_ink_vote(monkeypatch):
    monkeypatch.setattr(app_module, "run", lambda **kw: None)
    monkeypatch.setattr(sys, "argv", ["draw", "--markers", "--ink-vote", "0.92"])
    with pytest.raises(SystemExit) as error:
        app_module.main()
    assert error.value.code == 2


def test_main_combines_ink_vote_gain_and_rotation(monkeypatch):
    """As três opções coexistem em main(): soltar qualquer uma na montagem
    do PipelineConfig/run passaria despercebido sem este teste (review
    #53). Voto estrito, ganho e rotação chegam juntos ao run()."""
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(
        sys,
        "argv",
        ["draw", "--ink-vote", "0.85", "--rotate", "180"],
    )
    app_module.main()
    kwargs = seen[-1]
    shift = 0.85 - LEGACY_VOTE_LEVEL
    assert kwargs["pipeline_config"].ink_auto_gain is True
    assert kwargs["pipeline_config"].ink.vote_score_strong == pytest.approx(
        1.9 + 3 * shift
    )
    assert kwargs["pipeline_config"].ink.seen_threshold == 0.7
    assert kwargs["rotation"] == 180


def test_sketch_profile_keeps_the_legacy_vote_and_drops_ruling_removal():
    """Esboço/lápis: campos de voto idênticos ao legado (fidelidade do
    grafite fino), com os mecanismos de produção ligados e a remoção de
    pauta desligada (PR #56: ela pode comer lápis sobre a linha)."""
    cfg = InkConfig.sketch()
    legacy = InkConfig()
    for field in ("vote_hit", "vote_score", "vote_score_strong", "seen_threshold"):
        assert getattr(cfg, field) == getattr(legacy, field)
    assert cfg.ruling_removal is False
    # Sem hysteresis: o portão de âncora mata traço de lápis isolado
    # (Devin Review PR #122) — ver teste de acumulador abaixo.
    assert cfg.hysteresis is False
    assert cfg.motion_veto is True
    assert cfg.pass_interval_s == PRODUCTION_PASS_INTERVAL_S


def test_sketch_confirms_an_isolated_faint_stroke_through_the_accumulator():
    """Traço de lápis isolado a 0,75 passa o voto legado (2,25 >= 1,9)
    mas morreria no portão de âncora da hysteresis (nenhum pixel com
    média >= 0,90 e nenhuma tinta confirmada por perto — Devin Review
    PR #122). sketch() tem que confirmá-lo; ligar a histerese de volta
    tem que derrubá-lo."""
    assert _confirmed_px(InkConfig.sketch(), 0.75, passes=3) > 0
    gated = dataclasses.replace(InkConfig.sketch(), hysteresis=True)
    assert _confirmed_px(gated, 0.75, passes=3) == 0


def test_technical_profile_uses_the_measured_092_vote_level():
    """Técnico/CAD: o nível medido 0,92 mantém a escrita (109%) e quase
    zera a pauta residual (48%) — vetores mais limpos para DXF."""
    cfg = InkConfig.technical()
    shift = TECHNICAL_VOTE_LEVEL - LEGACY_VOTE_LEVEL
    assert cfg.vote_score_strong == pytest.approx(1.9 + 3 * shift)
    assert cfg.ruling_removal is True
    assert cfg.hysteresis is True
    assert cfg.motion_veto is True
    assert cfg.pass_interval_s == PRODUCTION_PASS_INTERVAL_S


def test_main_preset_flag_builds_the_profile_config(monkeypatch):
    seen = []
    monkeypatch.setattr(app_module, "run", lambda **kw: seen.append(kw))
    monkeypatch.setattr(sys, "argv", ["draw", "--preset", "sketch"])
    app_module.main()
    assert seen[-1]["pipeline_config"].ink == InkConfig.sketch()
    monkeypatch.setattr(sys, "argv", ["draw", "--preset", "ruled"])
    app_module.main()
    assert seen[-1]["pipeline_config"].ink == InkConfig.production()
    monkeypatch.setattr(
        sys, "argv", ["draw", "--preset", "technical", "--ink-interval", "0.25"]
    )
    app_module.main()
    # --ink-interval compõe sobre qualquer perfil (cadência é ortogonal).
    assert seen[-1]["pipeline_config"].ink == InkConfig.technical(pass_interval_s=0.25)


def test_main_preset_rejects_conflicting_flags(monkeypatch):
    monkeypatch.setattr(app_module, "run", lambda **kw: None)
    for argv in (
        ["draw", "--preset", "sketch", "--ink-vote", "0.9"],
        ["draw", "--preset", "ruled", "--legacy-ink"],
        ["draw", "--markers", "--preset", "sketch"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as error:
            app_module.main()
        assert error.value.code == 2


def test_production_preset_keeps_paper_and_border_guards():
    # O preset nunca pode perder as duas guardas: paper_ok segura a
    # janela pre-sementeira do PageModel e a remocao sob oclusor nao
    # mascarado (contrato 9); o inset automatico (None ->
    # resolve_border_guard_px) e a unica exclusao de borda nao-papel
    # sem mascara de cobertura (contrato 4, faixa de mesa). O default
    # do campo border_guard_px e 0 (InkConfig() e a pilha legada) —
    # production() precisa do None explicito.
    cfg = InkConfig.production()
    assert cfg.paper_guard is True
    assert cfg.border_guard_px is None
