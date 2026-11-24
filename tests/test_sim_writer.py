"""Testes do parser de flags do lab/sim_writer (geracao/auditoria
precisam de corpus e modelos externos ao repo)."""

import pytest


def test_parse_flags_defaults():
    """Sem flags: a base legada da regua (ver o teste do preset producao)."""
    from lab.sim_writer import _parse_flags

    from draw.ink.config import InkConfig
    from draw.settings import PipelineConfig

    assert _parse_flags([]) == PipelineConfig(
        ink=InkConfig(), ink_auto_gain=False, detect_wake=False
    )


def test_parse_flags_pipeline_and_ink_fields():
    from lab.sim_writer import _parse_flags

    config = _parse_flags(["coverage=0", "ink.reg_deadband=1"])
    assert config.coverage is False
    assert config.ink.reg_deadband is True
    assert config.canvas_persist is True
    assert config.ink.hysteresis is False


@pytest.mark.parametrize(
    "flags",
    [
        ["DRAW_INK_VOTE_E3=1"],  # variavel de ambiente removida
        ["nao_existe=1"],
        ["ink.nao_existe=1"],
        ["ink=1"],
        ["coverage"],
        ["coverage=talvez"],
        ["=1"],
    ],
)
def test_parse_flags_rejects_unknown(flags):
    from lab.sim_writer import _parse_flags

    with pytest.raises(ValueError):
        _parse_flags(flags)


def test_audit_refuses_ink_divnorm_with_frozen_model(tmp_path):
    """Review #48: o sim_writer audita o INK_MODEL congelado (v2);
    divnorm nele mediria o v2 fora do domínio — recusa antes de rodar."""
    from lab.sim_writer import audit_one

    with pytest.raises(ValueError, match="v4"):
        audit_one(tmp_path, tmp_path / "out", ["ink_divnorm=1"])


def test_parse_flags_validates_final_hysteresis_thresholds():
    from lab.sim_writer import _parse_flags

    with pytest.raises(ValueError):
        _parse_flags(["ink.hysteresis=1", "ink.t_high=0.60"])
    for order in (
        ["ink.t_low=0.5", "ink.t_high=0.8"],
        ["ink.t_high=0.8", "ink.t_low=0.5"],
    ):
        cfg = _parse_flags(order)
        assert (cfg.ink.t_low, cfg.ink.t_high) == (0.5, 0.8)


def test_parse_flags_base_is_legacy_and_producao_preset_matches_the_default():
    """A regua so compara se a base nao muda com o padrao de producao: sem
    flags e o legado; o preset `producao` do bench reproduz o padrao real."""
    from lab.bench_ink import preset_flags
    from lab.sim_writer import _parse_flags

    from draw.ink.config import InkConfig
    from draw.settings import PipelineConfig

    base = _parse_flags([])
    assert base == PipelineConfig(
        ink=InkConfig(), ink_auto_gain=False, detect_wake=False
    )
    assert _parse_flags(preset_flags("producao")) == PipelineConfig()


def test_sim_writer_rejects_none_for_non_nullable_flags():
    from lab import sim_writer

    with pytest.raises(ValueError, match="booleano|invalido"):
        sim_writer._parse_flags(["coverage=none"])


def test_audit_timeout_imprime_rabos_e_sai(monkeypatch, tmp_path, capsys):
    """Cenário travado: o audit mostra os rabos da saída parcial do
    audit-one e sai nomeando o cenário, em vez de pendurar a bateria
    inteira atrás dele."""
    import subprocess

    from lab import sim_writer

    captured = {}

    def fake_run(cmd, **kw):
        captured.update(kw)
        raise subprocess.TimeoutExpired(
            cmd, kw["timeout"], output="stdout parcial", stderr="stderr parcial"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="tempo limite.*cursive"):
        sim_writer.audit("corpus", tmp_path, [], ["cursive"])
    assert captured["timeout"] == 900
    streams = capsys.readouterr()
    assert "stdout parcial" in streams.out
    assert "stderr parcial" in streams.err
