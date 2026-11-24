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
    import pytest
    from lab.sim_writer import audit_one

    with pytest.raises(ValueError, match="v4"):
        audit_one(tmp_path, tmp_path / "out", ["ink_divnorm=1"])


def test_audit_appends_allow_unverified_to_audit_one_cmd(tmp_path, monkeypatch):
    """O --allow-unverified do `audit` atravessa o subprocesso do audit-one;
    sem a flag o cmd sai limpo."""
    import json
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace

    import lab.sim_writer as sw

    cmds = []

    def fake_run(cmd, **_kw):
        cmds.append(list(cmd))
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps({"scenario": "x"}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    # O patch vai no objeto-módulo subprocess, não em sw.subprocess —
    # assim o teste sobrevive a sim_writer passar a chamar run_captured
    # ou a remover o import próprio (mesmo resolvedor de nome).
    monkeypatch.setattr(subprocess, "run", fake_run)
    argv = [
        "audit",
        "--corpus",
        str(tmp_path / "corpus"),
        "--out",
        str(tmp_path / "out"),
        "--scenarios",
        "cursive",
    ]
    assert sw.main(argv) == 0
    assert sw.main([*argv, "--allow-unverified"]) == 0
    assert len(cmds) == 2
    assert "--allow-unverified" not in cmds[0]
    assert "--allow-unverified" in cmds[1]


def test_audit_one_resolves_manifest_verify_and_none_opt_out(tmp_path, monkeypatch):
    """audit_one converte "manifest" no callable verify_model e aceita o
    opt-out None — o detector é construído com o verify resolvido."""
    import lab.sim_writer as sw

    from draw.infra.manifest import verify_model

    # audit_one lê o sidecar antes de construir os modelos; "{}" basta
    # porque a sonda aborta no construtor do detector.
    (tmp_path / "sidecar.json").write_text("{}")

    seen = []

    class _Bail(Exception):
        pass

    def _probe(path, *, verify=None):
        seen.append(verify)
        raise _Bail

    monkeypatch.setattr("draw.perception.NeuralSheetDetector", _probe)

    for verify in ("manifest", None):
        with pytest.raises(_Bail):
            sw.audit_one(tmp_path, tmp_path / "out", [], verify=verify)

    assert seen == [verify_model, None]


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
