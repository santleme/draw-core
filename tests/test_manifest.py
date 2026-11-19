"""Contrato do manifesto de modelos (draw/infra/manifest.py).

O manifesto é o mecanismo do R1 da auditoria de 2026-10-31: o
binário de produção não mora no repositório, então a carga confere
sha256 e tamanho contra um registro versionado. Estes testes fixam
o comportamento nos casos que importam. O padrão é estrito:
divergência e arquivo não registrado (ou manifesto ausente) levantam
RuntimeError. Só o opt-out explícito DRAW_MODEL_STRICT=0 (ou
--allow-unverified no app) rebaixa os dois para aviso.
"""

import hashlib
import json
import re

import pytest

from draw.infra import manifest


@pytest.fixture
def default_policy(monkeypatch):
    """Sem DRAW_MODEL_STRICT: o padrão de produção (estrito)."""
    monkeypatch.delenv("DRAW_MODEL_STRICT", raising=False)


def _model(tmp_path, name="m.onnx", payload=b"pesos"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_manifest(tmp_path, models):
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"schema": 1, "models": models}))
    return path


def _entry(path, **extra):
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        **extra,
    }


def test_without_manifest_load_is_refused_by_default(tmp_path, default_policy):
    model = _model(tmp_path)
    missing = tmp_path / "nao-existe.json"
    with pytest.raises(RuntimeError, match="NAO REGISTRADO"):
        manifest.verify_model(model, missing)


def test_without_manifest_opt_out_only_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DRAW_MODEL_STRICT", "0")
    model = _model(tmp_path)
    assert manifest.verify_model(model, tmp_path / "nao-existe.json") is None
    assert "NAO REGISTRADO" in capsys.readouterr().out


def test_unregistered_model_is_refused_by_default(tmp_path, default_policy):
    registered = _model(tmp_path, "ok.onnx")
    stranger = _model(tmp_path, "intruso.onnx", b"outros pesos")
    path = _write_manifest(tmp_path, {"ok.onnx": _entry(registered)})
    with pytest.raises(RuntimeError, match="NAO REGISTRADO"):
        manifest.verify_model(stranger, path)


def test_missing_file_is_left_to_the_caller(tmp_path, default_policy):
    path = _write_manifest(tmp_path, {})
    assert manifest.verify_model(tmp_path / "nao.onnx", path) is None


@pytest.mark.parametrize("value", ["", "1", "true", "on", "qualquer"])
def test_strict_unless_explicit_opt_out(monkeypatch, value):
    monkeypatch.setenv("DRAW_MODEL_STRICT", value)
    assert manifest.strict_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_explicit_opt_out_values(monkeypatch, value):
    monkeypatch.setenv("DRAW_MODEL_STRICT", value)
    assert manifest.strict_enabled() is False


def test_allow_unverified_sets_opt_out(default_policy):
    manifest.allow_unverified()
    assert manifest.strict_enabled() is False


def test_strict_argument_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAW_MODEL_STRICT", "0")
    model = _model(tmp_path)
    with pytest.raises(RuntimeError):
        manifest.verify_model(model, tmp_path / "nao.json", strict=True)


def test_unregistered_model_files_lists_extras(tmp_path):
    registered = _model(tmp_path, "ok.onnx")
    _model(tmp_path, "extra.onnx", b"x")
    _model(tmp_path, "extra.task", b"y")
    _model(tmp_path, "leia.txt", b"z")
    path = _write_manifest(tmp_path, {"ok.onnx": _entry(registered)})
    names = [p.name for p in manifest.unregistered_model_files(tmp_path, path)]
    assert names == ["extra.onnx", "extra.task"]


@pytest.mark.parametrize(
    "entry",
    [
        "texto",
        {"bytes": 1},
        {"sha256": "abc"},
        {"sha256": "g" * 64},
        {"sha256": "0" * 64, "bytes": -1},
        {"sha256": "0" * 64, "bytes": "10"},
        {"sha256": "0" * 64, "bytes": True},
    ],
)
def test_malformed_entries_are_refused(tmp_path, entry):
    path = _write_manifest(tmp_path, {"m.onnx": entry})
    with pytest.raises(RuntimeError, match="entrada"):
        manifest.load_manifest(path)


def test_oversized_manifest_is_refused(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text('{"schema": 1, "models": {}, "pad": "' + "x" * (2 << 20) + '"}')
    with pytest.raises(RuntimeError, match="ilegível"):
        manifest.load_manifest(path)


@pytest.mark.parametrize("cut", [1, 10, 40])
def test_truncated_manifest_is_refused(tmp_path, cut):
    model = _model(tmp_path)
    full = json.dumps({"schema": 1, "models": {"m.onnx": _entry(model)}})
    path = tmp_path / "MANIFEST.json"
    path.write_text(full[:-cut])
    with pytest.raises(RuntimeError, match="ilegível"):
        manifest.load_manifest(path)


def test_matching_entry_passes(tmp_path):
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": _entry(model)})
    assert manifest.verify_model(model, path) is not None


def test_entry_by_relative_path_also_matches(tmp_path):
    model = _model(tmp_path, "sub/m.onnx")
    path = _write_manifest(tmp_path, {"sub/m.onnx": _entry(model)})
    assert manifest.verify_model(model, path) is not None


def test_divergence_is_fatal_by_default(tmp_path, default_policy):
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": {"sha256": "0" * 64}})
    with pytest.raises(RuntimeError, match="DIVERGENTE"):
        manifest.verify_model(model, path)


def test_divergence_only_warns_with_explicit_opt_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DRAW_MODEL_STRICT", "0")
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": {"sha256": "0" * 64, "bytes": 1}})
    assert manifest.verify_model(model, path) is not None
    printed = capsys.readouterr().out
    assert "DIVERGENTE" in printed


def test_strict_mode_turns_divergence_into_an_error(tmp_path, monkeypatch):
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": {"sha256": "0" * 64, "bytes": 1}})
    monkeypatch.setenv("DRAW_MODEL_STRICT", "1")
    with pytest.raises(RuntimeError, match="DIVERGENTE"):
        manifest.verify_model(model, path)


def test_size_mismatch_is_reported_even_with_the_right_hash(tmp_path, monkeypatch):
    model = _model(tmp_path)
    entry = _entry(model)
    entry["bytes"] = 999
    path = _write_manifest(tmp_path, {"m.onnx": entry})
    monkeypatch.setenv("DRAW_MODEL_STRICT", "1")
    with pytest.raises(RuntimeError, match="bytes"):
        manifest.verify_model(model, path)


def test_corrupted_manifest_is_an_explicit_error(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text("{ nao e json")
    with pytest.raises(RuntimeError, match="ilegível"):
        manifest.load_manifest(path)


def test_unknown_schema_is_refused(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"schema": 99, "models": {}}))
    with pytest.raises(RuntimeError, match="schema"):
        manifest.load_manifest(path)


def test_default_manifest_path_points_at_repo_assets():
    """O default é assets/manifest.json na raiz do repositório."""
    assert manifest.DEFAULT_MANIFEST_PATH.name == "manifest.json"
    assert manifest.DEFAULT_MANIFEST_PATH.parent.name == "assets"
    assert manifest.DEFAULT_MANIFEST_PATH.exists()


def test_repository_manifest_is_valid():
    """O assets/manifest.json versionado precisa carregar sem erro."""
    path, models = manifest.load_manifest()
    if not path.exists():
        pytest.skip("repositório sem assets/manifest.json")
    assert isinstance(models, dict)


def test_repository_manifest_covers_every_default_model():
    """Todo modelo que o app abre por padrão tem entrada registrada."""
    _path, models = manifest.load_manifest()
    for name in (
        "draw_seg_v3_dyn.onnx",
        "page_clean_unet.onnx",
        "hand_landmarker.task",
    ):
        assert name in models, name


def test_repository_manifest_entries_have_valid_shape():
    """O manifesto real: 3 modelos, sha256 de 64 hex, bytes > 0."""
    path, models = manifest.load_manifest()
    if not path.exists():
        pytest.skip("repositório sem assets/manifest.json")
    assert len(models) == 3
    for name, entry in models.items():
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), name
        assert isinstance(entry["bytes"], int) and entry["bytes"] > 0, name


def test_unregistered_model_files_defaults_to_repo_models_dir(tmp_path, monkeypatch):
    """Sem models_dir, a varredura olha para models/ na raiz do repo."""
    stray = tmp_path / "solto.onnx"
    stray.write_bytes(b"x")
    monkeypatch.setattr(manifest, "DEFAULT_MODELS_DIR", tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": 1, "models": {}}))
    names = [
        p.name for p in manifest.unregistered_model_files(manifest_path=manifest_path)
    ]
    assert names == ["solto.onnx"]


def test_matching_entry_keeps_calibration(tmp_path, default_policy):
    model = _model(tmp_path)
    path = _write_manifest(
        tmp_path, {"m.onnx": _entry(model, runtime_threshold=0.79, calibration={})}
    )
    entry = manifest.verify_model(model, path)
    assert entry["runtime_threshold"] == 0.79
    assert "calibration" in entry


def test_relaxed_mismatch_drops_calibration(tmp_path, capsys):
    """Binário trocado com --allow-unverified não herda o limiar do aprovado."""
    approved = _model(tmp_path, "m.onnx", b"pesos aprovados")
    entry = _entry(approved, runtime_threshold=0.79, calibration={"iou": 0.9})
    path = _write_manifest(tmp_path, {"m.onnx": entry})
    approved.write_bytes(b"outros pesos!!!")
    got = manifest.verify_model(approved, path, strict=False)
    assert "DIVERGENTE" in capsys.readouterr().out
    assert got["sha256"] == entry["sha256"]
    for key in manifest.CALIBRATION_KEYS:
        assert key not in got
    # A entrada do manifesto em si não é mutada.
    assert manifest.load_manifest(path)[1]["m.onnx"]["runtime_threshold"] == 0.79
