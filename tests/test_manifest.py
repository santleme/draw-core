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
import http.server
import json
import re
import threading
import urllib.request

import pytest

from draw.infra import manifest, safe_io


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


class _FakeHttpResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        if self._offset >= len(self._payload):
            return b""
        if size is None or size < 0:
            chunk = self._payload[self._offset :]
            self._offset = len(self._payload)
            return chunk
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def read1(self, size=-1):
        return self.read(size)


class _FakeClock:
    """Relógio determinístico: o teste avança o tempo a cada read."""

    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now


class _DrippingResponse:
    """Resposta que entrega 1 byte por read e avança o relógio em `step`.

    Simula o servidor que goteja devagar demais para estourar o timeout
    de cada operação de socket — só o deadline da transferência o pega.
    """

    def __init__(self, clock, step=30.0):
        self._clock = clock
        self._step = step
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        self.reads += 1
        self._clock.now += self._step
        return b"x"

    def read1(self, size=-1):
        return self.read(size)


def test_download_model_verifies_sha256_and_size_and_writes_atomically(tmp_path):
    payload = b"pesos-oficiais-v3"
    entry = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    seen_urls = []

    def fake_urlopen(request, timeout=None):
        seen_urls.append((request.full_url, timeout))
        return _FakeHttpResponse(payload)

    target = manifest.download_model(
        "page_clean_unet.onnx",
        entry,
        models_dir=tmp_path,
        release_url="https://example.com/releases/download/models-v3",
        urlopen=fake_urlopen,
    )
    assert target == (tmp_path / "page_clean_unet.onnx").resolve()
    assert target.read_bytes() == payload
    assert seen_urls == [
        (
            "https://example.com/releases/download/models-v3/page_clean_unet.onnx",
            manifest._DOWNLOAD_TIMEOUT_S,
        )
    ]


def test_download_model_rejects_hash_mismatch_and_leaves_no_file(tmp_path):
    payload = b"pesos-corrompidos"
    entry = {
        "sha256": "0" * 64,
        "bytes": len(payload),
    }
    with pytest.raises(RuntimeError, match="sha256 divergente"):
        manifest.download_model(
            "page_clean_unet.onnx",
            entry,
            models_dir=tmp_path,
            urlopen=lambda req, timeout=None: _FakeHttpResponse(payload),
        )
    assert list(tmp_path.iterdir()) == []


def test_download_model_rejects_oversized_stream_and_cleans_up(tmp_path):
    payload = b"x" * 32
    entry = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": 10,
    }
    with pytest.raises(RuntimeError, match="excedeu 10 bytes"):
        manifest.download_model(
            "page_clean_unet.onnx",
            entry,
            models_dir=tmp_path,
            urlopen=lambda req, timeout=None: _FakeHttpResponse(payload),
        )
    assert list(tmp_path.iterdir()) == []


def test_download_model_deadline_zero_refuses_before_first_read(tmp_path):
    """timeout=0 estoura o deadline antes da primeira leitura do socket."""
    payload = b"pesos"
    entry = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    response = _FakeHttpResponse(payload)
    reads = []
    original_read = response.read

    def counting_read(size=-1):
        reads.append(size)
        return original_read(size)

    response.read = counting_read
    with pytest.raises(RuntimeError, match="deadline"):
        manifest.download_model(
            "page_clean_unet.onnx",
            entry,
            models_dir=tmp_path,
            urlopen=lambda req, timeout=None: response,
            timeout=0,
        )
    assert reads == []
    assert list(tmp_path.iterdir()) == []


def test_download_model_deadline_stops_dripping_server(tmp_path, monkeypatch):
    """Servidor que goteja 1 byte por read sem estourar o timeout de cada
    operação de socket é cortado pelo deadline da transferência."""
    clock = _FakeClock()
    monkeypatch.setattr("draw.infra.manifest.time.monotonic", clock.monotonic)
    response = _DrippingResponse(clock, step=30.0)
    entry = {"sha256": "0" * 64, "bytes": 100}
    with pytest.raises(RuntimeError, match="deadline"):
        manifest.download_model(
            "page_clean_unet.onnx",
            entry,
            models_dir=tmp_path,
            urlopen=lambda req, timeout=None: response,
            timeout=60,
        )
    # t=0 e t=30 ainda cabem no deadline de 60s; em t=60 ele estoura.
    assert response.reads == 2
    assert list(tmp_path.iterdir()) == []


def test_download_model_deadline_aborts_drip_on_real_socket(tmp_path):
    """Servidor real gotejando 1 byte por vez: com socket de verdade,
    read() ficaria preso num único recv atrás do drip. Só read1 + o
    timeout de socket encolhido fazem o deadline valer de fato."""
    stop = threading.Event()

    class _DripHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(1 << 30))
            self.end_headers()
            while not stop.is_set():
                try:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                except OSError:
                    return
                stop.wait(0.05)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _DripHandler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    port = server.server_address[1]
    entry = {"sha256": "0" * 64, "bytes": 1 << 20}
    outcome = {}
    finished = threading.Event()

    def run():
        try:
            manifest.download_model(
                "page_clean_unet.onnx",
                entry,
                models_dir=tmp_path,
                urlopen=lambda req, timeout=None: urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/page_clean_unet.onnx",
                    timeout=timeout,
                ),
                timeout=0.5,
            )
        except Exception as error:  # o assert abaixo decide o tipo esperado
            outcome["error"] = error
        finally:
            finished.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        # Sem o fix o download não terminaria nunca: o próprio wait é o
        # limite de tempo do teste, não um assert sobre sleep. O erro é
        # RuntimeError tanto quando o check do deadline dispara quanto
        # quando o timeout de socket encolhido vence uma iteração antes
        # (perto do fim, `remaining` fica menor que o intervalo do drip).
        assert finished.wait(15)
        assert isinstance(outcome.get("error"), RuntimeError)
        assert list(tmp_path.iterdir()) == []
    finally:
        stop.set()
        server.shutdown()
        serving.join(timeout=5)
        server.server_close()
        worker.join(timeout=5)


def test_download_model_rejects_path_traversal_and_non_https(tmp_path):
    entry = {"sha256": "0" * 64, "bytes": 4}
    with pytest.raises(RuntimeError, match="inválido"):
        manifest.download_model("../escape.onnx", entry, models_dir=tmp_path)
    with pytest.raises(RuntimeError, match="https://"):
        manifest.download_model(
            "m.onnx",
            entry,
            models_dir=tmp_path,
            release_url="http://example.com/models",
        )


def test_ensure_models_downloads_only_missing_files(tmp_path):
    existing = _model(tmp_path / "models", "ok.onnx", b"ja-existe")
    missing_payload = b"novo-modelo"
    manifest_path = _write_manifest(
        tmp_path,
        {
            "ok.onnx": _entry(existing),
            "novo.onnx": {
                "sha256": hashlib.sha256(missing_payload).hexdigest(),
                "bytes": len(missing_payload),
            },
        },
    )
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _FakeHttpResponse(missing_payload)

    downloaded = manifest.ensure_models(
        models_dir=tmp_path / "models",
        manifest_path=manifest_path,
        urlopen=fake_urlopen,
    )
    assert [p.name for p in downloaded] == ["novo.onnx"]
    assert len(calls) == 1
    assert (tmp_path / "models" / "novo.onnx").read_bytes() == missing_payload
    assert existing.read_bytes() == b"ja-existe"


def test_ensure_models_forwards_download_deadline(tmp_path):
    """O deadline de ensure_models vale para cada download individual."""
    missing_payload = b"novo-modelo"
    manifest_path = _write_manifest(
        tmp_path,
        {
            "novo.onnx": {
                "sha256": hashlib.sha256(missing_payload).hexdigest(),
                "bytes": len(missing_payload),
            },
        },
    )
    with pytest.raises(RuntimeError, match="deadline"):
        manifest.ensure_models(
            models_dir=tmp_path / "models",
            manifest_path=manifest_path,
            urlopen=lambda req, timeout=None: _FakeHttpResponse(missing_payload),
            timeout=0,
        )
    assert not (tmp_path / "models" / "novo.onnx").exists()
    # atomic_publish criou a pasta, mas o temporário foi removido no erro.
    assert list((tmp_path / "models").iterdir()) == []


def test_ensure_models_accepts_existing_external_symlink_without_rejecting(tmp_path):
    cache_dir = tmp_path / "external_cache"
    external_sheet = _model(cache_dir, "draw_seg_v3_dyn.onnx", b"sheet-weights")
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "draw_seg_v3_dyn.onnx").symlink_to(external_sheet)

    ink_payload = b"ink-weights"
    manifest_path = _write_manifest(
        tmp_path,
        {
            "draw_seg_v3_dyn.onnx": _entry(external_sheet),
            "page_clean_unet.onnx": {
                "sha256": hashlib.sha256(ink_payload).hexdigest(),
                "bytes": len(ink_payload),
            },
        },
    )

    downloaded = manifest.ensure_models(
        models_dir=models_dir,
        manifest_path=manifest_path,
        urlopen=lambda req, timeout=None: _FakeHttpResponse(ink_payload),
    )
    assert [p.name for p in downloaded] == ["page_clean_unet.onnx"]
    assert (models_dir / "page_clean_unet.onnx").read_bytes() == ink_payload


def test_ensure_models_does_not_overwrite_existing_resolved_target_of_broken_symlink(
    tmp_path,
):
    models_dir = tmp_path / "models"
    existing_weights = _model(
        models_dir, "checkpoints/ink.onnx", b"pesos-locais-intactos"
    )
    # Symlink atravessa pasta intermediaria inexistente ("scratch/.."),
    # entao lexical_target.exists() e False no SO, mas Path.resolve()
    # colapsa "scratch/.." para models/checkpoints/ink.onnx que ja existe.
    (models_dir / "page_clean_unet.onnx").symlink_to("scratch/../checkpoints/ink.onnx")

    release_payload = b"pesos-da-release"
    manifest_path = _write_manifest(
        tmp_path,
        {
            "page_clean_unet.onnx": {
                "sha256": hashlib.sha256(release_payload).hexdigest(),
                "bytes": len(release_payload),
            },
        },
    )
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _FakeHttpResponse(release_payload)

    downloaded = manifest.ensure_models(
        models_dir=models_dir,
        manifest_path=manifest_path,
        urlopen=fake_urlopen,
    )
    assert downloaded == []
    assert calls == []
    assert existing_weights.read_bytes() == b"pesos-locais-intactos"


def test_download_model_translates_mkstemp_and_replace_oserror_to_runtime_error(
    tmp_path, monkeypatch
):
    payload = b"pesos"
    entry = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }

    def boom_mkstemp(*a, **kw):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr(safe_io.tempfile, "mkstemp", boom_mkstemp)
    with pytest.raises(RuntimeError, match="read-only filesystem") as exc_info:
        manifest.download_model(
            "hand_landmarker.task",
            entry,
            models_dir=tmp_path,
            urlopen=lambda req, timeout=None: _FakeHttpResponse(payload),
        )
    assert isinstance(exc_info.value.__cause__, PermissionError)

    monkeypatch.undo()

    def boom_replace(src, dst):
        raise PermissionError("cannot replace target")

    monkeypatch.setattr(safe_io.os, "replace", boom_replace)
    with pytest.raises(RuntimeError, match="cannot replace target") as exc_info2:
        manifest.download_model(
            "hand_landmarker.task",
            entry,
            models_dir=tmp_path,
            urlopen=lambda req, timeout=None: _FakeHttpResponse(payload),
        )
    assert isinstance(exc_info2.value.__cause__, PermissionError)
    assert list(tmp_path.iterdir()) == []
