"""Integridade do hvcam-host.exe provisionado no cache do Windows.

Cobre a politica de dois niveis de `_provision.ensure_hvcam_host`
(plano_execucao Parte 24): com sha256 registrado — kwarg
`expected_exe_sha256`, constante `HVCAM_HOST_EXE_SHA256` ou sidecar
`hvcam-host.exe.sha256` ao lado do exe — o cache e os candidatos locais
so sao aceitos se o sha conferir; sem registro, mantem a confianca
legada para nao quebrar builds locais. O sidecar so e gravado quando o
exe gravado foi verificado — nunca para copia local sem verificacao.
"""

import hashlib
import io
import zipfile

import pytest

from draw import hvcam as hvcam_module

_EXE_BYTES = b"MZ-fake-hvcam-exe"


def _zip_with_exe(exe_bytes=_EXE_BYTES):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(hvcam_module.HVCAM_HOST_ZIP_MEMBER, exe_bytes)
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _fake_urlopen(calls, zip_bytes):
    def fake_urlopen(req, timeout=None):
        calls.append((req.full_url, timeout))
        return _Resp(zip_bytes)

    return fake_urlopen


def _no_download(*_a, **_k):
    pytest.fail("downloader chamado quando o provisionamento nao devia baixar")


def test_verified_download_writes_sidecar_and_cache_hit_verifies_it(tmp_path):
    """Download verificado grava exe + sidecar; o sidecar ancora o cache-hit."""
    zip_bytes = _zip_with_exe()
    zip_sha = hashlib.sha256(zip_bytes).hexdigest()
    exe_sha = hashlib.sha256(_EXE_BYTES).hexdigest()
    calls = []

    kwargs = dict(
        cache_dir=tmp_path,
        urlopen=_fake_urlopen(calls, zip_bytes),
        expected_sha256=zip_sha,
        local_candidates=(),
    )
    exe = hvcam_module.ensure_hvcam_host(**kwargs)
    assert exe == tmp_path / "hvcam-host.exe"
    assert exe.read_bytes() == _EXE_BYTES
    assert len(calls) == 1

    sidecar = tmp_path / "hvcam-host.exe.sha256"
    assert sidecar.read_text(encoding="ascii").strip() == exe_sha

    # Cache-hit sem kwargs de sha: e o sidecar quem registra e verifica.
    exe2 = hvcam_module.ensure_hvcam_host(
        cache_dir=tmp_path,
        urlopen=_no_download,
        local_candidates=(),
    )
    assert exe2 == exe
    assert len(calls) == 1


def test_tampered_cache_reprovisions_via_sidecar(tmp_path):
    """Exe em cache adulterado diverge do sidecar -> apaga e baixa de novo."""
    zip_bytes = _zip_with_exe()
    zip_sha = hashlib.sha256(zip_bytes).hexdigest()
    calls = []
    kwargs = dict(
        cache_dir=tmp_path,
        urlopen=_fake_urlopen(calls, zip_bytes),
        expected_sha256=zip_sha,
        local_candidates=(),
    )
    exe = hvcam_module.ensure_hvcam_host(**kwargs)
    assert len(calls) == 1

    exe.write_bytes(b"MZ-tampered")
    exe2 = hvcam_module.ensure_hvcam_host(**kwargs)
    assert exe2 == exe
    assert exe2.read_bytes() == _EXE_BYTES
    assert len(calls) == 2


def test_registered_cache_mismatch_reprovisions(tmp_path, capsys):
    """Com sha registrado via kwarg, cache divergente cai no download."""
    zip_bytes = _zip_with_exe()
    exe_sha = hashlib.sha256(_EXE_BYTES).hexdigest()
    target = tmp_path / "hvcam-host.exe"
    target.write_bytes(b"MZ-planted")
    calls = []

    exe = hvcam_module.ensure_hvcam_host(
        cache_dir=tmp_path,
        urlopen=_fake_urlopen(calls, zip_bytes),
        expected_sha256=hashlib.sha256(zip_bytes).hexdigest(),
        expected_exe_sha256=exe_sha,
        local_candidates=(),
    )
    assert exe == target
    assert exe.read_bytes() == _EXE_BYTES
    assert len(calls) == 1
    assert (tmp_path / "hvcam-host.exe.sha256").read_text().strip() == exe_sha
    assert "AVISO" in capsys.readouterr().out


def test_local_candidate_matching_hash_is_copied_with_sidecar(tmp_path):
    """Candidato local so e copiado quando o sha confere — e gera sidecar."""
    exe_sha = hashlib.sha256(_EXE_BYTES).hexdigest()
    cand_dir = tmp_path / "candidates"
    cand_dir.mkdir()
    cand = cand_dir / "hvcam-host.exe"
    cand.write_bytes(_EXE_BYTES)
    cache = tmp_path / "cache"

    exe = hvcam_module.ensure_hvcam_host(
        cache_dir=cache,
        urlopen=_no_download,
        expected_exe_sha256=exe_sha,
        local_candidates=(cand,),
    )
    assert exe == cache / "hvcam-host.exe"
    assert exe.read_bytes() == _EXE_BYTES
    assert (cache / "hvcam-host.exe.sha256").read_text().strip() == exe_sha


def test_local_candidate_mismatch_is_skipped_and_downloads(tmp_path):
    """Candidato divergente do sha registrado e ignorado; cai no download."""
    exe_sha = hashlib.sha256(_EXE_BYTES).hexdigest()
    cand = tmp_path / "hvcam-host.exe"
    cand.write_bytes(b"MZ-other-build")
    cache = tmp_path / "cache"
    zip_bytes = _zip_with_exe()
    calls = []

    exe = hvcam_module.ensure_hvcam_host(
        cache_dir=cache,
        urlopen=_fake_urlopen(calls, zip_bytes),
        expected_sha256=hashlib.sha256(zip_bytes).hexdigest(),
        expected_exe_sha256=exe_sha,
        local_candidates=(cand,),
    )
    assert exe == cache / "hvcam-host.exe"
    assert exe.read_bytes() == _EXE_BYTES
    assert len(calls) == 1


def test_module_constant_registers_hash(monkeypatch, tmp_path):
    """HVCAM_HOST_EXE_SHA256 registrado vale como sha esperado do exe."""
    prov = hvcam_module._provision
    exe_sha = hashlib.sha256(_EXE_BYTES).hexdigest()
    monkeypatch.setattr(prov, "HVCAM_HOST_EXE_SHA256", exe_sha)
    cand = tmp_path / "hvcam-host.exe"
    cand.write_bytes(_EXE_BYTES)
    cache = tmp_path / "cache"

    exe = hvcam_module.ensure_hvcam_host(
        cache_dir=cache,
        urlopen=_no_download,
        local_candidates=(cand,),
    )
    assert exe.read_bytes() == _EXE_BYTES
    assert (cache / "hvcam-host.exe.sha256").read_text().strip() == exe_sha


def test_unregistered_cache_keeps_legacy_trust(monkeypatch, tmp_path):
    """Sem sha registrado o cache nao-vazio e aceito sem verificacao."""
    prov = hvcam_module._provision
    monkeypatch.setattr(prov, "HVCAM_HOST_EXE_SHA256", None)
    target = tmp_path / "hvcam-host.exe"
    target.write_bytes(b"MZ-garbage")

    exe = hvcam_module.ensure_hvcam_host(
        cache_dir=tmp_path,
        urlopen=_no_download,
        local_candidates=(),
    )
    assert exe == target
    assert exe.read_bytes() == b"MZ-garbage"
    assert not (tmp_path / "hvcam-host.exe.sha256").exists()


def test_unverified_candidate_copied_without_sidecar(tmp_path):
    """Sem sha registrado a copia local segue o legado — e NAO grava sidecar.

    Gravar sidecar para um binario nao verificado cimentaria um exe
    possivelmente plantado no PATH/checkout como se fosse confiavel.
    """
    cand = tmp_path / "hvcam-host.exe"
    cand.write_bytes(b"MZ-local-build")
    cache = tmp_path / "cache"

    exe = hvcam_module.ensure_hvcam_host(
        cache_dir=cache,
        urlopen=_no_download,
        local_candidates=(cand,),
    )
    assert exe == cache / "hvcam-host.exe"
    assert exe.read_bytes() == b"MZ-local-build"
    assert not (cache / "hvcam-host.exe.sha256").exists()


def test_malformed_explicit_hash_is_error_not_unregistered(tmp_path):
    """Kwarg fora do formato falha fechado — nao degrada para confianca legada."""
    target = tmp_path / "hvcam-host.exe"
    target.write_bytes(b"MZ-garbage")

    with pytest.raises(RuntimeError, match="fora do formato"):
        hvcam_module.ensure_hvcam_host(
            cache_dir=tmp_path,
            urlopen=_no_download,
            expected_exe_sha256="not-a-sha",
            local_candidates=(),
        )


def test_garbage_sidecar_counts_as_unregistered(tmp_path):
    """Sidecar fora do formato [0-9a-f]{64} nao registra sha algum."""
    target = tmp_path / "hvcam-host.exe"
    target.write_bytes(b"MZ-garbage")
    (tmp_path / "hvcam-host.exe.sha256").write_text("not-a-sha\n")

    exe = hvcam_module.ensure_hvcam_host(
        cache_dir=tmp_path,
        urlopen=_no_download,
        local_candidates=(),
    )
    assert exe == target
    assert exe.read_bytes() == b"MZ-garbage"
