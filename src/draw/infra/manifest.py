"""Integridade dos modelos ONNX contra o manifesto do repositório.

O repositório não publica binário de modelo: o `draw_seg_v3_dyn.onnx`
de produção vive fora dele. O efeito colateral é que qualquer arquivo
com o nome certo é aceito na carga, sem nenhuma verificação — um
modelo trocado, truncado ou de outra corrida entra no app em
silêncio.

Este módulo é o mecanismo que fecha esse buraco sem obrigar o
binário a morar no git: `assets/manifest.json` registra, por nome de
arquivo, o sha256, o tamanho em bytes, a régua que aprovou o modelo
e a data da aprovação. Toda carga de modelo (detector neural,
página->limpa, máscara de mão) passa por verify_model antes de abrir
a sessão.

Política (estrita por padrão):

- entrada conferindo: a carga segue;
- hash ou tamanho divergente: RuntimeError;
- arquivo de modelo sem entrada no manifesto (ou manifesto ausente):
  RuntimeError — um modelo não registrado não é carregado em silêncio.

O rebaixamento para aviso é opt-in explícito: DRAW_MODEL_STRICT=0
(ou false/no/off), ou a flag --allow-unverified do app, que define a
mesma variável. Qualquer outro valor, inclusive vazio, é estrito.

Formato:

.. code-block:: json

    {
      "schema": 1,
      "models": {
        "draw_seg_v3_dyn.onnx": {
          "sha256": "751e74da...",
          "bytes": 15191109,
          "approved_by": "tools/eval_real_frames.py: 43/43, 0 FP, IoU 0,962",
          "date": "2026-10-29"
        }
      }
    }

O manifesto é lido com teto de tamanho e cada entrada é validada
(sha256 hexadecimal de 64 caracteres, bytes inteiro não negativo):
um manifesto corrompido nunca vira "sem verificação".
"""

import hashlib
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .safe_io import InputLimitError, read_json

_REPO_ROOT = Path(__file__).resolve().parents[3]
# O manifesto mora em assets/ na raiz do repositório; os binários
# que ele registra ficam em models/ (distribuídos por Release, nunca
# versionados).
DEFAULT_MANIFEST_PATH = _REPO_ROOT / "assets" / "manifest.json"
DEFAULT_MODELS_DIR = _REPO_ROOT / "models"
DEFAULT_RELEASE_URL = (
    "https://github.com/santleme/draw-core/releases/download/models-v3"
)
_ENV_STRICT_VAR = "DRAW_MODEL_STRICT"
_RELAXED_VALUES = ("0", "false", "no", "off")
_SCHEMA = 1
_CHUNK = 1 << 20
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_MODEL_DOWNLOAD_BYTES = 128 << 20
_DOWNLOAD_TIMEOUT_S = 60
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_SUFFIXES = (".onnx", ".task")


def sha256_of(path):
    """sha256 do arquivo, em blocos de 1 MiB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path=None):
    """Lê o manifesto. Devolve (caminho, dicionário de modelos).

    Manifesto ausente devolve (caminho, {}) — e não é erro: o
    repositório publica o esquema, não os binários. Manifesto
    ilegível (JSON inválido, schema desconhecido) é erro explícito:
    um manifesto corrompido não pode virar "sem verificação" em
    silêncio.
    """
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    if not manifest_path.exists():
        return manifest_path, {}
    try:
        data = read_json(manifest_path, max_bytes=_MAX_MANIFEST_BYTES)
    except InputLimitError as error:
        raise RuntimeError(
            f"manifesto de modelos ilegível em {manifest_path}: {error}"
        ) from error
    if not isinstance(data, dict) or "models" not in data:
        raise RuntimeError(
            f"manifesto de modelos sem a chave 'models': {manifest_path}"
        )
    schema = data.get("schema", _SCHEMA)
    if schema != _SCHEMA:
        raise RuntimeError(
            f"manifesto de modelos com schema {schema!r}; este código "
            f"entende o schema {_SCHEMA}"
        )
    models = data["models"]
    if not isinstance(models, dict):
        raise RuntimeError(
            f"'models' do manifesto precisa ser um objeto: {manifest_path}"
        )
    for name, entry in models.items():
        _validate_entry(manifest_path, name, entry)
    return manifest_path, models


def _validate_entry(manifest_path, name, entry):
    if not isinstance(entry, dict):
        raise RuntimeError(f"entrada {name!r} do manifesto não é um objeto")
    sha = str(entry.get("sha256", "")).strip().lower()
    if not sha:
        raise RuntimeError(
            f"entrada {name!r} do manifesto não tem sha256: {manifest_path}"
        )
    if not _SHA256_RE.match(sha):
        raise RuntimeError(
            f"entrada {name!r} do manifesto com sha256 malformado: {manifest_path}"
        )
    size = entry.get("bytes")
    if size is not None and (
        isinstance(size, bool) or not isinstance(size, int) or size < 0
    ):
        raise RuntimeError(
            f"entrada {name!r} do manifesto com 'bytes' inválido: {size!r}"
        )


def strict_enabled():
    """True salvo opt-out explícito (DRAW_MODEL_STRICT=0/false/no/off)."""
    value = os.environ.get(_ENV_STRICT_VAR, "").strip().lower()
    return value not in _RELAXED_VALUES


def allow_unverified():
    """Rebaixa a verificação para aviso neste processo (--allow-unverified)."""
    os.environ[_ENV_STRICT_VAR] = "0"


def _refuse_or_warn(message, strict):
    if strict:
        raise RuntimeError(
            message + f" Para carregar mesmo assim, defina {_ENV_STRICT_VAR}=0 "
            "ou use --allow-unverified."
        )
    print(f"AVISO: {message}")


def find_entry(model_path, manifest_path, models):
    """Entrada do manifesto para o arquivo (por nome ou caminho relativo)."""
    path = Path(model_path)
    entry = models.get(path.name)
    if entry is None:
        try:
            relative = (
                path.resolve()
                .relative_to(Path(manifest_path).parent.resolve())
                .as_posix()
            )
        except ValueError:
            relative = None
        if relative is not None:
            entry = models.get(relative)
    return entry


def unregistered_model_files(models_dir=None, manifest_path=None):
    """Arquivos .onnx/.task da pasta de modelos sem entrada no manifesto."""
    resolved_manifest, models = load_manifest(manifest_path)
    folder = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix in MODEL_SUFFIXES
        and find_entry(path, resolved_manifest, models) is None
    )


# Campos da entrada que descrevem o comportamento de UM binário (o do
# sha256): só valem quando o arquivo confere. Divergência relaxada
# devolve a entrada sem eles — o binário trocado não herda a
# calibração do aprovado.
CALIBRATION_KEYS = ("runtime_threshold", "calibration")


def verify_model(model_path, manifest_path=None, strict=None):
    """Confere o modelo contra o manifesto. Devolve a entrada ou None.

    Arquivo inexistente devolve None (o chamador trata a ausência).
    Entrada conferindo devolve a entrada. Divergência ou arquivo sem
    entrada levantam RuntimeError no modo estrito (o padrão); com
    DRAW_MODEL_STRICT=0 (ou strict=False) imprimem o aviso e devolvem
    a entrada sem os CALIBRATION_KEYS (divergência) ou None (não
    registrado).
    """
    if strict is None:
        strict = strict_enabled()
    path = Path(model_path)
    if not path.exists():
        return None
    resolved_manifest, models = load_manifest(manifest_path)
    # A chave é o nome do arquivo (model.onnx) ou o caminho relativo
    # à pasta do manifesto (pasta/model.onnx), para o registro não
    # depender de dois modelos homônimos.
    entry = find_entry(path, resolved_manifest, models)
    if entry is None:
        where = (
            f"no manifesto {resolved_manifest}"
            if resolved_manifest.exists()
            else f"(manifesto ausente: {resolved_manifest})"
        )
        _refuse_or_warn(
            f"MODELO NAO REGISTRADO {where}: {path}. Nenhum sha256 aprovado "
            "cobre este arquivo.",
            strict,
        )
        return None
    expected = str(entry["sha256"]).strip().lower()
    size_expected = entry.get("bytes")
    size_actual = path.stat().st_size
    problems = []
    if isinstance(size_expected, int) and size_expected != size_actual:
        problems.append(f"{size_actual} bytes != {size_expected} esperados")
    else:
        actual = sha256_of(path)
        if actual != expected:
            problems.append(f"sha256 {actual} != {expected}")
    if not problems:
        return entry
    message = (
        f"MODELO DIVERGENTE do manifesto {resolved_manifest.name}: "
        f"{path.name} — "
        + "; ".join(problems)
        + ". O arquivo em disco não é o modelo que a régua aprovou."
    )
    _refuse_or_warn(message, strict)
    return {key: value for key, value in entry.items() if key not in CALIBRATION_KEYS}


def _manifest_release_url(manifest_path):
    """Lê release_url do manifesto, caindo para DEFAULT_RELEASE_URL."""
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST_PATH
    if not path.exists():
        return DEFAULT_RELEASE_URL
    try:
        data = read_json(path, max_bytes=_MAX_MANIFEST_BYTES)
    except InputLimitError:
        return DEFAULT_RELEASE_URL
    if not isinstance(data, dict):
        return DEFAULT_RELEASE_URL
    raw = data.get("release_url")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().rstrip("/")
    return DEFAULT_RELEASE_URL


def _validate_lexical_name(name):
    """Rejeita nomes absolutos ou com '..' antes de qualquer acesso a disco."""
    rel = Path(name)
    if not str(name).strip() or rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(
            f"nome de modelo inválido no manifesto para download: {name!r}"
        )
    return rel


def _validate_download_target(models_dir, name):
    """Garante que o destino de escrita não escapa da pasta de modelos."""
    rel = _validate_lexical_name(name)
    folder = Path(models_dir).resolve()
    target = (folder / rel).resolve()
    try:
        target.relative_to(folder)
    except ValueError as error:
        raise RuntimeError(f"caminho de modelo fora de {folder}: {name!r}") from error
    return target


def download_model(
    name,
    entry,
    *,
    models_dir=None,
    release_url=None,
    urlopen=None,
    timeout=_DOWNLOAD_TIMEOUT_S,
):
    """Baixa um modelo da Release oficial, confere bytes + sha256 e grava atômico.

    Nunca publica arquivo parcial ou com hash divergente: o download vai
    para um temporário na mesma pasta e só substitui o destino se o
    tamanho e o sha256 baterem exatamente com `entry`.
    """
    folder = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    target = _validate_download_target(folder, name)
    expected_sha = str(entry.get("sha256", "")).strip().lower()
    if not _SHA256_RE.match(expected_sha):
        raise RuntimeError(f"entrada {name!r} sem sha256 válido para download")
    expected_bytes = entry.get("bytes")
    if expected_bytes is not None and (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or expected_bytes > _MAX_MODEL_DOWNLOAD_BYTES
    ):
        raise RuntimeError(
            f"entrada {name!r} com 'bytes' fora do limite de download: {expected_bytes!r}"
        )
    max_bytes = (
        expected_bytes if isinstance(expected_bytes, int) else _MAX_MODEL_DOWNLOAD_BYTES
    )
    base_url = (release_url or DEFAULT_RELEASE_URL).strip().rstrip("/")
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError(f"release_url precisa ser https://: {base_url!r}")
    asset_name = Path(name).name
    url = f"{base_url}/{urllib.parse.quote(asset_name)}"
    opener = urllib.request.urlopen if urlopen is None else urlopen
    fd = None
    tmp_path = None
    digest = hashlib.sha256()
    total = 0
    try:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".part", dir=str(target.parent)
            )
            tmp_path = Path(tmp_name)
            request = urllib.request.Request(
                url, headers={"User-Agent": "draw-core/0.1"}
            )
            with (
                opener(request, timeout=timeout) as response,
                os.fdopen(fd, "wb") as handle,
            ):
                fd = None
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError(
                            f"download de {name} excedeu {max_bytes} bytes ({url})"
                        )
                    digest.update(chunk)
                    handle.write(chunk)
        except (urllib.error.URLError, OSError) as error:
            raise RuntimeError(f"falha ao baixar {name} de {url}: {error}") from error
        if isinstance(expected_bytes, int) and total != expected_bytes:
            raise RuntimeError(
                f"download de {name} truncado: {total} bytes != {expected_bytes} esperados"
            )
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"sha256 divergente no download de {name}: {actual_sha} != {expected_sha}"
            )
        try:
            os.replace(tmp_path, target)
        except OSError as error:
            raise RuntimeError(
                f"falha ao publicar {name} em {target}: {error}"
            ) from error
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return target


def ensure_models(
    names=None,
    *,
    models_dir=None,
    manifest_path=None,
    release_url=None,
    urlopen=None,
):
    """Garante que os modelos registrados em assets/manifest.json estão em models/.

    Só baixa o que está AUSente em `models_dir`; arquivos já presentes em
    disco (inclusive symlinks válidos para um cache externo) não são
    sobrescritos (a verificação subsequente decide se conferem ou se
    falham no modo estrito). Devolve a lista de Paths baixados nesta
    chamada.
    """
    resolved_manifest, models = load_manifest(manifest_path)
    if not models:
        return []
    folder = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    wanted = sorted(models) if names is None else list(names)
    base_url = release_url or _manifest_release_url(resolved_manifest)
    downloaded = []
    for name in wanted:
        entry = models.get(name)
        if entry is None:
            continue
        rel = _validate_lexical_name(name)
        lexical_target = folder / rel
        if lexical_target.exists():
            continue
        target = _validate_download_target(folder, name)
        if target.exists():
            continue
        print(
            f"Baixando {name} ({entry.get('bytes', '?')} bytes) da Release oficial..."
        )
        download_model(
            name,
            entry,
            models_dir=folder,
            release_url=base_url,
            urlopen=urlopen,
        )
        print(f"Modelo {name} salvo e verificado (sha256 confere).")
        downloaded.append(target)
    return downloaded
