"""Manifesto versionado com sha256 dos clipes do corpus."""

import hashlib
import json
from pathlib import Path

from lab.corpus.clip import ClipError, clip_frames

MANIFEST_NAME = "corpus.json"
MANIFEST_VERSION = 1


def iter_clip_dirs(root) -> list[Path]:
    """Diretorios imediatamente abaixo de ``root`` que contem ``clip.json``."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "clip.json").is_file()),
        key=lambda p: p.name,
    )


def hash_clip(clip_dir) -> str:
    """Sha256 deterministico sobre ``clip.json``, ``regions.json`` e ``f*.png``.

    O digest segue a ordem LEXICA de nomes (o ``sort`` abaixo), nao a
    ordem numerica de ``clip_frames`` — reordenar o inventario nao muda
    o sha256 de clipes ja registrados no manifesto.
    """
    clip_dir = Path(clip_dir)
    files = []
    for name in ("clip.json", "regions.json"):
        candidate = clip_dir / name
        if candidate.is_file():
            files.append(candidate)
    files.extend(clip_frames(clip_dir))
    files.sort(key=lambda p: p.name)

    digest = hashlib.sha256()
    for path in files:
        data = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def build_manifest(root) -> dict:
    """Constroi o manifesto ``corpus.json`` a partir dos clipes em ``root``."""
    clips = {}
    for clip_dir in iter_clip_dirs(root):
        meta_path = clip_dir / "clip.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClipError(f"clip.json ilegivel em {clip_dir.name}: {exc}") from exc
        if not isinstance(meta, dict):
            raise ClipError(f"clip.json deve ser objeto JSON: {meta_path}")
        try:
            clips[clip_dir.name] = {
                "sha256": hash_clip(clip_dir),
                "frames": int(meta["frames"]),
                "conditions": dict(meta.get("conditions") or {}),
                "issues": list(meta.get("issues") or []),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ClipError(f"clip.json invalido em {clip_dir.name}: {exc}") from exc
    return {"schema": MANIFEST_VERSION, "clips": clips}


def write_manifest(root, manifest: dict) -> Path:
    """Grava ``corpus.json`` em ``root`` e devolve o caminho."""
    out_path = Path(root) / MANIFEST_NAME
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def load_manifest(root) -> dict | None:
    """Carrega ``corpus.json`` ou devolve ``None`` se ausente."""
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema") != MANIFEST_VERSION
        or not isinstance(data.get("clips"), dict)
        or not all(isinstance(entry, dict) for entry in data["clips"].values())
    ):
        raise ValueError(
            f"manifesto invalido ou schema diferente de {MANIFEST_VERSION}"
        )
    return data


def verify_manifest(root) -> list[str]:
    """Confere o diretorio ``root`` contra ``corpus.json`` (fail-closed)."""
    root = Path(root)
    try:
        manifest = load_manifest(root)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return [f"manifesto invalido em {root}: {exc}"]
    if manifest is None:
        return [f"{MANIFEST_NAME} ausente em {root}"]

    problems = []
    manifest_clips = manifest["clips"]
    disk_dirs = {d.name: d for d in iter_clip_dirs(root)}

    for name in sorted(manifest_clips):
        if name not in disk_dirs:
            problems.append(f"clipe {name!r} no manifesto mas ausente no disco")
            continue
        clip_dir = disk_dirs[name]
        entry = manifest_clips[name]
        if hash_clip(clip_dir) != entry.get("sha256"):
            problems.append(f"sha256 divergente em {name!r}")
        try:
            meta = json.loads((clip_dir / "clip.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"clip.json ilegivel em {name!r}: {exc}")
            continue
        if not isinstance(meta, dict):
            problems.append(f"clip.json nao e objeto em {name!r}")
            continue
        if meta.get("frames") != entry.get("frames"):
            problems.append(
                f"frames divergentes em {name!r}: "
                f"clip.json={meta.get('frames')} vs manifesto={entry.get('frames')}"
            )

    for name in sorted(disk_dirs):
        if name not in manifest_clips:
            problems.append(f"clipe {name!r} no disco fora do manifesto")

    return problems
