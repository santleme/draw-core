"""Schema e validacao de clipes reais do corpus (v1 e v2)."""

import dataclasses
import json
import re
from pathlib import Path

CLIP_JSON_VERSION = 2
CONDITION_KEYS = (
    "lighting",
    "surface",
    "content",
    "occluder",
    "motion",
    "camera",
)
EXPECTED_RULES = ("no_ink", "ink_present", "canvas_stable")
REGION_KEYS = ("writing", "ruling_only")

# Extensoes de imagem que as fontes-de-diretorio do lab aceitam
# (cv2.imread decodifica todas): uniao das listas que replay_lab,
# sync_workers e seg_scene_audit mantinham — .bmp chega de brinde a
# auditoria de cena, que so listava .png/.jpg/.jpeg.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class ClipError(Exception):
    """Erro ao carregar um clipe ou seus metadados."""


@dataclasses.dataclass(frozen=True)
class ClipSpec:
    """Especificacao carregada de um clipe do corpus."""

    name: str
    dir: Path
    fps: float
    orientation: str | None
    frames: int
    desc: str
    conditions: dict
    provenance: dict
    expected: list
    issues: list
    regions: dict | None
    version: int


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def frame_sort_key(path):
    """Chave de ordem de quadros: o INTEIRO do sufixo numerico do stem.

    O sufixo e a corrida de digitos no FIM do nome — ``f2.png``,
    ``frame_00010.jpg``, ``seq_9.bmp``: quem termina em digito ordena
    pelo valor, entao ``f2.png`` precede ``f10.png`` (a ordem lexica
    os invertia) e empates de numero (``f1.png`` x ``f0001.png``)
    desempatam pelo nome. Nomes sem sufixo numerico vao ao fim, por
    nome.

    A captura grava ``f{indice:04d}`` — minimo de 4 digitos, que
    cresce a partir do quadro 10000 — entao ``f10000.png`` segue
    ``f9999.png``, nao ``f1000.png`` (ordem lexica rebotava quadros
    longos).
    """
    digits = re.search(r"(\d+)$", path.stem)
    if digits is not None:
        return (0, int(digits.group(1)), path.name)
    return (1, 0, path.name)


def clip_frames(clip_dir) -> list[Path]:
    """Quadros ``f*.png`` do nivel raiz: ``f<digitos>`` em ordem
    numerica, demais nomes ao fim por nome.

    Inventario unico do clipe: a contagem de ``count_frames``, o hash
    do manifesto e o replay do corpus compartilham esta lista — o que
    o sha256 cobre e o que entra no replay sao os mesmos arquivos. O
    indice i dos rotulos (``frames`` de ``expected``) e a posicao
    nesta ordem — so coincide com o numero do arquivo quando os
    sufixos sao 0..n-1 continuos; ``validate_clip`` aponta a lacuna
    que faria as duas leituras divergir.

    A ordem numerica vale só para o padrão da captura (``f<digitos>``):
    um ``f_backup0.png`` terminaria em dígito e cairia na posição 0,
    deslocando todos os rótulos posicionais — fora do padrão ele vai
    ao fim, como já ia na ordem léxica.

    O manifesto NAO muda com a ordem numerica: ``hash_clip`` reordena
    por nome antes de hashear, entao o sha256 de clipes ja registrados
    se mantem.
    """
    clip_dir = Path(clip_dir)
    if not clip_dir.is_dir():
        return []
    return sorted(
        (p for p in clip_dir.glob("f*.png") if p.is_file()), key=_clip_sort_key
    )


def _clip_sort_key(path):
    """``f<digitos>`` pelo número; demais ``f*`` ao fim por nome."""
    match = re.fullmatch(r"f(\d+)", path.stem)
    if match is not None:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def count_frames(clip_dir) -> int:
    """Conta quadros ``f*.png`` no nivel raiz do diretorio do clipe."""
    return len(clip_frames(clip_dir))


def span_problem(span) -> str | None:
    """``None`` se ``span`` é um ``[a, b]`` utilizável como janela inclusiva.

    Dois inteiros com ``0 <= a <= b`` — posições no inventário de
    ``clip_frames``. Mesma disciplina de ``rect_problem``: o texto é
    devolvido para ``_validate_expected`` compor a mensagem, e
    ``window_missing`` o converte em ``ValueError``.
    """
    if not isinstance(span, list) or len(span) != 2:
        return f"deve ser [a, b]: {span!r}"
    if not all(_is_int(v) for v in span):
        return f"indices nao inteiros: {span!r}"
    if span[0] < 0 or span[0] > span[1]:
        return f"limites invalidos {span!r} (precisa 0 <= a <= b)"
    return None


def window_missing(span, observed) -> list[int]:
    """Indices pedidos pela janela ``[a, b]`` ausentes de ``observed``.

    Regua unica de completude de janela, compartilhada entre
    ``validate_clip`` (``observed`` = o inventario do clipe) e a
    avaliacao de ``expected`` (``observed`` = os quadros que o replay
    registrou no trace — quadro pulado por imread ilegivel deixa lacuna).
    Janela malformada levanta ValueError pela regua de ``span_problem``
    — a mesma do schema, para o replay nao medir a janela vazia de um
    span invertido como "sem quadro faltante".
    """
    problem = span_problem(span)
    if problem is not None:
        raise ValueError(problem)
    a, b = span
    seen = set(observed)
    return [i for i in range(a, b + 1) if i not in seen]


def rect_problem(rect, size=None):
    """``None`` se ``rect`` é um ``[x0, y0, x1, y1]`` utilizável como fatia.

    Quatro inteiros, não negativos e estritamente ordenados; ``size``
    ``(w, h)`` do canvas, quando conhecido, também confere os limites
    superiores. O texto do problema é devolvido para composição de
    mensagens por ``check_regions`` e pela validação de ``expected``.
    """
    if not isinstance(rect, list) or len(rect) != 4:
        return "deve ser [x0, y0, x1, y1]"
    if not all(_is_int(v) for v in rect):
        return f"coordenadas nao inteiras: {rect!r}"
    x0, y0, x1, y1 = rect
    if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
        return f"limites invalidos {rect!r} (precisa 0 <= x0 < x1, 0 <= y0 < y1)"
    if size is not None:
        w, h = size
        if x1 > w or y1 > h:
            return f"fora do canvas {w}x{h}: {rect!r}"
    return None


def check_regions(spec, size=None) -> list:
    """Problemas de um ``regions.json``: chaves suportadas com retas válidas."""
    if not isinstance(spec, dict):
        return ["regions.json deve ser objeto"]
    problems = []
    for key, value in spec.items():
        if key not in REGION_KEYS:
            problems.append(f"regions.json: chave desconhecida {key!r}")
            continue
        if not isinstance(value, list):
            problems.append(f"regions.{key} deve ser lista de retas")
            continue
        for idx, rect in enumerate(value):
            problem = rect_problem(rect, size)
            if problem is not None:
                problems.append(f"regions.{key}[{idx}] {problem}")
    return problems


def parse_regions(spec, size=None) -> dict:
    """Normaliza ``regions.json`` -> ``{chave: [(x0, y0, x1, y1), ...]}``.

    Mesma régua de ``check_regions``: qualquer problema vira ``ClipError``
    — quem conta pixels nunca fatia uma caixa que a validação rejeitaria.
    """
    problems = check_regions(spec, size)
    if problems:
        raise ClipError("regions.json invalido: " + "; ".join(problems))
    return {
        key: [tuple(rect) for rect in spec[key]] for key in REGION_KEYS if key in spec
    }


def _read_json_dict(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ClipError(f"{label} ausente em {path.parent}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClipError(f"{label} invalido em {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ClipError(f"{label} deve ser objeto JSON: {path}")
    return data


def load_clip(clip_dir) -> ClipSpec:
    """Carrega ``clip.json`` (v1 ou v2) e ``regions.json`` opcional."""
    clip_dir = Path(clip_dir)
    raw = _read_json_dict(clip_dir / "clip.json", "clip.json")

    if "version" not in raw:
        version = 1
    else:
        raw_ver = raw["version"]
        if not _is_int(raw_ver) or raw_ver not in (1, CLIP_JSON_VERSION):
            raise ClipError(f"versao de clip.json nao suportada: {raw_ver!r}")
        version = int(raw_ver)

    for key in ("scene_fps", "frames", "desc"):
        if key not in raw:
            raise ClipError(f"campo obrigatorio ausente em clip.json: {key!r}")

    fps = raw["scene_fps"]
    if not _is_number(fps) or fps <= 0:
        raise ClipError(f"scene_fps deve ser > 0: {fps!r}")
    frames = raw["frames"]
    if not _is_int(frames) or frames < 0:
        raise ClipError(f"frames invalido: {frames!r}")
    desc = raw["desc"]
    if not isinstance(desc, str):
        raise ClipError(f"desc deve ser string: {desc!r}")

    orientation = raw.get("orientation")
    if orientation is not None and not isinstance(orientation, str):
        raise ClipError(f"orientation invalida: {orientation!r}")

    conditions = raw.get("conditions", {})
    provenance = raw.get("provenance", {})
    expected = raw.get("expected", [])
    issues = raw.get("issues", [])
    if not isinstance(conditions, dict):
        raise ClipError("conditions deve ser objeto")
    if not isinstance(provenance, dict):
        raise ClipError("provenance deve ser objeto")
    if not isinstance(expected, list):
        raise ClipError("expected deve ser lista")
    if not isinstance(issues, list):
        raise ClipError("issues deve ser lista")

    regions_path = clip_dir / "regions.json"
    regions = (
        _read_json_dict(regions_path, "regions.json") if regions_path.exists() else None
    )

    return ClipSpec(
        name=clip_dir.name,
        dir=clip_dir,
        fps=float(fps),
        orientation=orientation,
        frames=int(frames),
        desc=desc,
        conditions=dict(conditions),
        provenance=dict(provenance),
        expected=list(expected),
        issues=list(issues),
        regions=regions,
        version=version,
    )


def _validate_expected(expected, frames=None, size=None, observed=None) -> list[str]:
    problems = []
    if not isinstance(expected, list):
        return ["expected deve ser uma lista"]
    for idx, entry in enumerate(expected):
        if not isinstance(entry, dict):
            problems.append(f"expected[{idx}] deve ser objeto")
            continue
        rule = entry.get("rule")
        if rule not in EXPECTED_RULES:
            problems.append(f"expected[{idx}].rule desconhecida: {rule!r}")
        if "frames" in entry:
            span = entry["frames"]
            problem = span_problem(span)
            if problem is not None:
                problems.append(f"expected[{idx}].frames {problem}")
            elif frames is not None and span[1] >= frames:
                problems.append(
                    f"expected[{idx}].frames {span!r} fora do clipe ({frames} quadros)"
                )
            elif observed is not None:
                missing = window_missing(span, observed)
                if missing:
                    problems.append(
                        f"expected[{idx}].frames {span!r} sem quadro no "
                        f"inventario: {missing}"
                    )
        if "region" in entry:
            problem = rect_problem(entry["region"], size)
            if problem is not None:
                problems.append(f"expected[{idx}].region {problem}")
        if "min_px" in entry and not _is_int(entry["min_px"]):
            problems.append(f"expected[{idx}].min_px deve ser inteiro")
    return problems


def _canvas_size(raw):
    """``(w, h)`` do canvas retificado do clipe, ou ``None`` se incógnito.

    O replay roda sempre na grade a4 na orientação declarada em
    ``clip.json`` (``run_clip`` assume retrato na ausência dela).
    Importação preguiçosa: ``draw.config`` carrega cv2.
    """
    from draw import config

    orientation = raw.get("orientation") or "retrato"
    if orientation not in config.ORIENTATIONS:
        return None
    return config.paper_grid("a4", orientation)


def validate_clip(clip_dir) -> list[str]:
    """Valida ``clip.json``, contagem de quadros e ``regions.json``.

    Sufixos fora de ``0..n-1`` contíguos (``f0000``, ``f0005``) viram
    problema: índice de rótulo é posição na ordem de ``clip_frames`` —
    número de arquivo só coincide sem lacuna, e uma janela rotulada
    "pelo nome" mediria outro quadro em silêncio.
    """
    clip_dir = Path(clip_dir)
    try:
        raw = _read_json_dict(clip_dir / "clip.json", "clip.json")
    except ClipError as exc:
        return [str(exc)]

    problems = []
    if "version" in raw:
        ver = raw["version"]
        if not _is_int(ver) or ver not in (1, CLIP_JSON_VERSION):
            problems.append(f"version invalida: {ver!r} (esperado 1 ou 2)")

    if "scene_fps" not in raw:
        problems.append("scene_fps ausente")
    elif not _is_number(raw["scene_fps"]) or raw["scene_fps"] <= 0:
        problems.append(f"scene_fps deve ser > 0: {raw['scene_fps']!r}")

    inventory = clip_frames(clip_dir)
    actual = len(inventory)
    if [int(p.stem[1:]) for p in inventory if p.stem[1:].isdecimal()] != list(
        range(actual)
    ):
        problems.append(
            "sufixos de f*.png sem 0..n-1 contiguo: indices de rotulo "
            "sao posicoes no inventario ordenado, nao numeros de arquivo"
        )
    if "frames" not in raw:
        problems.append("frames ausente")
    elif not _is_int(raw["frames"]) or raw["frames"] < 0:
        problems.append(f"frames invalido: {raw['frames']!r}")
    elif actual != raw["frames"]:
        problems.append(
            f"contagem de f*.png ({actual}) difere de frames ({raw['frames']})"
        )

    if "desc" not in raw:
        problems.append("desc ausente")
    elif not isinstance(raw["desc"], str):
        problems.append("desc deve ser string")

    if "conditions" in raw:
        conds = raw["conditions"]
        if not isinstance(conds, dict):
            problems.append("conditions deve ser objeto")
        else:
            for key, val in conds.items():
                if key not in CONDITION_KEYS:
                    problems.append(f"condicao desconhecida: {key!r}")
                elif not isinstance(val, str) or not val:
                    problems.append(f"condicao {key!r} deve ter valor texto nao vazio")

    if "provenance" in raw and not isinstance(raw["provenance"], dict):
        problems.append("provenance deve ser objeto")

    regions_path = clip_dir / "regions.json"
    size = _canvas_size(raw) if "expected" in raw or regions_path.exists() else None

    if "expected" in raw:
        declared = raw.get("frames")
        frames = declared if _is_int(declared) and declared >= 0 else None
        problems.extend(
            _validate_expected(raw["expected"], frames, size, range(actual))
        )

    if "issues" in raw:
        issues = raw["issues"]
        if not isinstance(issues, list) or not all(_is_int(v) for v in issues):
            problems.append("issues deve ser lista de inteiros")

    if regions_path.exists():
        try:
            spec = _read_json_dict(regions_path, "regions.json")
        except ClipError as exc:
            problems.append(str(exc))
        else:
            problems.extend(check_regions(spec, size))

    return problems


def write_clip_json(clip_dir, data: dict) -> Path:
    """Grava ``clip.json`` (garantindo ``version`` v2 se omitida)."""
    clip_dir = Path(clip_dir)
    payload = dict(data)
    payload.setdefault("version", CLIP_JSON_VERSION)
    out_path = clip_dir / "clip.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path
