"""Testes de schema, rotulagem e manifesto do corpus de clipes reais."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from lab.corpus.clip import (
    CLIP_JSON_VERSION,
    CONDITION_KEYS,
    EXPECTED_RULES,
    ClipError,
    clip_frames,
    count_frames,
    load_clip,
    validate_clip,
    write_clip_json,
)
from lab.corpus.labels import ClipInitError, init_clip, upgrade_clip
from lab.corpus.manifest import (
    MANIFEST_NAME,
    build_manifest,
    hash_clip,
    iter_clip_dirs,
    load_manifest,
    verify_manifest,
    write_manifest,
)


def _write_frames(clip_dir, count=3, *, value=200):
    clip_dir.mkdir(parents=True, exist_ok=True)
    img = np.full((8, 8, 3), value, dtype=np.uint8)
    for i in range(count):
        cv2.imwrite(str(clip_dir / f"f{i:04d}.png"), img)


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_v1_clip_loads_with_empty_defaults_and_validates(tmp_path):
    clip_dir = tmp_path / "c1"
    _write_frames(clip_dir, count=2)
    _write_json(
        clip_dir / "clip.json",
        {
            "scene_fps": 10.0,
            "orientation": "retrato",
            "frames": 2,
            "desc": "clipe v1 antigo",
        },
    )
    spec = load_clip(clip_dir)
    assert spec.version == 1
    assert spec.fps == 10.0
    assert spec.orientation == "retrato"
    assert spec.frames == 2
    assert spec.conditions == {}
    assert spec.provenance == {}
    assert spec.expected == []
    assert spec.issues == []
    assert spec.regions is None
    assert validate_clip(clip_dir) == []


def test_v2_full_clip_loads_and_validate_catches_bad_fields(tmp_path):
    clip_dir = tmp_path / "c2"
    _write_frames(clip_dir, count=3)
    write_clip_json(
        clip_dir,
        {
            "version": CLIP_JSON_VERSION,
            "scene_fps": 8.0,
            "orientation": "paisagem",
            "frames": 3,
            "desc": "sombra movel",
            "conditions": {CONDITION_KEYS[0]: "sombra-movel"},
            "provenance": {"source": "desk_capture"},
            "expected": [{"rule": EXPECTED_RULES[1], "min_px": 100}],
            "issues": [133],
        },
    )
    _write_json(clip_dir / "regions.json", {"writing": [[0, 0, 8, 8]]})

    spec = load_clip(clip_dir)
    assert spec.version == 2
    assert spec.conditions == {"lighting": "sombra-movel"}
    assert spec.provenance == {"source": "desk_capture"}
    assert spec.expected == [{"rule": "ink_present", "min_px": 100}]
    assert spec.issues == [133]
    assert spec.regions == {"writing": [[0, 0, 8, 8]]}
    assert validate_clip(clip_dir) == []

    # Condicao desconhecida, regra ruim e contagem divergente aparecem na validacao.
    _write_json(
        clip_dir / "clip.json",
        {
            "version": 2,
            "scene_fps": 8.0,
            "frames": 99,
            "desc": "ruim",
            "conditions": {"clima_inventado": "sol"},
            "expected": [{"rule": "regra_inexistente"}],
        },
    )
    problems = validate_clip(clip_dir)
    joined = "\n".join(problems)
    assert "clima_inventado" in joined
    assert "regra_inexistente" in joined
    assert "99" in joined

    with pytest.raises(ClipError):
        _write_json(clip_dir / "clip.json", {"version": 9, "scene_fps": 1})
        load_clip(clip_dir)


def test_validate_clip_rejects_bad_region_boxes(tmp_path):
    """Caixas de regions.json: mesma régua que o replay usa ao fatiar."""
    clip_dir = tmp_path / "regioes"
    _write_frames(clip_dir, count=2)
    init_clip(clip_dir, fps=8.0)

    # Coordenada faltante: antes passava no check e abortava no replay.
    _write_json(clip_dir / "regions.json", {"writing": [[10, 20, 40]]})
    assert any("writing[0]" in p for p in validate_clip(clip_dir))

    # Invertida: antes voltava fatia vazia e contagem zero enganosa.
    _write_json(clip_dir / "regions.json", {"writing": [[40, 20, 10, 50]]})
    assert any("limites" in p for p in validate_clip(clip_dir))

    # Fora do canvas (retrato a4 = 850x1200) e chave desconhecida.
    _write_json(clip_dir / "regions.json", {"writing": [[0, 0, 2000, 2000]]})
    assert any("canvas" in p for p in validate_clip(clip_dir))
    _write_json(clip_dir / "regions.json", {"writting": [[0, 0, 10, 10]]})
    assert any("desconhecida" in p for p in validate_clip(clip_dir))

    _write_json(
        clip_dir / "regions.json",
        {"writing": [[0, 0, 100, 100]], "ruling_only": [[10, 10, 800, 1100]]},
    )
    assert validate_clip(clip_dir) == []


def test_validate_clip_checks_expected_scopes(tmp_path):
    """frames/region das regras sao validados contra clipe e canvas."""
    clip_dir = tmp_path / "escopo"
    _write_frames(clip_dir, count=2)
    _write_json(
        clip_dir / "clip.json",
        {
            "version": 2,
            "scene_fps": 8.0,
            "frames": 2,
            "desc": "escopo",
            "expected": [
                {"rule": "ink_present", "min_px": 10, "frames": [0, 5]},
                {"rule": "no_ink", "region": [50, 50, 10, 10]},
                {"rule": "canvas_stable", "region": [0, 0, 3000, 10]},
                {"rule": "no_ink", "frames": [-2, 1]},
            ],
        },
    )
    joined = "\n".join(validate_clip(clip_dir))
    assert "fora do clipe" in joined  # frames [0,5] num clipe de 2 quadros
    assert "limites" in joined  # region invertida
    assert "canvas" in joined  # region fora da grade 850x1200
    assert "0 <= a <= b" in joined  # frames negativo

    _write_json(
        clip_dir / "clip.json",
        {
            "version": 2,
            "scene_fps": 8.0,
            "frames": 2,
            "desc": "escopo",
            "expected": [
                {
                    "rule": "ink_present",
                    "min_px": 10,
                    "frames": [0, 1],
                    "region": [0, 0, 100, 100],
                }
            ],
        },
    )
    assert validate_clip(clip_dir) == []


def test_count_frames_ignores_non_frame_pngs(tmp_path):
    clip_dir = tmp_path / "c_extra"
    _write_frames(clip_dir, count=2)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(clip_dir / "mask_0000.png"), img)
    cv2.imwrite(str(clip_dir / "notes.png"), img)
    sub = clip_dir / "nested"
    sub.mkdir()
    cv2.imwrite(str(sub / "f0000.png"), img)
    assert count_frames(clip_dir) == 2


def test_clip_frames_is_the_shared_inventory(tmp_path):
    """O mesmo f*.png ordenado alimenta contagem e hash: preview.png,
    raw/ e imagens de outra extensao ficam fora dos dois."""
    clip_dir = tmp_path / "inv"
    _write_frames(clip_dir, count=2)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(clip_dir / "preview.png"), img)
    raw = clip_dir / "raw"
    raw.mkdir()
    cv2.imwrite(str(raw / "f9999.png"), img)
    cv2.imwrite(str(clip_dir / "f0002.jpg"), img)

    assert [p.name for p in clip_frames(clip_dir)] == [
        "f0000.png",
        "f0001.png",
    ]
    assert count_frames(clip_dir) == 2
    h = hash_clip(clip_dir)
    cv2.imwrite(str(clip_dir / "mais_um.png"), img)
    assert hash_clip(clip_dir) == h  # fora do padrao, fora do hash


def test_clip_frames_ordena_por_sufixo_numerico(tmp_path):
    """A captura grava minimo 4 digitos, nao largura fixa: f10000.png
    segue f9999.png — ordem lexica o rebotava para depois de f1000.png.
    Nomes sem sufixo numerico vao ao fim; numero empatado desempata
    pelo nome."""
    clip_dir = tmp_path / "ord"
    clip_dir.mkdir()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    for name in (
        "f10000.png",
        "fbackup.png",
        "f9999.png",
        "f1.png",
        "f1000.png",
        "f0001.png",
        "f2.png",
    ):
        cv2.imwrite(str(clip_dir / name), img)

    assert [p.name for p in clip_frames(clip_dir)] == [
        "f0001.png",
        "f1.png",
        "f2.png",
        "f1000.png",
        "f9999.png",
        "f10000.png",
        "fbackup.png",
    ]

    # O hash nao depende da ordem nova: o manifesto segue valido.
    init_clip(clip_dir, fps=10.0)
    write_manifest(tmp_path, build_manifest(tmp_path))
    assert verify_manifest(tmp_path) == []


def test_clip_frames_nome_fora_do_padrao_vai_ao_fim(tmp_path):
    """f_backup0.png termina em dígito mas não é f<dígitos>: vai ao
    fim em vez de ocupar a posição 0 e deslocar todos os rótulos
    posicionais do clipe."""
    clip_dir = tmp_path / "cl"
    clip_dir.mkdir()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    for name in ("f_backup0.png", "f0001.png", "f0000.png"):
        cv2.imwrite(str(clip_dir / name), img)

    assert [p.name for p in clip_frames(clip_dir)] == [
        "f0000.png",
        "f0001.png",
        "f_backup0.png",
    ]


def test_frame_sort_key_ordena_qualquer_sufixo_numerico():
    """A chave vale alem de f<N>: todo stem que termina em digito
    ordena pelo valor; sem sufixo numerico vai ao fim e empate de
    numero desempata pelo nome."""
    from lab.corpus.clip import frame_sort_key

    nomes = ["seq_10.png", "f2.png", "f10.png", "capa.png", "f0002.png"]
    ordenados = sorted(nomes, key=lambda n: frame_sort_key(Path(n)))
    assert ordenados == [
        "f0002.png",  # 2, desempate pelo nome
        "f2.png",  # 2
        "f10.png",  # 10
        "seq_10.png",  # 10, perde o desempate para f10.png
        "capa.png",  # sem digito no fim: por ultimo
    ]


def test_validate_clip_flags_window_beyond_inventory(tmp_path):
    """Janela dentro do ``frames`` declarado mas alem do inventario real
    e apontada quadro a quadro — alem da divergencia de contagem."""
    clip_dir = tmp_path / "gap"
    _write_frames(clip_dir, count=2)
    _write_json(
        clip_dir / "clip.json",
        {
            "version": 2,
            "scene_fps": 8.0,
            "frames": 5,
            "desc": "declarado > inventario",
            "expected": [{"rule": "no_ink", "frames": [3, 4]}],
        },
    )
    problems = validate_clip(clip_dir)
    joined = "\n".join(problems)
    assert "difere de frames" in joined  # contagem 2 x declarado 5
    assert "inventario" in joined and "[3, 4]" in joined


def test_init_clip_writes_v2_and_refuses_invalid_cases(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ClipInitError):
        init_clip(empty_dir, fps=10.0)

    clip_dir = tmp_path / "ok"
    _write_frames(clip_dir, count=2)
    with pytest.raises(ClipInitError, match="desconhecida"):
        init_clip(clip_dir, fps=10.0, conditions={"foo": "bar"})

    out = init_clip(
        clip_dir,
        fps=12.0,
        orientation="retrato",
        desc="teste",
        conditions={"lighting": "baixa"},
        provenance={"source": "desk"},
        expected=[{"rule": "no_ink"}],
        issues=[129],
    )
    assert out == clip_dir / "clip.json"
    spec = load_clip(clip_dir)
    assert spec.version == 2
    assert spec.frames == 2
    assert spec.fps == 12.0
    assert spec.issues == [129]

    with pytest.raises(ClipInitError):
        init_clip(clip_dir, fps=12.0)

    init_clip(clip_dir, fps=15.0, overwrite=True)
    assert load_clip(clip_dir).fps == 15.0


def test_upgrade_clip_v1_to_v2_preserves_fields_and_is_idempotent(tmp_path):
    clip_dir = tmp_path / "legacy"
    _write_frames(clip_dir, count=2)
    _write_json(
        clip_dir / "clip.json",
        {
            "scene_fps": 7.5,
            "orientation": "paisagem",
            "frames": 2,
            "desc": "original",
            "rotate_applied": 180,
        },
    )
    upgrade_clip(clip_dir, provenance={"source": "bench-clips"})
    spec = load_clip(clip_dir)
    assert spec.version == 2
    assert spec.fps == 7.5
    assert spec.orientation == "paisagem"
    assert spec.desc == "original"
    assert spec.provenance == {"source": "bench-clips"}
    raw1 = (clip_dir / "clip.json").read_text(encoding="utf-8")
    assert json.loads(raw1)["rotate_applied"] == 180

    # Idempotente na segunda chamada
    upgrade_clip(clip_dir)
    raw2 = (clip_dir / "clip.json").read_text(encoding="utf-8")
    assert raw1 == raw2


def test_hash_clip_deterministic_sensitive_to_frames_ignores_scratch(tmp_path):
    clip_dir = tmp_path / "h"
    _write_frames(clip_dir, count=2, value=120)
    init_clip(clip_dir, fps=10.0)
    h1 = hash_clip(clip_dir)
    assert h1 == hash_clip(clip_dir)

    (clip_dir / "scratch.txt").write_text("anotacao irrelevante", encoding="utf-8")
    assert hash_clip(clip_dir) == h1

    _write_frames(clip_dir, count=2, value=40)
    assert hash_clip(clip_dir) != h1


def test_manifest_roundtrip_and_tamper_detection(tmp_path):
    c1 = tmp_path / "clip-a"
    c2 = tmp_path / "clip-b"
    _write_frames(c1, count=2, value=10)
    _write_frames(c2, count=3, value=20)
    init_clip(c1, fps=8.0, conditions={"lighting": "normal"}, issues=[129])
    init_clip(c2, fps=8.0)

    assert [d.name for d in iter_clip_dirs(tmp_path)] == ["clip-a", "clip-b"]
    manifest = build_manifest(tmp_path)
    out = write_manifest(tmp_path, manifest)
    assert out == tmp_path / MANIFEST_NAME
    loaded = load_manifest(tmp_path)
    assert loaded == manifest
    assert verify_manifest(tmp_path) == []

    # Alterar um quadro invalida o sha256.
    _write_frames(c1, count=2, value=99)
    problems = verify_manifest(tmp_path)
    assert any("sha256" in p and "clip-a" in p for p in problems)

    # Restaura o manifesto para testar diretorio faltante e clipe extra.
    write_manifest(tmp_path, build_manifest(tmp_path))
    for p in c2.iterdir():
        p.unlink()
    c2.rmdir()
    c3 = tmp_path / "clip-novo"
    _write_frames(c3, count=1)
    init_clip(c3, fps=8.0)
    problems2 = verify_manifest(tmp_path)
    assert any("clip-b" in p for p in problems2)
    assert any("clip-novo" in p for p in problems2)


def test_load_manifest_absent_and_bad_schema(tmp_path):
    assert load_manifest(tmp_path) is None
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"schema": 99, "clips": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_manifest(tmp_path)


# --- Pin do sha256 do manifesto ---------------------------------------------
#
# O stream que ``hash_clip`` alimenta ao hasher e contrato de
# compatibilidade: todo ``corpus.json`` ja gravado so revalida se, para cada
# arquivo na ordem LEXICA do nome, entrarem
#
#     name_utf8 + b"\0" + str(len(data)).encode("ascii") + b"\0" + data + b"\0"
#
# cobrindo clip.json, regions.json (quando presente) e os f*.png. A literal
# abaixo foi calculada a partir dessa especificacao (hashlib sobre o stream
# descrito) e conferida uma unica vez contra a implementacao — nao e "lida"
# de hash_clip. Mudar ordem, separadores ou cobertura reescreve todos os
# digests registrados e DEVE falhar aqui.
_PINNED_CLIP_SHA256 = "51033e2a9f78e6599c2cfa3b19ef97f1b23855a65ea1a980c8310f88e7f7eb9e"

_PINNED_CLIP_FILES = {
    "clip.json": b'{"scene_fps": 10.0, "frames": 3, "desc": "pin"}\n',
    # Quadros com sufixos cuja ordem numerica (1, 2, 10) diverge da lexica
    # (f1.png, f10.png, f2.png) — o digest segue a lexica.
    "f1.png": b"frame-one",
    "f2.png": b"frame-two",
    "f10.png": b"frame-ten",
    "regions.json": b'{"writing": [[0, 0, 8, 8]]}\n',
}


def _write_pinned_clip(clip_dir):
    clip_dir.mkdir(parents=True)
    for name, data in _PINNED_CLIP_FILES.items():
        (clip_dir / name).write_bytes(data)
    return clip_dir


def test_hash_clip_digest_is_pinned_to_known_byte_stream(tmp_path):
    """Pin anti-tautologico: build_manifest e verify_manifest usam o mesmo
    hash_clip, entao uma mudanca que reescrevesse todo sha256 armazenado
    (ordem do stream, separadores, arquivos cobertos) ainda se
    auto-verificaria. A literal congela o stream inteiro."""
    clip_dir = _write_pinned_clip(tmp_path / "pin")

    assert hash_clip(clip_dir) == _PINNED_CLIP_SHA256
    # E e esse mesmo digest pinado que o manifesto registra no corpus.json.
    manifest = build_manifest(tmp_path)
    assert manifest["clips"]["pin"]["sha256"] == _PINNED_CLIP_SHA256


def test_hash_clip_orders_files_lexically_not_numerically(tmp_path):
    """O inventario numerico e [f1, f2, f10], mas o stream do sha256 segue
    a ordem LEXICA dos nomes (clip.json, f1.png, f10.png, f2.png,
    regions.json): a literal foi computada nessa ordem, entao hashear na
    ordem numerica mudaria o digest e quebraria manifests gravados."""
    clip_dir = _write_pinned_clip(tmp_path / "pin")

    assert [p.name for p in clip_frames(clip_dir)] == [
        "f1.png",
        "f2.png",
        "f10.png",
    ]
    assert hash_clip(clip_dir) == _PINNED_CLIP_SHA256
