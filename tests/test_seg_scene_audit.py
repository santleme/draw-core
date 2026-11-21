"""lab/seg_scene_audit.py: as medidas da issue #106 por frame e resumo.

O caminho íntegro usa o NeuralSheetDetector real com sessão falsa
(_Session/_logits_for_classes de test_neural_onnx_contract, sem
onnxruntime). Os cenários explosivos — quad colado nas bordas,
extrapolando o frame — são roteirizados num detector de mentira: por
máscara sintética o ajuste ou é recusado pela guarda de
plausibilidade ou fecha plausível demais para exercitar os
contadores.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab import seg_scene_audit as sa  # noqa: E402

from draw.perception import NeuralSheetDetector  # noqa: E402
from draw.protocol import HEADER_SIZE, pack_record  # noqa: E402
from test_neural_onnx_contract import _logits_for_classes, _Session  # noqa: E402


def _detector_for_classes(classes):
    """Detector real com sessão falsa de 4 classes sobre a grade dada."""
    session = _Session({"logits": _logits_for_classes(classes)})
    detector = NeuralSheetDetector("unused.onnx", verify=None)
    detector._session = session
    return detector


class _FakeDet:
    """Detector de mentira: roteiro de cantos/máscaras por chamada.

    Mesmo protocolo do real: callable com return_mask devolvendo
    (cantos, veredito, máscara) e os atributos last_* que a auditoria
    lê depois da chamada. O último item do roteiro repete se a
    chamada passar do fim.
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.last_coverage_mask = None
        self.last_coverage_raw = None
        self.last_hand_mask = None
        self.last_presence_logit = None

    def __call__(self, frame, *, return_mask=True):
        item = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        self.last_coverage_mask = item.get("coverage")
        # Sem "raw_coverage" no roteiro, None: a auditoria cai na
        # máscara preenchida, como nos dublês antigos.
        self.last_coverage_raw = item.get("raw_coverage")
        self.last_hand_mask = item.get("hand")
        self.last_presence_logit = item.get("presence")
        mask = item.get("coverage") if return_mask else None
        return item.get("corners"), item.get("verdict"), mask


def _exploded_item():
    """O quad mentiroso da issue: abraça o frame e extrapola a borda.

    Frame 100x80 (w x h), cobertura nativa 50x40 com a metade de cima
    não-fundo: coverage_ratio = 0.5, border_ratio = 0.5 (topo e 19 px
    de cada lateral). O quad cobre o frame inteiro e sai 60 px pela
    esquerda — área ~2.2x o frame, fill ~0.5, oob 60 px.
    """
    coverage = np.zeros((40, 50), np.uint8)
    coverage[:20] = 255
    corners = np.array([[-60, -10], [119, -10], [119, 89], [-60, 89]], np.float32)
    return {
        "corners": corners,
        "verdict": "paper",
        "coverage": coverage,
        "hand": None,
        "presence": 6.5,
    }


def test_quadro_limpo_mede_geometria_da_folha():
    """Retângulo central de papel: quad plausível, fill cheio, oob 0."""
    classes = np.zeros((8, 8), np.uint8)
    classes[2:6, 2:6] = 1  # papel: 16 px de 64 -> coverage 0.25
    det = _detector_for_classes(classes)
    frame = np.zeros((64, 64, 3), np.uint8)

    (record,) = sa.audit_frames(iter([("clean.png", frame)]), det, log=lambda *_a: None)

    assert record["frame"] == 0 and record["file"] == "clean.png"
    assert record["presence_logit"] == 1.0
    assert record["verdict"] == "paper"
    assert record["coverage_ratio"] == pytest.approx(0.25)
    assert record["components"] == 1
    assert record["largest_share"] == pytest.approx(1.0)
    assert record["border_ratio"] == pytest.approx(0.0)
    assert record["hand_ratio"] == pytest.approx(0.0)
    assert record["quad_oob_px"] == 0.0
    assert record["quad_area_ratio"] < sa.QUAD_AREA_GE
    assert record["quad_fill"] >= sa.FILL_BELOW


def test_frame_sem_folha_nao_fecha_quad():
    """Grade toda fundo: presença negativa forte, sem campos de quad."""
    det = _detector_for_classes(np.zeros((8, 8), np.uint8))
    frame = np.zeros((64, 64, 3), np.uint8)

    (record,) = sa.audit_frames(iter([("bg.png", frame)]), det, log=lambda *_a: None)

    assert record["verdict"] is None
    assert record["presence_logit"] == -10.0
    assert record["coverage_ratio"] == 0.0
    assert record["components"] == 0
    assert record["largest_share"] is None
    assert "quad_area_ratio" not in record
    assert "quad_fill" not in record


def test_contrato_dois_canais_hand_ratio_nulo():
    """O contrato legado (2 canais) não sabe o que é mão: null."""
    logits = np.full((1, 2, 8, 8), -4.0, np.float32)
    logits[0, 1, 1:7, 1:7] = 4.0
    session = _Session(
        {
            "mask_logits": logits,
            "presence_logit": np.array([2.5], np.float32),
        }
    )
    det = NeuralSheetDetector("unused.onnx", verify=None)
    det._session = session
    frame = np.zeros((64, 64, 3), np.uint8)

    (record,) = sa.audit_frames(iter([("x.png", frame)]), det, log=lambda *_a: None)

    assert record["hand_ratio"] is None
    assert record["coverage_ratio"] == pytest.approx(36 / 64)


def test_quad_explodido_medido_por_det_falso():
    """O cenário da issue: fill baixo, área > frame, canto 60 px fora."""
    det = _FakeDet([_exploded_item()])
    frame = np.zeros((80, 100, 3), np.uint8)

    (record,) = sa.audit_frames(iter([("boom.png", frame)]), det, log=lambda *_a: None)

    assert record["coverage_ratio"] == pytest.approx(0.5)
    assert record["components"] == 1
    assert record["largest_share"] == pytest.approx(1.0)
    assert record["border_ratio"] == pytest.approx(0.5)
    assert record["hand_ratio"] is None
    assert record["quad_oob_px"] == pytest.approx(60.0)
    assert record["quad_area_ratio"] > sa.QUAD_AREA_GE
    assert record["quad_fill"] == pytest.approx(0.5, abs=0.05)


def test_resumo_conta_as_tres_familias_da_issue():
    """fill_below, quad_area_ge e oob_ge_px sobem só onde deviam."""
    exploded = _exploded_item()
    clean_coverage = np.zeros((40, 50), np.uint8)
    # O quad 40x40 do frame (100x80) escala para [10,10]-[30,30] na
    # grade 50x40 — a cobertura limpa preenche o quad inteiro.
    clean_coverage[10:30, 10:30] = 255
    clean = {
        "corners": np.array([[20, 20], [60, 20], [60, 60], [20, 60]], np.float32),
        "verdict": "paper",
        "coverage": clean_coverage,
        "hand": None,
        "presence": 4.0,
    }
    no_quad = {"corners": None, "verdict": None, "coverage": None, "presence": -9.0}
    det = _FakeDet([exploded, clean, no_quad])
    frame = np.zeros((80, 100, 3), np.uint8)
    frames = iter([(f"f{i}.png", frame) for i in range(3)])

    records = sa.audit_frames(frames, det, log=lambda *_a: None)
    summary = sa.summarize(records, model="m.onnx")

    assert summary["frames"] == 3
    assert summary["frames_with_quad"] == 2
    assert summary["model"] == "m.onnx"
    assert summary["thresholds"] == {
        "fill_below": sa.FILL_BELOW,
        "quad_area_ge": sa.QUAD_AREA_GE,
        "oob_ge_px": sa.OOB_GE_PX,
    }
    assert records[1]["quad_fill"] >= sa.FILL_BELOW
    assert summary["failures"] == {
        "fill_below": 1,  # só o explodido; o limpo tem fill cheio
        "quad_area_ge": 1,
        "oob_ge_px": 1,
    }
    assert summary["quad_fill_median"] == pytest.approx(
        (records[0]["quad_fill"] + records[1]["quad_fill"]) / 2
    )
    # O frame sem máscara (coverage None) fica fora da média.
    assert summary["coverage_ratio_mean"] == pytest.approx(
        (records[0]["coverage_ratio"] + records[1]["coverage_ratio"]) / 2
    )


def test_resumo_frame_sem_mascara_nao_entra_na_media():
    """coverage None (det sem máscara auxiliar) não puxa a média."""
    det = _FakeDet(
        [
            {"corners": None, "verdict": None, "coverage": None},
            {
                "corners": None,
                "verdict": None,
                "coverage": np.full((4, 4), 255, np.uint8),
            },
        ]
    )
    frame = np.zeros((8, 8, 3), np.uint8)
    records = sa.audit_frames(
        iter([("a.png", frame), ("b.png", frame)]), det, log=lambda *_a: None
    )
    summary = sa.summarize(records)
    assert summary["frames"] == 2 and summary["frames_with_quad"] == 0
    assert summary["coverage_ratio_mean"] == pytest.approx(1.0)
    assert summary["quad_fill_p5"] is None
    assert summary["quad_fill_median"] is None
    assert summary["failures"] == {"fill_below": 0, "quad_area_ge": 0, "oob_ge_px": 0}


def test_max_frames_para_no_limite():
    det = _FakeDet([_exploded_item()])
    frame = np.zeros((8, 8, 3), np.uint8)
    frames = iter([(f"f{i}.png", frame) for i in range(5)])
    records = sa.audit_frames(frames, det, max_frames=2, log=lambda *_a: None)
    assert len(records) == 2 and det.calls == 2


def test_max_frames_nao_puxa_quadro_extra():
    """O limite fecha o iterável ANTES: a fonte nunca avança além.

    Reproduz o achado da revisão: o laço for pedia o item N+1 antes
    de checar o limite — uma fonte que quebra ao avançar (imread
    falho, registro truncado) derrubava a corrida já gravada.
    """
    det = _FakeDet([_exploded_item()])
    frame = np.zeros((8, 8, 3), np.uint8)

    def _guarded():
        for i in range(2):
            yield f"f{i}.png", frame
        raise AssertionError("a fonte avançou além do --max-frames")

    records = sa.audit_frames(_guarded(), det, max_frames=2, log=lambda *_a: None)
    assert len(records) == 2 and det.calls == 2


def test_buraco_de_fundo_conta_no_fill_e_nas_falhas():
    """A cobertura crua mede o buraco que a máscara de voto remenda.

    O detector preenche buracos internos para o voto de tinta; a
    auditoria mede o mapa cru — um fundo cercado de papel é falha de
    segmentação, não pixel preenchido (achado da revisão): o fill
    cai abaixo do piso e o contador registra.
    """
    raw = np.zeros((40, 50), np.uint8)
    raw[10:30, 10:30] = 255  # anel de papel na grade da cobertura
    raw[14:28, 14:28] = 0  # buraco de fundo cercado (~44% do quad)
    filled = raw.copy()
    filled[14:28, 14:28] = 255  # o que o voto de tinta enxerga
    det = _FakeDet(
        [
            {
                "corners": np.array(
                    [[20, 20], [60, 20], [60, 60], [20, 60]], np.float32
                ),
                "verdict": "paper",
                "coverage": filled,
                "raw_coverage": raw,
                "presence": 4.0,
            }
        ]
    )
    frame = np.zeros((80, 100, 3), np.uint8)

    (record,) = sa.audit_frames(iter([("hole.png", frame)]), det, log=lambda *_a: None)

    assert record["quad_fill"] < sa.FILL_BELOW
    summary = sa.summarize([record])
    assert summary["failures"]["fill_below"] == 1


def test_run_sobre_diretorio_de_pngs_grava_jsonl_e_resumo(tmp_path):
    """Fonte = diretório: imread por arquivo, file = nome do PNG."""
    src = tmp_path / "frames"
    src.mkdir()
    frame = np.zeros((80, 100, 3), np.uint8)
    for i in range(3):
        assert cv2.imwrite(str(src / f"frame_{i:05d}.png"), frame)
    (src / "notas.txt").write_text("ignorado")  # fora do glob
    no_quad = {"corners": None, "verdict": None, "coverage": None, "presence": -9.0}
    det = _FakeDet([_exploded_item(), no_quad, _exploded_item()])
    out = tmp_path / "audit"

    sa.run(src, det, out_dir=out, model="fake.onnx", log=lambda *_a: None)

    lines = (out / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    assert [r["file"] for r in rows] == [f"frame_{i:05d}.png" for i in range(3)]
    assert rows[0]["quad_oob_px"] == pytest.approx(60.0)
    assert "quad_fill" not in rows[1]
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["model"] == "fake.onnx"
    assert audit["frames"] == 3 and audit["frames_with_quad"] == 2
    assert audit["failures"] == {"fill_below": 2, "quad_area_ge": 2, "oob_ge_px": 2}
    assert audit["interrupted"] is False and audit["source_error"] is None


def test_run_drawcap_truncado_marca_source_error(tmp_path):
    """Registro cortado no payload: o resumo sai marcando incompleto.

    Reproduz o achado da revisão: HelperCapture.read() devolve
    (False, None) no registro truncado igual ao EOF limpo — sem o
    last_error o prefixo legível passava por captura completa.
    """
    from test_protocol import _jpeg

    src = tmp_path / "sessao.drawcap"
    payload = _jpeg()
    full = pack_record(0, 0, payload)
    truncated = pack_record(1, 33_333, payload)[: HEADER_SIZE + 10]
    src.write_bytes(full + truncated)
    det = _FakeDet([_exploded_item()])
    out = tmp_path / "audit"

    summary = sa.run(src, det, out_dir=out, model="m.onnx", log=lambda *_a: None)

    assert summary["frames"] == 1  # só o registro íntegro entra
    assert summary["interrupted"] is False
    assert "truncado" in summary["source_error"]
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["source_error"] == summary["source_error"]


def test_run_interrompida_grava_resumo_marcado(tmp_path):
    """Ctrl+C no meio: as linhas medidas viram resumo interrupted.

    A política do desk_capture aplicada à auditoria (achado da
    revisão): a corrida morta no meio não some com o resumo — as
    linhas já medidas entram na conta e o JSON diz que parou.
    """
    src = tmp_path / "frames"
    src.mkdir()
    frame = np.zeros((80, 100, 3), np.uint8)
    for i in range(3):
        assert cv2.imwrite(str(src / f"f_{i:05d}.png"), frame)

    class _InterruptDet(_FakeDet):
        def __call__(self, frame, *, return_mask=True):
            if self.calls == 1:
                raise KeyboardInterrupt
            return super().__call__(frame, return_mask=return_mask)

    det = _InterruptDet([_exploded_item()])
    out = tmp_path / "audit"

    summary = sa.run(src, det, out_dir=out, log=lambda *_a: None)

    assert summary["interrupted"] is True
    assert summary["frames"] == 1  # a linha medida antes do Ctrl+C
    assert summary["source_error"] is None
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["interrupted"] is True
    assert len((out / "frames.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_main_interrompida_avisa_e_sai_130(tmp_path, monkeypatch, capsys):
    """Ctrl+C: resumo parcial gravado, CLI avisa e sai 130.

    Reproduz o achado da revisão: o resumo parcial continua saindo
    em audit.json, mas a linha de comando precisa dizer que a
    auditoria não cobriu a captura toda — não o "fim" de sucesso.
    """
    src = tmp_path / "frames"
    src.mkdir()
    frame = np.zeros((80, 100, 3), np.uint8)
    for i in range(3):
        assert cv2.imwrite(str(src / f"f_{i:05d}.png"), frame)

    class _InterruptDet(_FakeDet):
        def __call__(self, frame, *, return_mask=True):
            if self.calls == 1:
                raise KeyboardInterrupt
            return super().__call__(frame, return_mask=return_mask)

    det = _InterruptDet([_exploded_item()])
    monkeypatch.setattr("draw.perception.NeuralSheetDetector", lambda *a, **k: det)
    out = tmp_path / "audit"

    code = sa.main(
        [
            "--source",
            str(src),
            "--model",
            "fake.onnx",
            "--out",
            str(out),
            "--allow-unverified",
        ]
    )

    assert code == 130
    printed = capsys.readouterr().out
    assert "interrompido" in printed and "fim" not in printed
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["interrupted"] is True and audit["frames"] == 1


def test_run_com_dump_overlays_grava_overlay(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    frame = np.zeros((80, 100, 3), np.uint8)
    assert cv2.imwrite(str(src / "f_00000.png"), frame)
    det = _FakeDet([_exploded_item()])
    out = tmp_path / "audit"

    sa.run(src, det, out_dir=out, dump_overlays=True, log=lambda *_a: None)

    overlay = cv2.imread(str(out / "overlay_00000.png"))
    assert overlay is not None
    # metade de cima tintada de avermelhado (BGR: R > B), fundo intacto
    assert overlay[10, 50, 2] > overlay[10, 50, 0]
    assert overlay[70, 50].tolist() == [0, 0, 0]


def test_run_recusa_out_com_auditoria_anterior(tmp_path):
    """--out reutilizado não mistura corridas: recusa, não apaga."""
    for artifact in ("audit.json", "frames.jsonl", "overlay_00007.png"):
        out = tmp_path / artifact.replace(".", "_")
        out.mkdir()
        (out / artifact).write_text("{}")
        with pytest.raises(ValueError, match="artefatos"):
            sa.run(
                tmp_path / "sem_frames",
                _FakeDet([]),
                out_dir=out,
                log=lambda *_a: None,
            )
        assert (out / artifact).exists()  # evidência intocada


def test_artefatos_ignoram_nomes_de_outras_ferramentas(tmp_path):
    """overlay_1.png (sem os 5 dígitos) e PNGs comuns não colidem."""
    out = tmp_path / "audit"
    out.mkdir()
    (out / "overlay_1.png").write_bytes(b"")
    (out / "scene_00000.png").write_bytes(b"")
    assert not sa._audit_artifacts(out)
    (out / "overlay_00001.png").write_bytes(b"")
    assert sa._audit_artifacts(out)


def test_main_recusa_out_sujo_sem_carregar_modelo(tmp_path):
    """A recusa acontece antes do ONNX: missing.onnx nem é lido."""
    out = tmp_path / "audit"
    out.mkdir()
    (out / "frames.jsonl").write_text("")
    with pytest.raises(SystemExit) as exc:
        sa.main(
            [
                "--source",
                str(tmp_path / "frames"),
                "--model",
                "missing.onnx",
                "--out",
                str(out),
                "--allow-unverified",
            ]
        )
    assert exc.value.code == 2


def test_audit_frames_sem_out_nao_grava_nada(tmp_path, monkeypatch):
    """Núcleo puro: sem out_dir, nenhum arquivo aparece."""
    monkeypatch.chdir(tmp_path)
    det = _FakeDet([_exploded_item()])
    frame = np.zeros((8, 8, 3), np.uint8)
    sa.audit_frames(iter([("x.png", frame)]), det, log=lambda *_a: None)
    assert list(tmp_path.iterdir()) == []
