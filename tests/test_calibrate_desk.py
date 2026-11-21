"""lab/calibrate_desk.py: recalibração dos limiares sem denominador.

A ferramenta aplica a MESMA regra da calibração de referência sobre
a telemetria de uma mesa nova (issue #105): piso de cobertura no
ponto médio fantasma↔folha, veto de presença no maior logit de
mesa vazia + margem. Sem um dos lados da distribuição, a sugestão
é None — calibrar com metade do dado é o chute a evitar.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab import calibrate_desk as cd  # noqa: E402


def _acq(source, logit=None, fit_reject=None):
    return {
        "event": "acquisition",
        "source": source,
        "presence_logit": logit,
        "fit_reject": fit_reject,
    }


def _snapshot(rejects=(), accepts=()):
    return {
        "coverage_iou_reject_recent": list(rejects),
        "coverage_iou_accept_recent": list(accepts),
        "t_epoch": 1.0,
    }


def _write_telemetry(tmp_path, records):
    path = tmp_path / "telemetry.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_sugere_piso_no_ponto_medio_da_regra_original(tmp_path):
    """Mesa nova com fantasma ≤0.10 e folha ≥0.30: a mesma regra
    que produziu 0.2 na referência (0.108/0.298) propõe ~0.2 aqui."""
    path = _write_telemetry(
        tmp_path,
        [
            _snapshot(rejects=[0.05, 0.10], accepts=[0.32, 0.30, 0.41]),
            _snapshot(rejects=[0.10], accepts=[0.30]),
        ],
    )
    summary = cd.summarize([path])
    # Anéis se repetem entre snapshots: deduplicados por valor.
    assert summary["coverage_reject"] == [0.05, 0.10]
    assert summary["coverage_accept"] == [0.30, 0.32, 0.41]
    sug = cd.suggest(summary)
    floor = sug["coverage_iou_floor"]["value"]
    assert floor == pytest.approx(0.2)


def test_sugere_veto_sobre_o_maior_logit_de_mesa_vazia(tmp_path):
    """Regra de referência: -6.98 na mesa vazia → veto -3.0.
    Aqui a mesa nova chega a -5.0 → veto -1.0, e nenhuma folha
    observada cai sob ele."""
    path = _write_telemetry(
        tmp_path,
        [
            _acq("veto", -5.0),
            _acq("veto", -8.2),
            _acq("neural", 2.1),
            _acq("classic", -0.4),
            _acq("presence_only", 1.0),
        ],
    )
    summary = cd.summarize([path])
    assert summary["presence_empty"] == [-5.0, -8.2]
    assert summary["presence_sheet"] == [2.1, -0.4, 1.0]
    sug = cd.suggest(summary)["presence_veto_logit"]
    assert sug["value"] == pytest.approx(-1.0)
    assert sug["sheets_vetoed"] == 0


def test_folha_sob_o_veto_sugerido_anula_a_sugestao(tmp_path):
    """Uma folha real com logit fraco cairia sob o veto da regra:
    o veto não separa nesta mesa — reportar a sobreposição com a
    contagem, não entregar um limiar que mata folha real."""
    path = _write_telemetry(
        tmp_path,
        [
            _acq("veto", -6.0),
            _acq("none", -4.0),  # mesa vazia sem veto: logit fraco
            _acq("classic", -2.5),  # folha real de logit fraco
            _acq("neural", 3.0),
        ],
    )
    summary = cd.summarize([path])
    assert summary["presence_empty"] == [-6.0, -4.0]
    sug = cd.suggest(summary)["presence_veto_logit"]
    # -4.0 + 4.0 = 0.0 vetaria a folha a -2.5 → sem sugestão.
    assert sug["value"] is None
    assert sug["sheets_vetoed"] == 1
    assert "não separa" in sug["evidence"]


def test_sem_um_lado_nao_ha_sugestao(tmp_path):
    """Só mesa com folha (nenhum fantasma rejeitado): piso fica sem
    sugestão — metade da distribuição não basta para posicioná-lo."""
    path = _write_telemetry(tmp_path, [_snapshot(accepts=[0.4, 0.5])])
    sug = cd.suggest(cd.summarize([path]))
    assert sug["coverage_iou_floor"]["value"] is None
    assert "rejeitados" in sug["coverage_iou_floor"]["evidence"]


def test_distribuicoes_sobrepostas_nao_separam_piso(tmp_path):
    """Fantasma acima da folha mais baixa: o ponto médio não separa
    — reportar a sobreposição em vez de inventar um limiar."""
    path = _write_telemetry(tmp_path, [_snapshot(rejects=[0.35], accepts=[0.30, 0.6])])
    sug = cd.suggest(cd.summarize([path]))
    assert sug["coverage_iou_floor"]["value"] is None
    assert "sobrepõem" in sug["coverage_iou_floor"]["evidence"]


def test_fit_reject_e_pixel_scale_entram_na_sumarizacao(tmp_path):
    path = _write_telemetry(
        tmp_path,
        [
            {
                **_acq(
                    "veto",
                    -9.0,
                    fit_reject={"reason": "min_area", "area_ratio": 0.008},
                ),
                "tracker_after": {"pixel_scale": 1.5},
            },
            _acq("none", None, {"reason": "unsupported"}),
        ],
    )
    summary = cd.summarize([path])
    assert summary["fit_reject_reasons"] == {"min_area": 1, "unsupported": 1}
    assert summary["fit_reject_areas"] == [0.008]
    assert summary["pixel_scales"] == [1.5]
    assert "min_area" in cd.suggest(summary)["min_contour_area_ratio"]["evidence"]


def test_main_emite_relatorio_json(tmp_path, capsys):
    path = _write_telemetry(
        tmp_path,
        [_acq("veto", -7.0), _snapshot(rejects=[0.05], accepts=[0.35])],
    )
    assert cd.main([str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["acquisitions"] == 1
    assert report["snapshots"] == 1
    sugs = report["suggestions"]
    assert sugs["presence_veto_logit"]["value"] == pytest.approx(-3.0)
    assert sugs["coverage_iou_floor"]["value"] == pytest.approx(0.2)


def test_main_aceita_diretorio_de_sessao(tmp_path, capsys):
    session = tmp_path / "sessao-nova"
    session.mkdir()
    (session / "telemetry.jsonl").write_text(
        json.dumps(_acq("veto", -4.0)) + "\n", encoding="utf-8"
    )
    assert cd.main([str(session)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["suggestions"]["presence_veto_logit"]["value"] == pytest.approx(0.0)


def test_linhas_quebradas_nao_derrubam(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        '{"event": "acquisition", "source": "veto", "presence_logit": -6.0}\n'
        "{linha truncada\n"
        "\n"
        '{"coverage_iou_reject_recent": [0.07]}\n',
        encoding="utf-8",
    )
    summary = cd.summarize([path])
    assert summary["presence_empty"] == [-6.0]
    assert summary["coverage_reject"] == [0.07]
