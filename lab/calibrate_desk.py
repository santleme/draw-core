"""Recalibra limiares sem denominador natural a partir de telemetria.

`coverage_iou_floor` e `PRESENCE_VETO_LOGIT` foram calibrados na
mesa de referência e não escalam com a resolução (issue #105). Este
script lê o telemetry.jsonl de uma ou mais sessões capturadas na
mesa nova e aplica a MESMA regra da calibração original sobre os
valores observados:

- coverage_iou_floor: ponto médio entre o maior IoU rejeitado
  (fantasma) e o menor IoU aceito (folha real). Referência:
  fantasma <= 0.108, folha >= 0.298 -> 0.2.
- PRESENCE_VETO_LOGIT: maior logit de presença medido em aquisição
  sem folha + 4.0 de margem. Referência: -6.98 + 4.0 -> -3.0.

As saídas são SUGESTÕES com a evidência ao lado: cabe a quem opera
confirmar que os rejeitados eram mesmo fantasmas e que a folha da
mesa nova não cai na faixa do veto.

Uso:
  python -m lab.calibrate_desk captures/telemetry.jsonl [mais.jsonl ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Fontes de aquisição que significam "a folha estava lá" —
# neural/classic fecharam cantos, presence_only viu papel sem quad.
_SHEET_SOURCES = {"neural", "classic", "presence_only"}
_EMPTY_SOURCES = {"veto", "none"}


def _iter_records(paths):
    """Abre cada telemetry.jsonl e devolve os dicts na ordem.

    Diretórios também valem: procura <dir>/telemetry.jsonl. Linha
    quebrada não derruba a leitura — só é pulada.
    """
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            path = path / "telemetry.jsonl"
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def summarize(paths):
    """Separa as amostras da telemetria por lado da calibração.

    As amostras de IoU vêm dos snapshots de stats (rejects e
    accepts ficam em anel de 32 por linha — juntados, repetem-se
    entre snapshots e por isso vão a conjuntos deduplicados por
    valor). Os logits vêm dos eventos de aquisição: `presence_logit`
    classificado pelo desfecho da detecção.
    """
    summary = {
        "coverage_reject": [],
        "coverage_accept": [],
        "presence_empty": [],
        "presence_sheet": [],
        "fit_reject_reasons": {},
        "fit_reject_areas": [],
        "pixel_scales": set(),
        "acquisitions": 0,
        "snapshots": 0,
    }
    cov_rej, cov_acc = set(), set()
    for rec in _iter_records(paths):
        if rec.get("event") == "acquisition":
            summary["acquisitions"] += 1
            logit = rec.get("presence_logit")
            source = rec.get("source")
            if isinstance(logit, (int, float)):
                if source in _SHEET_SOURCES:
                    summary["presence_sheet"].append(float(logit))
                elif source in _EMPTY_SOURCES:
                    summary["presence_empty"].append(float(logit))
            fit = rec.get("fit_reject")
            if isinstance(fit, dict):
                reason = fit.get("reason")
                if reason:
                    summary["fit_reject_reasons"][reason] = (
                        summary["fit_reject_reasons"].get(reason, 0) + 1
                    )
                area = fit.get("area_ratio")
                if isinstance(area, (int, float)):
                    summary["fit_reject_areas"].append(float(area))
            for side in ("tracker_before", "tracker_after"):
                diag = rec.get(side) or {}
                scale = diag.get("pixel_scale")
                if isinstance(scale, (int, float)):
                    summary["pixel_scales"].add(round(float(scale), 6))
        else:
            summary["snapshots"] += 1
            for recent in rec.get("coverage_iou_reject_recent") or []:
                cov_rej.add(round(float(recent), 4))
            for recent in rec.get("coverage_iou_accept_recent") or []:
                cov_acc.add(round(float(recent), 4))
    summary["coverage_reject"] = sorted(cov_rej)
    summary["coverage_accept"] = sorted(cov_acc)
    summary["pixel_scales"] = sorted(summary["pixel_scales"])
    return summary


def suggest(summary, margin=4.0):
    """Converte a sumarização em limiares sugeridos + evidência.

    Regras idênticas às da calibração original da mesa de
    referência; quando um lado da distribuição não apareceu na
    captura, a sugestão é None com o motivo — calibrar sem os dois
    lados é o chute que a issue quer evitar.
    """
    out = {}

    rej, acc = summary["coverage_reject"], summary["coverage_accept"]
    if rej and acc:
        high_ghost, low_sheet = rej[-1], acc[0]
        if high_ghost < low_sheet:
            out["coverage_iou_floor"] = {
                "value": round((high_ghost + low_sheet) / 2, 4),
                "evidence": (
                    f"maior IoU rejeitado {high_ghost} < menor IoU "
                    f"aceito {low_sheet} (regra do ponto médio)"
                ),
            }
        else:
            out["coverage_iou_floor"] = {
                "value": None,
                "evidence": (
                    f"distribuições se sobrepõem: rejeitado até "
                    f"{high_ghost}, aceito desde {low_sheet} — o "
                    f"piso não separa nesta mesa"
                ),
            }
    else:
        missing = "rejeitados" if not rej else "aceitos"
        out["coverage_iou_floor"] = {
            "value": None,
            "evidence": f"sem amostras {missing}: capture a mesa vazia e com folha",
        }

    empty = summary["presence_empty"]
    if empty:
        veto = round(max(empty) + margin, 4)
        # Folhas cuja presença cairia sob o veto sugerido: cada uma
        # é uma folha real que o veto mataria nesta mesa — logo a
        # regra não separa aqui e a sugestão honesta é None.
        sheets_below = sum(1 for logit in summary["presence_sheet"] if logit <= veto)
        if sheets_below:
            out["presence_veto_logit"] = {
                "value": None,
                "evidence": (
                    f"maior logit sem folha {max(empty)} + margem {margin} "
                    f"= {veto}, mas {sheets_below} aquisição(ões) com folha "
                    f"(menor logit {min(summary['presence_sheet'])}) cairiam "
                    f"sob ele — veto não separa nesta mesa"
                ),
                "sheets_vetoed": sheets_below,
            }
        else:
            out["presence_veto_logit"] = {
                "value": veto,
                "evidence": (
                    f"maior logit sem folha {max(empty)} + margem {margin}; "
                    f"nenhuma aquisição com folha cai sob o veto"
                ),
                "sheets_vetoed": 0,
            }
    else:
        out["presence_veto_logit"] = {
            "value": None,
            "evidence": "sem aquisições de mesa vazia na captura",
        }

    out["min_contour_area_ratio"] = {
        "value": None,
        "evidence": (
            f"{summary['fit_reject_reasons'].get('min_area', 0)} "
            f"rejeição(ões) min_area; menores razões de área "
            f"rejeitadas: {sorted(summary['fit_reject_areas'])[:5]} — "
            f"confirme visualmente se eram folhas antes de baixar o piso"
        ),
    }
    out["pixel_scale"] = {
        "value": summary["pixel_scales"],
        "evidence": "escalas de pixel vistas pelo tracker nesta captura",
    }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sugere limiares sem denominador a partir da "
        "telemetria de capturas em outra mesa (issue #105)."
    )
    parser.add_argument(
        "telemetry",
        nargs="+",
        help="telemetry.jsonl ou diretório de sessão que o contém",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=4.0,
        help="margem sobre o maior logit de mesa vazia (default 4.0, "
        "a mesma da calibração de referência)",
    )
    args = parser.parse_args(argv)

    summary = summarize(args.telemetry)
    report = {
        "inputs": [str(p) for p in args.telemetry],
        "acquisitions": summary["acquisitions"],
        "snapshots": summary["snapshots"],
        "fit_reject_reasons": summary["fit_reject_reasons"],
        "suggestions": suggest(summary, margin=args.margin),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
