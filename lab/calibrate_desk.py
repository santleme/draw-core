"""Recalibra limiares sem denominador natural a partir de telemetria.

`coverage_iou_floor` e `PRESENCE_VETO_LOGIT` foram calibrados na
mesa de referência e não escalam com a resolução (issue #105). Este
script lê o telemetry.jsonl de sessões capturadas na mesa nova e
reaplica a MESMA regra da calibração original:

- coverage_iou_floor: ponto médio entre o maior IoU de fantasma
  (mesa vazia) e o menor IoU de folha real. Referência:
  fantasma <= 0.108, folha >= 0.298 -> 0.2.
- PRESENCE_VETO_LOGIT: maior logit de presença medido em cena sem
  folha + 4.0 de margem. Referência: -6.98 + 4.0 -> -3.0.

Os lados vêm de RÓTULOS de captura, não do desfecho do detector:
--empty declara "esta captura é mesa vazia" e --sheet "esta tem
folha". É o operador quem sabe o que a câmera viu; os desfechos
(veto, classic, neural...) ficam no relatório como evidência
diagnóstica. Sem rótulos, a única separação disponível seria o
próprio limiar medido — os anéis de IoU são censurados por ele
(rejeitados < piso <= aceitos, sempre "separados" por construção)
e uma folha perdida contaminaria o lado vazio: é a contaminação
que o rótulo elimina.

A SessionTelemetry faz APPEND no telemetry.jsonl do save-dir:
grave cada cena num --save-dir novo — um arquivo com duas sessões
é recusado, porque o rótulo se aplica ao arquivo inteiro e
sessões misturadas misturariam os lados da calibração.

Uso:
  python -m lab.calibrate_desk --empty cap_mesa_vazia/ --sheet cap_folha/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[1]
    for _p in (
        str(_repo_root / "packages" / "draw-dral" / "src"),
        str(_repo_root / "src"),
        str(_repo_root),
    ):
        if _p not in sys.path:
            sys.path.insert(0, _p)

# Fontes que o veto de presença alcança: a detecção neural devolve
# cantos ANTES do check do logit (app.py) e é imune a ele — numa
# captura rotulada como com-folha, só classic/presence_only/veto/
# none estão sob o veto.
_VETOABLE_SOURCES = {"classic", "presence_only", "veto", "none"}
# Razões de rejeição cujo area_ratio é razão de contorno — a
# métrica do _MIN_CONTOUR_AREA_RATIO. low_presence carrega fração
# de foreground da máscara (métrica diferente) e ficaria misturada.
_CONTOUR_AREA_REASONS = {"min_area", "unsupported", "fit_failed", "implausible"}


class MixedSessionError(RuntimeError):
    """Stream que não prova uma sessão única e rotulável.

    A SessionTelemetry faz append: regravar no mesmo save-dir junta
    duas cenas (segundo session_open), e rotações além da primeira
    descartam o início da sessão — o stream fica sem cabeçalho. Nos
    dois casos o rótulo --empty/--sheet deixaria de valer para a
    captura inteira. Grave cada cena num --save-dir novo.
    """


def _iter_records(paths):
    """Abre cada telemetry.jsonl e devolve os dicts na ordem.

    Diretórios também valem: procura <dir>/telemetry.jsonl. Linha
    quebrada não derruba a leitura — só é pulada. A rotação por
    tamanho move o pedaço antigo para <nome>.1 e continua no mesmo
    arquivo: quando o .1 existe ele é encadeado ANTES do ativo, na
    ordem cronológica — assim uma sessão única rotacionada no meio
    (até num --save-dir novo) volta a ser um stream completo.

    O primeiro registro do stream PRECISA ser `session_open` (senão
    o início da sessão ficou numa rotação ainda mais antiga, já
    perdida) e único (senão o append juntou duas cenas): ambos
    sobem MixedSessionError em vez de calibrar com cena errada.
    """
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            path = path / "telemetry.jsonl"
        files = [path]
        rotated = path.with_name(path.name + ".1")
        if rotated.exists():
            files.insert(0, rotated)
        sessions = 0
        for file in files:
            with file.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") == "session_open":
                        sessions += 1
                        if sessions > 1:
                            raise MixedSessionError(
                                f"{path}: segundo session_open — duas "
                                "sessões no mesmo arquivo"
                            )
                    elif sessions == 0:
                        raise MixedSessionError(
                            f"{path}: registros antes do primeiro "
                            "session_open — o início da sessão ficou "
                            "numa rotação anterior, já descartada"
                        )
                    yield rec


def _summarize_side(paths, *, sheet):
    """Amostras de UMA captura sob o rótulo declarado pelo operador.

    Os dois anéis de IoU (rejects+accepts) entram juntos: a divisão
    entre eles é o piso VIGENTE, não a cena — um fantasma que o piso
    velho deixou passar continua sendo fantasma. `presence_sheet`
    reúne os logits das fontes que o veto alcança; `presence_neural`
    fica à parte porque é imune ao veto.
    """
    side = {
        "coverage": set(),
        "presence": [],
        "presence_neural": [],
        "source_counts": {},
        "fit_reject_reasons": {},
        "fit_reject_areas": [],
        "pixel_scales": set(),
        "acquisitions": 0,
        "snapshots": 0,
    }
    for rec in _iter_records(paths):
        if rec.get("event") == "acquisition":
            side["acquisitions"] += 1
            source = rec.get("source")
            if source:
                side["source_counts"][source] = side["source_counts"].get(source, 0) + 1
            logit = rec.get("presence_logit")
            if isinstance(logit, (int, float)):
                if sheet and source == "neural":
                    side["presence_neural"].append(float(logit))
                elif sheet or source in _VETOABLE_SOURCES:
                    # Na mesa rotulada com folha, veto/none também
                    # são amostras de folha — folhas perdidas pelo
                    # detector, exatamente as que a calibração busca.
                    side["presence"].append(float(logit))
            fit = rec.get("fit_reject")
            if isinstance(fit, dict):
                reason = fit.get("reason")
                if reason:
                    side["fit_reject_reasons"][reason] = (
                        side["fit_reject_reasons"].get(reason, 0) + 1
                    )
                area = fit.get("area_ratio")
                if reason in _CONTOUR_AREA_REASONS and isinstance(area, (int, float)):
                    side["fit_reject_areas"].append(float(area))
            for diag in (rec.get("tracker_before"), rec.get("tracker_after")):
                scale = (diag or {}).get("pixel_scale")
                if isinstance(scale, (int, float)):
                    side["pixel_scales"].add(round(float(scale), 6))
        else:
            side["snapshots"] += 1
            for recent in rec.get("coverage_iou_reject_recent") or []:
                side["coverage"].add(round(float(recent), 4))
            for recent in rec.get("coverage_iou_accept_recent") or []:
                side["coverage"].add(round(float(recent), 4))
    return side


def summarize(empty_paths, sheet_paths):
    """Sumariza as capturas rotuladas por lado da calibração."""
    return {
        "empty": _summarize_side(empty_paths, sheet=False),
        "sheet": _summarize_side(sheet_paths, sheet=True),
    }


def suggest(summary, margin=4.0):
    """Converte a sumarização em limiares sugeridos + evidência.

    Regras idênticas às da calibração original da mesa de
    referência; quando um lado da distribuição não foi capturado,
    a sugestão é None com o motivo — calibrar com metade do dado é
    o chute que a issue quer evitar.
    """
    out = {}

    ghosts = sorted(summary["empty"]["coverage"])
    sheets = sorted(summary["sheet"]["coverage"])
    if ghosts and sheets:
        high_ghost, low_sheet = ghosts[-1], sheets[0]
        if high_ghost < low_sheet:
            # Sem arredondar: o gate rejeita iou < piso ESTRITO e o
            # ponto médio de amostras de 4 casas tem 5 — round(,4)
            # colapsaria sobre o fantasma e o piso deixaria de
            # separar (revisão do PR #115).
            out["coverage_iou_floor"] = {
                "value": (high_ghost + low_sheet) / 2,
                "evidence": (
                    f"maior IoU de mesa vazia {high_ghost} < menor IoU "
                    f"com folha {low_sheet} (regra do ponto médio)"
                ),
            }
        else:
            out["coverage_iou_floor"] = {
                "value": None,
                "evidence": (
                    f"distribuições se sobrepõem: mesa vazia até "
                    f"{high_ghost}, folha desde {low_sheet} — o "
                    f"piso não separa nesta mesa"
                ),
            }
    else:
        missing = "mesa vazia (--empty)" if not ghosts else "com folha (--sheet)"
        out["coverage_iou_floor"] = {
            "value": None,
            "evidence": f"sem amostras de {missing}: capture os dois cenários",
        }

    empty_logits = summary["empty"]["presence"]
    sheet_logits = summary["sheet"]["presence"]
    if empty_logits and sheet_logits:
        # Também sem arredondar: um veto arredondado para cima
        # poderia alcançar uma folha que o valor exato pouparia.
        veto = max(empty_logits) + margin
        # Folhas cuja presença cairia sob o veto sugerido: cada uma
        # é uma folha real que o veto mataria nesta mesa — logo a
        # regra não separa aqui e a sugestão honesta é None. As
        # detecções neurais são imunes ao veto e ficaram à parte.
        sheets_below = sum(1 for logit in sheet_logits if logit <= veto)
        if sheets_below:
            out["presence_veto_logit"] = {
                "value": None,
                "evidence": (
                    f"maior logit sem folha {max(empty_logits)} + margem "
                    f"{margin} = {veto}, mas {sheets_below} aquisição(ões) "
                    f"com folha (menor logit {min(sheet_logits)}) cairiam "
                    f"sob ele — veto não separa nesta mesa"
                ),
                "sheets_vetoed": sheets_below,
            }
        else:
            out["presence_veto_logit"] = {
                "value": veto,
                "evidence": (
                    f"maior logit sem folha {max(empty_logits)} + margem "
                    f"{margin}; nenhuma aquisição com folha cai sob o veto"
                ),
                "sheets_vetoed": 0,
            }
    else:
        if not empty_logits:
            neural_empty = summary["empty"]["source_counts"].get("neural", 0)
            if neural_empty:
                # Detecções neurais não passam pelo caminho do veto —
                # faltam as amostras de mesa vazia que ele alcança.
                missing = (
                    f"mesa vazia só vista por detecções neurais "
                    f"({neural_empty} aquisições), que o veto não "
                    f"alcança — faltam amostras do lado vetável"
                )
            else:
                missing = "sem aquisições de mesa vazia (--empty)"
        elif summary["sheet"]["presence_neural"]:
            missing = (
                f"folhas só detectadas pela neural "
                f"({len(summary['sheet']['presence_neural'])} aquisições), "
                f"imune ao veto — faltam amostras do lado vetável"
            )
        else:
            missing = "sem aquisições com folha (--sheet)"
        out["presence_veto_logit"] = {
            "value": None,
            "evidence": f"{missing} — o veto precisa dos dois lados para ser validado",
        }

    reasons = {}
    areas = []
    for side in ("empty", "sheet"):
        for reason, count in summary[side]["fit_reject_reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + count
        areas += summary[side]["fit_reject_areas"]
    out["min_contour_area_ratio"] = {
        "value": None,
        "evidence": (
            f"{reasons.get('min_area', 0)} rejeição(ões) min_area; "
            f"menores razões de contorno rejeitadas: "
            f"{sorted(areas)[:5]} — confirme visualmente se eram "
            f"folhas antes de baixar o piso"
        ),
    }
    out["pixel_scale"] = {
        "value": sorted(
            set(summary["empty"]["pixel_scales"])
            | set(summary["sheet"]["pixel_scales"])
        ),
        "evidence": "escalas de pixel vistas pelo tracker nestas capturas",
    }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sugere limiares sem denominador a partir da "
        "telemetria de capturas ROTULADAS em outra mesa (issue #105)."
    )
    parser.add_argument(
        "--empty",
        nargs="+",
        default=[],
        metavar="CAP",
        help="telemetry.jsonl/diretório de captura de MESA VAZIA",
    )
    parser.add_argument(
        "--sheet",
        nargs="+",
        default=[],
        metavar="CAP",
        help="telemetry.jsonl/diretório de captura COM FOLHA",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=4.0,
        help="margem sobre o maior logit de mesa vazia (default 4.0, "
        "a mesma da calibração de referência)",
    )
    args = parser.parse_args(argv)
    if not args.empty and not args.sheet:
        parser.error("rotule as capturas: --empty e/ou --sheet são obrigatórios")

    try:
        summary = summarize(args.empty, args.sheet)
    except MixedSessionError as exc:
        print(
            f"erro: {exc}\n"
            "  grave cada cena num --save-dir novo (a SessionTelemetry "
            "faz append e a rotação move o arquivo cheio para .1)",
            file=sys.stderr,
        )
        return 2
    report = {
        "inputs": {
            "empty": [str(p) for p in args.empty],
            "sheet": [str(p) for p in args.sheet],
        },
        "empty": {
            "acquisitions": summary["empty"]["acquisitions"],
            "snapshots": summary["empty"]["snapshots"],
            "source_counts": summary["empty"]["source_counts"],
            "fit_reject_reasons": summary["empty"]["fit_reject_reasons"],
        },
        "sheet": {
            "acquisitions": summary["sheet"]["acquisitions"],
            "snapshots": summary["sheet"]["snapshots"],
            "source_counts": summary["sheet"]["source_counts"],
            "fit_reject_reasons": summary["sheet"]["fit_reject_reasons"],
        },
        "suggestions": suggest(summary, margin=args.margin),
        "caveats": [
            "os lados seguem os rótulos declarados (--empty/--sheet); "
            "um rótulo errado contamina a calibração — confira nas "
            "pranchas da captura (lab/desk_capture) antes de aplicar",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
