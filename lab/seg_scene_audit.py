"""Auditoria de cena da segmentação: as medidas da issue #106 por frame.

Fora da mesa de referência o modelo rotula fundo claro como papel — na
captura frames_s3_geometria_1117_1754 (640x480, câmera oblíqua, piso
claro, pessoa à vista) 13 dos 36 frames produziram quads cuja região
não-fundo cobria só 56-80% do quad (folha real >= 0.84), quads a
96-100% da área do frame ou extrapolando 80-100 px além das bordas.
A proposta da issue é coletar .drawcap em outras mesas e entrar essas
cenas num finetune; esta ferramenta é o instrumento da medida — reproduz
a falha numa captura qualquer hoje e aceita o modelo retreinado amanhã,
com o mesmo relatório.

Cada quadro da fonte passa pelo detector de folha e vira uma linha JSON
em frames.jsonl: presença, veredito, cobertura não-fundo, componentes
grandes, fração de borda ocupada, mão e — quando o quadrilátero fecha —
área relativa do quad, fill (a fração do quad que a cobertura preenche,
medida na grade nativa da máscara) e extrapolação dos cantos em pixels.
Ao fim, audit.json resume a corrida com os contadores das três famílias
de falha da issue: fill abaixo de FILL_BELOW (o piso 0.82 das guardas
geométricas do PR #107), área do quad >= QUAD_AREA_GE do frame e canto
>= OOB_GE_PX além da borda. "interrupted" marca Ctrl+C no meio da
corrida e "source_error" registra a fonte que terminou por erro (um
.drawcap truncado não passa por completo). --dump-overlays grava
overlay_NNNNN.png com a cobertura crua em tinta translúcida e o quad
desenhado, para conferir olho a olho os frames que dispararam os
contadores.

A fonte é um diretório de frames (*.png, *.jpg em ordem de nome) ou
qualquer fonte que draw.camera.parse_source/open_capture aceitem
(.drawcap, vídeo, tcp://, hvcam://). Um --out que já contém artefatos
de auditoria anterior é recusado — corridas não se misturam e a
evidência anterior não é apagada.

Uso:
  python -m lab.seg_scene_audit --source frames_s3_geometria_1117_1754 \
      --model models/draw_seg_v3_dyn.onnx --out /tmp/audit
  python -m lab.seg_scene_audit --source sessao.drawcap \
      --model m.onnx --out /tmp/audit --dump-overlays
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Limiares das famílias de falha, na medida que a issue citou: o piso
# de fill das guardas geométricas (PR #107 — a folha real mede >=
# 0.84), a fração de frame que os piores quads cobriram (96-100%) e o
# chão da extrapolação medida (80-100 px; 40 acende o alarme cedo, sem
# esperar o caso cheio).
FILL_BELOW = 0.82
QUAD_AREA_GE = 0.96
OOB_GE_PX = 40.0

# Piso de área para um componente contar: 0.5% do frame, medido na
# grade da máscara de cobertura — a razão é a mesma nas duas grades.
_MIN_COMPONENT_FRAC = 0.005

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# Mesma política do _desk_artifacts: sufixo estritamente numérico —
# um overlay_1.png de outra ferramenta não colide com os nossos.
_AUDIT_OVERLAY_RE = re.compile(r"overlay_\d{5,}\.png")


def _write_png(path, image):
    """imwrite checado: False silencioso vira OSError nomeando o destino."""
    if not cv2.imwrite(str(path), image):
        raise OSError(f"falha ao gravar {path}")


def _component_areas(cov_bin):
    """Áreas em px das regiões conectadas (8) — os contornos externos.

    connectedComponentsWithStats devolve a área em pixels de cada
    região direto; o conjunto é o mesmo que RETR_EXTERNAL enumeraria
    (uma região com buraco continua sendo um contorno externo só).
    """
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cov_bin.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros(0, np.int64)
    return stats[1:, cv2.CC_STAT_AREA]


def _quad_oob_px(quad, width, height):
    """A maior distância de um canto além da borda do frame, >= 0.

    Para cada canto: max(-x, x-(w-1), -y, y-(h-1)) — canto dentro do
    quadro contribui <= 0, canto fora mede a extrapolação em pixels.
    Os 80-100 px da issue saíram desta conta sobre a captura.
    """
    oob = 0.0
    for x, y in quad:
        oob = max(oob, -x, x - (width - 1), -y, y - (height - 1))
    return float(oob)


def _quad_fill(quad, cov_bin, frame_shape):
    """Fração do quad preenchida por não-fundo, na grade da cobertura.

    Os cantos vivem em coordenadas do frame; a máscara de cobertura
    fica na resolução nativa do modelo — escalam-se os cantos para a
    grade dela e conta-se quanto do polígono é não-fundo. A mancha de
    falso-papel da issue descia este número: o quad abraçava piso e
    braço que a máscara não rotulou. None sem cobertura ou com o
    polígono fora da grade.
    """
    if cov_bin is None:
        return None
    mask_h, mask_w = cov_bin.shape
    frame_h, frame_w = frame_shape
    scaled = quad * np.array([mask_w / frame_w, mask_h / frame_h])
    inside = np.zeros((mask_h, mask_w), np.uint8)
    cv2.fillPoly(inside, [np.round(scaled).astype(np.int32)], 1)
    inside = inside.astype(bool)
    quad_px = int(np.count_nonzero(inside))
    if quad_px == 0:
        return None
    return float(np.count_nonzero(cov_bin & inside) / quad_px)


def _coverage_of(det):
    """A cobertura CRUA que o último predict mediu.

    NeuralSheetDetector publica duas: last_coverage_raw é o argmax do
    modelo como saiu — o mapa que _scene_stats mede e que esta
    auditoria reporta; last_coverage_mask é o mesmo mapa já com os
    buracos internos preenchidos para o voto de tinta — uma falha de
    segmentação cercada de papel some nele. Detectores e dublês
    antigos sem o atributo cru caem na máscara preenchida.
    """
    raw = getattr(det, "last_coverage_raw", None)
    if raw is not None:
        return raw
    return getattr(det, "last_coverage_mask", None)


def measure_frame(det, frame, corners, verdict, *, index, name):
    """As medidas da issue sobre um quadro já detectado.

    Lê as máscaras auxiliares que a chamada deixou no detector
    (last_coverage_raw/last_hand_mask, na resolução nativa do
    modelo): cobertura é "tudo que não é fundo" — a mesma região que
    a mancha de falso-papel inflava na captura da issue. Devolve o
    dict pronto para a linha JSONL; os campos de quad só existem
    quando o ajuste fechou.
    """
    height, width = frame.shape[:2]
    coverage = _coverage_of(det)
    hand = getattr(det, "last_hand_mask", None)
    record = {
        "frame": index,
        "file": name,
        "presence_logit": getattr(det, "last_presence_logit", None),
        "verdict": verdict,
        "coverage_ratio": None,
        "components": 0,
        "largest_share": None,
        "border_ratio": None,
        "hand_ratio": None,
    }
    cov_bin = None
    if coverage is not None:
        cov_bin = np.asarray(coverage) > 0
        total = cov_bin.size
        non_bg = int(np.count_nonzero(cov_bin))
        record["coverage_ratio"] = non_bg / total
        areas = _component_areas(cov_bin)
        record["components"] = int(
            np.count_nonzero(areas >= _MIN_COMPONENT_FRAC * total)
        )
        if non_bg:
            record["largest_share"] = float(areas.max() / non_bg)
        edge = np.concatenate(
            [cov_bin[0], cov_bin[-1], cov_bin[1:-1, 0], cov_bin[1:-1, -1]]
        )
        record["border_ratio"] = float(np.count_nonzero(edge) / edge.size)
    if hand is not None:
        hand_bin = np.asarray(hand) > 0
        record["hand_ratio"] = float(np.count_nonzero(hand_bin) / hand_bin.size)
    if corners is not None:
        quad = np.asarray(corners, np.float64).reshape(-1, 2)
        record["quad_area_ratio"] = float(
            cv2.contourArea(quad.astype(np.float32)) / (height * width)
        )
        record["quad_oob_px"] = _quad_oob_px(quad, width, height)
        record["quad_fill"] = _quad_fill(quad, cov_bin, (height, width))
    return record


def _num(value):
    return "-" if value is None else f"{value:.3f}"


def _log_line(record):
    return (
        f"f{record['frame']:5d} {record['file']} "
        f"verdict={record['verdict'] or '-'} "
        f"cov={_num(record['coverage_ratio'])} "
        f"comp={record['components']} "
        f"fill={_num(record.get('quad_fill'))} "
        f"oob={_num(record.get('quad_oob_px'))}"
    )


def _overlay_image(frame, coverage, corners):
    """Quadro + cobertura em tinta translúcida + quad desenhado.

    A cobertura sobe da grade nativa por NEAREST — a fronteira dura
    mostra exatamente a região que o modelo rotulou, sem anti-alias
    para esconder degraus.
    """
    vis = frame.copy()
    if coverage is not None:
        height, width = frame.shape[:2]
        up = (
            cv2.resize(
                np.asarray(coverage),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        tinted = vis.copy()
        tinted[up] = (40, 40, 220)  # avermelhado: o "não-fundo"
        vis = cv2.addWeighted(vis, 0.65, tinted, 0.35, 0)
    if corners is not None:
        poly = np.round(np.asarray(corners, np.float64).reshape(-1, 1, 2)).astype(
            np.int32
        )
        cv2.polylines(vis, [poly], True, (60, 220, 60), 2)
    return vis


def audit_frames(
    frames,
    det,
    *,
    max_frames=None,
    out_dir=None,
    dump_overlays=False,
    log=print,
    records=None,
):
    """Mede cada (nome, quadro) do iterável com o detector dado.

    O detector chega construído (sessão real ou dublê de teste): um
    callable quadro -> (cantos, veredito, máscara) com os atributos
    last_coverage_raw/last_hand_mask/last_presence_logit. Com
    out_dir, frames.jsonl é gravado linha a linha — uma corrida que
    morre no meio deixa a evidência parcial no disco. dump_overlays
    exige out_dir e grava overlay_NNNNN.png por quadro. Devolve a
    lista de registros (a mesma das linhas do JSONL); records, quando
    dada, é essa lista — uma interrupção no meio do laço deixa nela
    as linhas já medidas.

    max_frames limita o iterável ANTES do laço (islice): pedir N
    quadros nunca puxa o N+1 — uma fonte que quebra ao avançar além
    do pedido não derruba a corrida já gravada.
    """
    if dump_overlays and out_dir is None:
        raise ValueError("--dump-overlays precisa de out_dir")
    if records is None:
        records = []
    jsonl = None
    try:
        if out_dir is not None:
            jsonl = (Path(out_dir) / "frames.jsonl").open("w", encoding="utf-8")
        for index, (name, frame) in enumerate(itertools.islice(frames, max_frames)):
            corners, verdict, _mask = det(frame, return_mask=True)
            record = measure_frame(det, frame, corners, verdict, index=index, name=name)
            records.append(record)
            if jsonl is not None:
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl.flush()
            if dump_overlays:
                _write_png(
                    Path(out_dir) / f"overlay_{index:05d}.png",
                    _overlay_image(frame, _coverage_of(det), corners),
                )
            log(_log_line(record))
    finally:
        if jsonl is not None:
            jsonl.close()
    return records


def summarize(
    records,
    *,
    model=None,
    fill_below=FILL_BELOW,
    quad_area_ge=QUAD_AREA_GE,
    oob_ge_px=OOB_GE_PX,
):
    """O audit.json: contagens e percentis sobre as linhas de frame.

    Os contadores espelham as três famílias de falha da issue — fill
    abaixo do piso das guardas, quad cobrindo quase o frame todo,
    cantos extrapolando a borda — com os limiares registrados ao lado
    para o relatório se ler sozinho.
    """
    quads = [r for r in records if r.get("quad_area_ratio") is not None]
    fills = [r["quad_fill"] for r in quads if r.get("quad_fill") is not None]
    ratios = [
        r["coverage_ratio"] for r in records if r.get("coverage_ratio") is not None
    ]
    return {
        "model": None if model is None else str(model),
        "frames": len(records),
        "frames_with_quad": len(quads),
        "coverage_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "quad_fill_p5": float(np.percentile(fills, 5)) if fills else None,
        "quad_fill_median": float(np.median(fills)) if fills else None,
        "thresholds": {
            "fill_below": fill_below,
            "quad_area_ge": quad_area_ge,
            "oob_ge_px": oob_ge_px,
        },
        "failures": {
            "fill_below": int(sum(1 for f in fills if f < fill_below)),
            "quad_area_ge": int(
                sum(1 for r in quads if r["quad_area_ratio"] >= quad_area_ge)
            ),
            "oob_ge_px": int(sum(1 for r in quads if r["quad_oob_px"] >= oob_ge_px)),
        },
    }


@contextlib.contextmanager
def _open_frames(source, *, helper_token_file=None):
    """Iterável de (nome, quadro) sobre o --source, fechando a fonte.

    Produz (iterável, info): info["source_error"] registra COMO a
    fonte terminou quando o iterável foi consumido até o fim — o
    last_error da captura separa o registro truncado do EOF limpo
    (os dois chegam como o mesmo read()==False); None é fim limpo
    ou fonte sem a informação.

    Diretório: os *.png/*.jpg em ordem de nome, decodificados por
    cv2.imread — um arquivo indecodificável aborta a corrida com o
    nome dele, nunca vira quadro mudo. Qualquer outra fonte passa por
    parse_source/open_capture (.drawcap, vídeo, tcp://, hvcam://) e
    entrega quadros seq_NNNNN até o read() fechar.
    """
    info = {"source_error": None}
    path = Path(str(source))
    if path.is_dir():
        names = sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not names:
            raise ValueError(f"{path} nao contem frames (*.png, *.jpg)")

        def _iter_dir():
            for item in names:
                image = cv2.imread(str(item))
                if image is None:
                    raise ValueError(f"nao foi possivel decodificar {item}")
                yield item.name, image

        yield _iter_dir(), info
        return

    from draw.camera import open_capture, parse_source

    capture = open_capture(parse_source(source), helper_token_file=helper_token_file)
    try:

        def _iter_capture():
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    info["source_error"] = getattr(capture, "last_error", None)
                    break
                yield f"seq_{index:05d}", frame
                index += 1

        yield _iter_capture(), info
    finally:
        capture.release()


def _audit_artifacts(out_dir):
    """Algum arquivo que uma corrida anterior da auditoria deixou.

    Confere NOMES via iterdir, não exists(): um symlink pendurado
    chamado audit.json tem exists()==False e, sem a guarda, a escrita
    do resumo seria redirecionada para fora do --out.
    """
    names = {path.name for path in out_dir.iterdir()}
    return (
        "audit.json" in names
        or "frames.jsonl" in names
        or any(_AUDIT_OVERLAY_RE.fullmatch(name) for name in names)
    )


def run(
    source,
    det,
    *,
    out_dir,
    max_frames=None,
    dump_overlays=False,
    model=None,
    helper_token_file=None,
    log=print,
    fill_below=FILL_BELOW,
    quad_area_ge=QUAD_AREA_GE,
    oob_ge_px=OOB_GE_PX,
):
    """Auditoria completa: fonte -> frames.jsonl + audit.json.

    O detector chega construído; se ele expõe load() (o neural expõe),
    a carga acontece aqui dentro, depois da recusa de --out sujo —
    relatorio antigo se recusa sem tocar em modelo nem fonte. Devolve
    o resumo escrito em audit.json.

    Dois desfechos escrevem resumo incompleto em vez de nenhum:
    Ctrl+C no meio da corrida ("interrupted": true, as linhas já
    medidas entram na conta — mesma política do desk_capture) e a
    fonte que termina por erro ("source_error": o last_error da
    captura — um .drawcap truncado não passa por captura completa).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if _audit_artifacts(out_dir):
        raise ValueError(
            f"{out_dir} ja contem artefatos de auditoria anterior; "
            "escolha outro --out (a evidencia anterior nao e apagada)"
        )
    load = getattr(det, "load", None)
    if load is not None:
        load()
    if model is None:
        model = getattr(det, "model_path", None)
    records = []
    interrupted = False
    source_error = None
    try:
        with _open_frames(source, helper_token_file=helper_token_file) as (
            frames,
            info,
        ):
            audit_frames(
                frames,
                det,
                max_frames=max_frames,
                out_dir=out_dir,
                dump_overlays=dump_overlays,
                log=log,
                records=records,
            )
            source_error = info["source_error"]
    except KeyboardInterrupt:
        interrupted = True
    summary = summarize(
        records,
        model=model,
        fill_below=fill_below,
        quad_area_ge=quad_area_ge,
        oob_ge_px=oob_ge_px,
    )
    summary["interrupted"] = interrupted
    summary["source_error"] = None if source_error is None else str(source_error)
    serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "audit.json").write_text(serialized, encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="diretorio de frames, .drawcap, video, tcp:// ou hvcam://",
    )
    parser.add_argument(
        "--model", required=True, help="caminho do .onnx de segmentacao"
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="diretorio de saida (recusado se ja tiver auditoria)",
    )
    parser.add_argument(
        "--max-frames", type=int, help="encerra a auditoria apos N quadros"
    )
    parser.add_argument(
        "--dump-overlays",
        action="store_true",
        help="grava overlay_NNNNN.png por quadro",
    )
    parser.add_argument(
        "--helper-token-file", type=Path, help="token do ajudante (fontes tcp://)"
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="desliga a verificacao sha256 do modelo",
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames precisa ser >= 1")

    from draw.infra.manifest import verify_model
    from draw.perception import NeuralSheetDetector

    det = NeuralSheetDetector(
        args.model, verify=None if args.allow_unverified else verify_model
    )
    try:
        summary = run(
            args.source,
            det,
            out_dir=args.out,
            max_frames=args.max_frames,
            dump_overlays=args.dump_overlays,
            model=str(args.model),
            helper_token_file=args.helper_token_file,
        )
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as error:
        parser.exit(2, f"erro: {error}\n")
    if summary["interrupted"]:
        # O resumo parcial ja esta em audit.json; a saida diz que a
        # auditoria nao cobriu a captura toda (130 = 128+SIGINT).
        print(
            f"interrompido: resumo parcial de {summary['frames']} "
            f"quadros gravado em {args.out}"
        )
        return 130
    failures = summary["failures"]
    thresholds = summary["thresholds"]
    print(
        f"fim: {summary['frames']} quadros, {summary['frames_with_quad']} "
        f"com quad; fill<{thresholds['fill_below']}: "
        f"{failures['fill_below']}, area>={thresholds['quad_area_ge']}: "
        f"{failures['quad_area_ge']}, oob>={thresholds['oob_ge_px']}px: "
        f"{failures['oob_ge_px']} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
