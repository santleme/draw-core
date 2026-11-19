"""Diagnóstico de captura ao vivo: o que o pipeline enxerga, quadro a quadro.

Para cada quadro (da câmera real ou de uma pasta de PNGs) grava:

  NNN_raw.png        quadro bruto (já rotacionado, se --rotate 180)
  NNN_seg.png        máscara do draw_seg sobreposta (papel verde, mão
                     vermelha) + quad ajustado em amarelo
  NNN_warp.png       crop retificado na grade do formato (o que o app usa)
  NNN_warp_aspect.png crop retificado preservando a proporção do quad
  NNN_prob.png       probabilidade do page_clean (0..255) no warp do app
  NNN_mask.png       máscara binária no limiar do manifesto
  NNN_mask_aspect.png idem sobre o warp com proporção preservada
  summary.json       luminância (global e por metade do quad), proporção
                     do quad, FPS de leitura e pixels de tinta por limiar

Exemplo:
  python lab/diag_live.py --source hvcam:// --frames 8 --out /tmp/diag
  python lab/diag_live.py --input pasta_de_pngs --out /tmp/diag
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from draw import config  # noqa: E402

# Segundos sem nenhuma leitura boa que declaram a fonte parada. O hvcam
# devolve False tanto no timeout transitório quanto no stream caído, então
# só o tempo sem quadro distingue uma coisa da outra.
_STALL_TIMEOUT_S = 10.0
_RETRY_SLEEP_S = 0.05
# Artefato por quadro: prefixo de TRÊS OU MAIS dígitos + "_" (diagnose_frame
# usa :03d, então rodadas de 1000+ quadros geram 4 dígitos).
_FRAME_ARTIFACT = re.compile(r"^\d{3,}_.+\.png$")
# Leitura contínua (sem diagnóstico no meio) que mede o FPS da fonte.
_FPS_BURST_S = 0.5

# Limiares comparados no relatório: o voto ao vivo (seen_threshold),
# o corte calibrado do v1 e o do v2.
_REPORT_THRESHOLDS = (0.5, 0.7, 0.85, 0.97)


def quad_aspect(corners):
    """Largura/altura média do quad (lados opostos), em px de câmera."""
    c = np.asarray(corners, np.float32).reshape(4, 2)
    w = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
    h = 0.5 * (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1]))
    return float(w / max(h, 1e-6))


def aspect_grid(corners, long_side=None):
    """(largura, altura) com a proporção do quad e o lado maior da grade."""
    long_side = long_side or max(config.PAPER_W, config.PAPER_H)
    aspect = quad_aspect(corners)
    if aspect >= 1.0:
        return long_side, max(1, int(round(long_side / aspect)))
    return max(1, int(round(long_side * aspect))), long_side


def warp_to(frame, corners, size):
    w, h = size
    src = np.asarray(corners, np.float32).reshape(4, 2)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (w, h))


def luminance_stats(frame, corners=None):
    """Luminância (0-255) global e, com quad, por metade esquerda/direita."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = {
        "frame_mean": float(gray.mean()),
        "frame_p05": float(np.percentile(gray, 5)),
        "frame_p95": float(np.percentile(gray, 95)),
    }
    if corners is not None:
        quad = np.zeros(gray.shape, np.uint8)
        c = np.round(np.asarray(corners, np.float32).reshape(4, 2)).astype(np.int32)
        cv2.fillConvexPoly(quad, c, 1)
        inside = quad.astype(bool)
        mid_x = float(c[:, 0].mean())
        xs = np.arange(gray.shape[1])[None, :]
        for name, side in (("left", xs < mid_x), ("right", xs >= mid_x)):
            region = inside & side
            if region.any():
                vals = gray[region]
                out[f"quad_{name}_mean"] = float(vals.mean())
                out[f"quad_{name}_p90"] = float(np.percentile(vals, 90))
    # Desequilíbrio de cor: canal médio / cinza (roxo = R e B acima de G).
    means = frame.reshape(-1, 3).mean(axis=0)
    out["bgr_mean"] = [float(v) for v in means]
    return out


def seg_overlay(frame, page_mask, hand_mask, corners):
    vis = frame.copy()
    if page_mask is not None:
        m = page_mask > 0
        vis[m] = (0.6 * vis[m] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
    if hand_mask is not None:
        m = hand_mask > 0
        vis[m] = (0.6 * vis[m] + 0.4 * np.array([0, 0, 220])).astype(np.uint8)
    if corners is not None:
        c = np.round(np.asarray(corners, np.float32).reshape(4, 2)).astype(np.int32)
        cv2.polylines(vis, [c], True, (0, 255, 255), 3)
    return vis


def _resize_to(mask, shape):
    h, w = shape[:2]
    if mask is None or mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


class SourceExhausted(RuntimeError):
    """A fonte ao vivo parou de entregar quadros antes do total pedido."""


def _write_png(path, image):
    """imwrite checado: False silencioso vira OSError nomeando o destino."""
    if not cv2.imwrite(str(path), image):
        raise OSError(f"falha ao gravar {path}")


def diagnose_frame(frame, detector, page_clean, out_dir, idx):
    """Roda detector + page_clean num quadro e grava os artefatos."""
    stem = f"{idx:03d}"
    _write_png(out_dir / f"{stem}_raw.png", frame)
    corners, verdict, page_mask = detector(frame, return_mask=True)
    hand = _resize_to(getattr(detector, "last_hand_mask", None), frame.shape)
    page_mask = _resize_to(page_mask, frame.shape)
    _write_png(
        out_dir / f"{stem}_seg.png", seg_overlay(frame, page_mask, hand, corners)
    )
    record = {
        "frame": idx,
        "verdict": verdict,
        "corners": None if corners is None else np.asarray(corners).tolist(),
        "luminance": luminance_stats(frame, corners),
    }
    if corners is None:
        return record
    record["quad_aspect"] = quad_aspect(corners)
    grids = {
        "warp": (config.PAPER_W, config.PAPER_H),
        "warp_aspect": aspect_grid(corners),
    }
    record["grids"] = {k: list(v) for k, v in grids.items()}
    record["ink_px"] = {}
    # As duas grades têm tamanhos diferentes: a contagem bruta não compara,
    # a fração (px de tinta / px da grade) sim.
    record["ink_frac"] = {}
    for name, size in grids.items():
        warped = warp_to(frame, corners, size)
        _write_png(out_dir / f"{stem}_{name}.png", warped)
        prob = page_clean.ink_prob(warped)
        if name == "warp":
            _write_png(
                out_dir / f"{stem}_prob.png",
                np.clip(prob * 255, 0, 255).astype(np.uint8),
            )
        suffix = "" if name == "warp" else "_aspect"
        mask = (prob > page_clean.threshold).astype(np.uint8) * 255
        _write_png(out_dir / f"{stem}_mask{suffix}.png", mask)
        counts = {str(t): int(np.count_nonzero(prob > t)) for t in _REPORT_THRESHOLDS}
        record["ink_px"][name] = counts
        record["ink_frac"][name] = {
            t: round(c / (size[0] * size[1]), 6) for t, c in counts.items()
        }
    return record


def _frames_from_dir(path, limit):
    """Quadros de uma pasta de PNGs; o limite conta só os que decodificam."""
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"--input não é uma pasta: {path}")
    got = 0
    for f in sorted(folder.glob("*.png")):
        if limit and got >= limit:
            break
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"AVISO: {f.name} ilegível; pulado", file=sys.stderr)
            continue
        got += 1
        yield img


def _frames_from_source(
    source,
    limit,
    warmup_s,
    spacing_s,
    stall_s=_STALL_TIMEOUT_S,
    opener=None,
    fps_burst_s=_FPS_BURST_S,
):
    """Lê `limit` quadros espaçados da fonte, pelo mesmo open_capture do app.

    Sem leitura boa por `stall_s` segundos a fonte é declarada parada
    (SourceExhausted) — o hvcam devolve False tanto no timeout transitório
    quanto no stream caído, e sem o limite o laço nunca terminava. O prazo
    é checado ENTRE leituras: a granularidade é o timeout de leitura da
    própria fonte (hvcam: 1 s), então um --stall-timeout menor que isso
    só vale na próxima volta. O FPS é medido numa rajada contínua de
    leituras (o aquecimento, no mínimo _FPS_BURST_S), antes de qualquer
    diagnóstico: durante o diagnóstico o gerador está suspenso e dividir
    por esse tempo subestimaria a câmera (review do PR #51).
    """
    from draw.camera import open_capture, parse_source

    cap = (opener or open_capture)(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(
            f"fonte {source} não abriu: {getattr(cap, 'open_error', '')}"
        )
    try:
        t_start = last_ok = time.monotonic()
        reads = 0
        t_first = None  # primeira leitura boa: a janela de FPS começa aqui
        n_first = 0

        def read():
            nonlocal reads, last_ok, t_first, n_first
            ok, frame = cap.read()
            now = time.monotonic()
            if ok and frame is not None:
                reads += 1
                last_ok = now
                if t_first is None:
                    t_first, n_first = now, reads
                return frame
            if now - last_ok > stall_s:
                raise SourceExhausted(
                    f"fonte parou de entregar: {now - last_ok:.1f} s sem quadro"
                )
            time.sleep(_RETRY_SLEEP_S)
            return None

        # Aquecimento (AE/AWB assentam) e janela de FPS: esta só abre na
        # primeira leitura boa, senão um timeout inicial maior que a janela
        # a fechava vazia e o resumo ficava com FPS 0 (review #51). O prazo
        # de parada segue ativo dentro de read() durante a partida.
        while True:
            read()
            now = time.monotonic()
            if (
                now - t_start >= warmup_s
                and t_first is not None
                and now - t_first >= fps_burst_s
            ):
                break
        fps = (reads - n_first) / max(now - t_first, 1e-6)
        got = 0
        next_t = time.monotonic()
        while got < limit:
            frame = read()
            if frame is None or time.monotonic() < next_t:
                continue
            got += 1
            next_t = time.monotonic() + spacing_s
            yield frame.copy(), fps
    finally:
        cap.release()


def _seconds(value):
    """Segundos finitos e >= 0: `nan` nunca satisfaz o `>=` do aquecimento
    (a ferramenta travava com a câmera funcionando) e `inf` não termina."""
    seconds = float(value)
    if not np.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError(
            f"precisa ser um número finito >= 0 (recebi {value!r})"
        )
    return seconds


def _positive_seconds(value):
    seconds = _seconds(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError(f"precisa ser > 0 (recebi {value!r})")
    return seconds


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"precisa ser >= 1 (recebi {value!r})")
    return number


def rotate_frame(frame, rotation):
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def _has_artifacts(out_dir):
    """Pasta de saída que já tem um diagnóstico: não se mistura execuções."""
    # Qualquer artefato por quadro (NNN_raw/seg/warp/prob/mask*.png) ou o
    # resumo: o operador pode ter guardado só as máscaras de uma rodada.
    return out_dir.is_dir() and (
        (out_dir / "summary.json").exists()
        or any(_FRAME_ARTIFACT.match(f.name) for f in out_dir.iterdir())
    )


def _mean(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="hvcam://, hvcam+tcp://host ou índice")
    src.add_argument("--input", help="pasta com PNGs de quadros brutos")
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "pasta de saída; recusada se já tiver summary.json ou artefatos "
            "NNN_*.png (prefixo de 3 ou mais dígitos)"
        ),
    )
    parser.add_argument("--frames", type=_positive_int, default=6)
    parser.add_argument(
        "--warmup", type=_seconds, default=2.0, help="s de AE/AWB (finito, >= 0)"
    )
    parser.add_argument(
        "--spacing", type=_seconds, default=0.5, help="s entre quadros (finito, >= 0)"
    )
    parser.add_argument(
        "--stall-timeout",
        type=_positive_seconds,
        default=_STALL_TIMEOUT_S,
        help=(
            "s sem quadro que declaram a fonte ao vivo parada; checado entre "
            "leituras, então a granularidade é o timeout de leitura da fonte "
            "(hvcam: 1 s)"
        ),
    )
    parser.add_argument("--rotate", type=int, choices=(0, 180), default=0)
    parser.add_argument(
        "--format",
        choices=list(config.FORMATS),
        default="a4",
        help="formato da sessão (a grade do warp do app); padrão a4",
    )
    parser.add_argument(
        "--orientation",
        choices=config.ORIENTATIONS,
        default="retrato",
        help="orientação da grade do warp do app (paisagem p/ caderno aberto)",
    )
    parser.add_argument("--models", default=str(_REPO_ROOT / "models"))
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="não confere os modelos contra o manifesto (binário fora dele)",
    )
    parser.add_argument("--label", default="", help="rótulo gravado no summary")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="limiar da máscara (vence o manifesto)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    if _has_artifacts(out_dir):
        print(
            f"erro: {out_dir} já tem um diagnóstico (summary.json ou NNN_*.png, "
            "3+ dígitos); "
            "use uma pasta nova para não misturar execuções",
            file=sys.stderr,
        )
        return 2
    # `is not None`, não verdade: --input "" é um caminho vazio (inválido), não
    # a ausência da opção — e Path("") é a pasta atual.
    if args.input is not None and (
        not args.input.strip() or not Path(args.input).is_dir()
    ):
        print(f"erro: --input não é uma pasta: {args.input!r}", file=sys.stderr)
        return 2
    # Mesma grade que o app usa na sessão: o warp e a máscara do diagnóstico
    # têm de ter as dimensões do que o runtime enxerga.
    config.apply_format(args.format, args.orientation)

    from draw.infra.manifest import verify_model
    from draw.page_clean import PageCleanInk
    from draw.perception import NeuralSheetDetector

    verify = None if args.allow_unverified else verify_model
    models = Path(args.models)
    detector = NeuralSheetDetector(str(models / "draw_seg_v3_dyn.onnx"), verify=verify)
    detector.load()
    page_clean = PageCleanInk(
        str(models / "page_clean_unet.onnx"), threshold=args.threshold, verify=verify
    )
    page_clean.load()

    out_dir.mkdir(parents=True, exist_ok=True)
    records, read_fps, error = [], None, None
    if args.input is not None:
        frames = ((f, None) for f in _frames_from_dir(args.input, args.frames))
    else:
        frames = _frames_from_source(
            args.source,
            args.frames,
            args.warmup,
            args.spacing,
            stall_s=args.stall_timeout,
        )
    try:
        for idx, (frame, fps) in enumerate(frames):
            read_fps = fps if fps is not None else read_fps
            frame = rotate_frame(frame, args.rotate)
            records.append(diagnose_frame(frame, detector, page_clean, out_dir, idx))
    except (SourceExhausted, OSError) as exc:
        # Fonte parada ou PNG que não gravou: os quadros já concluídos
        # ficam no resumo parcial em vez de órfãos.
        error = str(exc)
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()  # libera a câmera mesmo com a falha no meio
    if not records and error is None:
        error = "nenhum quadro foi lido"
    summary = {
        "label": args.label,
        "rotate": args.rotate,
        "format": args.format,
        "orientation": args.orientation,
        "read_fps": read_fps,
        "mask_threshold": page_clean.threshold,
        "complete": error is None,
        "error": error,
        "frames": records,
    }
    # O resumo sai também na falha: os quadros já gravados não ficam órfãos.
    # Se nem ele grava (volume cheio, permissão), erro claro e saída 1 — sem
    # traceback e sem perder a linha de resumo no terminal.
    summary_path = out_dir / "summary.json"
    summary_error = None
    try:
        summary_path.write_text(json.dumps(summary, indent=1))
    except OSError as exc:
        summary_error = f"não foi possível gravar {summary_path}: {exc}"
    lum = [r["luminance"] for r in records]
    if lum:
        left = _mean([v.get("quad_left_mean") for v in lum])
        right = _mean([v.get("quad_right_mean") for v in lum])
        print(
            f"{args.label or out_dir.name}: {len(records)} quadros, "
            f"fps_leitura={'-' if read_fps is None else round(read_fps, 1)}, "
            f"lum_media={_mean([v['frame_mean'] for v in lum]):.0f}, "
            f"quad_esq={'-' if left is None else round(left)}, "
            f"quad_dir={'-' if right is None else round(right)}"
        )
    if summary_error is not None:
        print(f"erro: {summary_error}", file=sys.stderr)
    if error is not None:
        # Só aponta o resumo parcial se ele de fato foi gravado.
        where = (
            f"resumo parcial em {summary_path}"
            if summary_error is None
            else "e o resumo parcial não pôde ser gravado"
        )
        print(f"erro: {error} ({where})", file=sys.stderr)
    return 1 if (error is not None or summary_error is not None) else 0


if __name__ == "__main__":
    sys.exit(main())
