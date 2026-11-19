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
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from draw import config  # noqa: E402

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


def diagnose_frame(frame, detector, page_clean, out_dir, idx):
    """Roda detector + page_clean num quadro e grava os artefatos."""
    stem = f"{idx:03d}"
    cv2.imwrite(str(out_dir / f"{stem}_raw.png"), frame)
    corners, verdict, page_mask = detector(frame, return_mask=True)
    hand = _resize_to(getattr(detector, "last_hand_mask", None), frame.shape)
    page_mask = _resize_to(page_mask, frame.shape)
    cv2.imwrite(
        str(out_dir / f"{stem}_seg.png"),
        seg_overlay(frame, page_mask, hand, corners),
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
    for name, size in grids.items():
        warped = warp_to(frame, corners, size)
        cv2.imwrite(str(out_dir / f"{stem}_{name}.png"), warped)
        prob = page_clean.ink_prob(warped)
        if name == "warp":
            cv2.imwrite(
                str(out_dir / f"{stem}_prob.png"),
                np.clip(prob * 255, 0, 255).astype(np.uint8),
            )
        suffix = "" if name == "warp" else "_aspect"
        mask = (prob > page_clean.threshold).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{stem}_mask{suffix}.png"), mask)
        record["ink_px"][name] = {
            str(t): int(np.count_nonzero(prob > t)) for t in _REPORT_THRESHOLDS
        }
    return record


def _frames_from_dir(path, limit):
    files = sorted(Path(path).glob("*.png"))[: limit or None]
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is not None:
            yield img


def _frames_from_source(source, limit, warmup_s, spacing_s):
    """Lê `limit` quadros da fonte (hvcam:// ou índice OpenCV), espaçados."""
    from draw.hvcam import HvcamCapture, is_hvcam_source

    cap = HvcamCapture(source) if is_hvcam_source(source) else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"fonte {source} não abriu: {getattr(cap, 'open_error', '')}"
        )
    try:
        t0 = time.monotonic()
        reads = 0
        while time.monotonic() - t0 < warmup_s:  # AE/AWB assentam
            ok, _ = cap.read()
            reads += ok
        fps = reads / max(warmup_s, 1e-6)
        got = 0
        next_t = time.monotonic()
        while got < limit:
            ok, frame = cap.read()
            if not ok:
                continue
            if time.monotonic() >= next_t:
                got += 1
                next_t = time.monotonic() + spacing_s
                yield frame.copy(), fps
    finally:
        cap.release()


def rotate_frame(frame, rotation):
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="hvcam://, hvcam+tcp://host ou índice")
    src.add_argument("--input", help="pasta com PNGs de quadros brutos")
    parser.add_argument("--out", required=True, help="pasta de saída")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--warmup", type=float, default=2.0, help="s de AE/AWB")
    parser.add_argument("--spacing", type=float, default=0.5, help="s entre quadros")
    parser.add_argument("--rotate", type=int, choices=(0, 180), default=0)
    parser.add_argument("--models", default=str(_REPO_ROOT / "models"))
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="não confere os modelos contra o manifesto (binário fora dele)",
    )
    parser.add_argument("--label", default="", help="rótulo gravado no summary")
    parser.add_argument(
        "--threshold", type=float, default=None, help="limiar da máscara (vence o manifesto)"
    )
    args = parser.parse_args(argv)

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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, read_fps = [], None
    if args.input:
        frames = ((f, None) for f in _frames_from_dir(args.input, args.frames))
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        frames = _frames_from_source(source, args.frames, args.warmup, args.spacing)
    for idx, (frame, fps) in enumerate(frames):
        read_fps = fps if fps is not None else read_fps
        frame = rotate_frame(frame, args.rotate)
        records.append(diagnose_frame(frame, detector, page_clean, out_dir, idx))
    summary = {
        "label": args.label,
        "rotate": args.rotate,
        "read_fps": read_fps,
        "mask_threshold": page_clean.threshold,
        "frames": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    lum = [r["luminance"] for r in records]
    if lum:
        print(
            f"{args.label or out_dir.name}: {len(records)} quadros, "
            f"fps_leitura={read_fps if read_fps is None else round(read_fps, 1)}, "
            f"lum_media={np.mean([v['frame_mean'] for v in lum]):.0f}, "
            f"quad_esq={np.mean([v.get('quad_left_mean', np.nan) for v in lum]):.0f}, "
            f"quad_dir={np.mean([v.get('quad_right_mean', np.nan) for v in lum]):.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
