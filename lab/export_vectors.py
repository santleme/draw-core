"""Ponte de vetores do Draw: canvas confirmado -> polilinhas -> SVG + DXF.

Driver de linha de comando sobre draw.exporters.export_vectors (os
primitivos: Otsu, microfragmentos, Zhang-Suen, traçado, Douglas-Peucker,
escritores SVG/DXF e a prova de ida e volta). Entrada: PNG na grade da
página (warp ou canvas de tinta acumulado). A máscara vem do modelo de
produção (draw.page_clean.PageCleanInk, mosaico 256, limiar de máscara
do manifesto — a inferência do app num quadro só; o canvas ao vivo vota
entre passadas com InkConfig.seen_threshold) ou de Otsu invertido.

Exemplo:
  python lab/export_vectors.py pagina.png --out-dir saida/
      [--mask model|otsu] [--onnx models/page_clean_unet.onnx]
      [--epsilon 1.5] [--format a4|a5|a3|carta|caderno]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from draw import config  # noqa: E402
from draw.exporters.export_vectors import (  # noqa: E402
    _export_scale,
    filter_microfragments,
    ink_mask_otsu,
    iou,
    make_8_minimal,
    median_stroke_width,
    parse_dxf_polylines,
    parse_svg_paths,
    rasterize,
    simplify_polylines,
    trace_polylines,
    write_dxf,
    write_svg,
    zhang_suen_thin,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = _REPO_ROOT / "models" / "page_clean_unet.onnx"


def ink_mask_model(bgr, onnx_path, verify, threshold=None):
    """Máscara de tinta do app: PageCleanInk (mosaico 256). -> (máscara, limiar)

    Usa draw.page_clean, o mesmo caminho de inferência do Draw —
    inclusive a ordem de canais e o limiar de produção, que vem do
    runtime_threshold do manifesto (ou DEFAULT_THRESHOLD). O limiar
    efetivo volta junto para a prova registrar o que foi usado.
    threshold explícito vence o manifesto e o default — é a saída
    para um modelo fora do manifesto (--allow-unverified), que de
    outro modo herdaria o DEFAULT_THRESHOLD do modelo vigente.
    """
    from draw.page_clean import PageCleanInk

    cleaner = PageCleanInk(str(onnx_path), threshold=threshold, verify=verify)
    cleaner.load()
    return cleaner.ink_mask(bgr) > 0, cleaner.threshold


def export_vectors(
    bgr,
    mask_source="model",
    onnx_path=None,
    epsilon=1.5,
    paper_mm=None,
    min_area=4,
    paper_format="a4",
    verify="manifest",
    threshold=None,
):
    """Imagem na grade da página -> dict com polilinhas e estatísticas.

    verify é o callable de integridade do modelo: o padrão "manifest"
    injeta draw.infra.manifest.verify_model; None desliga (bancadas e
    testes com modelos fora do manifesto). Só se aplica a --mask model.
    """
    if verify == "manifest":
        from draw.infra.manifest import verify_model

        verify = verify_model
    img_h, img_w = bgr.shape[:2]
    if paper_mm is None:
        paper_w_mm, paper_h_mm = config.FORMATS[paper_format]
        # A escala uniforme do formato só vale na grade canônica dele;
        # fora dela, por eixo a partir das dimensões (adendo da
        # auditoria #47 — ver _export_scale nos primitivos).
        sx, sy = _export_scale(
            img_w,
            img_h,
            paper_w_mm,
            paper_h_mm,
            formats=[
                (n, dims, config.paper_grid(n), config.mm_per_px(n))
                for n, dims in config.FORMATS.items()
            ],
        )
    else:
        paper_w_mm, paper_h_mm = paper_mm
        sx, sy = paper_w_mm / img_w, paper_h_mm / img_h
    if mask_source == "model":
        if onnx_path is None or not Path(onnx_path).exists():
            raise FileNotFoundError(f"modelo page-clean não encontrado: {onnx_path}")
        ink, mask_threshold = ink_mask_model(bgr, onnx_path, verify, threshold)
    else:
        ink, mask_threshold = ink_mask_otsu(bgr), None
    ink = filter_microfragments(ink, min_area=min_area)

    skel = make_8_minimal(zhang_suen_thin(ink))
    raw = trace_polylines(skel)
    simplified = simplify_polylines(raw, epsilon)
    width_px = median_stroke_width(ink, skel)

    return {
        "ink": ink,
        "skeleton": skel,
        "raw_polylines": raw,
        "polylines": simplified,
        "stroke_width_px": width_px,
        "stroke_width_mm": width_px * 0.5 * (sx + sy),
        "scale_mm_per_px": (sx, sy),
        "points_raw": int(sum(len(poly["points"]) for poly in raw)),
        "points_simplified": int(sum(len(poly["points"]) for poly in simplified)),
        "mask_threshold": mask_threshold,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exporta uma página/canvas do Draw para SVG + DXF vetoriais."
    )
    parser.add_argument("input", help="PNG na grade da página (warp ou canvas)")
    parser.add_argument("--out-dir", default="vector-out", help="pasta de saída")
    parser.add_argument(
        "--mask",
        choices=["model", "otsu"],
        default="model",
        help="fonte da máscara de tinta (padrão: model = page-clean titular)",
    )
    parser.add_argument(
        "--onnx",
        default=str(DEFAULT_ONNX),
        help="page-clean .onnx para --mask model",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.5,
        help="epsilon do Douglas-Peucker em px (padrão 1,5)",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="desliga a verificação do manifesto sobre o .onnx",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "limiar da máscara para --mask model; vence o runtime_threshold "
            "do manifesto. Use com um .onnx antigo sob --allow-unverified "
            "(ex.: 0.85 para o round-v1), que senão herda o DEFAULT_THRESHOLD"
        ),
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--format",
        choices=list(config.FORMATS),
        default="a4",
        help="formato da sessão; escala de draw.config (padrão: a4)",
    )
    size.add_argument(
        "--paper-mm",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="dimensões livres, API legada (sem --format)",
    )
    args = parser.parse_args(argv)
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("--threshold precisa estar em (0, 1)")

    bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"erro: não foi possível ler {args.input}", file=sys.stderr)
        return 1
    img_h, img_w = bgr.shape[:2]
    paper_w_mm, paper_h_mm = (
        config.FORMATS[args.format] if args.paper_mm is None else args.paper_mm
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    verify = None if args.allow_unverified else "manifest"
    t0 = time.perf_counter()
    result = export_vectors(
        bgr,
        mask_source=args.mask,
        onnx_path=args.onnx,
        epsilon=args.epsilon,
        paper_mm=args.paper_mm,
        paper_format=args.format,
        verify=verify,
        threshold=args.threshold,
    )
    svg_path = out_dir / f"{stem}.svg"
    dxf_path = out_dir / f"{stem}.dxf"
    write_svg(
        result["polylines"],
        svg_path,
        img_w,
        img_h,
        paper_w_mm,
        paper_h_mm,
        result["stroke_width_mm"],
        scale_mm_per_px=result["scale_mm_per_px"],
    )
    write_dxf(
        result["polylines"],
        dxf_path,
        img_w,
        img_h,
        paper_w_mm,
        paper_h_mm,
        result["stroke_width_mm"],
        scale_mm_per_px=result["scale_mm_per_px"],
    )

    # Prova: relê os DOIS arquivos do disco, rasteriza e compara com a máscara.
    sx, sy = result["scale_mm_per_px"]
    thickness = max(int(round(result["stroke_width_px"])), 1)
    ink = result["ink"]
    raster_svg = rasterize(parse_svg_paths(svg_path), ink.shape, sx, sy, thickness)
    raster_dxf = rasterize(
        parse_dxf_polylines(dxf_path),
        ink.shape,
        sx,
        sy,
        thickness,
        flip_y=True,
        paper_h_mm=paper_h_mm,
    )
    iou_svg = iou(raster_svg, ink)
    iou_dxf = iou(raster_dxf, ink)

    overlay = bgr.copy()
    overlay[raster_svg] = (0, 0, 255)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), (ink.astype(np.uint8) * 255))

    proof = {
        "input": str(args.input),
        "mask_source": (
            f"draw.page_clean.PageCleanInk sobre {args.onnx} "
            f"(mosaico 256x256, limiar da mascara de quadro unico "
            f"{result['mask_threshold']}; o canvas ao vivo vota com "
            "InkConfig.seen_threshold, nao com este limiar)"
            if args.mask == "model"
            else "otsu invertido (declarado)"
        ),
        "page_px": [img_w, img_h],
        "paper_mm": [paper_w_mm, paper_h_mm],
        "paper_format": args.format if args.paper_mm is None else None,
        "scale_mm_per_px": result["scale_mm_per_px"],
        "epsilon_px": args.epsilon,
        "ink_px": int(ink.sum()),
        "skeleton_px": int(result["skeleton"].sum()),
        "polylines": len(result["polylines"]),
        "points_before_simplification": result["points_raw"],
        "points_after_simplification": result["points_simplified"],
        "stroke_width_px_median": round(result["stroke_width_px"], 2),
        "svg_bytes": svg_path.stat().st_size,
        "dxf_bytes": dxf_path.stat().st_size,
        "iou_svg_roundtrip": round(iou_svg, 4),
        "iou_dxf_roundtrip": round(iou_dxf, 4),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / f"{stem}_proof.json").write_text(
        json.dumps(proof, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(proof, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
