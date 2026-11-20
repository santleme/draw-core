"""Partes puras da ponte de vetores do Draw (tools/export_vectors.py).

Extraidas sem a CLI e sem a fonte de mascara --mask model (que usa
draw.page_clean.PageCleanInk): restam a mascara otsu, o afinamento
Zhang-Suen vetorizado, o tracador de polilinhas, a escala de exportacao
(formatos injetados) e os writers/leitores SVG e DXF.

Diferente de exporters/vectorize.py, este modulo NAO usa cv2.ximgproc
(indisponivel em varios ambientes): o afinamento e Zhang-Suen
vetorizado em numpy, neste arquivo.
"""

import re
from pathlib import Path

import cv2
import numpy as np

_NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_DEGREE_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


# ---------------------------------------------------------------- máscara
def ink_mask_otsu(bgr):
    """Tinta escura por Otsu invertido; devolve também a polaridade."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = dark > 0
    # Se o "escuro" dominar a página, a polaridade está invertida.
    if mask.mean() > 0.5:
        mask = ~mask
    return mask


def filter_microfragments(ink, min_area=4):
    """Remove componentes conexas menores que `min_area` px."""
    if min_area <= 1:
        return ink.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros_like(ink)
    keep = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
    keep[0] = False
    return keep[labels]


# ------------------------------------------------------- afinamento (ZS)


def zhang_suen_thin(ink, *, native=False):
    """Esqueleto de 1 px por Zhang-Suen vetorizado (sem cv2.ximgproc).

    native=True usa o cv2.ximgproc.thinning (C++, ~10x mais rapido num
    canvas A4) quando existir; o padrao segue em numpy porque a saida NAO e
    identica: difere em dezenas de pixels num canvas real (2 e 34 px de ~20 mil
    medidos) e muito mais em manchas cheias. Quem liga aceita esse delta.
    """
    src = ink.astype(np.uint8)
    if native and hasattr(cv2, "ximgproc"):
        skel = (
            cv2.ximgproc.thinning(
                src * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
            > 0
        )
        return _keep_component_pixel(src, skel)
    img = src.copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = sum(((seq[i] == 0) & (seq[i + 1] == 1)) for i in range(8))
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            if step == 0:
                m = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                m = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            delete = (img == 1) & (b >= 2) & (b <= 6) & (transitions == 1) & m
            if delete.any():
                img[delete] = 0
                changed = True
    return _keep_component_pixel(src, img.astype(bool))


def _keep_component_pixel(src, skel):
    """Todo componente nao vazio mantem ao menos 1 px no esqueleto.

    Zhang-Suen apaga por completo componentes pequenos — um ponto 2x2
    some inteiro e o export o omitia. Para o componente apagado,
    conserva o pixel mais proximo do centroide; componentes cujo
    esqueleto sobreviveu ficam intocados.
    """
    n, labels, _, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
    for i in range(1, n):
        comp = labels == i
        if (comp & skel).any():
            continue
        ys, xs = np.nonzero(comp)
        cx, cy = centroids[i]
        pick = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        skel[ys[pick], xs[pick]] = True
    return skel


# ------------------------------------------------------------ travessia


def make_8_minimal(skel):
    """Remove pixels de canto L redundantes preservando a conectividade-8.

    O Zhang-Suen deixa degraus 4-conexos (pixel com vizinhos N e E, que
    já são 8-vizinhos entre si): esses triângulos viram falsos nós de
    grau 3 ao longo de traços diagonais e partem curvas simples em
    dezenas de arcos curtos. Remover cada orientação em sequência
    preserva a topologia 8-conexa. (Mesma técnica de
    tools/vectorize.py, reimplementada aqui sem ximgproc.)
    """
    s = skel.copy()
    for rot in range(4):
        p = np.pad(s, 1, constant_values=False)
        c = p[1:-1, 1:-1]
        n = p[0:-2, 1:-1]
        so = p[2:, 1:-1]
        e = p[1:-1, 2:]
        w = p[1:-1, 0:-2]
        nw = p[0:-2, 0:-2]
        ne = p[0:-2, 2:]
        sw = p[2:, 0:-2]
        se = p[2:, 2:]
        if rot == 0:
            rem = c & n & e & (~w) & (~so) & (~sw) & (~(nw & se))
        elif rot == 1:
            rem = c & e & so & (~n) & (~w) & (~nw) & (~(ne & sw))
        elif rot == 2:
            rem = c & so & w & (~n) & (~e) & (~ne) & (~(nw & se))
        else:
            rem = c & w & n & (~so) & (~e) & (~se) & (~(ne & sw))
        s[rem] = False
    return s


def _degree(skel):
    skel_u8 = skel.astype(np.uint8)
    deg = cv2.filter2D(
        skel_u8, cv2.CV_16U, _DEGREE_KERNEL, borderType=cv2.BORDER_CONSTANT
    )
    return (deg * skel_u8).astype(np.int32)


def _index_node_components(labels, n_nodes):
    """Pixels e centroide de cada no, numa unica passada O(H*W).

    Antes: `np.nonzero(labels == nid)` por no varria a imagem inteira
    (1 M px) uma vez por no — ~2,8 mil nos custavam ~2,7 s num canvas real.
    Mesma saida (centroides e pixels em ordem de varredura), ~46x mais rapido.
    """
    labels_flat = labels.ravel()
    width = labels.shape[1]
    nz_flat = np.flatnonzero(labels_flat)
    if len(nz_flat) == 0 or n_nodes <= 1:
        return {}, {}
    lbl_vals = labels_flat[nz_flat]
    order = np.argsort(lbl_vals, kind="stable")
    sorted_flat = nz_flat[order]
    sorted_lbl = lbl_vals[order]
    counts = np.bincount(sorted_lbl, minlength=n_nodes)
    splits = np.cumsum(counts)
    sum_x = np.bincount(sorted_lbl, weights=sorted_flat % width, minlength=n_nodes)
    sum_y = np.bincount(sorted_lbl, weights=sorted_flat // width, minlength=n_nodes)
    anchors = {}
    node_pixels = {}
    for nid in range(1, n_nodes):
        count = int(counts[nid])
        if count == 0:
            continue
        anchors[nid] = (float(sum_x[nid] / count), float(sum_y[nid] / count))
        node_pixels[nid] = sorted_flat[int(splits[nid - 1]) : int(splits[nid])].tolist()
    return anchors, node_pixels


def trace_polylines(skel):
    """Esqueleto -> polilinhas de centro ({points Nx2 (x,y) px, closed})."""
    height, width = skel.shape
    deg = _degree(skel)
    node_mask = skel & (deg != 2)
    n_nodes, labels = cv2.connectedComponents(
        node_mask.astype(np.uint8), connectivity=8
    )
    labels_flat = labels.ravel()
    skel_flat = skel.ravel()

    anchors, node_pixels = _index_node_components(labels, n_nodes)

    def neighbors(flat_idx):
        y, x = divmod(flat_idx, width)
        out = []
        for dy, dx in _NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                nidx = ny * width + nx
                if skel_flat[nidx]:
                    out.append(nidx)
        return out

    polylines = []
    visited = set()  # arestas (a, b) sem direção

    def walk(start_pixel, first_step, start_nid):
        path = [first_step]
        visited.add((start_pixel, first_step))
        visited.add((first_step, start_pixel))
        prev, cur = start_pixel, first_step
        end_anchor = None
        while labels_flat[cur] == 0:
            nbrs = [n for n in neighbors(cur) if n != prev]
            if len(nbrs) != 1:
                end_anchor = (float(cur % width), float(cur // width))
                break
            nxt = nbrs[0]
            visited.add((cur, nxt))
            visited.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        if end_anchor is None:
            end_anchor = anchors[int(labels_flat[cur])]
        points = [anchors[start_nid]]
        points += [(float(p % width), float(p // width)) for p in path]
        points.append(end_anchor)
        closed = points[0] == points[-1] and len(path) >= 2
        return {"points": np.array(points, dtype=np.float64), "closed": closed}

    for nid, pixels in node_pixels.items():
        has_external = False
        for p in pixels:
            for q in neighbors(p):
                if labels_flat[q] == nid:
                    continue
                has_external = True
                if (p, q) not in visited:
                    polylines.append(walk(p, q, nid))
        if not has_external:
            polylines.append({"points": np.array([anchors[nid]]), "closed": False})

    covered = np.zeros(skel_flat.shape, dtype=bool)
    for a, _b in visited:
        covered[a] = True
    remaining = [
        int(i) for i in np.nonzero(skel_flat & ~covered & (labels_flat == 0))[0]
    ]
    seen = set()
    for start in remaining:
        if start in seen:
            continue
        path = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nbrs = [n for n in neighbors(cur) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt == start and len(path) >= 3:
                break
            if nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        points = [(float(p % width), float(p // width)) for p in path]
        polylines.append(
            {
                "points": np.array(points, dtype=np.float64),
                "closed": len(path) >= 3,
            }
        )
    return polylines


# --------------------------------------------------------- simplificação


def simplify_polylines(polylines, epsilon):
    """Douglas-Peucker via cv2.approxPolyDP, linha a linha."""
    out = []
    for line in polylines:
        pts = line["points"]
        if len(pts) <= 2:
            out.append(line)
            continue
        approx = (
            cv2.approxPolyDP(
                pts.reshape(-1, 1, 2).astype(np.float32),
                float(epsilon),
                bool(line["closed"]),
            )
            .reshape(-1, 2)
            .astype(np.float64)
        )
        if len(approx) >= 2:
            out.append({"points": approx, "closed": line["closed"]})
        else:
            out.append(line)
    return out


def median_stroke_width(ink, skel):
    """Largura típica do traço: 2 x mediana do distanceTransform no esqueleto."""
    if not skel.any():
        return 1.0
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    return float(2.0 * np.median(dt[skel]))


# -------------------------------------------------------------- escritas


def _export_scale(img_w, img_h, paper_w_mm, paper_h_mm, scale=None, formats=None):
    """Escala mm/px; formats sao quadruplas (nome, dimensoes, grade, mm_por_px).

    O chamador que tem draw.config monta a lista:
    [(n, dims, config.paper_grid(n), config.mm_per_px(n))
     for n, dims in config.FORMATS.items()].
    """
    if scale is not None:
        return scale
    for _name, dimensions, grid, unit in formats or ():
        if dimensions == (paper_w_mm, paper_h_mm):
            # A escala uniforme do formato so vale na grade canonica dele:
            # mm_per_px e definido como altura_mm/1200 sobre a grade, nao
            # sobre a imagem. Fora da grade, inferir por (mm, mm) aplicava
            # 0,2475 em x tambem a imagens cuja largura nao e a da grade
            # (erro silencioso de ate +17,9% em x; adendo da auditoria #47).
            if (img_w, img_h) == grid:
                return unit, unit
    return paper_w_mm / img_w, paper_h_mm / img_h


def write_svg(
    polylines,
    out_path,
    img_w,
    img_h,
    paper_w_mm,
    paper_h_mm,
    stroke_w_mm,
    scale_mm_per_px=None,
    formats=None,
):
    """SVG com <path> reais em mm (Y para baixo, como a imagem)."""
    sx, sy = _export_scale(
        img_w, img_h, paper_w_mm, paper_h_mm, scale_mm_per_px, formats
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- Draw vector bridge: polilinhas de centro do canvas, em mm -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{paper_w_mm}mm" height="{paper_h_mm}mm" '
        f'viewBox="0 0 {paper_w_mm} {paper_h_mm}">',
    ]
    for line in polylines:
        pts = line["points"]
        if len(pts) == 1:
            cx, cy = pts[0][0] * sx, pts[0][1] * sy
            lines.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" '
                f'r="{max(stroke_w_mm / 2, 0.1):.2f}" fill="#000"/>'
            )
            continue
        d = "M " + " L ".join(f"{x * sx:.2f} {y * sy:.2f}" for x, y in pts)
        if line["closed"]:
            d += " Z"
        lines.append(
            f'<path d="{d}" fill="none" stroke="#000" '
            f'stroke-width="{stroke_w_mm:.2f}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )
    lines.append("</svg>")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dxf(
    polylines,
    out_path,
    img_w,
    img_h,
    paper_w_mm,
    paper_h_mm,
    stroke_w_mm,
    scale_mm_per_px=None,
    formats=None,
):
    """DXF mínimo (AC1015) escrito à mão: LWPOLYLINE em mm, Y para cima."""
    sx, sy = _export_scale(
        img_w, img_h, paper_w_mm, paper_h_mm, scale_mm_per_px, formats
    )

    def conv(pts):
        return [(x * sx, paper_h_mm - y * sy) for x, y in pts]

    out = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "9",
        "$ACADVER",
        "1",
        "AC1015",
        "9",
        "$INSUNITS",
        "70",
        "4",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "TABLES",
        "0",
        "TABLE",
        "2",
        "LAYER",
        "70",
        "1",
        "0",
        "LAYER",
        "2",
        "INK",
        "70",
        "0",
        "62",
        "7",
        "6",
        "Continuous",
        "0",
        "ENDTAB",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "BLOCKS",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
    ]
    for line in polylines:
        pts = conv(line["points"])
        if len(pts) == 1:
            pts = pts + pts  # LWPOLYLINE degenerada de 1 vértice não é válida
        out += [
            "0",
            "LWPOLYLINE",
            "100",
            "AcDbEntity",
            "8",
            "INK",
            "100",
            "AcDbPolyline",
            "90",
            str(len(pts)),
            "70",
            "1" if line["closed"] else "0",
            "43",
            f"{stroke_w_mm:.3f}",
        ]
        for x, y in pts:
            out += ["10", f"{x:.3f}", "20", f"{y:.3f}"]
    out += ["0", "ENDSEC", "0", "EOF"]
    Path(out_path).write_text("\n".join(out) + "\n", encoding="utf-8")


# ------------------------------------------------- prova (round-trip IoU)


def parse_svg_paths(svg_path):
    """Lê de volta os <path> e os pontos isolados (<circle>) do SVG:
    lista de (Nx2 mm, closed); um <circle> vira uma entrada de 1 ponto."""
    text = Path(svg_path).read_text(encoding="utf-8")
    out = []
    for match in re.finditer(
        r'<path d="([^"]+)"|<circle cx="([^"]+)" cy="([^"]+)"', text
    ):
        if match.group(1) is not None:
            d = match.group(1)
            nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", d)]
            pts = np.array(nums, dtype=np.float64).reshape(-1, 2)
            out.append((pts, d.strip().upper().endswith("Z")))
        else:
            out.append(
                (
                    np.array(
                        [[float(match.group(2)), float(match.group(3))]],
                        dtype=np.float64,
                    ),
                    False,
                )
            )
    return out


def parse_dxf_polylines(dxf_path):
    """Lê de volta as LWPOLYLINE do DXF: lista de (Nx2 mm, closed)."""
    tokens = [
        t.strip() for t in Path(dxf_path).read_text(encoding="utf-8").splitlines()
    ]
    pairs = list(zip(tokens[0::2], tokens[1::2], strict=False))
    out = []
    i = 0
    while i < len(pairs):
        code, value = pairs[i]
        if code == "0" and value == "LWPOLYLINE":
            verts = []
            closed = False
            j = i + 1
            pending_x = None
            while j < len(pairs) and pairs[j][0] != "0":
                c, v = pairs[j]
                if c == "70":
                    closed = v == "1"
                elif c == "10":
                    pending_x = float(v)
                elif c == "20" and pending_x is not None:
                    verts.append((pending_x, float(v)))
                    pending_x = None
                j += 1
            if verts:
                out.append((np.array(verts, dtype=np.float64), closed))
            i = j
        else:
            i += 1
    return out


def rasterize(mm_polylines, shape, sx, sy, thickness_px, flip_y=False, paper_h_mm=None):
    """Desenha polilinhas (mm) de volta na grade de pixels da página."""
    raster = np.zeros(shape, dtype=np.uint8)
    for pts_mm, closed in mm_polylines:
        pts = pts_mm.copy()
        if flip_y:
            pts[:, 1] = paper_h_mm - pts[:, 1]
        px = np.round(pts / np.array([sx, sy])).astype(np.int32)
        if len(px) == 1:
            cv2.circle(raster, tuple(px[0]), max(thickness_px // 2, 0), 1, -1)
        else:
            cv2.polylines(
                raster,
                [px.reshape(-1, 1, 2)],
                bool(closed),
                1,
                thickness=max(int(thickness_px), 1),
                lineType=cv2.LINE_AA,
            )
    return raster > 0


def iou(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 1.0


# ------------------------------------------------------------------ core
