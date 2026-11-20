"""Testes de tools/vectorize.py v2: grafo O(H*W), limpeza, carimbos e Camada 2."""

import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from draw.exporters import vectorize


@pytest.mark.slow
def test_node_indexing_matches_naive_and_scales_linearly():
    h, w = 1200, 850
    mask = np.zeros((h, w), dtype=np.uint8)
    # ~2000 componentes de nós espalhadas pela página:
    for r in range(20, h - 20, 24):
        for c in range(20, w - 20, 20):
            mask[r : r + 2, c : c + 2] = 1
    n_nodes, labels = cv2.connectedComponents(mask, connectivity=8)
    assert n_nodes >= 1800

    t0 = time.perf_counter()
    anchors, node_pixels = vectorize._index_node_components(labels, n_nodes)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 250.0, f"indexacao O(H*W) demorou demais: {dt_ms:.1f} ms"
    assert len(anchors) == n_nodes - 1

    # Confere equivalência exata em amostra determinística de nós:
    for nid in (1, n_nodes // 2, n_nodes - 1):
        ys, xs = np.nonzero(labels == nid)
        want_anchor = (float(xs.mean()), float(ys.mean()))
        want_pixels = [int(y) * w + int(x) for y, x in zip(ys, xs, strict=False)]
        assert abs(anchors[nid][0] - want_anchor[0]) < 1e-9
        assert abs(anchors[nid][1] - want_anchor[1]) < 1e-9
        assert node_pixels[nid] == want_pixels


def test_thick_shapes_axis_aligned_rect_circle_lines_and_freehand_wave():
    img = np.full((vectorize.PAPER_H_PX, vectorize.PAPER_W_PX), 255, dtype=np.uint8)
    # 1. Retângulo alinhado aos eixos (traço 5 px — caso clássico que partia em 8+ pedaços):
    cv2.rectangle(img, (80, 90), (340, 290), 0, thickness=5)
    # 2. Círculo (traço 5 px):
    cv2.circle(img, (610, 210), 95, 0, thickness=5)
    # 3. Duas retas cruzadas ('X', traço 5 px):
    cv2.line(img, (110, 480), (390, 740), 0, thickness=5)
    cv2.line(img, (110, 740), (390, 480), 0, thickness=5)
    # 4. Curva senoidal à mão livre (DEVE continuar como 'polyline', sem alucinar reta/círculo):
    wave_pts = np.array(
        [[470 + i * 3, int(round(620 + 55.0 * np.sin(i * 0.14)))] for i in range(100)],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(img, [wave_pts], False, 0, thickness=4)
    # 5. Poeira isolada (1-3 px) que deve sumir em min_area=4:
    img[25, 25] = 0
    img[25, 26] = 0
    img[1150, 810] = 0

    ink = img < 128
    polys = vectorize.vectorize_clean(ink, epsilon=1.5)
    prims = vectorize.classify_primitives(polys)
    by_type = {}
    for p in prims:
        by_type.setdefault(p["type"], []).append(p)

    assert len(prims) == 5, (
        f"esperava 5 tracos limpos, obteve {len(prims)}: {[p['type'] for p in prims]}"
    )
    assert len(by_type.get("rectangle", [])) == 1
    assert len(by_type.get("circle", [])) == 1
    assert len(by_type.get("line", [])) == 2
    assert len(by_type.get("polyline", [])) == 1

    circ = by_type["circle"][0]
    assert abs(circ["r_mm"] - 95.0 * vectorize.MM_PER_PX) < 1.2
    rect = by_type["rectangle"][0]
    dims = sorted([rect["width_mm"], rect["height_mm"]])
    want_dims = sorted([200.0 * vectorize.MM_PER_PX, 260.0 * vectorize.MM_PER_PX])
    assert abs(dims[0] - want_dims[0]) < 2.0
    assert abs(dims[1] - want_dims[1]) < 2.0


def test_stamps_order_strokes_chronologically_and_orient_direction():
    h, w = vectorize.PAPER_H_PX, vectorize.PAPER_W_PX
    img = np.full((h, w), 255, dtype=np.uint8)
    first_seen = np.zeros((h, w), dtype=np.uint32)
    last_update = np.zeros((h, w), dtype=np.uint32)

    # Três retas desenhadas de baixo para cima e da DIREITA para a ESQUERDA:
    strokes = [
        ((700, 950), (180, 950), 1000),  # desenhada primeiro (embaixo, dir->esq)
        ((700, 550), (180, 550), 3000),  # desenhada segundo (meio, dir->esq)
        ((700, 180), (180, 180), 6000),  # desenhada terceiro (topo, dir->esq)
    ]
    for (x0, y0), (x1, y1), base_t in strokes:
        cv2.line(img, (x0, y0), (x1, y1), 0, thickness=3)
        for step in range(101):
            alpha = step / 100.0
            cx = int(round(x0 + alpha * (x1 - x0)))
            cy = int(round(y0 + alpha * (y1 - y0)))
            t_val = np.uint32(base_t + step * 5)
            first_seen[cy - 3 : cy + 4, cx - 3 : cx + 4] = t_val
            last_update[cy - 3 : cy + 4, cx - 3 : cx + 4] = t_val + np.uint32(100)

    with tempfile.TemporaryDirectory() as tmp:
        png_path = Path(tmp) / "canvas.png"
        npz_path = Path(tmp) / "canvas_stamps.npz"
        svg_path = Path(tmp) / "out.svg"
        json_path = Path(tmp) / "out.json"
        cv2.imwrite(str(png_path), img)
        np.savez_compressed(
            npz_path, first_seen_ms=first_seen, last_update_ms=last_update
        )

        rc = vectorize.main(
            [
                str(png_path),
                "--clean",
                "--stamps",
                str(npz_path),
                "--primitives",
                "--out",
                str(svg_path),
                "--json",
                str(json_path),
            ]
        )
        assert rc == 0
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        prims = payload["primitives"]
        assert [p["type"] for p in prims] == ["line", "line", "line"]
        # Ordem cronológica: y decrescente (950 -> 550 -> 180):
        ys_mm = [p["y1_mm"] for p in prims]
        assert ys_mm[0] > ys_mm[1] > ys_mm[2], f"ordem temporal incorreta: {ys_mm}"
        # Orientação direita->esquerda preservada (x1 > x2 em todas as três):
        for p in prims:
            assert p["x1_mm"] > p["x2_mm"], f"sentido temporal invertido: {p}"

        svg_txt = svg_path.read_text(encoding="utf-8")
        assert "<line " in svg_txt


def main():
    test_node_indexing_matches_naive_and_scales_linearly()
    test_thick_shapes_axis_aligned_rect_circle_lines_and_freehand_wave()
    test_stamps_order_strokes_chronologically_and_orient_direction()
    print("TESTES DE VECTORIZE V2 E CAMADA 2 OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_dp_open_keeps_collinear_backtrack_overshoot():
    """Douglas-Peucker media distancia a reta infinita das ancoras:
    um retrocesso colinear que ultrapassa a ancora tinha distancia
    zero e sumia da simplificacao. Com distancia ao SEGMENTO, o
    vai-e-vem fica."""
    pts = np.array([[0, 0], [30, 0], [10, 0]], dtype=np.float32)
    simplified = vectorize._dp_open(pts, epsilon=1.0)
    assert len(simplified) == 3
    assert (simplified[1] == [30, 0]).all()


# --- Higiene topológica v2 (Etapas A/B/C) --------------------------------


def _canvas():
    return np.full((400, 400), 255, dtype=np.uint8)


def _strokes(img):
    ink = img < 128
    return vectorize.vectorize_mask(
        ink,
        epsilon=1.5,
        min_area=vectorize.DUST_MIN_AREA,
        fill_holes=True,
        topology="v2",
    )


def test_fill_pinholes_closes_small_holes_and_keeps_real_ones():
    ink = np.zeros((60, 60), dtype=bool)
    ink[10:20, 10:20] = True
    ink[14:17, 14:17] = False  # furo de 9 px: vira tinta
    ink[30:50, 30:50] = True
    ink[37:42, 37:42] = False  # furo de 25 px: miolo real, fica
    ink[0:5, 50:60] = True
    ink[0, 54] = False  # toca a borda: não é furo fechado
    out = vectorize.fill_pinholes(ink)
    assert out[14:17, 14:17].all()
    assert not out[37:42, 37:42].any()
    assert not out[0, 54]
    assert (out & ~ink).sum() == 9


def test_pinhole_no_longer_splits_a_thick_stroke():
    img = _canvas()
    cv2.line(img, (40, 200), (360, 200), 0, thickness=9)
    img[199:202, 199:202] = 255  # furo de 9 px no meio do traço
    assert len(_strokes(img)) == 1


def test_short_spur_is_pruned_and_long_branch_kept():
    img = _canvas()
    cv2.line(img, (40, 200), (360, 200), 0, thickness=7)
    cv2.line(img, (200, 200), (200, 207), 0, thickness=3)  # barba curta
    assert len(_strokes(img)) == 1
    cv2.line(img, (200, 200), (200, 330), 0, thickness=7)  # haste real
    assert len(_strokes(img)) == 2


def test_cross_pairs_into_two_straight_strokes():
    img = _canvas()
    cv2.line(img, (60, 60), (340, 340), 0, thickness=5)
    cv2.line(img, (60, 340), (340, 60), 0, thickness=5)
    strokes = _strokes(img)
    assert len(strokes) == 2
    for s in strokes:
        pts = s["points"]
        span = float(np.hypot(*(pts[-1] - pts[0])))
        assert span > 330.0, f"traço do X não atravessou a junção: {span:.1f}"


def test_tee_keeps_bar_and_leaves_stem_unpaired():
    img = _canvas()
    cv2.line(img, (60, 120), (340, 120), 0, thickness=5)
    cv2.line(img, (200, 120), (200, 340), 0, thickness=5)
    strokes = _strokes(img)
    assert len(strokes) == 2
    spans = sorted(
        float(np.hypot(*(s["points"][-1] - s["points"][0]))) for s in strokes
    )
    assert spans[1] > 270.0  # a barra inteira
    assert 200.0 < spans[0] < 230.0  # a haste sozinha


def test_thick_cross_contracts_double_junction():
    img = _canvas()
    # 'X' grosso a 60 graus: o afinamento parte o cruzamento em dois 'Y'
    # ligados por uma aresta curta; a contração junta os dois nós.
    # Limite conhecido: abaixo de ~45 graus a sobreposição passa de
    # 1,2*(r_u + r_v) e o 'X' sai em 3 traços (o legado dava 5).
    dx, dy = 170 * np.cos(np.radians(30)), 170 * np.sin(np.radians(30))
    p0, p1 = (int(200 - dx), int(200 - dy)), (int(200 + dx), int(200 + dy))
    q0, q1 = (int(200 - dx), int(200 + dy)), (int(200 + dx), int(200 - dy))
    cv2.line(img, p0, p1, 0, thickness=11)
    cv2.line(img, q0, q1, 0, thickness=11)
    raw = vectorize.traverse_graph(
        vectorize._make_8_minimal(vectorize.thin(img < 128)), include_topology=True
    )
    junctions = {
        nid
        for e in raw
        for nid, deg in (
            (e["start_node"], e["start_deg"]),
            (e["end_node"], e["end_deg"]),
        )
        if nid is not None and deg >= 3
    }
    assert len(junctions) >= 2, "a bancada precisa de um cruzamento partido"
    assert len(_strokes(img)) == 2


def test_closed_rectangle_stays_one_closed_stroke():
    img = _canvas()
    cv2.rectangle(img, (60, 80), (330, 300), 0, thickness=6)
    strokes = _strokes(img)
    assert len(strokes) == 1
    assert strokes[0]["closed"]


def test_topology_rejects_unknown_mode():
    try:
        vectorize.vectorize_mask(np.zeros((8, 8), dtype=bool), topology="v3")
    except ValueError:
        return
    raise AssertionError("topology desconhecida precisa falhar")


def _order_reference(enriched, first_idx):
    """O laço escalar anterior, referência do desempate."""
    remaining = [i for i in range(len(enriched)) if i != first_idx]
    ordered = [enriched[first_idx]]
    cur = ordered[0]["points"][0 if ordered[0]["closed"] else -1]
    while remaining:
        best_i, best_flip, best_d2 = None, False, 1e30
        for i in remaining:
            pts = enriched[i]["points"]
            d0 = float(np.sum((pts[0] - cur) ** 2))
            if d0 < best_d2:
                best_d2, best_i, best_flip = d0, i, False
            if not enriched[i]["closed"] and len(pts) > 1:
                d1 = float(np.sum((pts[-1] - cur) ** 2))
                if d1 < best_d2:
                    best_d2, best_i, best_flip = d1, i, True
        remaining.remove(best_i)
        chosen = dict(enriched[best_i])
        if best_flip:
            chosen["points"] = chosen["points"][::-1].copy()
        ordered.append(chosen)
        cur = chosen["points"][0 if chosen["closed"] else -1]
    return ordered


def test_vectorized_ordering_matches_scalar_reference_including_ties():
    rng = np.random.default_rng(7)
    lines = []
    for k in range(300):
        n = int(rng.integers(1, 6))
        # Coordenadas inteiras numa grade pequena forçam empates.
        pts = rng.integers(0, 40, size=(n, 2)).astype(np.float64)
        lines.append({"points": pts, "closed": bool(k % 7 == 0 and n >= 3)})
    got = vectorize.annotate_and_order_polylines(lines)
    first_idx = min(
        range(len(lines)),
        key=lambda i: float(
            np.hypot(lines[i]["points"][0, 0], lines[i]["points"][0, 1])
        ),
    )
    want = _order_reference(lines, first_idx)
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        assert np.array_equal(a["points"], b["points"])
        assert a["closed"] == b["closed"]


@pytest.mark.slow
def test_clean_path_on_dense_page_stays_fast():
    rng = np.random.default_rng(3)
    img = np.full((vectorize.PAPER_H_PX, vectorize.PAPER_W_PX), 255, dtype=np.uint8)
    for _ in range(400):
        x0, y0 = rng.integers(20, 830), rng.integers(20, 1180)
        dx, dy = rng.integers(-40, 41), rng.integers(-40, 41)
        cv2.line(img, (int(x0), int(y0)), (int(x0 + dx), int(y0 + dy)), 0, thickness=3)
    ink = img < 128
    vectorize.vectorize_clean(ink)
    t0 = time.perf_counter()
    vectorize.vectorize_clean(ink)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 1500.0, f"caminho limpo lento demais: {dt_ms:.0f} ms"


def test_line_crossing_two_close_parallels_keeps_each_line_whole():
    # Dois cruzamentos completos a 12 px um do outro não são um 'X'
    # partido: fundi-los ligava a ponta esquerda de uma paralela à
    # ponta direita da outra (review do PR #65).
    img = _canvas()
    cv2.line(img, (40, 150), (360, 150), 0, thickness=7)
    cv2.line(img, (40, 162), (360, 162), 0, thickness=7)
    cv2.line(img, (200, 40), (200, 360), 0, thickness=7)
    strokes = _strokes(img)
    assert len(strokes) == 3
    horizontals = []
    for s in strokes:
        a, b = s["points"][0], s["points"][-1]
        if abs(a[1] - b[1]) < abs(a[0] - b[0]):
            horizontals.append((a, b))
    assert len(horizontals) == 2
    for a, b in horizontals:
        assert abs(a[1] - b[1]) <= 3.0, f"paralela trocou de ponta: {a} -> {b}"
        assert abs(a[0] - b[0]) > 300.0


def test_circle_with_tangent_stem_keeps_both_primitives():
    # O laço volta ao próprio nó (u == v): fica fora do pareamento,
    # como na costura legada (review do PR #65). Quando a haste grossa
    # se sobrepõe ao arco por um trecho, o nó de junção sai do laço e
    # o arco continua tangente na haste — leitura ambígua (um 'd'),
    # resolvida pela continuidade.
    img = np.full((150, 120), 255, dtype=np.uint8)
    cv2.circle(img, (60, 45), 20, 0, thickness=7)
    cv2.line(img, (40, 45), (40, 105), 0, thickness=7)
    kinds = sorted(p["type"] for p in vectorize.classify_primitives(_strokes(img)))
    assert "circle" in kinds, kinds
    assert "line" in kinds, kinds


def test_clean_json_is_strict_when_dust_removes_everything():
    img = np.full((vectorize.PAPER_H_PX, vectorize.PAPER_W_PX), 255, dtype=np.uint8)
    img[100:103, 100:105] = 0  # 15 px de tinta: abaixo de DUST_MIN_AREA

    def reject(token):
        raise ValueError(f"JSON fora do padrão: {token}")

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "c.png"
        out_json = Path(tmp) / "c.json"
        cv2.imwrite(str(png), img)
        rc = vectorize.main(
            [str(png), "--clean", "--json", str(out_json), "--out", str(png) + ".svg"]
        )
        assert rc == 0
        payload = json.loads(
            out_json.read_text(encoding="utf-8"), parse_constant=reject
        )
        assert payload["summary"]["polilinhas"] == 0


def test_loop_at_junction_does_not_force_sharp_join_between_branches():
    # O laço preso ao nó segue contando no grau: os dois ramos que
    # saem dele em 'V' passam pelo custo da junção, não pela passagem
    # obrigatória de grau 2 (review do PR #65). Montado direto no
    # grafo: no raster o afinamento abre o 'V' alguns px fora do laço
    # e um dos ramos herda esse toco (limite conhecido).
    node = np.array([80.0, 80.0])
    loop = np.array([node, [80.0, 30.0], [30.0, 30.0], [30.0, 130.0], [80.0, 130.0]])
    items = [
        {"points": loop, "closed": True, "start_node": 1, "end_node": 1},
        {
            "points": np.array([node, [125.0, 68.0], [170.0, 56.0]]),
            "closed": False,
            "start_node": 1,
            "end_node": 2,
        },
        {
            "points": np.array([node, [125.0, 92.0], [170.0, 104.0]]),
            "closed": False,
            "start_node": 1,
            "end_node": 3,
        },
    ]
    graph = vectorize._TopoGraph(items, np.full((160, 200), 2.0, np.float32))
    assert graph.degrees()[1] == 4
    link = vectorize._pair_junctions(graph, vectorize.PAIR_TAU, 0.5)
    assert not link, f"ramos em V foram ligados: {link}"
    strokes = vectorize._chain_strokes(graph, link) + graph.loose
    assert sorted(s["closed"] for s in strokes) == [False, False, True]


def test_loop_connector_to_nearby_tee_is_not_contracted():
    # Laço com um conector de 13 px até um 'T' (raios 6 e 7: limite de
    # contração 15,6 px). O nó do laço é grau 3 sem ser metade de um
    # 'X'; contrair apagava o conector e soltava o laço (review do
    # PR #65).
    dist = np.zeros((200, 200), np.float32)
    dist[100, 60] = 6.0  # nó do laço
    dist[100, 73] = 7.0  # nó do 'T'
    loop_node, tee = np.array([60.0, 100.0]), np.array([73.0, 100.0])
    loop = np.array([loop_node, [40.0, 80.0], [20.0, 100.0], [40.0, 120.0]])
    items = [
        {"points": loop, "closed": True, "start_node": 1, "end_node": 1},
        {
            "points": np.array([loop_node, [66.0, 100.0], tee]),
            "closed": False,
            "start_node": 1,
            "end_node": 2,
        },
        {
            "points": np.array([tee, [73.0, 60.0], [73.0, 20.0]]),
            "closed": False,
            "start_node": 2,
            "end_node": 3,
        },
        {
            "points": np.array([tee, [73.0, 140.0], [73.0, 180.0]]),
            "closed": False,
            "start_node": 2,
            "end_node": 4,
        },
    ]
    graph = vectorize._TopoGraph(items, dist)
    vectorize._prune_and_contract(graph, 1.5, 1.2)
    assert all(e["alive"] for e in graph.edges), "conector do laço foi contraído"
    link = vectorize._pair_junctions(graph, vectorize.PAIR_TAU, 0.5)
    strokes = vectorize._chain_strokes(graph, link) + graph.loose
    ends = [tuple(p) for s in strokes if not s["closed"] for p in s["points"][[0, -1]]]
    assert (60.0, 100.0) in ends, "o laço ficou sem conector"
