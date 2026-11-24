"""Vetorização offline do canvas de tinta: raster -> grafo limpo -> polilinhas/primitivas -> SVG/JSON.

Ferramenta offline, sem câmera: recebe o PNG de um canvas acumulado
(850x1200, espaço do papel) — e opcionalmente o arquivo `_stamps.npz`
gerado ao salvar no Draw — e devolve as polilinhas e primitivas
geométricas (Camada 2 de VISION.md: reta, círculo, retângulo, polilinha).

Pipeline:
  1. binarização + preenchimento opcional de pinholes (<= 9 px) +
     filtro opcional de microfragmentos de poeira;
  2. afinamento com cv2.ximgproc.thinning (Zhang-Suen ou Guo-Hall);
  3. travessia indexada O(H*W) do esqueleto como grafo (~225x mais
     rápida que varredura por nó em páginas densas);
  4. higiene topológica opcional (topology="v2", a de vectorize_clean):
     poda radial de esporões (1,5*r), contração de junções vizinhas
     (1,2*(r_u+r_v)) e pareamento de ramos por tangente + espessura
     em cada junção;
  5. ordenação temporal (por first_seen_ms) ou espacial (vizinho mais
     próximo com inversão de sentido para reduzir saltos de caneta);
  6. simplificação Douglas-Peucker + classificação de primitivas
     geométricas (line, circle, rectangle, polygon, polyline) e
     exportação SVG/JSON em milímetros do A4.

Uso:
  python -m draw.exporters.vectorize canvas.png [--out saida.svg] [--clean]
      [--stamps canvas_stamps.npz] [--primitives] [--json saida.json]
      [--epsilon 1.5] [--thinning zhangsuen|guohall] [--stats]
  python -m draw.exporters.vectorize --selfcheck
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from .export_vectors import (
    _dp_open,
    _keep_component_pixel,
    filter_microfragments,
    simplify,
    traverse_graph,
)
from .export_vectors import (
    _index_node_components as _index_node_components,
)
from .export_vectors import (
    make_8_minimal as _make_8_minimal,
)

PAPER_W_PX = 850
PAPER_H_PX = 1200
PAPER_W_MM = 210.0
PAPER_H_MM = 297.0
MM_PER_PX_X = PAPER_W_MM / PAPER_W_PX
MM_PER_PX_Y = PAPER_H_MM / PAPER_H_PX
MM_PER_PX = 0.5 * (MM_PER_PX_X + MM_PER_PX_Y)


# Modo de medida: False = legado A4; True = papel custom. É a FLAG que
# distingue os dois modos, nunca o valor da escala — um papel explícito
# pode reproduzir as razões do default (ex.: 420x594 mm em 1700x2400 px)
# e mesmo assim pede a medida métrica por eixo.
_PAPER_OVERRIDE = False


def _set_paper_mm(paper_w_mm, paper_h_mm, img_w_px, img_h_px):
    """Reaponta a escala px->mm para uma folha não-A4 (override de CLI).

    Os denominadores em px vêm da imagem carregada, não das constantes
    ``PAPER_W_PX``/``PAPER_H_PX``: uma folha paisagem chega com o canvas
    já rotacionado (ex.: 1200x850), então a grade legada não se aplica.
    """
    global PAPER_W_MM, PAPER_H_MM, MM_PER_PX_X, MM_PER_PX_Y, MM_PER_PX
    global _PAPER_OVERRIDE
    PAPER_W_MM = float(paper_w_mm)
    PAPER_H_MM = float(paper_h_mm)
    MM_PER_PX_X = PAPER_W_MM / img_w_px
    MM_PER_PX_Y = PAPER_H_MM / img_h_px
    MM_PER_PX = 0.5 * (MM_PER_PX_X + MM_PER_PX_Y)
    _PAPER_OVERRIDE = True


def _reset_paper_mm():
    """Restaura escala e modo de medida do A4 default (finally de main)."""
    global _PAPER_OVERRIDE
    _set_paper_mm(210.0, 297.0, PAPER_W_PX, PAPER_H_PX)
    _PAPER_OVERRIDE = False


def _decision_space():
    """Espaço em que a geometria das primitivas é decidida.

    Devolve ``(scale_dec, unit_mm, mm_axis)``: pontos de decisão =
    ``pts_px * scale_dec``, medidas escalares = ``len_dec * unit_mm`` e
    coordenadas de saída = ``p_dec * mm_axis``.

    Sem override de papel, o espaço de decisão é o próprio raster em px,
    as medidas saem pela média legada ``MM_PER_PX`` e as coordenadas pela
    escala de cada eixo — bit a bit idêntico ao histórico. Com papel
    custom, os pontos são normalizados para "pixels isotrópicos" — a
    correção clássica de pixel aspect ratio, sem reamostrar o raster:
    ``pts * (sx/ref, sy/ref)`` com ``ref = min(sx, sy)`` torna a métrica
    uniforme (1 unidade = ``ref`` mm em qualquer direção). Nesse espaço
    as mesmas tolerâncias do legado (gap 8, raio 6, desvios 2.5/3.5/4,
    lado 10) medem geometria real: um círculo que o cisalhamento px->mm
    tornou elipse no raster volta a ser redondo, e um oval do raster
    não vira círculo na saída.
    """
    if not _PAPER_OVERRIDE:
        return np.ones(2), MM_PER_PX, np.array([MM_PER_PX_X, MM_PER_PX_Y])
    ref = min(MM_PER_PX_X, MM_PER_PX_Y)
    return np.array([MM_PER_PX_X / ref, MM_PER_PX_Y / ref]), ref, np.full(2, ref)


# Nomes estaticos: o cv2.ximgproc so e resolvido dentro de thin(), para
# que o import do modulo nao quebre num wheel opencv-*-headless sem
# contrib (AttributeError em nivel de modulo matava o arquivo inteiro).
THINNING_MODES = ("zhangsuen", "guohall")


def _thinning_type(name):
    """cv2.ximgproc.THINNING_* para `name`, resolvido no uso.

    Sem ximgproc (wheel sem contrib), levanta RuntimeError explicando a
    dependencia — o resto do modulo (grafo, primitivas, writers) segue
    utilizavel sem ele.
    """
    if name not in THINNING_MODES:
        raise ValueError(f"thinning desconhecido: {name!r}")
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None:
        raise RuntimeError(
            "cv2.ximgproc indisponivel: o afinamento de thin()/vectorize_mask "
            "exige opencv-contrib-python-headless"
        )
    return getattr(ximgproc, "THINNING_" + name.upper())


def load_ink(path):
    """Lê um PNG de canvas e devolve (mascara booleana de tinta, polaridade)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"nao foi possivel ler {path}")
    dark = img < 128
    if int(dark.sum()) <= int((~dark).sum()):
        return dark, "escura"
    return ~dark, "clara"


def load_stamps(path):
    """Carrega planos de carimbo (`first_seen_ms`, `last_update_ms`) de um `.npz`."""
    data = np.load(str(path), allow_pickle=False)
    first = data["first_seen_ms"] if "first_seen_ms" in data else None
    last = (
        data["last_update_ms"]
        if "last_update_ms" in data
        else (data["last_seen_ms"] if "last_seen_ms" in data else None)
    )
    return {"first_seen_ms": first, "last_update_ms": last}


def thin(ink, thinning="zhangsuen"):
    """Afina a máscara de tinta para um esqueleto de 1 px (booleano).

    Como zhang_suen_thin de export_vectors, conserva 1 px (o mais
    proximo do centroide) de cada componente que o ximgproc apagou por
    completo — um ponto 2x2 com min_area=0 nao some do resultado.
    """
    src = ink.astype(np.uint8)
    thinning_type = _thinning_type(thinning)  # valida ximgproc antes do acesso
    skel = cv2.ximgproc.thinning(src * np.uint8(255), None, thinning_type)
    return _keep_component_pixel(src, skel > 0)


def _polyline_arc_length(points):
    if len(points) <= 1:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


# --- Higiene topológica v2 (Etapas A/B/C) -------------------------------
# Medido em 2026-11-19 nos quatro clipes reais da régua (e5 e agora
# estáticos, mão-sombra, folha-movida; canvas confirmado do page_clean
# v2 com ruling + histerese): a Etapa A sozinha derrubou os ciclos
# falsos do grafo de 265 para 14 (-95%), e A+B+C levou os estáticos de
# ~960 para ~510 traços em ~107 ms (thinning incluso), contra o ciclo
# de 300 ms da passada. Nenhum prior temporal entra aqui: o relógio do
# canvas tem a resolução das publicações do page_clean (6-11 por clipe,
# ~1 s), grossa demais para desempatar cruzamentos.
PINHOLE_MAX_AREA = 9  # buraco 4-conexo <= 9 px dentro do traço vira tinta
DUST_MIN_AREA = 25  # componente 8-conexa < 25 px sai antes do afinamento
SPUR_RADIUS_RATIO = 1.5  # ramo-folha mais curto que 1,5*r da junção é esporão
CONTRACT_RADIUS_RATIO = 1.2  # junções a <= 1,2*(r_u + r_v) viram uma só
PAIR_TAU = 0.25  # custo máximo para dois ramos atravessarem a junção juntos
PAIR_RADIUS_WEIGHT = 0.5  # peso da diferença relativa de espessura no custo
_PAIR_EXACT_MAX = 10  # até 10 ramos na junção o pareamento é exato (DP)


def fill_pinholes(ink, max_area=PINHOLE_MAX_AREA):
    """Preenche buracos minúsculos dentro do traço (Etapa A).

    Um furo de 1-9 px no meio de um traço grosso vira um ciclo no
    esqueleto — o afinamento contorna o furo — e cada ciclo falso
    parte o traço em junções que não existem. Buracos 4-conexos que
    não tocam a borda e têm área <= max_area são tinta; o miolo de um
    'o' ou de um laço (dezenas de px) fica.
    """
    ink = ink.astype(bool, copy=False)
    if max_area <= 0:
        return ink.copy()
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (~ink).astype(np.uint8), connectivity=4
    )
    x0 = stats[:, cv2.CC_STAT_LEFT]
    y0 = stats[:, cv2.CC_STAT_TOP]
    x1 = x0 + stats[:, cv2.CC_STAT_WIDTH]
    y1 = y0 + stats[:, cv2.CC_STAT_HEIGHT]
    fill = (
        (stats[:, cv2.CC_STAT_AREA] <= int(max_area))
        & (x0 > 0)
        & (y0 > 0)
        & (x1 < w)
        & (y1 < h)
    )
    fill[0] = False
    if n <= 1 or not fill.any():
        return ink.copy()
    return ink | fill[labels]


class _TopoGraph:
    """Grafo do esqueleto: nós com raio local e arestas com polilinha."""

    def __init__(self, items, dist):
        self.dist = dist
        self.edges = []  # dicts: u, v, pts, length, alive
        self.loose = []  # itens sem nós (anéis livres, pontos isolados)
        self.loops = []  # nó de cada laço fechado preso a um nó (u == v)
        self.center = {}
        self.radius = {}
        self.parent = {}
        for item in items:
            u, v = item.get("start_node"), item.get("end_node")
            pts = item["points"]
            if u is None or v is None or len(pts) < 2 or item["closed"]:
                # Laço fechado que volta ao próprio nó (o círculo com
                # uma haste tangente) sai como traço próprio e fica
                # fora do pareamento: ligar uma ponta dele à haste
                # trocaria um círculo e uma reta por uma polilinha
                # aberta. As duas pontas dele seguem contando no grau
                # do nó (ver degrees).
                self.loose.append({"points": pts, "closed": item["closed"]})
                if item["closed"] and u is not None and u == v:
                    self._add_node(u, pts[0])
                    self.loops.append(u)
                continue
            self._add_node(u, pts[0])
            self._add_node(v, pts[-1])
            self.edges.append(
                {
                    "u": u,
                    "v": v,
                    "pts": pts,
                    "length": _polyline_arc_length(pts),
                    "alive": True,
                }
            )

    def _add_node(self, nid, point):
        if nid in self.center:
            return
        h, w = self.dist.shape
        self.center[nid] = np.asarray(point, dtype=np.float64).copy()
        x = int(np.clip(round(float(point[0])), 0, w - 1))
        y = int(np.clip(round(float(point[1])), 0, h - 1))
        self.radius[nid] = max(1.0, float(self.dist[y, x]))
        self.parent[nid] = nid

    def find(self, nid):
        root = nid
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[nid] != root:
            self.parent[nid], nid = root, self.parent[nid]
        return root

    def degrees(self):
        deg = {}
        for e in self.edges:
            if e["alive"]:
                for nid in (self.find(e["u"]), self.find(e["v"])):
                    deg[nid] = deg.get(nid, 0) + 1
        for nid in self.loops:
            root = self.find(nid)
            deg[root] = deg.get(root, 0) + 2
        return deg


def _prune_and_contract(graph, spur_ratio, contract_ratio):
    """Etapa B: poda radial de esporões e contração de junções vizinhas.

    Esporão é o ramo-folha preso a uma junção (grau >= 3) mais curto
    que spur_ratio vezes o raio do traço ali — a barba que o
    afinamento deixa nos cantos de traço grosso. Duas junções ligadas
    por uma aresta mais curta que contract_ratio*(r_u + r_v) são o
    mesmo cruzamento visto como dois nós (o 'X' grosso afina em dois
    'Y'); viram um nó no centróide ponderado pelos raios. Só o par
    3+3 contrai: dois cruzamentos completos (grau 4) próximos — uma
    reta cortando duas paralelas — são cruzamentos de verdade, e
    fundi-los trocava as pontas das paralelas entre si.
    """
    changed = True
    while changed:
        changed = False
        deg = graph.degrees()
        for e in graph.edges:
            if not e["alive"]:
                continue
            u, v = graph.find(e["u"]), graph.find(e["v"])
            leaf_u, leaf_v = deg.get(u) == 1, deg.get(v) == 1
            if leaf_u == leaf_v:
                continue
            hub = v if leaf_u else u
            if deg[hub] >= 3 and e["length"] < spur_ratio * graph.radius[hub]:
                e["alive"] = False
                deg[u] -= 1
                deg[v] -= 1
                changed = True
    loop_nodes = {graph.find(nid) for nid in graph.loops}
    deg = graph.degrees()
    for e in graph.edges:
        if not e["alive"]:
            continue
        u, v = graph.find(e["u"]), graph.find(e["v"])
        if u == v or deg.get(u, 0) != 3 or deg.get(v, 0) != 3:
            continue
        if u in loop_nodes or v in loop_nodes:
            # Nó de laço com um conector é grau 3 sem ser metade de um
            # 'X': contrair levava o conector embora e deixava o laço,
            # que não se move com o centro, solto (review do PR #65).
            continue
        ru, rv = graph.radius[u], graph.radius[v]
        if e["length"] > contract_ratio * (ru + rv):
            continue
        graph.center[u] = (graph.center[u] * ru + graph.center[v] * rv) / (ru + rv)
        graph.radius[u] = max(ru, rv)
        graph.parent[v] = u
        e["alive"] = False
        # Grau do nó fundido sem varrer o grafo: as duas pontas da
        # aresta contraída saem; o resto de v passa a contar em u
        # (grau 4, que não volta a contrair).
        deg[u] += deg.pop(v) - 2


def _half_edge_feature(graph, edge, at_start, root):
    """Tangente de saída e raio mediano de um ramo junto à junção."""
    pts = edge["pts"] if at_start else edge["pts"][::-1]
    c = graph.center[root]
    r = graph.radius[root]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    # A tangente é lida fora do disco da junção (>= 2r): dentro dele o
    # esqueleto entorta para o centro e todos os ramos parecem iguais.
    k = int(np.searchsorted(cum, max(2.0 * r, 3.0)))
    k = min(max(k, 1), len(pts) - 1)
    vec = pts[k] - c
    norm = float(np.hypot(vec[0], vec[1]))
    tangent = vec / norm if norm > 1e-6 else np.array([1.0, 0.0])
    h, w = graph.dist.shape
    sample = pts[: k + 1]
    ys = np.clip(np.round(sample[:, 1]).astype(int), 0, h - 1)
    xs = np.clip(np.round(sample[:, 0]).astype(int), 0, w - 1)
    rad = max(1.0, float(np.median(graph.dist[ys, xs])))
    return tangent, rad


def _max_weight_pairing(cost, tau):
    """Pares (i, j) que maximizam a soma de (tau - custo), cada ramo uma vez.

    Exato por DP em máscara até _PAIR_EXACT_MAX ramos (junções reais
    têm 3-6); acima disso, guloso pelo menor custo.
    """
    k = len(cost)
    if k > _PAIR_EXACT_MAX:
        cand = sorted(
            (cost[i][j], i, j)
            for i in range(k)
            for j in range(i + 1, k)
            if cost[i][j] < tau
        )
        used, out = set(), []
        for _c, i, j in cand:
            if i not in used and j not in used:
                used.update((i, j))
                out.append((i, j))
        return out

    full = (1 << k) - 1
    memo = {}

    def best(mask):
        if mask == full:
            return 0.0, ()
        if mask in memo:
            return memo[mask]
        i = 0
        while mask >> i & 1:
            i += 1
        top_w, top_p = best(mask | 1 << i)  # i fica sem par
        for j in range(i + 1, k):
            if mask >> j & 1 or cost[i][j] >= tau:
                continue
            w_rest, p_rest = best(mask | 1 << i | 1 << j)
            w_pair = w_rest + tau - cost[i][j]
            if w_pair > top_w:
                top_w, top_p = w_pair, ((i, j), *p_rest)
        memo[mask] = (top_w, top_p)
        return memo[mask]

    return list(best(0)[1])


def _pair_junctions(graph, tau, radius_weight):
    """Etapa C: liga as meias-arestas que atravessam cada nó.

    Em nó de grau 2 a passagem é obrigatória. Em junção, o custo de
    atravessar do ramo a para o b é (1 + t_a.t_b)/2 (0 quando saem em
    sentidos opostos) mais radius_weight vezes a diferença relativa
    de espessura; o pareamento de peso máximo abaixo de tau decide
    quem continua quem. Devolve {(edge, end): (edge, end)}.
    """
    incident = {}
    for idx, e in enumerate(graph.edges):
        if e["alive"]:
            incident.setdefault(graph.find(e["u"]), []).append((idx, 0))
            incident.setdefault(graph.find(e["v"]), []).append((idx, 1))
    loop_ends = {}
    for nid in graph.loops:
        root = graph.find(nid)
        loop_ends[root] = loop_ends.get(root, 0) + 2
    link = {}
    for root, halves in incident.items():
        # Passagem obrigatória só em nó de grau 2 de verdade: com um
        # laço preso ao nó, os dois ramos restantes são junção e
        # passam pelo custo (dois ramos em 'V' não viram um traço).
        if len(halves) == 2 and not loop_ends.get(root):
            a, b = halves
            link[a] = b
            link[b] = a
            continue
        if len(halves) < 2:
            continue
        feats = [
            _half_edge_feature(graph, graph.edges[i], end == 0, root)
            for i, end in halves
        ]
        k = len(halves)
        cost = [[math.inf] * k for _ in range(k)]
        for a in range(k):
            ta, ra = feats[a]
            for b in range(a + 1, k):
                tb, rb = feats[b]
                c = (1.0 + float(ta @ tb)) / 2.0
                c += radius_weight * abs(ra - rb) / max(ra, rb)
                cost[a][b] = cost[b][a] = c
        for a, b in _max_weight_pairing(cost, tau):
            link[halves[a]] = halves[b]
            link[halves[b]] = halves[a]
    return link


def _edge_points(graph, idx, forward):
    """Polilinha da aresta no sentido pedido, pontas no centro do nó."""
    e = graph.edges[idx]
    pts = e["pts"].copy()
    pts[0] = graph.center[graph.find(e["u"])]
    pts[-1] = graph.center[graph.find(e["v"])]
    return pts if forward else pts[::-1]


def _chain_strokes(graph, link):
    """Encadeia as arestas pelos pares das junções em traços."""
    alive = [i for i, e in enumerate(graph.edges) if e["alive"]]
    used = set()
    strokes = []

    def walk(start_idx, start_end):
        # Entra pela ponta start_end e segue pelo outro lado.
        idx, entry = start_idx, start_end
        parts = []
        while True:
            used.add(idx)
            parts.append(_edge_points(graph, idx, forward=entry == 0))
            nxt = link.get((idx, 1 - entry))
            if nxt is None:
                return parts, False
            if nxt[0] == start_idx and nxt[1] == start_end:
                return parts, True
            if nxt[0] in used:
                return parts, False
            idx, entry = nxt

    # Primeiro as cadeias abertas, a partir de uma ponta sem par.
    for idx in alive:
        if idx in used:
            continue
        for end in (0, 1):
            if (idx, end) not in link:
                parts, closed = walk(idx, end)
                strokes.append((parts, closed))
                break
    # O que sobrou são ciclos (toda ponta tem par).
    for idx in alive:
        if idx not in used:
            strokes.append(walk(idx, 0))

    out = []
    for parts, closed in strokes:
        pts = np.vstack([parts[0]] + [p[1:] for p in parts[1:]])
        if closed and len(pts) >= 4:
            out.append({"points": pts[:-1], "closed": True})
        else:
            out.append({"points": pts, "closed": False})
    return out


def topology_strokes(
    skel,
    ink,
    *,
    spur_ratio=SPUR_RADIUS_RATIO,
    contract_ratio=CONTRACT_RADIUS_RATIO,
    tau=PAIR_TAU,
    radius_weight=PAIR_RADIUS_WEIGHT,
):
    """Esqueleto -> traços (Etapas B e C): poda, contração e pareamento.

    Mesmo formato de saída de traverse_graph sem topologia:
    lista de {"points": Nx2 (x, y) px, "closed": bool}.
    """
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    items = traverse_graph(_make_8_minimal(skel), include_topology=True)
    graph = _TopoGraph(items, dist)
    _prune_and_contract(graph, spur_ratio, contract_ratio)
    link = _pair_junctions(graph, tau, radius_weight)
    return _chain_strokes(graph, link) + graph.loose


def annotate_and_order_polylines(polylines, stamps=None):
    """Anota carimbos temporais quando presentes e ordena para minimizar saltos."""
    first_plane = None
    last_plane = None
    if isinstance(stamps, dict):
        first_plane = stamps.get("first_seen_ms")
        last_plane = stamps.get("last_update_ms")
    elif isinstance(stamps, np.ndarray):
        first_plane = stamps

    enriched = []
    for item in polylines:
        pts = item["points"].copy()
        closed = bool(item["closed"])
        t_start = None
        t_end = None
        if first_plane is not None and len(pts) > 0:
            h, w = first_plane.shape
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            vals = first_plane[ys, xs]
            valid = vals[vals > 0]
            if len(valid) > 0:
                t_start = int(np.percentile(valid, 10))
                lvals = last_plane[ys, xs] if last_plane is not None else vals
                lvalid = lvals[lvals > 0]
                t_end = int(np.percentile(lvalid, 90)) if len(lvalid) > 0 else t_start
                # Orienta polilinha aberta do menor para o maior carimbo.
                if not closed and len(pts) >= 4:
                    q = max(1, len(pts) // 4)
                    head_v = vals[:q][vals[:q] > 0]
                    tail_v = vals[-q:][vals[-q:] > 0]
                    if len(head_v) > 0 and len(tail_v) > 0:
                        if float(np.median(head_v)) > float(np.median(tail_v)):
                            pts = pts[::-1].copy()
        out = {"points": pts, "closed": closed}
        if t_start is not None:
            out["t_start_ms"] = t_start
            out["t_end_ms"] = t_end
        enriched.append(out)

    if len(enriched) <= 1:
        return enriched

    if any("t_start_ms" in it for it in enriched):
        return sorted(
            enriched,
            key=lambda it: (
                it.get("t_start_ms", 1 << 60),
                float(it["points"][0, 1]),
                float(it["points"][0, 0]),
            ),
        )

    # Sem carimbos: encadeamento guloso pelo extremo mais próximo para
    # minimizar o deslocamento com caneta levantada (saltos no plotter/SVG).
    remaining = list(range(len(enriched)))
    # Começa pelo traço mais próximo da origem superior-esquerda:
    first_idx = min(
        remaining,
        key=lambda i: float(
            np.hypot(enriched[i]["points"][0, 0], enriched[i]["points"][0, 1])
        ),
    )
    remaining.remove(first_idx)
    ordered = [enriched[first_idx]]
    cur_pos = ordered[0]["points"][0 if ordered[0]["closed"] else -1]

    # Pontas candidatas em vetores: a varredura escalar (np.sum por
    # traço, O(n^2) chamadas) custava ~0,7 s em páginas reais de 400+
    # traços. Intercalar (início, fim) por traço e pegar o primeiro
    # argmin reproduz o desempate do laço antigo (o '<' estrito
    # varria início antes do fim, traço a traço, em ordem de índice).
    n = len(enriched)
    heads = np.array([it["points"][0] for it in enriched], dtype=np.float64)
    tails = np.array([it["points"][-1] for it in enriched], dtype=np.float64)
    can_flip = np.array(
        [(not it["closed"]) and len(it["points"]) > 1 for it in enriched]
    )
    live = np.ones(n, dtype=bool)
    live[first_idx] = False
    for _ in range(n - 1):
        idx = np.flatnonzero(live)
        d0 = ((heads[idx] - cur_pos) ** 2).sum(axis=1)
        d1 = ((tails[idx] - cur_pos) ** 2).sum(axis=1)
        d1[~can_flip[idx]] = np.inf
        pick = int(np.argmin(np.column_stack([d0, d1]).ravel()))
        best_i = int(idx[pick // 2])
        best_flip = bool(pick % 2)
        live[best_i] = False
        chosen = dict(enriched[best_i])
        if best_flip:
            chosen["points"] = chosen["points"][::-1].copy()
        ordered.append(chosen)
        cur_pos = chosen["points"][0 if chosen["closed"] else -1]
    return ordered


def _fit_circle_kasa(pts):
    """Ajuste algébrico de mínimos quadrados (Kåsa) para (cx, cy, r)."""
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank < 3:
        return None
    cx, cy, c0 = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c0 + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def _merge_collinear_closed_vertices(verts, collinear_tol_deg=16.0):
    """Remove vértices quase rasos (~180°) ao longo das arestas de um polígono fechado."""
    pts = list(verts)
    if len(pts) >= 2 and np.hypot(*(pts[0] - pts[-1])) < 1e-6:
        pts = pts[:-1]
    changed = True
    while changed and len(pts) > 4:
        changed = False
        n = len(pts)
        for i in range(n):
            p_prev = pts[(i - 1) % n]
            p_cur = pts[i]
            p_next = pts[(i + 1) % n]
            v1 = p_cur - p_prev
            v2 = p_next - p_cur
            n1 = float(np.hypot(v1[0], v1[1]))
            n2 = float(np.hypot(v2[0], v2[1]))
            if n1 < 2.0 or n2 < 2.0:
                pts.pop(i)
                changed = True
                break
            cos_turn = float(np.dot(v1, v2) / (n1 * n2))
            turn_deg = math.degrees(math.acos(min(1.0, max(-1.0, cos_turn))))
            if turn_deg <= collinear_tol_deg:
                pts.pop(i)
                changed = True
                break
    return np.array(pts, dtype=np.float64)


def classify_primitive(polyline):
    """Classifica uma polilinha em primitiva geométrica da Camada 2 (VISION.md).

    Retorna um dicionário com `type` em `{"point", "line", "circle",
    "rectangle", "polygon", "polyline"}` e parâmetros métricos em mm/px.
    """
    scale_dec, unit_mm, mm_axis = _decision_space()
    pts = np.asarray(polyline["points"], dtype=np.float64) * scale_dec
    closed = bool(polyline["closed"])
    meta = {}
    if "t_start_ms" in polyline:
        meta["t_start_ms"] = polyline["t_start_ms"]
        meta["t_end_ms"] = polyline["t_end_ms"]
    if "width_px" in polyline:
        # stroke_w_mm, não width_mm: o retângulo já usa width_mm para a
        # largura geométrica da caixa — colidir os dois o apagaria.
        meta["stroke_w_mm"] = round(float(polyline["width_px"]) * MM_PER_PX, 3)

    if len(pts) == 1:
        return {
            "type": "point",
            "x_mm": round(float(pts[0, 0] * mm_axis[0]), 3),
            "y_mm": round(float(pts[0, 1] * mm_axis[1]), 3),
            **meta,
        }

    dense = dense_samples(pts, closed, step=2.0)

    # 1. Reta aberta:
    if not closed:
        p0, p1 = pts[0], pts[-1]
        chord_vec = p1 - p0
        chord_len = float(np.hypot(chord_vec[0], chord_vec[1]))
        arc_len = _polyline_arc_length(pts)
        if chord_len >= 10.0 and arc_len <= 1.05 * chord_len:
            cross = np.abs(
                chord_vec[0] * (p0[1] - dense[:, 1])
                - (p0[0] - dense[:, 0]) * chord_vec[1]
            )
            max_dev = float(np.max(cross) / chord_len)
            if max_dev <= max(2.5, 0.025 * chord_len):
                return {
                    "type": "line",
                    "x1_mm": round(float(p0[0] * mm_axis[0]), 3),
                    "y1_mm": round(float(p0[1] * mm_axis[1]), 3),
                    "x2_mm": round(float(p1[0] * mm_axis[0]), 3),
                    "y2_mm": round(float(p1[1] * mm_axis[1]), 3),
                    "length_mm": round(float(chord_len * unit_mm), 3),
                    "max_dev_px": round(max_dev, 3),
                    **meta,
                }

    # 2. Círculo (fechado ou quase fechado):
    end_gap = 0.0 if closed else float(np.hypot(*(pts[-1] - pts[0])))
    perim = _polyline_arc_length(np.vstack([pts, pts[:1]]) if closed else pts)
    if (closed or end_gap <= max(8.0, 0.08 * perim)) and len(dense) >= 16:
        fit = _fit_circle_kasa(dense)
        if fit is not None:
            cx, cy, r = fit
            if r >= 6.0:
                radii = np.hypot(dense[:, 0] - cx, dense[:, 1] - cy)
                rmse = float(np.sqrt(np.mean((radii - r) ** 2)))
                max_err = float(np.max(np.abs(radii - r)))
                theta = np.arctan2(dense[:, 1] - cy, dense[:, 0] - cx)
                bins = np.unique((np.mod(np.degrees(theta), 360.0) // 15.0).astype(int))
                # >= 21 de 24 setores de 15° cobertos (>= 315°).
                circle_like = (
                    len(bins) >= 21
                    and rmse / r <= 0.055
                    and max_err <= max(3.5, 0.10 * r)
                )
                if circle_like and _PAPER_OVERRIDE:
                    # No espaço normalizado a excentricidade é física: um
                    # oval tem o desvio radial dominado pelo 2º harmônico
                    # (polos opostos sistemáticos), ruído de raster não.
                    # Sem essa auditoria uma elipse suave com rmse baixo
                    # exportaria como círculo de raio único.
                    h2 = np.column_stack([np.cos(2.0 * theta), np.sin(2.0 * theta)])
                    coef, _, _, _ = np.linalg.lstsq(h2, radii - r, rcond=None)
                    circle_like = float(np.hypot(coef[0], coef[1])) <= max(
                        1.5, 0.035 * r
                    )
                if circle_like:
                    return {
                        "type": "circle",
                        "cx_mm": round(cx * mm_axis[0], 3),
                        "cy_mm": round(cy * mm_axis[1], 3),
                        "r_mm": round(r * unit_mm, 3),
                        "radial_rmse_px": round(rmse, 3),
                        **meta,
                    }

    # 3. Retângulo fechado:
    if closed and len(pts) >= 4:
        coarse = _dp_open(np.vstack([pts, pts[:1]]), epsilon=3.0)
        corners = _merge_collinear_closed_vertices(coarse)
        if len(corners) == 4:
            sides = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
            lengths = [float(np.hypot(s[0], s[1])) for s in sides]
            if min(lengths) >= 10.0:
                angles_ok = True
                for i in range(4):
                    u = -sides[(i - 1) % 4] / lengths[(i - 1) % 4]
                    v = sides[i] / lengths[i]
                    ang = math.degrees(
                        math.acos(min(1.0, max(-1.0, float(np.dot(u, v)))))
                    )
                    if abs(ang - 90.0) > 12.0:
                        angles_ok = False
                        break
                opp0 = abs(lengths[0] - lengths[2]) / max(lengths[0], lengths[2])
                opp1 = abs(lengths[1] - lengths[3]) / max(lengths[1], lengths[3])
                # Garante também que o contorno real segue os 4 lados.
                # O desvio é medido com sinal sobre a normal externa de
                # cada lado: um contorno curvo (oval simplificado a 4
                # cantos) fica sistematicamente FORA do quadrilátero
                # inscrito, enquanto as arestas de um retângulo real
                # oscilam em torno da aresta (~metade dentro, metade fora).
                center = corners.mean(axis=0)
                normals = []
                for i, s in enumerate(sides):
                    n = np.array([s[1], -s[0]], dtype=np.float64) / lengths[i]
                    if float(np.dot(n, center - corners[i])) > 0:
                        n = -n
                    normals.append(n)
                # "Fora" só conta acima de uma tolerância numérica: a
                # interpolação deixa resíduos de ~1e-14 em amostras que
                # estão exatamente sobre a aresta, e sem o eps a fração
                # de fora depende do sinal do último bit (posição).
                out_eps = (
                    64.0
                    * np.finfo(np.float64).eps
                    * max(1.0, float(np.abs(dense).max()))
                )
                rect_dev = []
                n_out = 0
                for p in dense:
                    ds = [float(np.dot(p - corners[i], normals[i])) for i in range(4)]
                    j = int(np.argmin(np.abs(ds)))
                    rect_dev.append(abs(ds[j]))
                    n_out += ds[j] > out_eps
                if (
                    angles_ok
                    and opp0 <= 0.12
                    and opp1 <= 0.12
                    and max(rect_dev) <= max(4.0, 0.05 * max(lengths))
                    and (not _PAPER_OVERRIDE or n_out / len(rect_dev) <= 0.7)
                ):
                    w_mm = 0.5 * (lengths[0] + lengths[2]) * unit_mm
                    h_mm = 0.5 * (lengths[1] + lengths[3]) * unit_mm
                    ang_deg = math.degrees(math.atan2(sides[0][1], sides[0][0]))
                    return {
                        "type": "rectangle",
                        "center_mm": [
                            round(float(center[0] * mm_axis[0]), 3),
                            round(float(center[1] * mm_axis[1]), 3),
                        ],
                        "width_mm": round(w_mm, 3),
                        "height_mm": round(h_mm, 3),
                        "angle_deg": round(ang_deg, 2),
                        "corners_mm": [
                            [
                                round(float(c[0] * mm_axis[0]), 3),
                                round(float(c[1] * mm_axis[1]), 3),
                            ]
                            for c in corners
                        ],
                        **meta,
                    }

    return {
        "type": "polygon" if closed else "polyline",
        "vertices": len(pts),
        "points_mm": [
            [round(float(x * mm_axis[0]), 3), round(float(y * mm_axis[1]), 3)]
            for x, y in pts
        ],
        **meta,
    }


def classify_primitives(polylines):
    """Classifica todas as polilinhas em primitivas geométricas estruturadas."""
    return [classify_primitive(line) for line in polylines]


def vectorize_mask(
    ink,
    epsilon=1.5,
    thinning="zhangsuen",
    *,
    min_area=0,
    stamps=None,
    order_strokes=False,
    fill_holes=False,
    topology="legacy",
):
    """Máscara de tinta -> polilinhas simplificadas (px, espaço do papel).

    topology="v2" aplica a higiene topológica (topology_strokes);
    "legacy" devolve a travessia crua do esqueleto. Os dois caminhos
    devolvem o mesmo formato.
    """
    if topology not in ("legacy", "v2"):
        raise ValueError(f"topology desconhecida: {topology!r}")
    work = fill_pinholes(ink) if fill_holes else ink
    work = filter_microfragments(work, min_area=min_area) if min_area > 1 else work
    skel = thin(work, thinning)
    if topology == "v2":
        raw = topology_strokes(skel, work)
    else:
        raw = traverse_graph(skel)
    if order_strokes or stamps is not None:
        raw = annotate_and_order_polylines(raw, stamps=stamps)
    out = []
    # Largura por traço amostrada no esqueleto cru (pseudo-pressão para os
    # exportadores): um distanceTransform da máscara limpa por vetorização.
    dt_width = cv2.distanceTransform(work.astype(np.uint8), cv2.DIST_L2, 5)
    w_h, w_w = work.shape[:2]
    for line in raw:
        pts = simplify(line["points"], epsilon, line["closed"])
        item = {"points": pts, "closed": line["closed"]}
        if "t_start_ms" in line:
            item["t_start_ms"] = line["t_start_ms"]
            item["t_end_ms"] = line["t_end_ms"]
        rpts = np.asarray(line["points"]).reshape(-1, 2)
        if len(rpts):
            xs = np.clip(np.round(rpts[:, 0]).astype(int), 0, w_w - 1)
            ys = np.clip(np.round(rpts[:, 1]).astype(int), 0, w_h - 1)
            item["width_px"] = round(float(2.0 * np.median(dt_width[ys, xs])), 2)
        out.append(item)
    return out


def vectorize_clean(ink, epsilon=1.5, thinning="zhangsuen", stamps=None):
    """Atalho recomendado para canvas reais: higiene topológica v2 e ordenação.

    Preenche pinholes, remove poeira (< DUST_MIN_AREA px), poda
    esporões, contrai junções e pareia ramos nas junções; os carimbos,
    quando presentes, só orientam e ordenam os traços.
    """
    return vectorize_mask(
        ink,
        epsilon=epsilon,
        thinning=thinning,
        min_area=DUST_MIN_AREA,
        fill_holes=True,
        topology="v2",
        stamps=stamps,
        order_strokes=True,
    )


def dense_samples(points, closed, step=1.0):
    """Amostra pontos ao longo da polilinha com espaçamento <= step px."""
    if len(points) == 1:
        return points.copy()
    pts = list(points)
    if closed:
        pts = pts + [pts[0]]
    samples = []
    for a, b in zip(pts[:-1], pts[1:], strict=False):
        length = float(np.hypot(*(np.array(b) - np.array(a))))
        n = max(1, int(np.ceil(length / step)))
        for i in range(n):
            t = i / n
            samples.append(np.array(a) * (1 - t) + np.array(b) * t)
    samples.append(np.array(pts[-1], dtype=np.float64))
    return np.array(samples)


def distance_stats(ink, polylines):
    """Distâncias polilinha<->tinta nos dois sentidos, em px e mm."""
    dt_to_ink = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)
    samples = [dense_samples(line["points"], line["closed"]) for line in polylines]
    samples = np.vstack(samples) if samples else np.zeros((0, 2))
    h, w = ink.shape
    if len(samples) > 0:
        sx = np.clip(np.round(samples[:, 0]).astype(int), 0, w - 1)
        sy = np.clip(np.round(samples[:, 1]).astype(int), 0, h - 1)
        d_fwd = dt_to_ink[sy, sx]
    else:
        d_fwd = np.zeros((0,), dtype=np.float32)

    raster = np.zeros(ink.shape, dtype=np.uint8)
    for line in polylines:
        pts = np.round(line["points"]).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) == 1:
            raster[pts[0, 0, 1], pts[0, 0, 0]] = 1
        else:
            cv2.polylines(
                raster,
                [pts],
                bool(line["closed"]),
                1,
                thickness=1,
                lineType=cv2.LINE_8,
            )
    if raster.any():
        dt_to_poly = cv2.distanceTransform(
            (raster == 0).astype(np.uint8), cv2.DIST_L2, 5
        )
        d_bwd = dt_to_poly[ink]
    else:
        # Sem polilinha nenhuma (ex.: a poeira levou tudo) o
        # distanceTransform não tem alvo e devolve o sentinela de
        # float32 — a média estourava para Infinity no JSON.
        d_bwd = np.zeros((0,), dtype=np.float32)

    def summary(d):
        if len(d) == 0:
            return {"n": 0}
        if not np.isfinite(d).all():
            return {"n": int(len(d))}
        return {
            "n": int(len(d)),
            "media_px": float(d.mean()),
            "mediana_px": float(np.median(d)),
            "p95_px": float(np.percentile(d, 95)),
            "max_px": float(d.max()),
            "media_mm": float(d.mean() * MM_PER_PX),
        }

    return {
        "polilinha_para_tinta": summary(d_fwd),
        "tinta_para_polilinha": summary(d_bwd),
    }


def write_svg(polylines, out_path, epsilon, thinning, primitives=None):
    """Escreve o SVG com viewBox em mm do papel A4 (210x297)."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- Draw vectorize: epsilon={epsilon} px, thinning={thinning}, "
        f"escala x={MM_PER_PX_X:.5f} mm/px, y={MM_PER_PX_Y:.5f} mm/px -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{PAPER_W_MM}mm" height="{PAPER_H_MM}mm" '
        f'viewBox="0 0 {PAPER_W_MM} {PAPER_H_MM}">',
    ]
    if primitives is not None:
        for prim in primitives:
            kind = prim["type"]
            # stroke_w_mm medido na vetorização (meta de classify_primitive);
            # sem ele, as larguras fixas de sempre.
            w = prim.get("stroke_w_mm")
            if kind == "point":
                lines.append(
                    f'<circle cx="{prim["x_mm"]:.2f}" cy="{prim["y_mm"]:.2f}" '
                    f'r="{max((w or 0.60) / 2, 0.10):.2f}" fill="#000"/>'
                )
            elif kind == "line":
                lines.append(
                    f'<line x1="{prim["x1_mm"]:.2f}" y1="{prim["y1_mm"]:.2f}" '
                    f'x2="{prim["x2_mm"]:.2f}" y2="{prim["y2_mm"]:.2f}" '
                    f'stroke="#000" stroke-width="{w or 0.35:.2f}" '
                    f'stroke-linecap="round"/>'
                )
            elif kind == "circle":
                lines.append(
                    f'<circle cx="{prim["cx_mm"]:.2f}" cy="{prim["cy_mm"]:.2f}" '
                    f'r="{prim["r_mm"]:.2f}" fill="none" stroke="#000" '
                    f'stroke-width="{w or 0.35:.2f}"/>'
                )
            elif kind == "rectangle":
                coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in prim["corners_mm"])
                lines.append(
                    f'<polygon data-primitive="rectangle" points="{coords}" '
                    f'fill="none" stroke="#000" stroke-width="{w or 0.35:.2f}"/>'
                )
            else:
                coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in prim["points_mm"])
                tag = "polygon" if kind == "polygon" else "polyline"
                lines.append(
                    f'<{tag} points="{coords}" fill="none" stroke="#000" '
                    f'stroke-width="{w or 0.30:.2f}"/>'
                )
    else:
        for line in polylines:
            pts = line["points"]
            w_px = line.get("width_px")
            w_mm = None if w_px is None else float(w_px) * MM_PER_PX
            if len(pts) == 1:
                cx, cy = pts[0][0] * MM_PER_PX_X, pts[0][1] * MM_PER_PX_Y
                lines.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" '
                    f'r="{max((w_mm or 0.60) / 2, 0.10):.2f}" fill="#000"/>'
                )
                continue
            coords = " ".join(
                f"{x * MM_PER_PX_X:.2f},{y * MM_PER_PX_Y:.2f}" for x, y in pts
            )
            tag = "polygon" if line["closed"] else "polyline"
            lines.append(
                f'<{tag} points="{coords}" fill="none" stroke="#000" '
                f'stroke-width="{w_mm or 0.30:.2f}"/>'
            )
    lines.append("</svg>")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe(ink, polylines, stats, primitives=None):
    """Resumo textual de uma vetorização."""
    n_points = sum(1 for line in polylines if len(line["points"]) == 1)
    n_closed = sum(1 for line in polylines if line["closed"])
    out = {
        "pixels_tinta": int(ink.sum()),
        "polilinhas": len(polylines),
        "polilinhas_fechadas": n_closed,
        "polilinhas_de_um_ponto": n_points,
        "saltos_de_caneta": max(0, len(polylines) - 1),
        "distancias": stats,
    }
    if primitives is not None:
        counts = {}
        for p in primitives:
            counts[p["type"]] = counts.get(p["type"], 0) + 1
        out["primitivas_contagem"] = counts
    return out


def _selfcheck():
    """Valida tanto o grafo exato de 1 px quanto limpeza + primitivas em 4 px."""
    rect = [(250, 70), (430, 250), (250, 430), (70, 250)]
    img = np.full((PAPER_H_PX, PAPER_W_PX), 255, dtype=np.uint8)
    for a, b in zip(rect, rect[1:] + rect[:1], strict=False):
        cv2.line(img, a, b, 0, thickness=1)
    cv2.circle(img, (620, 250), 110, 0, thickness=1)
    cv2.line(img, (150, 700), (450, 1000), 0, thickness=1)
    cv2.line(img, (150, 1000), (450, 700), 0, thickness=1)
    ink = img < 128

    polylines = vectorize_mask(ink, epsilon=1.5)
    n_closed = sum(1 for line in polylines if line["closed"])
    n_open = len(polylines) - n_closed
    print(
        f"polilinhas (1 px bruto): {len(polylines)} (fechadas {n_closed}, "
        f"abertas {n_open}); esperado 6 (2 fechadas, 4 abertas)"
    )
    assert len(polylines) == 6, f"contagem inesperada: {len(polylines)}"
    assert n_closed == 2, f"fechadas inesperadas: {n_closed}"

    def seg_dist(p, a, b):
        a, b = np.array(a, float), np.array(b, float)
        pnt = np.array(p, float)
        ab = b - a
        t = np.clip(np.dot(pnt - a, ab) / np.dot(ab, ab), 0, 1)
        return float(np.hypot(*(pnt - (a + t * ab))))

    def dist_to_shapes(p):
        d_rect = min(
            seg_dist(p, a, b) for a, b in zip(rect, rect[1:] + rect[:1], strict=False)
        )
        d_circ = abs(float(np.hypot(p[0] - 620, p[1] - 250)) - 110.0)
        d_x = min(
            seg_dist(p, (150, 700), (450, 1000)),
            seg_dist(p, (150, 1000), (450, 700)),
        )
        return min(d_rect, d_circ, d_x)

    samples = np.vstack(
        [dense_samples(line["points"], line["closed"]) for line in polylines]
    )
    dists = np.array([dist_to_shapes(p) for p in samples])
    print(
        f"erro contra a forma geradora: n={len(dists)} amostras, "
        f"média {dists.mean():.3f} px ({dists.mean() * MM_PER_PX:.3f} mm), "
        f"p95 {np.percentile(dists, 95):.3f} px, "
        f"máx {dists.max():.3f} px"
    )
    assert dists.mean() <= 1.5, "erro médio acima do esperado"

    # Bancada 2: traço real de 4 px + poeira de 1-2 px + higiene + Camada 2:
    img_thick = np.full((PAPER_H_PX, PAPER_W_PX), 255, dtype=np.uint8)
    for a, b in zip(rect, rect[1:] + rect[:1], strict=False):
        cv2.line(img_thick, a, b, 0, thickness=4)
    cv2.circle(img_thick, (620, 250), 110, 0, thickness=4)
    cv2.line(img_thick, (150, 700), (450, 1000), 0, thickness=4)
    cv2.line(img_thick, (150, 1000), (450, 700), 0, thickness=4)
    img_thick[40, 40] = 0
    img_thick[41, 40] = 0
    ink_thick = img_thick < 128

    clean_polys = vectorize_clean(ink_thick, epsilon=1.5)
    prims = classify_primitives(clean_polys)
    kinds = sorted(p["type"] for p in prims)
    print(f"primitivas (4 px limpo): {kinds}")
    assert kinds == ["circle", "line", "line", "rectangle"], (
        f"primitivas inesperadas em traco 4 px: {kinds}"
    )

    stats = distance_stats(ink, polylines)
    print("distâncias raster:", json.dumps(stats, indent=1))
    print("SELFCHECK OK")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Vetoriza um canvas de tinta (PNG) para polilinhas/primitivas/SVG."
    )
    parser.add_argument("input", nargs="?", help="PNG do canvas de tinta")
    parser.add_argument("--out", default=None, help="SVG de saída")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.5,
        help="epsilon do Douglas-Peucker em px (padrao 1,5)",
    )
    parser.add_argument(
        "--thinning",
        choices=THINNING_MODES,
        default="zhangsuen",
        help="algoritmo de afinamento (padrao zhangsuen)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="ativa limpeza completa: remove microfragmentos, poda esporões, "
        "contrai junções vizinhas, pareia ramos e ordena os traços",
    )
    parser.add_argument(
        "--stamps",
        default=None,
        metavar="ARQUIVO.npz",
        help="carimbos temporais (`_stamps.npz`) para orientar e ordenar os "
        "traços na ordem real em que foram desenhados",
    )
    parser.add_argument(
        "--primitives",
        action="store_true",
        help="classifica primitivas geométricas da Camada 2 (line, circle, "
        "rectangle) e emite tags SVG semânticas",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="SAIDA.json",
        help="exporta primitivas/polilinhas estruturadas em JSON",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="imprime as distâncias nos dois sentidos",
    )
    parser.add_argument(
        "--paper-w-mm",
        type=float,
        default=None,
        help="largura da folha em mm (default A4 retrato: 210); os "
        "denominadores em px saem da imagem carregada, então serve "
        "também para paisagem e outras grades",
    )
    parser.add_argument(
        "--paper-h-mm",
        type=float,
        default=None,
        help="altura da folha em mm (default A4 retrato: 297); exige --paper-w-mm",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="roda a bancada sintética e sai",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return 0
    if not args.input:
        parser.error("informe o PNG de entrada ou --selfcheck")
    if (args.paper_w_mm is None) != (args.paper_h_mm is None):
        parser.error("--paper-w-mm e --paper-h-mm devem ser passados juntos")
    if args.paper_w_mm is not None and (
        not math.isfinite(args.paper_w_mm)
        or not math.isfinite(args.paper_h_mm)
        or args.paper_w_mm <= 0
        or args.paper_h_mm <= 0
    ):
        parser.error("--paper-w-mm/--paper-h-mm devem ser positivos finitos")

    ink, polarity = load_ink(args.input)
    if args.paper_w_mm is None:
        return _run_pipeline(args, ink, polarity)
    # Folha não-A4: reaponta a escala a partir da imagem real e restaura
    # o default no fim — main() nunca deixa o módulo mutado.
    img_h_px, img_w_px = ink.shape[:2]
    _set_paper_mm(args.paper_w_mm, args.paper_h_mm, img_w_px, img_h_px)
    try:
        return _run_pipeline(args, ink, polarity)
    finally:
        _reset_paper_mm()


def _run_pipeline(args, ink, polarity):
    stamps = load_stamps(args.stamps) if args.stamps else None
    if args.clean:
        polylines = vectorize_clean(
            ink, epsilon=args.epsilon, thinning=args.thinning, stamps=stamps
        )
    else:
        polylines = vectorize_mask(
            ink,
            epsilon=args.epsilon,
            thinning=args.thinning,
            stamps=stamps,
            order_strokes=stamps is not None,
        )

    prims = classify_primitives(polylines) if (args.primitives or args.json) else None
    out = args.out or str(Path(args.input).with_suffix(".svg"))
    write_svg(
        polylines,
        out,
        args.epsilon,
        args.thinning,
        primitives=prims if args.primitives else None,
    )
    stats = distance_stats(ink, polylines)
    summary = describe(ink, polylines, stats, primitives=prims)
    summary["polaridade_tinta"] = polarity
    summary["epsilon_px"] = args.epsilon
    summary["thinning"] = args.thinning
    if args.paper_w_mm is not None:
        summary["paper_mm"] = [PAPER_W_MM, PAPER_H_MM]
    summary["svg"] = out
    if args.json:
        payload = {"summary": summary, "primitives": prims}
        Path(args.json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary["json"] = args.json

    if args.stats:
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    else:
        print(
            f"{args.input}: {summary['polilinhas']} polilinhas "
            f"({summary['polilinhas_fechadas']} fechadas), "
            f"{summary['saltos_de_caneta']} saltos de caneta, "
            f"dist. média polilinha->tinta "
            f"{stats['polilinha_para_tinta'].get('media_px', float('nan')):.3f} px, "
            f"tinta->polilinha "
            f"{stats['tinta_para_polilinha'].get('media_px', float('nan')):.3f} px "
            f"-> {out}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
