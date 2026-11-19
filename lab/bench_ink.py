"""Régua do motor de tinta: sintético com verdade absoluta + clipes reais.

Mede, antes e depois de cada mudança no pipeline de tinta:

  Sintético (gabarito exato por pixel, página pautada com pauta curva):
    (a) ruling_removed     fração da pauta ISOLADA (fora do traço) removida;
    (b) crossing_preserved fração do traço a <= `band` px de uma pauta que
                           segue na saída (cruzamentos não podem picotar);
        ticks_residual     pixels de pauta colados ao traço (os "tracinhos");
    (c) short_stroke       recall dos traços horizontais curtos de E T S H;
        on_line_recall     recall do traço deitado EXATAMENTE sobre a pauta
                           (o risco conhecido de qualquer filtro de pauta);
    e o custo: ms adicionados por passada (p50/p90) de um filtro sobre o
    mapa de probabilidade.

  Clipes reais (câmera, mão, sombra): o laço completo do app via
  lab.replay_lab.run_replay, com CPU-segundos por segundo de cena, tinta
  provisória exibida × retratada e, se o clipe tiver `regions.json`, pixels
  de escrita × pauta pura do canvas final.

Exemplos:
  python lab/bench_ink.py --synthetic
  python lab/bench_ink.py --clips ~/draw-data/bench-clips \\
      --presets legado,ganho,ganho+voto0.85 --json /tmp/bench.json

O `regions.json` de um clipe é {"writing": [[x0,y0,x1,y1], ...],
"ruling_only": [[x0,y0,x1,y1], ...]} em coordenadas do canvas final.
"""

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
# `python lab/bench_ink.py` põe só lab/ no path: run_clip importa lab.replay_lab.
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Glifos do requisito (c): traços horizontais curtos de E, T, S e H.
SHORT_STROKE_GLYPHS = ("E", "T", "T2", "S", "H")

# Presets de configuração comparáveis nos clipes reais (flags de run_replay).
PRESETS = {
    "legado": [],
    "ganho": ["ink_auto_gain=1"],
    "ganho+voto0.85": ["ink_auto_gain=1", "@strict=0.85"],
    "ganho+voto0.92": ["ink_auto_gain=1", "@strict=0.92"],
}

# ---------------------------------------------------------------- sintético


@dataclass(frozen=True)
class SyntheticPage:
    """Página pautada sintética com todo o gabarito.

    prob é o que o modelo de tinta devolveria: pauta e traço altos (o
    page_clean v2 vê a pauta como tinta, medido ao vivo) com bordas suaves
    e ruído de fundo.
    """

    prob: np.ndarray
    strokes: np.ndarray
    ruling: np.ndarray
    ruling_only: np.ndarray
    crossing_strokes: np.ndarray
    ticks_zone: np.ndarray
    glyphs: dict = field(default_factory=dict)
    # Só os braços horizontais (sem as hastes): o que as métricas (c) e
    # "sobre a linha" medem, para a haste intacta não inflar o recall.
    arms: dict = field(default_factory=dict)
    period: float = 0.0
    h_pauta: int = 0
    pen: int = 0


def ruling_y(k, x, *, y0, period, slope, bend_amp, bend_x, bend_w, tilt=0.0):
    """Ordenada da linha k da pauta em x: inclinada e curva perto da lombada.

    `tilt` abre as duas páginas em "V" (inclinações opostas de cada lado da
    lombada), como num caderno aberto torto: slope - tilt à esquerda e
    slope + tilt à direita.
    """
    x = np.asarray(x, np.float64)
    bend = bend_amp * np.exp(-(((x - bend_x) / bend_w) ** 2))
    side = np.where(x < bend_x, slope - tilt, slope + tilt)
    return y0 + k * period + side * (x - bend_x) + bend


def _polyline(mask, points, thickness):
    pts = np.round(np.asarray(points, np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], False, 255, thickness=thickness, lineType=cv2.LINE_8)


def _glyph_masks(shape, specs, pen):
    """Máscaras por glifo a partir de especificações (nome, polilinhas)."""
    out = {}
    for name, polylines in specs:
        mask = np.zeros(shape, np.uint8)
        for points in polylines:
            _polyline(mask, points, pen)
        out[name] = mask > 0
    return out


def make_synthetic_page(
    shape=(850, 1200),
    *,
    period=34,
    h_pauta=2,
    pen=5,
    slope=0.03,
    bend_amp=22.0,
    tilt=0.0,
    seed=0,
    noise=0.02,
):
    """Gera página pautada com glifos E T S H e traços cruzando a pauta.

    Determinística por `seed`. Os glifos cobrem: hastes verticais que cruzam
    várias linhas, braços horizontais entre linhas, uma diagonal, um traço
    fino e um E com o braço de baixo EXATAMENTE sobre uma linha (caso
    difícil, reportado à parte).
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    geo = dict(
        y0=0.0,
        period=float(period),
        slope=slope,
        bend_amp=bend_amp,
        bend_x=w * 0.5,
        bend_w=w * 0.12,
        tilt=tilt,
    )
    ruling = np.zeros(shape, np.uint8)
    xs = np.arange(0, w, 4)
    k = -2 - int(abs(tilt) * w * 0.5 / period) - 2  # linhas que entram pela inclinação
    while True:
        ys = ruling_y(k, xs, **geo) + period * 0.7
        if ys.min() > h:
            break
        if ys.max() >= 0:
            _polyline(ruling, np.stack([xs, ys], 1), h_pauta)
        k += 1
    ruling = ruling > 0

    def line_y(kk, x):
        return float(ruling_y(kk, x, **geo)) + period * 0.7

    def on_line(kk, xa, xb):
        """Pontos sobre a polilinha EXATA da linha kk entre xa e xb."""
        grid = np.arange(0, w, 4)
        gy = ruling_y(kk, grid, **geo) + period * 0.7
        xsel = [xa, *[g for g in grid if xa < g < xb], xb]
        return [(float(x), float(np.interp(x, grid, gy))) for x in xsel]

    gh = int(period * 2.4)  # altura de glifo: a haste cruza ~2 linhas
    gw = int(gh * 0.62)
    specs = []
    # (xe, ye) = canto superior esquerdo de cada glifo
    for name, x0, y0 in (
        ("E", 120, 150),
        ("T", 300, 160),
        ("S", 470, 150),
        ("H", 650, 165),
        ("T2", 860, 150),
    ):
        specs.append((name, _glyph_polylines(name, x0, y0, gw, gh)))
    # Diagonal e traço fino (cruzam a pauta em ângulo)
    specs.append(("diag", [[(260, 540), (330, 670)], [(360, 520), (420, 670)]]))
    # E com o braço de baixo exatamente sobre uma linha
    ex, kk = 160, 14
    bottom = on_line(kk, ex, ex + gw)  # o braço de baixo: sobre a linha
    base = bottom[0][1]
    specs.append(
        (
            "E_on_line",
            [
                [(ex, base - gh), (ex, base)],
                [(ex, base - gh), (ex + gw, base - gh)],
                [(ex, base - gh // 2), (ex + int(gw * 0.8), base - gh // 2)],
                bottom,
            ],
        )
    )
    glyphs = _glyph_masks(shape, specs, pen)
    thin = _glyph_masks(shape, [("thin", [[(520, 470), (540, 640)]])], max(2, pen - 2))
    glyphs.update(thin)
    # Lápis fino deitado SOBRE a linha, com a espessura da própria pauta: o
    # caso ambíguo de qualquer filtro de pauta (só a probabilidade ou a
    # continuidade com a haste o separam). Reportado à parte, sem meta fixa.
    px, pk = 700, 14
    pencil_arm = on_line(pk, px, px + gw)
    pbase = pencil_arm[0][1]
    pencil_pts = [[(px, pbase - gh), (px, pbase)], pencil_arm]
    glyphs.update(_glyph_masks(shape, [("pencil_on_line", pencil_pts)], h_pauta))
    # Braços: polilinhas horizontais (|dy| <= |dx|/2) de cada glifo; o S (uma
    # curva só) conta inteiro; nos "sobre a linha" é só o braço que está
    # sobre a pauta.
    arms = {}
    for name, polylines in specs:
        if name in ("E", "T", "T2", "S", "H"):
            keep = (
                polylines
                if len(polylines) == 1
                else [pl for pl in polylines if _is_horizontal(pl)]
            )
            arms[name] = _glyph_masks(shape, [(name, keep)], pen)[name]
    arms["E_on_line"] = _glyph_masks(shape, [("a", [bottom])], pen)["a"]
    arms["pencil_on_line"] = _glyph_masks(shape, [("a", [pencil_arm])], h_pauta)["a"]
    strokes = np.zeros(shape, bool)
    for mask in glyphs.values():
        strokes |= mask

    band = 2
    dil = cv2.dilate(strokes.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ruling_only = ruling & ~dil
    near_ruling = (
        cv2.dilate(ruling.astype(np.uint8), np.ones((2 * band + 1,) * 2, np.uint8)) > 0
    )
    # Os glifos sobre a linha têm métrica própria (E_on_line, pencil_on_line):
    # contá-los aqui misturaria o caso ambíguo com o cruzamento de verdade.
    off_line = np.zeros(shape, bool)
    for name, mask in glyphs.items():
        if name not in ("E_on_line", "pencil_on_line"):
            off_line |= mask
    crossing_strokes = off_line & near_ruling
    ticks_zone = ruling_only & (
        cv2.dilate(strokes.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    )

    prob = np.zeros(shape, np.float32)
    prob[ruling] = rng.uniform(0.80, 0.98, int(ruling.sum()))
    prob[strokes] = rng.uniform(0.88, 0.99, int(strokes.sum()))
    # sigma 0,5: uma pauta de 2 px mantém ~0,82 de prob (o v2 real dá 0,8-0,98)
    prob = cv2.GaussianBlur(prob, (0, 0), 0.5)
    prob = np.clip(prob + noise * np.abs(rng.normal(size=shape)), 0, 1)
    return SyntheticPage(
        prob=prob.astype(np.float32),
        strokes=strokes,
        ruling=ruling,
        ruling_only=ruling_only,
        crossing_strokes=crossing_strokes,
        ticks_zone=ticks_zone,
        glyphs=glyphs,
        arms=arms,
        period=float(period),
        h_pauta=h_pauta,
        pen=pen,
    )


def _is_horizontal(points):
    pts = np.asarray(points, np.float64)
    dx, dy = np.abs(np.diff(pts[:, 0])), np.abs(np.diff(pts[:, 1]))
    return bool((dy <= 0.5 * dx).all())


def _glyph_polylines(name, x0, y0, w, h):
    """Polilinhas (em px) das letras E, T, S e H, com canto superior esquerdo."""
    x1, y1, ym = x0 + w, y0 + h, y0 + h // 2
    name = name.rstrip("2")  # T2 = segundo T (nome único no gabarito)
    if name == "E":
        return [
            [(x0, y0), (x0, y1)],
            [(x0, y0), (x1, y0)],
            [(x0, ym), (x1 - w // 5, ym)],
            [(x0, y1), (x1, y1)],
        ]
    if name == "T":
        return [[(x0, y0), (x1, y0)], [((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1)]]
    if name == "H":
        return [[(x0, y0), (x0, y1)], [(x1, y0), (x1, y1)], [(x0, ym), (x1, ym)]]
    if name == "S":
        pts = [
            (1.0, 0.1),
            (0.35, 0.0),
            (0.05, 0.22),
            (0.3, 0.46),
            (0.8, 0.62),
            (1.0, 0.82),
            (0.7, 1.0),
            (0.0, 0.92),
        ]
        return [[(x0 + px * w, y0 + py * h) for px, py in pts]]
    raise ValueError(name)


# ------------------------------------------------------------------ métricas


def _frac(num, den):
    return float(num) / float(den) if den else float("nan")


def ruling_removed(pred, page):
    """(a) Fração da pauta isolada (fora do traço) que NÃO está na saída."""
    return 1.0 - _frac((pred & page.ruling_only).sum(), page.ruling_only.sum())


def crossing_preserved(pred, page):
    """(b) Fração do traço a <= band px da pauta que segue na saída."""
    return _frac((pred & page.crossing_strokes).sum(), page.crossing_strokes.sum())


def ticks_residual(pred, page):
    """(b) Pixels de pauta colados ao traço que sobraram (os tracinhos)."""
    return int((pred & page.ticks_zone).sum())


def glyph_recall(pred, page):
    """Recall por glifo (E, T, S, H, diag, thin, E_on_line, pencil_on_line)."""
    return {
        name: _frac((pred & mask).sum(), mask.sum())
        for name, mask in page.glyphs.items()
    }


def arm_recall(pred, page):
    """Recall por braço horizontal (sem as hastes, que um filtro de pauta
    preserva de qualquer jeito): E T T2 H S, E_on_line e pencil_on_line."""
    return {
        name: _frac((pred & mask).sum(), mask.sum()) for name, mask in page.arms.items()
    }


def short_stroke_recall(pred, page):
    """(c) Recall agregado dos braços horizontais curtos de E T S H."""
    keys = SHORT_STROKE_GLYPHS
    num = sum((pred & page.arms[k]).sum() for k in keys)
    den = sum(page.arms[k].sum() for k in keys)
    return _frac(num, den)


def score_filter(page, fn=None, threshold=0.5):
    """Aplica `fn(prob) -> prob` (None = identidade) e mede (a), (b), (c)."""
    prob = page.prob if fn is None else fn(page.prob.copy())
    if np.shape(prob) != page.prob.shape:
        # Mapas empilhados repetiriam a máscara 2D no eixo extra e dariam
        # recall > 100% (review do PR #54).
        raise ValueError(
            f"o filtro devolveu forma {np.shape(prob)}; esperado {page.prob.shape}"
        )
    pred = prob > threshold
    recalls = glyph_recall(pred, page)
    arms = arm_recall(pred, page)
    return {
        "ruling_removed": ruling_removed(pred, page),
        "crossing_preserved": crossing_preserved(pred, page),
        "ticks_residual": ticks_residual(pred, page),
        "short_stroke_recall": short_stroke_recall(pred, page),
        "on_line_recall": arms["E_on_line"],
        "pencil_on_line_recall": arms["pencil_on_line"],
        "recalls": recalls,
        "arm_recalls": arms,
    }


def time_ms(fn, arg, *, runs=30, warmup=3):
    """Tempo (ms) de fn(arg): p50 e p90. Cada medida recebe uma cópia nova."""
    for _ in range(warmup):
        fn(arg.copy())
    samples = []
    for _ in range(runs):
        data = arg.copy()
        t0 = time.perf_counter()
        fn(data)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p90_ms": float(np.percentile(samples, 90)),
    }


# ------------------------------------------------------------- clipes reais


def preset_flags(name):
    """Preset -> flags de run_replay (resolve @strict=NIVEL em campos vote_*)."""
    from draw.ink.config import InkConfig

    out = []
    for flag in PRESETS[name]:
        if flag.startswith("@strict="):
            cfg = InkConfig.strict_vote(float(flag.split("=", 1)[1]))
            out += [
                f"ink.{f}={getattr(cfg, f)}"
                for f in ("vote_hit", "vote_score", "vote_score_strong")
            ]
        else:
            out.append(flag)
    return out


def _region_px(mask, boxes):
    return int(sum(np.count_nonzero(mask[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes))


def run_clip(clip_dir, flags, *, detector=None, page_clean=None, verify=None):
    """Roda um clipe pelo laço real do app e devolve as medidas.

    Sem cache de inferência: o custo de CPU medido é o real. Devolve:
    canvas_px, writing_px/ruling_px (se houver regions.json), provisório
    (mediana e último) × retratado, e CPU-segundos por segundo de cena.
    """
    import lab.replay_lab as rl
    from draw import config
    from draw import pipeline as pipeline_module

    clip_dir = Path(clip_dir)
    meta = json.loads((clip_dir / "clip.json").read_text())
    config.apply_format("a4", meta.get("orientation", "retrato"))
    series = []
    retracted = {"carry": 0, "last": 0}

    def _track_retractions(stats):
        # O acumulador zera quando a folha muda: o que já foi retratado antes
        # do zero entra no total (review do PR #54).
        now = int(stats.get("retracted_px_total") or 0)
        if now < retracted["last"]:
            retracted["carry"] += retracted["last"]
        retracted["last"] = now

    real = pipeline_module.Pipeline
    holder = {}

    class Spy(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["loop"] = self

        def step(self, packet):
            result = super().step(packet)
            stats = self.ink_worker.stats()
            series.append(stats.get("provisional_px") or 0)
            _track_retractions(stats)
            return result

    pipeline_module.Pipeline = Spy
    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        with tempfile.TemporaryDirectory(prefix="bench_ink_") as scratch:
            records = rl.run_replay(
                clip_dir,
                scratch,
                flags=flags,
                detector=detector,
                page_clean=page_clean,
                scene_fps=meta["scene_fps"],
                verify=verify,
            )
    finally:
        pipeline_module.Pipeline = real
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    loop = holder["loop"]
    mask = (
        np.zeros((config.PAPER_H, config.PAPER_W), bool)
        if loop.clean_mask is None
        else np.asarray(loop.clean_mask) > 0
    )
    # O replay publica a tinta do último quadro DEPOIS de Pipeline.step: a
    # amostra final vem do estado final, não do último passo.
    stats = loop.ink_worker.stats()
    _track_retractions(stats)
    final_provisional = int(stats.get("provisional_px") or 0)
    scene_s = len(records) / meta["scene_fps"]
    out = {
        "clip": clip_dir.name,
        "frames": len(records),
        "canvas_px": int(mask.sum()),
        "provisional_median_px": float(np.median(series)) if series else 0.0,
        "provisional_last_px": final_provisional,
        "retracted_px_total": retracted["carry"] + retracted["last"],
        "passes": int(stats.get("passes_total") or 0),
        "cpu_s_per_scene_s": cpu / scene_s if scene_s else float("nan"),
        "wall_s": wall,
    }
    out["retracted_over_canvas"] = _frac(out["retracted_px_total"], out["canvas_px"])
    regions = clip_dir / "regions.json"
    if regions.exists():
        spec = json.loads(regions.read_text())
        out["writing_px"] = _region_px(mask, spec.get("writing", []))
        out["ruling_px"] = _region_px(mask, spec.get("ruling_only", []))
        out["ruling_over_writing"] = _frac(out["ruling_px"], out["writing_px"])
    out["_canvas"] = mask
    return out


# ------------------------------------------------------------------------ CLI


def _json_safe(value):
    """NaN/inf (razão sem denominador) viram null: JSON estrito não os aceita."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def identity_filter():
    """Fábrica do filtro identidade (exemplo e teste de --filter)."""
    return lambda prob: prob


def _load_filter(spec):
    """`pacote.modulo:fabrica` -> fn(prob) -> prob (a fábrica é chamada sem args)."""
    import importlib

    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"--filter espera modulo:fabrica, recebeu {spec!r}")
    return getattr(importlib.import_module(module), attr)()


def _ms(value):
    return "-" if value is None else f"{value:.2f}"


def _print_synthetic(results):
    print("== sintético (gabarito exato) ==")
    print(
        f"{'filtro':22s} {'(a) pauta':>10s} {'(b) cruz.':>10s} {'ticks':>7s} "
        f"{'(c) E T S H':>12s} {'sobre linha':>12s} {'lapis s/ linha':>15s} "
        f"{'p50 ms':>8s} {'p90 ms':>8s}"
    )
    for name, r in results.items():
        t = r.get("timing", {})
        print(
            f"{name:22s} {r['ruling_removed']:10.1%} {r['crossing_preserved']:10.1%} "
            f"{r['ticks_residual']:7d} {r['short_stroke_recall']:12.1%} "
            f"{r['on_line_recall']:12.1%} {r['pencil_on_line_recall']:15.1%} "
            f"{_ms(t.get('p50_ms')):>8s} {_ms(t.get('p90_ms')):>8s}"
        )


def _print_clips(rows):
    print("== clipes reais ==")
    print(
        f"{'clipe':24s} {'preset':16s} {'canvas':>8s} {'escrita':>8s} {'pauta':>7s} "
        f"{'prov.med':>9s} {'retr/canvas':>11s} {'CPU s/s':>8s}"
    )
    for r in rows:
        print(
            f"{r['clip']:24s} {r['preset']:16s} {r['canvas_px']:8d} "
            f"{r.get('writing_px', float('nan')):8} {r.get('ruling_px', float('nan')):7} "
            f"{r['provisional_median_px']:9.0f} {r['retracted_over_canvas']:11.1f} "
            f"{r['cpu_s_per_scene_s']:8.2f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true", help="roda o sintético")
    parser.add_argument("--clips", help="pasta com subpastas de clipes (clip.json)")
    parser.add_argument("--only", help="só o clipe com este nome")
    parser.add_argument(
        "--presets",
        default="legado,ganho",
        help="presets separados por vírgula: " + ", ".join(PRESETS),
    )
    parser.add_argument(
        "--filter",
        help="modulo:fabrica de um filtro prob->prob a medir no sintético "
        "(custo p50/p90 incluso); sem ele, só a linha de base sem filtro",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", default=str(_REPO_ROOT / "models"))
    parser.add_argument("--json", help="grava o resultado completo neste arquivo")
    args = parser.parse_args(argv)
    if not args.synthetic and not args.clips:
        parser.error("informe --synthetic e/ou --clips")
    unknown = [p for p in args.presets.split(",") if p not in PRESETS]
    if unknown:
        parser.error(f"preset desconhecido: {unknown}; válidos: {list(PRESETS)}")

    report = {}
    if args.synthetic:
        pages = {
            "": make_synthetic_page(seed=args.seed),
            ", pagina torta (V 0,35)": make_synthetic_page(seed=args.seed, tilt=0.35),
        }
        results = {
            f"sem filtro (identidade){suffix}": score_filter(pg)
            for suffix, pg in pages.items()
        }
        if args.filter:
            try:
                for suffix, pg in pages.items():
                    fn = _load_filter(args.filter)
                    row = score_filter(pg, fn)
                    row["timing"] = time_ms(fn, pg.prob)
                    results[f"{args.filter}{suffix}"] = row
            except (ValueError, ImportError, AttributeError) as error:
                parser.error(str(error))
        report["synthetic"] = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("recalls", "arm_recalls")}
            for k, v in results.items()
        }
        _print_synthetic(results)
    if args.clips:
        from draw.infra.manifest import verify_model
        from draw.page_clean import PageCleanInk
        from draw.perception import NeuralSheetDetector

        models = Path(args.models)

        def fresh_detector():
            # O detector guarda bordas e máscara entre detecções: reusá-lo
            # entre clipes/presets faria o resultado depender da ordem de
            # execução (review do PR #54).
            detector = NeuralSheetDetector(
                str(models / "draw_seg_v3_dyn.onnx"), verify=verify_model
            )
            detector.load()
            return detector

        page_clean = PageCleanInk(
            str(models / "page_clean_unet.onnx"), verify=verify_model
        )
        page_clean.load()
        rows = []
        for clip in sorted(Path(args.clips).iterdir()):
            if not (clip / "clip.json").exists() or (
                args.only and clip.name != args.only
            ):
                continue
            for preset in args.presets.split(","):
                row = run_clip(
                    clip,
                    preset_flags(preset),
                    detector=fresh_detector(),
                    page_clean=page_clean,
                    verify=None,
                )
                row["preset"] = preset
                row.pop("_canvas")
                rows.append(row)
        report["clips"] = rows
        _print_clips(rows)
    if args.json:
        Path(args.json).write_text(
            json.dumps(_json_safe(report), indent=1, allow_nan=False)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
