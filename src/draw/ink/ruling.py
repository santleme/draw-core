"""Remoção de pauta do mapa de probabilidade da tinta (runtime, sem treino).

O page_clean v2 enxerga a pauta do caderno como tinta (medido ao vivo): ela
entra no canvas, e onde cruza um traço deixa "tracinhos" além da largura da
caneta. Este módulo a tira do mapa de probabilidade ANTES do registro e do
voto, em três passos:

1. Periodicidade e sementes: por faixas verticais, a projeção 1D de um mapa
   de "linearidade" (alto-passa vertical, que ignora hastes) dá, por
   autocorrelação, o período; os picos da projeção dão a posição de cada
   linha. Sem pico claro a página não tem pauta e o módulo não faz nada.
2. Caminhos estáveis por programação dinâmica: o pente de linhas é rastreado
   uma vez, com um único caminho de deslocamento `d(x)` (|dy/dx| <= 1) que
   acompanha inclinação e curvatura, e depois cada linha é refinada numa
   banda estreita. As DPs são vetorizadas sobre todos os `y` (e, no
   refinamento, sobre todas as linhas) por coluna; o único laço Python é
   sobre colunas, em resolução reduzida. A metade esquerda e a direita do
   caderno aberto são rastreadas de forma independente, então um degrau
   entre as duas páginas não quebra o caminho.
3. Portão de run-length vertical: ao longo de cada caminho mede-se a
   espessura vertical contínua da tinta e apaga-se só o que é compatível com
   pauta isolada (`<= h_pauta + 2`). Cruzamentos verticais
   (`> h_pauta + 2` por 2+ colunas seguidas) são âncoras e ficam intactos, com folga de 2 colunas nas
   bordas; um traço de caneta deitado sobre a linha (o braço de um E) também
   é preservado quando a sua probabilidade média está acima da mediana
   daquela pauta, mesmo que a espessura seja parecida.

O custo está em duas parcelas: a estimativa (cara, ~poucos ms) roda a cada
`reestimate_every` passadas, e a cada passada só se reancora o pente
(deslocamento vertical) e se aplica o portão.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

# Resolução da estimativa: colunas x6, linhas x2 (a pauta de 2 px vira ~1).
_DS_X = 6
_DS_Y = 2
_MIN_PERIOD = 14.0  # px da grade da página
_MAX_PERIOD = 90.0
_STRIP_W = 96  # largura das faixas de autocorrelação, px
_MIN_PEAK = 0.25  # força mínima do pico de autocorrelação
_MIN_LINES = 4
_REFINE_BAND = 8  # linhas reduzidas de cada lado, no refinamento
_SLOPE_ROWS = 3  # |dy/dx| <= 1: 6 px de coluna = até 3 linhas reduzidas
_JUMP_PENALTY = 0.02
# Portão
_T_RUN = 0.5  # probabilidade que conta como "tinta" para o run-length
_R = 6  # janela vertical de medida, px de cada lado
_OCCLUDED_BELOW = 0.85  # fração visível abaixo da qual um negativo é suspeito
_HAND_LEFT = 0.1  # aumento da fração visível que conta como 'a mão saiu'
_RETRY_PASSES = 3  # passadas ate reestimar um negativo suspeito
_SUPPORT_W = 48  # px de cada janela do teste de suporte
_MIN_SPAN = 0.6  # extensao minima (fração da metade) do suporte
_MIN_SUPPORT = 0.35  # fração minima de janelas com tinta sobre a linha
_ANCHOR_PAD = 2  # colunas protegidas em volta de um cruzamento
# Espessura apagável: <= h_pauta + _THIN_MARGIN (a pauta real oscila 3-6 px, medido no e5).
_THIN_MARGIN = 2
_PROB_MARGIN = 0.045  # margem sobre a mediana da pauta para proteger um braço
_PROB_SMOOTH = 9  # colunas da média móvel da probabilidade ao longo da linha
_PROB_SIGMAS = 4.0  # margem em desvios robustos do ruído da própria pauta
_PROB_MIN_RUN = 12  # colunas mínimas contínuas para proteger um braço
# Vinco sombreado: a sombra engrossa TODAS as linhas numa faixa larga de colunas.
# Um cruzamento de caneta também engrossa todas, mas só por poucas colunas; por
# isso a faixa exige >= _GUTTER_MIN_W colunas seguidas com a maioria das linhas
# grossas, e nela a espessura apagável sobe de _GUTTER_EXTRA px.
_GUTTER_FRAC = 0.6  # fração das linhas grossas na coluna
_GUTTER_MIN_W = 16  # colunas contínuas (mais largo que qualquer traço de caneta)
_GUTTER_JOIN_PAD = 3  # colunas olhadas em cada ponta da faixa
_GUTTER_EXTRA = 4  # px a mais de espessura apagável dentro da faixa


@dataclass(frozen=True)
class Segment:
    """Caminhos de uma metade do caderno: y(x) de cada linha em [x0, x1)."""

    x0: int
    x1: int
    paths: np.ndarray  # (K, x1 - x0) float32, px da grade da página
    period: float


@dataclass(frozen=True)
class RulingModel:
    segments: tuple
    h_pauta: float
    shape: tuple
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------- estimativa


def _line_score(prob, visible=None):
    """Mapa reduzido de 'linearidade': alto onde há tinta fina e horizontal."""
    src = prob if visible is None else prob * visible
    h, w = src.shape
    small = cv2.resize(
        src, (w // _DS_X, h // _DS_Y), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    # alto-passa vertical: hastes (colunas inteiras altas) somem, linhas ficam
    return np.maximum(small - cv2.blur(small, (1, 9)), 0.0)


def _autocorr_period(profile):
    """(lag em linhas reduzidas, força do pico) pela autocorrelação, ou None."""
    p = profile - profile.mean()
    n = p.size
    if n < 8 or float(np.abs(p).max()) < 1e-6:
        return None
    spec = np.fft.rfft(p, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    ac = ac / ac[0]
    lo = max(2, int(_MIN_PERIOD / _DS_Y))
    hi = min(n - 2, int(_MAX_PERIOD / _DS_Y))
    if hi <= lo:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    if not (ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]):
        return None
    if ac[lag] < _MIN_PEAK:
        return None
    # refino parabólico do pico
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
    return lag + float(np.clip(shift, -0.5, 0.5)), float(ac[lag])


def _peaks(profile, min_dist, min_height):
    """Picos do perfil, com supressão de não-máximos (altura desc.)."""
    mid = profile[1:-1]
    cand = (
        np.nonzero((mid > profile[:-2]) & (mid >= profile[2:]) & (mid > min_height))[0]
        + 1
    )
    order = cand[np.argsort(profile[cand])[::-1]]
    kept = []
    for idx in order:
        if all(abs(idx - j) >= min_dist for j in kept):
            kept.append(idx)
    return np.array(sorted(kept), np.float32)


# Inclinações candidatas das linhas (linhas reduzidas por coluna reduzida; a
# inclinação física é s * _DS_Y / _DS_X = s / 3, então |m| <= 0,6).
_SLOPES = np.arange(-1.8, 1.81, 0.2)
_SEED_COLS = 24  # largura da faixa de sementes, em colunas reduzidas


def _sheared_profile(score, a, b, slope):
    """Perfil vertical das colunas [a, b) com as linhas 'endireitadas' por `slope`.

    Com a inclinação certa as linhas se empilham e o perfil fica pontudo; com
    a errada elas se borram (uma faixa de 144 px de pauta a 25 graus desloca
    ~60 px, bem mais que o período).
    """
    cols = np.arange(a, b)
    shift = np.round(slope * (cols - (a + b) // 2)).astype(np.int64)
    rows = np.clip(
        np.arange(score.shape[0])[:, None] + shift[None, :], 0, score.shape[0] - 1
    )
    return score[rows, cols[None, :]].mean(axis=1)


def _seeds(score, c_lo, c_hi):
    """Melhor faixa da metade: (coluna-semente, posições das linhas, lag, inclinação)."""
    best = None
    for frac in (0.5, 0.3, 0.7, 0.15, 0.85):
        center = int(c_lo + frac * (c_hi - c_lo))
        a = max(c_lo, center - _SEED_COLS // 2)
        b = min(c_hi, center + _SEED_COLS // 2)
        if b - a < 8:
            continue
        slope, profile, sharp = 0.0, None, -1.0
        for cand in _SLOPES:
            prof = _sheared_profile(score, a, b, cand)
            var = float(prof.var())
            if var > sharp:
                slope, profile, sharp = float(cand), prof, var
        found = _autocorr_period(profile)
        if found is None:
            continue
        lag, strength = found
        if best is None or strength > best[0]:
            best = (strength, (a + b) // 2, profile, lag, slope)
        if strength >= 0.5:
            break
    if best is None:
        return None
    _strength, c_seed, profile, lag, slope = best
    smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
    peaks = _peaks(smooth, 0.6 * lag, float(smooth.mean() + 0.5 * smooth.std()))
    if peaks.size < _MIN_LINES:
        return None
    # com a folha inclinada, linhas fora da faixa-semente entram na vista pelo
    # lado: estende além das bordas pelo deslocamento máximo da inclinação
    reach = int(np.ceil(abs(slope) * (c_hi - c_lo))) + 1
    return c_seed, _extend_peaks(peaks, score.shape[0], reach), lag, slope


def _extend_peaks(peaks, h_s, reach=0):
    """Estende o pente, no passo mediano, até as bordas do mapa.

    Os picos da faixa-semente somem onde a pauta enfraquece (sombra, margens
    e, com a folha inclinada, as linhas que só entram na vista mais longe da
    semente): as linhas além do primeiro/último pico ficavam sem caminho e
    sobreviviam como riscos soltos. A pauta de verdade corre a folha toda; o
    portão de run-length segue decidindo, ponto a ponto, o que apagar.
    """
    step = float(np.median(np.diff(peaks)))
    if step < 2:
        return peaks
    lo = np.arange(peaks[0] - step, -1.0 - reach, -step)[::-1]
    hi = np.arange(peaks[-1] + step, h_s - 1.0 + reach, step)
    return np.concatenate([lo, peaks, hi]).astype(np.float32)


def _shifted_max(prev, slope, penalty):
    """max_s (prev[i + s] - penalty*|s|) e o argmax s, vetorizado em i."""
    d2 = prev.shape[-1]
    pad = np.full(prev.shape[:-1] + (slope,), -np.inf, prev.dtype)
    padded = np.concatenate([pad, prev, pad], axis=-1)
    cand = np.stack(
        [
            padded[..., slope + s : slope + s + d2] - penalty * abs(s)
            for s in range(-slope, slope + 1)
        ]
    )
    return cand.max(axis=0), cand.argmax(axis=0) - slope


def _comb_track(score, seeds, c_seed, c_lo, c_hi, dmax, slope=0.0):
    """Deslocamento d(c) do pente por DP, a partir da coluna-semente.

    Vetorizada sobre todos os deslocamentos: um laço de colunas, nenhum
    laço em y. Devolve (c_hi - c_lo,) em linhas reduzidas.
    """
    h_s = score.shape[0]
    d = np.arange(-dmax, dmax + 1)
    cols = np.arange(c_lo, c_hi)
    ref = np.round(seeds[None, :] + slope * (cols - c_seed)[:, None]).astype(np.int64)
    rows = np.clip(ref[:, :, None] + d[None, None, :], 0, h_s - 1)
    comb = score[rows, cols[:, None, None]].sum(axis=1) / len(seeds)  # (C, D2)
    n = cols.size
    i_seed = c_seed - c_lo
    out = np.zeros(n, np.int64)
    for step in (+1, -1):
        idxs = range(i_seed + step, n if step > 0 else -1, step)
        dp = np.full(d.size, -np.inf, np.float32)
        dp[dmax] = comb[i_seed, dmax]
        ptr = []
        for i in idxs:
            best, arg = _shifted_max(dp, _SLOPE_ROWS, _JUMP_PENALTY)
            dp = (comb[i] + best).astype(np.float32)
            ptr.append(arg)
        if not ptr:
            continue
        pos = int(np.argmax(dp))
        track = [pos]
        for arg in reversed(ptr):
            pos = int(np.clip(pos + arg[pos], 0, d.size - 1))
            track.append(pos)
        track.reverse()  # do mais perto da semente para fora
        # track[0] é a posição na semente; track[1:] casa com idxs
        for i, p in zip(idxs, track[1:], strict=False):
            out[i] = p
        out[i_seed] = dmax
    return (out - dmax).astype(np.float32)


def _refine(score, seeds, d_cols, c_lo, c_hi, c_seed=0, slope=0.0):
    """Refina cada linha numa banda estreita (DP vetorizada sobre as linhas)."""
    h_s = score.shape[0]
    band = np.arange(-_REFINE_BAND, _REFINE_BAND + 1)
    cols = np.arange(c_lo, c_hi)
    ref = seeds[None, :] + slope * (cols - c_seed)[:, None] + d_cols[:, None]
    base = np.round(ref).astype(np.int64)  # (C, K)
    rows = np.clip(base[:, :, None] + band[None, None, :], 0, h_s - 1)
    sc = score[rows, cols[:, None, None]]  # (C, K, B2)
    dp = sc[0].astype(np.float32)
    ptr = []
    for i in range(1, cols.size):
        best, arg = _shifted_max(dp, 2, _JUMP_PENALTY)
        dp = (sc[i] + best).astype(np.float32)
        ptr.append(arg)
    pos = np.argmax(dp, axis=-1)
    track = [pos]
    for arg in reversed(ptr):
        pos = np.clip(
            pos + np.take_along_axis(arg, pos[:, None], -1)[:, 0], 0, band.size - 1
        )
        track.append(pos)
    track.reverse()
    off = np.stack(track)  # (C, K)
    return (base + band[off]).astype(np.float32)  # (C, K) linhas reduzidas


def _segment(score, c_lo, c_hi, width, height):
    found = _seeds(score, c_lo, c_hi)
    if found is None:
        return None
    c_seed, seeds, lag, slope = found
    dmax = int(1.2 * lag) + 2
    d_cols = _comb_track(score, seeds, c_seed, c_lo, c_hi, dmax, slope)
    rows = _refine(score, seeds, d_cols, c_lo, c_hi, c_seed, slope)  # (C, K)
    # A grade reduzida vem de um resize: o passo real é width/w_s (nao _DS_X
    # quando a largura nao e multipla de 6: 850 -> 141 colunas deixava as
    # ultimas 4 sem caminho). A ultima metade vai ate a borda.
    w_s, h_s = score.shape[1], score.shape[0]
    rx, ry = width / w_s, height / h_s
    # centro do bloco reduzido em coordenadas de INDICE de pixel: -0,5
    xc = (np.arange(c_lo, c_hi) + 0.5) * rx - 0.5
    x0 = int(round(c_lo * rx))
    x1 = width if c_hi >= w_s else int(round(c_hi * rx))
    xs = np.arange(x0, x1, dtype=np.float32)
    full = (rows + 0.5) * ry - 0.5
    # linhas estendidas que nunca entram na imagem não têm o que apagar (e o
    # grampo da janela as jogaria para a borda)
    inside = ((full >= 0) & (full <= height - 1)).any(axis=0)
    paths = np.stack(
        [np.interp(xs, xc, full[:, k]) for k in np.flatnonzero(inside)]
    ).astype(np.float32)
    return Segment(x0=x0, x1=x1, paths=paths, period=float(lag * ry))


def estimate(prob, visible=None):
    """Estima os caminhos da pauta de uma página, ou None se não houver."""
    h, w = prob.shape
    score = _line_score(prob, visible)
    w_s = score.shape[1]
    split = w_s // 2
    segments = []
    for c_lo, c_hi in ((0, split), (split, w_s)):
        if c_hi - c_lo < 20:
            continue
        seg = _segment(score, c_lo, c_hi, w, h)
        if seg is not None:
            segments.append(seg)
    segments = [seg for seg in segments if _support(prob, seg) >= _MIN_SUPPORT]
    if not segments:
        return None
    h_pauta = _estimate_h(prob, segments)
    if h_pauta is None:
        return None
    return RulingModel(
        segments=tuple(segments),
        h_pauta=h_pauta,
        shape=(h, w),
        info={"lines": [s.paths.shape[0] for s in segments]},
    )


# --------------------------------------------------------------------- portão

_J = np.arange(-_R, _R + 1)
_POW = (1 << np.arange(2 * _R + 1)).astype(np.int16)  # bit i = linha j = i - _R


def _build_lut():
    """Para cada janela vertical de 2R+1 bits: (run, up, down, jstar).

    O centro é a linha ligada mais próxima de 0 dentro de |j| <= 2 (a
    estimativa do caminho erra por ~1-2 px); `run` é a sequência contínua de
    linhas ligadas que contém o centro, `up`/`down` os trechos acima/abaixo.
    """
    n = 1 << (2 * _R + 1)
    run = np.zeros(n, np.int8)
    up = np.zeros(n, np.int8)
    down = np.zeros(n, np.int8)
    jstar = np.zeros(n, np.int8)
    order = (0, -1, 1, -2, 2)
    for pat in range(n):
        bits = [(pat >> i) & 1 for i in range(2 * _R + 1)]
        c = next((j for j in order if bits[j + _R]), None)
        if c is None:
            continue
        u = 0
        while c - u - 1 >= -_R and bits[c - u - 1 + _R]:
            u += 1
        d = 0
        while c + d + 1 <= _R and bits[c + d + 1 + _R]:
            d += 1
        run[pat], up[pat], down[pat], jstar[pat] = u + d + 1, u, d, c
    return run, up, down, jstar


_LUT = None


def _lut():
    global _LUT
    if _LUT is None:
        _LUT = _build_lut()
    return _LUT


class _Track:
    """Caminhos de um segmento, com os mapas de amostragem em cache."""

    def __init__(self, seg, shape):
        self.x0, self.x1 = seg.x0, seg.x1
        self.paths = seg.paths.copy()
        self.shape = shape
        self._mapx = self._mapy = self._yi = None
        # nível típico e ruído da própria linha (por estimativa, não por passada)
        self.base_level = None
        self.base_sigma = None

    def _build(self):
        h, _ = self.shape
        k, x = self.paths.shape
        self._yi = np.clip(np.round(self.paths).astype(np.int32), _R, h - 1 - _R)
        rows = self._yi[:, None, :] + _J[None, :, None]  # (K, 2R+1, X)
        self._mapy = rows.reshape(k * (2 * _R + 1), x).astype(np.float32)
        cols = (np.arange(x) + self.x0).astype(np.float32)
        self._mapx = np.ascontiguousarray(
            np.broadcast_to(cols[None, :], self._mapy.shape)
        )

    def shift(self, delta):
        """Desloca o pente (px) e invalida o cache."""
        self.paths += delta
        self._mapx = self._mapy = self._yi = None

    def window(self, ink):
        """Padrão de bits da janela vertical ao longo de cada caminho: (K, X).

        `ink` é o mapa binário (H, W) uint8 0/1; o cv2.remap amostra a janela
        de 2R+1 linhas de todos os caminhos de uma vez.
        """
        if self._mapy is None:
            self._build()
        k, x = self.paths.shape
        bits = cv2.remap(ink, self._mapx, self._mapy, cv2.INTER_NEAREST)
        pattern = np.einsum(
            "i,kix->kx", _POW, bits.reshape(k, 2 * _R + 1, x), dtype=np.int16
        )
        return pattern, self._yi


def _along(a, k):
    """Média móvel ao longo do eixo x (último) por soma acumulada."""
    pad = k // 2
    padded = np.pad(a, ((0, 0), (pad + 1, pad)), mode="edge")
    c = np.cumsum(padded, axis=1, dtype=np.float64)
    return ((c[:, k:] - c[:, :-k]) / k).astype(np.float32)[:, : a.shape[1]]


def _dilate_cols(mask, pad):
    out = mask.copy()
    for s in range(1, pad + 1):
        out[:, s:] |= mask[:, :-s]
        out[:, :-s] |= mask[:, s:]
    return out


def _support(prob, seg):
    """Fração das janelas de _SUPPORT_W px, ao longo da pagina, em que a linha
    tem tinta fina sobre o caminho (mediana das linhas).

    Pauta de verdade corre a largura da folha; um tracejado de caneta, ou
    qualquer coisa periodica numa faixa so, nao: seu suporte e local.
    """
    ink = (prob > _T_RUN).view(np.uint8)
    pattern, _ = _Track(seg, prob.shape).window(ink)
    has = _lut()[0][pattern] >= 1  # (K, X)
    k, x = has.shape
    n = max(1, x // _SUPPORT_W)
    win = has[:, : n * _SUPPORT_W].reshape(k, n, _SUPPORT_W).any(axis=2)
    frac = float(np.median(win.mean(axis=1)))
    # Extensão: da primeira à última COLUNA com tinta, em fração da metade
    # (em pixels, não em janelas: o corte não depende do alinhamento). Um
    # tracejado local ocupa pouco da metade, mesmo cobrindo > 35% das janelas
    # de uma metade estreita.
    anyc = has.any(axis=1)
    first = has.argmax(axis=1)
    last = x - 1 - has[:, ::-1].argmax(axis=1)
    span = float(np.median(np.where(anyc, last - first + 1, 0) / x))
    return min(frac, span / _MIN_SPAN * _MIN_SUPPORT)


def _estimate_h(prob, segments):
    """Espessura típica da pauta (px): mediana dos runs finos ao longo dos caminhos."""
    ink = (prob > _T_RUN).view(np.uint8)
    run_t = _lut()[0]
    runs = []
    for seg in segments:
        track = _Track(seg, prob.shape)
        pattern, _ = track.window(ink)
        run = run_t[pattern]
        ok = (run >= 1) & (run <= 6)
        if ok.any():
            runs.append(run[ok])
    if not runs:
        return None
    return float(np.median(np.concatenate(runs)))


def _gutter_cols(raw):
    """Colunas (X,) dentro de uma faixa de sombra: quase todas as linhas
    engrossam por uma largura maior que um traço de caneta."""
    return _opening_cols((raw.mean(axis=0) >= _GUTTER_FRAC)[None, :], _GUTTER_MIN_W)[0]


def _joined_through(raw, gutter):
    """(K, X): trecho da linha dentro da faixa que liga dois trechos grossos.

    A sombra engrossa todas as linhas por igual, então por espessura um traço
    de caneta deitado sobre ela some na sombra. O que o separa é a continuidade:
    se a linha é grossa (>= pauta + margem) logo antes E logo depois da faixa,
    o que há dentro é o mesmo traço atravessando o vinco, não sombra.
    """
    k, x = raw.shape
    out = np.zeros_like(raw)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], gutter.view(np.uint8), [0]])))
    for a, b in zip(edges[::2], edges[1::2], strict=False):
        pad = _GUTTER_JOIN_PAD
        # O vinco fica na divisa das metades: cada trilha vê só um lado da
        # faixa, e a borda da trilha não conta como ponta a conferir.
        left = raw[:, max(a - pad, 0) : a].any(axis=1) if a > 0 else None
        right = raw[:, b : b + pad].any(axis=1) if b < x else None
        sides = [s for s in (left, right) if s is not None]
        if not sides:
            continue
        both = np.logical_and.reduce(sides)
        out[:, a:b] = both[:, None]
    return out


def _opening_cols(mask, length):
    """Abertura booleana ao longo de x: só sobrevivem trechos de >= length colunas."""
    core = _along(mask.astype(np.float32), length) > 0.999
    return _dilate_cols(core, length // 2)


def _baseline(track, peak, thin):
    """Nível mediano e ruído robusto (1,4826 x MAD) da probabilidade de cada linha."""
    vals = np.where(thin, peak, np.inf).astype(np.float32)
    srt = np.sort(vals, axis=1)
    n = np.maximum(thin.sum(axis=1), 1)
    mid = np.take_along_axis(srt, (n // 2)[:, None], 1)[:, 0]
    mid = np.where(np.isfinite(mid), mid, 1.0)
    dev = np.where(thin, np.abs(peak - mid[:, None]), np.inf).astype(np.float32)
    srt = np.sort(dev, axis=1)
    mad = np.take_along_axis(srt, (n // 2)[:, None], 1)[:, 0]
    mad = np.where(np.isfinite(mad), mad, 0.0)
    track.base_level = mid.astype(np.float32)
    track.base_sigma = np.maximum(1.4826 * mad, 0.01).astype(np.float32)


def erase_tracks(prob, tracks, h_pauta):
    """Apaga a pauta isolada ao longo dos caminhos. Devolve (prob, n, deslocamento).

    Modifica `prob` no lugar. Antes de apagar, reancora o pente: se a mediana
    do desvio entre o caminho e o pico de tinta passa de 1 px (a folha andou
    desde a estimativa), os caminhos são deslocados e o cache refeito.
    """
    run_t, up_t, _down_t, jstar_t = _lut()
    h_run = int(round(h_pauta))
    ink = (prob > _T_RUN).view(np.uint8)
    flat_prob = prob.reshape(-1)  # vista: as escritas valem para `prob`
    h, w = prob.shape
    n_total, shift_total = 0, 0
    for track in tracks:
        pattern, yi = track.window(ink)
        run = run_t[pattern].astype(np.int16)
        thin_any = (run >= 1) & (run <= h_run + _THIN_MARGIN)
        if thin_any.any():
            delta = int(np.round(np.median(jstar_t[pattern][thin_any])))
            if delta != 0:
                track.shift(delta)
                shift_total += delta
                pattern, yi = track.window(ink)
                run = run_t[pattern].astype(np.int16)
        jstar = jstar_t[pattern].astype(np.int64)
        thin = (run >= 1) & (run <= h_run + _THIN_MARGIN)
        # Âncora = cruzamento de verdade: run vertical acima da pauta em
        # >= 2 colunas SEGUIDAS. Um ponto isolado é flutuação de espessura da
        # própria pauta (medido no e5: 14% dos pontos, runs de 6 a 13 px) e,
        # tratado como âncora, protegia 5 colunas e deixava a pauta tracejada.
        raw = run > h_run + _THIN_MARGIN
        gutter = _gutter_cols(raw)
        if gutter.any():
            cap = h_run + _THIN_MARGIN + _GUTTER_EXTRA
            shaded = (run >= 1) & (run <= cap) & gutter[None, :]
            shaded &= ~_joined_through(raw, gutter)
            thin |= shaded
            raw = raw & ~(gutter[None, :] & (run <= cap))
        anchor = np.zeros_like(raw)
        both = raw[:, 1:] & raw[:, :-1]
        anchor[:, 1:] |= both
        anchor[:, :-1] |= both
        protect = _dilate_cols(anchor, _ANCHOR_PAD)
        centers = yi + jstar
        cols = (np.arange(track.paths.shape[1]) + track.x0)[None, :]
        peak = flat_prob.take(centers * w + cols)
        if track.base_level is None:
            _baseline(track, peak, thin)
        # traço de caneta deitado sobre a linha: trecho contínuo cuja
        # probabilidade média passa de 4 sigma do ruído da própria pauta
        smooth = _along(peak, _PROB_SMOOTH)
        margin = np.maximum(_PROB_MARGIN, _PROB_SIGMAS * track.base_sigma)
        above = thin & (smooth >= (track.base_level + margin)[:, None])
        protect |= _opening_cols(above, _PROB_MIN_RUN) & thin
        kill = thin & ~protect
        ks, xs = np.nonzero(kill)
        if ks.size == 0:
            continue
        # apaga a corrida (+1 linha de franja de cada lado) coluna a coluna,
        # uma passada por linha do span, por índice plano (1D)
        up = up_t[pattern[ks, xs]].astype(np.int64)
        length = run[ks, xs].astype(np.int64) + 2
        top = centers[ks, xs] - up - 1
        flat_cols = xs + track.x0
        for j in range(int(length.max())):
            sel = (j < length) & (top + j >= 0) & (top + j < h)
            if sel.any():
                flat_prob[(top[sel] + j) * w + flat_cols[sel]] = 0.0
                n_total += int(sel.sum())
    return prob, n_total, shift_total


class RulingFilter:
    """Filtro com estado: reestima de tempos em tempos e reancora a cada passada."""

    def __init__(self, reestimate_every=24):
        self.reestimate_every = reestimate_every
        self.reset()

    def reset(self):
        self._model = None
        self._tracks = None
        self._age = 0
        self._neg_frac = 1.0
        self.last_info = {}

    def apply(self, prob, visible=None):
        """Devolve `prob` sem a pauta isolada (no lugar). Sem pauta: inalterado."""
        frac = 1.0 if visible is None else float(visible.mean())
        stale = (
            self._tracks is None
            or self._age >= self.reestimate_every
            or self._model.shape != prob.shape
            # negativo dado com a mao tapando boa parte da folha: reestima
            # assim que ela sai (a pauta nao pode ser votada nesse meio tempo)
            or (not self._tracks and frac >= self._neg_frac + _HAND_LEFT)
        )
        if stale:
            self._model = estimate(prob, visible) or RulingModel((), 0.0, prob.shape)
            self._tracks = [_Track(seg, prob.shape) for seg in self._model.segments]
            self._age = 0
            self._neg_frac = frac if not self._tracks else 1.0
            if not self._tracks and frac < _OCCLUDED_BELOW:
                # Nao achou pauta com a mao tapando boa parte da folha: o
                # negativo vale so para esta vista. Tenta de novo em poucas
                # passadas (nao a cada uma: em papel liso a mao esta quase
                # sempre no quadro e a estimativa custa ~13 ms).
                self._age = max(0, self.reestimate_every - _RETRY_PASSES)
        self._age += 1
        if not self._tracks:
            self.last_info = {"lines": 0, "estimated": stale}
            return prob
        prob, erased, shifted = erase_tracks(prob, self._tracks, self._model.h_pauta)
        self.last_info = {
            "lines": sum(t.paths.shape[0] for t in self._tracks),
            "erased": erased,
            "shift": shifted,
            "h_pauta": self._model.h_pauta,
            "estimated": stale,
        }
        return prob


def make_filter():
    """Fábrica para `lab/bench_ink.py --filter draw.ink.ruling:make_filter`."""
    return RulingFilter().apply
