"""Remoção de pauta do mapa de probabilidade da tinta (runtime, sem treino).

O page_clean v2 enxerga a pauta do caderno como tinta (medido ao vivo): ela
entra no canvas, e onde cruza um traço deixa "tracinhos" além da largura da
caneta. Este módulo a tira do mapa de probabilidade ANTES do registro e do
voto, em três passos:

1. Periodicidade e sementes: por faixas verticais, a projeção 1D de um mapa
   de "linearidade" (alto-passa vertical, que ignora hastes) dá, por
   autocorrelação, o período; os picos da projeção dão a posição de cada
   linha, e o pente é estendido até as bordas (linhas extrapoladas só ficam
   se tiverem suporte próprio ao longo da folha).
2. Caminhos estáveis por programação dinâmica: o pente de linhas é rastreado
   com deslocamento `d(x)` (|dy/dx| <= 1) e refinado por linha em banda
   estreita, em resolução reduzida. Esquerda e direita do caderno são
   rastreadas de forma independente.
3. Portão de run-length vertical: mede a espessura contínua da tinta sobre
   cada caminho e apaga o que é compatível com pauta (`<= h_pauta + 2`).
   Cruzamentos (`>= 2` col acima da espessura) e braços sobre a pauta (acima
   do ruído de probabilidade por `_PROB_MIN_RUN` col) são protegidos. Numa
   faixa de sombra do vinco (`_gutter_mask`), o teto sobe de `_GUTTER_EXTRA`
   (`cap`), preservando desvios isolados acima da sombra (`_stroke_on_shade_tracks`)
   e traços que atravessam a faixa (`_joined_through_bands` + floresta conectada).

O custo está em duas parcelas: estimativa a cada `reestimate_every` passadas,
e reancoragem + portão a cada passada.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import NamedTuple

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
_MIN_VISIBLE = 2 * _SUPPORT_W  # colunas na imagem p/ valer uma linha extrapolada
_GUTTER_FRAC = 0.6  # fração das linhas grossas na coluna
_GUTTER_MIN_W = 16  # colunas contínuas (mais largo que qualquer traço de caneta)
_GUTTER_EDGE_RUN = 3  # colunas bastam num trecho que encosta na divisa
_SEAM_WIN = 4  # colunas de cada lado da costura onde as pautas defasadas se ligam
_GUTTER_VOTE_W = _PROB_MIN_RUN  # janela de continuação fora da faixa
_VOTE_FRAC = 0.85  # fração mínima de colunas grossas na janela de voto
_GUTTER_EXTRA = 4  # px a mais de espessura apagável dentro da faixa
_STROKE_RUN = 4  # colunas de desvio de 1 px numa so linha = traco dentro da sombra
_STROKE_SHORT_RUN = 2  # desvio >= 2 px basta em 2 colunas (cruzamento estreito)


def _min_bulk(k):
    """Mínimo de linhas votantes para formar padrão comum de ombro/ondulação."""
    return max(2, k // 4)


@dataclass(frozen=True)
class Segment:
    """Caminhos de uma metade do caderno: y(x) de cada linha em [x0, x1)."""

    x0: int
    x1: int
    paths: np.ndarray  # (K, x1 - x0) float32, px da grade da página
    period: float
    # (K,) True nas linhas extrapoladas além dos picos medidos (None: nenhuma)
    extended: np.ndarray | None = None


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
    """Melhor faixa da metade: (coluna-semente, linhas, extrapoladas?, lag, inclinação)."""
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
    full, ext = _extend_peaks(peaks, score.shape[0], reach)
    return c_seed, full, ext, lag, slope


def _extend_peaks(peaks, h_s, reach):
    """Estende o pente, no passo mediano, até as bordas do mapa.

    Os picos da faixa-semente somem onde a pauta enfraquece (sombra, margens
    e, com a folha inclinada, as linhas que só entram na vista mais longe da
    semente): as linhas além do primeiro/último pico ficavam sem caminho e
    sobreviviam como riscos soltos. As linhas extrapoladas ficam marcadas e só
    valem com suporte próprio (`_drop_unsupported`): uma previsão não se valida
    pelas linhas medidas, nem por uma marca local de caneta que caia nela.
    """
    step = float(np.median(np.diff(peaks)))
    if step < 2:
        return peaks, np.zeros(peaks.size, bool)
    lo = np.arange(peaks[0] - step, -1.0 - reach, -step)[::-1]
    hi = np.arange(peaks[-1] + step, h_s - 1.0 + reach, step)
    full = np.concatenate([lo, peaks, hi]).astype(np.float32)
    ext = np.concatenate(
        [np.ones(lo.size, bool), np.zeros(peaks.size, bool), np.ones(hi.size, bool)]
    )
    return full, ext


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
    c_seed, seeds, ext_rows, lag, slope = found
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
    keep = np.flatnonzero(inside)
    paths = np.stack([np.interp(xs, xc, full[:, k]) for k in keep]).astype(np.float32)
    return Segment(
        x0=x0,
        x1=x1,
        paths=paths,
        period=float(lag * ry),
        extended=ext_rows[keep],
    )


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
    marks = [_ink_on_path(prob, seg) for seg in segments]  # um remap por segmento
    kept = [
        (seg, has)
        for seg, has in zip(segments, marks, strict=True)
        if _support(prob, seg, has) >= _MIN_SUPPORT
    ]
    segments = [_drop_unsupported(prob, seg, has) for seg, has in kept]
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
        self.period = seg.period
        self.shape = shape
        self._mapx = self._mapy = self._yi = self._valid = None
        # nível típico e ruído da própria linha (por estimativa, não por passada)
        self.base_level = None
        self.base_sigma = None

    def _build(self):
        h, _ = self.shape
        k, x = self.paths.shape
        self._valid = (self.paths >= 0) & (self.paths <= h - 1)
        self._yi = np.clip(np.round(self.paths).astype(np.int32), 0, h - 1)
        rows = self._yi[:, None, :] + _J[None, :, None]  # (K, 2R+1, X)
        self._mapy = rows.reshape(k * (2 * _R + 1), x).astype(np.float32)
        cols = (np.arange(x) + self.x0).astype(np.float32)
        self._mapx = np.ascontiguousarray(
            np.broadcast_to(cols[None, :], self._mapy.shape)
        )

    def shift(self, delta):
        """Desloca o pente (px) e invalida o cache."""
        self.paths += delta
        self._mapx = self._mapy = self._yi = self._valid = None

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
        return pattern, self._yi, self._valid


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


def _ink_on_path(prob, seg):
    """(K, X) bool: há tinta fina sobre o caminho. Um remap por segmento."""
    ink = (prob > _T_RUN).view(np.uint8)
    pattern, _, valid = _Track(seg, prob.shape).window(ink)
    return (_lut()[0][pattern] >= 1) & valid


def _support(prob, seg, has):
    """Fração das janelas de _SUPPORT_W px, ao longo da pagina, em que a linha
    tem tinta fina sobre o caminho (mediana das linhas).

    Pauta de verdade corre a largura da folha; um tracejado de caneta, ou
    qualquer coisa periodica numa faixa so, nao: seu suporte e local.
    """
    if seg.extended is not None and not seg.extended.all():
        has = has[~seg.extended]  # só as linhas medidas decidem o segmento
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


def _window_support(has, inside):
    """(frac, span) por linha: fração das janelas de _SUPPORT_W px com tinta e
    extensão da tinta, só nas colunas em que o caminho está na imagem.

    Inclui a última janela, parcial, pela largura real: uma linha inclinada que
    só entra no fim da trilha tem o suporte justamente ali. Esse resto só entra
    na conta quando tem tinta; vazio, não dilui o suporte das janelas inteiras
    (a metade não é múltipla de _SUPPORT_W).
    """
    k, x = has.shape
    n = -(-x // _SUPPORT_W)
    pad = ((0, 0), (0, n * _SUPPORT_W - x))
    win_has = np.pad(has, pad).reshape(k, n, _SUPPORT_W).any(axis=2)
    cnt_in = np.pad(inside, pad).reshape(k, n, _SUPPORT_W).sum(axis=2)
    width = np.minimum(_SUPPORT_W, x - np.arange(n) * _SUPPORT_W)  # largura real
    win_in = cnt_in >= 0.5 * width[None, :]
    partial = width < _SUPPORT_W
    counted = win_in & (win_has | ~partial[None, :])
    frac = (win_has & win_in).sum(axis=1) / np.maximum(counted.sum(axis=1), 1)
    first = has.argmax(axis=1)
    last = x - 1 - has[:, ::-1].argmax(axis=1)
    span = np.where(has.any(axis=1), last - first + 1, 0) / np.maximum(
        inside.sum(axis=1), 1
    )
    return frac, span


def _drop_unsupported(prob, seg, has=None):
    """Tira as linhas extrapoladas sem suporte próprio ao longo da folha.

    Mesmo critério do segmento (janelas de _SUPPORT_W px com tinta fina e
    extensão), mas por linha e só nas colunas em que o caminho está dentro da
    imagem: uma linha inclinada que entra a meio da vista vale pelo trecho
    visível. Uma marca local de caneta não passa (suporte e extensão curtos).
    """
    if seg.extended is None or not seg.extended.any():
        return seg
    h = prob.shape[0]
    if has is None:
        has = _ink_on_path(prob, seg)
    inside = (seg.paths >= 0) & (seg.paths <= h - 1)
    frac, span = _window_support(has, inside)
    ok = np.minimum(frac, span / _MIN_SPAN * _MIN_SUPPORT) >= _MIN_SUPPORT
    # Um trecho visível curto não se distingue de uma marca de caneta alinhada:
    # sem pelo menos duas janelas de vista, a linha prevista não autoriza apagar.
    ok &= inside.sum(axis=1) >= _MIN_VISIBLE
    keep = ~seg.extended | ok
    if keep.all():
        return seg
    return dataclasses.replace(seg, paths=seg.paths[keep], extended=seg.extended[keep])


def _estimate_h(prob, segments):
    """Espessura típica da pauta (px): mediana dos runs finos ao longo dos caminhos."""
    ink = (prob > _T_RUN).view(np.uint8)
    run_t = _lut()[0]
    runs = []
    for seg in segments:
        track = _Track(seg, prob.shape)
        pattern, _, valid = track.window(ink)
        run = run_t[pattern]
        ok = valid & (run >= 1) & (run <= 6)
        if ok.any():
            runs.append(run[ok])
    if not runs:
        return None
    return float(np.median(np.concatenate(runs)))


def _gutter_mask(tracks_raw, width):
    """(W,) colunas da página dentro de uma faixa de sombra.

    A fração de linhas grossas é medida por coluna física e juntada entre as
    metades: o vinco costuma cair na divisa, e uma faixa de 16 colunas
    repartida em 8 + 8 não pode sumir. Conta só trecho contínuo (sem preencher
    a borda). A sombra engrossa quase todas as linhas; um cruzamento de
    caneta também, mas só por poucas colunas.
    """
    frac = np.zeros(width, np.float32)
    for x0, x1, raw, valid in tracks_raw:
        cnt = valid.sum(axis=0)
        frac[x0:x1] = np.where(cnt > 0, raw.sum(axis=0) / np.maximum(cnt, 1), 0.0)
    hot = np.concatenate([[0], frac >= _GUTTER_FRAC, [0]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(hot))
    out = np.zeros(width, bool)
    for a, b in zip(edges[::2], edges[1::2], strict=False):
        if b - a >= _GUTTER_MIN_W:
            out[a:b] = True
    return out


def _leading_run(mask):
    """(K,) comprimento do trecho True no início de cada linha."""
    inv = ~mask
    first = inv.argmax(axis=1)
    return np.where(inv.any(axis=1), first, mask.shape[1])


def _extent(cols, need):
    """(K,) até onde vai o trecho grosso a partir da borda da faixa.

    A primeira janela de `need` colunas vale com a mesma tolerância do voto
    (>= 85% grosso: uma coluna fina logo na borda não zera a medida); dali em
    diante o trecho tem de ser contínuo.
    """
    lead = _leading_run(cols)
    ok = cols[:, :need].mean(axis=1) >= _VOTE_FRAC
    # a faixa pode terminar a exatas `need` colunas da borda da trilha: sem
    # cauda depois da janela de voto, a extensão além dela vale zero
    tail = _leading_run(cols[:, need:]) if cols.shape[1] > need else 0
    return np.where(ok, need + tail, lead)


def _bulk_ext(vals):
    """O comprimento comum dos ombros: o topo do maior grupo de extensões.

    Ombros e traços viram grupos separados por um gap claro (> _GUTTER_EDGE_RUN
    entre extensões vizinhas). O padrão comum é o MAIOR grupo; no empate, o de
    extensões menores (argmax pega o primeiro = viés de preservar tinta). Dois
    traços iguais formam o próprio grupo: nenhum é "ombro um do outro".
    """
    srt = np.sort(vals)
    bounds = np.flatnonzero(np.diff(srt) > _GUTTER_EDGE_RUN) + 1
    groups = np.split(srt, bounds)
    return groups[int(np.argmax([g.size for g in groups]))][-1]


def _match_tracks(t_left, pass_left, t_right, pass_right):
    """Pares (k_left, k_right) com geometria próxima E toque vertical na costura."""
    pat_l, yi_l, v_left = pass_left.pattern, pass_left.yi, pass_left.valid
    pat_r, yi_r, v_right = pass_right.pattern, pass_right.yi, pass_right.valid
    y_l = t_left.paths[:, -1]
    y_r = t_right.paths[:, 0]
    tol = max(float(_R), 0.35 * min(t_left.period, t_right.period))
    dist = np.abs(y_l[:, None] - y_r[None, :]).astype(np.float32)
    _run_t, up_t, down_t, jstar_t = _lut()
    pl, pr = pat_l[:, -1], pat_r[:, 0]
    cl = yi_l[:, -1] + jstar_t[pl]
    cr = yi_r[:, 0] + jstar_t[pr]
    touch = (
        cl[:, None] + down_t[pl][:, None] + 2 >= cr[None, :] - up_t[pr][None, :]
    ) & (cr[None, :] + down_t[pr][None, :] + 2 >= cl[:, None] - up_t[pl][:, None])
    invalid = ~v_left[:, -1][:, None] | ~v_right[:, 0][None, :] | (dist > tol) | ~touch
    dist[invalid] = np.inf
    pairs = []
    while np.isfinite(dist).any():
        kl, kr = np.unravel_index(int(np.argmin(dist)), dist.shape)
        pairs.append((int(kl), int(kr)))
        dist[kl, :] = np.inf
        dist[:, kr] = np.inf
    return pairs


def _gather_side(tracks, thick, matches, idx, line_ids, edge_col, direction):
    """Colunas de `thick` a partir de `edge_col` para fora (cruzando a divisa se preciso)."""
    t = tracks[idx]
    rev = direction < 0
    part = (
        thick[idx][line_ids, : edge_col - t.x0]
        if rev
        else thick[idx][line_ids, edge_col - t.x0 :]
    )
    if rev:
        part = part[:, ::-1]
    j = idx + direction
    if 0 <= j < len(tracks) and (tracks[j].x1 == t.x0 if rev else tracks[j].x0 == t.x1):
        m = {kr: kl for kl, kr in matches[j]} if rev else dict(matches[idx])
        adj = np.zeros((line_ids.size, thick[j].shape[1]), bool)
        for pos, lid in enumerate(line_ids):
            if int(lid) in m:
                row = thick[j][m[int(lid)]]
                adj[pos] = row[::-1] if rev else row
        return np.concatenate([part, adj], axis=1)
    return part


# --- Vocabulário das transições -------------------------------------------
# O que cada função devolve e a próxima consome. NamedTuple mantém o wire
# posicional (desempacota como tupla), mas a construção é por nome — um
# campo não pode trocar de significado sem mudar a interface.


class _Run(NamedTuple):
    """Nó da floresta de tinta: intervalo conectado numa coluna."""

    x: int
    top: int
    bot: int
    pred: int = -1  # índice do nó pai em `nodes`; -1 na raiz

    def visit(self):
        return _Visit(self.x, self.top, self.bot)


class _Visit(NamedTuple):
    """Coluna visitada por caminho qualificado — o que vira marca no joined."""

    x: int
    top: int
    bot: int


class _Continuation(NamedTuple):
    """Prova terminal de um caminho de continuação.

    `x`, `top`, `bot` localizam o ÚLTIMO intervalo grosso do caminho — a
    cauda fina onde o traço parou não decide pousio nem veto de ombro.
    `tail` guarda o trecho grosso consecutivo que produziu esse intervalo
    — é ele, não só o ponto final, que diz se a saída SEGUE uma pauta.
    """

    ext: int  # extensão grossa medida por `_extent` (não é contagem de colunas)
    x: int
    top: int
    bot: int
    tail: tuple = ()  # (x, top, bot) das colunas grossas consecutivas do fim


class _SideStats(NamedTuple):
    """Evidência de um lado da faixa, medida num conjunto de linhas.

    Vale só para vereditos sobre as linhas medidas: `bulk` deste
    conjunto não decide por linhas de outro conjunto.
    """

    cols: np.ndarray  # (K, W) bool — tinta grossa por coluna
    vote: np.ndarray | None  # (K,) bool — passou no voto de espessura
    ext: np.ndarray | None  # (K,) int — extensão grossa por linha
    has_bulk: bool  # linhas bastantes p/ medir o ombro do conjunto
    bulk: int  # ombro típico DO conjunto medido


class _Pass(NamedTuple):
    """Passada de uma trilha: janela amostrada + alturas medidas."""

    pattern: np.ndarray  # (K, X) padrão de bits da janela vertical
    yi: np.ndarray  # (K, X) linha central do caminho
    valid: np.ndarray  # (K, X) bool — caminho dentro da folha
    run: np.ndarray  # (K, X) int — altura da tinta sobre o caminho


@dataclass(frozen=True)
class _PageFold:
    """Carga da página para o veredito do vinco, conferida na fronteira.

    A alfândega é o `__post_init__`: trilhas, passadas e mapas grossos
    atravessam alinhados por índice (um por segmento); `matches` é por
    divisa entre vizinhos e só pode faltar quando `has_shade` é falso.
    Carga fora de compasso é recusada antes de qualquer veredito.
    """

    tracks: list  # _Track por segmento
    passes: list  # _Pass alinhado a `tracks`
    thick: list  # (K, X) bool — tinta acima da pauta, alinhado a `tracks`
    matches: list  # pares (k_l, k_r) por divisa entre trilhas vizinhas
    has_shade: bool  # sombra detectada: obriga uma entrada por divisa

    def __post_init__(self):
        n = len(self.tracks)
        # com sombra, cada divisa entre vizinhas exige uma entrada (mesmo
        # vazia); sem sombra, a carga de pares é vazia
        want = n - 1 if self.has_shade else 0
        aligned = (
            len(self.passes) == n and len(self.thick) == n and len(self.matches) == want
        )
        if not aligned:
            raise ValueError("carga do vinco desalinhada")

    @property
    def valid(self):
        """(K, X) bool por trilha: caminho dentro da folha — campo da _Pass."""
        return [p.valid for p in self.passes]

    def check_joined(self, joined_list):
        """Confere a saída do veredito: uma máscara por trilha, no shape da grossa."""
        ok = len(joined_list) == len(self.thick) and all(
            j.shape == r.shape for j, r in zip(joined_list, self.thick, strict=True)
        )
        if not ok:
            raise ValueError("veredito do vinco desalinhado com a carga")


class _PairedVotes(NamedTuple):
    """Decisão de voto e veto do conjunto pareado numa faixa."""

    ok_l: np.ndarray | None  # voto esquerdo apos veto/borda (None: sem lado)
    ok_r: np.ndarray | None
    vl: np.ndarray | None  # voto cru esquerdo (serve ao resgate por traco)
    vr: np.ndarray | None
    cl: np.ndarray | None  # veto de ombro esquerdo (None: sem referencia)
    cr: np.ndarray | None
    has_pl: bool = False  # conjunto pareado sustenta referência própria (esq)
    has_pr: bool = False


class _FoldJob(NamedTuple):
    """Trabalho do vinco numa passada: a carga, a tinta e a saída parcial."""

    ink: np.ndarray  # mapa binário (H, W)
    page: _PageFold  # carga conferida na fronteira
    joined: list  # máscaras (K, X) — os blocos escrevem aqui
    h_run: int  # altura típica da pauta
    need: int  # janela de voto exigida


def _trace_ink_run(ink, yc0, seam_col, limit, direction, stop_col, clip_r=2 * _R):
    """Traça floresta de runs de tinta conectados a partir de `seam_col`.

    Devolve `(nodes, exit_tips)` onde `nodes` é lista de `_Run` até
    `stop_col` (crescimento vertical limitado a `±clip_r` do centro
    anterior) e `exit_tips` são os índices na última coluna da faixa.
    """
    h = ink.shape[0]
    lo0, hi0 = max(0, yc0 - _R), min(h, yc0 + _R + 1)
    if not ink[lo0:hi0, seam_col].any():
        return [], []
    ys0 = np.flatnonzero(ink[lo0:hi0, seam_col]) + lo0
    mid = int(ys0[int(np.argmin(np.abs(ys0 - yc0)))])
    t0 = b0 = mid
    while t0 > 0 and ink[t0 - 1, seam_col] and mid - (t0 - 1) <= clip_r:
        t0 -= 1
    while b0 + 1 < h and ink[b0 + 1, seam_col] and (b0 + 1) - mid <= clip_r:
        b0 += 1

    nodes = [_Run(seam_col, t0, b0)]
    active = [0]
    for x in range(seam_col + direction, limit, direction):
        inside_band = (x < stop_col) if direction > 0 else (x > stop_col)
        if not inside_band:
            break
        col = ink[:, x]
        nxt = []
        seen_top = {}
        for nid in active:
            nd = nodes[nid]
            pc = (nd.top + nd.bot) // 2
            lo, hi = max(0, nd.top), min(h, nd.bot + 1)
            if not col[lo:hi].any():
                continue
            for hit in np.flatnonzero(col[lo:hi]) + lo:
                top = bot = int(hit)
                while top > 0 and col[top - 1] and pc - (top - 1) <= clip_r:
                    top -= 1
                while bot + 1 < h and col[bot + 1] and (bot + 1) - pc <= clip_r:
                    bot += 1
                if top in seen_top:
                    continue
                idx = len(nodes)
                nodes.append(_Run(x, top, bot, nid))
                seen_top[top] = idx
                nxt.append(idx)
        if not nxt:
            return nodes, []
        active = nxt
    return nodes, active


def _trace_branch_out(ink, x_start, top0, bot0, limit, direction, h_run, need, stop):
    """Lista de `_Continuation` por caminho conectado que completou a janela.

    Cada intervalo vivo carrega a própria sequência de espessura e nunca a
    mistura com a de outro ramo: quando dois caminhos ocupam as mesmas
    linhas na coluna (um reencontro), cada um reivindica a própria janela
    — intervalos sobrepostos coexistem como camadas — e, ao se separarem,
    seguem individualmente sem interferência. Um ramo grosso não empresta
    evidência ao fino que passa pelos mesmos pixels, nem o fino enterra a
    história grossa do outro.
    """
    h = ink.shape[0]
    # (top, bot, hist, last_thick, tail, last_tail): cada caminho vivo
    # carrega a sequência de espessura, a posição do último intervalo
    # grosso e o trecho grosso consecutivo que o produziu — é o trecho,
    # não só o ponto final, que responde ao veto de ombro
    active = [(top0, bot0, (), None, (), ())]
    ends = []  # (hist, last_thick, last_tail) de cada caminho encerrado
    x = x_start
    for x in range(x_start + direction, limit, direction):
        col = ink[:, x]
        nxt = []
        for pt, pb, hist, lt, tail, lt_tail in active:
            pc = (pt + pb) // 2
            lo, hi = max(0, pt), min(h, pb + 1)
            mine = -1  # último intervalo deste pai em nxt
            for hit in np.flatnonzero(col[lo:hi]) + lo:
                if mine >= 0 and hit <= nxt[mine][1]:
                    continue  # mesmo trecho contíguo deste pai
                top = bot = int(hit)
                while top > 0 and col[top - 1] and pc - (top - 1) <= 2 * _R:
                    top -= 1
                while bot + 1 < h and col[bot + 1] and (bot + 1) - pc <= 2 * _R:
                    bot += 1
                nxt.append((top, bot, hist, lt, tail, lt_tail))
                mine = len(nxt) - 1
            if mine < 0:
                ends.append((hist, lt, lt_tail))
        if not nxt:
            active = []
            break
        seen = {}
        for top, bot, hist, lt, tail, lt_tail in nxt:
            thick = (bot - top + 1) > h_run + _THIN_MARGIN
            h2 = hist + (thick,)
            tail2 = (tail + ((x, top, bot),))[-stop:] if thick else ()
            lt2 = (x, top, bot) if thick else lt
            lt_tail2 = tail2 if thick else lt_tail
            if len(h2) >= need and (not h2[-1] or len(h2) >= stop):
                ends.append((h2, lt2, lt_tail2))
                continue
            state = (top, bot, h2, lt2, tail2, lt_tail2)
            # reencontros repetidos podem produzir estados idênticos: sem
            # dedup, cada cópia refaz o mesmo traço e dobra na próxima
            # bifurcação (16 ciclos medidos: 65.536 caminhos, >2 s). A
            # chave exclui o tail — em empate fica o trecho mais curto,
            # que é o menos provável de fechar o veto de ombro
            key = (top, bot, h2, lt2)
            prev = seen.get(key)
            if prev is None or len(lt_tail2) < len(prev[5]):
                seen[key] = state
        active = list(seen.values())
        if not active:
            break
    for _top, _bot, hist, lt, _tail, lt_tail in active:
        ends.append((hist, lt, lt_tail))
    out = []
    for hist, lt, lt_tail in ends:
        if len(hist) < need or lt is None:
            continue
        arr = np.array(hist, bool)[None, :]
        if arr[:, :need].mean() < _VOTE_FRAC:
            continue
        out.append(_Continuation(int(_extent(arr, need)[0]), *lt, lt_tail))
    return list(dict.fromkeys(out))


def _mark_visited_track(joined_track, track, valid_track, visited):
    """Marca em `joined_track` apenas (k, x) visitados pelo traço conectado."""
    for x, top, bot in visited:
        if not (track.x0 <= x < track.x1):
            continue
        lx = x - track.x0
        mid = (top + bot) // 2
        lo, hi = max(top, mid - 2 * _R), min(bot, mid + 2 * _R)
        hit = (
            valid_track[:, lx]
            & (track.paths[:, lx] >= lo - 1)
            & (track.paths[:, lx] <= hi + 1)
        )
        joined_track[hit, lx] = True


def _ruling_hit(tracks, valid_list, x, top, bot):
    """(i, k) do caminho válido coberto por (top, bot) na coluna x, ou None.

    O terminal pode sair da trilha da faixa quando ela termina antes da
    divisa: a trilha certa é a da coluna x, não a da faixa. Devolve o
    caminho coberto mais próximo do centro do intervalo.
    """
    for i, t in enumerate(tracks):
        if t.x0 <= x < t.x1:
            lx = x - t.x0
            hit = np.flatnonzero(
                valid_list[i][:, lx]
                & (t.paths[:, lx] >= top - 2)
                & (t.paths[:, lx] <= bot + 2)
            )
            if not hit.size:
                return None
            mid = (top + bot) / 2
            return i, int(hit[np.argmin(np.abs(t.paths[hit, lx] - mid))])
    return None


def _rule_run(tracks, i, k, tail):
    """Colunas grossas consecutivas do fim de `tail` que seguem a pauta k.

    A pauta segue por y(x): dentro da trilha, `paths[k, x-x0]`; fora, o
    valor extrapolado na borda (`paths[k, 0]`/`[-1]`) — é o que mantém o
    veto do ombro ALINHADO atravessando a costura e, ao mesmo tempo,
    libera a dobra que só pousa na pauta deslocada: o ombro segue a pauta
    em todas as colunas da saída, a dobra só nas últimas.
    """
    t = tracks[i]
    run = 0
    for x, ctop, cbot in reversed(tail):
        if t.x0 <= x < t.x1:
            y = t.paths[k, x - t.x0]
        elif x < t.x0:
            y = t.paths[k, 0]
        else:
            y = t.paths[k, -1]
        if ctop - 2 <= y <= cbot + 2:
            run += 1
        else:
            break
    return run


def _seam_bridge(nodes, tid, band_edge, direction, split, src_tops):
    """O caminho cruza a divisa por um remendo alto (sombra) ou por dobra?

    Uma dobra distribui o desnível entre as pautas ao longo das colunas: em
    nenhuma coluna o intervalo cobre o desnível inteiro. Um remendo de sombra
    o concentra: perto da divisa há um intervalo de altura >= espessura +
    desnível (da pauta de origem à de destino) ou, quando o remendo liga as
    pautas por intervalos que só se sobrepõem em colunas vizinhas, duas
    colunas seguidas quase tão altas (desnível menos `_THIN_MARGIN`).

    A evidência é medida ONDE as pautas se ligam — as colunas a até
    `_SEAM_WIN` da divisa `split` (primeira coluna do destino) —, nunca no
    trecho todo: um cruzamento de caneta longe da divisa é alto mas não liga
    pauta nenhuma, e um remendo largo não pode esconder a própria ponte
    subindo a mediana. Os níveis das duas pautas e a espessura de referência
    saem das colunas FORA da janela: o lado do destino vem da cadeia, o da
    origem vem da cadeia (quando o traço já nasce antes da divisa) e do
    traço de volta `src_tops`. Sem nível de origem observável, abstém.
    """
    chain = []
    cur = tid
    while cur != -1:
        n = nodes[cur]
        if (band_edge - n.x) * direction > 0:
            chain.append(n)
        cur = n.pred
    near = sorted(
        (n for n in chain if abs(n.x - split) <= _SEAM_WIN), key=lambda n: n.x
    )
    far_dst = [
        n
        for n in chain
        if (n.x - split) * direction >= 0 and abs(n.x - split) > _SEAM_WIN
    ]
    far_src_chain = [
        n.top
        for n in chain
        if (n.x - split) * direction < 0 and abs(n.x - split) > _SEAM_WIN
    ]
    src = list(src_tops) + far_src_chain
    if not near or not far_dst or not src:
        return False
    base = float(np.median([n.bot - n.top + 1 for n in far_dst]))
    # desnível entre os topos das duas pautas, medidos fora da janela
    gap = abs(float(np.median([n.top for n in far_dst])) - float(np.median(src)))
    if gap <= _THIN_MARGIN:
        return False
    h = [n.bot - n.top + 1 for n in near]
    if max(h) >= base + gap:
        return True
    # remendo escalonado: duas colunas seguidas cobrem o desnível (menos a
    # margem) E passam de 1,5 espessura. O piso evita que uma dobra suave,
    # cujo intervalo só engorda ao encostar na pauta sombreada (espessura +
    # poucos px), passe por remendo quando o desnível é pequeno
    thr = max(base + gap - _THIN_MARGIN, 1.5 * base)
    tall = [x >= thr for x in h]
    return any(a and b for a, b in zip(tall, tall[1:], strict=False))


def _project_forest(
    ink,
    t_src,
    k_src,
    valid_src,
    seam_col,
    band_in_edge,
    band_out_edge,
    direction,
    h_run,
    need,
    tracks,
    valid_list,
    has_dst=False,
    bulk_dst=0,
    require_src_out=False,
    require_bend=False,
):
    """(visited_src, visited_dst): listas de `_Visit`, floresta + ramos filtrados.

    O veto de ombro é por caminho conectado: só vale quando a continuação
    do próprio caminho acaba numa pauta do destino sem passar do ombro
    comum daquele lado. Caminhos terminando em espaço em branco não têm
    ombro de referência e não são vetados — uma ponta irmã seguindo pauta
    não decide por eles. A janela exigida é sempre `need`: sem espaço para
    medi-la, quem decide é o chamador (abster), não a continuação parcial.
    """
    if not valid_src[seam_col - t_src.x0]:
        return [], []
    w = ink.shape[1]
    yc0 = int(round(float(t_src.paths[k_src, seam_col - t_src.x0])))
    limit_dst = w if direction > 0 else -1
    nodes, tips = _trace_ink_run(ink, yc0, seam_col, limit_dst, direction, band_in_edge)
    # sem ponta de saída o veredito é vazio de qualquer forma: o traço de
    # volta na origem (o ramo mais caro) não precisa rodar
    if not tips:
        return [], []
    # traça de volta na origem: preserva só colunas conectadas ao traço na costura
    limit_src = -1 if direction > 0 else w
    stop_src = band_out_edge - 1 if direction > 0 else band_out_edge + 1
    nodes_s, tips_s = _trace_ink_run(
        ink, yc0, seam_col, limit_src, -direction, stop_src
    )
    # divisa entre as trilhas e topos da pauta de origem (floresta de volta,
    # fora da janela da divisa)
    split = t_src.x1 if direction > 0 else t_src.x0
    src_tops = [n.top for n in nodes_s if abs(n.x - split) > _SEAM_WIN]
    stop_dst = max(need, bulk_dst + _GUTTER_EDGE_RUN + 1) if has_dst else need + 1
    qual_dst = set()
    qual_tips = []
    for tid in tips:
        tip = nodes[tid]
        paths = _trace_branch_out(
            ink, tip.x, tip.top, tip.bot, limit_dst, direction, h_run, need, stop_dst
        )
        # o veto de ombro decide por caminho: só descarta quando TODA
        # continuação qualificada acaba curta sobre pauta do destino — um
        # irmão de sombra não veta o ramo que termina em espaço em branco
        clean = False
        for cont in paths:
            hit = None
            if has_dst and cont.ext <= bulk_dst + _GUTTER_EDGE_RUN:
                hit = _ruling_hit(tracks, valid_list, cont.x, cont.top, cont.bot)
            shadow = hit is not None and _rule_run(tracks, *hit, cont.tail) >= min(
                cont.ext, need
            )
            if not shadow:
                clean = True
                break
        if not clean:
            continue
        if require_bend and _seam_bridge(
            nodes, tid, band_in_edge, direction, split, src_tops
        ):
            continue
        qual_tips.append(tid)
        cur = tid
        while cur != -1:
            qual_dst.add(cur)
            cur = nodes[cur].pred
    if not qual_dst:
        return [], []
    visited_dst = [nodes[i].visit() for i in sorted(qual_dst)]
    if require_src_out:
        # a saída do lado de origem é provada a partir das pontas qualificadas:
        # uma dobra pode sair da faixa por uma linha diferente da pauta de onde
        # o traço começou, então a floresta de volta nasce na ponta e não no
        # caminho da linha
        qual_s = set()
        for tid in qual_tips:
            tip = nodes[tid]
            nodes_b, tips_b = _trace_ink_run(
                ink, (tip.top + tip.bot) // 2, tip.x, limit_src, -direction, stop_src
            )
            for tb in tips_b:
                tip_b = nodes_b[tb]
                if not _trace_branch_out(
                    ink,
                    tip_b.x,
                    tip_b.top,
                    tip_b.bot,
                    limit_src,
                    -direction,
                    h_run,
                    need,
                    need + 1,
                ):
                    continue
                cur = tb
                while cur != -1:
                    qual_s.add(nodes_b[cur].visit())
                    cur = nodes_b[cur].pred
        if not qual_s:
            return [], []
        visited_src = sorted(qual_s)
    else:
        visited_src = [n.visit() for n in nodes_s]
    return visited_src, visited_dst


def _side_bulks(tracks, thick, matches, idx, a, b, need):
    """(left_info, right_info) por lado: `_SideStats` do conjunto da trilha."""
    t = tracks[idx]
    all_k = np.arange(t.paths.shape[0], dtype=np.int64)
    min_b = _min_bulk(t.paths.shape[0])
    out = []
    for edge, direction in ((a, -1), (b, +1)):
        cols = _gather_side(tracks, thick, matches, idx, all_k, edge, direction)
        vote = (
            cols[:, :need].mean(axis=1) >= _VOTE_FRAC if cols.shape[1] >= need else None
        )
        has_b = vote is not None and vote.sum() > min_b
        ext = _extent(cols, need) if vote is not None else None
        bulk = _bulk_ext(ext[vote]) if has_b else 0
        out.append(_SideStats(cols, vote, ext, has_b, bulk))
    return out[0], out[1]


def _project_line(
    ink,
    tracks,
    valid_list,
    joined,
    i_src,
    k_src,
    direction,
    a,
    b,
    h_run,
    need,
    i_dst=None,
    has_d=False,
    bulk_d=0,
    require_src_out=False,
    require_bend=False,
    seam_override=None,
):
    """Projeta uma linha pela costura via `_project_forest` e marca só visitadas.

    Devolve True quando algum ramo se qualificou (saída grossa do lado do
    destino, mais a da origem se `require_src_out`).
    """
    t_s = tracks[i_src]
    if seam_override is not None:
        seam = int(np.clip(seam_override, t_s.x0, t_s.x1 - 1))
    else:
        seam = t_s.x1 - 1 if direction > 0 else t_s.x0
    in_e = b if direction > 0 else a - 1
    out_e = a if direction > 0 else b
    t_d = tracks[i_dst] if i_dst is not None else None
    v_d = valid_list[i_dst] if i_dst is not None else None
    v_s, v_dst = _project_forest(
        ink,
        t_s,
        int(k_src),
        valid_list[i_src][k_src],
        seam,
        in_e,
        out_e,
        direction,
        h_run,
        need,
        tracks,
        valid_list,
        has_d,
        bulk_d,
        require_src_out,
        require_bend,
    )
    if not v_dst:
        return False
    for x, _t, _b in v_s:
        if t_s.x0 <= x < t_s.x1:
            joined[i_src][k_src, x - t_s.x0] = True
    if seam_override is not None:
        _mark_visited_track(joined[i_src], t_s, valid_list[i_src], v_dst)
    if t_d is not None:
        _mark_visited_track(joined[i_dst], t_d, v_d, v_dst)
    return True


def _verdict_paired(side_l, side_r, kl, kr, l_vot, r_vot, l_edge, r_edge):
    """Voto + veto de ombro do conjunto pareado (ou trilha única) na faixa.

    Bloco puro de decisão — não olha tinta, não marca nada. `ok_*` é o voto
    após o veto e as bordas; `v*`, `c*` seguem crus para o resgate por traço.
    """
    vl = side_l.vote[kl] if l_vot else None
    vr = side_r.vote[kr] if r_vot else None
    # Referência de ombro comum por lado, em hierarquia: o conjunto pareado
    # quando ele próprio a sustenta (has_p), senão o lado inteiro (ombro
    # comum não precisa de par para ser medido), senão nada.
    has_pl = vl is not None and int(vl.sum()) > _min_bulk(kl.size)
    has_pr = vr is not None and int(vr.sum()) > _min_bulk(kl.size)
    ref_l = (
        _bulk_ext(side_l.ext[kl][vl])
        if has_pl
        else (side_l.bulk if l_vot and side_l.has_bulk else None)
    )
    ref_r = (
        _bulk_ext(side_r.ext[kr][vr])
        if has_pr
        else (side_r.bulk if r_vot and side_r.has_bulk else None)
    )
    # c*: a linha continua ALÉM do ombro comum do lado? ext > ref subsum o
    # voto — quem não votou tem ext < need <= ref e falha sozinho.
    cl = side_l.ext[kl] > ref_l + _GUTTER_EDGE_RUN if ref_l is not None else None
    cr = side_r.ext[kr] > ref_r + _GUTTER_EDGE_RUN if ref_r is not None else None
    ok_l = vl.copy() if vl is not None else None
    ok_r = vr.copy() if vr is not None else None
    if vl is not None and vr is not None:
        if not (has_pl or has_pr):
            # par esparso: nenhum conjunto pareado sustenta referência
            # própria. As refs de fallback do lado inteiro medem o ombro
            # para o veto/traço, mas não autorizam o voto cru — uma linha
            # pode votar nas bordas e ainda estar desconectada no meio da
            # faixa. A linha votada é provada pelo traço (_mark_paired)
            ok_l = np.zeros(kl.size, bool)
            ok_r = np.zeros(kl.size, bool)
        elif cl is not None and cr is not None:
            # o veto de ombro só fecha o par quando os dois lados têm
            # referência (um ombro unilateral não veta)
            ok_l &= cl
            ok_r &= cr
    else:
        # lado único observável: borda de imagem aceita o voto com o veto do
        # próprio ombro; cauda curta decide linha a linha pelo traço conectado
        if vl is not None and r_edge and cl is not None:
            ok_l &= cl
        if vr is not None and l_edge and cr is not None:
            ok_r &= cr
    if l_edge and r_edge:
        # nenhum lado observável: sem evidência não há isenção — sombra
        # cobrindo o crop inteiro sai por completo
        ok_l = ok_r = np.zeros(kl.size, bool)
    else:
        if l_edge and ok_l is None:
            ok_l = np.ones(kl.size, bool)
        if r_edge and ok_r is None:
            ok_r = np.ones(kl.size, bool)
    return _PairedVotes(ok_l, ok_r, vl, vr, cl, cr, has_pl, has_pr)


def _mark_paired(job, a, b, i_l, i_r, kl, kr, pv, side_l, side_r):
    """Trecho decidido por voto/borda: marca a faixa e resgata dobras.

    Protege o trecho inteiro da faixa nas linhas que passaram nos dois
    lados; linha que passou só num lado (dobra saindo pela pauta vizinha)
    ainda é verificada pelo traço conectado de `_project_line`.
    """
    tracks, valid_list, joined = job.page.tracks, job.page.valid, job.joined
    cont = pv.ok_l & pv.ok_r
    if i_l == i_r:
        t = tracks[i_l]
        joined[i_l][kl, a - t.x0 : b - t.x0] |= cont[:, None]
        if pv.vl is not None:
            for idx_c in np.flatnonzero(~cont & pv.vl):
                if pv.cl is None or pv.cl[idx_c] or not pv.has_pl:
                    _project_line(
                        job.ink,
                        tracks,
                        valid_list,
                        joined,
                        i_l,
                        kl[idx_c],
                        1,
                        a,
                        b,
                        job.h_run,
                        job.need,
                        i_l,
                        side_r.has_bulk,
                        side_r.bulk,
                        seam_override=a,
                    )
        if pv.vr is not None:
            for idx_c in np.flatnonzero(~cont & pv.vr):
                if pv.cr is None or pv.cr[idx_c] or not pv.has_pr:
                    _project_line(
                        job.ink,
                        tracks,
                        valid_list,
                        joined,
                        i_l,
                        kr[idx_c],
                        -1,
                        a,
                        b,
                        job.h_run,
                        job.need,
                        i_l,
                        side_l.has_bulk,
                        side_l.bulk,
                        seam_override=b - 1,
                    )
    else:
        t_l, t_r = tracks[i_l], tracks[i_r]
        joined[i_l][kl, a - t_l.x0 :] |= cont[:, None]
        joined[i_r][kr, : b - t_r.x0] |= cont[:, None]
        # dobra dentro da faixa: passou num lado e falhou no outro (esq <-> dir)
        if pv.vl is not None and pv.vr is not None:
            # o ombro do destino é medido no conjunto pareado, não nas
            # linhas soltas do lado — um par esparso (sem ref própria) não
            # pode morrer pelo ombro de pautas que não são as dele; nesse
            # caso a ponte/bend decide por geometria
            bulk_pr = _bulk_ext(side_r.ext[kr][pv.vr]) if pv.has_pr else 0
            bulk_pl = _bulk_ext(side_l.ext[kl][pv.vl]) if pv.has_pl else 0
            for idx_c in np.flatnonzero(~cont & pv.vl):
                if pv.cl is None or pv.cl[idx_c] or not pv.has_pl:
                    _project_line(
                        job.ink,
                        tracks,
                        valid_list,
                        joined,
                        i_l,
                        kl[idx_c],
                        1,
                        a,
                        b,
                        job.h_run,
                        job.need,
                        i_r,
                        pv.has_pr,
                        bulk_pr,
                        require_bend=not pv.has_pr,
                    )
            for idx_c in np.flatnonzero(~cont & pv.vr):
                if pv.cr is None or pv.cr[idx_c] or not pv.has_pr:
                    _project_line(
                        job.ink,
                        tracks,
                        valid_list,
                        joined,
                        i_r,
                        kr[idx_c],
                        -1,
                        a,
                        b,
                        job.h_run,
                        job.need,
                        i_l,
                        pv.has_pl,
                        bulk_pl,
                        require_bend=not pv.has_pl,
                    )


def _verdict_short_tail(
    job, a, b, i_l, i_r, kl, kr, pv, side_l, side_r, l_short, r_short
):
    """Cauda curta: lado sem janela decide linha a linha pelo traço.

    O lado sem espaço para a janela de voto é provado pelo traço conectado
    a partir do limite da faixa (`require_src_out` prova também a saída do
    outro lado, cobrindo o resgate de linha reprovada só no voto).
    """
    tracks, valid_list, joined = job.page.tracks, job.page.valid, job.joined
    for idx_c in range(kl.size):
        l_ok = pv.ok_l[idx_c] if pv.ok_l is not None else None
        r_ok = pv.ok_r[idx_c] if pv.ok_r is not None else None
        if r_short:
            l_resc = pv.vl is not None and not pv.vl[idx_c]
            if l_ok is False and not l_resc and (pv.has_pl or pv.has_pr):
                # veto de ombro fechado: não traça nem marca. Par esparso
                # (sem ref própria) tem ok zerado só para forçar a prova —
                # o traço ainda tem de rodar
                continue
            r_ok = _project_line(
                job.ink,
                tracks,
                valid_list,
                joined,
                i_l,
                kl[idx_c],
                1,
                a,
                b,
                job.h_run,
                job.need,
                i_dst=(i_r if i_l != i_r else i_l),
                has_d=pv.has_pr,
                bulk_d=(_bulk_ext(side_r.ext[kr][pv.vr]) if pv.has_pr else 0),
                require_src_out=l_short or l_resc,
                require_bend=not pv.has_pr,
                seam_override=a,
            )
            if l_short or l_resc:
                l_ok = r_ok
        elif l_short:
            r_resc = pv.vr is not None and not pv.vr[idx_c]
            if r_ok is False and not r_resc and (pv.has_pl or pv.has_pr):
                continue
            l_ok = _project_line(
                job.ink,
                tracks,
                valid_list,
                joined,
                i_l if i_l == i_r else i_r,
                (kl if i_l == i_r else kr)[idx_c],
                -1,
                a,
                b,
                job.h_run,
                job.need,
                i_dst=i_l,
                has_d=pv.has_pl,
                bulk_d=(_bulk_ext(side_l.ext[kl][pv.vl]) if pv.has_pl else 0),
                require_src_out=r_resc,
                require_bend=not pv.has_pl,
                seam_override=b - 1,
            )
            if r_resc:
                r_ok = l_ok
        # manter exige os dois lados; a marcação já veio do traço


def _verdict_unmatched(job, a, b, i_l, i_r, kl, kr, side_l, side_r):
    """Linhas sem par na divisa: cada uma decide pelo traço do destino.

    O ombro da origem é provisório — um traço com comprimento de ombro pode
    continuar em branco (ombro de verdade não passa do veto por ramo).
    """
    if i_l == i_r:
        return
    tracks, valid_list, joined = job.page.tracks, job.page.valid, job.joined
    for i_s, i_d, matched_s, has_d, b_d, d in (
        (i_l, i_r, kl, side_r.has_bulk, side_r.bulk, 1),
        (i_r, i_l, kr, side_l.has_bulk, side_l.bulk, -1),
    ):
        unmatched = np.setdiff1d(np.arange(tracks[i_s].paths.shape[0]), matched_s)
        for k in unmatched:
            _project_line(
                job.ink,
                tracks,
                valid_list,
                joined,
                i_s,
                k,
                d,
                a,
                b,
                job.h_run,
                job.need,
                i_d,
                has_d,
                b_d,
            )


def _joined_through_bands(page, gutter_full, ink, h_run):
    """Lista de máscaras (K, X): trecho da linha na faixa que continua grosso fora dela.

    Cada faixa física de `gutter_full` é julgada uma vez só: linhas pareadas
    usam o veredito conjunto das duas pontas; linhas sem par ou dobrando na
    faixa usam a floresta conectada de `_project_forest`, marcando apenas as
    colunas efetivamente visitadas por caminhos com continuação qualificada.
    """
    tracks, thick, matches = page.tracks, page.thick, page.matches
    joined = [np.zeros_like(raw) for raw in thick]
    edges = np.flatnonzero(
        np.diff(np.concatenate([[0], gutter_full.view(np.uint8), [0]]))
    )
    if not tracks or edges.size == 0:
        return joined
    w = gutter_full.size
    need = _GUTTER_VOTE_W
    job = _FoldJob(ink, page, joined, h_run, need)
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        over = [i for i, t in enumerate(tracks) if t.x0 < b and t.x1 > a]
        if not over:
            continue
        if len(over) == 1:
            i_l = i_r = over[0]
            kl = kr = np.arange(tracks[i_l].paths.shape[0], dtype=np.int64)
        elif (
            len(over) == 2
            and over[1] == over[0] + 1
            and tracks[over[0]].x1 == tracks[over[1]].x0
        ):
            i_l, i_r = over
            pairs = matches[i_l]
            kl = np.array([p[0] for p in pairs], dtype=np.int64)
            kr = np.array([p[1] for p in pairs], dtype=np.int64)
        else:
            continue

        side_l, _ = _side_bulks(tracks, thick, matches, i_l, a, b, need)
        _, side_r = _side_bulks(tracks, thick, matches, i_r, a, b, need)
        # Evidência por lado: janela de voto (>= need colunas), cauda curta
        # (espaço observável — trilha ou fora dela — sem janela: a continuação
        # tem de sair do traço, nunca de voto unilateral) ou borda da imagem
        # (nada observável).
        l_vot = side_l.cols.shape[1] >= need
        r_vot = side_r.cols.shape[1] >= need
        l_edge = a == 0
        r_edge = b == w
        l_short = not l_vot and not l_edge
        r_short = not r_vot and not r_edge

        # 1. Linhas pareadas (ou trilha única): o bloco de veredito decide,
        #    os blocos de marcação executam
        if kl.size > 0:
            pv = _verdict_paired(side_l, side_r, kl, kr, l_vot, r_vot, l_edge, r_edge)
            if not (l_short or r_short):
                _mark_paired(job, a, b, i_l, i_r, kl, kr, pv, side_l, side_r)
            else:
                _verdict_short_tail(
                    job,
                    a,
                    b,
                    i_l,
                    i_r,
                    kl,
                    kr,
                    pv,
                    side_l,
                    side_r,
                    l_short,
                    r_short,
                )

        # 2. Linhas sem par na divisa (esq -> dir e dir -> esq)
        _verdict_unmatched(job, a, b, i_l, i_r, kl, kr, side_l, side_r)
    return joined


def _long_runs(mask, length, edge_length):
    """(K, X): trechos contínuos de `mask` com >= length colunas.

    Um trecho que encosta na borda da trilha (a divisa das metades) vale com
    `edge_length`: o resto dele está na outra trilha, que a daqui não vê.
    """
    k, x = mask.shape
    padded = np.zeros((k, x + 2), np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1)
    rs, rc = np.nonzero(d == 1)  # início de cada trecho
    es, ec = np.nonzero(d == -1)  # fim (exclusivo)
    out = np.zeros_like(mask)
    n = ec - rc
    need = np.where((rc == 0) | (ec == x), edge_length, length)
    for r, a, b in zip(rs[n >= need], rc[n >= need], ec[n >= need], strict=True):
        out[r, a:b] = True
    return out


def _shade_deviations(run, gutter, valid):
    """(above, strong): desvios isolados da espessura em relação às linhas vizinhas."""
    n = valid.sum(axis=0)
    med = np.zeros(run.shape[1], np.float32)
    has = n > 0
    if has.any():
        srt = np.sort(
            np.where(valid[:, has], run[:, has].astype(np.float32), np.inf), axis=0
        )
        nh = n[has]
        cols = np.arange(nh.size)
        med[has] = 0.5 * (srt[(nh - 1) // 2, cols] + srt[nh // 2, cols])
    above = valid & (run > med[None, :]) & gutter[None, :]
    # uma ondulação da sombra desvia várias linhas na mesma coluna; um traço
    # desvia só a linha que ele toca — conta só o desvio isolado
    base = np.broadcast_to(med[None, :].astype(np.float32), run.shape).copy()
    shared = above.sum(axis=0) > _min_bulk(above.shape[0])
    if shared.any():
        sel = above[:, shared]
        vals = np.where(sel, run[:, shared].astype(np.float32), np.inf)
        srt = np.sort(vals, axis=0)
        n = sel.sum(axis=0)
        # o nível da ondulação é o desvio comum da coluna; quem vai acima
        # dele segue valendo como traço
        level = srt[(n - 1) // 2, np.arange(sel.shape[1])]
        base[:, shared] = level[None, :]
        above[:, shared] = sel & (run[:, shared] > level[None, :])
    strong = above & (run >= base + 2)
    return above, strong


def _stroke_on_shade_tracks(page, gutter_full):
    """Lista de máscaras (K, X) de traços na sombra, contando desvios através da divisa."""
    tracks, matches = page.tracks, page.matches
    strokes = [np.zeros_like(p.valid) for p in page.passes]
    if not tracks or not gutter_full.any():
        return strokes
    devs = [None] * len(tracks)
    for i, (t, p) in enumerate(zip(tracks, page.passes, strict=True)):
        gutter = gutter_full[t.x0 : t.x1]
        if not gutter.any():
            continue
        above, strong = _shade_deviations(p.run, gutter, p.valid)
        devs[i] = (above, strong)
        strokes[i] = _long_runs(above, _STROKE_RUN, _STROKE_RUN) | _long_runs(
            strong, _STROKE_SHORT_RUN, _STROKE_SHORT_RUN
        )
    for i in range(len(tracks) - 1):
        xc = tracks[i].x1
        if (
            tracks[i + 1].x0 != xc
            or not (gutter_full[xc - 1] and gutter_full[xc])
            or devs[i] is None
            or devs[i + 1] is None
            or not matches[i]
        ):
            continue
        kl = np.array([p[0] for p in matches[i]], dtype=np.int64)
        kr = np.array([p[1] for p in matches[i]], dtype=np.int64)
        wl = min(_STROKE_RUN, tracks[i].paths.shape[1])
        wr = min(_STROKE_RUN, tracks[i + 1].paths.shape[1])
        comb_above = np.concatenate(
            [devs[i][0][kl, -wl:], devs[i + 1][0][kr, :wr]], axis=1
        )
        comb_strong = np.concatenate(
            [devs[i][1][kl, -wl:], devs[i + 1][1][kr, :wr]], axis=1
        )
        comb_ok = _long_runs(comb_above, _STROKE_RUN, _STROKE_RUN + 1) | _long_runs(
            comb_strong, _STROKE_SHORT_RUN, _STROKE_SHORT_RUN + 1
        )
        strokes[i][kl, -wl:] |= comb_ok[:, :wl]
        strokes[i + 1][kr, :wr] |= comb_ok[:, wl:]
    return strokes


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
    passes = []
    for track in tracks:
        pattern, yi, valid = track.window(ink)
        run = np.where(valid, run_t[pattern], 0).astype(np.int16)
        thin_any = (run >= 1) & (run <= h_run + _THIN_MARGIN)
        if thin_any.any():
            delta = int(np.round(np.median(jstar_t[pattern][thin_any])))
            if delta != 0:
                track.shift(delta)
                shift_total += delta
                pattern, yi, valid = track.window(ink)
                run = np.where(valid, run_t[pattern], 0).astype(np.int16)
        passes.append(_Pass(pattern, yi, valid, run))
    # Âncora = cruzamento de verdade: run vertical acima da pauta em
    # >= 2 colunas SEGUIDAS. Um ponto isolado é flutuação de espessura da
    # própria pauta (medido no e5: 14% dos pontos, runs de 6 a 13 px) e,
    # tratado como âncora, protegia 5 colunas e deixava a pauta tracejada.
    thick = [p.run > h_run + _THIN_MARGIN for p in passes]
    gutter_full = _gutter_mask(
        [
            (t.x0, t.x1, raw, p.valid)
            for t, raw, p in zip(tracks, thick, passes, strict=True)
        ],
        w,
    )
    cap = h_run + _THIN_MARGIN + _GUTTER_EXTRA
    matches = (
        [
            _match_tracks(tracks[i], passes[i], tracks[i + 1], passes[i + 1])
            if tracks[i].x1 == tracks[i + 1].x0
            else []
            for i in range(len(tracks) - 1)
        ]
        if gutter_full.any()
        else []
    )
    page = _PageFold(tracks, passes, thick, matches, bool(gutter_full.any()))
    joined_list = _joined_through_bands(page, gutter_full, ink, h_run)
    page.check_joined(joined_list)
    shade_strokes = _stroke_on_shade_tracks(page, gutter_full)
    for i, (track, p, raw) in enumerate(
        zip(page.tracks, page.passes, page.thick, strict=True)
    ):
        jstar = jstar_t[p.pattern].astype(np.int64)
        thin = (p.run >= 1) & (p.run <= h_run + _THIN_MARGIN)
        gutter = gutter_full[track.x0 : track.x1]
        if gutter.any():
            shaded = (p.run >= 1) & (p.run <= cap) & gutter[None, :]
            shaded &= ~joined_list[i]
            shaded &= ~shade_strokes[i]
            thin |= shaded
            raw = raw & ~(gutter[None, :] & (p.run <= cap))
        anchor = np.zeros_like(raw)
        both = raw[:, 1:] & raw[:, :-1]
        anchor[:, 1:] |= both
        anchor[:, :-1] |= both
        protect = _dilate_cols(anchor, _ANCHOR_PAD)
        centers = p.yi + jstar
        cols = (np.arange(track.paths.shape[1]) + track.x0)[None, :]
        peak = flat_prob.take(np.clip(centers, 0, h - 1) * w + cols)
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
        up = up_t[p.pattern[ks, xs]].astype(np.int64)
        length = p.run[ks, xs].astype(np.int64) + 2
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
