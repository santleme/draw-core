"""Modelo persistente da pagina limpa na grade da ancora (PageModel).

A fila de robustez de 2026-11-21 (#129-#133, #130) e um unico defeito
batendo em portas diferentes: cada passada decide so com o quadro da
vez — prob por passada, ganho por passada, mascara de mao por passada
— e uma cascata de guardas remenda cada sintoma (mascara vazia =
pagina toda visivel, ganho p90 sem memoria, sem modelo do que ja
estava impresso, sem faixa de guarda na borda). O mecanismo que falta
e o classico da captura de quadro/documento por camera fixa (He, Liu
& Zhang, MSR 2004, "Real-Time Whiteboard Capture"): uma imagem de
referencia da superficie limpa mantida no tempo, contra a qual cada
pixel e classificado — fundo conhecido, conteudo novo ou ocluido/
pendente.

O PageModel mantem, na grade da ancora (a mesma do acumulador):

- `ref` (float32 HxW): a luminancia de referencia do papel limpo.
  Semeada pixel a pixel quando a passada o observa por
  `confirm_obs` vezes seguidas dentro do seu bloco; rastreada
  lentamente (`rate`) onde a passada afirma papel — absorve a
  deriva de iluminacao (#133) sem nunca afundar um traco (o
  chamador passa `paper` excluindo evidencia de tinta).
- `seeded` (bool HxW): o pixel tem referencia. Regiao nunca vista
  limpa fica pendente — a mao que cobre o caderno na abertura nao
  entra na referencia, e o conteudo revelado depois semeia COM ele
  dentro (a semantica "primeira observacao limpa" do #130: o que
  estava la quando o pixel foi visto pela primeira vez e pagina,
  nao tinta nova).
- `_streak` (uint8 por bloco de `block` px): quantas observacoes
  vistas seguidas o bloco tem. A grade por blocos e a da guarda
  fotometrica medida no #129 (32 px): fina o bastante para uma mao
  nao contaminar a pagina, grossa o bastante para o custo ficar
  abaixo de 1 ms por passada.

Saidas para a trabalhadora: `admissible` (so pixels com referencia
entregam evidencia — regiao pendente congela, o "freeze por falta
de testemunha" que a guarda de papel do #129 aproxima por quadro) e
`novel` (so o que esta mais escuro que a referencia e conteudo novo —
suprime pauta impressa, desenho antigo e sombra de vinco que a regra
do #132 persegue por heuristica). Apagar tinta e simetricamente
seguro: papel re-exposto fica MAIS claro que a referencia, nao mais
escuro, entao remocao nunca le como conteudo novo.

Opt-in via InkConfig.page_model; desligado, nada muda. O limiar de
"novo" (_NOVEL_RATIO, contraste de Weber (ref - I) / ref) e
invariante de escala — vale para mesa clara e cena escura; a
calibracao nas capturas reais esta registrada na constante.
"""

import cv2
import numpy as np

from . import paper_evidence

# Bloco da grade de semeadura: a mesma ordem do 32 px medido na
# guarda fotometrica (#129). Fino o bastante para a mao nao
# contaminar a pagina inteira; grosso o bastante para a fracao por
# bloco custar < 1 ms.
_BLOCK = 32
# Fracao de pixels visiveis que conta o bloco como "visto" na
# passada. Blocos parciais de borda contam pelo que tem.
_SEEN_FRACTION = 0.5
# Observacoes vistas seguidas ate o bloco poder semear. Tres
# passadas a 0,15-0,25 s = 0,45-0,75 s: cabe no periodo de
# estabilizacao que a folha ja exige e cobre uma mao parada breve.
_CONFIRM_OBS = 3
# Passo do rastreio lento da referencia onde ha papel: ~1/rate
# passadas para absorver uma deriva (~20 a 0,15 s = ~3 s) — lento
# demais para apagar jitter de exposicao entre passadas, rapido o
# bastante para acompanhar nuvem e lampada (#133).
_UPDATE_RATE = 0.125
# Quanto mais escuro que a referencia um pixel precisa estar para
# ser "conteudo novo", como RAZAO do proprio nivel da referencia
# (contraste de Weber, (ref - I) / ref) — a literatura canonica testa
# sombra por razao (MOG2 Tau, penumbra; docs/research/
# nota-page-model-literatura.md no repo GUZZBR1/-draw-site-) e so ela
# e invariante entre mesa clara e cena escura. Calibrado nas capturas
# reais de captures/pagina_reorientada_*.png (2026-11-22): conteudo
# impresso/escrito real mede razao ~1,0; marca impressa lendo ~27%
# abaixo da referencia sob dimming (o caso do teste da suíte) fica
# abaixo do limiar e segue suprimida; sombra local suave (~20%)
# tambem. O residuo conhecido: sombra local mais profunda que ~30%
# ainda parece conteudo novo — a mesma classe de furo que o delta
# absoluto tinha, agora invariante de escala. Lapis bem fraco
# (< ~30% de contraste) fica fora: e o preco de nao chamar sombra de
# tinta.
_NOVEL_RATIO = 0.30
# Limiar absoluto do casamento de tom salvo (`saved_tone_evidence`):
# ali a pergunta nao e "isto e conteudo novo contra o papel" mas
# "este tom casa com a tinta salva" — uma distancia entre dois niveis
# escuros, dominio em que a razao de Weber nao se aplica. Todo o resto
# que julga "mais escuro que a aparencia limpa" (novel, appearance,
# returned, darker_than_reference) usa a razao — o mesmo predicado
# (_novel_diff) para nao divergir.
_NOVEL_DELTA = 24.0
# Tolerancia de "voltou a referencia", em luminancia normalizada. Bem
# mais apertada que _NOVEL_DELTA de proposito: histerese de limiar duplo
# (Rosin, BMVC 1995) — acima de NOVEL e conteudo novo, abaixo de RETURN
# e o fundo de volta, e entre os dois e incerto e nao decide nada. Sem
# isso um traco que perdeu contraste (reflexo) pareceria apagado.
_RETURN_TOL = 8.0


def _block_fraction(mask, block, raw=False):
    """Fracao de True por bloco de `block` px, sobre a area REAL do bloco.

    Blocos de borda (grade nao multipla de `block`) contam pelos
    pixels que existem — o padding fora da grade nao dilui a fracao,
    senao a faixa externa nunca seria "vista" nem semeada. Com
    `raw=True` devolve a contagem de True por bloco (a area real e a
    contagem da mascara de uns — o chamador cacheia e divide).
    """
    height, width = mask.shape
    bh = -(-height // block)
    bw = -(-width // block)
    padded = np.zeros((bh * block, bw * block), dtype=bool)
    padded[:height, :width] = mask
    sums = padded.reshape(bh, block, bw, block).sum(axis=(1, 3))
    if raw:
        return sums
    return sums / _block_fraction(np.ones((height, width), dtype=bool), block, raw=True)


def _nearest_reference_gap(ref_inf, ys, xs, values, reach):
    """min |ref_inf[q] - value| sobre q a ate `reach` px (Chebyshev) de
    cada (ys, xs); inf onde nenhuma referencia semeada e alcancavel.

    Varre os vizinhos por indice: os pontos sao poucos perto da grade
    (fora dela, ou inf, nao reproduz nada).
    """
    height, width = ref_inf.shape
    gap_min = np.full(ys.shape, np.inf, dtype=np.float32)
    for oy in range(-reach, reach + 1):
        qy = ys + oy
        in_y = (qy >= 0) & (qy < height)
        qy = np.clip(qy, 0, height - 1)
        for ox in range(-reach, reach + 1):
            qx = xs + ox
            inside = in_y & (qx >= 0) & (qx < width)
            gap = np.abs(ref_inf[qy, np.clip(qx, 0, width - 1)] - values)
            np.minimum(gap_min, np.where(inside, gap, np.inf), out=gap_min)
    return gap_min


def _expand(blocks, shape, block):
    """Mascara por bloco de volta a grade de pixels."""
    height, width = shape
    full = np.repeat(np.repeat(blocks, block, axis=0), block, axis=1)
    return full[:height, :width]


class PageModel:
    """Referencia persistente da pagina limpa na grade da ancora.

    Uma instancia por geracao e por forma de grade: a trabalhadora a
    recria quando a geracao ou o tamanho do mapa muda. A re-ancoragem
    (veredito de pouso, follow) NAO reseta: a grade e a da folha e o
    conteudo anda com ela, exatamente como o estado do acumulador.
    """

    def __init__(
        self,
        shape,
        *,
        block=_BLOCK,
        confirm_obs=_CONFIRM_OBS,
        rate=_UPDATE_RATE,
        novel_ratio=_NOVEL_RATIO,
        novel_delta=_NOVEL_DELTA,
    ):
        if block < 1:
            raise ValueError("block deve ser >= 1")
        if confirm_obs < 1:
            raise ValueError("confirm_obs deve ser >= 1")
        if not 0.0 < rate <= 1.0:
            raise ValueError("rate deve estar em (0, 1]")
        if not 0.0 < novel_ratio < 1.0:
            raise ValueError("novel_ratio deve estar em (0, 1)")
        if not novel_delta > 0.0:
            raise ValueError("novel_delta deve ser > 0")
        self.shape = tuple(shape)
        self.block = int(block)
        self.confirm_obs = int(confirm_obs)
        self.rate = float(rate)
        self.novel_ratio = float(novel_ratio)
        self.novel_delta = float(novel_delta)
        height, width = self.shape
        self._ref = np.zeros((height, width), dtype=np.float32)
        self._norm_ref = np.zeros((height, width), dtype=np.float32)
        self._diff_buf = np.empty((height, width), dtype=np.float32)
        self._blue_ref = np.zeros((height, width), dtype=np.float32)
        self._red_ref = np.zeros((height, width), dtype=np.float32)
        self._has_chroma_ref = False
        self._has_seeded = False
        self._all_seeded = False
        self._last_chroma_ok = True
        self._last_visible = None
        self._last_norm = None
        # Ultima aparencia normalizada VISTA por pixel (NaN = nunca
        # visto): o que a mao cobriu na ultima passada guarda o tom
        # da ultima observacao valida, nao o da mao.
        self._observed_norm = np.full((height, width), np.nan, dtype=np.float32)
        self._observed_shared = False
        self._seeded = np.zeros((height, width), dtype=bool)
        # Referencia com inf fora do semeado: derivada, refeita sob
        # demanda depois de cada observe (que a altera).
        self._ref_inf = None
        self._occluder_hold = np.zeros((height, width), dtype=np.uint8)
        self._hold_mask = np.zeros((height, width), dtype=bool)
        self._recovering_px = 0
        self._seeding_remaining = 0
        self._streak = np.zeros(
            (-(-height // self.block), -(-width // self.block)), dtype=np.uint8
        )
        # Area real de cada bloco (grade nao multipla de `block` tem
        # borda parcial): cacheada — a fracao por bloco a divide todo
        # observe e a constante nao muda na vida do modelo.
        self._block_area = _block_fraction(
            np.ones(self.shape, dtype=bool), self.block, raw=True
        )

    @property
    def reference(self):
        """Luminancia de referencia do papel limpo (float32, grade da ancora)."""
        return self._ref

    @property
    def seeded(self):
        """Pixels que ja tem referencia (bool, grade da ancora)."""
        return self._seeded

    @property
    def seeding_remaining(self):
        """Observacoes restantes para blocos ja expostos em curso de semeadura.

        Considera apenas blocos com 0 < streak < confirm_obs e pixels visiveis
        ainda nao semeados — blocos permanentemente ocultos (streak == 0) nao
        seguram o descanso da trabalhadora.
        """
        return self._seeding_remaining

    @property
    def recovering_px(self):
        """Pixels onde o papel ja reapareceu mas segue retido pelo hold."""
        return self._recovering_px

    @property
    def hold_mask(self):
        """Mascara bool HxW dos pixels retidos exclusivamente pelo hold atual."""
        return self._hold_mask

    @property
    def seeded_fraction(self):
        """Fracao da grade com referencia (telemetria de prontidao)."""
        if self._seeded.size == 0:
            return 0.0
        return float(np.count_nonzero(self._seeded) / self._seeded.size)

    def _reference_inf(self):
        if self._ref_inf is None:
            self._ref_inf = np.where(self._seeded, self._norm_ref, np.float32(np.inf))
        return self._ref_inf

    def observe(self, gray, visible=None, paper=None, bgr=None):
        """Alimenta o modelo com a luminancia desta passada (grade da ancora).

        `gray` e o warp em luminancia na grade da ancora. `visible`
        (None = tudo visivel) e a mascara de observacao da passada —
        o que ela viu de verdade. `paper` e o subconjunto que o
        chamador julga papel limpo (a trabalhadora passa
        `visible & (prob <= piso)`); onde ele e False a referencia
        nao se move — traco nunca afunda no fundo. `paper=None`
        semeia sem rastrear, o modo conservador. `bgr` (opcional,
        uint8 HxWx3) registra a cromaticidade B/R do papel para a
        guarda fotometrica (`exposed`).

        Semear exige `confirm_obs` observacoes vistas seguidas no
        bloco; o pixel semeia com a observacao da vez, nao com media
        — a media borraria o conteudo que o "nao e novo" precisa
        reconhecer depois.
        """
        gray = np.asarray(gray, dtype=np.float32)
        if gray.shape != self.shape:
            raise ValueError(f"gray na grade {gray.shape}, modelo na {self.shape}")
        if visible is None:
            visible = np.ones(self.shape, dtype=bool)
        else:
            visible = np.asarray(visible, dtype=bool)
            if visible.shape != self.shape:
                raise ValueError(
                    f"visible na grade {visible.shape}, modelo na {self.shape}"
                )
        seen = (
            _block_fraction(visible, self.block, raw=True) / self._block_area
            >= _SEEN_FRACTION
        )
        streak = self._streak
        streak[seen] = np.minimum(streak[seen].astype(np.uint16) + 1, 255).astype(
            np.uint8
        )
        streak[~seen] = 0
        seed_now = None
        has_seed_now = False
        if not self._all_seeded:
            ready = _expand(streak >= self.confirm_obs, self.shape, self.block)
            seed_now = ready & visible & ~self._seeded
            has_seed_now = bool(np.any(seed_now))
        gain, chroma_ok = self._estimate_gain(gray, bgr=bgr, visible=visible)
        self._last_chroma_ok = chroma_ok
        self._last_visible = visible
        norm_gray = gray if abs(gain - 1.0) <= 1e-4 else (gray / np.float32(gain))
        self._last_norm = norm_gray
        if visible.all():
            # Sem copia: a passada inteira vista E o ultimo visto.
            self._observed_norm = norm_gray
            self._observed_shared = True
        else:
            if self._observed_shared:
                # Nunca escrever no gray de uma passada anterior.
                self._observed_norm = self._observed_norm.copy()
                self._observed_shared = False
            np.copyto(self._observed_norm, norm_gray, where=visible)
        if has_seed_now:
            self._ref[seed_now] = gray[seed_now]
            self._ref_inf = None
            self._norm_ref[seed_now] = norm_gray[seed_now]
            self._seeded |= seed_now
            self._has_seeded = True
            self._all_seeded = bool(np.all(self._seeded))
        update = None
        has_update = False
        if paper is not None:
            paper = np.asarray(paper, dtype=bool)
            if paper.shape != self.shape:
                raise ValueError(
                    f"paper na grade {paper.shape}, modelo na {self.shape}"
                )
            update = self._seeded & visible & paper
            has_update = bool(np.any(update))
            if has_update:
                diff = self._diff_buf
                rate = np.float32(self.rate)
                np.subtract(gray, self._ref, out=diff, where=update)
                np.multiply(diff, rate, out=diff, where=update)
                np.add(self._ref, diff, out=self._ref, where=update)
                self._ref_inf = None
                np.subtract(norm_gray, self._norm_ref, out=diff, where=update)
                np.multiply(diff, rate, out=diff, where=update)
                np.add(self._norm_ref, diff, out=self._norm_ref, where=update)
        if (
            (has_seed_now or has_update)
            and bgr is not None
            and getattr(bgr, "ndim", 0) == 3
            and bgr.shape[:2] == self.shape
            and bgr.shape[2] >= 3
        ):
            blue = bgr[..., 0].astype(np.float32)
            red = bgr[..., 2].astype(np.float32)
            if has_seed_now:
                self._blue_ref[seed_now] = blue[seed_now]
                self._red_ref[seed_now] = red[seed_now]
                self._has_chroma_ref = True
            if has_update and self._has_chroma_ref:
                self._blue_ref[update] += self.rate * (
                    blue[update] - self._blue_ref[update]
                )
                self._red_ref[update] += self.rate * (
                    red[update] - self._red_ref[update]
                )
        active_streak = (streak > 0) & (streak < self.confirm_obs)
        if not self._all_seeded and np.any(active_streak):
            unseeded_vis_blocks = (
                _block_fraction(~self._seeded & visible, self.block, raw=True) > 0
            )
            active = active_streak & unseeded_vis_blocks
            self._seeding_remaining = (
                int(np.max(self.confirm_obs - streak[active])) if np.any(active) else 0
            )
        else:
            self._seeding_remaining = 0

    def _estimate_gain(
        self,
        gray_f,
        *,
        bgr=None,
        visible=None,
        chroma_tol=paper_evidence._CHROMA_TOL,
    ):
        """Estima (gain, chroma_valid) da iluminacao global contra _norm_ref."""
        if not self._has_seeded:
            return 1.0, True
        stride = paper_evidence._SAMPLE_STRIDE
        seeded_s = self._seeded[::stride, ::stride]
        ref_s = self._norm_ref[::stride, ::stride]
        gray_s = gray_f[::stride, ::stride]
        if visible is None:
            seeded_vis_grid = seeded_s & (ref_s > 1.0)
        else:
            seeded_vis_grid = seeded_s & visible[::stride, ::stride] & (ref_s > 1.0)
        seeded_total = int(np.count_nonzero(seeded_s))
        min_samples = min(
            paper_evidence._MIN_SAMPLES,
            max(8, seeded_total // 2),
        )
        if int(np.count_nonzero(seeded_vis_grid)) < min_samples:
            return 1.0, True
        ref_cut = float(np.percentile(ref_s[seeded_vis_grid], 50))
        paper_keep = seeded_vis_grid & (ref_s >= ref_cut)
        if int(np.count_nonzero(paper_keep)) < max(4, min_samples // 2):
            return 1.0, True
        ratios = gray_s[paper_keep] / ref_s[paper_keep]
        gain_p90 = float(np.percentile(ratios, 90))
        gray_p90 = float(np.percentile(gray_s[paper_keep], 90))
        if not (0.35 <= gain_p90 <= 2.0 and gray_p90 >= 60.0):
            return 1.0, True
        plateau = (ratios >= 0.85 * gain_p90) & (ratios <= 1.15 * gain_p90)
        if int(np.count_nonzero(plateau)) < 0.25 * ratios.size:
            return 1.0, True
        chroma_valid = True
        if (
            self._has_chroma_ref
            and bgr is not None
            and getattr(bgr, "ndim", 0) == 3
            and bgr.shape[:2] == self.shape
            and bgr.shape[2] >= 3
        ):
            bgr_s = bgr[::stride, ::stride]
            b_s = bgr_s[..., 0][paper_keep][plateau].astype(np.float32)
            r_s = bgr_s[..., 2][paper_keep][plateau].astype(np.float32)
            b_ref_s = self._blue_ref[::stride, ::stride][paper_keep][plateau]
            r_ref_s = self._red_ref[::stride, ::stride][paper_keep][plateau]
            r_mean = float(r_s.mean())
            r_ref_mean = float(r_ref_s.mean())
            if r_ref_mean > 1.0:
                ref_ratio = float(b_ref_s.mean()) / r_ref_mean
                spread_g = chroma_tol * max(ref_ratio, 1e-3) + 1e-6
                cur_ratio = float(b_s.mean()) / r_mean if r_mean > 1.0 else 0.0
                chroma_valid = (r_mean > 1.0) and (
                    abs(cur_ratio - ref_ratio) <= spread_g
                )
        if not chroma_valid:
            return 1.0, False
        return gain_p90, True

    def exposed(
        self,
        gray,
        bgr=None,
        *,
        visible=None,
        min_ratio=paper_evidence._MIN_RATIO,
        chroma_tol=paper_evidence._CHROMA_TOL,
        hold_passes=0,
    ):
        """Mascara bool HxW: True onde a regiao semeada parece papel exposto.

        Impede que maos ou sombras perdidas pelo detector alcancem o voto ou
        corrompam `self._ref`: tracos de caneta preservam papel adjacente
        na janela deslizante de `self.block` px, enquanto maos/bracos/sombras
        caem abaixo de `min_ratio * ref_peak` ou saem da razao B/R do papel.
        """
        gray_f = np.asarray(gray, dtype=np.float32)
        if gray_f.shape != self.shape:
            raise ValueError(f"gray na grade {gray_f.shape}, modelo na {self.shape}")
        input_vis = (
            np.ones(self.shape, dtype=bool)
            if visible is None
            else np.asarray(visible, dtype=bool)
        )
        if input_vis.shape != self.shape:
            raise ValueError(
                f"visible na grade {input_vis.shape}, modelo na {self.shape}"
            )
        if not self._has_seeded:
            self._recovering_px = 0
            self._hold_mask.fill(False)
            return np.ones(self.shape, dtype=bool)
        height, width = self.shape
        step = paper_evidence._REDUCE_PX
        cells = max(1, self.block // step)
        kernel = np.ones((cells, cells), np.uint8)
        ref_grid = np.where(
            self._seeded[::step, ::step], self._norm_ref[::step, ::step], 0.0
        )
        gray_grid = gray_f[::step, ::step]
        ref_peak = cv2.dilate(ref_grid, kernel)
        gray_peak = cv2.dilate(gray_grid, kernel)
        has_ref = ref_peak > 1.0
        gain, chroma_ok = self._estimate_gain(
            gray_f, bgr=bgr, visible=input_vis, chroma_tol=chroma_tol
        )
        self._last_chroma_ok = chroma_ok
        eff_gain = min(gain, 1.0)
        ok = ~has_ref | (gray_peak >= (min_ratio * eff_gain) * ref_peak)
        if (
            self._has_chroma_ref
            and bgr is not None
            and getattr(bgr, "ndim", 0) == 3
            and bgr.shape[:2] == self.shape
            and bgr.shape[2] >= 3
        ):
            peak_bgr = cv2.dilate(bgr[::step, ::step], kernel)
            blue_f = peak_bgr[..., 0].astype(np.float32)
            red_f = peak_bgr[..., 2].astype(np.float32)
            b_ref_peak = cv2.dilate(
                np.where(self._seeded, self._blue_ref, 0.0)[::step, ::step], kernel
            )
            r_ref_peak = cv2.dilate(
                np.where(self._seeded, self._red_ref, 0.0)[::step, ::step], kernel
            )
            paper_ratio = np.where(
                r_ref_peak > 1.0, b_ref_peak / np.maximum(r_ref_peak, 1.0), 1.0
            )
            spread = chroma_tol * np.maximum(paper_ratio, 1e-3) + 1e-6
            chroma_ok = (
                (red_f > 1.0)
                & (blue_f >= (paper_ratio - spread) * red_f)
                & (blue_f <= (paper_ratio + spread) * red_f)
            )
            ok &= ~has_ref | (r_ref_peak <= 1.0) | chroma_ok
        if visible is not None:
            vis_peak = cv2.dilate(input_vis.astype(np.uint8)[::step, ::step], kernel)
            ok |= vis_peak == 0
        half = max(1, cells // 2) + 1
        ok = cv2.erode(
            ok.astype(np.uint8),
            np.ones((2 * half + 1, 2 * half + 1), np.uint8),
            borderValue=1,
        )
        grown = cv2.resize(ok, None, fx=step, fy=step, interpolation=cv2.INTER_NEAREST)
        raw_exposed = grown[:height, :width].astype(bool)
        hold = self._occluder_hold
        if hold_passes > 0:
            trigger = self._seeded & input_vis & ~raw_exposed
            if np.any(trigger):
                hold[trigger] = hold_passes
            release = self._seeded & input_vis & raw_exposed & (hold > 0)
            if np.any(release):
                hold[release] -= 1
            self._recovering_px = int(np.count_nonzero(release))
            self._hold_mask = release
            return ~self._seeded | (raw_exposed & ~release)
        if np.any(hold):
            hold.fill(0)
        self._recovering_px = 0
        self._hold_mask.fill(False)
        return ~self._seeded | raw_exposed

    def admissible(self, visible=None):
        """Pixels que podem entregar evidencia: os que tem referencia.

        Com `visible`, devolve `seeded & visible` — a mascara pronta
        para o voto. Pixel pendente (nunca visto limpo) nao entrega:
        e a extensao temporal do "quem nao ve, se abstem" da
        oclusao, da passada para a vida inteira da geracao.
        """
        if visible is None:
            return self._seeded.copy()
        return self._seeded & np.asarray(visible, dtype=bool)

    def normalize(self, gray, visible=None):
        """`gray` no dominio normalizado pela iluminacao da passada.

        O ganho e estimado na folha inteira (o custo dominante): quem
        compara a mesma passada varias vezes normaliza uma vez e passa
        `prenormalized=True` aos metodos de comparacao.
        """
        return self._normalized(gray, visible)

    def _normalized(self, gray, visible=None):
        gray = np.asarray(gray, dtype=np.float32)
        if gray.shape != self.shape:
            raise ValueError(f"gray na grade {gray.shape}, modelo na {self.shape}")
        vis = self._last_visible if visible is None else visible
        gain = (
            self._estimate_gain(gray, visible=vis)[0] if self._last_chroma_ok else 1.0
        )
        return gray if abs(gain - 1.0) <= 1e-4 else (gray / np.float32(gain))

    def _novel_diff(self, diff, ref):
        """O predicado de "conteudo novo": `diff > novel_ratio * ref`.

        `diff` e `ref - gray` no dominio normalizado — positivo onde a
        observacao esta mais escura que a referencia. `ref` pode ser a
        referencia semeada (novel/appearance/returned) ou a aparencia
        limpa propagada (darker_than_reference); inf nunca e novo.
        Unico lugar onde a razao de Weber vira comparacao — os quatro
        caminhos (voto, remocao, retorno, identidade) classificam pelo
        mesmo criterio.
        """
        return diff > self.novel_ratio * np.maximum(ref, 1.0)

    def novel(self, gray, visible=None):
        """Pixels mais escuros que a referencia alem de `novel_ratio`
        vezes a propria referencia (contraste de Weber).

        Compara `gray` e a referencia no mesmo dominio normalizado pela
        iluminacao do papel exposto: escurecimento global da folha nao
        transforma marca impressa ja semeada em tinta nova.
        """
        norm_gray = self._normalized(gray, visible)
        if not self._has_seeded:
            return np.zeros(self.shape, dtype=bool)
        return self._seeded & self._novel_diff(
            self._norm_ref - norm_gray, self._norm_ref
        )

    def appearance(self, gray, visible=None):
        """(novel, returned) numa unica normalizacao de iluminacao.

        A estimativa de ganho e o custo dominante de ambos; o caminho
        quente da trabalhadora precisa dos dois por passada. As
        mascaras sao DISJUNTAS por construcao: `returned` exclui quem
        ainda passa o teste de novidade — sem isso um traco sobre
        pauta escura (razao > _NOVEL_RATIO com |diff| <= _RETURN_TOL)
        seria lido como tinta nova e apagado ao mesmo tempo.
        """
        norm_gray = self._normalized(gray, visible)
        if not self._has_seeded:
            none = np.zeros(self.shape, dtype=bool)
            return none, none.copy()
        diff = self._norm_ref - norm_gray
        novel_mask = self._novel_diff(diff, self._norm_ref)
        return (
            self._seeded & novel_mask,
            self._seeded & (np.abs(diff) <= _RETURN_TOL) & ~novel_mask,
        )

    def returned(self, gray, visible=None):
        """Pixels de volta a aparencia semeada, dentro de `_RETURN_TOL`.

        E o outro lado da histerese de `novel`: so quem casa com a
        referencia de perto e evidencia de ausencia. Pixel entre as duas
        tolerancias (traco com contraste reduzido) nao e nem novo nem
        de volta. Quem ainda passa o teste de novidade nunca conta como
        devolvido — sobre pauta escura as duas faixas se tocariam.
        """
        norm_gray = self._normalized(gray, visible)
        if not self._has_seeded:
            return np.zeros(self.shape, dtype=bool)
        diff = self._norm_ref - norm_gray
        return (
            self._seeded
            & (np.abs(diff) <= _RETURN_TOL)
            & ~self._novel_diff(diff, self._norm_ref)
        )

    def darker_than_reference(
        self,
        gray,
        *,
        dy=0,
        dx=0,
        near=0,
        visible=None,
        region=None,
        prenormalized=False,
        with_expected=True,
    ):
        """Pixels mais escuros que a aparencia limpa ALINHADA, por `_novel_diff`.

        Usa o mesmo contraste de Weber de `novel()` — a identidade
        nao pode chamar de "indistinguivel da pauta" um traco que o
        voto acabou de admitir como conteudo novo (review #174).

        Compara `gray` (grade da ancora) a `_norm_ref` na posicao
        deslocada de (dy, dx) — a mesma translacao que a pauta mede —
        no dominio normalizado pela iluminacao da passada, entao um
        escurecimento global nao vira marca nova. Cada pixel julga
        contra a pior aparencia limpa que poderia ocupar o lugar: a
        referencia semeada mais escura a ate `near` px — a variacao
        de borda da marca impressa deslocada. Pixel sem referencia
        alcancavel nao prova nada (False).

        `region`=(y0, y1, x0, x1) restringe a computacao pesada: a
        margem necessaria (near + |deslocamento|) e acrescentada por
        dentro, e posicoes fora dela voltam darker=False, expected=inf.
        Em folha A4 a comparacao full-frame dominava a avaliacao de
        identidade.

        Devolve (darker, expected): `expected` e a aparencia limpa
        propagada usada no julgamento (float32; inf onde nenhuma
        referencia semeada alcanca). A comparacao com o nivel de
        cinza da tinta salva (`expected - level` dentro de
        `novel_delta`) e o que separa "indistinguivel" de
        "aparencia de pauta provando ausencia". `with_expected=False`
        devolve expected=None (evita a grade float inteira);
        `prenormalized=True` recebe `gray` ja vindo de `normalize`."""
        # O ganho de iluminacao e global — medido na folha inteira,
        # nao no recorte.
        norm_gray = (
            np.asarray(gray, dtype=np.float32)
            if prenormalized
            else self._normalized(gray, visible)
        )
        height, width = self.shape
        if not self._has_seeded:
            return (
                np.zeros(self.shape, dtype=bool),
                np.full(self.shape, np.inf, dtype=np.float32)
                if with_expected
                else None,
            )
        if region is None:
            cy0, cy1, cx0, cx1 = 0, height, 0, width
        else:
            pad = near + max(abs(dy), abs(dx))
            cy0 = max(0, int(region[0]) - pad)
            cy1 = min(height, int(region[1]) + pad)
            cx0 = max(0, int(region[2]) - pad)
            cx1 = min(width, int(region[3]) + pad)
        ref = self._reference_inf()[cy0:cy1, cx0:cx1]
        if dy or dx:
            ch, cw = ref.shape
            shifted = np.full(ref.shape, np.inf, dtype=np.float32)
            src_y0, src_y1 = max(0, -dy), min(ch, ch - dy)
            src_x0, src_x1 = max(0, -dx), min(cw, cw - dx)
            if src_y1 > src_y0 and src_x1 > src_x0:
                shifted[src_y0 + dy : src_y1 + dy, src_x0 + dx : src_x1 + dx] = ref[
                    src_y0:src_y1, src_x0:src_x1
                ]
            ref = shifted
        if near > 0:
            ref = cv2.erode(ref, np.ones((2 * near + 1, 2 * near + 1), dtype=np.uint8))
        darker = np.zeros(self.shape, dtype=bool)
        darker[cy0:cy1, cx0:cx1] = np.isfinite(ref) & self._novel_diff(
            ref - norm_gray[cy0:cy1, cx0:cx1], ref
        )
        if not with_expected:
            return darker, None
        expected = np.full(self.shape, np.inf, dtype=np.float32)
        expected[cy0:cy1, cx0:cx1] = ref
        return darker, expected

    def saved_tone_evidence(
        self,
        gray,
        levels,
        at,
        *,
        dy=0,
        dx=0,
        near=0,
        visible=None,
        region=None,
        prenormalized=False,
        with_unsure=False,
        sampled=None,
    ):
        """Evidencia fotometrica da tinta salva nas posicoes `at`.

        `levels` (float32 HxW, NaN = desconhecido) e o tom normalizado
        da tinta salva; (dy, dx) a translacao medida da folha. Devolve
        (distinct, matched_obs), bool HxW:

        - distinct: posicao de `at` cujo tom salvo difere alem de
          `novel_delta` de TODA aparencia limpa semeada a ate
          near + |deslocamento| px — nenhuma copia limpa, nem com a
          pauta deslocada, o reproduz. Lapis de 70 colado a pauta de
          40 e distinto (a pior aparencia vizinha nao o imita); traco
          de 40 colado a pauta de 40 nao. Tom desconhecido ou nenhuma
          referencia alcancavel: False. Tambem False onde o unico veto
          ao match vem de pauta do quadro medido cuja amostra esta
          escondida — nem presenca nem ausencia provada (ambiguo).
        - matched_obs: posicoes observadas (a salva ou a deslocada, na
          grade atual) de pixel distinto onde o tom salvo reapareceu
          dentro de `novel_delta` — presenca provada pela aparencia.

        `region`=(y0, y1, x0, x1) e um bbox que contem `at` (evita a
        varredura da grade inteira); `prenormalized` como em
        `darker_than_reference`. `with_unsure` acrescenta dois masks:
        unsure, as posicoes de `at` tiradas de `distinct` so pela
        amostra escondida — nao julgaveis nesta passada — e occluded,
        o subconjunto cujo veto depende de amostra escondida DENTRO de
        `sampled` (a pegada do registro; None = grade inteira): mao
        sobre a folha, nao recorte geometrico.
        """
        height, width = self.shape
        distinct = np.zeros(self.shape, dtype=bool)
        matched_obs = np.zeros(self.shape, dtype=bool)
        unsure_saved = np.zeros(self.shape, dtype=bool)
        occluded_saved = np.zeros(self.shape, dtype=bool)

        def result():
            if with_unsure:
                return distinct, matched_obs, unsure_saved, occluded_saved
            return distinct, matched_obs

        if not self._has_seeded or levels is None or levels.shape != self.shape:
            return result()
        y0, y1, x0, x1 = (0, height, 0, width) if region is None else region
        ys, xs = np.nonzero(np.asarray(at, dtype=bool)[y0:y1, x0:x1])
        ys = ys + y0
        xs = xs + x0
        lv = levels[ys, xs]
        known = np.isfinite(lv)
        ys, xs, lv = ys[known], xs[known], lv[known]
        if ys.size == 0:
            return result()
        reach = int(near) + max(abs(int(dy)), abs(int(dx)))
        ref_inf = self._reference_inf()
        min_gap = _nearest_reference_gap(ref_inf, ys, xs, lv, reach)
        keep = np.isfinite(min_gap) & (min_gap > self.novel_delta)
        ys, xs, lv = ys[keep], xs[keep], lv[keep]
        if ys.size == 0:
            return result()
        distinct[ys, xs] = True
        norm_gray = (
            np.asarray(gray, dtype=np.float32)
            if prenormalized
            else self._normalized(gray, visible)
        )
        vis = None if visible is None else np.asarray(visible, dtype=bool)
        foot = None if sampled is None else np.asarray(sampled, dtype=bool)
        matched_saved = np.zeros(self.shape, dtype=bool)
        for oy, ox in {(0, 0), (int(dy), int(dx))}:
            qy, qx = ys + oy, xs + ox
            ok = (qy >= 0) & (qy < height) & (qx >= 0) & (qx < width)
            if vis is not None:
                ok[ok] = vis[qy[ok], qx[ok]]
            py, px = ys[ok], xs[ok]
            qy, qx, lq = qy[ok], qx[ok], lv[ok]
            obs = norm_gray[qy, qx]
            saved_gap = np.abs(obs - lq)
            # A observacao tem de ser melhor explicada pela tinta salva
            # que pela aparencia limpa: pauta de uma copia limpa um
            # pouco mais escura (85 vs 95) cai dentro do delta da
            # escrita salva (65), mas segue mais perto da pauta. A
            # referencia mora na grade salva e cada quadro so alcanca
            # `near`: o do candidato (q volta para a posicao salva) e
            # o da folha medida (q - (dy, dx)) — o traco no lugar
            # salvo nao prova que a pauta tambem nao andou. O
            # deslocamento e global: a pauta vizinha que provadamente
            # ficou no lugar sai do quadro medido e nao veta o traco
            # que tambem ficou.
            near_i = int(near)
            clean_gap = _nearest_reference_gap(ref_inf, py, px, obs, near_i)
            unsure = occluded = None
            if (oy, ox) != (int(dy), int(dx)) and py.size:
                moved_ref, known_ref, occl_ref, my0, mx0 = self._moved_reference(
                    ref_inf,
                    norm_gray,
                    vis,
                    foot,
                    py - int(dy),
                    px - int(dx),
                    near_i,
                    int(dy),
                    int(dx),
                )
                ry, rx = py - int(dy) - my0, px - int(dx) - mx0
                if known_ref is not None:
                    known_gap = np.minimum(
                        clean_gap,
                        _nearest_reference_gap(known_ref, ry, rx, obs, near_i),
                    )
                    occl_gap = np.minimum(
                        clean_gap,
                        _nearest_reference_gap(occl_ref, ry, rx, obs, near_i),
                    )
                np.minimum(
                    clean_gap,
                    _nearest_reference_gap(moved_ref, ry, rx, obs, near_i),
                    out=clean_gap,
                )
                if known_ref is not None:
                    fits = saved_gap <= self.novel_delta
                    unsure = fits & (saved_gap < known_gap)
                    occluded = fits & (saved_gap < occl_gap)
            hit = (saved_gap <= self.novel_delta) & (saved_gap < clean_gap)
            matched_obs[qy[hit], qx[hit]] = True
            matched_saved[py[hit], px[hit]] = True
            if unsure is not None:
                # Veto so por referencia cuja amostra do quadro medido
                # esta escondida (ou fora da grade): nem presenca nem
                # ausencia provada — o pixel volta a ser ambiguo.
                unsure &= ~hit
                unsure_saved[py[unsure], px[unsure]] = True
                occluded &= unsure
                occluded_saved[py[occluded], px[occluded]] = True
        unsure_saved &= ~matched_saved
        occluded_saved &= unsure_saved
        distinct &= ~unsure_saved
        return result()

    @staticmethod
    def _moved_reference(ref_inf, norm_gray, vis, foot, cy, cx, reach, dy, dx):
        """Recortes de `ref_inf` cobrindo (cy, cx) +- reach para o quadro
        medido (dy, dx). Devolve (moved, known, occluded, y0, x0):

        - moved: inf so onde a aparencia limpa provadamente FICOU no
          lugar — a observacao em r reproduz ref[r] mais de
          `_RETURN_TOL` melhor que a em r + (dy, dx), ambas visiveis.
          Amostra que nao distingue as duas posicoes (faixa larga
          cobrindo ambas, oclusao, borda) mantem a referencia.
        - known: `moved` tambem com inf onde alguma das duas amostras
          esta escondida ou fora da grade — so a referencia julgada
          pela aparencia visivel.
        - occluded: `moved` com inf so onde a amostra escondida esta
          dentro da grade e de `foot` (mao, nao recorte do registro):
          o veto que sobra nele independe da oclusao.
        """
        height, width = ref_inf.shape
        y0 = max(int(cy.min()) - reach, 0)
        y1 = min(int(cy.max()) + reach + 1, height)
        x0 = max(int(cx.min()) - reach, 0)
        x1 = min(int(cx.max()) + reach + 1, width)
        moved = ref_inf[y0:y1, x0:x1].copy()
        ty0, ty1 = max(y0, -dy), min(y1, height - dy)
        tx0, tx1 = max(x0, -dx), min(x1, width - dx)
        if ty1 <= ty0 or tx1 <= tx0:
            known = np.full(moved.shape, np.inf, dtype=np.float32)
            return moved, known, moved.copy(), y0, x0
        full = (ty0, ty1, tx0, tx1) == (y0, y1, x0, x1)
        ref = ref_inf[ty0:ty1, tx0:tx1]
        here = norm_gray[ty0:ty1, tx0:tx1]
        there = norm_gray[ty0 + dy : ty1 + dy, tx0 + dx : tx1 + dx]
        stayed = np.abs(here - ref) + _RETURN_TOL < np.abs(there - ref)
        judged = np.ones(ref.shape, dtype=bool)
        if vis is not None:
            judged = (
                vis[ty0:ty1, tx0:tx1] & vis[ty0 + dy : ty1 + dy, tx0 + dx : tx1 + dx]
            )
            stayed &= judged
        win = (slice(ty0 - y0, ty1 - y0), slice(tx0 - x0, tx1 - x0))
        moved[win][stayed] = np.inf
        if full and (vis is None or judged.all()):
            # Toda amostra visivel e na grade: nenhum veto depende de
            # oclusao — sem os recortes extras.
            return moved, None, None, y0, x0
        known = np.full(moved.shape, np.inf, dtype=np.float32)
        occluded = ref_inf[y0:y1, x0:x1].copy()
        known[win] = np.where(judged & ~stayed, ref, np.inf)
        hidden = ~judged
        if foot is not None:
            hidden &= (
                foot[ty0:ty1, tx0:tx1] & foot[ty0 + dy : ty1 + dy, tx0 + dx : tx1 + dx]
            )
        occluded[win] = np.where(stayed | hidden, np.inf, ref)
        return moved, known, occluded, y0, x0

    @property
    def observed_normalized_gray(self):
        """Ultimo tom normalizado VISTO por pixel (float32, NaN = nunca visto).

        A referencia fotometrica da tinta salva: diferente de
        `last_normalized_gray`, o pixel coberto pela mao (ou recusado
        pela guarda) na ultima passada guarda a ultima observacao
        valida, nao o tom do oclusor.
        """
        return self._observed_norm

    @property
    def last_normalized_gray(self):
        """O gray normalizado pela iluminacao da ultima observacao.

        None antes do primeiro observe — a referencia fotometrica
        que a maquina de identidade guarda junto da mascara de
        tinta salva.
        """
        return self._last_norm

    def seeding_pending(self, visible=None):
        """Visivel sem referencia em bloco com semeadura em curso.

        So blocos que ja contam observacoes (0 < streak < confirm_obs):
        bloco com menos de `_SEEN_FRACTION` de papel visivel nunca conta
        e nunca semeia — esperar por ele seria esperar para sempre. E o
        que a prontidao precisa: "falta pouco", nao "falta tudo".
        """
        if visible is None:
            visible = np.ones(self.shape, dtype=bool)
        active = _expand(
            (self._streak > 0) & (self._streak < self.confirm_obs),
            self.shape,
            self.block,
        )
        return ~self._seeded & np.asarray(visible, dtype=bool) & active

    def pending(self, visible=None):
        """Pixels visiveis ainda sem referencia (diagnostico/telemetria)."""
        if visible is None:
            visible = np.ones(self.shape, dtype=bool)
        return ~self._seeded & np.asarray(visible, dtype=bool)
