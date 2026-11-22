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
"novo" (_NOVEL_DELTA, em unidades de luminancia 0-255) e uma
semente razoavel para um traco escuro sobre papel claro; o corpus
de clipes reais e quem calibra de vez.
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
# ser "conteudo novo", em unidades de luminancia. Semente para
# traco escuro sobre papel claro; a calibracao real sai do corpus.
# A literatura canonica testa sombra por RAZAO (MOG2 Tau, penumbra e o
# caso dificil — docs/research/nota-page-model-literatura.md no repo
# GUZZBR1/-draw-site-); o delta
# absoluto falha justamente no gradiente de sombra movel do #133. E o
# erro mais seguro p/ prototipo: falso `novel` cai na pipeline normal
# (status quo), falso `~novel` suprime traco claro para sempre — mas a
# decisao delta vs razao vs entrada `division_normalize` e medicao,
# nao chute: pendente de clipes reais.
_NOVEL_DELTA = 24.0


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
        novel_delta=_NOVEL_DELTA,
    ):
        if block < 1:
            raise ValueError("block deve ser >= 1")
        if confirm_obs < 1:
            raise ValueError("confirm_obs deve ser >= 1")
        if not 0.0 < rate <= 1.0:
            raise ValueError("rate deve estar em (0, 1]")
        if not novel_delta > 0.0:
            raise ValueError("novel_delta deve ser > 0")
        self.shape = tuple(shape)
        self.block = int(block)
        self.confirm_obs = int(confirm_obs)
        self.rate = float(rate)
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
        self._seeded = np.zeros((height, width), dtype=bool)
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
        if has_seed_now:
            self._ref[seed_now] = gray[seed_now]
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

    def novel(self, gray, visible=None):
        """Pixels mais escuros que a referencia alem de `novel_delta`.

        Compara `gray` e a referencia no mesmo dominio normalizado pela
        iluminacao do papel exposto: escurecimento global da folha nao
        transforma marca impressa ja semeada em tinta nova.
        """
        gray = np.asarray(gray, dtype=np.float32)
        if gray.shape != self.shape:
            raise ValueError(f"gray na grade {gray.shape}, modelo na {self.shape}")
        if not self._has_seeded:
            return np.zeros(self.shape, dtype=bool)
        vis = self._last_visible if visible is None else visible
        gain = (
            self._estimate_gain(gray, visible=vis)[0] if self._last_chroma_ok else 1.0
        )
        norm_gray = gray if abs(gain - 1.0) <= 1e-4 else (gray / np.float32(gain))
        return self._seeded & (self._norm_ref - norm_gray > self.novel_delta)

    def pending(self, visible=None):
        """Pixels visiveis ainda sem referencia (diagnostico/telemetria)."""
        if visible is None:
            visible = np.ones(self.shape, dtype=bool)
        return ~self._seeded & np.asarray(visible, dtype=bool)
