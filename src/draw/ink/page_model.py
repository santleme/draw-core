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

import numpy as np

# Amostragem do estimador de ganho global: passo da grade e minimo
# de pixels semeados com referencia para a razao mediana ser confiavel.
_SAMPLE_STRIDE = 8
_MIN_SAMPLES = 64
# Desvio maximo da cromaticidade B/R para a leitura ser papel.
_CHROMA_TOL = 0.18

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
# Razao de clareamento que faz a referencia subir direto ate a
# observacao em vez de derivar: papel a 222 semeado sob mao a 130
# mede 1,7 — muito acima de 1,25; relampagos e nuvens reais ficam
# na deriva lenta.
_REF_UP_RATIO = 1.25
# Passo do rastreio lento da referencia onde ha papel: ~1/rate
# passadas para absorver uma deriva (~20 a 0,15 s = ~3 s) — lento
# demais para apagar jitter de exposicao entre passadas, rapido o
# bastante para acompanhar nuvem e lampada (#133).
_UPDATE_RATE = 0.125
# Quanto mais escuro que a referencia um pixel precisa estar para
# ser "conteudo novo", como RAZAO do proprio nivel da referencia
# (contraste de Weber, (ref - I) / ref) — a literatura canonica testa
# sombra por razao (MOG2 Tau, penumbra; docs/research/
# nota-page-model-literatura.md) e so ela e invariante entre mesa
# clara e cena escura. Calibrado nas capturas reais de
# captures/pagina_reorientada_*.png (2026-11-22): conteudo impresso/
# escrito real mede razao ~1,0; marca impressa lendo ~27% abaixo da
# referencia sob dimming (o caso do teste da suíte) fica abaixo do
# limiar e segue suprimida; sombra local suave (~20%) tambem. O
# residuo conhecido: sombra local mais profunda que ~30% ainda parece
# conteudo novo — a mesma classe de furo que o delta absoluto tinha,
# agora invariante de escala. Lapis bem fraco (< ~30% de contraste)
# fica fora: e o preco de nao chamar sombra de tinta.
_NOVEL_RATIO = 0.30


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
        novel_ratio=_NOVEL_RATIO,
    ):
        if block < 1:
            raise ValueError("block deve ser >= 1")
        if confirm_obs < 1:
            raise ValueError("confirm_obs deve ser >= 1")
        if not 0.0 < rate <= 1.0:
            raise ValueError("rate deve estar em (0, 1]")
        if not 0.0 < novel_ratio < 1.0:
            raise ValueError("novel_ratio deve estar em (0, 1)")
        self.shape = tuple(shape)
        self.block = int(block)
        self.confirm_obs = int(confirm_obs)
        self.rate = float(rate)
        self.novel_ratio = float(novel_ratio)
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
        uint8 HxWx3) registra a cromaticidade B/R do papel — leitura
        fora da razao do papel nao participa do ganho global.

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
                # Observacao muito mais clara que a referencia e um
                # oclusor que saiu (mao nao mascarada semeada como
                # papel, sombra que abriu): a deriva lenta levaria
                # dezenas de passadas para reexpor o papel — a
                # referencia sobe direto ate a observacao.
                brighter = update & (norm_gray > self._norm_ref * _REF_UP_RATIO)
                np.subtract(gray, self._ref, out=diff, where=update)
                np.multiply(diff, rate, out=diff, where=update)
                np.add(self._ref, diff, out=self._ref, where=update)
                np.subtract(norm_gray, self._norm_ref, out=diff, where=update)
                np.multiply(diff, rate, out=diff, where=update)
                np.add(self._norm_ref, diff, out=self._norm_ref, where=update)
                if np.any(brighter):
                    self._ref[brighter] = gray[brighter]
                    self._norm_ref[brighter] = norm_gray[brighter]
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
        chroma_tol=_CHROMA_TOL,
    ):
        """Estima (gain, chroma_valid) da iluminacao global contra _norm_ref."""
        if not self._has_seeded:
            return 1.0, True
        stride = _SAMPLE_STRIDE
        seeded_s = self._seeded[::stride, ::stride]
        ref_s = self._norm_ref[::stride, ::stride]
        gray_s = gray_f[::stride, ::stride]
        if visible is None:
            seeded_vis_grid = seeded_s & (ref_s > 1.0)
        else:
            seeded_vis_grid = seeded_s & visible[::stride, ::stride] & (ref_s > 1.0)
        seeded_total = int(np.count_nonzero(seeded_s))
        min_samples = min(
            _MIN_SAMPLES,
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
        """Pixels mais escuros que a referencia alem de `novel_ratio`
        vezes a propria referencia (contraste de Weber).

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
        return self._seeded & (
            self._norm_ref - norm_gray
            > self.novel_ratio * np.maximum(self._norm_ref, 1.0)
        )

    def pending(self, visible=None):
        """Pixels visiveis ainda sem referencia (diagnostico/telemetria)."""
        if visible is None:
            visible = np.ones(self.shape, dtype=bool)
        return ~self._seeded & np.asarray(visible, dtype=bool)
