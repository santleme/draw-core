"""Preset de producao do motor de tinta (o que as flags Tier A/B liam do ambiente)."""

import math
from dataclasses import dataclass

# Nivel de voto que reproduz o legado: a media das 3 passadas que fecha a
# soma strong de 1,9. strict_vote desloca os campos a partir dele.
LEGACY_VOTE_LEVEL = 1.9 / 3

# Preset de producao (PipelineConfig): voto estrito e cadencia medidos nos
# PRs #53 (voto 0,85, com ganho) e #68 (0,15 s com o page_clean INT8).
PRODUCTION_VOTE_LEVEL = 0.85
PRODUCTION_PASS_INTERVAL_S = 0.15

# Nível de voto do perfil técnico/CAD (medido na bancada do strict_vote:
# 0,92 com ganho -> escrita 109%, pauta 48% do legado sem ganho — traco
# mais fino ainda, aceitável e até desejado em linhas de CAD).
TECHNICAL_VOTE_LEVEL = 0.92


def resolve_border_guard_px(shape, configured, *, min_auto_side=0):
    """Largura (px) da faixa de guarda na borda externa da grade da pagina.

    Quando `configured` e None, deriva ~0,8% do lado menor (minimo 6 px)
    para isolar jitter de 1-3 px do quad contra fundo escuro sem cortar
    tracos da area util; `min_auto_side` desliga o automatico em grades
    minusculas de testes unitarios.
    """
    h, w = shape[:2]
    short = min(h, w)
    limit = max(0, (short - 1) // 2)
    if limit == 0:
        return 0
    if configured is not None:
        return min(int(configured), limit)
    if short < min_auto_side:
        return 0
    return min(max(6, int(round(0.008 * short))), limit)


def resolve_border_guard_yx(
    shape,
    configured,
    corners=None,
    *,
    min_auto_side=0,
    camera_jitter_px=3.5,
):
    """(guard_y, guard_x) em px da grade retificada.

    No modo automatico (`configured is None`), quando `corners` (4x2 em
    espaco de camera) e informado, projeta `camera_jitter_px` pela
    escala de cada eixo para cobrir o estiramento de folhas pequenas no
    quadro da camera. Larguras explicitas seguem em px retificados.
    """
    h, w = shape[:2]
    limit_y = max(0, (h - 1) // 2)
    limit_x = max(0, (w - 1) // 2)
    if configured is not None:
        val = int(configured)
        return min(val, limit_y), min(val, limit_x)
    base = resolve_border_guard_px(shape, None, min_auto_side=min_auto_side)
    if base == 0 or corners is None:
        return min(base, limit_y), min(base, limit_x)
    gy = gx = base
    try:
        c0, c1, c2, c3 = corners[0], corners[1], corners[2], corners[3]
        top_w = math.hypot(float(c1[0] - c0[0]), float(c1[1] - c0[1]))
        bot_w = math.hypot(float(c2[0] - c3[0]), float(c2[1] - c3[1]))
        left_h = math.hypot(float(c3[0] - c0[0]), float(c3[1] - c0[1]))
        right_h = math.hypot(float(c2[0] - c1[0]), float(c2[1] - c1[1]))
        min_w = min(top_w, bot_w)
        min_h = min(left_h, right_h)
        if min_h > 0.0:
            gy = max(gy, math.ceil(camera_jitter_px * h / min_h))
        if min_w > 0.0:
            gx = max(gx, math.ceil(camera_jitter_px * w / min_w))
    except (IndexError, TypeError, ValueError):
        pass
    return min(gy, limit_y), min(gx, limit_x)


@dataclass(frozen=True, slots=True)
class InkConfig:
    """Comportamento do motor de tinta, injetado por construtor.

    O default e o preset de producao validado no legado: os
    mecanismos do Tier A ligados (remocao de tinta, arbitragem e
    follow de ancora, evento geometrico de follow_move) e os do
    Tier B desligados (banda morta e ancora-mediana do registro).
    Os recursos dos Tiers C/D nao viram campos: foram removidos do
    codigo junto com as flags que os ligavam.
    """

    removal: bool = True
    anchor_arbitration: bool = True
    anchor_follow: bool = True
    follow_move: bool = True
    reg_deadband: bool = False
    reg_anchor_median: bool = False
    # Limiar de "visto como tinta" do motor ao vivo: camada
    # provisoria, ancora da porta de fronteira, julgamento de virada
    # e identidade da pagina no reencontro. Fica no 0,7 legado de
    # proposito, separado do runtime_threshold calibrado da mascara
    # (PageCleanInk.threshold): aquela calibracao mede IoU de um
    # quadro e nao cobre o voto entre passadas — subir este valor
    # sem replay propria engrossa traco novo (a ancora some) e faz o
    # reencontro perder a pagina (review do PR #43).
    seen_threshold: float = 0.7
    # Regra de confirmacao do voto entre passadas (janela de 3, ver
    # draw.ink.votes): um pixel confirma quando a soma das 3 passadas
    # chega a vote_score_strong (media >= ~0,63) ou quando aparece com
    # prob >= vote_hit nas 3 passadas e a soma chega a vote_score. Os
    # defaults sao os valores historicos de draw.ink.votes (o teste de
    # paridade garante que nao divergem). Com o page_clean v2, que e
    # mais confiante que o v1, a pauta cruza essa barra: veja
    # strict_vote para o regime calibrado.
    vote_hit: float = 0.5
    vote_score: float = 1.4
    vote_score_strong: float = 1.9
    # Remocao de pauta por programacao dinamica no mapa de probabilidade
    # (draw.ink.ruling), antes do registro e do voto. Opt-in: o default
    # fica desligado ate a regua (lab/bench_ink.py) e o uso ao vivo
    # provarem o ganho.
    ruling_removal: bool = False
    # Histerese espaco-temporal (draw.ink.accumulator): poeira isolada nao
    # confirma nem aparece. Escala: media da janela de votos (0-1).
    #  - confirmacao: um pixel que a regra de voto confirmaria so entra se a
    #    media for >= t_low E o componente 8-conectado dele contiver uma
    #    ancora (media >= t_high ou tinta ja confirmada);
    #  - camada provisoria exibida: so componentes 8-conectados com >=
    #    provisional_min_px pixels E com uma ancora (prob. desta passada >=
    #    t_high, ou colados a tinta confirmada).
    # Nota: a media divide sempre pela janela cheia (passada faltante = zero),
    # e um traco vizinho de tinta confirmada serve de ancora; por isso duas
    # leituras fortes + um zero (media ~0,66 >= t_low) ainda estendem um traco
    # ja confirmado. E intencional; endurecer exigiria hits == janela.
    # Calibracao medida (2026-11-19; page_clean v2, clipe real de escrita
    # `escrita-mao-sombra`, 58 passadas, SEM gabarito) — confirmado relativo
    # ao voto sem histerese, e quanto do que foi cortado e fragmento < 12 px:
    #   t_high 0.90: 75,5% (30%)   0.85: 82,6%   0.80: 87,3% (63%)
    #   t_high 0.75: 90,1%         0.70: 92,5% (90%)
    # O default 0.90 corta ~25% do confirmado nesse clipe, 70% disso em
    # componentes grandes: sem rotulo nao da para dizer se e traco fraco.
    # Nao mudar o default sem um teste de mesa rotulado.
    # Campos proprios, separados do seen_threshold (que governa identidade da
    # pagina, testemunha de remocao e ancora da porta de fronteira: ver #53).
    # Opt-in: o default fica desligado ate a regua e o uso ao vivo provarem.
    hysteresis: bool = False
    t_high: float = 0.90
    t_low: float = 0.65
    provisional_min_px: int = 12
    # Arbitro de registro e veto por movimento residual (draw.ink.motion):
    # mede por consenso de blocos o deslocamento da pagina desta passada
    # contra a referencia da ancora; desalinhada alem de motion_veto_px,
    # a passada vota sem registro (se o warp cru esta alinhado), entra
    # corrigida pelo modelo afim medido (ate 4 px) ou e vetada. Opt-in
    # (medido so em replay).
    motion_veto: bool = False
    motion_veto_px: float = 1.0
    # Intervalo minimo entre inicios de passadas ativas da tinta (s).
    # None = o piso de draw.ink.worker._PAGE_CLEAN_INTERVAL_SECONDS
    # (0,25 s); substitui apenas esse piso, mantendo sonda ociosa e
    # descanso. Simulado em 2026-11-20 (Ryzen 7700X, draw_seg a 6 Hz em
    # quadros reais, tinta INT8 de pagina inteira em 4 threads, sem a mao
    # do MediaPipe): 0,25 s -> 4,0 passadas/s em 2,3 nucleos; 0,15 s ->
    # 6,7/s em 3,1; 0,10 s -> 9,9/s em 4,3 (o fp32 em mosaico de hoje, a
    # 0,25 s: 3,9/s em 4,9). Opt-in ate a validacao ao vivo com a mao.
    pass_interval_s: float | None = None
    # Janela completa de confirmacao apos a saida da mao (worker). Sem ela,
    # o voto legado gasta 1 passada assumindo que o traco ja foi lido uma
    # vez com a mao presente; um traco revelado so depois da saida fica
    # aquem do voto e o descanso total o congela (Devin Review PR #122).
    post_exit_full_window: bool = False
    # Faixa de guarda (px) na borda externa da grade da pagina antes do
    # voto: 0 desliga (legado); None deriva do tamanho retificado em
    # grades de pagina inteira (>= 200 px).
    border_guard_px: int | None = 0

    def __post_init__(self):
        if not 0.0 < self.t_low < self.t_high <= 1.0:
            raise ValueError(
                f"histerese exige 0 < t_low < t_high <= 1, recebeu "
                f"t_low={self.t_low}, t_high={self.t_high}"
            )
        if self.provisional_min_px < 1:
            raise ValueError("provisional_min_px deve ser >= 1")
        if not self.motion_veto_px > 0.0:
            raise ValueError("motion_veto_px deve ser > 0")
        if self.pass_interval_s is not None and not 0.0 < self.pass_interval_s <= 5.0:
            raise ValueError(
                f"pass_interval_s fora de (0, 5] s: {self.pass_interval_s!r}"
            )
        if self.border_guard_px is not None and self.border_guard_px < 0:
            raise ValueError(
                f"border_guard_px deve ser >= 0 ou None: {self.border_guard_px!r}"
            )

    @classmethod
    def strict_vote(cls, level, **overrides):
        """Voto no regime do v2: confirma so o que a media das passadas
        sustenta acima de `level`. LEGACY_VOTE_LEVEL (1,9 / 3, ~0,633)
        reproduz o legado exatamente; o maximo e 0,99.

        Desloca os tres campos do voto por `shift = level - LEGACY`:
        vote_score_strong = 1,9 + 3 x shift, vote_score = 1,4 + 3 x
        shift e vote_hit = 0,5 + shift. O seen_threshold NAO muda, de
        proposito: ele governa a identidade da pagina no reencontro, a
        testemunha da remocao e a ancora da porta de fronteira (decisao
        do PR #43; subi-lo junto do voto faz a folha que volta perder
        a tinta confirmada, a borracha parcial deixar tinta velha e o
        halo do traco entrar — review do PR #53). Medido, mante-lo no
        0,7 rende quase o mesmo que subi-lo.

        Medidas e trade-offs (page_clean v2, 2026-11-19; uma sessao ao vivo de caderno pautado, 40 quadros, regioes
        marcadas a mao; relativo ao legado SEM ganho = 100%):
          legado com ganho:    escrita 149%, pauta 82%
          0,85 com ganho:      escrita 121%, pauta 57%
          0,92 com ganho:      escrita 109%, pauta 48%
          0,85 SEM ganho:      escrita  77%, pauta 54% (o N perde a diagonal)
        TRADE-OFFS: (1) so use com entrada bem exposta (ver ink_auto_gain):
        sem isso o traco real tem confianca baixa e o voto estrito quebra
        as letras. (2) Em sessao bem iluminada (s4, escrita) o ruido fora
        da escrita cai ~90%, mas o traco afina: a faixa do texto perde 48%
        dos pixels em 0,85 e 60% em 0,92 (segue legivel); no s2, 27% e 39%.
        (3) Os tracinhos onde a pauta cruza o traco sobrevivem a qualquer
        limiar — so um modelo que conheca a pauta (v4) os remove. Por isso
        `InkConfig()` segue no legado; o preset de producao, que usa este
        voto junto do ganho, e `InkConfig.production()`.
        """
        if not LEGACY_VOTE_LEVEL <= level <= 0.99:
            raise ValueError(
                f"nivel de voto fora de [{LEGACY_VOTE_LEVEL:.3f}, 0,99]: {level!r}"
            )
        shift = level - LEGACY_VOTE_LEVEL
        fields = {
            "vote_hit": 0.5 + shift,
            "vote_score": 1.4 + 3 * shift,
            "vote_score_strong": 1.9 + 3 * shift,
        }
        fields.update(overrides)
        return cls(**fields)

    @classmethod
    def production(
        cls,
        *,
        vote_level=PRODUCTION_VOTE_LEVEL,
        pass_interval_s=PRODUCTION_PASS_INTERVAL_S,
        ruling_removal=True,
    ):
        """Preset de producao do motor de tinta: voto estrito (0,85), remocao
        de pauta, histerese, veto por movimento e passadas a 0,15 s.

        E o que `PipelineConfig()` usa; `InkConfig()` continua sendo o
        legado (paridade byte a byte com os campos desligados). Riscos
        medidos: a pauta pode apagar lapis fino deitado sobre a linha
        impressa (PR #56, recall do braco 8,6%), e o voto 0,85 afina o
        traco (-27% a -48%, PR #53) e so funciona com `ink_auto_gain`.
        `ruling_removal=False` (--no-ink-ruling) deixa so a pauta de fora.
        """
        return cls.strict_vote(
            vote_level,
            ruling_removal=ruling_removal,
            hysteresis=True,
            motion_veto=True,
            pass_interval_s=pass_interval_s,
            border_guard_px=None,
        )

    @classmethod
    def sketch(cls, *, pass_interval_s=PRODUCTION_PASS_INTERVAL_S):
        """Perfil esboço/lápis: fidelidade primeiro.

        Voto legado (o strict_vote medido afina tracos -27..48% e exige
        ink_auto_gain; para desenho a lápis a perda de grafite fino
        pesa mais que a sujeira residual) + motion_veto para a poeira
        temporal. Sem hysteresis: o portão de âncora exige média >=
        t_high (0,90) ou contato com tinta confirmada, e um traço de
        lápis isolado a ~0,75 por três passadas passa o voto legado
        (2,25 >= 1,9) mas morre sem âncora (Devin Review PR #122).
        ruling_removal desligado: o removedor pode comer lápis
        deitado sobre a linha impressa (PR #56, recall do braço 8,6%).
        post_exit_full_window: o atalho de 1 passada pos-saida assume
        uma leitura com a mao presente; traco so revelado depois da
        saida precisa da janela inteira (Devin Review PR #122).
        """
        return cls.strict_vote(
            LEGACY_VOTE_LEVEL,
            ruling_removal=False,
            hysteresis=False,
            motion_veto=True,
            pass_interval_s=pass_interval_s,
            post_exit_full_window=True,
        )

    @classmethod
    def technical(
        cls,
        *,
        vote_level=TECHNICAL_VOTE_LEVEL,
        pass_interval_s=PRODUCTION_PASS_INTERVAL_S,
        ruling_removal=True,
    ):
        """Perfil técnico/CAD: vetores mais limpos para export DXF.

        Voto mais estrito (0,92 medido: escrita 109%, pauta 48% — quase
        zera a pauta residual) + removedor de pauta; o traço mais fino
        é desejado em linhas de construção.
        """
        return cls.production(
            vote_level=vote_level,
            pass_interval_s=pass_interval_s,
            ruling_removal=ruling_removal,
        )
