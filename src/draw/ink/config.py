"""Preset de producao do motor de tinta (o que as flags Tier A/B liam do ambiente)."""

from dataclasses import dataclass

# Nivel de voto que reproduz o legado: a media das 3 passadas que fecha a
# soma strong de 1,9. strict_vote desloca os campos a partir dele.
LEGACY_VOTE_LEVEL = 1.9 / 3


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
        o default segue no legado.
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
