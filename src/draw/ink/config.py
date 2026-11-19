"""Preset de producao do motor de tinta (o que as flags Tier A/B liam do ambiente)."""

from dataclasses import dataclass


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
