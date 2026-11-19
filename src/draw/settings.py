"""Configuração imutável da orquestração (draw.pipeline).

Substitui as leituras soltas de `DRAW_*` que o laço legado fazia no
meio do `step`: cada campo é um mecanismo do preset de produção
validado (docs/SIMULATOR.md do legado, commit a95eec8), resolvido
uma vez na construção. Nada aqui lê ambiente — a única leitura de
env permitida no runtime é `DRAW_MODEL_STRICT`, em
`draw.infra.manifest`.
"""

from dataclasses import dataclass

from .ink.config import InkConfig


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Comportamento do laço de orquestração, injetado por construtor.

    O default é o preset de produção:

    - coverage (era DRAW_SHEET_COVERAGE): a máscara neural da última
      detecção acompanha o resultado; retificada com a homografia de
      cada warp de tinta, veda a mesa vazada no retificado de virar
      voto de tinta nova.
    - canvas_persist (era DRAW_CANVAS_PERSIST): a queda do rastreador
      congela o canvas na tela até a folha voltar (re-registro) ou a
      sessão reabrir com página nova.
    - skip_stale (era DRAW_INK_SKIP_STALE): passadas sobre cantos
      retidos na janela de graça (sem confirmação nova além do piso
      da queda) são trabalho morto por construção e não sobem.
    - seg_hand_occlusion (era DRAW_SEG_HAND_OCCLUSION): a classe mão
      do segmentador vira segundo sinal de oclusão — basta um dizer
      "mão" para a passada não enxergar a folha ali.
    - coverage_iou_floor: piso do IoU entre o quad retificado e a
      máscara de cobertura seg, no espaço da câmera. O quad fantasma
      da mesa vazia — a máscara marca um blob espúrio e o fit infla
      quatro cantos sobre mesa — mede IoU perto de zero; a folha
      real com quad estourado em movimento segue dominada pela
      interseção. Abaixo do piso a passada de tinta não sobe:
      submeter a região é trabalho morto e cada pixel é mesa.
      Medido nas capturas reais: quads fantasmas de mesa vazia até
      0,082; pior folha real em movimento 0,514 — 0,2 separa os
      dois lados com margem >2x de cada um. 0 desliga o piso.
    - ink_divnorm: aplica draw.photometry.division_normalize no
      retificado antes do page_clean — o domínio de entrada do
      modelo de tinta v4 (sombra da mão e pauta viram problema
      supervisionado). Desligado: o v2 congelado espera RGB cru.
    - follow_move (era DRAW_INK_FOLLOW_MOVE): com a geometria de
      tinta congelada sem mão, um deslocamento sustentado da folha
      reparte o histórico de cantos na posição nova.
    - ink: o preset do motor de tinta que a trabalhadora recebe na
      construção (quem monta a Pipeline cria a trabalhadora com ele).
    """

    coverage: bool = True
    canvas_persist: bool = True
    skip_stale: bool = True
    seg_hand_occlusion: bool = True
    coverage_iou_floor: float = 0.2
    ink_divnorm: bool = False
    follow_move: bool = True
    ink: InkConfig = InkConfig()
