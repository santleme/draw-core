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
      Medido nas capturas s1–s4 com o draw_seg de produção
      (2026-11-19, 1 frame a cada 5): 52 quads fantasmas de mesa
      vazia com IoU 0,044–0,108; folhas reais com IoU ≥ 0,298 (pior
      caso em s3, geometria; s2/s4 ≥ 0,86). 0,2 separa os dois lados
      com margem ~1,5x para cada um. Fração do quad coberta e IoU por
      componente conexo foram medidos e NÃO separam (se sobrepõem).
      A máscara é a cobertura (papel+mão+tinta) de propósito: na mesa
      vazia o seg marca 15%–24% do frame como mão fora do quad, e é
      isso que derruba o IoU do fantasma — sem a classe mão o
      fantasma sobe para 0,23–0,41 e cruza a folha real. A união do
      IoU (draw.pipeline._coverage_iou) conta os componentes não-mão
      que tocam o quad dilatado pelo anel de rasterização mais TODA a
      mão prevista: a mancha de papel falsa disjunta da issue #106
      sai da conta sem levar a proteção do fantasma junto — salvo
      quando a mão fecha um anel ao redor do vão e o preenchimento
      de buracos da máscara de voto o reconecta à folha (resíduo
      conhecido; o mapa cru não atravessa o contrato de detecção).
      LIMITAÇÃO CONHECIDA (não medida): a classe mão segue contando
      inteira na união do IoU — mão ou antebraço FORA da folha incham
      o denominador. Folha pequena com um antebraço grande em cena
      pode cair abaixo do piso e ser vetada (a tinta não sobe até a
      mão sair). Nenhuma captura real desse caso existe ainda; o pior
      caso medido com mão (s4, escrita) fica em IoU 0,864. Se
      aparecer, `coverage_iou_floor=0` desliga o piso.
      0 desliga o piso.
    - ink_divnorm: o retificado sobe cru e a trabalhadora aplica
      draw.photometry.division_normalize na passada, antes do
      page_clean — o domínio de entrada do modelo de tinta v4
      (sombra da mão e pauta viram problema supervisionado).
      Desligado: o v2 congelado espera RGB cru. O flag precisa
      bater nos dois lados: `PipelineConfig.ink_divnorm` desliga o
      preenchimento/ganho na pipeline e o `ink_divnorm` da
      PageCleanWorker normaliza — ligado num lado só entrega warp
      sem normalização ao modelo.
    - follow_move (era DRAW_INK_FOLLOW_MOVE): com a geometria de
      tinta congelada sem mão, um deslocamento sustentado da folha
      reparte o histórico de cantos na posição nova.
    - ink_auto_gain: ganho digital no retificado antes do page_clean
      (draw.exposure): leva o papel ao nível ~200 quando a câmera
      entrega a página escura (teto físico de 1/30 s a 30 fps, driver
      sem gain UVC). Nunca escurece; ligado por padrão (PR #52 mediu
      ~1 ms e o voto estrito de produção depende dele).
      Ignorado com ink_divnorm, que já normaliza a escala (o ganho
      antes dele clipava o papel claro e apagava traço fraco).
    - ink: o preset do motor de tinta que a trabalhadora recebe na
      construção (quem monta a Pipeline cria a trabalhadora com ele).
      Padrão: `InkConfig.production()`; `InkConfig()` é o legado.
    - detect_wake: cadência adaptativa do detector com sentinela de
      bordas (PR #69), ligada por padrão. Cai para 1 Hz quando a folha
      assenta e acorda no mesmo quadro se ela mexer.
    - spread_mode: como combinar contornos de duas páginas. `auto`
      conserva o limite de calha de 10%; `always` aceita calhas maiores
      quando tamanho e alinhamento combinam; `never` mantém as páginas
      separadas.
    """

    coverage: bool = True
    canvas_persist: bool = True
    skip_stale: bool = True
    seg_hand_occlusion: bool = True
    coverage_iou_floor: float = 0.2
    # Liga junto com o ink_divnorm da PageCleanWorker: os dois lados
    # precisam bater — ligado num lado só entrega warp sem
    # normalização ao modelo (ver a docstring acima).
    ink_divnorm: bool = False
    follow_move: bool = True
    ink_auto_gain: bool = True
    ink: InkConfig = InkConfig.production()
    detect_wake: bool = True
    detect_wake_interval_s: float = 1.0
    spread_mode: str = "auto"

    def __post_init__(self):
        if not 0.0 < self.detect_wake_interval_s <= 5.0:
            raise ValueError(
                "detect_wake_interval_s fora de (0, 5] s: "
                f"{self.detect_wake_interval_s!r}"
            )
        if self.spread_mode not in ("auto", "always", "never"):
            raise ValueError(
                f"spread_mode deve ser um de: auto, always, never: {self.spread_mode!r}"
            )
