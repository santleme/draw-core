"""Contratos tipados entre os estágios do pipeline.

Dataclasses congeladas (frozen, slots): os módulos trocam objetos
imutáveis destes tipos, nunca dicts soltos ou tuplas posicionais
(docs/MIGRATION_SPEC.md, "Rules for what gets copied"). Nenhuma
dependência além de numpy — o contrato é a fronteira para uma
futura implementação nativa dos estágios quentes.
"""

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np

_CORNERS_SHAPE = (4, 2)
_PageDecision = Literal["same", "new_page", "defer"]

# Todo veredito que o pipeline legado produz ou propaga no slot
# `verdict`: "paper" (o interior do candidato passa no portão de
# textura — paper.py/neural.py), "blank_maybe" (candidato branco o
# bastante, mas liso demais para afirmar papel — o rastreador pede
# confirmação mais longa) e "none" (o laço rebaixa para "none" uma
# detecção fraca barrada no portão de primeira confirmação). Ausência
# de veredito é representada por None, não por "none".
SHEET_VERDICTS = frozenset({"paper", "blank_maybe", "none"})


def _corners_float32(value, name):
    """Normaliza cantos para (4, 2) float32, com cópia própria."""
    corners = np.array(value, dtype=np.float32, copy=True)
    if corners.shape != _CORNERS_SHAPE:
        raise ValueError(f"{name} precisa ter forma {_CORNERS_SHAPE}: {corners.shape}")
    corners.flags.writeable = False
    return corners


def _readonly(value, copy=False):
    """Devolve visão (ou cópia) somente-leitura do array."""
    out = value.copy() if copy else value.view()
    out.flags.writeable = False
    return out


def _corners_field(obj, name):
    """Normaliza obj.<name> para (4, 2) float32 somente-leitura; None passa."""
    if (value := getattr(obj, name)) is not None:
        object.__setattr__(obj, name, _corners_float32(value, name))


def _mask_2d(obj, name, *, copy=False, dtypes=(np.uint8,), optional=True):
    """Valida a máscara 2D obj.<name> e grava a forma somente-leitura."""
    if (value := getattr(obj, name)) is None:
        if optional:
            return
    elif isinstance(value, np.ndarray) and value.ndim == 2 and value.dtype in dtypes:
        object.__setattr__(obj, name, _readonly(value, copy=copy))
        return
    desc = "/".join(np.dtype(t).name for t in dtypes)
    if optional:
        desc += " ou None"
    raise ValueError(f"{name} precisa ser máscara 2D {desc}")


def _check_captured_at(value, optional=True):
    """Exige carimbo monotônico finito; None passa só se optional."""
    if value is None:
        if optional:
            return
    elif isinstance(value, (int, float)) and np.isfinite(value):
        return
    raise ValueError(f"captured_at precisa ser float finito: {value!r}")


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Um quadro capturado: imagem BGR (H, W, 3) uint8 e identidade.

    captured_at é um carimbo monotônico em segundos (time.monotonic);
    sequence é o número de sequência da fonte, >= 0. image é guardada
    como visão somente-leitura do buffer recebido (sem copiar o frame):
    o produtor não pode mutar o buffer depois de publicar o pacote.

    source_frame_id e source_ts_ms são a identidade dada pelo emissor
    quando a fonte a expõe (ajudante Windows/.drawcap): o frame_id da
    sessão dele e o relógio monotônico de captura em milissegundos.
    É outro domínio de relógio — só serve para diferenças dentro da
    mesma sessão, nunca para comparar com captured_at. None para
    fontes sem identidade própria (webcam local, URLs genéricas).
    """

    image: np.ndarray
    captured_at: float
    sequence: int
    source_frame_id: int | None = None
    source_ts_ms: int | None = None

    def __post_init__(self):
        if not isinstance(self.image, np.ndarray) or self.image.ndim != 3:
            raise ValueError("image precisa ser um ndarray (H, W, 3)")
        if self.image.shape[2] != 3:
            raise ValueError(f"image precisa ter 3 canais: {self.image.shape}")
        if self.image.dtype != np.uint8:
            raise ValueError(f"image precisa ser uint8: {self.image.dtype}")
        if self.sequence < 0:
            raise ValueError(f"sequence precisa ser >= 0: {self.sequence}")
        if self.source_frame_id is not None and self.source_frame_id < 0:
            raise ValueError(
                f"source_frame_id precisa ser >= 0: {self.source_frame_id}"
            )
        if self.source_ts_ms is not None and self.source_ts_ms < 0:
            raise ValueError(f"source_ts_ms precisa ser >= 0: {self.source_ts_ms}")
        object.__setattr__(self, "image", _readonly(self.image))


@dataclass(frozen=True, slots=True)
class SheetQuad:
    """O quadrilátero da folha no espaço da câmera.

    corners é (4, 2) float32 ou None; verdict é None ou um de
    SHEET_VERDICTS ("paper"/"blank_maybe"/"none" — a folha pode
    confirmar com o rótulo "blank_maybe" que a sustentou). O par
    sempre vem junto: sem folha confirmada é (None, None).
    detector nomeia a fonte.
    """

    corners: np.ndarray | None
    verdict: str | None
    detector: str = ""

    def __post_init__(self):
        if (self.corners is None) != (self.verdict is None):
            raise ValueError(
                "corners e verdict precisam vir juntos: "
                "(cantos, 'paper') ou (None, None)"
            )
        _corners_field(self, "corners")
        if self.verdict is not None and self.verdict not in SHEET_VERDICTS:
            raise ValueError(
                f"verdict precisa ser None ou um de {sorted(SHEET_VERDICTS)}: "
                f"{self.verdict!r}"
            )

    @property
    def present(self):
        """True quando há quadrilátero de folha."""
        return self.corners is not None


@dataclass(frozen=True, slots=True)
class PageIdentity:
    """Decisão de identidade da página (F-11).

    generation identifica a sessão de página; decision é o desfecho
    do juiz; recall/containment são as evidências do confronto, em
    [0, 1] quando presentes.
    """

    generation: int
    decision: _PageDecision
    recall: float | None = None
    containment: float | None = None

    def __post_init__(self):
        if self.decision not in get_args(_PageDecision):
            raise ValueError(f"decision inválida: {self.decision!r}")
        for name in ("recall", "containment"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} precisa estar em [0, 1]: {value}")


@dataclass(frozen=True, slots=True)
class InkState:
    """Estado da tinta numa geração de página.

    confirmed e provisional são máscaras 2D uint8 (None enquanto a
    passada não produziu), guardadas como visões somente-leitura do
    buffer recebido: o produtor não pode mutar a máscara depois de
    publicar. anchor_corners é o quad (4, 2) que ancora a geometria
    da tinta na cena.
    """

    confirmed: np.ndarray | None
    provisional: np.ndarray | None
    generation: int
    anchor_corners: np.ndarray | None = None

    def __post_init__(self):
        for name in ("confirmed", "provisional"):
            _mask_2d(self, name)
        _corners_field(self, "anchor_corners")


@dataclass(frozen=True, slots=True)
class WarpedPage:
    """Uma página retificada entregue à trabalhadora de tinta.

    image é o warp: ndarray 2D ou (H, W, 3) uint8 — a saída de
    warp_sheet sobre o quadro da câmera — guardada como visão
    somente-leitura do buffer recebido. generation é a geração de
    sessão da folha, >= 0. corners é o quad (4, 2) float32
    somente-leitura que retificou ESTE warp, ou None. visible é a
    máscara de visibilidade do FRAME que originou este warp
    (verdadeiro onde a câmera vê a página; falso sob a mão) e
    coverage a cobertura de folha do mesmo frame (verdadeiro onde
    o quad cobre papel de verdade): ambas 2D bool/uint8 na grade
    (altura, largura) do FRAME DE CÂMERA quando corners vem junto
    — a trabalhadora retifica câmera→página ou câmera→âncora com
    corners (issue #100). Sem corners não há geometria de câmera:
    as máscaras são lidas na grade do próprio warp, como antes.
    captured_at é o carimbo monotônico (s) do quadro de câmera que
    originou o warp, ou None.
    """

    image: np.ndarray
    generation: int
    corners: np.ndarray | None = None
    visible: np.ndarray | None = None
    coverage: np.ndarray | None = None
    captured_at: float | None = None

    def __post_init__(self):
        _check_captured_at(self.captured_at)
        if not isinstance(self.image, np.ndarray) or self.image.ndim not in (2, 3):
            raise ValueError("image precisa ser um ndarray 2D ou (H, W, 3)")
        if self.image.ndim == 3 and (
            self.image.shape[2] != 3 or self.image.dtype != np.uint8
        ):
            raise ValueError(f"image (H, W, 3) precisa ser uint8: {self.image.dtype}")
        object.__setattr__(self, "image", _readonly(self.image))
        if self.generation < 0:
            raise ValueError(f"generation precisa ser >= 0: {self.generation}")
        _corners_field(self, "corners")
        for name in ("visible", "coverage"):
            mask = getattr(self, name)
            if mask is None:
                continue
            _mask_2d(self, name, dtypes=(bool, np.uint8))
            if self.corners is None and mask.shape != self.image.shape[:2]:
                raise ValueError(
                    f"{name} sem corners segue a grade do warp: "
                    f"{mask.shape} != {self.image.shape[:2]}"
                )


@dataclass(frozen=True, slots=True)
class InkBaselineState:
    """Prontidao da referencia inicial da folha (InkConfig.page_model).

    phase: "disabled", "waiting_page", "seeding", "recapturing",
    "ready_partial" (ha area visivel ainda sem referencia) ou "ready".
    fraction e a fracao da grade ja semeada; pending_px os pixels de
    papel rastreado ainda sem referencia, inclusive os cobertos agora.
    "ready" so sem nenhum pendente.
    """

    enabled: bool
    ready: bool
    fraction: float
    pending_px: int
    phase: str


@dataclass(frozen=True, slots=True)
class InkSnapshot:
    """Uma publicação do canvas de tinta confirmada.

    version é o número da publicação dentro da geração; confirmed é
    a máscara 2D uint8 do canvas, guardada como cópia
    somente-leitura: mutações posteriores no buffer do produtor
    não atravessam a fronteira.
    """

    version: int
    confirmed: np.ndarray

    def __post_init__(self):
        _mask_2d(self, "confirmed", copy=True, optional=False)


@dataclass(frozen=True, slots=True)
class InkDelta:
    """O que a tinta confirmada ganhou desde `since_version`.

    Cobre (since_version, version] de uma geração. mask é a máscara
    2D uint8 (255 = pixel recém-confirmado) recortada em bbox =
    (y0, y1, x0, x1), meio-aberto, em pixels da grade da página
    page_grid = (altura, largura); bbox None e mask (0, 0) quando
    nada mudou. full=True é descontinuidade (primeira publicação da
    geração, tinta removida ou migrada, ou histórico insuficiente
    para o since_version pedido): mask/bbox descrevem TODO o canvas
    confirmado e o consumidor deve substituir o seu, não somar.
    removed_px conta pixels que saíram do canvas nesse intervalo
    (limite inferior se o histórico foi truncado).
    full também cobre âncora movida (re-ancoragem) no intervalo.
    captured_at é o carimbo monotônico (s, mesmo relógio de
    FramePacket.captured_at) do quadro da passada que produziu a
    publicação mais recente, ou None. mm_per_px e anchor_corners
    (quad (4, 2) da âncora da geração) levam a página ao mundo.
    """

    generation: int
    since_version: int
    version: int
    captured_at: float | None
    bbox: tuple[int, int, int, int] | None
    mask: np.ndarray
    page_grid: tuple[int, int]
    mm_per_px: float | None
    anchor_corners: np.ndarray | None
    removed_px: int
    full: bool

    def __post_init__(self):
        _mask_2d(self, "mask", copy=True, optional=False)
        mask = self.mask
        if self.bbox is None:
            if mask.size:
                raise ValueError("bbox None exige mask vazia")
        else:
            y0, y1, x0, x1 = self.bbox
            if mask.shape != (y1 - y0, x1 - x0) or y1 <= y0 or x1 <= x0:
                raise ValueError(f"mask {mask.shape} não bate com bbox {self.bbox}")
        if self.version < self.since_version:
            raise ValueError("version precisa ser >= since_version")
        if self.removed_px < 0:
            raise ValueError("removed_px precisa ser >= 0")
        _corners_field(self, "anchor_corners")


@dataclass(frozen=True, slots=True)
class InkDisplaySnapshot:
    """Uma publicação das camadas de tinta para a tela.

    version é o número da publicação dentro da geração; confirmed e
    provisional são as máscaras 2D uint8 das camadas, de mesma
    forma, guardadas como cópias somente-leitura.
    """

    version: int
    confirmed: np.ndarray
    provisional: np.ndarray

    def __post_init__(self):
        for name in ("confirmed", "provisional"):
            _mask_2d(self, name, copy=True, optional=False)
        if self.confirmed.shape != self.provisional.shape:
            raise ValueError(
                f"confirmed e provisional precisam ter a mesma forma: "
                f"{[self.confirmed.shape, self.provisional.shape]}"
            )


@dataclass(frozen=True, slots=True)
class InkPublication:
    """Uma publicação da tinta confirmada para a saída ao vivo.

    generation e version identificam a publicação (a geração da trabalhadora
    e a versão dentro dela); confirmed é a máscara 2D uint8 somente-leitura,
    ou None para "canvas limpo" (troca de câmera: a tinta antiga morreu).
    captured_at (monotônico, s), anchor_corners (quad (4, 2) da âncora) e
    full (descontinuidade: tinta removida/migrada ou âncora movida) vêm do
    InkDelta da mesma passada; None/False quando não se pediu a metadados ou
    a publicação não trouxe tinta nova.
    """

    generation: int
    version: int
    confirmed: np.ndarray | None = None
    captured_at: float | None = None
    anchor_corners: np.ndarray | None = None
    full: bool = False

    def __post_init__(self):
        _check_captured_at(self.captured_at)
        _corners_field(self, "anchor_corners")
        _mask_2d(self, "confirmed")


@dataclass(frozen=True, slots=True)
class GateInfo:
    """O pacote do portão de primeira confirmação (frente 88).

    Extraído do rastro da detecção combinada quando ela traz os
    campos do portão: a fonte do desfecho, o logit de presença e a
    fração de aresta da máscara. Rastro sem esses campos não tem
    pacote — o laço não julga por falta de sinal.
    """

    source: str | None = None
    presence_logit: float | None = None
    edge_in: float | None = None


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Um resultado de detecção de folha publicado pelo worker.

    sequence é o número de sequência do resultado, >= 0. corners é
    (4, 2) float32 somente-leitura quando presente; verdict é None
    ou um de SHEET_VERDICTS ("paper"/"blank_maybe"/"none") — os dois
    NÃO precisam vir juntos, porque "papel visto, quadrilátero não
    fechou" é um caso legítimo (presença sem posição). captured_at é
    o carimbo de captura do quadro
    examinado; detect_ms é a duração da detecção em ms, >= 0.
    gate_info é o pacote do portão quando o rastro o traz.
    coverage_mask e hand_mask são máscaras 2D uint8 (visões
    somente-leitura do buffer recebido) publicadas só quando o
    detector roda com cobertura; sem cobertura ficam None.
    """

    sequence: int
    corners: np.ndarray | None
    verdict: str | None
    captured_at: float
    detect_ms: float
    gate_info: GateInfo | None = None
    coverage_mask: np.ndarray | None = None
    hand_mask: np.ndarray | None = None

    def __post_init__(self):
        if self.sequence < 0:
            raise ValueError(f"sequence precisa ser >= 0: {self.sequence}")
        _corners_field(self, "corners")
        if self.verdict is not None and self.verdict not in SHEET_VERDICTS:
            raise ValueError(
                f"verdict precisa ser None ou um de {sorted(SHEET_VERDICTS)}: "
                f"{self.verdict!r}"
            )
        if self.detect_ms < 0:
            raise ValueError(f"detect_ms precisa ser >= 0: {self.detect_ms}")
        for name in ("coverage_mask", "hand_mask"):
            _mask_2d(self, name)


@dataclass(frozen=True, slots=True)
class HandSignature:
    """Assinatura de presença de mão num quadro.

    area é a área do casco convexo dos marcos (ou dos pixels da
    máscara, na fonte de reserva), em px, >= 0; centroid_x e
    centroid_y são o centroide em px do quadro. Os três são
    floats finitos. HandSignature(0.0, 0.0, 0.0) é o "sem mão" de
    uma detecção que respondeu; indisponível é None, não a
    assinatura zero.
    """

    area: float
    centroid_x: float
    centroid_y: float

    def __post_init__(self):
        for name in ("area", "centroid_x", "centroid_y"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{name} precisa ser float finito: {value!r}")
        if self.area < 0:
            raise ValueError(f"area precisa ser >= 0: {self.area}")


@dataclass(frozen=True, slots=True)
class HandSample:
    """Uma amostra de presença de mão.

    present é True (mão), False (sem mão) ou None (sinal
    indisponível — sem modelo ou landmarker). moved é o sinal de
    atividade da assinatura: passo de centroide ou variação
    relativa de área além dos limiares.
    """

    present: bool | None
    moved: bool

    def __post_init__(self):
        if self.present not in (True, False, None):
            raise ValueError(
                f"present precisa ser True, False ou None: {self.present!r}"
            )
        if not isinstance(self.moved, bool):
            raise ValueError(f"moved precisa ser bool: {self.moved!r}")


@dataclass(frozen=True, slots=True)
class HandPresenceResult:
    """Amostra publicada com o timestamp de captura em segundos.

    sequence ordena publicações da trabalhadora; source_id é a geração
    da fonte de câmera e muda também quando a orientação muda.
    """

    sequence: int
    captured_at: float
    source_id: int
    sample: HandSample


@dataclass(frozen=True, slots=True)
class HandOcclusion:
    """Evidência de oclusão produzida para um quadro da câmera.

    Máscaras booleanas na grade da captura. None é evidência
    indisponível e uma máscara vazia é uma observação livre.
    source_id identifica a geração da fonte, incluindo sua orientação.
    present e moved seguem a semântica de HandSample.
    """

    frame_sequence: int
    captured_at: float
    source_id: int
    occlusion_mask: np.ndarray | None = None
    orientation_mask: np.ndarray | None = None
    present: bool | None = None
    moved: bool = False

    def __post_init__(self):
        if self.frame_sequence < 0 or self.source_id < 0:
            raise ValueError("frame_sequence e source_id precisam ser >= 0")
        _check_captured_at(self.captured_at, optional=False)
        HandSample(self.present, self.moved)
        for name in ("occlusion_mask", "orientation_mask"):
            _mask_2d(self, name, dtypes=(bool,), copy=True)
        a, b = self.occlusion_mask, self.orientation_mask
        if a is not None and b is not None and a.shape != b.shape:
            raise ValueError("as máscaras precisam usar a mesma grade de câmera")


@dataclass(frozen=True, slots=True)
class ExposureAssessment:
    """Veredito fotométrico de uma passada da trabalhadora de tinta.

    Produzido pelo guarda de exposição do warp (ink/occlusion) e
    consumido pelo voto: máscaras independentes porque adicionar e
    remover tinta pedem provas diferentes. visible é máscara 2D
    bool/uint8 na grade do warp — True onde a imagem sustenta
    evidência — ou None quando a guarda não encontrou região
    bloqueada (não é "evidência indisponível": imagem sem fotometria
    confiável devolve None no observe inteiro). paper_exposed é
    bool (página inteira verificada) ou máscara 2D — True onde a
    imagem prova papel exposto, o requisito separado da remoção de
    tinta confirmada.
    """

    visible: np.ndarray | None
    paper_exposed: np.ndarray | bool

    def __post_init__(self):
        # O tipo declarado de paper_exposed não admite None: uma
        # avaliação emitida sem veredito de papel não é "sem avaliação"
        # (None no acumulador reabriria o contrato legado — Devin
        # Review do PR #144), é "nada prova papel" — False congela.
        if self.paper_exposed is None:
            object.__setattr__(self, "paper_exposed", False)
        for name in ("visible", "paper_exposed"):
            mask = getattr(self, name)
            if mask is None:
                continue
            # Escalar só vale para paper_exposed (veredito da página
            # inteira); visible é sempre máscara ou None.
            if isinstance(mask, (bool, np.bool_)):
                if name == "visible":
                    raise ValueError(
                        "visible precisa ser None ou máscara 2D bool/uint8"
                    )
                continue
            _mask_2d(self, name, dtypes=(bool, np.uint8))
