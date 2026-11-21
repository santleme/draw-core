"""Deteccao automatica de orientacao: camera invertida (180°) e folha retrato/paisagem.

Observa sinais ja produzidos pelo pipeline (mascara semantica de mao do
detector neural e cantos estabilizados da folha) para eliminar ajustes
manuais de `--rotate 180` e `--orientation paisagem`:

- Camera invertida (180°): exige contato real do componente principal da mao
  com a borda superior (ou inferior) do quadro; entradas laterais perto do
  topo se abstem. Depois de `confirm_votes` deteccoes distintas seguidas
  indicando entrada pelo topo, sinaliza a inversao uma unica vez por fonte.
- Orientacao da pagina (retrato <-> paisagem): usa a razao corrigida de
  perspectiva (`paper.estimate_format`) quando o formato do quadro esta
  disponivel (abstendo-se em projecoes ambiguas/obliquas), com histerese e
  `confirm_frames` quadros consecutivos antes de alternar.
"""

import cv2
import numpy as np

from . import paper as paper_module

_HAND_MIN_AREA_RATIO = 0.005
_HAND_EDGE_TOL_RATIO = 0.02
_HAND_EDGE_MIN_PIXELS = 12
_HAND_BORDER_DOMINANCE = 3.0


def hand_mask_orientation_vote(hand_mask):
    """Infere se a mao entra por baixo ('upright') ou por cima ('inverted').

    Avalia o maior componente conectado da mascara de mao e exige contato
    direto com a borda superior ou inferior do quadro (tolerancia de 2% da
    altura para recortes de segmentacao). Entradas apenas pelas laterais
    (esquerda/direita) devolvem None para nao virar cameras corretas.
    """
    if hand_mask is None or hand_mask.ndim < 2 or hand_mask.size == 0:
        return None
    h, w = hand_mask.shape[:2]
    binary = (hand_mask > 0).astype(np.uint8)
    min_area = max(40, int(h * w * _HAND_MIN_AREA_RATIO))
    if int(np.count_nonzero(binary)) < min_area:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return None
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[largest_label, cv2.CC_STAT_AREA]) < min_area:
        return None

    comp = labels == largest_label
    tol = max(3, int(round(h * _HAND_EDGE_TOL_RATIO)))
    top_contact = int(np.count_nonzero(comp[:tol, :]))
    bot_contact = int(np.count_nonzero(comp[h - tol :, :]))

    if top_contact >= _HAND_EDGE_MIN_PIXELS and top_contact >= (
        _HAND_BORDER_DOMINANCE * max(bot_contact, 1)
    ):
        return "inverted"
    if bot_contact >= _HAND_EDGE_MIN_PIXELS and bot_contact >= (
        _HAND_BORDER_DOMINANCE * max(top_contact, 1)
    ):
        return "upright"
    return None


class AutoRotationDetector:
    """Confirma camera invertida (180°) apos votos consecutivos distintos.

    Dispara no maximo uma inversao por fonte (chame `reset()` ao trocar de
    camera) para evitar oscilacao durante a sessao.
    """

    def __init__(self, confirm_votes=4):
        if confirm_votes < 1:
            raise ValueError("confirm_votes precisa ser >= 1")
        self._confirm_votes = int(confirm_votes)
        self._inverted_streak = 0
        self._flipped = False

    @property
    def flipped(self):
        return self._flipped

    def reset(self):
        """Reinicia o estado ao trocar para outra fonte de camera."""
        self._inverted_streak = 0
        self._flipped = False

    def observe_vote(self, vote):
        """Processa 'inverted', 'upright' ou None. True no instante do flip."""
        if self._flipped or vote is None:
            return False
        if vote == "inverted":
            self._inverted_streak += 1
            if self._inverted_streak >= self._confirm_votes:
                self._flipped = True
                self._inverted_streak = 0
                return True
            return False
        if vote == "upright":
            self._inverted_streak = 0
        return False

    def observe_mask(self, hand_mask):
        return self.observe_vote(hand_mask_orientation_vote(hand_mask))


def quad_aspect_ratio(corners, frame_shape=None):
    """Razao largura / altura de um quadrilatero TL/TR/BR/BL.

    Quando `frame_shape` e informado, usa a estimativa corrigida de
    perspectiva de `paper.estimate_format` para que uma folha em pe vista
    em angulo obliquo (encurtamento vertical) nao seja confundida com
    paisagem; devolve None quando a projecao e ambigua/degenerada.
    """
    quad = np.asarray(corners, dtype=np.float64)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        return None
    if frame_shape is not None:
        est = paper_module.estimate_format(quad, frame_shape)
        ratio = est.get("ratio")
        if ratio is None or not np.isfinite(ratio) or ratio <= 0:
            return None
        return float(ratio)
    top = float(np.linalg.norm(quad[1] - quad[0]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    width = 0.5 * (top + bottom)
    height = 0.5 * (left + right)
    if width <= 1e-6 or height <= 1e-6:
        return None
    return width / height


class AutoPageOrientation:
    """Alterna entre 'retrato' e 'paisagem' com histerese pela forma da folha.

    Caderno aberto (2 paginas) ou folha deitada tem largura > altura
    (ratio >= landscape_ratio); folha em pe tem largura < altura
    (ratio <= portrait_ratio). A zona morta entre 0,85 e 1,18 impede
    trocas espurias em folhas quase quadradas ou muito inclinadas.
    """

    def __init__(
        self,
        initial="retrato",
        *,
        confirm_frames=5,
        landscape_ratio=1.18,
        portrait_ratio=0.85,
    ):
        if initial not in ("retrato", "paisagem"):
            raise ValueError(f"orientacao inicial invalida: {initial!r}")
        self._current = initial
        self._confirm_frames = int(confirm_frames)
        self._landscape_ratio = float(landscape_ratio)
        self._portrait_ratio = float(portrait_ratio)
        self._pending = None
        self._streak = 0

    @property
    def current(self):
        return self._current

    def reset_streak(self):
        self._pending = None
        self._streak = 0

    def observe(self, corners, frame_shape=None):
        """Observa cantos rastreados; devolve a nova orientacao ou None."""
        ratio = quad_aspect_ratio(corners, frame_shape=frame_shape)
        if ratio is None:
            return None
        target = None
        if self._current == "retrato" and ratio >= self._landscape_ratio:
            target = "paisagem"
        elif self._current == "paisagem" and ratio <= self._portrait_ratio:
            target = "retrato"
        if target is None:
            self._pending = None
            self._streak = 0
            return None
        if target == self._pending:
            self._streak += 1
        else:
            self._pending = target
            self._streak = 1
        if self._streak >= self._confirm_frames:
            self._current = target
            self._pending = None
            self._streak = 0
            return target
        return None
