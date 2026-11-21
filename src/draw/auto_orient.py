"""Deteccao automatica de orientacao: camera invertida (180°) e folha retrato/paisagem.

Modulo puro (apenas numpy) que observa sinais ja produzidos pelo pipeline
(mascara semantica de mao do detector neural e cantos estabilizados da folha)
para eliminar ajustes manuais de `--rotate 180` e `--orientation paisagem`:

- Camera invertida (180°): numa mesa normal, o braco/mao entra pela borda
  inferior do quadro; com a webcam montada de cabeca para baixo, entra pela
  borda superior. Depois de `confirm_votes` deteccoes distintas seguidas
  indicando entrada pelo topo, sinaliza a inversao uma unica vez (sem oscilar).
- Orientacao da pagina (retrato <-> paisagem): mede a razao largura/altura
  dos cantos rastreados com histerese e exige `confirm_frames` quadros
  consecutivos antes de alternar entre retrato e paisagem (caderno aberto).
"""

import numpy as np

_HAND_MIN_AREA_RATIO = 0.005
_HAND_BORDER_STRIP_RATIO = 0.12
_HAND_BORDER_MIN_PIXELS = 20
_HAND_BORDER_DOMINANCE = 3.0


def hand_mask_orientation_vote(hand_mask):
    """Infere se a mao entra por baixo ('upright') ou por cima ('inverted').

    Recebe a mascara uint8 (H, W) da classe mao (nao-zero = mao).
    Devolve 'inverted', 'upright' ou None quando a evidencia e ambigua
    (mao pequena demais, sem tocar bordas ou tocando topo e base juntos).
    """
    if hand_mask is None or hand_mask.ndim < 2 or hand_mask.size == 0:
        return None
    h, w = hand_mask.shape[:2]
    total = int(np.count_nonzero(hand_mask))
    if total < max(40, int(h * w * _HAND_MIN_AREA_RATIO)):
        return None
    strip = max(4, int(round(h * _HAND_BORDER_STRIP_RATIO)))
    top_px = int(np.count_nonzero(hand_mask[:strip, :]))
    bot_px = int(np.count_nonzero(hand_mask[h - strip :, :]))
    if top_px >= _HAND_BORDER_MIN_PIXELS and top_px >= _HAND_BORDER_DOMINANCE * max(
        bot_px, 1
    ):
        return "inverted"
    if bot_px >= _HAND_BORDER_MIN_PIXELS and bot_px >= _HAND_BORDER_DOMINANCE * max(
        top_px, 1
    ):
        return "upright"
    return None


class AutoRotationDetector:
    """Confirma camera invertida (180°) apos votos consecutivos distintos.

    Dispara no maximo uma inversao por sessao para evitar oscilacao caso
    o usuario passe a mao por cima da folha mais tarde.
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


def quad_aspect_ratio(corners):
    """Razao largura / altura media de um quadrilatero TL/TR/BR/BL."""
    quad = np.asarray(corners, dtype=np.float64)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        return None
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

    def observe(self, corners):
        """Observa cantos rastreados; devolve a nova orientacao ou None."""
        ratio = quad_aspect_ratio(corners)
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
