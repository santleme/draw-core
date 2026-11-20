"""Veto de passada por movimento residual da folha (InkConfig.motion_veto).

Medido em 2026-11-20 no replay da escrita-folha-movida (preset ganho +
pauta + histerese): nas passadas 5-9 a folha deslizou 1,9-3,3 px na
grade da ancora DEPOIS do registro, e essas passadas carimbaram ~42 mil
dos 107 mil px novos do clipe (39%) — tinta duplicada ao lado da tinta
assentada. O registro nao viu o deslizamento porque o quad que ele le
estava parado: o detector alternava entre dois modos (rodape fora do
quadro, saltos crus de 60-120 px de camera) e o rastreador segurava o
quad confirmado por ~1,75 s esperando o salto persistir. Nesse
intervalo o deslocamento so aparece na imagem.

A medida e a correlacao de fase entre a pagina registrada desta
passada e a referencia da geracao, em meia escala e so na textura
(passa-alta: pauta e tinta, sem iluminacao), fora da mao. Medido na
pagina retificada real (e5): piso 0,35-0,40 px parada, erro <= 0,22 px
em deslocamentos sinteticos de ate 3 px; ~6,5 ms por passada (p50) no
replay (a passada do modelo custa ~260 ms).

Decisao por passada:
  - residuo <= limiar: aceita (historicos zeram);
  - residuo > limiar e a folha ainda anda: veta (a passada nao vota);
  - residuo > limiar ESTAVEL por `settle_passes` passadas, pequeno
    (<= MOTION_CORRECT_MAX_PX) e RIGIDO (as quatro metades da pagina
    concordam com o global ate MOTION_RIGID_TOL_PX): a folha parou
    deslocada por translacao pura, abaixo da banda morta do
    rastreador — o quad pode nunca mais andar. A passada entra
    corrigida pela translacao medida, em vez de vetar para sempre;
  - deslocamento grande ou nao-rigido segue vetado: grande, os cantos
    andam alem da banda morta e o registro alcanca a folha; nao-rigido
    (rotacao, papel deformando), a translacao global desalinha a outra
    metade. Medido na folha-movida: corrigir sem a checagem de rigidez
    criou um terceiro jogo de copias na metade direita (+17 mil px), com
    esquerda e direita discordando em 2,7 px (e ate 30 px depois);
  - trava de vivacidade: MOTION_FAIL_OPEN_PASSES vetos seguidos e a
    passada entra como entraria sem o veto (o comportamento de hoje).
    O veto nunca silencia o acumulador para sempre, e no pior caso o
    resultado e o de sem veto.

Re-ancorar na posicao nova, a outra saida possivel, foi descartado: a
tinta ja confirmada fica na grade velha, e a evidencia nova cairia
deslocada dela — a mesma duplicata, por outro caminho.
"""

from collections import deque

import cv2
import numpy as np

# Residuo acima do qual a passada e suspeita: ~2,5x o piso da meia
# escala, abaixo do 1,9 px da menor passada duplicadora medida.
MOTION_VETO_PX = 1.0
# Passadas seguidas com o mesmo deslocamento para tratar a folha
# como parada (na cadencia de ~4 passadas/s, ~0,75 s).
MOTION_SETTLE_PASSES = 3
# Concordancia entre as medidas que fecha o "parado": ~o piso.
MOTION_SETTLE_TOL_PX = 0.5
# Maior translacao corrigida: acima disso (~2,5-5 px de camera, alem
# da banda morta de 2 px do rastreador) o quad acompanha a folha.
MOTION_CORRECT_MAX_PX = 4.0
# Deslocamento de uma metade da pagina acima do qual a passada e
# suspeita mesmo com o global parado (rotacao em torno do centro). O
# registro do app ja deixa as metades discordarem: no e5 parado (quad
# reajustado pelo rastreador entre passadas) uma metade mede ate
# 1,5 px com o global em 0,2 px, e a tinta sai certa. Pega rotacao a
# partir de ~0,5 grau (meia pagina andando ~2,5 px).
MOTION_HALF_VETO_PX = 2.5
# Discordancia maxima entre uma metade e o global para tratar o
# movimento como translacao pura: acima do piso do registro (~1,3 px
# no e5 parado) e abaixo da discordancia medida quando a folha girou
# na folha-movida (2,7 px e crescendo).
MOTION_RIGID_TOL_PX = 1.5
# Vetos seguidos ate a passada entrar sem juizo (~5 s na cadencia).
MOTION_FAIL_OPEN_PASSES = 20
_SCALE = 0.5
_HIGHPASS_SIGMA = 12.0  # em px da meia escala (24 px na pagina)
# Abaixo desta fracao de pagina visivel nas duas imagens a medida
# nao e confiavel (mao cobrindo a pagina): a passada segue sem juizo.
_MIN_OVERLAP = 0.3


def _texture(gray_small):
    blur = cv2.GaussianBlur(gray_small, (0, 0), _HIGHPASS_SIGMA)
    return gray_small - blur


def _dft_side(n):
    """Maior lado <= n que o phaseCorrelate nao preenche, com a
    metade tambem sem preenchimento (as quatro metades da pagina)."""
    for side in range(n - n % 2, 1, -2):
        half = side // 2
        if cv2.getOptimalDFTSize(side) == side and cv2.getOptimalDFTSize(half) == half:
            return side
    return n


def _dft_crop(image):
    """Recorta para lados que a DFT do phaseCorrelate usa sem preencher.

    O phaseCorrelate completa a imagem ate getOptimalDFTSize, e o
    preenchimento desloca o pico: duas imagens identicas de 224
    colunas (otimo 225) medem 0,5 px — 1 px de residuo falso na
    meia escala, veto em pagina parada.
    """
    h, w = image.shape[:2]
    return image[: _dft_side(h), : _dft_side(w)]


def _scaled_registration(matrix):
    """A matriz de registro de escala cheia agindo na meia escala."""
    s = np.diag([_SCALE, _SCALE, 1.0])
    return s @ np.asarray(matrix, dtype=np.float64) @ np.linalg.inv(s)


def _phase_shift(reference, texture, overlap):
    """(dx, dy) em px da meia escala, ou None sem pagina visivel bastante."""
    if float(overlap.mean()) < _MIN_OVERLAP:
        return None
    size = (texture.shape[1], texture.shape[0])
    window = cv2.createHanningWindow(size, cv2.CV_32F) * overlap.astype(np.float32)
    # Copias contiguas: o phaseCorrelate escreveu na referencia
    # guardada quando ela chegava como fatia (recorte/metade) — a
    # mesma imagem medida 4 vezes derivava 0,07 -> 0,27 px.
    (dx, dy), _response = cv2.phaseCorrelate(
        np.array(reference, dtype=np.float32, order="C"),
        np.array(texture, dtype=np.float32, order="C"),
        window,
    )
    return float(dx), float(dy)


def translate(image, dx, dy, *, nearest=False):
    """Desloca uma imagem por (dx, dy) px; borda vazia."""
    affine = np.float32([[1, 0, dx], [0, 1, dy]])
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(
        image,
        affine,
        (image.shape[1], image.shape[0]),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


class MotionGate:
    """Mede o deslocamento residual da pagina registrada e decide a passada."""

    def __init__(
        self,
        threshold_px=MOTION_VETO_PX,
        settle_passes=MOTION_SETTLE_PASSES,
        settle_tol_px=MOTION_SETTLE_TOL_PX,
        correct_max_px=MOTION_CORRECT_MAX_PX,
        half_veto_px=MOTION_HALF_VETO_PX,
        rigid_tol_px=MOTION_RIGID_TOL_PX,
        fail_open_passes=MOTION_FAIL_OPEN_PASSES,
    ):
        self.threshold_px = float(threshold_px)
        self.settle_passes = int(settle_passes)
        self.settle_tol_px = float(settle_tol_px)
        self.correct_max_px = float(correct_max_px)
        self.half_veto_px = float(half_veto_px)
        self.rigid_tol_px = float(rigid_tol_px)
        self.fail_open_passes = int(fail_open_passes)
        self.reset()

    def reset(self):
        """Esquece a referencia (geracao nova ou ancora nova)."""
        self._reference = None
        self._reference_visible = None
        self._reference_anchor = None
        self._history = deque(maxlen=max(1, self.settle_passes))
        self._vetoes_in_row = 0

    def measure(self, warped, matrix, visible, anchor):
        """Deslocamento da pagina desta passada contra a referencia.

        Devolve (dx, dy, half_max, spread) em px da grade da ancora, ou
        None sem juizo. (dx, dy) e o deslocamento global; half_max, o
        maior deslocamento entre as quatro metades da pagina (pega
        rotacao); spread, a maior discordancia entre uma metade e o
        global (inf quando alguma metade nao tem pagina visivel
        bastante).

        matrix leva o warp desta passada para a grade da ancora
        (None = identidade); visible e a visibilidade DESTE warp
        (None = pagina toda). A primeira passada medida de uma ancora
        vira a referencia; ancora diferente da guardada recomeca.
        """
        anchor = None if anchor is None else np.asarray(anchor, dtype=np.float32)
        if (
            self._reference_anchor is not None
            and anchor is not None
            and float(np.max(np.abs(anchor - self._reference_anchor))) > 1e-3
        ):
            self.reset()
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
        small = cv2.resize(
            np.asarray(gray, dtype=np.float32),
            None,
            fx=_SCALE,
            fy=_SCALE,
            interpolation=cv2.INTER_AREA,
        )
        size = (small.shape[1], small.shape[0])
        vis = (
            np.ones(small.shape, dtype=np.uint8)
            if visible is None
            else cv2.resize(
                visible.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
            )
        )
        if matrix is not None:
            scaled = _scaled_registration(matrix)
            small = cv2.warpPerspective(
                small,
                scaled,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            vis = cv2.warpPerspective(
                vis, scaled, size, flags=cv2.INTER_NEAREST, borderValue=0
            )
        texture = _dft_crop(_texture(small))
        vis = _dft_crop(vis)
        if self._reference is None:
            self._reference = texture
            self._reference_visible = vis > 0
            self._reference_anchor = anchor
            return None
        overlap = (vis > 0) & self._reference_visible
        shift = _phase_shift(self._reference, texture, overlap)
        if shift is None:
            return None
        dx, dy = shift
        # As quatro metades sempre: rotacao em torno do centro deixa o
        # global perto de zero com as bordas andando px.
        half_max = 0.0
        spread = 0.0
        h, w = texture.shape
        halves = (
            np.s_[:, : w // 2],
            np.s_[:, w // 2 :],
            np.s_[: h // 2, :],
            np.s_[h // 2 :, :],
        )
        for part in halves:
            local = _phase_shift(self._reference[part], texture[part], overlap[part])
            if local is None:
                spread = float("inf")
                continue
            half_max = max(half_max, float(np.hypot(*local)))
            spread = max(spread, float(np.hypot(local[0] - dx, local[1] - dy)))
        return dx / _SCALE, dy / _SCALE, half_max / _SCALE, spread / _SCALE

    def decide(self, measured):
        """("accept" | "veto" | "correct" | "fail_open", deslocamento a desfazer).

        measured e a saida de measure: (dx, dy, half_max, spread) ou None.
        """
        if measured is None:
            return "accept", None
        dx, dy, half_max, spread = measured
        residual = float(np.hypot(dx, dy))
        if residual <= self.threshold_px and half_max <= self.half_veto_px:
            self._history.clear()
            self._vetoes_in_row = 0
            return "accept", None
        self._history.append(np.array([dx, dy], dtype=np.float64))
        if (
            len(self._history) >= self.settle_passes
            and residual <= self.correct_max_px
            and spread <= self.rigid_tol_px
        ):
            stack = np.stack(self._history)
            center = stack.mean(axis=0)
            if float(np.max(np.linalg.norm(stack - center, axis=1))) <= (
                self.settle_tol_px
            ):
                self._vetoes_in_row = 0
                return "correct", (float(center[0]), float(center[1]))
        if self._vetoes_in_row >= self.fail_open_passes:
            return "fail_open", None
        self._vetoes_in_row += 1
        return "veto", None
