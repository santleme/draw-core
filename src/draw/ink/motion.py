"""Arbitro de registro e veto por movimento residual (InkConfig.motion_veto).

O problema (replay da escrita-folha-movida, 2026-11-20): passadas
carimbaram tinta deslocada 1,9-4 px ao lado da tinta assentada — 39%
da tinta nova do clipe. A causa medida e o proprio registro: ele leva
cada passada para a grade da ancora por M = A @ C^-1 (homografias
camera->pagina da ancora e da passada), o que supoe a folha PARADA e
so a estimativa do quad oscilando. Quando a folha anda de verdade e o
quad acompanha, o warp da passada ja sai alinhado (1,2-1,6 px nas
passadas 5-9) e o registro desfaz o movimento real (3,1-4,0 px depois
dele). Na maioria das passadas o registro acerta (e5 parado: 1,24 ->
0,42 px; mao-sombra: melhora em 15 passadas, piora em 3): ele nao pode
sair, so ser conferido.

A medida: correlacao de fase, bloco a bloco, entre a textura (passa-alta,
meia escala) da pagina desta passada e a referencia da ancora. Votam no
consenso so os blocos com:
  - textura 2D (menor autovalor do tensor de estrutura acima do piso e
    nao muito menor que o maior): papel liso vota (0, 0) ou ruido e
    mascara o deslizamento; pauta so horizontal nao mede X (problema da
    abertura);
  - pagina visivel nas duas imagens fora da mao (mascara dilatada);
  - pico de correlacao confiavel.
Os deslocamentos dos blocos validos ajustam um modelo afim robusto
(rejeita os blocos fora do consenso, como a mao que a mascara nao
cobre), o que pega translacao e rotacao. Poucos blocos validos ou
sem juizo (a passada segue como hoje, nunca veto falso). Blocos de
sobra que discordam entre si sao outra coisa: prova de movimento, e a
mediana deles decide.

Decisao por passada (MotionGate.decide):
  - registrada alinhada (<= MOTION_VETO_PX nos cantos da grade do
    modelo) ou sem juizo: aceita como esta;
  - registrada desalinhada mas o warp SEM registro alinhado: a folha
    andou com o quad; a passada vota na propria geometria;
  - nenhuma alinhada, modelo afim consistente e deslocamento <=
    MOTION_CORRECT_MAX_PX: a passada
    entra corrigida pelo modelo afim da hipotese mais proxima. O modelo
    e medido na MESMA imagem de onde sai a probabilidade da passada,
    entao vale para ela mesmo com a folha andando. A folha que para
    deslocada (o quad congelado pela banda morta de 2 px do rastreador,
    ou pela espera de persistencia de salto que o segurou por 1,75 s na
    folha-movida) e corrigida em toda passada — nao ha deadlock a
    escapar. Re-ancorar foi descartado: a tinta ja confirmada ficaria
    na grade velha e a nova cairia deslocada dela;
  - acima disso, ou sem consenso afim com blocos de sobra (folha
    deformando): veta. Deslocamento desse tamanho o quad acompanha e o
    registro alcanca a folha;
  - trava de vivacidade: MOTION_FAIL_OPEN_PASSES vetos seguidos e a
    passada entra como entraria sem o portao. O acumulador nunca fica
    mudo, e o pior caso e o comportamento de hoje.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Maior deslocamento aceito sem juizo, em px da pagina: ~3x o piso do
# consenso na pagina parada, abaixo do 1,9 px da menor passada
# duplicadora medida.
MOTION_VETO_PX = 1.0
# Maior deslocamento corrigido: acima disso (alem da banda morta do
# rastreador) o quad acompanha a folha e o registro alcanca.
MOTION_CORRECT_MAX_PX = 4.0
# Vetos seguidos ate a passada entrar sem juizo (~5 s na cadencia).
MOTION_FAIL_OPEN_PASSES = 20

_SCALE = 0.5
_HIGHPASS_SIGMA = 12.0  # px da meia escala (24 px na pagina)
_BLOCK = 100  # lado do bloco na meia escala (200 px da pagina)
_HAND_MARGIN = 15  # dilatacao da regiao invisivel, px da meia escala
_BLOCK_MIN_VISIBLE = 0.9
_TEXTURE_MIN = 60.0  # menor autovalor do tensor de estrutura (papel liso ~12)
_TEXTURE_MIN_RATIO = 0.15  # menor/maior: abaixo e textura de uma direcao so
_PEAK_MIN = 0.08  # resposta do phaseCorrelate (identico ~1, ruido ~0,02)
_MIN_BLOCKS = 6
# Sem consenso so vira prova de movimento com a maior parte da grade
# (24 blocos na pagina de 850x1200) visivel e com textura. Medido:
# na folha-movida as passadas que carimbavam a copia tinham 20-23
# blocos validos (8-13 em acordo); na mao-sombra, com a folha parada,
# a mao, a sombra e a escrita nova deixavam 6-10 blocos validos sem
# acordo nenhum (0-1) — ler isso como movimento vetava escrita real.
_MIN_BLOCKS_DISAGREEING = 15
_INLIER_PX = 1.5
_MIN_INLIER_FRACTION = 0.5


@dataclass(frozen=True)
class Motion:
    """Modelo afim do deslocamento da pagina, em px da pagina.

    Um ponto p (px da pagina, origem no centro) aparece deslocado de
    jacobian @ p + shift. max_px e o maior deslocamento do modelo nos
    blocos dos cantos da grade; blocks e inliers contam o consenso.
    """

    jacobian: np.ndarray
    shift: np.ndarray
    center: np.ndarray
    max_px: float
    blocks: int
    inliers: int
    corners: np.ndarray  # deslocamento nos 4 blocos dos cantos da grade
    # False: muitos blocos com textura, mas sem consenso afim (folha
    # deformando, movimento grande). max_px e entao a mediana do
    # deslocamento dos blocos — prova de movimento, nunca correcao.
    consistent: bool = True

    def warp_back(self, image, *, nearest=False):
        """Leva uma imagem desta passada de volta para a grade da referencia."""
        j, t, c = self.jacobian, self.shift, self.center
        matrix = np.hstack([np.eye(2) + j, (t - j @ c).reshape(2, 1)])
        flags = (cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR) | (
            cv2.WARP_INVERSE_MAP
        )
        return cv2.warpAffine(
            image,
            matrix.astype(np.float64),
            (image.shape[1], image.shape[0]),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )


def _texture(gray_small):
    return gray_small - cv2.GaussianBlur(gray_small, (0, 0), _HIGHPASS_SIGMA)


def _scaled_registration(matrix):
    s = np.diag([_SCALE, _SCALE, 1.0])
    return s @ np.asarray(matrix, dtype=np.float64) @ np.linalg.inv(s)


def _structure(block):
    gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
    a, b, c = float(np.mean(gx * gx)), float(np.mean(gx * gy)), float(np.mean(gy * gy))
    root = np.sqrt(max(0.0, ((a - c) / 2) ** 2 + b * b))
    return (a + c) / 2 - root, (a + c) / 2 + root


def _fit_affine(points, shifts):
    """Ajuste afim robusto: parte dos blocos perto da mediana (a mao ou a
    sombra que a mascara nao cobre puxariam o primeiro minimos quadrados
    e a rejeicao descartaria os blocos bons), depois rejeicao iterativa."""
    median = np.median(shifts, axis=0)
    keep = np.linalg.norm(shifts - median, axis=1) <= _INLIER_PX
    full = np.hstack([points, np.ones((len(points), 1))])
    params = None
    for _ in range(4):
        if np.count_nonzero(keep) < 3:
            return None, keep
        params, *_ = np.linalg.lstsq(full[keep], shifts[keep], rcond=None)
        new_keep = np.linalg.norm(full @ params - shifts, axis=1) <= _INLIER_PX
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if np.count_nonzero(keep) < 3:
        return None, keep
    params, *_ = np.linalg.lstsq(full[keep], shifts[keep], rcond=None)
    return params, keep


class MotionGate:
    """Mede o movimento residual da pagina por consenso de blocos e decide."""

    def __init__(
        self,
        threshold_px=MOTION_VETO_PX,
        correct_max_px=MOTION_CORRECT_MAX_PX,
        fail_open_passes=MOTION_FAIL_OPEN_PASSES,
    ):
        self.threshold_px = float(threshold_px)
        self.correct_max_px = float(correct_max_px)
        self.fail_open_passes = int(fail_open_passes)
        self.reset()

    def reset(self):
        """Esquece a referencia (geracao nova ou ancora nova)."""
        self._reference = None
        self._reference_visible = None
        self._reference_anchor = None
        self._vetoes_in_row = 0

    def _prepare(self, warped, matrix, visible):
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
        # A mascara de mao do app cobre so a mao, nao a sombra nem a
        # borda borrada: a regiao invisivel ganha margem.
        hidden = cv2.dilate(
            (vis == 0).astype(np.uint8),
            np.ones((2 * _HAND_MARGIN + 1, 2 * _HAND_MARGIN + 1), np.uint8),
        )
        return _texture(small), hidden == 0

    def _textured_blocks(self, texture, vis):
        h, w = texture.shape
        count = 0
        for y in range(0, h - _BLOCK + 1, _BLOCK):
            for x in range(0, w - _BLOCK + 1, _BLOCK):
                part = np.s_[y : y + _BLOCK, x : x + _BLOCK]
                if vis[part].mean() < _BLOCK_MIN_VISIBLE:
                    continue
                small_eig, big_eig = _structure(
                    np.array(texture[part], dtype=np.float32, order="C")
                )
                if (
                    small_eig >= _TEXTURE_MIN
                    and small_eig >= _TEXTURE_MIN_RATIO * big_eig
                ):
                    count += 1
        return count

    def _align_to_confirmed(self, texture, vis, confirmed_ink, ink_prob):
        """Alinha referencia tardia a tinta ja confirmada; None se nao der."""
        h, w = texture.shape
        ref_ink = cv2.resize(
            confirmed_ink.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA
        )
        cur_ink = cv2.resize(
            ink_prob.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA
        )
        ref_m = np.where(vis, ref_ink, 0.0).astype(np.float32)
        cur_m = np.where(vis, cur_ink, 0.0).astype(np.float32)
        if np.count_nonzero(ref_m > 0.1) < 10 or np.count_nonzero(cur_m > 0.1) < 10:
            return None
        (dx, dy), peak = cv2.phaseCorrelate(ref_m, cur_m)
        shift_px = float(np.hypot(dx, dy)) / _SCALE
        if peak < _PEAK_MIN or shift_px > 20.0:
            return None
        if float(np.hypot(dx, dy)) < 0.1:
            return texture, vis, shift_px
        shift_m = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
        aligned_tex = cv2.warpAffine(
            texture, shift_m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0
        )
        aligned_vis = (
            cv2.warpAffine(
                vis.astype(np.uint8),
                shift_m,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
            > 0
        )
        return aligned_tex, aligned_vis, shift_px

    def measure(
        self,
        warped,
        matrix,
        visible,
        anchor,
        *,
        confirmed_ink=None,
        ink_prob=None,
    ):
        """Motion desta passada contra a referencia, ou None sem juizo.

        warped e a pagina retificada (BGR ou cinza, 0-255: o portao de
        textura e calibrado nessa escala); matrix leva o warp desta
        passada para a grade da ancora (None = sem registro); visible e
        a visibilidade DESTE warp. A primeira passada com blocos de
        textura bastantes de uma ancora vira a referencia (registrada);
        ancora diferente da guardada recomeca. Se o canvas ja tiver
        tinta confirmada quando a referencia tardia qualifica, a
        referencia e alinhada a essa tinta antes de ser guardada.
        """
        anchor = None if anchor is None else np.asarray(anchor, dtype=np.float32)
        if (
            self._reference_anchor is not None
            and anchor is not None
            and float(np.max(np.abs(anchor - self._reference_anchor))) > 1e-3
        ):
            self.reset()
        texture, vis = self._prepare(warped, matrix, visible)
        if self._reference is None:
            if self._textured_blocks(texture, vis) < _MIN_BLOCKS:
                return None
            ref_tex, ref_vis, shift_px = texture, vis, 0.0
            if (
                confirmed_ink is not None
                and ink_prob is not None
                and np.any(confirmed_ink)
            ):
                aligned = self._align_to_confirmed(
                    texture, vis, confirmed_ink, ink_prob
                )
                if aligned is None:
                    return None
                ref_tex, ref_vis, shift_px = aligned
                if self._textured_blocks(ref_tex, ref_vis) < _MIN_BLOCKS:
                    return None
            self._reference, self._reference_visible = ref_tex, ref_vis
            self._reference_anchor = anchor
            if shift_px > self.threshold_px:
                return self._consensus(texture, vis)
            return None
        return self._consensus(texture, vis)

    def _consensus(self, texture, vis):
        ref, ref_vis = self._reference, self._reference_visible
        h, w = texture.shape
        window = cv2.createHanningWindow((_BLOCK, _BLOCK), cv2.CV_32F)
        points, shifts = [], []
        for y in range(0, h - _BLOCK + 1, _BLOCK):
            for x in range(0, w - _BLOCK + 1, _BLOCK):
                part = np.s_[y : y + _BLOCK, x : x + _BLOCK]
                if (
                    vis[part].mean() < _BLOCK_MIN_VISIBLE
                    or ref_vis[part].mean() < _BLOCK_MIN_VISIBLE
                ):
                    continue
                a = np.array(ref[part], dtype=np.float32, order="C")
                b = np.array(texture[part], dtype=np.float32, order="C")
                small_eig, big_eig = _structure(a)
                if small_eig < _TEXTURE_MIN or small_eig < _TEXTURE_MIN_RATIO * big_eig:
                    continue
                # Copias contiguas: o phaseCorrelate escreve na entrada
                # quando ela chega como fatia (medido: a referencia
                # derivava a cada medida).
                (dx, dy), peak = cv2.phaseCorrelate(a, b, window)
                if peak < _PEAK_MIN:
                    continue
                points.append(((x + _BLOCK / 2 - w / 2), (y + _BLOCK / 2 - h / 2)))
                shifts.append((dx, dy))
        if len(points) < _MIN_BLOCKS:
            return None
        points = np.asarray(points, dtype=np.float64) / _SCALE
        shifts = np.asarray(shifts, dtype=np.float64) / _SCALE
        params, keep = _fit_affine(points, shifts)
        inliers = int(np.count_nonzero(keep))
        if params is None or inliers < max(
            _MIN_BLOCKS, _MIN_INLIER_FRACTION * len(points)
        ):
            if len(points) < _MIN_BLOCKS_DISAGREEING:
                return None
            # Sem consenso com blocos de sobra: a folha deformou ou andou
            # demais para um modelo afim (folha-movida, passadas 5-9:
            # ~20 blocos validos, sem consenso — e eram as que
            # carimbavam a copia deslocada). A mediana dos blocos resiste
            # a mao cobrindo ate metade deles.
            median = np.median(shifts, axis=0)
            return Motion(
                jacobian=np.zeros((2, 2)),
                shift=median,
                center=np.array([w / 2, h / 2]) / _SCALE,
                max_px=float(np.median(np.linalg.norm(shifts, axis=1))),
                blocks=len(points),
                inliers=inliers,
                corners=np.tile(median, (4, 1)),
                consistent=False,
            )
        jacobian, shift = params[:2].T, params[2]
        # Pontos fixos (centros dos blocos dos cantos da grade): o
        # mesmo lugar da pagina em toda passada, para comparar modelos.
        nx, ny = w // _BLOCK, h // _BLOCK
        gx = (np.array([0, nx - 1]) * _BLOCK + _BLOCK / 2 - w / 2) / _SCALE
        gy = (np.array([0, ny - 1]) * _BLOCK + _BLOCK / 2 - h / 2) / _SCALE
        corners = np.array(
            [[gx[0], gy[0]], [gx[1], gy[0]], [gx[1], gy[1]], [gx[0], gy[1]]]
        )
        displaced = corners @ jacobian.T + shift
        return Motion(
            jacobian=jacobian,
            shift=shift,
            center=np.array([w / 2, h / 2]) / _SCALE,
            max_px=float(np.max(np.linalg.norm(displaced, axis=1))),
            blocks=len(points),
            inliers=inliers,
            corners=displaced,
        )

    def aligned(self, motion):
        return motion is None or motion.max_px <= self.threshold_px

    def decide(self, registered, unregistered=None):
        """(decisao, Motion, hipotese) desta passada.

        decisao: "accept" | "unregistered" | "correct" | "veto" | "fail_open".
        registered e o Motion da passada registrada; unregistered, o do
        warp sem registro (so medido quando o registrado falha). Em
        "correct", o Motion leva a evidencia da hipotese indicada
        ("registered" ou "unregistered") de volta para a grade da
        referencia: o modelo foi medido na MESMA imagem de onde sai a
        probabilidade da passada, entao vale para ela mesmo com a folha
        andando — e a folha parada fora do lugar e corrigida em toda
        passada, sem deadlock.
        """
        if self.aligned(registered):
            self._vetoes_in_row = 0
            return "accept", None, "registered"
        if unregistered is not None and self.aligned(unregistered):
            self._vetoes_in_row = 0
            return "unregistered", None, "unregistered"
        candidates = [(registered, "registered")]
        if unregistered is not None:
            candidates.append((unregistered, "unregistered"))
        candidates = [c for c in candidates if c[0].consistent]
        best, hypothesis = min(
            candidates, key=lambda c: c[0].max_px, default=(None, None)
        )
        if best is not None and best.max_px <= self.correct_max_px:
            self._vetoes_in_row = 0
            return "correct", best, hypothesis
        if self._vetoes_in_row >= self.fail_open_passes:
            return "fail_open", None, "registered"
        self._vetoes_in_row += 1
        return "veto", None, "registered"
