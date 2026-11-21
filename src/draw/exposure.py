"""Ganho digital da página retificada antes do page_clean.

A 30 fps a exposição do sensor tem teto físico de 1/30 s (UVC -5 =
1/32 s); numa mesa pouco iluminada o papel chega ao modelo em
~80-100 de 255 e o page_clean v2 perde a confiança nos traços leves
— medido ao vivo (2026-11-19, lab/diag_live.py): o NYMES sai
picotado em -5 e inteiro em -3, que só entrega 8 fps. Muitos drivers
(o da mesa de referência incluso) não expõem VideoProcAmp_Gain, então
o ganho que o sensor não dá é aplicado aqui, no warp.

Ganho linear que leva o percentil 90 do papel a _TARGET_P90. Com a
máscara de cobertura da folha, só os pixels da folha entram na conta:
a mesa clara que o quad abrange não cancela o ganho do papel escuro
(review do PR #52). Nunca escurece (ganho >= 1: página bem exposta
passa intacta) e tem teto _MAX_GAIN (frame quase preto não vira ruído
amplificado). O ganho continua linear, mas sua estimativa é suavizada entre
quadros para o ruído do p90 não oscilar a entrada do modelo.

Custo: roda no laço de UI (orçamento de 33 ms a 30 fps). O percentil
sai de uma amostra esparsa (1 pixel a cada _SAMPLE_STRIDE por eixo) e o
ganho se aplica por tabela (cv2.LUT): ~1 ms num warp A4, contra ~24 ms
do cálculo ingênuo sobre todos os pixels.
"""

import math

import cv2
import numpy as np

_TARGET_P90 = 200.0
_MAX_GAIN = 3.0
_SAMPLE_STRIDE = 8
_GAIN_TIME_CONSTANT_SECONDS = 0.35
_GAIN_DEADBAND_LOG = math.log1p(0.03)


def paper_gain(image, mask=None):
    """Ganho que levaria o p90 do papel a _TARGET_P90, em [1, _MAX_GAIN].

    mask (bool, forma da imagem): pixels da folha. Vazia ou None, vale a
    imagem inteira.
    """
    sample = image[::_SAMPLE_STRIDE, ::_SAMPLE_STRIDE]
    gray = sample.mean(axis=2) if sample.ndim == 3 else sample
    if mask is not None:
        sampled = np.asarray(mask)[::_SAMPLE_STRIDE, ::_SAMPLE_STRIDE] > 0
        if sampled.any():
            gray = gray[sampled]
    p90 = float(np.percentile(gray, 90))
    if p90 <= 0:
        return _MAX_GAIN
    return float(np.clip(_TARGET_P90 / p90, 1.0, _MAX_GAIN))


def apply_paper_gain(image, mask=None):
    """(imagem uint8 com ganho aplicado, ganho usado)."""
    gain = paper_gain(image, mask)
    return apply_gain(image, gain), gain


def apply_gain(image, gain):
    """Aplica um ganho linear em [1, _MAX_GAIN] com LUT uint8."""
    gain = float(np.clip(gain, 1.0, _MAX_GAIN))
    if gain == 1.0:
        return image
    table = np.clip(np.arange(256, dtype=np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


class PaperGainSmoother:
    """Amortece pequenas oscilações do ganho estimado entre passadas.

    A banda morta de 3% evita que ruído do p90 altere a LUT. Mudanças
    maiores são seguidas em escala logarítmica, com constante de tempo
    curta para não prender a exposição quando a iluminação realmente muda.
    Alvo 1,0 é bypass, não amortecimento: papel já bem exposto não pode
    esperar a banda morta liberar o resíduo — cada passada ganhada clipa
    o traço claro junto do papel.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._gain = None
        self._updated_at = None

    def update(self, target_gain, timestamp):
        """Retorna o ganho temporalmente estabilizado para `timestamp` (s)."""
        target = float(np.clip(target_gain, 1.0, _MAX_GAIN))
        now = float(timestamp)
        if not math.isfinite(now):
            raise ValueError(f"timestamp precisa ser finito: {timestamp!r}")
        if self._gain is None or self._updated_at is None:
            self._gain = target
            self._updated_at = now
            return self._gain

        elapsed = now - self._updated_at
        if elapsed <= 0.0:
            return self._gain
        self._updated_at = now

        if target == 1.0:
            self._gain = target
            return self._gain

        current_log = math.log(self._gain)
        target_log = math.log(target)
        error = target_log - current_log
        if abs(error) <= _GAIN_DEADBAND_LOG:
            return self._gain

        alpha = -math.expm1(-elapsed / _GAIN_TIME_CONSTANT_SECONDS)
        self._gain = math.exp(current_log + alpha * error)
        return self._gain
