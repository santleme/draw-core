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
amplificado). Linear e por quadro: não muda a ordem dos níveis.

Custo: roda no laço de UI (orçamento de 33 ms a 30 fps). O percentil
sai de uma amostra esparsa (1 pixel a cada _SAMPLE_STRIDE por eixo) e o
ganho se aplica por tabela (cv2.LUT): ~1 ms num warp A4, contra ~24 ms
do cálculo ingênuo sobre todos os pixels.
"""

import cv2
import numpy as np

_TARGET_P90 = 200.0
_MAX_GAIN = 3.0
_SAMPLE_STRIDE = 8


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
    if gain == 1.0:
        return image, gain
    table = np.clip(np.arange(256, dtype=np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table), gain
