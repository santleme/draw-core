"""Ganho digital da página retificada antes do page_clean.

A 30 fps a exposição do sensor tem teto físico de 1/30 s (UVC -5 =
1/32 s); numa mesa pouco iluminada o papel chega ao modelo em
~80-100 de 255 e o page_clean v2 perde a confiança nos traços leves
— medido ao vivo (2026-11-19, lab/diag_live.py): o NYMES sai
picotado em -5 e inteiro em -3, que só entrega 8 fps. Muitos drivers
(o da mesa de referência incluso) não expõem VideoProcAmp_Gain, então
o ganho que o sensor não dá é aplicado aqui, no warp.

Ganho linear que leva o percentil 90 do warp (o papel: a tinta é
minoria) a _TARGET_P90. Nunca escurece (ganho >= 1: página bem
exposta passa intacta) e tem teto _MAX_GAIN (frame quase preto não
vira ruído amplificado). Linear e por quadro: não muda a ordem dos
níveis, só a escala — o que a rede vê é uma foto mais clara da mesma
página.
"""

import numpy as np

_TARGET_P90 = 200.0
_MAX_GAIN = 3.0


def paper_gain(image):
    """Ganho que levaria o p90 da página a _TARGET_P90, em [1, _MAX_GAIN]."""
    gray = image.mean(axis=2) if image.ndim == 3 else image
    p90 = float(np.percentile(gray, 90))
    if p90 <= 0:
        return _MAX_GAIN
    return float(np.clip(_TARGET_P90 / p90, 1.0, _MAX_GAIN))


def apply_paper_gain(image):
    """(imagem uint8 com ganho aplicado, ganho usado)."""
    gain = paper_gain(image)
    if gain == 1.0:
        return image, gain
    out = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out, gain
