"""Evidencia fotometrica de "papel exposto", independente do MediaPipe.

A remocao de tinta confirmada (draw.ink.accumulator) so pode confiar em
uma regiao quando ela PARECE papel: luminancia compativel com a
referencia da folha e sem anomalia escura de bloco (mao, braco, caneta,
sombra). Incerteza congela a regiao — nunca a libera.

Tudo em blocos (media por INTER_AREA): custo de ~1 passada de resize,
sem float de pagina inteira.
"""

import cv2
import numpy as np

_SAMPLE_STRIDE = 8
_MIN_SAMPLES = 64
# Bloco so e papel se a media dele for >= esta fracao da referencia. Traco
# de caneta cobre poucos % do bloco; mao/sombra derrubam a media em >25%.
_MIN_RATIO = 0.78
_BLOCK_PX = 32
# Reducao previa por area (px por lado) antes do maximo por bloco.
_REDUCE_PX = 4
# Cor: pele tem B/R bem abaixo do papel (que e quase neutro). Tolera
# desvio relativo do B/R do bloco contra a referencia.
_CHROMA_TOL = 0.18


def paper_reference(image, visible=None):
    """(nivel de papel p90, razao B/R do papel) ou None se incerto.

    Amostrado com stride em pixels visiveis. Menos de _MIN_SAMPLES
    amostras => None (a regiao inteira congela).
    """
    sample = image[::_SAMPLE_STRIDE, ::_SAMPLE_STRIDE]
    gray = sample.mean(axis=2) if sample.ndim == 3 else sample.astype(np.float32)
    keep = None
    if visible is not None:
        keep = np.asarray(visible)[::_SAMPLE_STRIDE, ::_SAMPLE_STRIDE] > 0
        if keep.shape != gray.shape or int(np.count_nonzero(keep)) < _MIN_SAMPLES:
            return None
        values = gray[keep]
    else:
        values = gray.ravel()
    if values.size < _MIN_SAMPLES:
        return None
    level = float(np.percentile(values, 90))
    if level <= 1.0:
        return None
    ratio = 1.0
    if sample.ndim == 3 and sample.shape[2] >= 3:
        bright = gray >= level * 0.9
        if keep is not None:
            bright &= keep
        if int(np.count_nonzero(bright)) >= _MIN_SAMPLES // 4:
            blue = float(sample[..., 0][bright].mean())
            red = float(sample[..., 2][bright].mean())
            ratio = blue / red if red > 1.0 else 1.0
    return level, ratio


def paper_exposed(image, reference, *, min_ratio=_MIN_RATIO, block=_BLOCK_PX):
    """Mascara bool HxW: True onde ha evidencia de papel exposto.

    reference e o par de paper_reference; None => tudo False (incerteza
    congela). A estatistica do bloco e o MAXIMO (por canal) do bloco: o
    traco de caneta, mesmo grosso, deixa papel claro no bloco, enquanto
    mao, braco e sombra escurecem ate o pixel mais claro. Bloco com o
    claro abaixo de min_ratio * nivel, ou com B/R fora da tolerancia da
    referencia (pele, sombra colorida), nao e papel.
    """
    height, width = image.shape[:2]
    if reference is None:
        return np.zeros((height, width), dtype=bool)
    level, ratio = reference
    # Sub-amostragem em vista (o traco fino pode sumir da amostra, o que
    # so favorece o maximo) e maximo por bloco por dilatacao ancorada:
    # amostrar [::cells] dela da o maximo exato de cada bloco.
    cells = max(1, block // _REDUCE_PX)
    sample = image[::_REDUCE_PX, ::_REDUCE_PX]
    peak = cv2.dilate(sample, np.ones((cells, cells), np.uint8), anchor=(0, 0))
    peak = peak[::cells, ::cells]
    if peak.ndim == 3 and peak.shape[2] >= 3:
        peak = peak.astype(np.float32)
        gray = peak.mean(axis=2)
        ok = gray >= min_ratio * level
        red = peak[..., 2]
        local = np.divide(peak[..., 0], red, out=np.ones_like(red), where=red > 1.0)
        ok &= np.abs(local - ratio) <= _CHROMA_TOL * max(ratio, 1e-3)
    else:
        ok = peak.astype(np.float32) >= min_ratio * level
    # Fail-closed na borda: bloco so a fracao coberta pela mao ainda passa;
    # erodir 1 bloco congela o vizinho de toda anomalia.
    ok = cv2.erode(ok.astype(np.uint8), np.ones((3, 3), np.uint8), borderValue=1)
    grown = cv2.resize(ok, (width, height), interpolation=cv2.INTER_NEAREST)
    return grown.view(bool)
