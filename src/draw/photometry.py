"""Fotometria determinística da página retificada.

Division normalization: I / GaussianBlur(I) por canal. Anula a
componente suave de iluminação — sombra da mão, vinheta de lente,
gradiente da lâmpada — sem tocar nas frequências altas onde o
traço de tinta mora. O papel plano sai ~branco de referência e a
tinta segue escura; pauta colorida vira traco de contraste normal
para a rede suprimir com supervisão.

É o domínio de entrada do modelo de tinta v4: determinístico,
mesmos parâmetros no treino (draw-training draw.divnorm) e na
inferência. O v2 congelado NÃO passa por aqui — o toggle
`PipelineConfig.ink_divnorm` controla a aplicação e fica
desligado até o modelo v4 chegar.

Medido sem retreino no corpus real (2026-11-18): página de
desenho sob sombra de mão IoU 0,195 -> 0,389.
"""

import cv2
import numpy as np

# ~10% do lado maior separa iluminação (dezenas de cm) de traço e
# pauta (mm). Idêntico ao draw-training draw.divnorm: mudar aqui
# muda o domínio do modelo e exige retreino.
_K_FRAC = 0.10
_EPS = 4.0  # piso pré-divisão: pixel escuro de mesa não explode
_WHITE = 235.0  # referência de papel; headroom para especular


def division_normalize(image):
    """Normaliza a iluminação por divisão, por canal, em uint8."""
    img = image.astype(np.float32) + _EPS
    side = max(image.shape[:2])
    k = max(31, int(round(side * _K_FRAC)) | 1)  # ímpar, >=31 px
    blur = cv2.GaussianBlur(img, (k, k), 0)
    return np.clip((img / blur) * _WHITE, 0, 255).astype(np.uint8)
