"""Registro da evidencia na geometria ancora e limites do voto."""

import cv2
import numpy as np

# Registro da evidencia na geometria ancora da geracao (medido
# na regua desta frente, 2026-11-05, draw-lab/parallel-2026-11-05/
# repassada): o acumulador vive numa grade fixa da pagina, mas a
# geometria que retifica cada passada anda — a cadeia rastreador
# + mediana do app sustenta deslocamentos de varios px por
# dezenas de passadas quando um vies persistente (a mao sobre um
# canto) puxa o quad, e ate o tremor de +-1 px entre passadas
# basta. Sem registro, a evidencia entra na grade na posicao em
# que AQUELE quad a projetou: a janela do voto e de 0,75 s e nao
# distingue "tinta nova" de "a mesma tinta re-projetada 2 px ao
# lado" — cada alinhamento confirma a sua copia e a confirmada
# degenera para a uniao de todas as alinhagens, a tinta assentada
# reimpressa camada sobre camada. Na regua (pagina real assentada
# do cache de 2026-11-01, 300 passadas, nenhum conteudo novo):
# geometria congelada confirma 20.137 px e para; tremor de +-1 px
# confirma 45.528 px; deriva lenta de 3 px, 42.930 px; vies de
# canto sustentado, 71.548 px — contra unioes de 47.472/42.711/
# 76.302 px das mascaras das alinhagens: o voto, sob geometria
# movel, e a uniao antiga com outro nome.
# O registro: a primeira passada da geracao fixa a geometria
# ancora; o mapa de cada passada seguinte e re-projetado para a
# grade da ancora (M = A @ C^-1, homografias camera->pagina da
# ancora e da passada) antes do voto. Mover o caderno vira so
# re-registro: o conteudo confirmado nao anda nem duplica
# (registrar uma vez, adaptar por substituicao). O custo e um
# warpPerspective por passada sobre o mapa float, ~ms contra os
# ~168 ms da passada do modelo; geometria igual a da ancora (o
# caso assentado, com o quad congelado) pula o warp.
_REGISTRATION_IDENTITY_TOL = 1e-4


def _corners_page_matrix(corners, width, height):
    """Homografia camera->pagina de um quad, na grade do mapa.

    Mesma convencao de draw.paper.perspective_matrix (os cantos
    na ordem do rastreador, casando posicionalmente com o destino
    (0,0), (W,0), (W,H), (0,H)), com o destino no tamanho do
    mapa da passada: e nessa grade que ancora e passada
    precisam conversar.
    """
    destination = np.array(
        [(0, 0), (width, 0), (width, height), (0, height)],
        dtype=np.float32,
    )
    source = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(source, destination)


def _permute_prob(prob, k):
    """A prob como o warp posicional a entrega sob rotacao de k x 90
    graus: rot90 + resize de volta ao canonico nos k impares (a
    forma medida na frente 23 — NAO e o registro com cantos
    rolados, que a frente 24 provou nao reconstruir a evidencia
    entregue: escores 0,005 contra 0,23-0,27 da permutacao)."""
    k %= 4
    if k == 0:
        return prob
    if k == 2:
        return np.rot90(prob, 2).copy()
    rotated = np.rot90(prob, 1 if k == 1 else -1)
    return cv2.resize(
        rotated,
        (prob.shape[1], prob.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def _registration_matrix(anchor_corners, corners, shape):
    """M que leva o mapa desta passada para a grade da ancora.

    Um ponto fisico da pagina tem UMA coordenada na ancora:
    p_ancora = A @ C^-1 @ p_passada, com A e C as homografias
    camera->pagina da ancora e desta passada.
    """
    height, width = shape
    anchor = _corners_page_matrix(anchor_corners, width, height)
    current = _corners_page_matrix(corners, width, height)
    return anchor @ np.linalg.inv(current)


def _registration_is_identity(matrix):
    """True se a matriz nao move pixel nenhum (tol sub-pixel)."""
    return bool(
        np.max(np.abs(matrix - np.eye(3, dtype=np.float64)))
        < _REGISTRATION_IDENTITY_TOL
    )


# Protocolo inicial F-10: ausência forte, visível e persistente, com
# testemunho positivo da mesma página. Não confundir apagar tudo com virada.
_REMOVAL_LOW_PROB = 0.1
_REMOVAL_PASSES = 6
_REMOVAL_RETAINED_FRACTION = 0.65
# Piso absoluto de testemunho: a fração de 65% veda a remoção quando
# mais de ~35% do canvas confirmado está velho (cópia defasada após
# migração de âncora, contorno de mão confirmado) — os próprios pixels
# errados impedem a testemunha de existir, trava de catch-22 medida no
# sim-hand (retained 0,33%, canvas congelado com folha limpa à vista).
# Um modelo que re-testemunha >= piso px do canvas está comprovadamente
# vendo tinta: a evidência negativa dele é real. O caso que a fração
# protege (saída do modelo escurecendo por inteiro) continua vedado:
# modelo escuro não retém piso nenhum. Os pixels fora de vista seguem
# imunes pela máscara `visible` dentro de `missing`.
_REMOVAL_WITNESS_FLOOR_PX = 4000


# Disciplina de registro (M1, auditoria de 2026-11-10, draw-lab/
# live-audit-2026-11-10/, provas P2/P3a/P3b): o voto de tinta e
# destruido pelo jitter do quad de registro, e o jitter real e
# evento discreto (flip de modo do detector, pouso de salto), nao
# ruido. Duas correcoes aprovadas em replay, ambas opt-in via InkConfig e
# so sobre o quad que alimenta _registration_matrix (warp do app,
# rastreador e exibicao intocados):
# - InkConfig.reg_anchor_median: a ancora da geracao e a mediana por
#   componente dos cantos das _REG_ANCHOR_MEDIAN_SAMPLES primeiras
#   passadas com quad valido (mediana das que houver antes disso),
#   nao a 1a amostra — um quad espurio na passada 0 nao projeta a
#   pagina inteira no quadro errado.
# - InkConfig.reg_deadband: o registro usa um quad congelado por geracao
#   enquanto o deslocamento maximo de canto contra ele for
#   < _REG_DEADBAND_TAU_PX; >= tau por _REG_DEADBAND_RELEASE_PASSES
#   passadas seguidas libera (o congelado salta para o quad da
#   passada). Spike de 1 passada nunca libera. tau precisa exceder o
#   diametro do jitter assentado: tau=3 foi medido e reprovado
#   (libera no ruido). EMA/One Euro foram reprovados na P3b (a tinta
#   fica para tras quando a pagina anda de verdade).
_REG_ANCHOR_MEDIAN_SAMPLES = 3
_REG_DEADBAND_TAU_PX = 10
_REG_DEADBAND_RELEASE_PASSES = 2


def _median_anchor(samples):
    """Mediana por componente de uma lista de quads (4, 2)."""
    return np.median(np.stack(samples), axis=0).astype(np.float32)


def _deadband_step(frozen, corners, over):
    """Um passo da banda morta: (quad de registro, congelado, over, liberou, desvio).

    over conta as passadas seguidas com desvio >= tau contra o
    congelado; a liberacao so ocorre na _REG_DEADBAND_RELEASE_PASSES-esima.
    """
    shift = float(np.max(np.linalg.norm(corners - frozen, axis=1)))
    if shift < _REG_DEADBAND_TAU_PX:
        return frozen, frozen, 0, False, shift
    over += 1
    if over >= _REG_DEADBAND_RELEASE_PASSES:
        frozen = corners.copy()
        return frozen, frozen, 0, True, shift
    return frozen, frozen, over, False, shift


def _register_ink_probability(
    prob, anchor, corners, visible=None, visible_cam=None, pass_corners=None
):
    """Registra a passada na geometria ancora; devolve (prob, visible).

    `visible` é a máscara já na grade da página; `visible_cam` (issue
    #100) é a mesma máscara na grade da câmera — quando presente, a
    versão registrada sai por UM resample câmera→âncora, em vez do
    NEAREST página→âncora de sempre (dois resamples acumulados por
    pixel). `corners` é o quad de registro (congelado na banda
    morta); `pass_corners` é o quad ao vivo que produziu a prob — a
    máscara percorre o MESMO caminho da tinta (page_matrix da
    passada, depois M), senão drift suprimido desalinha máscara e
    traço em ~tau px (review do PR #118).
    """
    if visible is not None:
        prob = prob.copy()
        prob[~visible] = 0.0
    if anchor is None or corners is None:
        return prob, visible
    matrix = _registration_matrix(anchor, corners, prob.shape)
    if not _registration_is_identity(matrix):
        size = (prob.shape[1], prob.shape[0])
        prob = cv2.warpPerspective(
            prob,
            matrix,
            size,
            flags=cv2.INTER_LINEAR,
            borderValue=0.0,
        )
        if visible_cam is not None:
            visible = (
                cv2.warpPerspective(
                    visible_cam.astype(np.uint8),
                    matrix
                    @ _corners_page_matrix(
                        corners if pass_corners is None else pass_corners,
                        size[0],
                        size[1],
                    ),
                    size,
                    flags=cv2.INTER_NEAREST,
                    borderValue=0,
                )
                > 0
            )
            # A prob so cobre a pegada que M alcanca do quad desta
            # passada; fora dela nao houve observacao. Marcar
            # "visivel" ai contaria tinta velha fora do recorte como
            # ausencia — a folha re-posta noutra pose apagaria o
            # traco (review do PR #118).
            visible &= (
                cv2.warpPerspective(
                    np.ones(prob.shape, np.uint8),
                    matrix,
                    size,
                    flags=cv2.INTER_NEAREST,
                    borderValue=0,
                )
                > 0
            )
        elif visible is not None:
            visible = (
                cv2.warpPerspective(
                    visible.astype(np.uint8),
                    matrix,
                    size,
                    flags=cv2.INTER_NEAREST,
                    borderValue=0,
                )
                > 0
            )
    return prob, visible
