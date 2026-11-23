"""Trabalhadora pagina->tinta: fila, passadas, cadencia e voto."""

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import config as paper_config
from .. import photometry
from ..contracts import InkBaselineState, InkDelta, InkDisplaySnapshot, InkSnapshot
from . import (
    accumulator,
    identity,
    motion,
    occlusion,
    page_model,
    paper_evidence,
    registration,
    ruling,
    votes,
)
from . import anchor as anchor_mod
from .config import InkConfig, resolve_border_guard_px


def _inset_quad(corners, inset_px):
    """Recua um quadrilatero convexo em `inset_px` para dentro no espaco da camera."""
    pts = np.asarray(corners, dtype=np.float64)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return None
    center = pts.mean(axis=0)
    normals = []
    offsets = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        edge = p1 - p0
        length = float(np.hypot(edge[0], edge[1]))
        if length < 1e-6:
            return None
        normal = np.array([-edge[1] / length, edge[0] / length], dtype=np.float64)
        if float(np.dot(normal, center - p0)) < 0.0:
            normal = -normal
        normals.append(normal)
        offsets.append(float(np.dot(normal, p0)) + float(inset_px))
    inset_pts = np.empty((4, 2), dtype=np.float64)
    for i in range(4):
        prev = (i - 1) % 4
        mat = np.vstack((normals[prev], normals[i]))
        det = float(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
        if abs(det) < 1e-6:
            return None
        inset_pts[i] = np.linalg.solve(mat, np.array([offsets[prev], offsets[i]]))
    if not np.isfinite(inset_pts).all():
        return None
    return inset_pts


def _border_guard_mask(
    shape,
    configured,
    corners=None,
    *,
    min_auto_side=0,
    camera_jitter_px=3.5,
):
    """Mascara booleana da area util na grade retificada (False na faixa de guarda)."""
    h, w = shape[:2]
    base = resolve_border_guard_px(shape, configured, min_auto_side=min_auto_side)
    if base <= 0:
        return None
    mask = np.ones((h, w), dtype=bool)
    mask[:base, :] = False
    mask[-base:, :] = False
    mask[:, :base] = False
    mask[:, -base:] = False
    if configured is None and corners is not None:
        # Quad corrompido (NaN/degenerado): sem dominio para julgar
        # a passada — devolver uma mascara todo-False apagaria o
        # provisorio e contaria retracao falsa; None = a guarda se
        # abstem nesta passada.
        if not np.isfinite(corners).all():
            return None
        inset_cam = _inset_quad(corners, camera_jitter_px)
        if inset_cam is not None:
            # Mesma homografia camera->pagina do warp/registration
            # (destino (0,0)..(W,H)): a grade w-1/h-1 deslocava o
            # inset ~0,5 px e admitia amostras dentro do jitter.
            homography = registration._corners_page_matrix(corners, w, h)
            inset_page = cv2.perspectiveTransform(
                inset_cam.reshape(1, 4, 2).astype(np.float32),
                homography,
            )[0]
            # Rasteriza o inset testando os QUATRO cantos de cada
            # pixel contra os quatro semi-planos: o pixel so entra
            # na area util inteiro. Testar so o centro admitia a
            # amostra de fronteira ~0,5 px de pagina mais perto da
            # borda do que o inset de camera previsto — dentro da
            # faixa de jitter que a guarda existe para excluir.
            xs = np.arange(w, dtype=np.float64)[None, :]
            ys = np.arange(h, dtype=np.float64)[:, None]
            cx, cy = inset_page.mean(axis=0)
            inside = np.ones((h, w), dtype=bool)
            for i in range(4):
                p1 = inset_page[i].astype(np.float64)
                p2 = inset_page[(i + 1) % 4].astype(np.float64)
                ex, ey = p2[0] - p1[0], p2[1] - p1[1]
                ref = ex * (cy - p1[1]) - ey * (cx - p1[0])
                for dx, dy in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)):
                    cross = ex * (ys + dy - p1[1]) - ey * (xs + dx - p1[0])
                    inside &= cross * ref >= 0
            mask &= inside
    if not mask.any():
        # Dominio inspecionado vazio nao e "tudo falhou": a guarda
        # nao pode julgar esta passada, entao se abstem.
        return None
    return mask


# Porta de ociosidade sem mao: no maximo uma sonda a cada
# _IDLE_PROBE_SECONDS. A porta e a presenca da mao, nao um
# detector de mudanca por diff de warp: medido nesta webcam, o
# churn entre quadros parados (1,5-2,0) supera o sinal do lapis,
# entao diff cego dispara a toa ou engole o traco fraco. Escrever
# exige mao no papel; sem mao, a pagina esta assentada e so a
# sonda periodica confere.
_IDLE_PROBE_SECONDS = 1.0


# Sem sinal de mao (clone sem MediaPipe), a porta de ociosidade
# nunca fecha e a trabalhadora rodaria a 250 ms para sempre —
# CPU queimada a toa num app que o dono acha "lento". O descanso,
# nesse caso, vem da tinta parada: depois de
# _NO_HAND_REST_AFTER_STILL_PASSES passadas seguidas sem nenhuma
# mudanca no acumulador (nada confirmou, nada provisoriamente
# mudou), a cadencia cai para a da sonda ate algo mudar. O sinal
# e o do proprio acumulador, nao diff de quadro — o diff cego
# nesta webcam ja foi medido e descartado (churn de exposicao
# empata com o traco). Com sinal de mao presente, nada muda.
_NO_HAND_REST_AFTER_STILL_PASSES = 4


# Descanso total (especificacao do Lucas, 2026-11-01 e de novo
# 2026-11-03): folha assentada e saida confirmada quieta = ZERO
# passadas, nao sonda de 1 Hz para sempre. A sonda de 1 Hz
# seguia alimentando o voto sobre warps que tremem, e cada
# alinhamento de fase confirmava uma camada nova sobre o
# desenho parado (regua de replay, modo hold: salto de +9.999
# px na passada 89). O descanso entra quando a geracao ja rodou
# uma janela de voto inteira (_REST_MIN_GENERATION_PASSES) e as
# ultimas _REST_QUIET_PASSES passadas nao confirmaram nada novo
# — o portao e sobre a SAIDA CONFIRMADA do acumulador, nunca
# sobre diff de quadro (licao de 2026-11-01: o churn da camera
# supera o sinal do lapis). Acorda com evento que anuncia tinta
# possivel: a mao volta ou se mexe (set_hand_presence), a folha
# troca (new_generation) ou a geracao vira. Sem sinal de mao
# (None) o descanso total NAO fecha: nao existe evento que
# anuncie tinta nova, e a sonda de 1 Hz segue sendo o mecanismo
# de acordar desse clone — com a porta de fronteira, ela nao
# acumula mais.
_REST_MIN_GENERATION_PASSES = 3


# Pixels minimos de papel em recuperacao (hold) para re-armar a janela de
# confirmacao pos-saida; abaixo disso e ruido de borda.
_RECOVERY_MIN_PX = 256


_REST_QUIET_PASSES = 2


# Descanso total tambem COM a mao presente (decisao do Lucas,
# 2026-11-03): ele segurando o caderno parado, a cadencia de
# 250 ms seguia eterna. A extensao usa a mesma maquinaria de
# quietude da saida confirmada (_REST_*), mais a mao parada por
# _HAND_REST_STILL_SECONDS. Tinta nova com a mao ja no quadro
# nao se anuncia pela tinta — quem acorda e a mao se mexendo,
# medida na assinatura da mascara que o adaptador ja computa
# (area e centroide), nunca por diff do quadro da pagina (o
# churn da camera supera o sinal do lapis, licao de 2026-11-01).
# Nao ha sonda periodica de tinta neste caminho: sonda e
# passada, e o ponto do descanso e zero passadas.
_HAND_REST_STILL_SECONDS = 1.0


# Evento geometrico que acorda o descanso (frente do cenario E3):
# a premissa do descanso sem mao e "a pagina nao muda" — mas a
# folha que desliza sem mao na cena mostra exatamente o contrario
# (medido em 2026-11-12: o warp congelado seguiu vendo mesa e a
# trabalhadora dormiu no meio do movimento, porque submit nao
# acorda). Um quad
# submetido a mais de _GEO_WAKE_MOVE_PX da ancora e evidencia de
# cena nova — enquanto o deslocamento persistir, a quietude nao
# acumula e a sonda segue; quando a geometria retorna a moldura,
# _REST_QUIET_PASSES recoloca a trabalhadora para dormir.
# 25 px casa com o destrave do historico do laco: abaixo disso o
# rastreador ainda trata como jitter de deteccao.
_GEO_WAKE_MOVE_PX = 25.0


# Cadencia maxima da tinta: uma passada completa a cada ~250 ms.
# A passada custa 168 ms no Ryzen com 8 threads (perfil em
# draw-lab/page-clean-v0/PERF_BASELINE_RYZEN.md), entao quem manda
# no ritmo do canvas e a cadencia, nao o modelo — e o laco
# principal nao paga nenhum dos dois.
_PAGE_CLEAN_INTERVAL_SECONDS = 0.25


# Teto da espera pela trabalhadora de tinta na saida do app.
_PAGE_CLEAN_STOP_TIMEOUT_SECONDS = 5.0


# Watchdog do descanso total: o wait() sem prazo depende de um
# notify chegar — se algum evento se perder entre a avaliacao do
# predicado e o sono, a trabalhadora nunca mais olha a cena. O
# prazo so faz a volta reavaliar _pass_interval; o descanso
# continua decidido pelos predicados, nao pelo relogio.
_REST_WATCHDOG_SECONDS = 10.0


# Publicacoes confirmadas guardadas para poll_delta mesclar: ~10 s a 3
# passadas/s. Alem disso o consumidor atrasado recebe um delta full.
_DELTA_HISTORY = 32


# Tamanho dos historicos de telemetria da trabalhadora: registros
# por passada e eventos de identidade. Replays que medem episodios
# longos passam (None, None) para guardar tudo.
_WORKER_HISTORY = (256, 128)


# Folga do teste de cadencia no pump(): relogio de cena em float
# (indice * 1/30 s) nao pode perder uma passada por um ulp.
_SYNC_CADENCE_EPS_SECONDS = 1e-9


# Fracao de pixels semeados que ainda casam com a propria referencia
# para um tile anomalo ser lido como "conteudo", nao "coberto":
# impressao densa escurece a media do tile mantendo cada pixel na
# aparencia semeada; uma capa ou regua escura leva os pixels de
# papel semeados a `novel` e a fracao despenca bem abaixo disso.
_RETURNED_TILE_MATCH = 0.75


def _fill_page_outside(image, coverage):
    """Fora da folha := média BGR da própria página por canal.

    Mesma regra do laço: a transição folha->preto do warp é uma
    aresta artificial que o page_clean lê como tinta; a cor média
    do papel a apaga sem fabricar uma borda branca. A média usa o
    fancy-indexing do numpy de propósito — mesma ordem de soma da
    receita que o laco aplicava, bit-identica (cv2.mean diverge
    por 1 ulp no truncamento).
    """
    filled = image.copy()
    filled[~coverage] = image[coverage].mean(axis=0).astype(np.uint8)
    return filled


def _bbox_of(mask):
    """(y0, y1, x0, x1) meio-aberto do que e True em mask, ou None."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _provisional_for_display(provisional, min_px):
    """Camada provisoria como a tela a ve (issue #134).

    A mascara interna muda por pixel: uma poeira oscilando em
    volta do limiar republicava o painel a cada passada e ainda
    contava como atividade, segurando a trabalhadora na cadencia
    cheia. A camada exibida so contem os componentes 8-conectados
    com >= min_px pixels — poeira de borda, residuo de pauta e
    oscilacao isolada ficam de fora, sustentados ou nao — e e a
    mudanca desta camada que decide republicar e contar
    atividade, nunca a diff pixel a pixel. Um traco novo entra
    na primeira passada como antes: nesta grade o menor traco de
    lapis ja passa do piso (o mesmo do provisorio na histerese).
    """
    mask = provisional > 0
    out = np.zeros(provisional.shape, dtype=np.uint8)
    bbox = _bbox_of(mask)
    if bbox is None:
        return out
    y0, y1, x0, x1 = bbox
    roi = mask[y0:y1, x0:x1].view(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= min_px
    big[0] = False
    out[y0:y1, x0:x1] = np.where(big[labels], 255, 0).astype(np.uint8)
    return out


def _crop_all_visible(visible, corners):
    """Todo o recorte do quad visivel na mascara?

    Com corners a mascara e de camera (issue #100): so o interior do
    quad conta — mao na mesa fora do recorte nao e folha mascarada
    (review do PR #118). Sem corners a mascara ja e a grade do warp:
    toda ela e recorte.
    """
    if corners is None:
        return bool(np.all(visible))
    crop = np.zeros(visible.shape, np.uint8)
    cv2.fillConvexPoly(crop, np.round(corners).astype(np.int32), 1)
    return bool(visible[crop.astype(bool)].all())


def _warp_exposure_mask(mask, matrix, shape):
    """Materializa escalar e warpa uma máscara bool (INTER_NEAREST)."""
    if isinstance(mask, (bool, np.bool_)):
        mask = np.full(shape, mask, dtype=bool)
    return (
        cv2.warpPerspective(
            mask.astype(np.uint8),
            matrix,
            (shape[1], shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    )


def _exposure_masks_in_geometry(
    assessment, reg_anchor, reg_corners, shape, motion_fix=None, fill_clear=False
):
    """As máscaras fotométricas da guarda na grade de registro pedida.

    A guarda mede o warp desta passada; quem julga identidade e quem
    vota vivem na grade da âncora da geração. O par (âncora, quad de
    registro) tem que ser o MESMO que registrou a prob — na passada de
    re-ancoragem eles já são a âncora nova, e transformar pela âncora
    velha cobre os pixels errados (Devin Review do PR #144). Com
    motion_fix a correção medida acompanha a evidência, como na prob
    registrada. fill_clear materializa o "tudo à vista" do veredito
    quando a guarda é a única fonte de visibilidade da passada: sem
    máscara alguma o acumulador não tem evidência para a remoção, e o
    materializado antes do warp mantém o fora-do-quad mascarado.
    """
    photo_visible = assessment.visible
    exposed = assessment.paper_exposed
    if photo_visible is None and fill_clear:
        photo_visible = np.ones(shape, dtype=bool)
    if reg_anchor is not None and reg_corners is not None:
        matrix = registration._registration_matrix(reg_anchor, reg_corners, shape)
        if not registration._registration_is_identity(matrix):
            if photo_visible is not None:
                photo_visible = _warp_exposure_mask(photo_visible, matrix, shape)
            exposed = _warp_exposure_mask(exposed, matrix, shape)
    if motion_fix is not None:
        if photo_visible is not None:
            photo_visible = (
                motion_fix.warp_back(photo_visible.astype(np.uint8), nearest=True) > 0
            )
        if isinstance(exposed, (bool, np.bool_)):
            exposed = np.full(shape, exposed, dtype=bool)
        exposed = motion_fix.warp_back(exposed.astype(np.uint8), nearest=True) > 0
    return photo_visible, exposed


def _reg_triple(anchor, reg_corners, corners):
    """Normaliza tripla (ancora, registro, submetido) ou None sem geometria."""
    if anchor is None or reg_corners is None:
        return None
    return (
        anchor.copy(),
        reg_corners.copy(),
        None if corners is None else corners.copy(),
    )


def _mask_bbox(mask):
    """(y0, y1, x0, x1) exclusivo dos pixels ligados, ou None.

    Reducoes por eixo em vez de np.nonzero: numa grade A4 com tinta
    espalhada o nonzero completo custava ms por chamada.
    """
    rows = np.flatnonzero(np.any(mask, axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(np.any(mask[rows[0] : rows[-1] + 1], axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _shift_bool_mask(mask, dy, dx):
    if dy == 0 and dx == 0:
        return mask
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    out[src_y0 + dy : src_y1 + dy, src_x0 + dx : src_x1 + dx] = mask[
        src_y0:src_y1, src_x0:src_x1
    ]
    return out


def _align_identity_baseline(identity_baseline, seen, prior, radius, kernel):
    """Alinha a pauta impressa a evidencia independente do prior.

    Devolve (baseline_alinhado, dy, dx): o deslocamento medido da
    pauta e o mesmo da folha — a escrita salva reaparece no quadro
    deslocado junto com ela.
    """
    base_nonprior = identity_baseline if prior is None else identity_baseline & ~prior
    if radius <= 0 or not np.any(base_nonprior) or not np.any(seen):
        return identity_baseline, 0, 0
    active_rows = np.flatnonzero(np.any(base_nonprior, axis=1))
    if active_rows.size == 0:
        return identity_baseline, 0, 0
    base_rows = base_nonprior[active_rows]
    active_cols = np.flatnonzero(np.any(base_rows, axis=0))
    x0 = int(active_cols[0])
    x1 = int(active_cols[-1]) + 1
    h, w = seen.shape
    # A evidencia independente so e lida dentro da banda ao redor do
    # bbox do base (± radius nas duas direcoes): o recorte evita
    # copiar e dilatar a folha inteira — em grade A4 era a parte
    # mais cara do ajuste.
    sy0 = max(0, int(active_rows[0]) - radius)
    sy1 = min(h, int(active_rows[-1]) + radius + 1)
    sx0 = max(0, x0 - radius)
    sx1 = min(w, x1 + radius)
    seen_indep = seen[sy0:sy1, sx0:sx1]
    if prior is not None:
        # ~prior dilatado so difere de `seen` num raio do bbox do
        # prior, recortado ainda mais para dentro da banda.
        prior_box = _mask_bbox(prior)
        if prior_box is not None:
            pm = kernel.shape[0] // 2
            py0 = max(sy0, prior_box[0] - pm)
            py1 = min(sy1, prior_box[1] + pm)
            px0 = max(sx0, prior_box[2] - pm)
            px1 = min(sx1, prior_box[3] + pm)
            if py1 > py0 and px1 > px0:
                # A dilatacao le o prior com a margem do kernel ALEM da
                # banda: tinta salva logo fora do recorte ainda projeta
                # o halo dela para dentro.
                wy0, wy1 = max(0, py0 - pm), min(h, py1 + pm)
                wx0, wx1 = max(0, px0 - pm), min(w, px1 + pm)
                prior_d = (
                    cv2.dilate(prior[wy0:wy1, wx0:wx1].astype(np.uint8), kernel) > 0
                )[py0 - wy0 : py1 - wy0, px0 - wx0 : px1 - wx0]
                seen_indep = seen_indep.copy()
                iy0, ix0 = py0 - sy0, px0 - sx0
                seen_indep[iy0 : iy0 + (py1 - py0), ix0 : ix0 + (px1 - px0)] &= ~prior_d
    if not np.any(seen_indep):
        return identity_baseline, 0, 0
    packed_base_active = np.packbits(base_rows[:, x0:x1], axis=1)
    row_band_mask = np.zeros(h + 2 * radius, dtype=bool)
    for dy in range(-radius, radius + 1):
        row_band_mask[active_rows + (radius + dy)] = True
    band_rows = np.flatnonzero(row_band_mask)
    row_to_band = np.empty(h + 2 * radius, dtype=np.intp)
    row_to_band[band_rows] = np.arange(band_rows.size, dtype=np.intp)
    col_span = x1 - x0 + 2 * radius
    seen_band = np.zeros((band_rows.size, col_span), dtype=bool)
    src_rows = band_rows - radius
    valid_r = (src_rows >= 0) & (src_rows < h)
    valid_band_idx = np.flatnonzero(valid_r)
    valid_src_rows = src_rows[valid_r]
    src_x0 = max(0, x0 - radius)
    src_x1 = min(w, x1 + radius)
    dst_x0 = src_x0 - (x0 - radius)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if valid_band_idx.size > 0 and src_x1 > src_x0:
        seen_band[valid_band_idx, dst_x0:dst_x1] = seen_indep[
            valid_src_rows - sy0, src_x0 - sx0 : src_x1 - sx0
        ]
    span_w = x1 - x0
    packed_shifts = [
        np.packbits(
            seen_band[:, radius + dx : radius + dx + span_w],
            axis=1,
        )
        for dx in range(-radius, radius + 1)
    ]
    rows_center = row_to_band[active_rows + radius]
    best_overlap = int(
        np.bitwise_count(packed_base_active & packed_shifts[radius][rows_center]).sum()
    )
    best_dy, best_dx = 0, 0
    best_dist = 0
    for dy in range(-radius, radius + 1):
        rows_shifted = row_to_band[active_rows + (radius + dy)]
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            overlap = int(
                np.bitwise_count(
                    packed_base_active & packed_shifts[radius + dx][rows_shifted]
                ).sum()
            )
            dist = abs(dy) + abs(dx)
            if overlap > best_overlap or (overlap == best_overlap and dist < best_dist):
                best_overlap = overlap
                best_dy, best_dx = dy, dx
                best_dist = dist
    return _shift_bool_mask(identity_baseline, best_dy, best_dx), best_dy, best_dx


def _evaluate_page_identity(
    identity_reference,
    identity_baseline,
    candidate,
    identity_visible,
    identity_sampled,
    identity_guard,
    hold_seen,
    threshold,
    identity_gray=None,
    identity_model=None,
    identity_gray_ref=None,
):
    """Compara a passada com a referencia de identidade, fora do lock.

    Funcao pura: devolve (keep_page, recall, containment,
    reference_px, visible_reference_px, deferred, quarantine_reason,
    identity_hold). O dominio julgavel e a tinta salva sob a guarda
    do quad de referencia, alcancada pela pegada amostrada e visivel
    sob a mao; marcas de baseline (pauta impressa) sao excluidas dos
    DOIS lados — a mesma folha tipo caderno compartilha a pauta com
    qualquer folha tipo caderno.
    """
    prior = identity_reference > 0
    # Visibilidade do registro antes de a incerteza fotometrica tirar
    # pixels dela: separa tinta coberta pela mao de tinta nao julgavel.
    hand_visible = identity_visible
    has_baseline = (
        identity_baseline is not None
        and identity_baseline.shape == candidate.shape
        and bool(np.any(identity_baseline))
    )
    prior_px = int(np.count_nonzero(prior))
    kernel = np.ones((identity._IDENTITY_DILATION_KERNEL,) * 2, np.uint8)
    radius = identity._IDENTITY_DILATION_KERNEL // 2
    if prior_px == 0:
        seen_any = candidate > threshold
        if has_baseline:
            baseline_d = cv2.dilate(identity_baseline.astype(np.uint8), kernel) > 0
            seen_any = seen_any & ~baseline_d
        if identity_visible is not None:
            seen_any = seen_any & identity_visible
        keep_page = bool(not np.any(seen_any))
        recall = 1.0 if keep_page else 0.0
        return (
            keep_page,
            recall,
            None,
            prior_px,
            None,
            False,
            None,
            False,
        )

    seen = candidate > threshold
    # Evidencia bruta antes da subtracao de pauta: a admissao de
    # re-observacao la embaixo volta a incluir pixels removidos por
    # ~aligned_baseline — mas so onde a aparencia prova escrita.
    seen_raw = seen
    # Marca impressa nao carrega escrita: sai da referencia julgavel
    # E da evidencia vista (alinhada sub-pixel antes de subtrair).
    if has_baseline:
        prior_judgeable = prior & ~identity_baseline
        if not np.any(prior_judgeable):
            prior_judgeable = prior
            prior_excluded = None
        else:
            prior_excluded = prior & ~prior_judgeable
        aligned_baseline, baseline_dy, baseline_dx = _align_identity_baseline(
            identity_baseline, seen, prior_judgeable, radius, kernel
        )
        # Pixel escuro na posicao salva DENTRO da margem impressa so e
        # escrita se estiver mais escuro que a pior aparencia limpa do
        # lugar — a referencia local do PageModel alinhada pelo mesmo
        # deslocamento que a pauta mediu, normalizada pela iluminacao
        # da passada. Escuro sem esse escurecimento local (a pauta
        # vista na posicao do traco, ou traco tao escuro quanto a
        # pauta) nao prova presenca nem ausencia: e ambiguo e sai do
        # dominio julgavel como a pauta coberta — abstencao, nao voto.
        baseline_d = cv2.dilate(aligned_baseline.astype(np.uint8), kernel) > 0
        darker = None
        if (
            identity_gray is not None
            and identity_model is not None
            and identity_model.shape == candidate.shape
        ):
            # Todo consumidor do resultado mora num raio do prior:
            # o ambiguo (subconjunto dele), a banda do ajuste de
            # deslocamento e o re-observado deslocado — a regiao ja
            # inclui esse raio, e o metodo completa com a margem
            # interna (near + |shift|). Fora dela a comparacao nao
            # pode provar nada — e numa folha A4 a varredura inteira
            # custava mais que o resto da avaliacao.
            prior_box = _mask_bbox(prior)
            region = (
                max(0, prior_box[0] - radius),
                min(candidate.shape[0], prior_box[1] + radius),
                max(0, prior_box[2] - radius),
                min(candidate.shape[1], prior_box[3] + radius),
            )
            # Uma normalizacao por passada: o ganho e a parte cara e
            # as duas comparacoes fotometricas usam a mesma.
            identity_norm = identity_model.normalize(identity_gray, identity_visible)
            darker, _ = identity_model.darker_than_reference(
                identity_norm,
                dy=baseline_dy,
                dx=baseline_dx,
                near=radius,
                visible=identity_visible,
                region=region,
                prenormalized=True,
                with_expected=False,
            )
        ambiguous = None
        ambiguous_px = 0
        unsure = occluded = None
        tone_obs = None
        photometric = (
            darker is not None
            and identity_gray_ref is not None
            and identity_gray_ref.shape == candidate.shape
        )
        if photometric and prior_excluded is not None:
            # Com o tom da tinta salva conhecido, o traco sobre a pauta
            # e julgavel como o resto: a fotometria abaixo o separa em
            # presente, ausente provado ou ambiguo. Exclui-lo inteiro
            # deixava a mesma folha mista sem containment (a evidencia
            # admitida sem referencia) e a copia limpa presa no hold.
            prior_judgeable = prior.copy()
            prior_excluded = None
        amb_prior = prior & baseline_d
        if np.any(amb_prior):
            ambiguous = amb_prior & seen_raw
            if darker is not None:
                ambiguous &= ~darker
                if photometric:
                    # Escuro na posicao salva sem escurecer a referencia
                    # local so e indistinguivel quando alguma aparencia
                    # limpa alcancavel reproduz o tom salvo. Tom salvo
                    # distinto de todas: casar com ele prova presenca
                    # (lapis de 70 colado a pauta de 40, que o `darker`
                    # contra a pior vizinha nao admite), nao casar
                    # prova ausencia (a pauta vista onde a tinta
                    # estava).
                    (
                        distinct,
                        tone_obs,
                        unsure,
                        occluded,
                    ) = identity_model.saved_tone_evidence(
                        identity_norm,
                        identity_gray_ref,
                        ambiguous,
                        dy=baseline_dy,
                        dx=baseline_dx,
                        near=radius,
                        visible=identity_visible,
                        region=region,
                        prenormalized=True,
                        with_unsure=True,
                        sampled=identity_sampled,
                    )
                    ambiguous &= ~(distinct | unsure)
                    if np.any(unsure):
                        # Veto que depende de amostra escondida pela mao
                        # nao e ambiguidade de tom: a tinta conta como
                        # escondida — fica no denominador dos pisos de
                        # evidencia (fragmento nao restaura o resto) e
                        # volta a ser julgada quando a mao sai.
                        identity_visible = (
                            ~unsure
                            if identity_visible is None
                            else (identity_visible & ~unsure)
                        )
            ambiguous_px = int(np.count_nonzero(ambiguous))
            if ambiguous_px == 0:
                ambiguous = None
        if ambiguous is not None:
            # Sem `&=`: prior_judgeable pode ser o proprio `prior` (tinta
            # toda sobre a pauta) e mutar zerava a referencia inteira.
            prior_judgeable = prior_judgeable & ~ambiguous
            prior_excluded = (
                ambiguous if prior_excluded is None else prior_excluded | ambiguous
            )
    else:
        prior_judgeable = prior
        prior_excluded = None
        baseline_dy = baseline_dx = 0
        baseline_d = None
        darker = None
        ambiguous = None
        ambiguous_px = 0
        unsure = occluded = None
    # Dominio inspecionado da comparacao: a guarda do quad salvo,
    # cruzada com a visibilidade. Resposta de borda fora da area
    # util nao pode derrubar o containment nem rejeitar a pagina
    # vazia; a guarda do voto final segue amarrada a ancora final.
    inspected = identity_visible
    if inspected is None:
        inspected = identity_guard
    elif identity_guard is not None:
        inspected = inspected & identity_guard
    admitted = None
    if has_baseline:
        # Pele da pauta vazada por alinhamento imperfeito entrava em
        # `seen` e a dilatacao do recall a projetava sobre a escrita
        # escondida — uma folha limpa do mesmo formulario herdava o
        # desenho. Sem referencia local alinhada confiavel (darker
        # ausente) a posicao so nao prova escrita: os pixels da
        # margem impressa saem da evidencia em vez de valer pelo lugar.
        seen = seen & ~baseline_d
        if darker is not None:
            # A evidencia que sustenta o match tem de distinguir
            # escrita de pauta: o pixel exato da tinta salva
            # (re-observado) ou pixel fora do alcance dilatado da
            # marca impressa.
            reobserved = prior
            if baseline_dy != 0 or baseline_dx != 0:
                # A pauta mediu o deslocamento da folha: a escrita
                # salva reaparece no quadro deslocado junto com ela.
                # Um traco fino colado a margem impressa que desceu
                # 1 px cai dentro de baseline_d sem tocar a posicao
                # salva — sem o quadro deslocado a evidencia dele
                # era descartada e a mesma folha votava como nova.
                # Os dois quadros valem: jitter de registro pode
                # deixar o traco entre um e outro.
                reobserved = prior | _shift_bool_mask(prior, baseline_dy, baseline_dx)
            if np.any(darker):
                # A pauta pode nao medir deslocamento nenhum: a
                # marca impressa colada a tinta salva sai do ajuste
                # por prior_d e nao sobra evidencia para transladar.
                # A evidencia local (darker) nao pode ser forjada por
                # pauta — alinhar o proprio traco salvo contra ela
                # mede o deslocamento real que a pauta nao viu,
                # mantendo "uma translacao por folha".
                ink_aligned, ink_dy, ink_dx = _align_identity_baseline(
                    prior, darker, None, radius, kernel
                )
                if ink_dy != 0 or ink_dx != 0:
                    reobserved = reobserved | ink_aligned
            # Posicao salva nao basta: dentro da margem impressa a
            # admissao exige evidencia LOCAL de escrita — o pixel tem
            # de estar mais escuro que a pior aparencia limpa que
            # poderia ocupar o lugar (o `darker` medido acima; o
            # `near` cobre a variacao de borda da marca). A pele da
            # pauta ou a copia limpa mantem a aparencia esperada e
            # ficam de fora; o traco escurece o local e volta a
            # contar — inclusive o tracado inteiro sobre a pauta, que
            # a subtracao de aligned tinha apagado da evidencia, e o
            # traco claro ao lado da pauta, que a comparacao contra
            # o proprio papel admite.
            admitted = reobserved & seen_raw
            if inspected is not None:
                admitted &= inspected
            admitted &= darker if tone_obs is None else (darker | tone_obs)
            seen |= admitted
    # A guarda e estatica: o que ela exclui nao volta a aparecer e
    # sai do denominador da oclusao, que mede so o escondido pela
    # mao DESTA passada — senao a tinta sob a faixa segura a
    # decisao ate o timeout de quarentena.
    prior_guard = (
        prior_judgeable
        if identity_guard is None
        else (prior_judgeable & identity_guard)
    )
    prior_guard_px = int(np.count_nonzero(prior_guard))
    # Dominio sob a guarda SEM descontar ambiguidade nem pauta:
    # o pixel ambiguo (escuro na posicao salva sem escurecer a
    # referencia local) ou coberto pela marca impressa continua
    # podendo conter a tinta — so nao e julgavel NESTA passada.
    # Os pisos que autorizam um veredito medem contra o dominio
    # completo: uma fracao julgavel minuscula nao pode chamar a
    # si mesma de "evidencia completa".
    if identity_guard is None:
        prior_guard_full = prior
        prior_guard_full_px = prior_px
    else:
        prior_guard_full = prior & identity_guard
        prior_guard_full_px = int(np.count_nonzero(prior_guard_full))
    # So pixel realmente amostrado pode ser julgado: a faixa que o
    # registro preencheu de zeros entra em ~visible como se fosse
    # mao, mas nada nasce ali — contar como escondido adiaria a
    # decisao ate o timeout da quarentena resetar a pagina com a
    # folha limpa a vista.
    usable_total_px = (
        candidate.shape[0] * candidate.shape[1]
        if identity_guard is None
        else int(np.count_nonzero(identity_guard))
    )
    if identity_sampled is not None:
        usable_domain = (
            identity_sampled
            if identity_guard is None
            else (identity_sampled & identity_guard)
        )
        usable_px = int(np.count_nonzero(usable_domain))
    prior_domain = (
        prior_guard if identity_sampled is None else (prior_guard & identity_sampled)
    )
    prior_domain_px = int(np.count_nonzero(prior_domain))
    # Tinta no dominio cujo veto limpo depende de amostra escondida
    # pela mao: nao julgavel nesta passada, mas pode ser a tinta salva.
    # Amostra fora da pegada do registro nao conta: a pose nao muda
    # sozinha, e o adiamento geometrico segue sujeito ao timeout.
    occluded_domain_px = (
        0
        if occluded is None or not np.any(occluded)
        else int(np.count_nonzero(occluded & prior_domain))
    )
    unsure_in_domain = unsure is not None and bool(np.any(unsure & prior_domain))
    # Incerteza geometrica (recorte) somada a tinta coberta direto pela
    # mao no dominio: descobrir essa tinta pode bastar para o piso, e
    # o prazo nao pode apagar o desenho antes de a mao sair.
    uncovered_px = (
        0
        if not unsure_in_domain or hand_visible is None
        else int(np.count_nonzero(prior_domain & ~hand_visible & ~unsure))
    )
    # Hold anterior so persiste abaixo do piso enquanto a mao ainda
    # cobre tinta do dominio: veto so geometrico (recorte) com a mao
    # fora nao e oclusao, e o prazo volta a correr nessa pose.
    hand_hidden = hand_visible is not None and bool(
        np.any(prior_domain & ~hand_visible)
    )
    if (
        not hand_hidden
        and unsure_in_domain
        and hand_visible is not None
        and (baseline_dy or baseline_dx)
    ):
        # A tinta inconclusiva tambem e comparada na posicao deslocada
        # com a pauta: a mao ali (dentro da pegada) ainda e oclusao.
        shifted = _shift_bool_mask(unsure & prior_domain, baseline_dy, baseline_dx)
        shifted &= ~hand_visible
        if identity_sampled is not None:
            shifted &= identity_sampled
        hand_hidden = bool(np.any(shifted))
    prior_eval = (
        prior_domain if identity_visible is None else (prior_domain & identity_visible)
    )
    prior_eval_px = int(np.count_nonzero(prior_eval))
    # Marca ambigua (coberta pelo baseline) mas visivel: nao pode
    # carregar atribuicao, entao relaxa o piso de evidencia minima
    # em vez de punir o match pelo que a passada nunca consegue
    # provar.
    ambig_visible_px = (
        0
        if prior_excluded is None
        or identity_visible is None
        or not np.any(prior_excluded)
        else int(np.count_nonzero(prior_excluded & identity_visible))
    )
    if identity_sampled is not None and usable_px < max(
        identity._IDENTITY_MIN_VISIBLE_PX,
        int(identity._IDENTITY_MIN_VISIBLE_FRACTION * usable_total_px),
    ):
        # Pegada fora (ou quase fora) da regiao usavel da ancora:
        # comparar seria decidir sem evidencia — o "match vazio"
        # de uma folha disjunta nao prova mesma pagina. A
        # referencia fica pendente e a quarentena segue o caminho
        # limitado ate o timeout.
        return (False, None, None, prior_px, 0, True, "no_overlap", False)
    if identity_visible is not None and prior_domain_px > 0:
        # Gate precoce de "evidencia suficiente para avaliar", nao de
        # restauracao: a tinta ambigua (sobre pauta) visivel nao pode
        # carregar atribuicao, entao desconta do bound INTEIRO. Um
        # match na fatia clara ainda passa pelo piso estrito la embaixo
        # — mas se entrar, pode estabelecer o hold de oclusao em vez de
        # queimar a quarentena adiando ate o timeout.
        min_eval_px = min(
            prior_guard_full_px,
            max(
                1,
                max(
                    identity._IDENTITY_MIN_VISIBLE_PX,
                    int(identity._IDENTITY_MIN_VISIBLE_FRACTION * prior_domain_px),
                )
                - ambig_visible_px,
            ),
        )
        if prior_eval_px < min_eval_px:
            # Tinta anterior quase toda coberta pela mao — ou
            # indistinguivel por aparencia dentro da margem da
            # pauta: decidir agora seria condenar a pagina pelo que
            # a passada nao ve. Adia; a referencia fica para a
            # proxima passada. Uma vez iniciado o hold por match
            # parcial, a oclusao apertando nao e evidencia nova:
            # queimar o prazo apagaria o desenho que o hold existe
            # para proteger. Ambiguidade fotometrica vale o mesmo:
            # o escuro indistinguivel pode ser a tinta salva —
            # estabelece o hold em vez de emprestar o hold_seen. Idem
            # a tinta bloqueada por amostra escondida: a mao sobre a
            # regra projetada nao pode queimar o prazo.
            return (
                False,
                None,
                None,
                prior_px,
                prior_eval_px,
                True,
                "occlusion",
                (hold_seen and (hand_hidden or not unsure_in_domain))
                or ambiguous_px > 0
                or occluded_domain_px > 0
                or (uncovered_px > 0 and prior_eval_px + uncovered_px >= min_eval_px),
            )
    if prior_eval_px == 0:
        unobserved_px = 0 if identity_sampled is None else usable_total_px - usable_px
        if (
            prior_domain_px == 0
            and prior_guard_px > 0
            and unobserved_px
            > max(
                identity._IDENTITY_MIN_VISIBLE_PX,
                int(identity._IDENTITY_MIN_VISIBLE_FRACTION * usable_total_px),
            )
        ):
            # TODA a tinta da faixa usavel ficou fora da pegada
            # desta passada E a faixa nao observada ainda e
            # substancial: o "vazio" aqui nao prova nem refuta a
            # mesma pagina. Inconclusivo geometrico: adia sem
            # votar e a quarentena limitada resolve pelo timeout.
            return (False, None, None, prior_px, 0, True, "no_overlap", False)
        # Sem tinta salva no dominio inspecionado (canvas vazio ou
        # tinta toda sob a guarda, que e estatica): a comparacao
        # vale so pela evidencia vista.
        seen_any = seen
        if inspected is not None:
            seen_any = seen_any & inspected
        ambiguous_seen = ambiguous is not None and bool(
            np.any(ambiguous if inspected is None else (ambiguous & inspected))
        )
        # Dominio julgavel: a tinta salva sob a guarda e na pegada
        # amostrada. Faixa de borda e faixa nao amostrada nunca
        # reaparecem — nao contam como escondidas para o hold.
        prior_dom = prior_guard_full
        if identity_sampled is not None:
            prior_dom = prior_dom & identity_sampled
        prior_dom_px = int(np.count_nonzero(prior_dom))
        prior_vis = prior_dom
        if identity_visible is not None:
            prior_vis = prior_vis & identity_visible
        prior_vis_px = int(np.count_nonzero(prior_vis))
        containment = None
        if admitted is not None and np.any(admitted) and prior_vis_px > 0:
            # O dominio julgavel esvaziou por ambiguidade, mas a
            # aparencia provou escrita em posicoes salvas: restaura so
            # se essa evidencia cobre a tinta visivel (recall) e
            # explica a tinta vista (containment) — um fragmento
            # coincidente numa folha com outro desenho nao basta.
            # Evidencia insuficiente e inconclusiva, nao voto.
            admitted_d = cv2.dilate(admitted.astype(np.uint8), kernel) > 0
            recall = float(np.count_nonzero(prior_vis & admitted_d)) / prior_vis_px
            seen_any_px = int(np.count_nonzero(seen_any))
            prior_vis_d = cv2.dilate(prior_vis.astype(np.uint8), kernel) > 0
            containment = (
                0.0
                if seen_any_px == 0
                else float(np.count_nonzero(seen_any & prior_vis_d)) / seen_any_px
            )
            if not (
                recall >= identity._PAGE_IDENTITY_MIN_RECALL
                and containment >= identity._PAGE_IDENTITY_MIN_CONTAINMENT
            ):
                # Hold (prazo congelado) so quando parte da tinta salva
                # esta escondida e ainda pode reaparecer. Com ela toda
                # a vista, outra passada igual nao traz evidencia nova:
                # o adiamento queima a quarentena e termina.
                return (
                    False,
                    recall,
                    containment,
                    prior_px,
                    0,
                    True,
                    "occlusion",
                    prior_vis_px < prior_dom_px,
                )
            keep_page = True
        elif ambiguous_seen:
            # Posicao salva escura dentro da margem sem escurecer a
            # referencia local: pode ser a pauta ou a tinta — a
            # passada nao distingue. Nem prova nem refuta: INCONCLUSIVO
            # com hold — a referencia fica protegida em vez de virar
            # match posicional (heranca) ou voto negativo (perda). Tinta
            # nova em outro lugar da folha nao prova que a salva sumiu.
            return (
                False,
                None,
                None,
                prior_px,
                0,
                True,
                "occlusion",
                True,
            )
        elif np.any(seen_any):
            keep_page = False
            recall = 0.0
        else:
            keep_page = True
            recall = 1.0
        hidden_domain = None
        if identity_visible is not None:
            hidden = ~identity_visible
            if identity_sampled is not None:
                hidden = hidden & identity_sampled
            hidden_domain = (
                hidden if identity_guard is None else (identity_guard & hidden)
            )
        if keep_page and hidden_domain is not None and np.any(hidden_domain):
            # "Vazio" nao prova pagina igual enquanto a mao pode
            # esconder tinta do candidato dentro da area
            # inspecionavel — adia SEM queimar a quarentena nem
            # votar: a copia fica recuperavel ate uma passada
            # desobstruida distinguir retorno de substituicao.
            return (
                keep_page,
                recall,
                containment,
                prior_px,
                prior_eval_px,
                True,
                "occlusion",
                True,
            )
        return (
            keep_page,
            recall,
            containment,
            prior_px,
            prior_eval_px,
            False,
            None,
            False,
        )
    if inspected is not None:
        seen = seen & inspected
    # recall e containment medem sobreposicao num raio da referencia:
    # todo o trabalho mora no bbox do prior_eval + a margem do kernel.
    # Dilatar a grade inteira dominava a avaliacao numa folha A4.
    pe_box = _mask_bbox(prior_eval)
    pe_y0 = max(0, pe_box[0] - radius)
    pe_y1 = min(candidate.shape[0], pe_box[1] + radius)
    pe_x0 = max(0, pe_box[2] - radius)
    pe_x1 = min(candidate.shape[1], pe_box[3] + radius)
    seen_d = cv2.dilate(seen[pe_y0:pe_y1, pe_x0:pe_x1].astype(np.uint8), kernel) > 0
    recall = (
        np.count_nonzero(prior_eval[pe_y0:pe_y1, pe_x0:pe_x1] & seen_d) / prior_eval_px
    )
    prior_d = (
        cv2.dilate(prior_eval[pe_y0:pe_y1, pe_x0:pe_x1].astype(np.uint8), kernel) > 0
    )
    seen_px = int(np.count_nonzero(seen))
    containment = (
        0.0
        if seen_px == 0
        else (np.count_nonzero(seen[pe_y0:pe_y1, pe_x0:pe_x1] & prior_d) / seen_px)
    )
    keep_page = bool(
        recall >= identity._PAGE_IDENTITY_MIN_RECALL
        and containment >= identity._PAGE_IDENTITY_MIN_CONTAINMENT
    )
    # O desconto de ambiguidade abriu a banda para a fatia clara
    # tentar match e entrar em hold — mas um mismatch medido com
    # evidencia abaixo do piso SEM desconto nao e prova de troca:
    # tinta nova na fatia vista ou a mao cobrindo o resto explicam
    # a divergencia sem a folha ter mudado. Estabelece o hold de
    # oclusao em vez de votar — como o match parcial, o mismatch
    # coberto tambem segura a quarentena: enquanto a evidencia nao
    # desobstrui, o timeout nao pode destruir a referencia.
    if (
        not keep_page
        and identity_visible is not None
        and prior_domain_px > 0
        and prior_eval_px
        < min(
            prior_guard_full_px,
            max(
                1,
                max(
                    identity._IDENTITY_MIN_VISIBLE_PX,
                    int(identity._IDENTITY_MIN_VISIBLE_FRACTION * prior_domain_px),
                ),
            ),
        )
    ):
        return (
            False,
            recall,
            containment,
            prior_px,
            prior_eval_px,
            True,
            "occlusion",
            True,
        )
    coverage_bound = max(
        1,
        max(
            identity._IDENTITY_MIN_VISIBLE_PX - ambig_visible_px,
            int(identity._PAGE_IDENTITY_MIN_REFERENCE_FRACTION * prior_guard_px),
        ),
    )
    # Sem o rastreio de baseline, o piso e o absoluto da
    # referencia classica: tinta menor que a massa minima nao
    # prova a folha. Com o rastreio ativo — PageModel ligado,
    # mesmo em folha em branco onde o baseline capturado veio
    # vazio — a comparacao ja esta ancorada e o piso nao pode
    # exigir mais do que a massa julgavel inteira: senao uma
    # marca de 20 px deferia ate o timeout e o modelo novo
    # aprenderia o traco dela como pagina.
    baseline_tracked = (
        identity_baseline is not None and identity_baseline.shape == candidate.shape
    )
    coverage_floor = (
        min(prior_guard_px, coverage_bound) if baseline_tracked else coverage_bound
    )
    if keep_page and prior_domain_px < coverage_floor:
        # Fragmento: a pegada alcancou so uma fatia da referencia
        # julgavel — restaurar o resto seria aceitar como prova o
        # que a passada nunca comparou. Inconclusivo geometrico:
        # adia sem votar; o timeout resolve o caso persistente.
        return (
            keep_page,
            recall,
            containment,
            prior_px,
            prior_eval_px,
            True,
            "no_overlap",
            False,
        )
    if keep_page and prior_eval_px < coverage_floor:
        # A geometria alcancou a referencia, mas a mao cobre a
        # maior parte da tinta julgavel: o match na fatia visivel
        # nao distingue a mesma folha de uma substituta cujas
        # marcas divergentes estao escondidas. O piso e sobre a
        # referencia julgavel INTEIRA, nao sobre a fatia
        # alcancada: dois pisos de 50% em cascata restaurariam
        # com so um quarto comparado. Hold de oclusao: segura a
        # decisao sem queimar a quarentena nem votar.
        return (
            keep_page,
            recall,
            containment,
            prior_px,
            prior_eval_px,
            True,
            "occlusion",
            True,
        )
    return (
        keep_page,
        recall,
        containment,
        prior_px,
        prior_eval_px,
        False,
        None,
        False,
    )


@dataclass(slots=True)
class _Pass:
    """Scratch mutavel de uma passada da trabalhadora.

    _run_pass quebrou em estagios nomeados; o que as ~50 variaveis
    locais acumulavam pela funcao agora viaja neste objeto, preenchido
    em ordem: job -> evidencia -> estado de ancora -> registro ->
    identidade/portoes -> voto/publicacao. Tudo continua na mesma
    thread e nos mesmos pontos de lock de antes.
    """

    # job
    warped: object = None
    generation: int = -1
    submitted_at: float = 0.0
    corners: object = None
    visible: object = None
    coverage: object = None
    pass_started_at: float = 0.0
    scheduled_interval: object = None
    scheduled_hand_state: object = None
    captured_at: object = None
    # evidencia
    pass_started: float = 0.0
    pass_ended_at: float = 0.0
    visible_cam: object = None
    coverage_cam: object = None
    coverage_page_static: object = None
    coverage_page: object = None
    coverage_anchor: object = None
    prob: object = None
    paper_ok_page: object = None
    exposure: object = None
    exposure_guard: object = None
    exposure_confirmed_ink: object = None
    exposure_confirmed_anchor: object = None
    # estado de ancora / registro
    first_pass: bool = False
    recovering: bool = False
    anchor: object = None
    anchor_ink: object = None
    registration_corners: object = None
    registered_prob: object = None
    registered_visible: object = None
    camera_prob: object = None
    camera_corners: object = None
    motion_decision: object = None
    motion_fix: object = None
    sampled: object = None
    # identidade (produzida em _pass_identity, lida no restante)
    identity_reference_snap: object = None
    identity_baseline_snap: object = None
    identity_corners_snap: object = None
    identity_hold_seen_snap: bool = False
    identity_precomputed: object = None
    identity_decision: object = None
    identity_recall: object = None
    identity_containment: object = None
    identity_reference_px: object = None
    identity_visible_reference_px: object = None
    identity_deferred: bool = False
    identity_hold: bool = False
    identity_timed_out: bool = False
    identity_switch_pending: bool = False
    identity_geometry_registered: bool = False
    identity_pending: bool = False
    identity_owned: bool = False
    pm_rezero: bool = False
    pm_rezero_applied: bool = False
    keep_page: bool = False
    quarantine_reason: object = None
    quarantine_frames: int = 0
    # portoes / voto
    watch_step_px: object = None
    force_publish: bool = False
    arbitration_abstain: bool = False
    follow_abstain: bool = False
    motion_abstain: bool = False
    used_registered_prob: bool = False
    pass_voted: bool = False
    last_reg: object = None
    pm_epoch: object = None
    confirmed_ink: object = None
    pm_baseline_ink: object = None
    next_baseline_ink: object = None
    paper_ok: object = None
    paper_exposed: object = None
    reg_coverage: object = None
    coverage_matches_last_reg: bool = False
    pm_seeding_remaining: int = 0
    pm_confirm_obs: int = 0
    pm_recovering_px: int = 0
    pm_hold_pending_px: int = 0
    reg_info: dict = field(default_factory=dict)


def _merge_deltas(records):
    """Uniao (OR) dos deltas incrementais de registros contiguos."""
    boxes = [r[3] for r in records if r[3] is not None]
    if not boxes:
        return None, np.zeros((0, 0), np.uint8)
    y0 = min(b[0] for b in boxes)
    y1 = max(b[1] for b in boxes)
    x0 = min(b[2] for b in boxes)
    x1 = max(b[3] for b in boxes)
    merged = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for r in records:
        if r[3] is None:
            continue
        by0, by1, bx0, bx1 = r[3]
        np.maximum(
            merged[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0],
            r[4],
            out=merged[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0],
        )
    return (y0, y1, x0, x1), merged


class PageCleanWorker(anchor_mod._AnchorSteps):
    """Tinta pagina->limpa em thread propria, latest-wins.

    O laco principal entrega o warp mais recente com submit() e
    segue: a fila tem tamanho 1 e um warp nao processado e
    substituido pelo proximo, nunca acumulado. A trabalhadora roda
    passadas completas do modelo com pelo menos
    _PAGE_CLEAN_INTERVAL_SECONDS entre inicios — e nenhuma
    passada em descanso total (folha assentada, saida confirmada
    quieta e mao fora ou parada; _at_rest) — e alimenta o
    accumulator.LayeredInkAccumulator da geracao: a tinta nova sai na hora
    como camada provisoria para a exibicao (poll_display) e so
    vira canvas pelos votos de confirmacao (poll). Publicar cedo
    sem voto era a uniao antiga, que integrava o tremor e o
    mobiliario para sempre; confirmar antes de mostrar era o
    atraso de segundos da cadeia assentada. As duas camadas
    separam as duas perguntas.

    Geracao e o contrato de sessao da folha: folha nova ou folha
    que sai chama new_generation(), que descarta pendente e
    publicado da geracao anterior; a uniao da trabalhadora zera na
    primeira passada da geracao nova. Resultado de geracao velha
    nunca entra no canvas novo, mesmo com uma passada em voo na
    troca. Dentro da geracao, a evidencia de cada passada e
    registrada na geometria da primeira (a ancora) antes do voto:
    o quad que retifica anda entre passadas, e sem o registro cada
    alinhamento confirmava a sua copia da mesma tinta — a
    reimpressao medida na regua de 2026-11-05 (ver
    registration._REGISTRATION_IDENTITY_TOL). Blocos sujos por diff de pixels NAO entram aqui: medido
    nesta webcam, o churn entre quadros consecutivos parados
    (media de |diff| por bloco ~1,5-2,0 niveis; maximo pos-blur
    4-6) empata ou supera o sinal de um traco novo no bloco
    (lapis: maximo pos-blur 6-7), entao pular bloco por diff
    arriscaria engolir justamente o lapis. A economia vem da
    assincronia e da cadencia, nao de pular bloco.
    """

    def __init__(
        self,
        page_clean,
        hand_masker=None,
        clock=time.monotonic,
        history=_WORKER_HISTORY,
        config=None,
        ink_divnorm=False,
    ):
        self._page_clean = page_clean
        self._config = config if config is not None else InkConfig()
        # ink_divnorm (PipelineConfig.ink_divnorm): aplica
        # photometry.division_normalize no warp dentro da passada —
        # o domínio do modelo de tinta v4. A normalização mora aqui
        # porque a passada tem folga de cadência; o laço principal
        # não paga o blur de página inteira (~42 ms/submit, #101).
        self._ink_divnorm = bool(ink_divnorm)
        # Filtro de pauta (InkConfig.ruling_removal, opt-in): modelo em cache
        # por folha; zerado quando a geracao vira (outra folha, outro quad).
        self._ruling = ruling.RulingFilter() if self._config.ruling_removal else None
        # Veto por movimento residual (InkConfig.motion_veto, opt-in):
        # ver draw.ink.motion.
        self._motion = (
            motion.MotionGate(self._config.motion_veto_px)
            if self._config.motion_veto
            else None
        )
        self._motion_generation = None
        self._motion_vetoes_total = 0
        self._motion_corrections_total = 0
        self._motion_fail_open_total = 0
        self._motion_unregistered_total = 0
        self._ruling_generation = None
        # Modelo persistente da pagina limpa (InkConfig.page_model,
        # opt-in): mantido por epoca (canvas_epoch, pm_reset_seq); a
        # trabalhadora o muta sob _pm_gate, e new_generation usa o mesmo
        # gate antes de preservar ou invalidar a epoca.
        self._page_model = None
        self._pm_epoch = None
        self._pm_reset_seq = 0
        self._pm_confirm_obs = page_model._CONFIRM_OBS
        self._pm_baseline_ink = None
        self._identity_baseline = None
        # Re-zero manual (tecla z, #130) pendente, e o ultimo estado
        # publicado do PageModel: (epoca do modelo, fracao semeada,
        # pixels visiveis pendentes).
        self._pm_rezero_requested = False
        # Semeadura pos re-zero em curso: segura o descanso ate o modelo
        # novo semear, senao a pagina parada dorme sem referencia.
        self._pm_rezero_seeding = False
        self._pm_status = None
        # Relogio monotonico injetavel: o padrao e o do processo;
        # replays com pump() passam o relogio da cena.
        self._clock = clock
        self._hand_masker = hand_masker
        self._paper_exposure = occlusion.PaperExposureGuard()
        self._condition = threading.Condition()
        # Serializa o trecho que muta o PageModel e comita a passada
        # contra a troca de geração. Ordem única: _pm_gate -> _condition.
        # submit e os demais leitores usam só _condition; nenhum caminho
        # toma _pm_gate enquanto já segura _condition.
        self._pm_gate = threading.Lock()
        self._thread = threading.Thread(
            target=self._work, name="draw-page-clean", daemon=True
        )
        self._pending = None
        self._generation = 0
        self._published = None
        self._version = 0
        # Historico de deltas confirmados para poll_delta: um registro por
        # publicacao confirmada da geracao corrente (ver _record_delta).
        self._delta_ring = deque(maxlen=_DELTA_HISTORY)
        self._delta_prev = None
        self._stopped = False
        self._error = None
        # Estado privado da trabalhadora (so ela toca):
        self._worker_generation = -1
        # Geometria ancora da geracao: os cantos da primeira
        # passada com cantos. Toda evidencia posterior e
        # registrada nela antes do voto (ver registration._registration_matrix).
        # None enquanto a geracao nao teve passada com cantos —
        # chamadores que nao informam cantos (ferramentas, testes
        # antigos) seguem com a evidencia na propria geometria.
        self._anchor_corners = None
        # Disciplina de registro (InkConfig.reg_*, desligadas por
        # padrao): amostras de cantos da ancora-mediana e o quad
        # congelado da banda morta, ambos por geracao.
        self._anchor_samples = []
        self._reg_frozen_corners = None
        self._reg_deadband_over = 0
        self._reg_deadband_releases = 0
        # Arbitragem de ancora: o evento aberto, o streak de
        # assentamento com seus escores por passada e o snapshot
        # do canvas pre-evento.
        self._arb_event_open = False
        self._arb_streak_quads = []
        self._arb_scores = []
        self._arb_snapshot = None
        self._arb_actions = 0
        self._arb_events = deque(maxlen=128)
        # Mobilidade da ancora: o evento de divergencia
        # assentada, o streak de quads assentados e a retencao
        # por veredito spurious da arbitragem.
        self._follow_event_open = False
        self._follow_streak = 0
        self._follow_streak_quads = []
        self._follow_hold = False
        self._anchor_reanchors = 0
        self._follow_events = deque(maxlen=128)
        # Rotulacao ciclica que a ancora corrente carrega em relacao
        # aos quads posicionais (0 = posicional; k = cantos rolados
        # por k apos um pouso rodado julgado pela arbitragem).
        self._anchor_roll = 0
        # Tripla (ancora, quad de registro, quad submetido) da
        # ultima passada que VOTOU: a ancora sozinha nao diz em
        # que pose a grade estava quando cada evidencia votou —
        # deadband congela o quad de registro (que difere do
        # submetido), pouso/follow trocam a ancora no meio da
        # passada, e passadas abstidas (arbitragem/follow/
        # consenso de identidade) nao depositam nada e ficam com
        # None aqui. Telemetria de leitura (last_registration).
        self._last_registration = None
        # Epoca do conteudo do acumulador: sobe a cada reset —
        # a tinta que sobrevive ao reset e so a que entrou
        # depois dele, entao registros de passadas anteriores
        # deixam de descrever o canvas (eval descarta).
        self._canvas_epoch = 0
        self._identity_reference = None
        self._identity_reference_corners = None
        self._identity_gray_ref = None
        self._identity_recovery_requested = False
        self._identity_defer_count = 0
        self._identity_switch_votes = 0
        self._identity_hold_seen = False
        # Memo da guarda por quad: a ancora fica estavel por longas
        # sequencias e rasterizar a mascara cheia a cada passada,
        # sob o lock, custa arrays da pagina inteira.
        self._guard_memo = {}
        # Degrau entre quads consecutivos (a entrada de cadencia
        # geometrica do follow e do portao de remocao).
        self._previous_corners = None
        self._paper_ref = None
        self._spent_confirm_pass = False
        self._accumulator = accumulator.LayeredInkAccumulator(
            threshold=self._config.seen_threshold, config=self._config
        )
        self._last_pass_start = None
        self._passes_total = 0
        self._telemetry_stale_errors = 0
        self._last_pass_ms = None
        self._provisional_px = 0
        self._confirmed_px = 0
        self._retracted_px_total = 0
        self._published_display = None
        self._display_version = 0
        # A camada provisoria tal como a ultima publicacao de
        # exibicao a mostrou (filtrada por _provisional_for_display):
        # a diff que republica o painel e conta atividade e contra
        # ela, nao contra a mascara interna pixel a pixel (issue
        # #134). None = ainda nada mostrado nesta geracao.
        self._display_provisional = None
        self._hand_state = None
        self._confirm_passes = 0
        self._confirm_masked_skips = 0
        self._hand_exit_submits = 0
        self._awaiting_clear_post_exit = False
        self._still_passes = 0
        # Passadas seguidas com recuperacao de preenchimento
        # pendente e NENHUMA borda de adocao. O teto
        # (_RECOVERY_WAIT_PASSES) nao corre sobre este total — o
        # pendente e global e um streak que nasce quando outro morre
        # herdaria a espera alheia: a porta compara com o teto mais
        # a idade do pendente mais velho, entao o que o teto limita
        # e o tempo morto anterior a ele (Devin Review do PR #144).
        # Zera em qualquer adocao ou quando o pendente some; uma
        # borda nova a reabre.
        self._fill_recovery_waited = 0
        self._marks_recovery_waited = 0
        # Descanso total (so a trabalhadora e o new_generation
        # tocam): passadas desta geracao e passadas seguidas sem
        # confirmar nada novo.
        self._generation_passes = 0
        self._quiet_passes = 0
        # Passadas seguidas sem a saida visivel mudar (confirmada
        # ou a provisoria filtrada da tela, issue #134), contadas
        # SO pelas passadas — nunca zeradas por
        # set_hand_presence, que o laco chama a cada amostra de
        # presenca e zeraria o sinal entre passadas. E o sinal que
        # a cadencia por atividade (frente 92) le; _still_passes
        # nao serve porque a presenca o zera.
        self._activity_quiet_passes = 0
        # Passadas retiradas da fila e descartadas ANTES do modelo
        # por a geracao ter trocado (ou a trabalhadora ter parado)
        # entre a retirada e a execucao: custo de modelo poupado,
        # contado para a telemetria nao confundir com passada.
        self._telemetry_skipped_stale = 0
        # Relogio de quietude da mao (carimbo monotonico do
        # ultimo movimento/aparicao reportado): o descanso com a
        # mao presente exige a mao parada alem de
        # _HAND_REST_STILL_SECONDS. None = nenhum movimento
        # reportado ainda — nao conta como parada.
        self._hand_motion_at = None
        # Estado do evento geometrico (frente do cenario E3):
        # True enquanto o quad submetido segue fora da moldura da
        # ancora — o periodo em que o descanso nao pode fechar.
        self._geo_away = False
        # Telemetria por etapa (U0 do loop v2): contadores e os
        # carimbos das ultimas passadas, para medir onde o tempo
        # submit->publicacao mora sem mudar comportamento nenhum.
        # Quem le usa telemetry_snapshot(); quem escreve e o submit
        # (laco) e a trabalhadora, sempre sob a Condition.
        self._telemetry_submits = 0
        self._telemetry_replaced = 0
        self._telemetry_passes = 0
        self._telemetry_publishes = 0
        self._previous_pass_started_at = None
        self._telemetry_records = deque(maxlen=history[0])
        self._identity_events = deque(maxlen=history[1])

    def start(self):
        self._thread.start()
        return self

    def new_generation(self, preserve_identity=False):
        """Abre geracao; opcionalmente identifica a pagina anterior.

        preserve_identity guarda uma copia do canvas confirmado para
        que a primeira passada possa decidir, por conteudo registrado,
        se uma folha que voltou e a mesma pagina em outra pose.
        """
        with self._pm_gate, self._condition:
            self._identity_recovery_requested = bool(preserve_identity)
            self._identity_defer_count = 0
            self._identity_switch_votes = 0
            self._identity_hold_seen = False
            # Fonte/folha nova (sem identidade preservada) cancela o
            # re-zero pendente: o pedido era sobre a folha anterior. Com
            # identidade preservada o pedido sobrevive: fica bloqueado
            # (p.identity_owned) ate a identidade decidir, e so cancela
            # se a folha for outra.
            if not preserve_identity:
                self._pm_rezero_requested = False
                self._pm_rezero_seeding = False
            self._guard_memo.clear()
            self._previous_corners = None
            self._paper_ref = None
            if preserve_identity:
                if self._identity_reference is None:
                    # MOVER (nao copiar) o desenho confirmado para a
                    # referencia de identidade: enquanto ela estiver
                    # pendente, o acumulador publica so o que esta
                    # geracao confirmar — o desenho anterior nao pode
                    # vazar para o Save/display da folha substituta.
                    # Uma referencia pendente de uma queda anterior
                    # nao e sobrescrita pelo canvas vazio — nem o
                    # baseline impresso que a acompanha.
                    self._identity_reference = self._accumulator._confirmed
                    self._identity_reference_corners = (
                        None
                        if self._anchor_corners is None
                        else self._anchor_corners.copy()
                    )
                    self._identity_baseline = (
                        None
                        if self._pm_baseline_ink is None
                        else self._pm_baseline_ink.copy()
                    )
                    # Nivel de cinza da tinta salva (normalizado): o
                    # que separa "a pauta vista na posicao" de "a
                    # tinta escondida na pauta" — mesma aparencia do
                    # modelo, mesma base da comparacao local. So a
                    # ultima observacao VALIDA de cada pixel: o tom da
                    # mao que cobria o traco na ultima passada nao e o
                    # da tinta (NaN onde nunca foi visto).
                    self._identity_gray_ref = (
                        None
                        if self._page_model is None
                        else self._page_model.observed_normalized_gray.copy()
                    )
                # Toda geracao preservada zera o acumulador, com ou
                # sem captura nova: a referencia pendente e a unica
                # dona do desenho anterior — tinta confirmada numa
                # geracao intermediaria e de outra folha e nao pode
                # fundir no restore. Custo aceito (review): o
                # confirmed restaurado e autoritativo e a janela de
                # votos se reconstroi em poucas passadas; preservar
                # _window/_removal_streak so para esse caso nao
                # compensa a maquina de estados extra.
                self._accumulator.reset()
            else:
                self._identity_reference = None
                self._identity_reference_corners = None
                self._identity_gray_ref = None
                self._identity_baseline = None
                self._pm_baseline_ink = None
                self._pm_reset_seq += 1
            self._generation += 1
            self._anchor_samples = []
            self._reg_frozen_corners = None
            self._reg_deadband_over = 0
            self._arb_event_open = False
            self._arb_streak_quads = []
            self._arb_scores = []
            self._arb_snapshot = None
            self._follow_event_open = False
            self._follow_streak = 0
            self._follow_streak_quads = []
            self._follow_hold = False
            self._anchor_roll = 0
            self._last_registration = None
            self._still_passes = 0
            self._confirm_passes = 0
            self._confirm_masked_skips = 0
            self._hand_exit_submits = self._telemetry_submits
            self._awaiting_clear_post_exit = False
            self._generation_passes = 0
            self._quiet_passes = 0
            self._activity_quiet_passes = 0
            # Troca a guarda por uma instancia fresca (swap atomico sob o
            # lock): uma passada em voo da geracao velha segue no objeto
            # antigo, ainda valido, e tem o resultado descartado no
            # check de geracao. Leitores de pending/recovering nunca
            # veem estado da geracao anterior.
            self._paper_exposure = occlusion.PaperExposureGuard()
            self._fill_recovery_waited = 0
            self._marks_recovery_waited = 0
            self._pending = None
            self._published = None
            self._published_display = None
            self._display_provisional = None
            self._delta_ring.clear()
            self._delta_prev = None
            self._condition.notify_all()
            return self._generation

    def _cached_border_guard(self, shape, corners):
        """Guarda da passada com memo por quad.

        Rasterizar a mascara cheia a cada passada, sob o lock,
        aloca arrays da pagina inteira; a ancora fica estavel por
        longas sequencias, entao a mascara depende so do quad.
        """
        key = (shape, None if corners is None else corners.tobytes())
        if key not in self._guard_memo:
            if len(self._guard_memo) >= 8:
                self._guard_memo.clear()
            self._guard_memo[key] = _border_guard_mask(
                shape,
                self._config.border_guard_px,
                corners,
                min_auto_side=200,
            )
        return self._guard_memo[key]

    def _pass_sampled(self, anchor, corners, shape, live_corners=None):
        """Pegada amostrada do registro, so quando ha consumidor.

        Sem guarda ativa o dominio valido nao a consome — o warp
        da pagina inteira seria descartado (o registro so precisa
        da dele quando recorta a visibilidade, e rasteriza a
        propria nesse caso). Com guarda, ela segue necessaria
        mesmo havendo mascara de visibilidade: a fusao provisoria
        distingue "fora do dominio" (limpa) de "coberto pela mao"
        (preserva), e a visibilidade sozinha confunde os dois
        casos. Computada antes do registro para ser reaproveitada
        no clipe de visibilidade em vez de warp'ada duas vezes.
        """
        if self._config.border_guard_px == 0:
            return None
        return registration._observed_footprint(anchor, corners, live_corners, shape)

    def _identity_gray(self, p):
        """Gray da passada na grade ancora, pelo MESMO warp que
        produziu registered_prob. A comparacao de aparencia da
        re-observacao (pauta limpa vs traco que a escurece) le essa
        luminancia — a prob do modelo de tinta nao distingue
        impressao de escrita quando as duas sao escuras."""
        gray = (
            p.warped
            if p.warped.ndim == 2
            else cv2.cvtColor(p.warped, cv2.COLOR_BGR2GRAY)
        )
        if p.anchor is None or p.registration_corners is None:
            return gray
        matrix = registration._registration_matrix(
            p.anchor, p.registration_corners, p.prob.shape
        )
        if registration._registration_is_identity(matrix):
            return gray
        # Papel (255) na borda: o padding nunca pode parecer tinta
        # mais escura que a pauta no teste de aparencia.
        return cv2.warpPerspective(
            gray,
            matrix,
            (p.prob.shape[1], p.prob.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )

    def _register_into(
        self, anchor, reg_corners, prob, visible, visible_cam, pass_corners=None
    ):
        """Re-registra a passada numa ancora nova e atualiza _last_registration.

        Devolve (prob, visible, sampled) registrados. `pass_corners` e o
        quad ao vivo que produziu a prob: sob banda morta `reg_corners`
        e o congelado, e sem ele a mascara de visibilidade desalinha do
        traco em ~tau px (review do PR #118) — os sitios de
        re-ancoragem o passam sempre.
        """
        sampled = self._pass_sampled(
            anchor, reg_corners, prob.shape, live_corners=pass_corners
        )
        registered_prob, registered_visible = registration._register_ink_probability(
            prob,
            anchor,
            reg_corners,
            visible,
            visible_cam=visible_cam,
            pass_corners=pass_corners,
            sampled=sampled,
        )
        self._last_registration = (
            None if anchor is None else anchor.copy(),
            None if reg_corners is None else reg_corners.copy(),
            None if pass_corners is None else pass_corners.copy(),
        )
        return registered_prob, registered_visible, sampled

    def submit(self, page):
        """Entrega a página retificada mais recente (latest-wins).

        page é o contracts.WarpedPage da passada: image é o warp,
        generation a geração de sessão da folha. corners e o quad
        que retificou ESTE warp (os mesmos
        cantos que o chamador passou ao warp_sheet): a
        trabalhadora registra a evidencia da passada na
        geometria ancora da geracao antes do voto. Cantos fora
        do formato (4, 2) sao tratados como ausentes — um quad
        malformado nao pode derrubar a thread da tinta; sem
        cantos, a passada entra na propria geometria.

        visible e a mascara de visibilidade do FRAME que originou
        este warp (True onde a camera ve a pagina; False sob a
        mao), NA GRADE DA CAMERA quando corners vem junto — a
        passada a retifica para a pagina ou direto para a ancora
        (resample unico, issue #100). Sem cantos nao ha geometria
        de camera: ela entra na grade do proprio warp, como antes.
        Onde ela e False, a passada se abstem: a evidencia
        nao vota, a virada nao julga e a decisao de identidade so
        olha a tinta anterior visivel. Formato nao-2D e tratado
        como ausente, pela mesma razao dos cantos.

        coverage e a cobertura de folha do mesmo FRAME (True onde
        a retificacao cobre papel de verdade; False onde o quad
        admitiu o que esta fora da folha — mesa que entra por
        quad largo ou atrasado depois de movimento), na mesma
        regra de grade do visible. Onde ela e False, a passada so vota
        sobre a tinta que a ancora ja conhece: re-confirmar e
        retrair o que existe, sim; tinta nova nascendo fora da
        folha, nunca. Assim o defeito semantico da mascara
        (traco denso virando "fundo" numa
        pose nova) nao congela a tinta real — ela ja e tinta —
        e a banda de mesa, que nunca foi tinta, segue vedada.
        """
        warped = page.image
        generation = page.generation
        corners = page.corners
        visible = page.visible
        coverage = page.coverage
        if corners is not None:
            try:
                corners = np.asarray(corners, dtype=np.float32)
            except (TypeError, ValueError):
                corners = None
            if corners is not None and corners.shape != (4, 2):
                corners = None
        if visible is not None:
            try:
                visible = np.asarray(visible, dtype=bool)
            except (TypeError, ValueError):
                visible = None
            if visible is not None and visible.ndim != 2:
                visible = None
        if coverage is not None:
            try:
                coverage = np.asarray(coverage, dtype=bool)
            except (TypeError, ValueError):
                coverage = None
            if coverage is not None and coverage.ndim != 2:
                coverage = None
        with self._condition:
            if self._stopped or generation != self._generation:
                return
            if corners is not None and self._anchor_corners is not None:
                # Evento geometrico: o quad deste warp saiu da
                # moldura da ancora (folha que se moveu de
                # verdade, nao jitter — quem envia aqui ja e a
                # geometria estabilizada do laco). Enquanto o
                # deslocamento persistir, a premissa do descanso
                # ("pagina parada, nada para ver") esta refutada:
                # a vista pode ter evidencia que o canvas nunca
                # contou. A quietude nao acumula — a sonda fica no
                # regime de 1 Hz do modo sem sinal de mao — e o
                # descanso so volta quando a geometria retorna a
                # moldura ou a saida prova que nao ha nada novo.
                # A comparacao e POSICIONAL, como a dos gatilhos
                # da arbitragem: uma ancora com rotulos rolados
                # (_anchor_roll, de um pouso rodado) nao pode
                # fazer toda passada seguinte parecer deslocada
                # pela permutacao — e o deslocamento de POSE que
                # afasta, nao o da rotulacao.
                anchor_positional = (
                    self._anchor_corners
                    if self._anchor_roll == 0
                    else np.roll(self._anchor_corners, -self._anchor_roll, axis=0)
                )
                away = (
                    float(np.max(np.linalg.norm(corners - anchor_positional, axis=1)))
                    > _GEO_WAKE_MOVE_PX
                )
                if away:
                    self._quiet_passes = 0
                    self._activity_quiet_passes = 0
                    self._still_passes = 0
                self._geo_away = away
            if self._pending is not None:
                # Latest-wins: um warp pendente substituido nunca
                # vira passada; a telemetria conta a substituicao
                # porque ela e atraso real de atualizacao.
                self._telemetry_replaced += 1
            self._telemetry_submits += 1
            if (
                self._hand_state is False
                and self._awaiting_clear_post_exit
                and self._post_exit_confirm_passes() > 1
                and (visible is None or _crop_all_visible(visible, corners))
            ):
                # Primeira submissao pos-saida com visibilidade limpa (sem
                # mascara de mao velha): re-arma a janela completa de
                # confirmacao a partir deste quadro.
                self._awaiting_clear_post_exit = False
                self._confirm_passes = self._post_exit_confirm_passes()
            self._pending = (
                warped,
                generation,
                self._clock(),
                corners,
                visible,
                coverage,
                page.captured_at,
            )
            self._condition.notify_all()

    def _post_exit_confirm_passes(self):
        """Passadas de confirmacao apos a saida da mao.

        No voto legado (vote_score_strong <= 1,9 e sem histerese), duas
        leituras fortes confirmam (uma antes da saida + uma depois),
        entao basta 1 passada. Com voto estrito ou histerese (ex.:
        InkConfig.production, onde vote_score_strong = 2,55 e t_high =
        0,90 sobre a media de 3 passadas), um traco forte exige 3
        leituras na janela: a saida da mao arma uma janela completa
        (_INK_VOTE_WINDOW) para fechar o traco antes do descanso total.
        Perfis com post_exit_full_window (ex.: sketch, voto legado sem
        histerese) tambem armam a janela: a passada unica so fecha um
        traco ja lido com a mao presente, e um traco revelado so depois
        da saida precisa das leituras restantes (Devin Review PR #122).
        """
        passes = (
            votes._INK_VOTE_WINDOW
            if (
                self._config.vote_score_strong > votes._INK_VOTE_SCORE_STRONG
                or self._config.hysteresis
                or self._config.post_exit_full_window
            )
            else 1
        )
        if self._config.page_model:
            passes = max(passes, self._pm_confirm_obs)
        return passes

    @property
    def _confirm_once(self):
        return self._confirm_passes > 0

    @_confirm_once.setter
    def _confirm_once(self, value):
        self._confirm_passes = self._post_exit_confirm_passes() if value else 0
        self._confirm_masked_skips = 0
        self._awaiting_clear_post_exit = bool(value)

    def rezero_baseline(self):
        """Agenda o re-zero da referencia (InkConfig.page_model, #130).

        A proxima passada que vota fora da recuperacao de identidade
        zera o canvas e o PageModel: o conteudo a vista passa a ser a
        referencia, semeada de novo. Devolve False sem page_model.
        """
        if not self._config.page_model:
            return False
        with self._condition:
            if self._stopped:
                return False
            self._pm_rezero_requested = True
            self._quiet_passes = 0
            self._still_passes = 0
            self._condition.notify_all()
            return True

    def baseline_state(self):
        """Prontidao da referencia inicial para a interface (#130)."""
        if not self._config.page_model:
            return InkBaselineState(False, False, 0.0, 0, "disabled")
        with self._condition:
            if self._pm_rezero_requested:
                return InkBaselineState(True, False, 0.0, 0, "recapturing")
            status = self._pm_status
            if status is None or status[0] != (self._canvas_epoch, self._pm_reset_seq):
                return InkBaselineState(True, False, 0.0, 0, "seeding")
            _, fraction, pending_px = status
            ready = fraction > 0.0
            phase = (
                "seeding" if not ready else "ready_partial" if pending_px else "ready"
            )
            return InkBaselineState(True, ready, fraction, pending_px, phase)

    def set_hand_presence(self, present, moved=False):
        """Alimenta a porta de ociosidade: True, False ou None.

        A transicao para False arma a janela de confirmacao pos-saida
        (1 passada no legado; _INK_VOTE_WINDOW no voto estrito/histerese):
        a mao saiu, a pagina assentou, e essas passadas fecham o voto do
        que foi escrito. Depois delas, sem mao, a trabalhadora sonda a
        no maximo 1 Hz enquanto a saida confirmada ainda muda — e entra
        em descanso total (zero passadas) quando ela para de mudar; ver
        _at_rest.

        moved=True e o sinal de atividade da mascara (ver
        _MediaPipeHandPresence.sample): reseta o relogio de
        quietude da mao e acorda o descanso com a mao presente.
        Transicao de presenca tambem conta como movimento: a mao
        chegando ou saindo nunca e mao parada.
        """
        with self._condition:
            if present is False and self._hand_state is not False:
                self._confirm_passes = self._post_exit_confirm_passes()
                self._confirm_masked_skips = 0
                self._hand_exit_submits = self._telemetry_submits
                self._awaiting_clear_post_exit = True
            elif present is not False and self._hand_state is False:
                # A janela pos-saida so e cancelada pela TRANSICAO
                # que encerra o episodio de mao fora (a mao de
                # volta ou o sinal perdido): a premissa "pagina
                # assentada sem mao" morreu. Uma amostra repetida
                # com a mao ja presente nao muda estado nenhum —
                # o laco chama isto a cada amostra de presenca, e
                # zerar aqui apagava o rearma da adocao de
                # preenchimento (_pass_anchor_state) na amostra
                # seguinte, antes de o preenchimento votar (Devin
                # Review do PR #144).
                self._confirm_passes = 0
                self._confirm_masked_skips = 0
                self._awaiting_clear_post_exit = False
            if present is not None:
                self._still_passes = 0
            if present is not None and (moved or present != self._hand_state):
                self._hand_motion_at = self._clock()
                if (
                    moved
                    and self._config.pass_interval_s is not None
                    and self._config.pass_interval_s > _PAGE_CLEAN_INTERVAL_SECONDS
                ):
                    self._quiet_passes = 0
            self._hand_state = present
            self._condition.notify_all()

    def _hand_still(self):
        """Mao presente e sem movimento alem da janela de quietude."""
        return (
            self._hand_motion_at is not None
            and self._clock() - self._hand_motion_at >= _HAND_REST_STILL_SECONDS
        )

    def _recovery_wait_open(self):
        """Espera por recuperacao pendente ainda dentro do prazo.

        O teto _RECOVERY_WAIT_PASSES cobre o tempo MORTO: as
        passadas pendentes anteriores ao item vivo mais velho,
        gastas em streaks que morreram no caminho. O pendente mais
        velho recebe a propria idade como carencia — um streak que
        so fica a vista tarde ainda completa o horizonte de adocao,
        em vez de herdar a espera alheia e ter a contagem cortada
        antes da leitura que o adota (Devin Review do PR #144).
        Como a idade pendente e sempre menor que o teto, a espera
        total fica limitada a 2*_RECOVERY_WAIT_PASSES - 1: o
        vaivem de sombra que nunca adota segue fechando a porta.

        Com a Condition na mao (le _paper_exposure e
        _fill_recovery_waited).
        """
        return (
            self._paper_exposure.recovery_pending
            and self._fill_recovery_waited
            < occlusion._RECOVERY_WAIT_PASSES + self._paper_exposure.max_recovery_age
        )

    def _marks_wait_open(self):
        """Allow a clean position veto to expire and cast its first vote.

        Age credit lets a late tile finish without allowing an endless
        chain of different pending tiles: total wait is below twice the
        marks horizon, independently of the fill recovery budget.
        """
        age = self._paper_exposure.max_marks_recovery_age
        return age > 0 and self._marks_recovery_waited < occlusion._MARKS_HOLD_MAX + age

    def _removal_pending(self):
        """Removal votes still need observations at the active cadence.

        O streak que não se move em toda uma janela de remoção — a
        sombra estável que nunca prova papel, por exemplo — não é
        trabalho urgente: como as carências de recovery e marks, a
        espera é limitada e o pendurado caduca depois de
        _REMOVAL_PASSES passadas sem progresso (Devin Review do
        PR #144)."""
        return (
            self._removal_outstanding()
            and self._accumulator._removal_stale_passes < registration._REMOVAL_PASSES
        )

    def _removal_outstanding(self):
        """Voto de remoção existe — fresco ou caducado.

        A caducidade do _removal_pending só rebaixa a cadência (de
        ativa para a sonda ociosa); não apaga o trabalho. Enquanto
        houver voto por resolver, o descanso total fica proibido: o
        loop em repouso ignora frames novos — a sombra sair sozinha
        não muda mão nem geometria — e a prova de papel que destrava
        a contagem nunca chegaria ao acumulador, deixando a tinta
        apagada congelada no canvas (Devin Review do PR #144)."""
        return self._accumulator._removal_streak is not None and np.any(
            self._accumulator._removal_streak
        )

    def _at_rest(self):
        """Descanso total: nenhuma passada ate um evento acordar.

        Sem mao: sinal presente e False, passada de confirmacao
        da saida da mao ja gasta, janela de voto da geracao
        completa e as ultimas passadas sem confirmar nada novo.
        Com mao (decisao do Lucas, 2026-11-03): a mesma quietude
        da saida confirmada, mais a mao parada alem de
        _HAND_REST_STILL_SECONDS — acorda com o movimento da
        propria mao, reportado pela assinatura da mascara.
        Ver _REST_MIN_GENERATION_PASSES/_REST_QUIET_PASSES.
        """
        if self._removal_outstanding():
            return False
        if self._marks_wait_open():
            return False
        if self._recovery_wait_open():
            # Streak de evidência de tinta contando num bloco
            # anômalo: as passadas que a guarda precisa para adotar
            # o preenchimento não podem contar como quietude —
            # dormir no meio da contagem congelava o traco antes de
            # ele votar. O prazo corre por item vivo (ver
            # _recovery_wait_open): pendente por mais passadas que
            # o teto mais a carencia e oscilacao de sombra, nao
            # preenchimento — segurar a porta aí deixava a
            # inferencia rodando sem fim depois da saida da mao
            # (Devin Review do PR #144).
            return False
        if self._identity_recovery_requested and self._identity_reference is not None:
            return False
        if self._pm_rezero_requested or self._pm_rezero_seeding:
            return False
        if (
            self._generation_passes < _REST_MIN_GENERATION_PASSES
            or self._quiet_passes < _REST_QUIET_PASSES
        ):
            return False
        if self._hand_state is False:
            return not self._confirm_once
        if self._hand_state is True:
            # A janela rearmada pela adocao tambem fala pela mao
            # parada: o preenchimento recem-adotado precisa das
            # passadas restantes para fechar o voto com a mao
            # segurando a folha (Devin Review do PR #144).
            return self._hand_still() and not self._confirm_once
        return False

    def _pass_interval(self):
        if self._at_rest():
            return None
        # Carencia de recuperacao anda na cadencia ativa: manter a
        # porta aberta mas amostrar uma vez por segundo esticava a
        # espera limitada muito alem do horizonte de adocao —
        # passadas, nao segundos, e o que a guarda conta
        # (Devin Review do PR #144).
        if (
            self._hand_state is False
            and not self._confirm_once
            and not self._recovery_wait_open()
            and not self._marks_wait_open()
            and not self._removal_pending()
        ):
            return _IDLE_PROBE_SECONDS
        if (
            self._hand_state is None
            and self._still_passes >= _NO_HAND_REST_AFTER_STILL_PASSES
            and not self._recovery_wait_open()
            and not self._marks_wait_open()
            and not self._removal_pending()
        ):
            return _IDLE_PROBE_SECONDS
        # Voto caducado sob mao parada: o descanso total e proibido
        # (o voto existe), mas ele nao e trabalho urgente — sem o
        # desvio para a sonda, o ativo de 250 ms rodava para sempre
        # sob a sombra estavel com a mao pousada (Devin Review do
        # PR #144).
        if (
            self._hand_state is True
            and self._hand_still()
            and self._removal_outstanding()
            and not self._confirm_once
            and not self._recovery_wait_open()
            and not self._marks_wait_open()
            and not self._removal_pending()
        ):
            return _IDLE_PROBE_SECONDS
        interval = (
            self._config.pass_interval_s
            if self._config.pass_interval_s is not None
            else _PAGE_CLEAN_INTERVAL_SECONDS
        )
        # Um voto de remoção pendente não pode esperar um
        # --ink-interval lento: a cadência urgente é o probe ocioso
        # (Devin Review do PR #144).
        if self._removal_pending():
            return min(interval, _IDLE_PROBE_SECONDS)
        return interval

    def _announce_rest_locked(self):
        """Entrada do descanso total: a tela assenta numa unica troca.

        O provisorio ainda visivel neste ponto e tinta que a
        janela de voto nao confirmou — poeira, nao traco (o traco
        real confirma antes da quietude fechar). Ele sai da camada
        exibida nesta publicacao unica por entrada: o painel para
        na saida confirmada em vez de congelar exibindo o ultimo
        provisorio para sempre (issue #134). Sem provisorio para
        limpar nenhuma versao sobe, entao a transicao para o
        descanso nao pisca. Idempotente por construcao: depois da
        limpeza a camada publicada esta vazia e, sem passadas no
        descanso, ninguem a repoe. So e chamada com a Condition na
        mao, no ramo em que _pass_interval devolveu descanso.
        """
        published = self._published_display
        if (
            published is None
            or published[0] != self._generation
            or not np.any(published[3])
        ):
            return
        self._display_version += 1
        settled = np.zeros_like(published[3])
        self._published_display = (
            published[0],
            self._display_version,
            published[2],
            settled,
        )
        self._display_provisional = settled
        self._condition.notify_all()

    def telemetry_snapshot(self):
        """Copia da telemetria por etapa (somente leitura).

        Contadores desde a criacao da trabalhadora e, por passada
        recente, os carimbos monotonicos e as duracoes derivadas:
        queue_wait_s (submit->inicio da passada; inclui a espera
        da cadencia de _PAGE_CLEAN_INTERVAL_SECONDS), pass_s
        (inicio->fim do modelo) e submit_to_publish_s (submit->
        fim da passada que decide a publicacao). Cada registro
        traz tambem interval_s (o intervalo que governou a espera
        daquela passada; None na primeira ou na acordada do
        descanso), hand_state (o sinal de presenca na retirada) e
        activity_quiet_passes (quietude da saida ao fim da
        passada) — sem esses tres, o orcamento de cadencia nao
        distingue sonda ociosa de cadencia ativa (frente 92). Na reconciliacao
        com a orquestracao em camadas, passes conta cada passada
        do modelo e publishes conta so as passadas que publicam
        canvas confirmado novo no poll(). Eventos de identidade de
        pagina ficam numa lista separada para sobreviverem ao buffer
        curto de passadas. Nenhum campo muda comportamento.
        """
        with self._condition:
            snapshot = {
                "submits": self._telemetry_submits,
                "submits_replaced": self._telemetry_replaced,
                "passes": self._telemetry_passes,
                "publishes": self._telemetry_publishes,
                "passes_skipped_stale": self._telemetry_skipped_stale,
                "passes_stale_errors": self._telemetry_stale_errors,
                "records": [dict(r) for r in self._telemetry_records],
                "page_identity_events": [
                    dict(event) for event in self._identity_events
                ],
            }
            if self._config.reg_deadband or self._reg_deadband_releases:
                snapshot["reg_deadband_releases"] = self._reg_deadband_releases
            snapshot["anchor_arbitration_actions"] = self._arb_actions
            snapshot["anchor_arbitration_events"] = [
                dict(event) for event in self._arb_events
            ]
            snapshot["anchor_reanchors"] = self._anchor_reanchors
            snapshot["anchor_reanchor_events"] = [
                dict(event) for event in self._follow_events
            ]
            return snapshot

    def poll(self, generation, after_version):
        """InkSnapshot se ha publicado novo da geracao depois de
        after_version; senao None.

        Este e o canal dos consumidores de canvas (snapshot,
        cursor, Paint, ponte): so tinta confirmada passa por aqui.
        A camada provisoria nunca sai por este poll.
        """
        with self._condition:
            published = self._published
            if (
                published is None
                or published[0] != generation
                or published[1] <= after_version
            ):
                return None
            return InkSnapshot(version=published[1], confirmed=published[2])

    def _record_delta(self, generation, version, confirmed, captured_at, anchor):
        """Calcula e guarda o delta desta publicacao (thread da tinta, fora
        do lock da passada; so o append pega o lock)."""
        now = confirmed > 0
        prev = self._delta_prev
        if prev is not None and (prev[0] != generation or prev[1].shape != now.shape):
            prev = None
        first = prev is None
        removed_px = 0
        anchor_moved = False
        if not first:
            removed_px = int(np.count_nonzero(prev[1] & ~now))
            old_anchor = prev[2]
            anchor_moved = (old_anchor is None) != (anchor is None) or (
                anchor is not None and not np.array_equal(old_anchor, anchor)
            )
        # Descontinuidade (primeira da geracao, tinta removida ou ancora
        # re-ancorada): o consumidor substitui; a mascara sai do canvas
        # publicado no poll, entao nao se aloca aqui.
        full = first or removed_px > 0 or anchor_moved
        bbox, mask = None, np.zeros((0, 0), np.uint8)
        if not full:
            new = now & ~prev[1]
            bbox = _bbox_of(new)
            if bbox is not None:
                mask = new[bbox[0] : bbox[1], bbox[2] : bbox[3]].astype(np.uint8) * 255
        record = (
            generation,
            version,
            captured_at,
            bbox,
            mask,
            removed_px,
            full,
            anchor,
            now.shape,
            first,
        )
        with self._condition:
            self._delta_ring.append(record)
            self._condition.notify_all()
        self._delta_prev = (generation, now, anchor)

    def poll_delta(self, generation, after_version, wait=0.0):
        """InkDelta com o que a tinta confirmada ganhou depois de
        after_version, ou None se nao ha publicacao nova da geracao
        (ou o delta dela ainda esta sendo calculado: tente de novo).

        Mesmo gatilho e mesma versao de poll(): quem ja consome o canvas
        pode trocar o diff de mascaras de pagina inteira por isto. Sem
        historico para cobrir (after_version antes do que o anel
        guarda), ou com tinta removida/migrada ou ancora movida no
        intervalo, o delta sai full: substitui o canvas, nao soma.
        removed_px soma as remocoes registradas no historico (limite
        inferior quando o historico foi truncado). Nao muda cadencia
        nem o voto: le registros que a passada ja produziu; o trabalho
        sobre mascaras roda fora do lock. wait (s, 0 = nao espera) cobre a
        janela entre a publicacao e o registro do delta (~1 ms): espera no
        maximo isso pelo registro da versao ja publicada.
        """
        with self._condition:
            if wait > 0 and self._published is not None:
                latest_now = self._published[1]
                self._condition.wait_for(
                    lambda: any(
                        r[0] == generation and r[1] >= latest_now
                        for r in self._delta_ring
                    ),
                    timeout=wait,
                )
            published = self._published
            if (
                published is None
                or published[0] != generation
                or published[1] <= after_version
            ):
                return None
            latest, confirmed = published[1], published[2]
            records = [
                r
                for r in self._delta_ring
                if r[0] == generation and r[1] > after_version
            ]
        if not records or records[-1][1] != latest:
            return None
        complete = all(
            records[i + 1][1] == records[i][1] + 1 for i in range(len(records) - 1)
        ) and (records[0][1] == after_version + 1 or records[0][9])
        last = records[-1]
        full = not complete or any(r[6] for r in records)
        removed_px = sum(r[5] for r in records)
        if full:
            full_mask = confirmed > 0
            bbox = _bbox_of(full_mask)
            mask = (
                np.zeros((0, 0), np.uint8)
                if bbox is None
                else full_mask[bbox[0] : bbox[1], bbox[2] : bbox[3]].astype(np.uint8)
                * 255
            )
        else:
            bbox, mask = _merge_deltas(records)
        try:
            mm = float(paper_config.mm_per_px())
        except ValueError:
            mm = None
        return InkDelta(
            generation=generation,
            since_version=after_version,
            version=latest,
            captured_at=last[2],
            bbox=bbox,
            mask=mask,
            page_grid=tuple(int(v) for v in confirmed.shape),
            mm_per_px=mm,
            anchor_corners=last[7],
            removed_px=removed_px,
            full=full,
        )

    def poll_display(self, generation, after_version):
        """InkDisplaySnapshot para a tela, ou None.

        As camadas vao separadas para a exibicao pintar a
        provisoria em estilo distinto. Este canal e so de
        exibicao: quem consome canvas usa poll().
        """
        with self._condition:
            published = self._published_display
            if (
                published is None
                or published[0] != generation
                or published[1] <= after_version
            ):
                return None
            return InkDisplaySnapshot(
                version=published[1],
                confirmed=published[2],
                provisional=published[3],
            )

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stopped
                )
                if self._stopped:
                    return
                # Cadencia: o intervalo corre com o pendente na
                # mao; um submit durante a espera o substitui
                # (latest-wins) e e ele que roda ao vencer. Se a
                # geracao trocou, new_generation limpou o pendente
                # e esta volta nao roda nada.
                while self._last_pass_start is not None:
                    interval = self._pass_interval()
                    if interval is None:
                        # Descanso total: espera por um
                        # evento que acorde (a mao voltar, a mao
                        # presente se mexer, geracao nova, stop).
                        # Submit NAO acorda: ele so troca o
                        # pendente, que e reavaliado quando um
                        # evento de verdade chegar. Sem isto a
                        # sonda de 1 Hz rodava para sempre sobre a
                        # pagina parada e o voto acumulava camadas.
                        # O prazo é só watchdog contra um notify
                        # perdido: quem decide continua sendo o
                        # predicado do _pass_interval.
                        self._announce_rest_locked()
                        self._condition.wait(timeout=_REST_WATCHDOG_SECONDS)
                        if self._stopped:
                            return
                        if self._pending is None:
                            break
                        continue
                    remaining = interval - (self._clock() - self._last_pass_start)
                    if remaining <= 0 or self._pending is None:
                        break
                    self._condition.wait(timeout=remaining)
                    if self._stopped:
                        return
                if self._pending is None:
                    continue
                job = self._take_pending_locked()
            if not self._run_pass(job):
                return

    def _take_pending_locked(self):
        """Retira o pendente e abre a passada (com a Condition na mao)."""
        (
            warped,
            generation,
            submitted_at,
            corners,
            visible,
            coverage,
            captured_at,
        ) = self._pending
        self._pending = None
        # Intervalo que governou a espera desta passada (o regime
        # de cadencia em vigor na retirada, antes de _last_pass_start
        # andar): vai ao registro de telemetria para o orcamento de
        # latencia separar espera de cadencia de custo de passada.
        # None na primeira passada da trabalhadora.
        scheduled_interval = (
            self._pass_interval() if self._last_pass_start is not None else None
        )
        scheduled_hand_state = self._hand_state
        self._last_pass_start = self._clock()
        pass_started_at = self._last_pass_start
        self._spent_confirm_pass = False
        if self._confirm_passes > 0:
            # Se a passada retirada logo apos a saida da mao veio de um submit
            # anterior a saida ou ainda carrega mascara de oclusao pre-saida
            # (visible com pixels False antes de o detector renovar
            # _hand_mask_det), nao gasta a janela de confirmacao limpa nela
            # (limitado a _INK_VOTE_WINDOW passadas para nunca prender o
            # descanso).
            is_pre_exit = self._telemetry_submits <= self._hand_exit_submits
            # A mascara e de camera quando ha cantos: so a oclusao
            # dentro do recorte mascara a passada — uma mao na mesa
            # fora do quad nao pode gastar a janela de confirmacao
            # pos-saida (review do PR #118).
            is_masked = visible is not None and not _crop_all_visible(visible, corners)
            fill_pending = self._recovery_wait_open()
            if (
                self._post_exit_confirm_passes() > 1
                and (is_pre_exit or is_masked or fill_pending)
                and self._confirm_masked_skips < votes._INK_VOTE_WINDOW
            ):
                # Passada com recuperação de preenchimento em curso
                # também não gasta a janela limpa: o veto da guarda
                # esconde o traco exatamente como uma máscara — o
                # orçamento serve às passadas em que ele pode votar
                # (Devin Review do PR #144). O teto da janela vale
                # para ela igual: pendente sem prazo rearmava a
                # janela para sempre e mantinha a inferencia a
                # toda depois da saida da mao — o orçamento
                # compartilhado e a espera limitada fecham juntos.
                self._confirm_masked_skips += 1
                self._confirm_passes = self._post_exit_confirm_passes()
            else:
                self._confirm_passes -= 1
                self._spent_confirm_pass = True
        return (
            warped,
            generation,
            submitted_at,
            corners,
            visible,
            coverage,
            pass_started_at,
            scheduled_interval,
            scheduled_hand_state,
            captured_at,
        )

    def pump(self):
        """Execucao sincrona (sem thread): roda a passada se ja venceu.

        Para quem injeta um relogio controlado (replays, testes) e
        nao chama start(): a mesma decisao de cadencia e descanso
        do _work, avaliada no instante do relogio, seguida da mesma
        passada. True se uma passada rodou.
        """
        with self._condition:
            if self._stopped or self._pending is None:
                return False
            if self._last_pass_start is not None:
                interval = self._pass_interval()
                if interval is None:
                    self._announce_rest_locked()
                    return False
                elapsed = self._clock() - self._last_pass_start
                if elapsed < interval - _SYNC_CADENCE_EPS_SECONDS:
                    return False
            job = self._take_pending_locked()
        self._run_pass(job)
        return True

    def _run_pass(self, job):
        """Uma passada do modelo + voto. False quando a trabalhadora para."""
        p = _Pass()
        (
            p.warped,
            p.generation,
            p.submitted_at,
            p.corners,
            p.visible,
            p.coverage,
            p.pass_started_at,
            p.scheduled_interval,
            p.scheduled_hand_state,
            p.captured_at,
        ) = job
        # Geracao trocou (ou a trabalhadora parou) entre a retirada
        # e a execucao: a passada esta morta antes de nascer — o
        # pos-modelo a descartaria do mesmo jeito, mas pagando o
        # custo inteiro do modelo por um resultado que nunca entra
        # no canvas. Descarta aqui e conta a economia; o desfecho
        # (True segue, False para) e o mesmo do descarte tardio.
        with self._condition:
            if self._stopped:
                return False
            if p.generation != self._generation:
                self._telemetry_skipped_stale += 1
                return True
            # Captura a guarda desta geracao sob o lock: se new_generation
            # trocar o objeto no meio da inferencia, o observe desta
            # passada escreve na instancia orfa, nao na guarda nova —
            # estado de pagina velha nao semeia referencia na pagina
            # nova (Devin Review do PR #144).
            p.exposure_guard = self._paper_exposure
            if p.generation == self._worker_generation:
                confirmed = self._accumulator._confirmed
                if confirmed is not None:
                    p.exposure_confirmed_ink = (confirmed > 0).copy()
                    p.exposure_confirmed_anchor = (
                        None
                        if self._anchor_corners is None
                        else self._anchor_corners.copy()
                    )
        # Uma excecao do modelo (shape errado, sessao morta, ONNX
        # sem memoria) matava esta thread em silencio: o canvas
        # congelava no ultimo quadro e nada era impresso. O erro
        # fica guardado, e o laco principal o reporta.
        p.pass_started = self._clock()
        try:
            self._pass_evidence(p)
            abort = self._pass_anchor_state(p)
            if abort is not None:
                return abort
            self._pass_register_motion_coverage(p)
            return self._pass_commit(p)
        except Exception as error:  # noqa: BLE001 - vai para o laco
            with self._condition:
                if p.generation != self._generation:
                    self._telemetry_stale_errors += 1
                    self._condition.notify_all()
                    return True
                self._error = error
                self._stopped = True
                self._condition.notify_all()
            return False

    def _pass_evidence(self, p):
        """Imagem -> evidencia de tinta: visivel na grade do warp,
        divnorm opcional, modelo, guarda de papel e pauta.

        Toda a parte cara da passada, fora do lock. Os campos de
        evidencia do _Pass sao preenchidos aqui.
        """
        # As mascaras do WarpedPage sao de CAMERA quando a
        # passada tem cantos (issue #100): visivel na grade da
        # pagina e o mesmo resultado de sempre — o LINEAR da
        # oclusao invertida *255 so e 0 onde TODO pixel-fonte
        # era visivel, e o borderValue=0 fora do frame cai em
        # "visivel" como o `warp(hand) == 0` que o laco
        # produzia (bit-identico). A versao da ancora sai pelo
        # resample unico no registro. Sem cantos nao ha
        # geometria de camera: quem submete sem quad entrega a
        # mascara ja na grade do warp, como antes.
        if p.visible is not None and p.corners is not None:
            p.visible_cam = p.visible
            p.visible = (
                cv2.warpPerspective(
                    (~p.visible).astype(np.uint8) * 255,
                    registration._corners_page_matrix(
                        p.corners, p.warped.shape[1], p.warped.shape[0]
                    ),
                    (p.warped.shape[1], p.warped.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderValue=0,
                )
                == 0
            )
        p.coverage_cam = p.coverage if p.corners is not None else None
        p.coverage_page_static = p.coverage if p.corners is None else None
        photometric_image = p.warped
        if self._ink_divnorm:
            # O warp sobe cru; a imagem que o modelo vê é a
            # mesma que o laço submetia: fora-da-folha
            # preenchido com a média do papel ANTES da divisão
            # (a mesa clara não escurece a borda do papel) e
            # de novo DEPOIS, sobre o normalizado. A cobertura
            # chega na grade da câmera quando há cantos
            # (issue #100) — retifica à grade da página para o
            # fill, no mesmo caminho do `_covered` abaixo.
            divnorm_coverage = p.coverage_page_static
            if divnorm_coverage is None and p.coverage_cam is not None:
                divnorm_coverage = (
                    cv2.warpPerspective(
                        p.coverage_cam.astype(np.uint8) * 255,
                        registration._corners_page_matrix(
                            p.corners, p.warped.shape[1], p.warped.shape[0]
                        ),
                        (p.warped.shape[1], p.warped.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderValue=0,
                    )
                    > 127
                )
            if divnorm_coverage is not None and divnorm_coverage.any():
                p.warped = photometry.division_normalize(
                    _fill_page_outside(p.warped, divnorm_coverage)
                )
                p.warped = _fill_page_outside(p.warped, divnorm_coverage)
            else:
                p.warped = photometry.division_normalize(p.warped)
        if p.visible is None and self._hand_masker is not None:
            hand_mask = self._hand_masker.mask(
                p.warped,
                timestamp_ms=(
                    None if p.captured_at is None else p.captured_at * 1000.0
                ),
            )
            if hand_mask is not None:
                p.visible = hand_mask == 0
        ink_prob = getattr(self._page_clean, "ink_prob", None)
        if ink_prob is not None:
            p.prob = ink_prob(p.warped)
        else:
            p.prob = (self._page_clean.ink_mask(p.warped) > 0).astype(np.float32)
        if self._config.paper_guard and p.warped.dtype == np.uint8:
            # Guarda fotometrica (issue #129): so papel exposto
            # pode apagar ou votar; mao/sombra/caneta congelam,
            # com ou sem a deteccao do MediaPipe. So imagem
            # fotometrica (uint8): mapa float nao e foto.
            reference = paper_evidence.paper_reference(p.warped, p.visible)
            with self._condition:
                if p.generation == self._generation:
                    if reference is not None:
                        self._paper_ref = reference
                    paper_ref = self._paper_ref
                else:
                    paper_ref = None
            # Sem NENHUMA referencia de papel (imagem preta desde o
            # inicio da geracao) nao ha o que comparar: a guarda se
            # abstem como antes. Com referencia previa, incerteza
            # (reference None) congela a pagina inteira.
            if paper_ref is not None:
                p.paper_ok_page = paper_evidence.paper_exposed(p.warped, paper_ref)
        if self._ruling is not None:
            if p.generation != self._ruling_generation:
                self._ruling.reset()
                self._ruling_generation = p.generation
            p.prob = self._ruling.apply(p.prob, p.visible)
        # O suporte de preenchimento conta a evidência no nível que
        # o voto confirma (vote_hit), não num piso fixo: um bloco
        # sustentado em [vote_hit, 0,7) confirmaria no voto mas
        # ficava vetado para sempre (Devin Review do PR #144). A
        # guarda mede a imagem FOTOMETRICA (antes do divnorm) — o
        # normalizado muda a luminancia que a referencia compara.
        confirmed_ink = p.exposure_confirmed_ink
        if (
            confirmed_ink is not None
            and p.corners is not None
            and p.exposure_confirmed_anchor is not None
        ):
            matrix = registration._registration_matrix(
                p.exposure_confirmed_anchor, p.corners, p.prob.shape
            )
            if not registration._registration_is_identity(matrix):
                confirmed_ink = (
                    cv2.warpPerspective(
                        confirmed_ink.astype(np.uint8),
                        np.linalg.inv(matrix),
                        (p.prob.shape[1], p.prob.shape[0]),
                        flags=cv2.INTER_NEAREST,
                        borderValue=0,
                    )
                    > 0
                )
        p.exposure = (
            p.exposure_guard.observe(
                photometric_image,
                p.visible,
                ink_prob=p.prob,
                ink_support_threshold=self._config.vote_hit,
                confirmed_ink=confirmed_ink,
            )
            if self._config.paper_guard
            else None
        )
        # Cobertura de folha (cobertura de folha do loop) viaja em
        # canal proprio ate depois do registro: o veto nao e
        # cego nem se confunde com oclusao de mao — ver o
        # gate de re-confirmacao abaixo, no espaco da ancora.

    def _pass_anchor_state(self, p):
        """Ancora da passada sob o lock: guarda de geracao, escolha da
        ancora, mediana e banda morta de registro.

        Devolve None para seguir a passada, ou o bool que _run_pass
        propaga (trabalhadora parada / geracao morta).
        """
        with self._condition:
            if self._stopped:
                return False
            if p.generation != self._generation:
                return True
            if p.exposure_guard.last_fill_adoption:
                # A guarda acabou de adotar um bloco como
                # preenchimento: as passadas que o votam precisam
                # cobrir a janela de voto — sem rearma o
                # orçamento, gasto durante o streak, expirava
                # antes da confirmação fechar (Devin Review do
                # PR #144). Rearmar é por borda de adoção; um
                # bloco adotado não segura o orçamento para
                # sempre. A borda tambem zera a espera: houve
                # progresso, a recuperacao nao e oscilacao.
                self._fill_recovery_waited = 0
                self._confirm_passes = max(
                    self._confirm_passes,
                    votes._INK_VOTE_WINDOW,
                )
            elif p.exposure_guard.recovery_pending:
                self._fill_recovery_waited += 1
            else:
                self._fill_recovery_waited = 0
            if p.exposure_guard.max_marks_recovery_age:
                self._marks_recovery_waited += 1
            else:
                self._marks_recovery_waited = 0
            p.first_pass = p.generation != self._worker_generation
            p.recovering = self._identity_reference is not None and (
                p.first_pass or self._identity_recovery_requested
            )
            if p.recovering:
                p.identity_reference_snap = self._identity_reference
                p.identity_baseline_snap = self._identity_baseline
                p.identity_corners_snap = self._identity_reference_corners
                p.identity_hold_seen_snap = self._identity_hold_seen
            p.anchor = (
                self._identity_reference_corners
                if p.recovering
                else p.corners
                if p.first_pass
                else self._anchor_corners
            )
            p.registration_corners = p.corners
            if not p.recovering and p.corners is not None:
                anchor_rebased = False
                if (
                    self._config.reg_anchor_median
                    and not p.first_pass
                    and 0
                    < len(self._anchor_samples)
                    < registration._REG_ANCHOR_MEDIAN_SAMPLES
                ):
                    self._anchor_samples.append(p.corners.copy())
                    self._anchor_corners = registration._median_anchor(
                        self._anchor_samples
                    )
                    p.anchor = self._anchor_corners
                    anchor_rebased = True
                if self._config.reg_anchor_median:
                    p.reg_info["reg_anchor_samples"] = max(
                        1 if p.first_pass else 0, len(self._anchor_samples)
                    )
                if self._config.reg_deadband:
                    if self._reg_frozen_corners is None:
                        self._reg_frozen_corners = p.corners.copy()
                        self._reg_deadband_over = 0
                    if anchor_rebased:
                        # Rebase (interacao das duas flags,
                        # prova composta de 2026-11-10): a
                        # ancora ainda esta assentando (a
                        # janela da mediana acabou de andar);
                        # o congelado acompanha a ancora em
                        # vez de ficar preso na amostra 1, e o
                        # contador de liberacao recomeca.
                        # Fechada a janela, vale a semantica
                        # da banda morta do #57.
                        self._reg_frozen_corners = self._anchor_corners.copy()
                        self._reg_deadband_over = 0
                        p.registration_corners = self._reg_frozen_corners
                        shift = float(
                            np.max(
                                np.linalg.norm(
                                    p.corners - self._reg_frozen_corners,
                                    axis=1,
                                )
                            )
                        )
                        p.reg_info.update(
                            {
                                "reg_deadband_shift_px": shift,
                                "reg_deadband_held": True,
                                "reg_deadband_pending": 0,
                                "reg_deadband_released": False,
                                "reg_deadband_rebased": True,
                            }
                        )
                    else:
                        (
                            p.registration_corners,
                            self._reg_frozen_corners,
                            self._reg_deadband_over,
                            released,
                            shift,
                        ) = registration._deadband_step(
                            self._reg_frozen_corners,
                            p.corners,
                            self._reg_deadband_over,
                        )
                        if released:
                            self._reg_deadband_releases += 1
                        p.reg_info.update(
                            {
                                "reg_deadband_shift_px": shift,
                                "reg_deadband_held": not released,
                                "reg_deadband_pending": self._reg_deadband_over,
                                "reg_deadband_released": released,
                            }
                        )
            confirmed = self._accumulator._confirmed
            p.anchor_ink = (
                confirmed > 0
                if confirmed is not None and confirmed.shape == p.prob.shape
                else None
            )
        return None

    def _pass_register_motion_coverage(self, p):
        """Registro -> portao de movimento -> portao de cobertura:
        a evidencia sai na grade da ancora, fora do lock."""
        # Pegada amostrada pelo registro (zeros de preenchimento
        # onde a passada nao alcancou a ancora), cruzada com a
        # pegada do quad vivo (o que a passada cobriu de fato):
        # separa "nao observei" de "observei vazio" quando nao ha
        # mascara de visibilidade — entra no dominio valido la no
        # voto, e o registro a reusa no clipe de visibilidade em
        # vez de rasterizar a dela de novo.
        p.sampled = self._pass_sampled(
            p.anchor, p.registration_corners, p.prob.shape, live_corners=p.corners
        )
        p.registered_prob, p.registered_visible = (
            registration._register_ink_probability(
                p.prob,
                p.anchor,
                p.registration_corners,
                p.visible,
                visible_cam=p.visible_cam,
                pass_corners=p.corners,
                sampled=p.sampled,
            )
        )
        p.last_reg = _reg_triple(p.anchor, p.registration_corners, p.corners)
        # Mapa registrado SO pela camera: e o que a arbitragem de
        # ancora compara com o mapa cru para julgar pouso x quad ruim.
        # A escolha/correcao do portao vale so para a evidencia que
        # vota (review do PR #67: o mapa corrigido inflava o escore de
        # camera e um movimento real virava "espurio").
        p.camera_prob = p.registered_prob
        p.camera_corners = p.registration_corners
        if (
            self._motion is not None
            and not p.recovering
            and p.anchor is not None
            and p.registration_corners is not None
        ):
            # Estado do portao so nesta thread: geracao nova zera
            # a referencia aqui, nao em new_generation (outra
            # thread, no meio de uma medida).
            if self._motion_generation != p.generation:
                self._motion.reset()
                self._motion_generation = p.generation
            motion_matrix = registration._registration_matrix(
                p.anchor, p.registration_corners, p.prob.shape
            )
            if registration._registration_is_identity(motion_matrix):
                motion_matrix = None
            measured = self._motion.measure(
                p.warped,
                motion_matrix,
                p.visible,
                p.anchor,
                confirmed_ink=None if p.first_pass else p.anchor_ink,
                ink_prob=p.registered_prob,
            )
            unregistered = None
            if motion_matrix is not None and not self._motion.aligned(measured):
                # O registro supoe a folha parada; se ela andou e o
                # quad acompanhou, o warp sem registro e o alinhado
                # (folha-movida, passadas 5-9: 1,2-1,6 px sem
                # registro contra 3,1-4,0 px com ele).
                unregistered = self._motion.measure(p.warped, None, p.visible, p.anchor)
            p.motion_decision, p.motion_fix, hypothesis = self._motion.decide(
                measured, unregistered
            )
            p.reg_info["motion_decision"] = p.motion_decision
            p.reg_info["motion_residual_px"] = (
                None if measured is None else measured.max_px
            )
            p.reg_info["motion_unregistered_px"] = (
                None if unregistered is None else unregistered.max_px
            )
            p.reg_info["motion_blocks"] = (
                None if measured is None else (measured.inliers, measured.blocks)
            )
            if hypothesis == "unregistered":
                # A passada vota na propria geometria: registro
                # identidade, a mesma da 1a passada da ancora.
                p.registration_corners = p.anchor.copy()
                # Registro identidade (ancora contra ela
                # mesma): a pegada e a grade inteira.
                p.sampled = None
                p.registered_prob, p.registered_visible = (
                    registration._register_ink_probability(
                        p.prob,
                        p.anchor,
                        p.registration_corners,
                        p.visible,
                        visible_cam=p.visible_cam,
                        pass_corners=p.corners,
                        sampled=p.sampled,
                    )
                )
                p.last_reg = _reg_triple(p.anchor, p.registration_corners, p.corners)
            if p.motion_decision == "correct":
                # A evidencia volta para a grade da ancora pelo
                # modelo afim medido nesta mesma imagem.
                p.registered_prob = p.motion_fix.warp_back(p.registered_prob)
                if p.registered_visible is not None:
                    p.registered_visible = (
                        p.motion_fix.warp_back(
                            p.registered_visible.astype(np.uint8), nearest=True
                        )
                        > 0
                    )
                # A correcao afim tambem recorta a pegada: a
                # faixa que o warp_back preenche de zeros nao
                # foi observada.
                if self._config.border_guard_px != 0:
                    p.sampled = (
                        p.motion_fix.warp_back(
                            np.ones(p.prob.shape, np.uint8)
                            if p.sampled is None
                            else p.sampled.astype(np.uint8),
                            nearest=True,
                        )
                        > 0
                    )
        if p.coverage_cam is not None or p.coverage_page_static is not None:
            # Gate de re-confirmacao (cobertura de folha do loop): onde
            # o warp nao e folha, so a tinta que a ancora JA
            # conhece vota — re-confirmar e retrair o que
            # existe, sim; tinta nova nascendo na mesa, nunca.
            # e o que separa o buraco semantico (a faixa de
            # texto que o segmentador marca como fundo na pose
            # nova, E3 2026-11-12) da banda de mesa: os dois
            # sao "fora da mascara perto dela", mas so um ja
            # era tinta. Sem tinta de ancora o gate e o veto
            # seco de sempre.
            p.coverage_page = p.coverage_page_static

            def _covered(reg_corners):
                size = (p.prob.shape[1], p.prob.shape[0])
                if (
                    p.coverage_cam is not None
                    and p.anchor is not None
                    and reg_corners is not None
                ):
                    reg_matrix = registration._registration_matrix(
                        p.anchor, reg_corners, p.prob.shape
                    )
                    if not registration._registration_is_identity(reg_matrix):
                        if p.coverage_anchor is None:
                            # Resample único câmera→âncora (issue
                            # #100): M @ page_matrix(passada) — o
                            # mesmo caminho da tinta; na banda
                            # morta o quad de registro != o ao
                            # vivo e A sozinha desalinha ~tau px
                            # (review do PR #118).
                            p.coverage_anchor = (
                                cv2.warpPerspective(
                                    p.coverage_cam.astype(np.uint8),
                                    reg_matrix
                                    @ registration._corners_page_matrix(
                                        p.corners, size[0], size[1]
                                    ),
                                    size,
                                    flags=cv2.INTER_NEAREST,
                                    borderValue=0,
                                )
                                > 0
                            )
                        return p.coverage_anchor
                if p.coverage_page is None:
                    # Sem registro (ou identidade): a grade da
                    # própria passada — o mesmo `warped > 127`
                    # que o laço produzia.
                    p.coverage_page = (
                        cv2.warpPerspective(
                            p.coverage_cam.astype(np.uint8) * 255,
                            registration._corners_page_matrix(
                                p.corners, size[0], size[1]
                            ),
                            size,
                            flags=cv2.INTER_LINEAR,
                            borderValue=0,
                        )
                        > 127
                    )
                return p.coverage_page

            def _gate(evidence, reg_coverage):
                if p.anchor_ink is None:
                    return evidence * reg_coverage
                return np.where(
                    reg_coverage | p.anchor_ink,
                    evidence,
                    0.0,
                ).astype(np.float32)

            # O mapa da camera que a arbitragem pontua passa pelo
            # MESMO gate de cobertura de sempre (na geometria da
            # camera): sem isso, pixels fora da folha diluiam o
            # escore de camera mesmo com o portao desligado
            # (review do PR #67). Portao sem efeito nesta passada:
            # o mapa da camera e o proprio mapa que vota, como antes.
            camera_untouched = (
                p.camera_prob is p.registered_prob
                and p.camera_corners is p.registration_corners
            )
            reg_coverage = _covered(p.registration_corners)
            if p.motion_decision == "correct":
                # A cobertura acompanha a correcao da evidencia: sem
                # isso o traco novo deslocado para fora da mascara
                # velha era cortado em toda passada corrigida.
                reg_coverage = (
                    p.motion_fix.warp_back(reg_coverage.astype(np.uint8), nearest=True)
                    > 0
                )
            p.reg_coverage = reg_coverage
            p.coverage_matches_last_reg = True
            p.registered_prob = _gate(p.registered_prob, reg_coverage)
            if camera_untouched:
                p.camera_prob = p.registered_prob
            else:
                p.camera_prob = _gate(p.camera_prob, _covered(p.camera_corners))
        if p.anchor is not None and p.registration_corners is not None:
            matrix = registration._registration_matrix(
                p.anchor, p.registration_corners, p.prob.shape
            )
            p.reg_info["registration_displacement_px"] = (
                anchor_mod._matrix_grid_displacement(matrix, p.prob.shape)
            )
            p.reg_info["registration_is_identity"] = (
                registration._registration_is_identity(matrix)
            )
        if p.identity_reference_snap is not None:
            identity_sampled = p.sampled
            if identity_sampled is None:
                # Recovering implica portao de movimento
                # parado: nao ha correcao afim a desfazer,
                # so a pegada do registro.
                identity_sampled = registration._observed_footprint(
                    p.anchor,
                    p.registration_corners,
                    p.corners,
                    p.prob.shape,
                )
            identity_visible = p.registered_visible
            if p.exposure is not None and p.exposure.visible is not None:
                # O bloqueio fotométrico também rege o
                # reencontro: a sombra que o detector de mão
                # perdeu não pode contar contra a página —
                # a tinta anterior escondida por ela deixa
                # de derrubar o recall (Devin Review do PR
                # #144). As máscaras entram na MESMA grade
                # do candidato: o par que registrou esta
                # passada, antes da arbitragem/follow.
                identity_photo, _unused_exposed = _exposure_masks_in_geometry(
                    p.exposure,
                    p.anchor,
                    p.registration_corners,
                    p.prob.shape,
                    (p.motion_fix if p.motion_decision == "correct" else None),
                )
                identity_visible = (
                    identity_photo
                    if identity_visible is None
                    else identity_visible & identity_photo
                )
            p.identity_precomputed = (
                p.identity_reference_snap,
                p.identity_baseline_snap,
                _evaluate_page_identity(
                    p.identity_reference_snap,
                    p.identity_baseline_snap,
                    p.registered_prob,
                    identity_visible,
                    identity_sampled,
                    self._cached_border_guard(p.prob.shape, p.identity_corners_snap),
                    p.identity_hold_seen_snap,
                    self._accumulator.threshold,
                    self._identity_gray(p),
                    self._page_model,
                    self._identity_gray_ref,
                ),
            )

    def _pass_commit(self, p):
        """Decisao, PageModel (fora do lock), voto e publicacao da passada.

        Estagios internos:
        - _condition: identidade, ancora e preparo do voto (fase A).
        - _pm_gate, fora de _condition: PageModel e registro fotometrico.
        - _pm_gate + _condition: valida epochs, vota e publica (fase B).
          new_generation toma o mesmo gate antes de _condition, entao nao
          observa nem preserva um PageModel parcialmente atualizado.
        Devolve o bool que _run_pass propaga.
        """
        delta_job = None
        with self._condition:
            if self._stopped:
                return False
            if p.generation != self._generation:
                return True
            p.first_pass = p.generation != self._worker_generation
            if p.first_pass:
                self._worker_generation = p.generation
                self._generation_passes = 0
                self._quiet_passes = 0
            # Degrau geometrico entre passadas consecutivas
            # (o follow so conta passada assentada; o portao de
            # remocao exige a mesma quietude).
            if p.corners is not None:
                if self._previous_corners is not None:
                    p.watch_step_px = float(
                        np.max(
                            np.linalg.norm(
                                p.corners - self._previous_corners,
                                axis=1,
                            )
                        )
                    )
                self._previous_corners = p.corners.copy()
            self._pass_identity(p)
            p.identity_owned = (
                p.first_pass
                or p.identity_pending
                or p.identity_deferred
                or p.identity_switch_pending
            )
            self._pass_anchor_gates(p)
            self._pass_prepare_vote(p)
        # A transição toma o mesmo gate antes do lock. Logo, ela ocorre
        # inteiramente antes deste trecho (e o token abaixo o descarta)
        # ou depois do modelo, voto, publicação e delta desta passada.
        with self._pm_gate:
            with self._condition:
                if self._stopped:
                    return False
                if p.generation != self._generation or p.pm_epoch != (
                    self._canvas_epoch,
                    self._pm_reset_seq,
                ):
                    return True
            self._pass_paper_and_page_model(p)
            with self._condition:
                if self._stopped:
                    return False
                step = 1 if p.pm_rezero_applied else 0
                if (
                    p.generation != self._generation
                    or p.pm_epoch
                    != (self._canvas_epoch + step, self._pm_reset_seq + step)
                    or (p.pm_rezero_applied and not self._pm_rezero_requested)
                ):
                    return True
                if p.pm_rezero_applied:
                    # Commit do re-zero (#130): canvas e modelo novos juntos,
                    # canvas vazio publicado nesta passada.
                    self._pm_rezero_requested = False
                    self._pm_rezero_seeding = True
                    self._accumulator.reset()
                    self._canvas_epoch += 1
                    self._pm_reset_seq += 1
                    self._pm_baseline_ink = None
                    p.force_publish = True
                delta_job = self._pass_vote_and_publish(p)
                self._condition.notify_all()
            if delta_job is not None:
                self._record_delta(*delta_job)
        return True

    def _pass_identity(self, p):
        """Identidade de pagina (F-11): roda na primeira passada e,
        enquanto estiver em quarentena, nas seguintes — a referencia
        so e consumida quando a decisao sai.

        Chamado com _condition tomado. Preenche os campos identity_*,
        keep_page e quarantine_* do _Pass; timeout ou pagina nova
        resetam acumulador e ancora.
        """
        p.identity_pending = (
            self._identity_recovery_requested and self._identity_reference is not None
        )
        if p.first_pass or p.identity_pending:
            identity_reference = self._identity_reference
            identity_baseline = self._identity_baseline
            identity_corners = self._identity_reference_corners
            identity_recovery_requested = self._identity_recovery_requested
            p.keep_page = False
            if identity_reference is not None:
                identity_visible = p.registered_visible
                if p.exposure is not None and p.exposure.visible is not None:
                    # O bloqueio fotométrico também rege o
                    # reencontro: a sombra que o detector de mão
                    # perdeu não pode contar contra a página —
                    # a tinta anterior escondida por ela deixa
                    # de derrubar o recall (Devin Review do PR
                    # #144). As máscaras entram na MESMA grade
                    # do candidato: o par que registrou esta
                    # passada, antes da arbitragem/follow.
                    identity_photo, _unused_exposed = _exposure_masks_in_geometry(
                        p.exposure,
                        p.anchor,
                        p.registration_corners,
                        p.prob.shape,
                        (p.motion_fix if p.motion_decision == "correct" else None),
                    )
                    identity_visible = (
                        identity_photo
                        if identity_visible is None
                        else identity_visible & identity_photo
                    )
                if identity_corners is not None and p.corners is not None:
                    p.identity_geometry_registered = True
                if (
                    p.identity_precomputed is not None
                    and p.identity_precomputed[0] is identity_reference
                    and p.identity_precomputed[1] is identity_baseline
                ):
                    result = p.identity_precomputed[2]
                else:
                    identity_sampled = p.sampled
                    if identity_sampled is None:
                        # Recovering implica portao de movimento
                        # parado: nao ha correcao afim a desfazer,
                        # so a pegada do registro.
                        identity_sampled = registration._observed_footprint(
                            p.anchor,
                            p.registration_corners,
                            p.corners,
                            p.prob.shape,
                        )
                    result = _evaluate_page_identity(
                        identity_reference,
                        identity_baseline,
                        p.registered_prob,
                        identity_visible,
                        identity_sampled,
                        self._cached_border_guard(p.prob.shape, identity_corners),
                        self._identity_hold_seen,
                        self._accumulator.threshold,
                        self._identity_gray(p),
                        self._page_model,
                        self._identity_gray_ref,
                    )
                (
                    p.keep_page,
                    p.identity_recall,
                    p.identity_containment,
                    p.identity_reference_px,
                    p.identity_visible_reference_px,
                    p.identity_deferred,
                    p.quarantine_reason,
                    p.identity_hold,
                ) = result
            if p.identity_deferred:
                # O hold por interior oculto nao queima a
                # quarentena: reter a referencia e separado de
                # apagar o canvas salvo — so um timeout de
                # evidencia geometrica (no_overlap) ainda destrói.
                if not p.identity_hold:
                    self._identity_defer_count += 1
                p.quarantine_frames = self._identity_defer_count
                p.identity_timed_out = (
                    p.quarantine_frames > identity._PAGE_IDENTITY_MAX_DEFERRALS
                )
            if p.identity_hold:
                # O hold e pegajoso na recuperacao: depois de um
                # match parcial ou dominio oculto, uma oclusao mais
                # pesada nas passadas seguintes tambem nao queima a
                # quarentena — o prazo so corre para evidencia
                # geometrica (no_overlap) ou oclusao sem hold previo.
                self._identity_hold_seen = True
            if p.identity_deferred:
                self._identity_switch_votes = 0
            elif identity_reference is not None and not p.keep_page:
                self._identity_switch_votes += 1
                p.identity_switch_pending = (
                    self._identity_switch_votes
                    < identity._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES
                )
            else:
                self._identity_switch_votes = 0
            if identity_recovery_requested:
                if p.identity_timed_out:
                    p.identity_decision = "quarantine_timeout"
                elif p.identity_deferred:
                    p.identity_decision = "deferred_occlusion"
                elif p.identity_switch_pending:
                    p.identity_decision = "deferred_switch_consensus"
                else:
                    p.identity_decision = "same_page" if p.keep_page else "new_page"
                self._identity_events.append(
                    {
                        "generation": p.generation,
                        "decision": p.identity_decision,
                        "recall": p.identity_recall,
                        "containment": p.identity_containment,
                        "reference_px": p.identity_reference_px,
                        "visible_reference_px": (p.identity_visible_reference_px),
                        "geometry_registered": (p.identity_geometry_registered),
                        "threshold": identity._PAGE_IDENTITY_MIN_RECALL,
                        "containment_threshold": (
                            identity._PAGE_IDENTITY_MIN_CONTAINMENT
                        ),
                        "quarantine_reason": p.quarantine_reason,
                        "quarantine_frames": p.quarantine_frames,
                        "deferred": p.identity_deferred,
                    }
                )
            if (
                p.identity_deferred and not p.identity_timed_out
            ) or p.identity_switch_pending:
                # A referencia NAO e consumida: a proxima
                # passada decide com registro e visao
                # confiaveis e consenso. ECC rejeitado nao vota.
                pass
            else:
                if p.identity_timed_out:
                    # Perda de lock, nao voto de pagina nova.
                    # A proxima passada estabelece outra ancora.
                    self._accumulator.reset()
                    self._pm_rezero_requested = False
                    self._pm_rezero_seeding = False
                    self._canvas_epoch += 1
                    self._page_model = None
                    self._pm_epoch = None
                    self._pm_baseline_ink = None
                    self._anchor_corners = None
                    self._anchor_samples = []
                    self._reg_frozen_corners = None
                    self._reg_deadband_over = 0
                    self._arb_event_open = False
                    self._arb_streak_quads = []
                    self._arb_scores = []
                    self._arb_snapshot = None
                    self._follow_event_open = False
                    self._follow_streak = 0
                    self._follow_streak_quads = []
                    self._follow_hold = False
                    self._worker_generation = -1
                    self._generation_passes = 0
                    self._quiet_passes = 0
                    self._still_passes = 0
                elif not p.keep_page:
                    self._accumulator.reset()
                    self._pm_rezero_requested = False
                    self._pm_rezero_seeding = False
                    self._canvas_epoch += 1
                    self._page_model = None
                    self._pm_epoch = None
                    self._pm_baseline_ink = None
                    self._anchor_corners = p.corners
                    self._anchor_samples = (
                        [p.corners.copy()]
                        if self._config.reg_anchor_median and p.corners is not None
                        else []
                    )
                elif identity_reference is not None:
                    # same_page: a copia volta para o acumulador
                    # antes do add desta passada — unida ao que a
                    # geracao ja confirmou nas passadas deferidas
                    # que votaram — e sai no publish desta
                    # resolucao.
                    self._accumulator.restore_confirmed(identity_reference)
                # Keep the old anchor when identity matched: all future
                # observations remain registered into the same canvas.
                self._identity_hold_seen = False
                self._identity_reference = None
                self._identity_reference_corners = None
                self._identity_gray_ref = None
                self._identity_baseline = None
                self._identity_recovery_requested = False
                self._identity_defer_count = 0
                self._identity_switch_votes = 0

    def _pass_anchor_gates(self, p):
        """Arbitragem e mobilidade da ancora, nesta ordem: o
        veredito da arbitragem nesta passada e a precedencia que
        o passo de follow le.

        Chamado com _condition tomado. Abstencoes abertas viram
        campos do _Pass lidos no voto; pouso/re-ancoragem
        re-registra a evidencia na ancora nova.
        """
        reg_anchor_median = self._config.reg_anchor_median
        reg_deadband = self._config.reg_deadband
        arb_status, arb_event = self._anchor_arbitration_step(
            p.generation,
            p.corners,
            p.prob.shape,
            p.identity_owned,
            reg_anchor_median,
            reg_deadband,
            p.prob,
            p.camera_prob,
            p.visible,
        )
        p.arbitration_abstain = arb_status == "open"
        p.reg_info["arb_event_open"] = self._arb_event_open
        p.reg_info["arb_abstained"] = p.arbitration_abstain
        p.reg_info["arb_streak"] = len(self._arb_streak_quads)
        if self._arb_scores:
            scores = self._arb_scores[-1]
            p.reg_info["arb_pass_score_page"] = scores[0]
            p.reg_info["arb_pass_score_camera"] = scores[1]
            p.reg_info["arb_pass_scores_cyclic"] = list(scores[2:5])
        if arb_event is not None:
            p.reg_info["arbitration_event"] = dict(arb_event)
            p.reg_info["arb_verdict"] = arb_event["verdict"]
            p.reg_info["arb_score_page"] = arb_event["score_page"]
            p.reg_info["arb_score_camera"] = arb_event["score_camera"]
            p.reg_info["arb_rotation_k"] = arb_event["rotation_k"]
        if arb_status == "verdict" and arb_event is not None and arb_event["action"]:
            # Pouso: a passada corrente entra ja
            # registrada na ancora nova, por cima do
            # estado INTOCADO (a v3 nao move o estado).
            # A decisao do portao foi contra a ancora velha.
            self._reregister_current(
                p,
                self._reg_frozen_corners if reg_deadband else p.corners,
            )
            p.force_publish = True
        # Mobilidade da ancora (frente 34): roda DEPOIS
        # da arbitragem — o veredito dela nesta passada
        # (arb_event) e a precedencia que o passo le.
        follow_status, follow_event = self._anchor_follow_step(
            p.generation,
            p.corners,
            p.prob.shape,
            p.identity_owned,
            reg_anchor_median,
            reg_deadband,
            p.watch_step_px,
            arb_event,
        )
        p.follow_abstain = follow_status == "open"
        p.reg_info["follow_event_open"] = self._follow_event_open
        p.reg_info["follow_abstained"] = p.follow_abstain
        p.reg_info["follow_streak"] = self._follow_streak
        p.reg_info["follow_hold"] = self._follow_hold
        if follow_event is not None:
            p.reg_info["follow_reanchored"] = True
            p.reg_info["follow_displacement_px"] = follow_event["displacement_px"]
            p.reg_info["follow_closed_arb_event"] = follow_event["closed_arb_event"]
            if follow_event["closed_arb_event"]:
                p.reg_info["arb_event_open"] = False
            # A passada corrente entra ja registrada na
            # ancora nova, por cima do estado INTOCADO
            # (a troca e so de ancora) — e o evento de
            # arbitragem que a abstinha foi fechado pela
            # re-ancoragem. A decisao do portao foi contra a
            # ancora velha.
            self._reregister_current(
                p,
                self._reg_frozen_corners if reg_deadband else p.corners,
            )
            p.arbitration_abstain = False
            p.force_publish = True

    def _reregister_current(self, p, registration_corners):
        """Re-registra a passada corrente na ancora nova (apos
        pouso/re-ancoragem): a evidencia entra na grade nova por
        cima do estado INTOCADO, e a decisao de movimento medida
        contra a ancora velha morre com ela."""
        p.motion_decision = None
        p.registered_prob, p.registered_visible, p.sampled = self._register_into(
            self._anchor_corners,
            registration_corners,
            p.prob,
            p.visible,
            p.visible_cam,
            pass_corners=p.corners,
        )
        p.last_reg = _reg_triple(self._anchor_corners, registration_corners, p.corners)
        p.coverage_matches_last_reg = False

    def _pass_prepare_vote(self, p):
        """Escolhe o mapa que vota, aplica abstencoes e tira snapshot do
        estado da epoca/acumulador antes de soltar o lock para o PageModel."""
        if (
            (p.first_pass or p.identity_pending)
            and not p.identity_deferred
            and not p.identity_switch_pending
            and not p.keep_page
        ):
            if p.visible is not None:
                p.prob = np.where(p.visible, p.prob, 0.0).astype(np.float32)
            # Voto sem registro (1a passada ou pagina nova na
            # identidade): a evidencia entra na propria
            # geometria do warp e a ancora e o quad da
            # passada — o par registrado antes (referencia
            # de identidade, quad atual) descreveria uma
            # matriz pela qual a tinta nunca passou.
            p.last_reg = _reg_triple(p.corners, p.corners, p.corners)
            p.coverage_matches_last_reg = False
            p.used_registered_prob = False
            # Voto na propria geometria: registro identidade,
            # a pegada e a grade inteira.
            p.sampled = None
        else:
            p.prob = p.registered_prob
            p.visible = p.registered_visible
            p.used_registered_prob = True
        # Veto de movimento so vale na passada registrada: a 1a
        # passada e a de identidade votam na propria geometria.
        p.motion_abstain = p.motion_decision == "veto" and p.used_registered_prob
        # Passada adiada por "no_overlap" ou segurada por
        # interior oculto nao vota: o adiamento de oclusao e
        # sobre a MESMA folha escondida pela mao e a evidencia
        # amostrada segue valida; os outros dois sao sobre OUTRA
        # folha possivel — a tinta nova registrada na regiao
        # parcial confirmaria no canvas velho antes de a
        # decisao sair, misturando os dois desenhos.
        identity_abstain = p.identity_switch_pending or (
            p.identity_deferred
            and (
                p.identity_timed_out
                or p.quarantine_reason == "no_overlap"
                or p.identity_hold
            )
        )
        p.pass_voted = not (
            identity_abstain
            or p.arbitration_abstain
            or p.follow_abstain
            or p.motion_abstain
        )
        if identity_abstain:
            p.visible = np.zeros(p.prob.shape, dtype=bool)
        if p.arbitration_abstain:
            # Evento de arbitragem aberto: a passada e
            # registrada e pontuada, mas nao vota.
            p.visible = np.zeros(p.prob.shape, dtype=bool)
        if p.follow_abstain:
            # Evento de divergencia assentada aberto: a
            # passada foi medida, mas nao vota — registrada
            # na ancora velha, ela depositaria a copia
            # deslocada que a re-ancoragem existe para
            # nao fabricar.
            p.visible = np.zeros(p.prob.shape, dtype=bool)
        if p.motion_abstain:
            # A folha anda sob um quad parado: registrada, esta
            # passada carimbaria a copia deslocada da tinta.
            p.visible = np.zeros(p.prob.shape, dtype=bool)
        if not p.pass_voted:
            # Passada abstida nao deposita: o registro fica
            # None para o avaliador nao projetar GT por uma
            # geometria que nunca chegou ao canvas.
            p.last_reg = None
        # Re-zero (#130): so candidata aqui. A troca de modelo e o clear
        # do canvas dependem de a passada ter papel utilizavel, o que so
        # se sabe depois da mascara de papel (fora do lock) — ver
        # _pass_paper_and_page_model e o commit em _pass_commit.
        p.pm_rezero = bool(
            self._pm_rezero_requested
            and self._config.page_model
            and p.pass_voted
            and not p.identity_owned
        )
        p.pm_epoch = (self._canvas_epoch, self._pm_reset_seq)
        if self._config.page_model and p.pass_voted:
            confirmed = self._accumulator._confirmed
            p.confirmed_ink = (
                (confirmed > 0).copy()
                if confirmed is not None and confirmed.shape == p.prob.shape
                else np.zeros(p.prob.shape, dtype=bool)
            )
            p.pm_baseline_ink = (
                None if self._pm_baseline_ink is None else self._pm_baseline_ink.copy()
            )

    def _pass_paper_and_page_model(self, p):
        """Registro da guarda e PageModel fora de _condition, sob _pm_gate."""
        paper_ok = p.paper_ok_page
        registration_used = p.last_reg
        if (
            paper_ok is not None
            and registration_used is not None
            and registration_used[0] is not None
            and p.used_registered_prob
        ):
            # Mesma transformacao da probabilidade; so o que sobrevive
            # inteiro ao resample (== 1) conta como papel.
            paper_ok = (
                registration._register_ink_probability(
                    paper_ok.astype(np.float32),
                    registration_used[0],
                    registration_used[1],
                    None,
                )[0]
                > 0.999
            )
            if p.motion_decision == "correct":
                # A evidencia de papel segue o mesmo modelo afim da
                # probabilidade e da visibilidade.
                paper_ok = (
                    p.motion_fix.warp_back(paper_ok.astype(np.uint8), nearest=True) > 0
                )
        p.paper_ok = paper_ok
        # Sem a guarda o comportamento é o legado: a evidência
        # de papel exposto não rege a remoção — mas também não há
        # avaliação fotométrica, então o opt-out entrega None (sem
        # avaliação), não True: com True o acumulador lia "avaliação
        # aprovada" e um streak de remoção armado drenava sem
        # testemunha nem prova de papel (Devin Review do PR #144).
        # Guarda ligada sem avaliação na passada (imagem fora do
        # contrato fotométrico) falha fechada para a remoção: nada
        # prova papel, a ausência não pode ser testemunhada —
        # incerteza congela.
        paper_exposed = None
        if self._config.paper_guard:
            paper_exposed = False
        physical_visible = p.visible
        if p.exposure is not None:
            # O par de registro é o da passada que VOTA
            # (p.last_reg), não o medido antes da
            # arbitragem/follow: na passada de re-ancoragem a prob
            # já está na âncora nova e a âncora velha cobriria os
            # pixels errados. Sem máscara de visibilidade própria
            # (caminho clássico, sem modelo de mão) o "sem bloco"
            # da guarda é materializado em máscara — senão o
            # acumulador nunca tem evidência para a remoção e o
            # apagamento congela na página limpa.
            last_reg = registration_used
            photo_visible, paper_exposed = _exposure_masks_in_geometry(
                p.exposure,
                None if last_reg is None else last_reg[0],
                None if last_reg is None else last_reg[1],
                p.prob.shape,
                (
                    p.motion_fix
                    if p.motion_decision == "correct" and p.used_registered_prob
                    else None
                ),
                fill_clear=p.visible is None,
            )
            if photo_visible is not None:
                p.visible = (
                    photo_visible if p.visible is None else p.visible & photo_visible
                )
        if p.paper_ok is not None:
            # As duas testemunhas de papel votam juntas na remocao:
            # so conta onde a guarda fotometrica E a referencia de
            # papel provam exposicao — fail-closed. Quem se abstem
            # (paper_ok None, sem referencia ainda, ou a guarda
            # desligada no modo legado) deixa o outro falar sozinho;
            # quem nega papel congela a regiao.
            paper_exposed = (
                p.paper_ok if paper_exposed is None else paper_exposed & p.paper_ok
            )
        p.paper_exposed = paper_exposed
        last_reg = registration_used
        if self._config.page_model and p.pass_voted:
            # Modelo persistente da pagina (draw.ink.page_model):
            # observa a passada na grade da ancora e so entrega
            # evidencia onde ha referencia semeada; o que nao e
            # conteudo novo contra ela (pauta, desenho antigo,
            # sombra) e suprimido antes do voto. Passada abstida
            # nao observa: geometria em fluxo envenenaria a
            # referencia. A 1a passada e a de identidade entram na
            # propria geometria — que e a da ancora por definicao.
            # Sem cantos (last_reg None), a passada ja esta na propria
            # grade (transformacao identidade).
            if (
                self._pm_epoch != p.pm_epoch
                or self._page_model is None
                or self._page_model.shape != p.prob.shape
            ):
                self._page_model = page_model.PageModel(p.prob.shape)
                self._pm_epoch = p.pm_epoch
                p.next_baseline_ink = np.zeros(p.prob.shape, dtype=bool)
            elif (
                p.pm_baseline_ink is not None
                and p.pm_baseline_ink.shape == p.prob.shape
            ):
                p.next_baseline_ink = p.pm_baseline_ink
            else:
                p.next_baseline_ink = np.zeros(p.prob.shape, dtype=bool)
            pm = self._page_model
            p.pm_confirm_obs = pm.confirm_obs
            gray = (
                p.warped
                if p.warped.ndim == 2
                else cv2.cvtColor(p.warped, cv2.COLOR_BGR2GRAY)
            )
            reg_bgr = p.warped if p.warped.ndim == 3 else None
            pm_matrix = None
            size = (p.prob.shape[1], p.prob.shape[0])
            if (
                last_reg is not None
                and last_reg[0] is not None
                and last_reg[1] is not None
            ):
                pm_matrix = registration._registration_matrix(
                    last_reg[0], last_reg[1], p.prob.shape
                )
                if not registration._registration_is_identity(pm_matrix):
                    gray = cv2.warpPerspective(
                        gray,
                        pm_matrix,
                        size,
                        flags=cv2.INTER_LINEAR,
                        borderValue=0,
                    )
                    if reg_bgr is not None:
                        reg_bgr = cv2.warpPerspective(
                            reg_bgr,
                            pm_matrix,
                            size,
                            flags=cv2.INTER_LINEAR,
                            borderValue=0,
                        )
            gray = np.asarray(gray, dtype=np.float32)
            if p.motion_decision == "correct" and p.used_registered_prob:
                gray = p.motion_fix.warp_back(gray)
                if reg_bgr is not None:
                    reg_bgr = p.motion_fix.warp_back(reg_bgr)
            pm_coverage = None
            if p.coverage_page_static is not None:
                pm_coverage = p.coverage_page_static
                if pm_matrix is not None and not registration._registration_is_identity(
                    pm_matrix
                ):
                    pm_coverage = (
                        cv2.warpPerspective(
                            pm_coverage.astype(np.uint8),
                            pm_matrix,
                            size,
                            flags=cv2.INTER_NEAREST,
                            borderValue=0,
                        )
                        > 0
                    )
                if p.motion_decision == "correct" and p.used_registered_prob:
                    pm_coverage = (
                        p.motion_fix.warp_back(
                            pm_coverage.astype(np.uint8), nearest=True
                        )
                        > 0
                    )
            elif p.coverage_cam is not None:
                if p.coverage_matches_last_reg and p.reg_coverage is not None:
                    pm_coverage = p.reg_coverage
                elif p.corners is not None:
                    page_mat = registration._corners_page_matrix(
                        p.corners, size[0], size[1]
                    )
                    if (
                        pm_matrix is not None
                        and not registration._registration_is_identity(pm_matrix)
                    ):
                        pm_coverage = (
                            cv2.warpPerspective(
                                p.coverage_cam.astype(np.uint8),
                                pm_matrix @ page_mat,
                                size,
                                flags=cv2.INTER_NEAREST,
                                borderValue=0,
                            )
                            > 0
                        )
                    else:
                        pm_coverage = (
                            cv2.warpPerspective(
                                p.coverage_cam.astype(np.uint8) * 255,
                                page_mat,
                                size,
                                flags=cv2.INTER_LINEAR,
                                borderValue=0,
                            )
                            > 127
                        )
                    if p.motion_decision == "correct" and p.used_registered_prob:
                        pm_coverage = (
                            p.motion_fix.warp_back(
                                pm_coverage.astype(np.uint8), nearest=True
                            )
                            > 0
                        )
            obs_visible = (
                pm_coverage
                if p.visible is None
                else (p.visible if pm_coverage is None else (p.visible & pm_coverage))
            )
            pm_sampled = p.sampled
            if pm_sampled is None and p.used_registered_prob:
                # A pegada nao depende da guarda de borda: sem ela
                # _pass_sampled devolve None e o padding do registro
                # semearia a referencia de pagina limpa em regiao que
                # a passada nunca observou. Quando ha correcao afim
                # aplicada a evidencia, a pegada calculada aqui segue
                # o mesmo warp_back (com a guarda ligada isso ja foi
                # feito em _pass_register_motion_coverage). So na
                # passada REGISTRADA: quando o voto sai na geometria
                # propria (primeira passada ou folha nova apos troca)
                # nao ha registro nem padding — recomputar a pegada
                # com a ancora velha podaria pixels validos da folha
                # nova na primeira observacao de baseline dela.
                pm_sampled = registration._observed_footprint(
                    p.anchor, p.registration_corners, p.corners, p.prob.shape
                )
                if p.motion_decision == "correct" and p.used_registered_prob:
                    pm_sampled = (
                        p.motion_fix.warp_back(
                            np.ones(p.prob.shape, np.uint8)
                            if pm_sampled is None
                            else pm_sampled.astype(np.uint8),
                            nearest=True,
                        )
                        > 0
                    )
            if pm_sampled is not None:
                # O PageModel aprende so onde a passada amostrou de
                # verdade: o padding do registro nao pode semear a
                # referencia de pagina limpa em regioes que o
                # acumulador rejeita pela pegada.
                obs_visible = (
                    pm_sampled if obs_visible is None else (obs_visible & pm_sampled)
                )
            # Papel coberto agora pela mao (mascara fisica), ainda sem
            # referencia: conta para a prontidao. A mascara e a de
            # ANTES do veto fotometrico — pixel visivel que a guarda
            # recusa como papel (logo escuro, impressao densa) nunca
            # semeia e fica de fora: esperar por ele seria esperar
            # para sempre (o "fantasma" de inicializacao, medido no
            # merge do #161 no PR #144).
            hand_covered = None
            if physical_visible is not None:
                hand_covered = (
                    ~physical_visible
                    if pm_coverage is None
                    else pm_coverage & ~physical_visible
                )
            # Dominio observavel antes dos vetos fotometricos: a
            # diferenca para o obs_visible final e o que a guarda
            # negou nesta passada — mao/oclusor que a mascara fisica
            # nao viu e que ainda nao tem referencia conta para a
            # prontidao com prazo curto (Devin Review do PR #144).
            obs_domain = obs_visible
            if self._config.paper_guard and p.warped.dtype == np.uint8:
                exposed_mask = pm.exposed(
                    gray,
                    bgr=reg_bgr,
                    visible=obs_visible,
                    hold_passes=self._config.occlusion_hold_passes,
                )
                p.pm_recovering_px = pm.recovering_px
                p.pm_hold_pending_px = int(
                    np.count_nonzero(
                        pm.hold_mask
                        & ~p.confirmed_ink
                        & (p.prob > self._accumulator.threshold)
                    )
                )
                p.visible = (
                    exposed_mask if p.visible is None else (p.visible & exposed_mask)
                )
                obs_visible = (
                    exposed_mask
                    if obs_visible is None
                    else (obs_visible & exposed_mask)
                )
            if p.paper_ok is not None:
                obs_visible = (
                    p.paper_ok if obs_visible is None else (obs_visible & p.paper_ok)
                )
            if p.pm_rezero and (obs_visible is None or bool(np.any(obs_visible))):
                # Papel utilizavel nesta passada: o modelo recomeca aqui e
                # o canvas e zerado no commit. Sem papel (sombra, mao,
                # guarda rejeitando tudo) o pedido segue pendente e o
                # canvas atual fica intacto.
                pm = page_model.PageModel(p.prob.shape)
                self._page_model = pm
                p.pm_epoch = (p.pm_epoch[0] + 1, p.pm_epoch[1] + 1)
                self._pm_epoch = p.pm_epoch
                p.next_baseline_ink = np.zeros(p.prob.shape, dtype=bool)
                p.pm_confirm_obs = pm.confirm_obs
                p.pm_rezero_applied = True
            cover_held = None
            if obs_domain is not None:
                blocked = (
                    obs_domain if obs_visible is None else obs_domain & ~obs_visible
                )
                cover_held = pm.covered_unseeded(
                    blocked,
                    hold_passes=self._config.unseeded_hold_passes,
                )
            pm.observe(
                gray,
                visible=obs_visible,
                paper=(
                    p.prob <= registration._REMOVAL_LOW_PROB
                    if obs_visible is None
                    else obs_visible & (p.prob <= registration._REMOVAL_LOW_PROB)
                ),
                bgr=(
                    reg_bgr
                    if (reg_bgr is not None and p.warped.dtype == np.uint8)
                    else None
                ),
            )
            p.pm_seeding_remaining = pm.seeding_remaining
            p.visible = pm.admissible(p.visible)
            obs_admissible = pm.admissible(obs_visible)
            confirmed_ink = p.confirmed_ink
            # `returned` so interessa onde ha traco confirmado sobre
            # impressao semeada (raro): sem candidato, o custo e o de sempre.
            erase_candidates = confirmed_ink & p.next_baseline_ink & pm.seeded
            if np.any(erase_candidates):
                novel_mask, returned_mask = pm.appearance(gray)
            else:
                novel_mask, returned_mask = pm.novel(gray), None
            suppress = pm.seeded & ~novel_mask & ~confirmed_ink
            voting_px = p.visible & (p.prob > self._accumulator.threshold)
            p.reg_info["pm_seeded_frac"] = pm.seeded_fraction
            p.reg_info["pm_novel_px"] = int(
                np.count_nonzero(novel_mask & voting_px & ~confirmed_ink)
            )
            p.reg_info["pm_suppressed_px"] = int(np.count_nonzero(suppress & voting_px))
            p.reg_info["pm_pending_px"] = int(np.count_nonzero(pm.pending(obs_visible)))
            # Sem referencia e semeavel: o que a mao cobre agora mais o
            # visivel ainda em semeadura. E o que a interface precisa para
            # nao dizer "pronto" antes de a mao liberar uma regiao.
            unseeded_px = int(np.count_nonzero(pm.seeding_pending(obs_visible)))
            if hand_covered is not None:
                unseeded_px += int(np.count_nonzero(hand_covered & ~pm.seeded))
            if cover_held is not None:
                unseeded_px += int(np.count_nonzero(cover_held))
            p.reg_info["pm_unseeded_px"] = unseeded_px
            baseline_seen = (
                suppress & obs_admissible & (p.prob > self._accumulator.threshold)
            )
            if np.any(baseline_seen):
                p.next_baseline_ink[baseline_seen] = True
            baseline_erased = obs_admissible & (
                p.prob <= registration._REMOVAL_LOW_PROB
            )
            if np.any(baseline_erased):
                p.next_baseline_ink[baseline_erased] = False
            # Traco confirmado sobre impresso que voltou a aparencia
            # semeada (apagado): a impressao ainda pontua como tinta e o
            # removedor nunca veria ausencia. Zera a evidencia ali para
            # que a remocao conte. Usa a tolerancia APERTADA de
            # PageModel.returned, nao ~novel: traco que so perdeu
            # contraste (reflexo) segue visto como tinta. As salvaguardas
            # do acumulador (visibilidade, testemunha, passadas
            # consecutivas) seguem.
            zero = suppress
            returned_seen = (
                None
                if returned_mask is None
                else erase_candidates & returned_mask & obs_admissible
            )
            if returned_seen is not None and np.any(returned_seen):
                # O retorno a aparencia semeada vale evidencia de
                # ausencia so fora dos pixels sob suspeita de oclusor
                # nesta passada: tile anomalo (luma/croma — a regua
                # escura que imita a guia) ou tinta inexplicada em
                # posicao nao aprendida (as marcas dela). A impressao
                # semeada que segue lendo tinta (next_baseline_ink) NAO
                # e marca nova: em tile denso demais para amostrar papel
                # a guarda nunca aprende posicoes e o veto cego
                # congelava a remocao (Devin Review do PR #144). O veto
                # de pixel "confirmada sobre pixel escuro" e o unico
                # que o casamento com a referencia do PageModel
                # sobrepoe.
                if p.exposure is not None:
                    anomalous_px = p.exposure_guard.anomalous_px
                    suspect = p.exposure_guard.suspect_ink
                    # A guarda mede o warp desta passada; o veto e o
                    # retorno julgados abaixo vivem na grade da ancora.
                    # Sem a projecao, pagina que desloca ou rotaciona
                    # deixa o veto cobrindo os pixels errados — e a
                    # regua escura sob o traco confirmado deixava de
                    # suprimir o retorno, virando ausencia provada
                    # (Devin Review do PR #144). Mesmo par (ancora,
                    # registro) e mesmo motion_fix das mascaras da
                    # avaliacao acima.
                    if last_reg is not None:
                        matrix = registration._registration_matrix(
                            last_reg[0], last_reg[1], p.prob.shape
                        )
                        if not registration._registration_is_identity(matrix):
                            if anomalous_px is not None:
                                anomalous_px = _warp_exposure_mask(
                                    anomalous_px, matrix, p.prob.shape
                                )
                            if suspect is not None:
                                suspect = _warp_exposure_mask(
                                    suspect, matrix, p.prob.shape
                                )
                    motion_fix = (
                        p.motion_fix
                        if p.motion_decision == "correct" and p.used_registered_prob
                        else None
                    )
                    if motion_fix is not None:
                        if anomalous_px is not None:
                            anomalous_px = (
                                motion_fix.warp_back(
                                    anomalous_px.astype(np.uint8), nearest=True
                                )
                                > 0
                            )
                        if suspect is not None:
                            suspect = (
                                motion_fix.warp_back(
                                    suspect.astype(np.uint8), nearest=True
                                )
                                > 0
                            )
                    if anomalous_px is not None:
                        veto = anomalous_px.copy()
                        if suspect is not None:
                            veto |= suspect & ~p.next_baseline_ink
                        # Tile anomalo que segue parecendo a propria
                        # semente nao esta coberto: a impressao densa
                        # escurece a media do tile sem mudar a
                        # aparencia de nenhum pixel semeado. Sob uma
                        # capa ou regua escura, os pixels de papel
                        # semeados viram `novel` e a fracao de
                        # casamento despenca — o veto permanece
                        # (Devin Review do PR #144).
                        tile = occlusion._REFERENCE_TILE_PX
                        grid = (
                            (p.prob.shape[1] + tile - 1) // tile,
                            (p.prob.shape[0] + tile - 1) // tile,
                        )
                        seeded_frac = cv2.resize(
                            pm.seeded.astype(np.float32),
                            grid,
                            interpolation=cv2.INTER_AREA,
                        )
                        matched_frac = cv2.resize(
                            (pm.seeded & ~novel_mask).astype(np.float32),
                            grid,
                            interpolation=cv2.INTER_AREA,
                        )
                        consistent = (
                            cv2.resize(
                                (
                                    (seeded_frac > 0.0)
                                    & (
                                        matched_frac
                                        >= seeded_frac * _RETURNED_TILE_MATCH
                                    )
                                ).astype(np.uint8),
                                (p.prob.shape[1], p.prob.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            > 0
                        )
                        veto &= ~consistent
                        returned_seen = returned_seen & ~veto
                zero = zero | returned_seen
                # O retorno a aparencia semeada tambem prova exposicao:
                # a impressao escura sob o traco apagado cai no veto
                # dark_ink da guarda ("tinta confirmada sobre pixel
                # escuro") e a remocao congelava para sempre (merge do
                # #161 no PR #144). O casamento com a referencia do
                # PageModel e a evidencia de que nada cobre o pixel —
                # abertura so pixel, nunca sobre veto de tile.
                if p.exposure is not None and np.any(returned_seen):
                    exposed_mask = p.paper_exposed
                    if isinstance(exposed_mask, (bool, np.bool_)):
                        exposed_mask = np.full(p.prob.shape, exposed_mask, dtype=bool)
                    else:
                        exposed_mask = exposed_mask.copy()
                    exposed_mask[returned_seen] = True
                    p.paper_exposed = exposed_mask
            if np.any(zero):
                p.prob = np.where(zero, 0.0, p.prob).astype(np.float32)

    def _pass_vote_and_publish(self, p):
        """Voto, acumulador e publicacao sob _condition: comita estado,
        roda acumulador, carimba pass_ended_at e publica telemetria."""
        delta_job = None
        self._last_registration = p.last_reg
        if p.motion_abstain:
            self._motion_vetoes_total += 1
        elif p.motion_decision == "correct" and p.used_registered_prob:
            self._motion_corrections_total += 1
        elif p.motion_decision == "unregistered" and p.used_registered_prob:
            self._motion_unregistered_total += 1
        elif p.motion_decision == "fail_open" and p.used_registered_prob:
            self._motion_fail_open_total += 1
        if p.pm_confirm_obs > 0:
            self._pm_confirm_obs = p.pm_confirm_obs
        if p.next_baseline_ink is not None:
            self._pm_baseline_ink = p.next_baseline_ink
        if self._config.page_model and "pm_seeded_frac" in p.reg_info:
            self._pm_status = (
                p.pm_epoch,
                float(p.reg_info["pm_seeded_frac"]),
                int(p.reg_info.get("pm_unseeded_px", 0)),
            )
            if (
                self._pm_rezero_seeding
                and p.pm_seeding_remaining == 0
                and p.reg_info["pm_seeded_frac"] > 0.0
            ):
                self._pm_rezero_seeding = False
        guard_corners = (
            self._anchor_corners if self._anchor_corners is not None else p.corners
        )
        guard_mask = self._cached_border_guard(p.prob.shape, guard_corners)
        # A pegada amostrada entra no dominio valido so com a
        # guarda ativa: sem ela, a abstencao legada sem mascara
        # de visibilidade continua valendo (zeros de
        # preenchimento do registro nao viram evidencia
        # negativa).
        valid_mask = guard_mask
        if valid_mask is not None and p.sampled is not None:
            valid_mask = valid_mask & p.sampled
        result = self._accumulator.add(
            p.prob,
            p.visible,
            valid_mask=valid_mask,
            paper_exposed=p.paper_exposed,
            removal_allowed=(
                not p.identity_owned
                and not p.arbitration_abstain
                and not p.follow_abstain
                and p.corners is not None
                and p.watch_step_px is not None
                and p.watch_step_px <= anchor_mod._ARB_SETTLE_PX
                and p.reg_info.get("registration_displacement_px", float("inf"))
                <= anchor_mod._ANCHOR_MIGRATION_DISPLACEMENT_PX
            ),
        )
        # A provisoria que a tela ve: so componentes grandes
        # (issue #134). A diff pixel a pixel republicava o
        # painel a cada poeira oscilando e zerava a contagem
        # de atividade — com o sinal de mao ausente, o
        # flicker mantinha a cadencia cheia para sempre.
        provisional_display = _provisional_for_display(
            result["provisional"], self._config.provisional_min_px
        )
        previous_display = self._display_provisional
        provisional_display_changed = (
            bool(np.any(provisional_display))
            if previous_display is None
            else previous_display.shape != provisional_display.shape
            or bool(np.any(provisional_display != previous_display))
        )
        if result["confirmed_changed"] or provisional_display_changed or result["turn"]:
            self._still_passes = 0
            self._activity_quiet_passes = 0
        else:
            self._still_passes += 1
            self._activity_quiet_passes += 1
        # Contadores do descanso total: so a saida
        # CONFIRMADA interessa (provisoria tremula com o
        # warp e nao pode segurar a trabalhadora acordada).
        if self._spent_confirm_pass and self._hand_state is False:
            pm_cap = self._pm_confirm_obs if self._config.page_model else 0
            pm_needed = (
                (self._confirm_passes + 1)
                if (self._config.page_model and not p.pass_voted)
                else p.pm_seeding_remaining
            )
            hold_needed = (
                self._recovery_wait_open()
                or p.pm_hold_pending_px > 0
                or p.pm_recovering_px >= _RECOVERY_MIN_PX
            )
            max_skips = (
                2 * max(votes._INK_VOTE_WINDOW, pm_cap)
                + self._config.occlusion_hold_passes
                + 1
            )
            if self._config.page_model:
                max_skips = max(max_skips, 2 * pm_cap + votes._INK_VOTE_WINDOW)
            if (
                hold_needed
                and self._confirm_passes < self._post_exit_confirm_passes()
                and self._confirm_masked_skips < max_skips
            ):
                # Pos-saida da mao: enquanto a espera de recuperacao da
                # guarda segue aberta (streak de preenchimento contando
                # ou tile em hold pos-anomalia, ainda dentro do prazo
                # de _recovery_wait_open) ou o PageModel segura oclusor,
                # as passadas gastas ate aqui nao deram voto limpo.
                # Re-arma a janela cheia (limitado pelo teto de
                # devolucoes) para que, solto o hold, haja uma janela
                # inteira de votos antes do descanso.
                self._confirm_masked_skips += 1
                self._confirm_passes = self._post_exit_confirm_passes()
            elif (
                self._config.page_model
                and pm_needed > self._confirm_passes
                and self._confirm_masked_skips < max_skips
            ):
                self._confirm_masked_skips += 1
                self._confirm_passes = min(max(self._confirm_passes, pm_needed), pm_cap)
        self._generation_passes += 1
        if result["confirmed_changed"] or result["turn"] or p.motion_abstain:
            # Passada vetada nao e quietude: a folha esta fora do
            # lugar e a tinta nova ainda nao entrou — descansar aqui
            # deixava o traco sem confirmar ate outro evento.
            self._quiet_passes = 0
        else:
            self._quiet_passes += 1
        self._passes_total += 1
        p.pass_ended_at = self._clock()
        self._last_pass_ms = (p.pass_ended_at - p.pass_started) * 1000.0
        self._provisional_px = result["provisional_px"]
        self._confirmed_px = result["confirmed_px"]
        self._retracted_px_total = result["retracted_px_total"]
        publish_confirmed = (
            p.first_pass
            or p.identity_timed_out
            or p.identity_decision == "new_page"
            or p.identity_decision == "same_page"
            or result["confirmed_changed"]
            # Migracao: o canvas migrado nao "muda" para o
            # voto, mas a tela mostraria a grade velha.
            or p.force_publish
        )
        if publish_confirmed:
            self._version += 1
            self._published = (
                p.generation,
                self._version,
                result["confirmed"],
            )
            # So a referencia (result["confirmed"] ja e uma copia da
            # passada): o calculo do delta roda depois de soltar o lock.
            delta_job = (
                p.generation,
                self._version,
                result["confirmed"],
                p.captured_at,
                None if self._anchor_corners is None else self._anchor_corners.copy(),
            )
        if publish_confirmed or provisional_display_changed:
            self._display_version += 1
            self._published_display = (
                p.generation,
                self._display_version,
                result["confirmed"],
                provisional_display,
            )
            self._display_provisional = provisional_display
        # Telemetria por etapa (loop v2) sobre o caminho em
        # camadas: um registro por passada; publishes conta
        # so as passadas que publicam canvas confirmado
        # novo, e o campo de mascara da passada vira
        # confirmed_px (a uniao nao existe mais).
        published_at = self._clock()
        self._telemetry_passes += 1
        if publish_confirmed:
            self._telemetry_publishes += 1
        self._telemetry_records.append(
            {
                **p.reg_info,
                "sequence": self._telemetry_passes,
                "pass_gap_s": None
                if self._previous_pass_started_at is None
                else p.pass_started_at - self._previous_pass_started_at,
                "corner_step_px": p.watch_step_px,
                "quarantine_reason": p.quarantine_reason,
                "quarantine_frames": p.quarantine_frames,
                "deferred": p.identity_deferred,
                "generation": p.generation,
                "version": self._version,
                "page_identity_decision": p.identity_decision,
                "page_identity_recall": p.identity_recall,
                "page_identity_containment": p.identity_containment,
                "page_identity_reference_px": p.identity_reference_px,
                "page_identity_visible_reference_px": (p.identity_visible_reference_px),
                "page_identity_geometry_registered": (p.identity_geometry_registered),
                "submitted_at": p.submitted_at,
                "pass_started_at": p.pass_started_at,
                "pass_ended_at": p.pass_ended_at,
                "published_at": published_at,
                "queue_wait_s": p.pass_started_at - p.submitted_at,
                "pass_s": p.pass_ended_at - p.pass_started_at,
                "submit_to_publish_s": published_at - p.submitted_at,
                "interval_s": p.scheduled_interval,
                "hand_state": p.scheduled_hand_state,
                "activity_quiet_passes": self._activity_quiet_passes,
                "mask_px": int(np.count_nonzero(p.prob > self._accumulator.threshold)),
                "confirmed_px": result["confirmed_px"],
                "new_confirmed_px": result["new_confirmed_px"],
                "removed_confirmed_px": result["removed_confirmed_px"],
                "visible_fraction": 1.0
                if p.visible is None
                else np.count_nonzero(p.visible) / p.visible.size,
            }
        )
        self._previous_pass_started_at = p.pass_started_at
        return delta_job

    def stats(self):
        """Contadores para a telemetria: passadas e custo da ultima."""
        with self._condition:
            return {
                "passes_total": self._passes_total,
                "pass_ms": (
                    None if self._last_pass_ms is None else round(self._last_pass_ms, 1)
                ),
                "canvas_px": self._confirmed_px,
                "provisional_px": self._provisional_px,
                "retracted_px_total": self._retracted_px_total,
                "motion_vetoes_total": self._motion_vetoes_total,
                "motion_corrections_total": self._motion_corrections_total,
                "motion_fail_open_total": self._motion_fail_open_total,
                "motion_unregistered_total": self._motion_unregistered_total,
                "worker_mode": (
                    "rest"
                    if self._at_rest()
                    else "active"
                    if self._hand_state
                    else "idle"
                    if self._hand_state is False
                    else "unknown"
                ),
                "resting": self._at_rest(),
                "hand_present": self._hand_state,
            }

    @property
    def error(self):
        """A excecao que matou a trabalhadora, ou None."""
        with self._condition:
            return self._error

    @property
    def anchor_corners(self):
        """Âncora publicada no frame atual (4×2) — cópia defensiva.

        Superfície de observabilidade para eval/replay: os call sites
        liam ``_anchor_corners`` direto; a propriedade formaliza o
        acesso sem expor o array interno à mutação.
        """
        with self._condition:
            return None if self._anchor_corners is None else self._anchor_corners.copy()

    @property
    def last_registration(self):
        """Tripla (âncora, quad de registro, quad submetido) — cópias.

        A grade armazenada e indexada pela retificacao da ancora,
        mas cada evidencia entra pela matriz P(ancora)·P(reg)⁻¹ da
        SUA passada, aplicada sobre o warp do quad SUBMETIDO: a
        folha pode andar entre o pouso da ancora e a passada
        seguinte, e o deadband/pouso/follow trocam os dois primeiros
        no meio do caminho. Avaliadores projetam o GT pelo mesmo
        caminho; a ancora sozinha nao basta. None quando a ultima
        passada nao votou (abstida por arbitragem, follow ou
        consenso de identidade) ou antes da primeira.
        """
        with self._condition:
            if self._last_registration is None:
                return None
            anchor, registration, submitted = self._last_registration
            return (
                None if anchor is None else anchor.copy(),
                None if registration is None else registration.copy(),
                None if submitted is None else submitted.copy(),
            )

    @property
    def canvas_epoch(self):
        """Época do conteúdo do acumulador — sobe a cada reset.

        Registros de registro de passadas anteriores ao ultimo
        reset nao descrevem mais o canvas (a tinta que eles
        projetavam foi descartada); consumidores como o
        eval_capture descartam os registros quando a epoca muda.
        """
        with self._condition:
            return self._canvas_epoch

    def stop(self):
        # O teto cobre uma passada em voo: a trabalhadora so ve o
        # stop entre passadas, e sair com ela dentro do ONNX
        # derruba o processo na finalizacao (visto na bancada
        # desta VM, passada de segundos). No Ryzen a passada e
        # 168 ms e o join volta muito antes do teto.
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=_PAGE_CLEAN_STOP_TIMEOUT_SECONDS)
