"""Trabalhadora pagina->tinta: fila, passadas, cadencia e voto."""

import threading
import time
from collections import deque

import cv2
import numpy as np

from .. import config as paper_config
from ..contracts import InkDelta, InkDisplaySnapshot, InkSnapshot
from . import accumulator, identity, motion, registration, ruling
from . import anchor as anchor_mod
from .config import InkConfig

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


# Evento geometrico que acorda o descanso (frente do cenario E3,
# InkConfig.follow_move do preset de producao): a premissa do
# descanso sem mao e "a pagina nao muda" — mas a folha que desliza
# sem mao na cena mostra exatamente o contrario (medido em
# 2026-11-12: o warp congelado seguiu vendo mesa e a trabalhadora
# dormiu no meio do movimento, porque submit nao acorda). Um quad
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


def _bbox_of(mask):
    """(y0, y1, x0, x1) meio-aberto do que e True em mask, ou None."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


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
    ):
        self._page_clean = page_clean
        self._config = config if config is not None else InkConfig()
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
        # Relogio monotonico injetavel: o padrao e o do processo;
        # replays com pump() passam o relogio da cena.
        self._clock = clock
        self._hand_masker = hand_masker
        self._condition = threading.Condition()
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
        # Arbitragem de ancora (InkConfig.anchor_arbitration): o
        # evento aberto, o streak de assentamento com seus escores
        # por passada e o snapshot do canvas pre-evento.
        self._arb_event_open = False
        self._arb_streak_quads = []
        self._arb_scores = []
        self._arb_snapshot = None
        self._arb_actions = 0
        self._arb_events = deque(maxlen=128)
        # Mobilidade da ancora (InkConfig.anchor_follow): o evento
        # de divergencia assentada, o streak de quads assentados e
        # a retencao por veredito spurious da arbitragem.
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
        self._identity_recovery_requested = False
        self._identity_defer_count = 0
        self._identity_switch_votes = 0
        # Degrau entre quads consecutivos (a entrada de cadencia
        # geometrica do follow e do portao de remocao).
        self._previous_corners = None
        self._accumulator = accumulator.LayeredInkAccumulator(
            threshold=self._config.seen_threshold, config=self._config
        )
        self._last_pass_start = None
        self._passes_total = 0
        self._last_pass_ms = None
        self._provisional_px = 0
        self._confirmed_px = 0
        self._retracted_px_total = 0
        self._published_display = None
        self._display_version = 0
        self._hand_state = None
        self._confirm_once = False
        self._still_passes = 0
        # Descanso total (so a trabalhadora e o new_generation
        # tocam): passadas desta geracao e passadas seguidas sem
        # confirmar nada novo.
        self._generation_passes = 0
        self._quiet_passes = 0
        # Passadas seguidas sem a saida (confirmada ou provisoria)
        # mudar, contadas SO pelas passadas — nunca zeradas por
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
        # Estado do evento geometrico (frente do cenario E3,
        # InkConfig.follow_move): True enquanto o quad submetido
        # segue fora da moldura da ancora — o periodo em que o
        # descanso nao pode fechar.
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
        with self._condition:
            self._identity_recovery_requested = bool(preserve_identity)
            self._identity_defer_count = 0
            self._identity_switch_votes = 0
            self._previous_corners = None
            if preserve_identity:
                reference = self._accumulator._confirmed
                self._identity_reference = (
                    None if reference is None else reference.copy()
                )
                self._identity_reference_corners = (
                    None
                    if self._anchor_corners is None
                    else self._anchor_corners.copy()
                )
            else:
                self._identity_reference = None
                self._identity_reference_corners = None
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
            self._generation_passes = 0
            self._quiet_passes = 0
            self._activity_quiet_passes = 0
            self._pending = None
            self._published = None
            self._published_display = None
            self._delta_ring.clear()
            self._delta_prev = None
            self._condition.notify_all()
            return self._generation

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

        visible e a mascara de visibilidade DESTE warp (True onde
        a camera ve a pagina; False sob a mao), na mesma grade do
        warp. Onde ela e False, a passada se abstem: a evidencia
        nao vota, a virada nao julga e a decisao de identidade so
        olha a tinta anterior visivel. Formato divergente do warp
        e tratado como ausente, pela mesma razao dos cantos.

        coverage e a cobertura de folha DESTE warp (True onde a
        retificacao cobre papel de verdade; False onde o quad
        admitiu o que esta fora da folha — mesa que entra por
        quad largo ou atrasado depois de movimento). Onde ela e
        False, a passada so vota sobre a tinta que a ancora ja
        conhece: re-confirmar e retrair o que existe, sim; tinta
        nova nascendo fora da folha, nunca. Assim o defeito
        semantico da mascara (traco denso virando "fundo" numa
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
            if visible is not None and visible.shape != warped.shape[:2]:
                visible = None
        if coverage is not None:
            try:
                coverage = np.asarray(coverage, dtype=bool)
            except (TypeError, ValueError):
                coverage = None
            if coverage is not None and coverage.shape != warped.shape[:2]:
                coverage = None
        with self._condition:
            if self._stopped or generation != self._generation:
                return
            if (
                self._config.follow_move
                and corners is not None
                and self._anchor_corners is not None
            ):
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

    def set_hand_presence(self, present, moved=False):
        """Alimenta a porta de ociosidade: True, False ou None.

        A transicao para False arma exatamente uma passada de
        confirmacao: a mao saiu, a pagina assentou, e essa passada
        fecha o voto do que foi escrito. Depois dela, sem mao, a
        trabalhadora sonda a no maximo 1 Hz enquanto a saida
        confirmada ainda muda — e entra em descanso total (zero
        passadas) quando ela para de mudar; ver _at_rest.

        moved=True e o sinal de atividade da mascara (ver
        _MediaPipeHandPresence.sample): reseta o relogio de
        quietude da mao e acorda o descanso com a mao presente.
        Transicao de presenca tambem conta como movimento: a mao
        chegando ou saindo nunca e mao parada.
        """
        with self._condition:
            if present is False and self._hand_state is not False:
                self._confirm_once = True
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
        if (
            self._config.removal
            and self._accumulator._removal_streak is not None
            and np.any(self._accumulator._removal_streak)
        ):
            return False
        if self._identity_recovery_requested and self._identity_reference is not None:
            return False
        if (
            self._generation_passes < _REST_MIN_GENERATION_PASSES
            or self._quiet_passes < _REST_QUIET_PASSES
        ):
            return False
        if self._hand_state is False:
            return not self._confirm_once
        if self._hand_state is True:
            return self._hand_still()
        return False

    def _pass_interval(self):
        if self._at_rest():
            return None
        if self._hand_state is False and not self._confirm_once:
            return _IDLE_PROBE_SECONDS
        if (
            self._hand_state is None
            and self._still_passes >= _NO_HAND_REST_AFTER_STILL_PASSES
        ):
            return _IDLE_PROBE_SECONDS
        if self._config.pass_interval_s is not None:
            return self._config.pass_interval_s
        return _PAGE_CLEAN_INTERVAL_SECONDS

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
                "anchor_arbitration_enabled": self._config.anchor_arbitration,
                "submits": self._telemetry_submits,
                "submits_replaced": self._telemetry_replaced,
                "passes": self._telemetry_passes,
                "publishes": self._telemetry_publishes,
                "passes_skipped_stale": self._telemetry_skipped_stale,
                "records": [dict(r) for r in self._telemetry_records],
                "page_identity_events": [
                    dict(event) for event in self._identity_events
                ],
            }
            if self._config.reg_deadband or self._reg_deadband_releases:
                snapshot["reg_deadband_releases"] = self._reg_deadband_releases
            if self._config.anchor_arbitration or self._arb_actions:
                snapshot["anchor_arbitration_actions"] = self._arb_actions
                snapshot["anchor_arbitration_events"] = [
                    dict(event) for event in self._arb_events
                ]
            if self._config.anchor_follow or self._anchor_reanchors:
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
                        # Descanso total: espera sem prazo por um
                        # evento que acorde (a mao voltar, a mao
                        # presente se mexer, geracao nova, stop).
                        # Submit NAO acorda: ele so troca o
                        # pendente, que e reavaliado quando um
                        # evento de verdade chegar. Sem isto a
                        # sonda de 1 Hz rodava para sempre sobre a
                        # pagina parada e o voto acumulava camadas.
                        self._condition.wait()
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
        if self._confirm_once:
            self._confirm_once = False
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
                    return False
                elapsed = self._clock() - self._last_pass_start
                if elapsed < interval - _SYNC_CADENCE_EPS_SECONDS:
                    return False
            job = self._take_pending_locked()
        self._run_pass(job)
        return True

    def _run_pass(self, job):
        """Uma passada do modelo + voto. False quando a trabalhadora para."""
        (
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
            if generation != self._generation:
                self._telemetry_skipped_stale += 1
                return True
        # Uma excecao do modelo (shape errado, sessao morta, ONNX
        # sem memoria) matava esta thread em silencio: o canvas
        # congelava no ultimo quadro e nada era impresso. O erro
        # fica guardado, e o laco principal o reporta.
        pass_started = self._clock()
        identity_decision = None
        identity_recall = None
        identity_containment = None
        identity_reference_px = None
        identity_visible_reference_px = None
        identity_deferred = False
        identity_timed_out = False
        identity_switch_pending = False
        quarantine_reason = None
        quarantine_frames = 0
        identity_geometry_registered = False
        reg_anchor_median = self._config.reg_anchor_median
        reg_deadband = self._config.reg_deadband
        anchor_arbitration = self._config.anchor_arbitration
        anchor_follow = self._config.anchor_follow
        force_publish = False
        reg_info = {}
        try:
            if visible is None and self._hand_masker is not None:
                hand_mask = self._hand_masker.mask(warped)
                if hand_mask is not None:
                    visible = hand_mask == 0
            ink_prob = getattr(self._page_clean, "ink_prob", None)
            if ink_prob is not None:
                prob = ink_prob(warped)
            else:
                prob = (self._page_clean.ink_mask(warped) > 0).astype(np.float32)
            if self._ruling is not None:
                if generation != self._ruling_generation:
                    self._ruling.reset()
                    self._ruling_generation = generation
                prob = self._ruling.apply(prob, visible)
            # Cobertura de folha (cobertura de folha do loop) viaja em
            # canal proprio ate depois do registro: o veto nao e
            # cego nem se confunde com oclusao de mao — ver o
            # gate de re-confirmacao abaixo, no espaco da ancora.
            pass_ended_at = self._clock()
            with self._condition:
                if self._stopped:
                    return False
                if generation != self._generation:
                    return True
                first_pass = generation != self._worker_generation
                recovering = self._identity_reference is not None and (
                    first_pass or self._identity_recovery_requested
                )
                anchor = (
                    self._identity_reference_corners
                    if recovering
                    else corners
                    if first_pass
                    else self._anchor_corners
                )
                registration_corners = corners
                if not recovering and corners is not None:
                    anchor_rebased = False
                    if (
                        reg_anchor_median
                        and not first_pass
                        and 0
                        < len(self._anchor_samples)
                        < registration._REG_ANCHOR_MEDIAN_SAMPLES
                    ):
                        self._anchor_samples.append(corners.copy())
                        self._anchor_corners = registration._median_anchor(
                            self._anchor_samples
                        )
                        anchor = self._anchor_corners
                        anchor_rebased = True
                    if reg_anchor_median:
                        reg_info["reg_anchor_samples"] = max(
                            1 if first_pass else 0, len(self._anchor_samples)
                        )
                    if reg_deadband:
                        if self._reg_frozen_corners is None:
                            self._reg_frozen_corners = corners.copy()
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
                            registration_corners = self._reg_frozen_corners
                            shift = float(
                                np.max(
                                    np.linalg.norm(
                                        corners - self._reg_frozen_corners,
                                        axis=1,
                                    )
                                )
                            )
                            reg_info.update(
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
                                registration_corners,
                                self._reg_frozen_corners,
                                self._reg_deadband_over,
                                released,
                                shift,
                            ) = registration._deadband_step(
                                self._reg_frozen_corners,
                                corners,
                                self._reg_deadband_over,
                            )
                            if released:
                                self._reg_deadband_releases += 1
                            reg_info.update(
                                {
                                    "reg_deadband_shift_px": shift,
                                    "reg_deadband_held": not released,
                                    "reg_deadband_pending": self._reg_deadband_over,
                                    "reg_deadband_released": released,
                                }
                            )
                confirmed = self._accumulator._confirmed
                anchor_ink = (
                    confirmed > 0
                    if confirmed is not None and confirmed.shape == prob.shape
                    else None
                )
            registered_prob, registered_visible = (
                registration._register_ink_probability(
                    prob, anchor, registration_corners, visible
                )
            )
            self._last_registration = (
                None if anchor is None else anchor.copy(),
                (None if registration_corners is None else registration_corners.copy()),
                None if corners is None else corners.copy(),
            )
            # Mapa registrado SO pela camera: e o que a arbitragem de
            # ancora compara com o mapa cru para julgar pouso x quad ruim.
            # A escolha/correcao do portao vale so para a evidencia que
            # vota (review do PR #67: o mapa corrigido inflava o escore de
            # camera e um movimento real virava "espurio").
            camera_prob = registered_prob
            camera_corners = registration_corners
            motion_decision = None
            if (
                self._motion is not None
                and not recovering
                and anchor is not None
                and registration_corners is not None
            ):
                # Estado do portao so nesta thread: geracao nova zera
                # a referencia aqui, nao em new_generation (outra
                # thread, no meio de uma medida).
                if self._motion_generation != generation:
                    self._motion.reset()
                    self._motion_generation = generation
                motion_matrix = registration._registration_matrix(
                    anchor, registration_corners, prob.shape
                )
                if registration._registration_is_identity(motion_matrix):
                    motion_matrix = None
                measured = self._motion.measure(
                    warped,
                    motion_matrix,
                    visible,
                    anchor,
                    confirmed_ink=None if first_pass else anchor_ink,
                    ink_prob=registered_prob,
                )
                unregistered = None
                if motion_matrix is not None and not self._motion.aligned(measured):
                    # O registro supoe a folha parada; se ela andou e o
                    # quad acompanhou, o warp sem registro e o alinhado
                    # (folha-movida, passadas 5-9: 1,2-1,6 px sem
                    # registro contra 3,1-4,0 px com ele).
                    unregistered = self._motion.measure(warped, None, visible, anchor)
                motion_decision, motion_fix, hypothesis = self._motion.decide(
                    measured, unregistered
                )
                reg_info["motion_decision"] = motion_decision
                reg_info["motion_residual_px"] = (
                    None if measured is None else measured.max_px
                )
                reg_info["motion_unregistered_px"] = (
                    None if unregistered is None else unregistered.max_px
                )
                reg_info["motion_blocks"] = (
                    None if measured is None else (measured.inliers, measured.blocks)
                )
                if hypothesis == "unregistered":
                    # A passada vota na propria geometria: registro
                    # identidade, a mesma da 1a passada da ancora.
                    registration_corners = anchor.copy()
                    registered_prob, registered_visible = (
                        registration._register_ink_probability(
                            prob, anchor, registration_corners, visible
                        )
                    )
                    self._last_registration = (
                        anchor.copy(),
                        registration_corners.copy(),
                        None if corners is None else corners.copy(),
                    )
                if motion_decision == "correct":
                    # A evidencia volta para a grade da ancora pelo
                    # modelo afim medido nesta mesma imagem.
                    registered_prob = motion_fix.warp_back(registered_prob)
                    if registered_visible is not None:
                        registered_visible = (
                            motion_fix.warp_back(
                                registered_visible.astype(np.uint8), nearest=True
                            )
                            > 0
                        )
            if coverage is not None and coverage.shape == prob.shape:
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
                def _covered(reg_corners):
                    if anchor is None or reg_corners is None:
                        return coverage
                    reg_matrix = registration._registration_matrix(
                        anchor, reg_corners, prob.shape
                    )
                    if registration._registration_is_identity(reg_matrix):
                        return coverage
                    return (
                        cv2.warpPerspective(
                            coverage.astype(np.uint8),
                            reg_matrix,
                            (prob.shape[1], prob.shape[0]),
                            flags=cv2.INTER_NEAREST,
                            borderValue=0,
                        )
                        > 0
                    )

                def _gate(evidence, reg_coverage):
                    if anchor_ink is None:
                        return evidence * reg_coverage
                    return np.where(
                        reg_coverage | anchor_ink,
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
                    camera_prob is registered_prob
                    and camera_corners is registration_corners
                )
                reg_coverage = _covered(registration_corners)
                if motion_decision == "correct":
                    # A cobertura acompanha a correcao da evidencia: sem
                    # isso o traco novo deslocado para fora da mascara
                    # velha era cortado em toda passada corrigida.
                    reg_coverage = (
                        motion_fix.warp_back(
                            reg_coverage.astype(np.uint8), nearest=True
                        )
                        > 0
                    )
                registered_prob = _gate(registered_prob, reg_coverage)
                if camera_untouched:
                    camera_prob = registered_prob
                else:
                    camera_prob = _gate(camera_prob, _covered(camera_corners))
            if anchor is not None and registration_corners is not None:
                matrix = registration._registration_matrix(
                    anchor, registration_corners, prob.shape
                )
                reg_info["registration_displacement_px"] = (
                    anchor_mod._matrix_grid_displacement(matrix, prob.shape)
                )
                reg_info["registration_is_identity"] = (
                    registration._registration_is_identity(matrix)
                )
        except Exception as error:  # noqa: BLE001 - vai para o laco
            with self._condition:
                self._error = error
                self._stopped = True
                self._condition.notify_all()
            return False
        delta_job = None
        with self._condition:
            if self._stopped:
                return False
            if generation != self._generation:
                return True
            first_pass = generation != self._worker_generation
            if first_pass:
                self._worker_generation = generation
                self._generation_passes = 0
                self._quiet_passes = 0
            # Degrau geometrico entre passadas consecutivas
            # (o follow so conta passada assentada; o portao de
            # remocao exige a mesma quietude).
            watch_step_px = None
            if corners is not None:
                if self._previous_corners is not None:
                    watch_step_px = float(
                        np.max(
                            np.linalg.norm(
                                corners - self._previous_corners,
                                axis=1,
                            )
                        )
                    )
                self._previous_corners = corners.copy()
            # A decisao de identidade roda na primeira passada e,
            # enquanto estiver em quarentena, nas seguintes
            # — a referencia so e consumida quando a decisao sai.
            identity_pending = (
                self._identity_recovery_requested
                and self._identity_reference is not None
            )
            if first_pass or identity_pending:
                identity_reference = self._identity_reference
                identity_corners = self._identity_reference_corners
                identity_recovery_requested = self._identity_recovery_requested
                keep_page = False
                if identity_reference is not None:
                    candidate = registered_prob
                    identity_visible = registered_visible
                    if identity_corners is not None and corners is not None:
                        identity_geometry_registered = True
                    prior = identity_reference > 0
                    prior_px = int(np.count_nonzero(prior))
                    identity_reference_px = prior_px
                    if prior_px == 0:
                        seen_any = candidate > self._accumulator.threshold
                        if identity_visible is not None:
                            seen_any = seen_any & identity_visible
                        keep_page = not np.any(seen_any)
                        identity_recall = 1.0 if keep_page else 0.0
                    else:
                        seen = candidate > self._accumulator.threshold
                        kernel = np.ones(
                            (identity._IDENTITY_DILATION_KERNEL,) * 2, np.uint8
                        )
                        seen_d = cv2.dilate(seen.astype(np.uint8), kernel) > 0
                        if identity_visible is not None:
                            prior_visible = prior & identity_visible
                            prior_visible_px = int(np.count_nonzero(prior_visible))
                            identity_visible_reference_px = prior_visible_px
                            if prior_visible_px < max(
                                identity._IDENTITY_MIN_VISIBLE_PX,
                                int(identity._IDENTITY_MIN_VISIBLE_FRACTION * prior_px),
                            ):
                                # Tinta anterior quase toda
                                # coberta: decidir agora seria
                                # condenar a pagina pela mao.
                                # Adia; a referencia fica para a
                                # proxima passada.
                                identity_deferred = True
                                quarantine_reason = "occlusion"
                            else:
                                prior_d = (
                                    cv2.dilate(
                                        prior_visible.astype(np.uint8),
                                        kernel,
                                    )
                                    > 0
                                )
                                identity_recall = (
                                    np.count_nonzero(prior_visible & seen_d)
                                    / prior_visible_px
                                )
                                seen_px = int(np.count_nonzero(seen))
                                identity_containment = (
                                    0.0
                                    if seen_px == 0
                                    else (np.count_nonzero(seen & prior_d) / seen_px)
                                )
                                keep_page = (
                                    identity_recall
                                    >= identity._PAGE_IDENTITY_MIN_RECALL
                                    and identity_containment
                                    >= identity._PAGE_IDENTITY_MIN_CONTAINMENT
                                )
                        else:
                            prior_d = cv2.dilate(prior.astype(np.uint8), kernel) > 0
                            identity_recall = (
                                np.count_nonzero(prior & seen_d) / prior_px
                            )
                            seen_px = int(np.count_nonzero(seen))
                            identity_containment = (
                                0.0
                                if seen_px == 0
                                else (np.count_nonzero(seen & prior_d) / seen_px)
                            )
                            keep_page = (
                                identity_recall >= identity._PAGE_IDENTITY_MIN_RECALL
                                and identity_containment
                                >= identity._PAGE_IDENTITY_MIN_CONTAINMENT
                            )
                if identity_deferred:
                    self._identity_defer_count += 1
                    quarantine_frames = self._identity_defer_count
                    identity_timed_out = (
                        quarantine_frames > identity._PAGE_IDENTITY_MAX_DEFERRALS
                    )
                if identity_deferred:
                    self._identity_switch_votes = 0
                elif identity_reference is not None and not keep_page:
                    self._identity_switch_votes += 1
                    identity_switch_pending = (
                        self._identity_switch_votes
                        < identity._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES
                    )
                else:
                    self._identity_switch_votes = 0
                if identity_recovery_requested:
                    if identity_timed_out:
                        identity_decision = "quarantine_timeout"
                    elif identity_deferred:
                        identity_decision = "deferred_occlusion"
                    elif identity_switch_pending:
                        identity_decision = "deferred_switch_consensus"
                    else:
                        identity_decision = "same_page" if keep_page else "new_page"
                    self._identity_events.append(
                        {
                            "generation": generation,
                            "decision": identity_decision,
                            "recall": identity_recall,
                            "containment": identity_containment,
                            "reference_px": identity_reference_px,
                            "visible_reference_px": (identity_visible_reference_px),
                            "geometry_registered": (identity_geometry_registered),
                            "threshold": identity._PAGE_IDENTITY_MIN_RECALL,
                            "containment_threshold": (
                                identity._PAGE_IDENTITY_MIN_CONTAINMENT
                            ),
                            "quarantine_reason": quarantine_reason,
                            "quarantine_frames": quarantine_frames,
                            "deferred": identity_deferred,
                        }
                    )
                if (
                    identity_deferred and not identity_timed_out
                ) or identity_switch_pending:
                    # A referencia NAO e consumida: a proxima
                    # passada decide com registro e visao
                    # confiaveis e consenso. ECC rejeitado nao vota.
                    pass
                else:
                    if identity_timed_out:
                        # Perda de lock, nao voto de pagina nova.
                        # A proxima passada estabelece outra ancora.
                        self._accumulator.reset()
                        self._canvas_epoch += 1
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
                    elif not keep_page:
                        self._accumulator.reset()
                        self._canvas_epoch += 1
                        self._anchor_corners = corners
                        self._anchor_samples = (
                            [corners.copy()]
                            if reg_anchor_median and corners is not None
                            else []
                        )
                    # Keep the old anchor when identity matched: all future
                    # observations remain registered into the same canvas.
                    self._identity_reference = None
                    self._identity_reference_corners = None
                    self._identity_recovery_requested = False
                    self._identity_defer_count = 0
                    self._identity_switch_votes = 0
            identity_owned = (
                first_pass
                or identity_pending
                or identity_deferred
                or identity_switch_pending
            )
            arbitration_abstain = False
            arb_event = None
            if anchor_arbitration:
                arb_status, arb_event = self._anchor_arbitration_step(
                    generation,
                    corners,
                    prob.shape,
                    identity_owned,
                    reg_anchor_median,
                    reg_deadband,
                    prob,
                    camera_prob,
                    visible,
                )
                arbitration_abstain = arb_status == "open"
                reg_info["arb_event_open"] = self._arb_event_open
                reg_info["arb_abstained"] = arbitration_abstain
                reg_info["arb_streak"] = len(self._arb_streak_quads)
                if self._arb_scores:
                    scores = self._arb_scores[-1]
                    reg_info["arb_pass_score_page"] = scores[0]
                    reg_info["arb_pass_score_camera"] = scores[1]
                    reg_info["arb_pass_scores_cyclic"] = list(scores[2:5])
                if arb_event is not None:
                    reg_info["arbitration_event"] = dict(arb_event)
                    reg_info["arb_verdict"] = arb_event["verdict"]
                    reg_info["arb_score_page"] = arb_event["score_page"]
                    reg_info["arb_score_camera"] = arb_event["score_camera"]
                    reg_info["arb_rotation_k"] = arb_event["rotation_k"]
                if (
                    arb_status == "verdict"
                    and arb_event is not None
                    and arb_event["action"]
                ):
                    # Pouso: a passada corrente entra ja
                    # registrada na ancora nova, por cima do
                    # estado INTOCADO (a v3 nao move o estado).
                    # A decisao do portao foi contra a ancora velha.
                    motion_decision = None
                    registered_prob, registered_visible = (
                        registration._register_ink_probability(
                            prob,
                            self._anchor_corners,
                            (self._reg_frozen_corners if reg_deadband else corners),
                            visible,
                        )
                    )
                    self._last_registration = (
                        None
                        if self._anchor_corners is None
                        else self._anchor_corners.copy(),
                        (
                            None
                            if (self._reg_frozen_corners if reg_deadband else corners)
                            is None
                            else (
                                self._reg_frozen_corners if reg_deadband else corners
                            ).copy()
                        ),
                        None if corners is None else corners.copy(),
                    )
                    force_publish = True
            follow_abstain = False
            if anchor_follow:
                # Mobilidade da ancora (frente 34): roda DEPOIS
                # da arbitragem — o veredito dela nesta passada
                # (arb_event) e a precedencia que o passo le.
                follow_status, follow_event = self._anchor_follow_step(
                    generation,
                    corners,
                    prob.shape,
                    identity_owned,
                    reg_anchor_median,
                    reg_deadband,
                    watch_step_px,
                    arb_event,
                )
                follow_abstain = follow_status == "open"
                reg_info["follow_event_open"] = self._follow_event_open
                reg_info["follow_abstained"] = follow_abstain
                reg_info["follow_streak"] = self._follow_streak
                reg_info["follow_hold"] = self._follow_hold
                if follow_event is not None:
                    reg_info["follow_reanchored"] = True
                    reg_info["follow_displacement_px"] = follow_event["displacement_px"]
                    reg_info["follow_closed_arb_event"] = follow_event[
                        "closed_arb_event"
                    ]
                    if follow_event["closed_arb_event"]:
                        reg_info["arb_event_open"] = False
                    # A passada corrente entra ja registrada na
                    # ancora nova, por cima do estado INTOCADO
                    # (a troca e so de ancora) — e o evento de
                    # arbitragem que a abstinha foi fechado pela
                    # re-ancoragem. A decisao do portao foi contra a
                    # ancora velha.
                    motion_decision = None
                    registered_prob, registered_visible = (
                        registration._register_ink_probability(
                            prob,
                            self._anchor_corners,
                            (self._reg_frozen_corners if reg_deadband else corners),
                            visible,
                        )
                    )
                    self._last_registration = (
                        None
                        if self._anchor_corners is None
                        else self._anchor_corners.copy(),
                        (
                            None
                            if (self._reg_frozen_corners if reg_deadband else corners)
                            is None
                            else (
                                self._reg_frozen_corners if reg_deadband else corners
                            ).copy()
                        ),
                        None if corners is None else corners.copy(),
                    )
                    arbitration_abstain = False
                    force_publish = True
            if (
                (first_pass or identity_pending)
                and not identity_deferred
                and not identity_switch_pending
                and not keep_page
            ):
                if visible is not None:
                    prob = np.where(visible, prob, 0.0).astype(np.float32)
                # Voto sem registro (1a passada ou pagina nova na
                # identidade): a evidencia entra na propria
                # geometria do warp e a ancora e o quad da
                # passada — o par registrado antes (referencia
                # de identidade, quad atual) descreveria uma
                # matriz pela qual a tinta nunca passou.
                self._last_registration = (
                    None
                    if corners is None
                    else (corners.copy(), corners.copy(), corners.copy())
                )
            else:
                prob = registered_prob
                visible = registered_visible
            # Veto de movimento so vale na passada registrada: a 1a
            # passada e a de identidade votam na propria geometria.
            motion_abstain = motion_decision == "veto" and prob is registered_prob
            if motion_abstain:
                self._motion_vetoes_total += 1
            elif motion_decision == "correct" and prob is registered_prob:
                self._motion_corrections_total += 1
            elif motion_decision == "unregistered" and prob is registered_prob:
                self._motion_unregistered_total += 1
            elif motion_decision == "fail_open" and prob is registered_prob:
                self._motion_fail_open_total += 1
            pass_voted = not (
                identity_switch_pending
                or (identity_deferred and identity_timed_out)
                or arbitration_abstain
                or follow_abstain
                or motion_abstain
            )
            if identity_switch_pending or (identity_deferred and identity_timed_out):
                visible = np.zeros(prob.shape, dtype=bool)
            if arbitration_abstain:
                # Evento de arbitragem aberto: a passada e
                # registrada e pontuada, mas nao vota.
                visible = np.zeros(prob.shape, dtype=bool)
            if follow_abstain:
                # Evento de divergencia assentada aberto: a
                # passada foi medida, mas nao vota — registrada
                # na ancora velha, ela depositaria a copia
                # deslocada que a re-ancoragem existe para
                # nao fabricar.
                visible = np.zeros(prob.shape, dtype=bool)
            if motion_abstain:
                # A folha anda sob um quad parado: registrada, esta
                # passada carimbaria a copia deslocada da tinta.
                visible = np.zeros(prob.shape, dtype=bool)
            if not pass_voted:
                # Passada abstida nao deposita: o registro fica
                # None para o avaliador nao projetar GT por uma
                # geometria que nunca chegou ao canvas.
                self._last_registration = None
            result = self._accumulator.add(
                prob,
                visible,
                removal_allowed=(
                    not identity_owned
                    and not arbitration_abstain
                    and not follow_abstain
                    and corners is not None
                    and watch_step_px is not None
                    and watch_step_px <= anchor_mod._ARB_SETTLE_PX
                    and reg_info.get("registration_displacement_px", float("inf"))
                    <= anchor_mod._ANCHOR_MIGRATION_DISPLACEMENT_PX
                ),
            )
            if (
                result["confirmed_changed"]
                or result["provisional_changed"]
                or result["turn"]
            ):
                self._still_passes = 0
                self._activity_quiet_passes = 0
            else:
                self._still_passes += 1
                self._activity_quiet_passes += 1
            # Contadores do descanso total: so a saida
            # CONFIRMADA interessa (provisoria tremula com o
            # warp e nao pode segurar a trabalhadora acordada).
            self._generation_passes += 1
            if result["confirmed_changed"] or result["turn"] or motion_abstain:
                # Passada vetada nao e quietude: a folha esta fora do
                # lugar e a tinta nova ainda nao entrou — descansar aqui
                # deixava o traco sem confirmar ate outro evento.
                self._quiet_passes = 0
            else:
                self._quiet_passes += 1
            self._passes_total += 1
            self._last_pass_ms = (self._clock() - pass_started) * 1000.0
            self._provisional_px = result["provisional_px"]
            self._confirmed_px = result["confirmed_px"]
            self._retracted_px_total = result["retracted_px_total"]
            publish_confirmed = (
                first_pass
                or identity_timed_out
                or identity_decision == "new_page"
                or result["confirmed_changed"]
                # Migracao: o canvas migrado nao "muda" para o
                # voto, mas a tela mostraria a grade velha.
                or force_publish
            )
            if publish_confirmed:
                self._version += 1
                self._published = (
                    generation,
                    self._version,
                    result["confirmed"],
                )
                # So a referencia (result["confirmed"] ja e uma copia da
                # passada): o calculo do delta roda depois de soltar o lock.
                delta_job = (
                    generation,
                    self._version,
                    result["confirmed"],
                    captured_at,
                    None
                    if self._anchor_corners is None
                    else self._anchor_corners.copy(),
                )
            if publish_confirmed or result["provisional_changed"]:
                self._display_version += 1
                self._published_display = (
                    generation,
                    self._display_version,
                    result["confirmed"],
                    result["provisional"],
                )
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
                    **reg_info,
                    "anchor_arbitration_enabled": anchor_arbitration,
                    "sequence": self._telemetry_passes,
                    "pass_gap_s": None
                    if self._previous_pass_started_at is None
                    else pass_started_at - self._previous_pass_started_at,
                    "corner_step_px": watch_step_px,
                    "quarantine_reason": quarantine_reason,
                    "quarantine_frames": quarantine_frames,
                    "deferred": identity_deferred,
                    "generation": generation,
                    "version": self._version,
                    "page_identity_decision": identity_decision,
                    "page_identity_recall": identity_recall,
                    "page_identity_containment": identity_containment,
                    "page_identity_reference_px": identity_reference_px,
                    "page_identity_visible_reference_px": (
                        identity_visible_reference_px
                    ),
                    "page_identity_geometry_registered": (identity_geometry_registered),
                    "submitted_at": submitted_at,
                    "pass_started_at": pass_started_at,
                    "pass_ended_at": pass_ended_at,
                    "published_at": published_at,
                    "queue_wait_s": pass_started_at - submitted_at,
                    "pass_s": pass_ended_at - pass_started_at,
                    "submit_to_publish_s": published_at - submitted_at,
                    "interval_s": scheduled_interval,
                    "hand_state": scheduled_hand_state,
                    "activity_quiet_passes": self._activity_quiet_passes,
                    "mask_px": int(
                        np.count_nonzero(prob > self._accumulator.threshold)
                    ),
                    "confirmed_px": result["confirmed_px"],
                    "new_confirmed_px": result["new_confirmed_px"],
                    "removed_confirmed_px": result.get("removed_confirmed_px", 0),
                    "visible_fraction": 1.0
                    if visible is None
                    else float(np.mean(visible)),
                }
            )
            self._previous_pass_started_at = pass_started_at
            self._condition.notify_all()
        if delta_job is not None:
            self._record_delta(*delta_job)
        return True

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
