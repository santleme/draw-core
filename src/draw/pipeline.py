"""Orquestracao do laco sem marcadores: folha, geracao e tinta.

E o miolo do laco do app (deteccao -> rastreador -> trabalhadora de
tinta -> canvas), extraido do SheetInkLoop legado sem mudar a ordem
das operacoes nem os padroes do caminho pagina->limpa. O app monta
este objeto com as dependencias reais (threads de deteccao, de
tinta e de presenca de mao, relogio monotonico, warp e LK do
OpenCV) e chama `step()` a cada quadro capturado; a exibicao, as
teclas, a telemetria por segundo e o epilogo ficam fora daqui.

As fronteiras falam contratos: o passo recebe contracts.FramePacket,
a deteccao devolve contracts.DetectionResult, a passada sobe como
contracts.WarpedPage e o canvas desce como
contracts.InkDisplaySnapshot. As bandeiras do ambiente viraram
campos do settings.PipelineConfig consultados via self._config.

Nao migraram daqui (Tiers C/D e a pilha de tinta antiga, que ficam
no legado): session_ink/ink_source/InkAccumulator e a geometria
congelada da tecla C (session_corners/freeze_pending/
restart_reference/legacy_generation), session_recorder e
trace_acquisitions (o gravador e do lado de treino; o callback
on_acquisition permanece), o portao de primeira confirmacao
(first_confirm_*), a cadencia assentada (detect_settled_every e o
salto de oportunidades), a propagacao LK do quad
(ReseededQuadPropagator/LkUpdateGate e contadores lk_*) e os
canais take_turn_event/take_identity_split (a virada e a divisao
por identidade moram dentro da trabalhadora nova).
"""

import time
from collections import deque

import cv2
import numpy as np

from . import config as config_module
from . import contracts, exposure, paper, tracker
from .ink import render as ink_render
from .settings import PipelineConfig
from .tracker import _SHEET_LOST_MIN_FRAMES

# A tinta e lida numa geometria estabilizada: a mediana dos ultimos
# conjuntos de cantos do rastreador. O jitter de canto entre deteccoes
# (media de ~5 px, com saltos de dezenas de pixels quando a mao cobre
# um canto) desloca o traco de 1 a 3 px no espaco do papel e desfaz o
# voto por pixel do canvas acumulativo; a mediana de 5 segura a
# geometria no lugar por tempo bastante para a janela do acumulador.
_INK_CORNER_HISTORY = 5


# Desfecho de step() quando a trabalhadora de tinta morreu: o app
# mostra o aviso na janela e sai do laco.
INK_WORKER_FAILED = "ink_worker_failed"


# Destrave da geometria de tinta congelada sem mao (config
# follow_move): com clean_hand_present=False o historico de cantos
# nao recebe o rastreador — decisao correta para a pagina parada (a
# mediana viva andaria com o jitter e arrastaria a tinta publicada),
# mas cega para a folha que desliza sem mao na cena: o warp segue na
# pose velha, passa a ver mesa e o modelo confirma a textura como
# tinta (a inundacao de borda medida em 2026-11-12). O rastreador so
# move o quad confirmado depois de filtro de mediana, banda morta e
# persistencia de salto — um deslocamento sustentado deste quad
# contra a mediana congelada e movimento real, nao ruido. Acima do
# limiar, o historico reparte da posicao nova e o registro/ECC/juiz
# da trabalhadora tratam a migracao, como num salto com mao.
# 25 px fica bem acima do jitter residual do rastreador (~2 px de
# banda morta) e abaixo do salto grande de _SHEET_BIG_JUMP_PX, que
# um deslize continuo nunca dispara.
_FROZEN_SHEET_MOVE_PX = 25.0


_STALE_PASS_MIN_FRAMES = _SHEET_LOST_MIN_FRAMES


def _acquisition_record(trace, frame_index, consumed, before, after, tracker_event):
    """Evento por aquisicao: rastro do combinado + rastreador.

    detected_frame_index (do rastro) e a passada cujo quadro o
    detector examinou; frame_index e a passada em que o rastreador
    consumiu o resultado (com deteccao atrasada, posterior).
    """
    return {
        **trace,
        "frame_index": frame_index,
        "consumed_by_tracker": consumed,
        "tracker_before": before,
        "tracker_after": after,
        "tracker_event": tracker_event,
    }


def _bgr_to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _ignore(*_args, **_kwargs):
    return None


class Pipeline:
    """Estado e passo do laco sem marcadores, dependencias injetadas.

    detection_worker: poll/submit/error/take_traces de
    `SheetDetectionWorker` (poll devolve contracts.DetectionResult).
    ink_worker: `PageCleanWorker` — submit recebe
    contracts.WarpedPage e poll_display devolve
    contracts.InkDisplaySnapshot. hand_presence_worker:
    `HandPresenceWorker` ou None. clock: relogio monotonico em
    segundos (o mesmo das trabalhadoras). warp_sheet, to_gray e
    track_display: o warp da folha e o LK do quad de exibicao;
    render_display: o painel em camadas (confirmada + provisoria).
    Callbacks: on_format_suggestion (sugestao de formato),
    on_acquisition (evento por aquisicao),
    save_page_snapshot(prefixo, mascara) (pagina anterior na
    virada ou na divisao) e log (console).
    """

    def __init__(
        self,
        *,
        detection_worker,
        ink_worker,
        hand_presence_worker=None,
        hand_masker=None,
        config=None,
        format_suggester=None,
        on_format_suggestion=_ignore,
        on_acquisition=_ignore,
        save_page_snapshot=_ignore,
        clock=time.monotonic,
        warp_sheet=paper.warp_sheet,
        to_gray=_bgr_to_gray,
        track_display=tracker.lk_track_corners,
        render_display=ink_render.render_layered_canvas,
        log=print,
    ):
        self.detection_worker = detection_worker
        self.ink_worker = ink_worker
        self.hand_presence_worker = hand_presence_worker
        self.hand_masker = hand_masker
        self._config = PipelineConfig() if config is None else config
        self.format_suggester = (
            paper.FormatSuggester() if format_suggester is None else format_suggester
        )
        self._on_format_suggestion = on_format_suggestion
        self._on_acquisition = on_acquisition
        self._save_page_snapshot = save_page_snapshot
        self._clock = clock
        self._warp_sheet = warp_sheet
        self._to_gray = to_gray
        self._track_display = track_display
        self._render_display = render_display
        self._log = log

        self.detect_ms = None
        self.sheet_present = False
        self.clean_hand_present = None
        self.hand_presence_seq = 0
        # Máscara neural da última detecção, em coordenadas de cena
        # (config.coverage). Retificada com a homografia de cada
        # warp de tinta, vira a cobertura que impede mesa vazada no
        # retificado de virar voto de tinta nova.
        self._coverage_mask = None
        self._hand_mask_det = None

        self.tracker = tracker.SheetTracker()
        self.corner_history = deque(maxlen=_INK_CORNER_HISTORY)
        self.ink_session_open = False
        self.accumulated_canvas = None
        # Canvas do caminho pagina->limpa: a mascara acumulada e a
        # uniao das mascaras confirmadas da trabalhadora. None
        # enquanto nenhuma folha foi lida nesta sessao.
        self.clean_mask = None
        # Geracao da tinta assincrona: acompanha a geracao da
        # trabalhadora; folha nova sobe a geracao. Ao cair do tracker,
        # a identidade fica pendente ate a folha voltar e o conteudo
        # decidir se e re-registro ou pagina nova. (A versao aplicada do poll()
        # saiu na orquestracao em camadas: o laco le o canvas pelo
        # poll_display(), e a clean_applied_version ficou morta.)
        self.clean_generation = 0
        self.clean_display_version = 0
        self.clean_provisional_mask = None
        self.clean_recovery_pending = False
        # Buffer dono do canvas acumulado: mora aqui de um frame de
        # deteccao ao outro, e nenhum outro uso pode escrever nele.
        self.canvas_buffer = np.empty(
            (config_module.PAPER_H, config_module.PAPER_W, 3),
            dtype=np.uint8,
        )
        self.frame_number = 0
        self.last_det_seq = 0
        self.display_corners = None
        self.display_gray = None
        self._detections_since = 0.0
        # Observabilidade (sem efeito no laco): ultimo evento do
        # rastreador e contagem das aberturas de geracao por causa.
        self.last_tracker_event = None
        # Ganho digital do ultimo warp de tinta (ink_auto_gain); 1.0 sem ganho.
        self.last_ink_gain = 1.0
        self.generation_events = {"drop": 0, "reopen": 0}

    def _sheet_coverage_page(self, warp_corners):
        """Cobertura da folha na grade da página, ou None sem evidência.

        A máscara da última detecção é retificada com a mesma
        homografia do warp de tinta deste quadro: o que a cena
        mostra fora da folha — mesa que entra por quad largo ou
        atrasado depois de movimento — só pode votar sobre a tinta
        que a âncora já tem (gate de re-confirmação na trabalhadora).
        Uma erosão leve desconta a borda da
        própria máscara. Cobertura toda vazia é devolvida como
        está: o quad que retifica uma região sem folha nenhuma é
        exatamente o caso que a cobertura existe para vedar —
        None seria "sem sinal", que confia.
        """
        if not self._config.coverage or self._coverage_mask is None:
            return None
        warped = self._warp_sheet(self._coverage_mask, warp_corners)
        if warped is None:
            return None
        coverage = (
            cv2.erode((warped > 127).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        )
        return coverage

    def _quad_coverage_iou(self, warp_corners):
        """IoU do quad com a mascara de cobertura no espaco da camera.

        None sem mascara publicada (coverage desligado ou resultado
        sem extras) — o piso fica inerte e o comportamento segue o
        de antes. Com mascara, mede o quanto o quad que retifica o
        warp confere com a folha segmentada: o quad fantasma sobre
        mesa vazia cobre quase tudo de fora da mascara; a folha
        real, mesmo com o quad estourado, domina a intersecao.
        """
        if self._coverage_mask is None:
            return None
        quad = np.zeros(self._coverage_mask.shape, np.uint8)
        cv2.fillConvexPoly(quad, np.round(warp_corners).astype(np.int32), 1)
        mask = self._coverage_mask > 0
        union = (quad.astype(bool) | mask).sum()
        if not union:
            return 0.0
        return float((quad.astype(bool) & mask).sum() / union)

    def _warp_and_submit_ink(self, frame, warp_corners):
        """Warp + visibilidade + submissao a trabalhadora de tinta.

        Os cantos sao o quad que retifica ESTE warp, e a
        trabalhadora registra a evidencia de cada passada na
        geometria ancora da geracao — sem isso, o quad que anda
        entre passadas re-projeta a tinta assentada e cada
        alinhamento confirma a sua copia.
        """
        if self._config.skip_stale and (
            self.tracker._frames_since_seen > _STALE_PASS_MIN_FRAMES
        ):
            # skip_stale: os cantos retidos sao de graça —
            # frames_since_seen passou do piso da queda sem
            # confirmacao nova. O warp olharia a mesa (o quad ficou
            # pra tras) ou pixels ja ocluidos (o canal visible ja
            # zera) — a passada e trabalho morto por construcao.
            # Passada fresca nao e afetada: _frames_since_seen = 0
            # nela.
            return
        iou = self._quad_coverage_iou(warp_corners)
        if iou is not None and iou < self._config.coverage_iou_floor:
            # Coverage floor: o quad retifica uma regiao que a
            # segmentacao nao reconhece como folha — o fit inflou
            # cantos sobre um blob espurio da mesa vazia. Submeter
            # o retificado de mesa e trabalho morto e cada pixel
            # e mesa na trabalhadora de tinta.
            return
        ink_warped = self._warp_sheet(frame, warp_corners)
        # Visibilidade na grade da pagina: onde a mao esta, a
        # passada nao ve a folha — a passada nao pode votar com
        # evidencia fabricada sob oclusao.
        visible = None
        if self.hand_masker is not None:
            camera_mask = self.hand_masker.mask(frame)
            if camera_mask is not None:
                hand_warped = self._warp_sheet(camera_mask, warp_corners)
                visible = hand_warped == 0
        if self._config.seg_hand_occlusion and self._hand_mask_det is not None:
            # seg_hand_occlusion: a classe mão do segmentador vira
            # segundo sinal de oclusão. Sem ela, a mão sobre a
            # folha conta como cobertura (classes != 0) e, quando o
            # masker falha, a textura da mão vota como tinta nova.
            # Os dois sinais se somam: basta um dizer "mão" para a
            # passada não enxergar a folha ali.
            seg_hand_warped = self._warp_sheet(self._hand_mask_det, warp_corners)
            if seg_hand_warped is not None:
                # Margem de ~1,2% da largura da página: o contorno da
                # mão é uma zona de transição que o segmentador
                # reparte entre classes — sem ela a aresta da mão vota
                # como traço.
                margin = max(5, int(round(0.012 * seg_hand_warped.shape[1])))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
                )
                seg_hand_dil = (
                    cv2.dilate((seg_hand_warped > 127).astype(np.uint8), kernel) > 0
                )
                seg_clear = ~seg_hand_dil
                visible = seg_clear if visible is None else (visible & seg_clear)
        coverage = self._sheet_coverage_page(warp_corners)
        if self._config.ink_auto_gain:
            # Página escura (exposição no teto de 30 fps): o ganho que o
            # sensor não dá entra aqui, antes do page_clean. Estimado só
            # sobre a folha (coverage): a mesa clara do quad não cancela
            # o ganho do papel, e o preenchimento abaixo já trabalha no
            # domínio clareado.
            # A mão visível fica fora da conta: a cobertura a inclui (o seg
            # a conta como folha), mas uma mão clara não é papel e
            # cancelaria o ganho do papel escuro ao redor (review do PR
            # #52). `visible` já une os dois sinais de oclusão.
            # Sem cobertura (detecção clássica não a fornece) a mão ainda
            # sai da conta: `visible` sozinha serve de máscara.
            gain_mask = coverage
            if visible is not None:
                gain_mask = visible if coverage is None else coverage & visible
            ink_warped, self.last_ink_gain = exposure.apply_paper_gain(
                ink_warped, gain_mask
            )
        if coverage is not None:
            # Fora da folha o warp é preto: a transição
            # folha->preto é uma aresta artificial que o page_clean
            # lê como tinta. A cor média da própria página preenche o
            # entorno — o modelo deixa de ver a aresta sintética e
            # a borda da folha para de votar como traço.
            ink_submit = ink_warped
            if coverage.any():
                ink_submit = ink_warped.copy()
                ink_submit[~coverage] = (
                    ink_warped[coverage].mean(axis=0).astype(np.uint8)
                )
            self.ink_worker.submit(
                contracts.WarpedPage(
                    ink_submit,
                    self.clean_generation,
                    warp_corners,
                    visible,
                    coverage,
                )
            )
        else:
            self.ink_worker.submit(
                contracts.WarpedPage(
                    ink_warped,
                    self.clean_generation,
                    warp_corners,
                    visible,
                )
            )

    def reset_source(self, since):
        """Fronteira de fonte: a camera nova comeca uma pagina nova.

        A troca de camera muda resolucao e cena: cantos do rastreador,
        quad e cinza de exibicao, mascaras e historico de cantos
        pertencem a camera anterior e sao descartados. A tinta nao e
        preservada: o canvas reinicia, como ao virar a folha, e a
        primeira folha da camera nova abre uma geracao limpa (a geracao
        antiga e invalidada: um snapshot dela ainda nao lido nao volta
        para a tela nem para o Save). Resultados de deteccao capturados antes de since (em voo na
        camera antiga) tambem sao descartados.
        """
        self.ink_session_open = False
        self.clean_recovery_pending = False
        self.clean_mask = None
        self.clean_provisional_mask = None
        self.accumulated_canvas = None
        # Nenhuma geracao da trabalhadora e negativa: a colheita deixa de
        # ver snapshots nao lidos da camera antiga ate a primeira folha
        # nova abrir a sua geracao.
        self.clean_generation = -1
        self.clean_display_version = 0
        self.tracker.reset()
        self.corner_history.clear()
        self.display_corners = None
        self.display_gray = None
        self._coverage_mask = None
        self._hand_mask_det = None
        self._detections_since = since

    def step(self, packet):
        """Um quadro do laco sem marcadores.

        packet e o contracts.FramePacket da captura: image e o
        quadro BGR e captured_at o carimbo monotonico em segundos
        (o relogio compartilhado com as trabalhadoras; quem precisa
        de milissegundos usa captured_at * 1000 nas mesmas contas
        de idade de antes). Devolve None, ou INK_WORKER_FAILED
        quando a trabalhadora de tinta morreu (o aviso ja saiu no
        log).
        """
        frame = packet.image
        det_result = self.detection_worker.poll(self.last_det_seq)
        if det_result is not None and det_result.captured_at < self._detections_since:
            self.last_det_seq = det_result.sequence
            det_result = None
        if det_result is not None:
            self.last_det_seq = det_result.sequence
            detected_corners = det_result.corners
            detected_verdict = det_result.verdict
            det_captured_at = det_result.captured_at
            self.detect_ms = det_result.detect_ms
            # A máscara de mão do detector alimenta a oclusão
            # semântica (seg_hand_occlusion) mesmo com a cobertura
            # desligada — no legado as duas vinham na mesma flag,
            # aqui são sinais separados. Um resultado sem máscara
            # limpa a evidência velha em vez de deixar oclusão
            # defasada.
            self._hand_mask_det = det_result.hand_mask
            if self._config.coverage:
                # A máscara de cobertura acompanha o quad que saiu
                # dela — chegou junto, guarda junto; sem ela limpa
                # em vez de deixar cobertura defasada.
                self._coverage_mask = det_result.coverage_mask
            ran_detection = True
        else:
            detected_corners, detected_verdict = None, None
            det_captured_at = None
            ran_detection = False
        if self.tracker.should_detect(self.frame_number):
            self.detection_worker.submit(frame, packet.captured_at, self.frame_number)
        if self.detection_worker.error is not None:
            raise self.detection_worker.error
        tracker_before = self.tracker.diagnostic_state() if ran_detection else None
        tracker_event = self.tracker.update(
            detected_corners,
            detected_verdict,
            ran_detection,
            det_captured_at if ran_detection else self._clock(),
        )
        if ran_detection:
            tracker_after = self.tracker.diagnostic_state()
            for trace in self.detection_worker.take_traces(self.last_det_seq):
                consumed = trace["result_seq"] == self.last_det_seq
                self._on_acquisition(
                    _acquisition_record(
                        trace,
                        self.frame_number,
                        consumed,
                        tracker_before if consumed else None,
                        tracker_after if consumed else None,
                        tracker_event if consumed else None,
                    )
                )
        self.last_tracker_event = tracker_event
        if tracker_event == "drop":
            self.generation_events["drop"] += 1
            self.ink_session_open = False
            # canvas_persist: a queda do tracker nao prova que a
            # pagina mudou — o canvas do usuario fica congelado na
            # tela ate a folha voltar (re-registro via
            # clean_recovery_pending) ou a sessao reabrir com pagina
            # nova (a reabertura limpa, como sempre). Com a opcao
            # desligada: comportamento historico, o canvas some da
            # tela.
            if not self._config.canvas_persist:
                self.clean_mask = None
                self.clean_provisional_mask = None
                self.accumulated_canvas = None
            # Guarda a geracao ate a folha voltar;
            # a primeira mascara decide pela tinta se e
            # re-registro ou pagina nova.
            self.clean_recovery_pending = True

        self.frame_number += 1
        # Presenca e o estado do rastreador: warp_sheet com
        # cantos nunca devolve None, entao o warp por quadro
        # era so um teste de presenca caro, e o resultado era
        # descartado. O unico consumo dos pixels fora do bloco
        # de tinta e o fallback de exibicao, que retifica sob
        # demanda.
        self.sheet_present = self.tracker.corners is not None
        if ran_detection and self.sheet_present and detected_corners is not None:
            suggestion = self.format_suggester.observe(detected_corners, frame.shape)
            if suggestion is not None:
                self._on_format_suggestion(suggestion)

        # Presenca de mao assincrona (T4): colhe o
        # resultado mais novo da trabalhadora (o estado
        # clean_hand_present que o congelamento de
        # geometria le abaixo) e alimenta a porta da
        # tinta. O atraso e de ate um ciclo de amostragem
        # frente ao desenho sincrono antigo — irrelevante
        # na cadencia de 250 ms da tinta e na janela de
        # quietude de 1 s da mao.
        if self.hand_presence_worker is not None:
            if self.hand_presence_worker.error is not None:
                # Presenca morta degrada para "sem sinal":
                # a cadencia trata None como ativo e o app
                # segue como antes da porta existir.
                self._log(
                    "Aviso: a presenca de mao falhou e foi "
                    "desligada: "
                    f"{self.hand_presence_worker.error}. A tinta "
                    "segue na cadencia normal."
                )
                self.hand_presence_worker = None
                self.clean_hand_present = None
                self.ink_worker.set_hand_presence(None)
            hp_result = (
                self.hand_presence_worker.poll(self.hand_presence_seq)
                if self.hand_presence_worker is not None
                else None
            )
            if hp_result is not None:
                (
                    self.hand_presence_seq,
                    self.clean_hand_present,
                    clean_hand_moved,
                ) = hp_result
                self.ink_worker.set_hand_presence(
                    self.clean_hand_present,
                    moved=clean_hand_moved,
                )
        # Quad de EXIBICAO (T9): em quadro com resultado de
        # deteccao, o exibido e o confirmado; nos demais,
        # propaga por LK sobre o cinza anterior. Falha de
        # propagacao volta ao confirmado. So exibicao: a
        # tinta nunca le este quad.
        if self.tracker.corners is None:
            self.display_corners = None
            self.display_gray = None
        else:
            gray_now = self._to_gray(frame)
            if (
                ran_detection
                or self.display_corners is None
                or self.display_gray is None
            ):
                self.display_corners = self.tracker.corners.copy()
            else:
                tracked_display = self._track_display(
                    self.display_gray, gray_now, self.display_corners
                )
                self.display_corners = (
                    tracked_display
                    if tracked_display is not None
                    else self.tracker.corners.copy()
                )
            self.display_gray = gray_now
        if ran_detection and self.tracker.corners is not None:
            if self.hand_presence_worker is not None:
                # Submete com o carimbo de captura: e ele
                # que alimenta o modo VIDEO do landmarker.
                self.hand_presence_worker.submit(frame, packet.captured_at * 1000.0)
            if not self.ink_session_open:
                # Folha nova na cena: canvas e geometria recomecam.
                self.clean_mask = None
                # accumulated_canvas tambem zera: sem isto, o
                # painel continuava exibindo o canvas da folha
                # ANTERIOR ate a trabalhadora publicar a
                # primeira mascara da geracao nova (ate ~250 ms
                # de cadencia mais a passada do modelo), e a
                # tecla s salvava esse canvas errado.
                self.accumulated_canvas = None
                self.generation_events["reopen"] += 1
                self.clean_generation = self.ink_worker.new_generation(
                    preserve_identity=self.clean_recovery_pending
                )
                self.clean_recovery_pending = False
                self.clean_display_version = 0
                self.clean_provisional_mask = None
                self.corner_history.clear()
                self.ink_session_open = True
            # Sem mao, a geometria da tinta congela na
            # ultima mediana: a pagina esta assentada e a
            # mediana viva seguiria andando com o balanco
            # do detector, deslocando a tinta ja publicada.
            if self.clean_hand_present is not False or (not self.corner_history):
                self.corner_history.append(self.tracker.corners.copy())
            elif self._config.follow_move:
                # Congelado sem mao: so um deslocamento real da folha
                # destrava — o quad do rastreador ja e posicao
                # confirmada (mediana + banda morta + persistencia),
                # entao distancia grande da mediana congelada e
                # movimento de verdade, nao jitter do detector. O
                # historico reparte da pose nova para a mediana
                # convergir ja, sem esperar a janela andar.
                _frozen_med = np.median(np.stack(self.corner_history), axis=0).astype(
                    np.float32
                )
                if (
                    tracker.corner_distance(_frozen_med, self.tracker.corners)
                    > _FROZEN_SHEET_MOVE_PX
                ):
                    self.corner_history.clear()
                    self.corner_history.append(self.tracker.corners.copy())
            ink_corners = np.median(np.stack(self.corner_history), axis=0).astype(
                np.float32
            )
            self._warp_and_submit_ink(frame, ink_corners)

        return self.collect_ink()

    def collect_ink(self):
        """Colhe a tinta assincrona: painel e erro.

        E o fim de step(); quem dirige a trabalhadora em pump() logo
        depois de step() chama de novo para o consumidor ver a passada
        no mesmo quadro. Devolve None ou INK_WORKER_FAILED.
        """
        # Colhe a tinta assincrona pelo canal de
        # exibicao: a confirmada vira o canvas
        # (clean_mask, o que snapshot e consumidores
        # leem) e a provisoria entra so no painel, em
        # cinza. Sem versao nova, o painel exibido e o
        # mesmo de antes, sem custo de modelo neste laco.
        got = self.ink_worker.poll_display(
            self.clean_generation, self.clean_display_version
        )
        if got is not None:
            self.clean_display_version = got.version
            self.clean_mask = got.confirmed
            self.clean_provisional_mask = got.provisional
            self.accumulated_canvas = self._render_display(
                self.clean_mask,
                self.clean_provisional_mask,
                out=self.canvas_buffer,
            )
        # Trabalhadora morta por excecao: sem isto o canvas
        # congelava no ultimo quadro e nem o console nem a
        # janela diziam nada (o erro morria na thread).
        worker_error = self.ink_worker.error
        if worker_error is not None:
            self._log(
                "A tinta pagina->limpa falhou e a thread "
                f"parou: {worker_error}. O canvas nao avanca."
            )
            return INK_WORKER_FAILED
        return None
