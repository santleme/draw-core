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
# 25 px e medido no quadro de referencia (diagonal 1280x720): no
# ponto de uso o limiar efetivo e _FROZEN_SHEET_MOVE_PX * pixel_scale
# — um deslocamento de 25 px e ~3,5% da diagonal a 720p e ~1% a 4K,
# entao sem a escala as resolucoes maiores marcariam deslize onde
# nao ha (issue #105). Fica bem acima do jitter residual do
# rastreador (~2 px de banda morta) e abaixo do salto grande de
# _SHEET_BIG_JUMP_PX, que um deslize continuo nunca dispara — a
# banda morta e o salto grande escalam pelo mesmo fator no
# rastreador, entao a ordem entre os limiares se preserva em
# qualquer resolucao.
_FROZEN_SHEET_MOVE_PX = 25.0


_STALE_PASS_MIN_FRAMES = _SHEET_LOST_MIN_FRAMES


def _quad_camera_width(corners):
    """Largura média do quad em px de câmera (arestas superior/inferior)."""
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))


def _quad_camera_mask(corners, shape):
    """Interior do quad rasterizado na grade do frame de câmera."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.asarray(corners, dtype=np.float32).astype(np.int32)], 1)
    return mask.astype(bool)


def _paper_mean_bgr(image, coverage):
    """Média BGR sobre os pixels cobertos, sem materializar a seleção.

    cv2.mean reduz numa passada sobre a máscara; o caminho
    `image[coverage].mean(axis=0)` copiava os ~700-800k pixels do
    papel (~24 ms por submit medidos na VM, issue #98) antes de
    reduzir — mesmo resultado, ~43x mais barato.

    cv2.mean multiplica a soma pelo recíproco da contagem em vez de
    dividir: uma média em inteiro exato pode sair 1 ulp abaixo e o
    cast para uint8 truncar um nível a menos. Recupera a soma exata
    com round(mean*count) e divide direto — bit a bit igual ao
    astype(uint8) do caminho antigo (floor para valores >= 0).
    """
    count = int(coverage.sum())
    means = np.array(cv2.mean(image, coverage.astype(np.uint8))[:3])
    return (np.round(means * count) // count).astype(np.uint8)


# Anel de tolerancia do teste de componentes no IoU de cobertura
# (_coverage_iou). A mascara chega ao espaco da camera por um resize
# NEAREST do mapa nativo de 384 px do segmentador — um pixel nativo
# vira ~2-3 px de frame em 720p — e o contorno rasterizado do quad
# arredonda os cantos ao pixel inteiro, entao a borda da folha real
# pode ficar a alguns pixels do poligono desenhado; sem o anel, essa
# lacuna de rasterizacao desconectava o componente da folha e o IoU
# mediria a folha certa como espuria. 1% do menor lado do frame
# (~5-7 px em 480p-720p, piso de 4 px) cobre a folga com margem; um
# blob falso tao perto do quad ja seria indistinguivel da folha sem
# um modelo melhor.
_COVERAGE_IOU_RING_MIN_PX = 4
_COVERAGE_IOU_RING_FRAC = 0.01


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


def _coverage_iou(corners, coverage_mask, hand_mask=None):
    """IoU do quad com a cobertura, so com os blobs que tocam o quad.

    corners sao os 4 cantos do quad em pixels na grade da mascara;
    coverage_mask e a mascara uint8 nao-fundo da ultima deteccao. A
    uniao conta apenas os componentes conexos nao-fundo que tocam o
    quad dilatado pelo anel de rasterizacao: a mancha de piso claro
    que o segmentador rotula como papel fora da mesa de referencia
    (issue #106) entrava inteira na uniao e o IoU do quad certo caia
    abaixo do piso — falso negativo que derrubava passadas de tinta
    validas. O blob colado no quad segue na uniao porque a mascara
    nao o distingue da folha: o quad dentro de um blob gigante
    continua medindo |quad|/|blob|, a unica guarda desse modo.

    hand_mask (classe mao da mesma deteccao, mesma grade) entra
    INTEIRA na uniao, tocando o quad ou nao: o piso 0,2 foi calibrado
    sobre a uniao total e a mesa fantasma so afunda por causa dos
    pixels de mao fora do quad (settings.py, coverage_iou_floor) —
    exclui-la por desconexao devolvia o fantasma a passada de tinta.
    A componentizacao, por sua vez, roda sobre o nao-fundo SEM a
    classe mao: uma mancha de falso papel que so alcanca o quad
    atraves de pixels de mao fica fora — a mao nao pode virar ponte
    para um blob espurio. Sem mascara de mao (contrato de dois
    canais), o nao-fundo inteiro e o melhor proxy da folha.

    LIMITACAO: a cobertura que chega aqui e a mascara de VOTO, com
    os buracos internos ja preenchidos (perception.last_coverage_mask
    — o mapa cru fica em last_coverage_raw e nao atravessa o contrato
    DetectionResult). Se os dedos fecham um anel ao redor do va
    entre a folha e a mancha, o preenchimento marca o va como
    nao-fundo e a mancha volta a contar contra a folha mesmo depois
    da classe mao removida.
    """
    quad = np.zeros(coverage_mask.shape, np.uint8)
    cv2.fillConvexPoly(quad, np.round(corners).astype(np.int32), 1)
    ring = max(
        _COVERAGE_IOU_RING_MIN_PX,
        round(_COVERAGE_IOU_RING_FRAC * min(coverage_mask.shape)),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    quad_reach = cv2.dilate(quad, kernel) > 0
    has_hand = hand_mask is not None and hand_mask.shape == coverage_mask.shape
    non_hand = coverage_mask > 0
    if has_hand:
        non_hand &= ~(np.asarray(hand_mask) > 0)
    n_labels, labels = cv2.connectedComponents(non_hand.astype(np.uint8))
    keep = np.zeros(n_labels, bool)
    keep[np.unique(labels[quad_reach])] = True
    keep[0] = False  # o rotulo 0 e o fundo
    touching = keep[labels]
    union_mask = (quad > 0) | touching
    if has_hand:
        union_mask |= np.asarray(hand_mask) > 0
    union = np.count_nonzero(union_mask)
    if not union:
        return 0.0
    # A intersecao usa a mascara inteira: todo pixel nao-fundo dentro
    # do quad pertence a um componente que toca o quad (o proprio
    # pixel esta dentro do anel), entao cruzar com `touching` seria
    # redundante.
    inter = np.count_nonzero((quad > 0) & (coverage_mask > 0))
    return float(inter / union)


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
    on_ink(contracts.InkPublication) a cada publicacao nova da tinta,
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
        on_ink=_ignore,
        ink_meta=False,
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
        self._on_ink = on_ink
        self._ink_meta = bool(ink_meta)
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
        self._detect_wake = (
            tracker.DetectWakeGate(
                settled_interval_s=self._config.detect_wake_interval_s
            )
            if self._config.detect_wake
            else None
        )
        self._detect_wake_frames = (
            deque(maxlen=16) if self._config.detect_wake else None
        )
        self._last_ink_submit_at = None
        self._hand_exit_at = None
        self._stale_hand_woke = False
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
        # Telemetria do piso de cobertura (issue #105): o
        # coverage_iou_floor nao tem denominador natural para
        # escalar com a resolucao como os limiares em pixels, entao
        # cada veto conta aqui e os ultimos IoUs rejeitados ficam
        # em anel — a recalibracao do piso em outras mesas se faz
        # com este dado, nao com chute. Sobem a SessionTelemetry
        # via stats() (merge no snapshot de 1 Hz em draw/app.py).
        # O lado ACEITO tambem e amostrado: o piso novo se posiciona
        # entre o maior fantasma e a menor folha real (calibracao
        # original: fantasma <= 0.108, folha >= 0.298 -> 0.2), e um
        # anel so de rejeitados nao mostra a distribuicao da folha.
        self._coverage_iou_rejects = 0
        self._coverage_iou_reject_recent = deque(maxlen=32)
        self._coverage_iou_accepts = 0
        self._coverage_iou_accept_recent = deque(maxlen=32)

    def _sheet_coverage_camera(self):
        """Cobertura da folha na grade do frame, ou None sem evidência.

        A máscara da última detecção sobe à trabalhadora em espaço
        de câmera (issue #100): quem consome retifica para a grade
        que precisa com a homografia do quad que a acompanha. Uma
        erosão leve desconta a borda da própria máscara — a mesma
        regra de quando a erosão rolava depois do warp, agora na
        fonte. Cobertura toda vazia é devolvida como está: o quad
        que retifica uma região sem folha nenhuma é exatamente o
        caso que a cobertura existe para vedar — None seria "sem
        sinal", que confia.
        """
        if not self._config.coverage or self._coverage_mask is None:
            return None
        return (
            cv2.erode(
                (self._coverage_mask > 127).astype(np.uint8),
                np.ones((5, 5), np.uint8),
            )
            > 0
        )

    def _quad_coverage_iou(self, warp_corners):
        """IoU do quad com a mascara de cobertura no espaco da camera.

        None sem mascara publicada (coverage desligado ou resultado
        sem extras) — o piso fica inerte e o comportamento segue o
        de antes. Com mascara, mede o quanto o quad que retifica o
        warp confere com a folha segmentada: o quad fantasma sobre
        mesa vazia cobre quase tudo de fora da mascara; a folha
        real, mesmo com o quad estourado, domina a intersecao.
        A uniao conta so os componentes nao-fundo que tocam o quad
        (_coverage_iou): a mancha falsa disjunta — o piso claro lido
        como papel da issue #106 — nao infla mais a medida, enquanto
        o blob colado no candidato a folha continua contando por ser
        indistinguivel dela sem um modelo melhor. A mao prevista pela
        mesma deteccao (_hand_mask_det) entra inteira na uniao: e o
        termo que o piso 0,2 usa para afundar o quad fantasma da mesa
        vazia. O custo e um connectedComponents por passada de tinta
        — cadencia de passada, nao de quadro.
        """
        if self._coverage_mask is None:
            return None
        return _coverage_iou(warp_corners, self._coverage_mask, self._hand_mask_det)

    def _warp_and_submit_ink(self, frame, warp_corners, captured_at=None):
        """Warp + visibilidade + submissao a trabalhadora de tinta.

        Os cantos sao o quad que retifica ESTE warp, e a
        trabalhadora registra a evidencia de cada passada na
        geometria ancora da geracao — sem isso, o quad que anda
        entre passadas re-projeta a tinta assentada e cada
        alinhamento confirma a sua copia.
        """
        settled_wake = (
            self._detect_wake is not None
            and self._detect_wake.settled
            and not self.tracker.missed
        )
        if (
            self._config.skip_stale
            and not settled_wake
            and self.tracker._frames_since_seen > _STALE_PASS_MIN_FRAMES
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
            # O piso foi calibrado nas capturas s1-s4 da mesa de
            # referencia; sem denominador natural para escala-lo,
            # os IoUs rejeitados sao registrados para que a
            # recalibracao em outras mesas vire dado (issue #105).
            self._coverage_iou_rejects += 1
            self._coverage_iou_reject_recent.append(round(float(iou), 4))
            return
        if iou is not None:
            # Lado aceito da distribuicao: a recalibracao do piso
            # precisa ver onde a folha real cai, nao so onde o
            # fantasma caiu (issue #105).
            self._coverage_iou_accepts += 1
            self._coverage_iou_accept_recent.append(round(float(iou), 4))
        # Visibilidade e cobertura sobem na grade do frame (issue
        # #100): retificar cada máscara câmera→página custava ~3
        # warpPerspective por submit no laço, e a trabalhadora ainda
        # re-retificava página→âncora — o mesmo pixel pagava dois
        # resamples. Com as máscaras em espaço de câmera a
        # trabalhadora resolve as duas grades com uma retificação
        # só (M·C = A), e o laço deixa de pagar warps de máscara.
        visible = None
        if self.hand_masker is not None:
            camera_mask = self.hand_masker.mask(frame)
            if camera_mask is not None:
                visible = camera_mask == 0
        if self._config.seg_hand_occlusion and self._hand_mask_det is not None:
            # seg_hand_occlusion: a classe mão do segmentador vira
            # segundo sinal de oclusão. Sem ela, a mão sobre a
            # folha conta como cobertura (classes != 0) e, quando o
            # masker falha, a textura da mão vota como tinta nova.
            # Os dois sinais se somam: basta um dizer "mão" para a
            # passada não enxergar a folha ali.
            # Margem de ~1,2% da largura da folha — agora medida no
            # quad em px de câmera: o contorno da mão é uma zona de
            # transição que o segmentador reparte entre classes e,
            # sem ela, a aresta da mão vota como traço.
            margin = max(5, int(round(0.012 * _quad_camera_width(warp_corners))))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
            )
            seg_clear = (
                cv2.dilate((self._hand_mask_det > 127).astype(np.uint8), kernel) == 0
            )
            visible = seg_clear if visible is None else (visible & seg_clear)
        coverage = self._sheet_coverage_camera()
        if self._config.ink_divnorm:
            # Com ink_divnorm a imagem sobe crua: o par de
            # preenchimentos e a normalização rolam dentro da
            # passada da trabalhadora (issue #101), fora do laço.
            # A cobertura de câmera segue subindo — a trabalhadora a
            # retifica para preencher a região fora da folha e para
            # o gate de re-confirmação.
            ink_submit = self._warp_sheet(frame, warp_corners)
        else:
            # O ganho e o preenchimento usam máscaras em grade de
            # página nos antigos — a grade da página É o quad
            # retificado; em câmera o equivalente é o interior do
            # quad, senão a mesa do frame inteiro entraria na conta.
            quad_mask = (
                _quad_camera_mask(warp_corners, frame.shape)
                if self._config.ink_auto_gain
                or (coverage is not None and coverage.any())
                else None
            )
            if self._config.ink_auto_gain:
                # Página escura (exposição no teto de 30 fps): o ganho
                # que o sensor não dá entra aqui, antes do page_clean.
                # Estimado só sobre a folha (coverage): a mesa clara do
                # quad não cancela o ganho do papel, e o preenchimento
                # abaixo já trabalha no domínio clareado.
                # A mão visível fica fora da conta: a cobertura a inclui
                # (o seg a conta como folha), mas uma mão clara não é
                # papel e cancelaria o ganho do papel escuro ao redor
                # (review do PR #52). `visible` já une os dois sinais
                # de oclusão.
                # Sem cobertura (detecção clássica não a fornece) a mão
                # ainda sai da conta: `visible` sozinha serve de
                # máscara.
                gain_mask = quad_mask.copy()
                if coverage is not None:
                    gain_mask &= coverage
                if visible is not None:
                    gain_mask &= visible
                frame, self.last_ink_gain = exposure.apply_paper_gain(frame, gain_mask)
            fill_color = None
            if coverage is not None and coverage.any():
                # Fora da folha o warp é preto: a transição
                # folha->preto é uma aresta artificial que o page_clean
                # lê como tinta. A cor média da própria página preenche
                # o entorno — o modelo deixa de ver a aresta sintética
                # e a borda da folha para de votar como traço. O
                # preenchimento rola no frame: a região dentro do quad
                # já sobe uniforme para o warp, e o ganho acima já
                # clareou o que a média vê (mesma ordem de antes).
                paper = coverage & quad_mask
                if paper.any():
                    fill_color = _paper_mean_bgr(frame, paper)
                    # fillConvexPoly sub-cobre a aresta do quad: sem a
                    # dilatação, uma faixa de ~2 px junto à borda da
                    # página não é preenchida e sobe com cor de mesa
                    # (review do PR #118). Pintar além do quad não
                    # custa nada — o warp só amostra o interior.
                    fill_region = cv2.dilate(
                        quad_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
                    ).astype(bool)
                    filled = frame.copy()
                    filled[fill_region & ~coverage] = fill_color
                    frame = filled
            ink_submit = self._warp_sheet(frame, warp_corners)
            if fill_color is not None:
                frame_h, frame_w = frame.shape[:2]
                crosses = (
                    warp_corners[:, 0].min() < 0
                    or warp_corners[:, 0].max() > frame_w - 1
                    or warp_corners[:, 1].min() < 0
                    or warp_corners[:, 1].max() > frame_h - 1
                )
                if crosses:
                    # Folha parcialmente fora do quadro: o warp só
                    # preenche o que a câmera viu — o resto sobe preto.
                    # Repõe a média do papel no padding, como o
                    # preenchimento pós-warp do caminho antigo.
                    valid = self._warp_sheet(
                        np.full((frame_h, frame_w), 255, np.uint8),
                        warp_corners,
                    )
                    ink_submit[valid == 0] = fill_color
        self.ink_worker.submit(
            contracts.WarpedPage(
                ink_submit,
                self.clean_generation,
                warp_corners,
                visible,
                coverage,
                captured_at=captured_at,
            )
        )

    # So o stream pede os metadados do InkDelta (ink_meta): sem consumidor o
    # laco nao paga o poll_delta. A versao confirmada do delta e outra do que
    # a de exibicao (poll_display sobe tambem com mudanca so do provisorio).
    _ink_meta = False
    _delta_version = -1

    def _ink_publication(self, got):
        captured_at = anchor = None
        full = False
        if self._ink_meta:
            poll_delta = getattr(self.ink_worker, "poll_delta", None)
            delta = (
                None
                if poll_delta is None
                else poll_delta(self.clean_generation, self._delta_version, wait=0.005)
            )
            if delta is not None:
                self._delta_version = delta.version
                captured_at, anchor, full = (
                    delta.captured_at,
                    delta.anchor_corners,
                    delta.full,
                )
        return contracts.InkPublication(
            self.clean_generation,
            got.version,
            got.confirmed,
            captured_at=captured_at,
            anchor_corners=anchor,
            full=full,
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
        # A tinta antiga morreu: a saida ao vivo precisa saber (canvas limpo).
        self._on_ink(contracts.InkPublication(-1, 0, None))
        self.tracker.reset()
        detect_wake = getattr(self, "_detect_wake", None)
        if detect_wake is not None:
            detect_wake.reset()
        detect_wake_frames = getattr(self, "_detect_wake_frames", None)
        if detect_wake_frames is not None:
            detect_wake_frames.clear()
        self._last_ink_submit_at = None
        self._hand_exit_at = None
        self._stale_hand_woke = False
        self.corner_history.clear()
        self.display_corners = None
        self.display_gray = None
        self._coverage_mask = None
        self._hand_mask_det = None
        self._detections_since = since
        want_shape = (config_module.PAPER_H, config_module.PAPER_W, 3)
        buf = getattr(self, "canvas_buffer", None)
        if buf is not None and buf.shape != want_shape:
            self.canvas_buffer = np.empty(want_shape, dtype=np.uint8)

    @property
    def last_hand_mask(self):
        """Ultima mascara semantica de mao publicada pelo detector, ou None."""
        return self._hand_mask_det

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
        # Escala de pixels do quadro atual contra o quadro de
        # referencia 1280x720 onde os limiares em pixels foram
        # calibrados (issue #105): alimenta o rastreador abaixo e o
        # destrave da geometria congelada mais adiante. Um hypot por
        # quadro — custo de ruido.
        pixel_scale = tracker.camera_pixel_scale(frame.shape[:2])
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
            had_hand_mask = self._hand_mask_det is not None and np.any(
                self._hand_mask_det
            )
            self._hand_mask_det = det_result.hand_mask
            has_hand_mask = self._hand_mask_det is not None and np.any(
                self._hand_mask_det
            )
            if (
                self._detect_wake is not None
                and self.clean_hand_present is False
                and not self._stale_hand_woke
                and has_hand_mask
                and self._hand_exit_at is not None
                and det_captured_at is not None
                and det_captured_at < self._hand_exit_at
            ):
                # Heartbeat capturado ANTES de a mao sair chegou atrasado
                # com mascara de mao velha: descarta a mascara defasada e
                # acorda o detector uma unica vez neste episodio de ausencia.
                self._hand_mask_det = None
                self._stale_hand_woke = True
                self._detect_wake.wake()
                # Re-arma a confirmacao caso a trabalhadora ja tenha gasto
                # a passada de confirmacao sob mascara velha e descansado.
                self.ink_worker.set_hand_presence(True)
                self.ink_worker.set_hand_presence(False)
            elif (
                self.clean_hand_present is False and had_hand_mask and not has_hand_mask
            ):
                # A deteccao pos-saida acabou de limpar a mascara de mao que
                # ainda cobria a pagina: re-arma a janela de confirmacao para
                # garantir as passadas limpas completas sobre o traco novo.
                self.ink_worker.set_hand_presence(True)
                self.ink_worker.set_hand_presence(False)
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
        base_should_detect = self.tracker.should_detect(self.frame_number)
        if self._detect_wake is not None:
            should_detect = self._detect_wake.should_detect(
                frame,
                packet.captured_at,
                base_should_detect=base_should_detect,
                hand_mask=self._hand_mask_det,
                wake_signal=self.tracker.missed or self.tracker.has_pending,
            )
        else:
            should_detect = base_should_detect
        if should_detect:
            if self._detect_wake_frames is not None:
                self._detect_wake_frames.append((packet.captured_at, frame))
            self.detection_worker.submit(frame, packet.captured_at, self.frame_number)
        if self.detection_worker.error is not None:
            raise self.detection_worker.error
        tracker_before = self.tracker.diagnostic_state() if ran_detection else None
        tracker_event = self.tracker.update(
            detected_corners,
            detected_verdict,
            ran_detection,
            det_captured_at if ran_detection else self._clock(),
            pixel_scale=pixel_scale,
        )
        if ran_detection and self._detect_wake is not None:
            ref_frame = None
            if self._detect_wake_frames is not None and det_captured_at is not None:
                while (
                    self._detect_wake_frames
                    and self._detect_wake_frames[0][0] <= det_captured_at + 1e-6
                ):
                    ts, submitted = self._detect_wake_frames.popleft()
                    if abs(ts - det_captured_at) < 1e-6:
                        ref_frame = submitted
            if ref_frame is not None:
                self._detect_wake.on_detection(
                    self.tracker.corners,
                    ref_frame,
                    det_captured_at,
                    hand_mask=self._hand_mask_det,
                    missed=self.tracker.missed,
                    has_pending=self.tracker.has_pending,
                )
            elif self.tracker.corners is None or self.tracker.missed:
                self._detect_wake.reset()
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
                prev_hand_present = self.clean_hand_present
                (
                    self.hand_presence_seq,
                    self.clean_hand_present,
                    clean_hand_moved,
                ) = hp_result
                self.ink_worker.set_hand_presence(
                    self.clean_hand_present,
                    moved=clean_hand_moved,
                )
                if self.clean_hand_present is False:
                    if prev_hand_present is not False:
                        self._hand_exit_at = packet.captured_at
                        self._stale_hand_woke = False
                    if (
                        not self._stale_hand_woke
                        and self._detect_wake is not None
                        and self._hand_mask_det is not None
                        and np.any(self._hand_mask_det)
                    ):
                        # Transicao para mao ausente com mascara de mao velha
                        # em cache: acorda uma vez para renovar _hand_mask_det
                        # no proximo quadro antes de fechar a confirmacao.
                        self._stale_hand_woke = True
                        self._detect_wake.wake(force_next=True)
                else:
                    self._hand_exit_at = None
                    self._stale_hand_woke = False
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
                    > _FROZEN_SHEET_MOVE_PX * pixel_scale
                ):
                    self.corner_history.clear()
                    self.corner_history.append(self.tracker.corners.copy())
            ink_corners = np.median(np.stack(self.corner_history), axis=0).astype(
                np.float32
            )
            self._last_ink_submit_at = packet.captured_at
            self._warp_and_submit_ink(
                frame, ink_corners, captured_at=packet.captured_at
            )
        elif (
            self._detect_wake is not None
            and (
                self._detect_wake.settled
                or (
                    self.clean_hand_present is False
                    and not self.tracker.missed
                    and not self.tracker.has_pending
                    and (self._hand_mask_det is None or not np.any(self._hand_mask_det))
                )
            )
            and self.tracker.corners is not None
            and self.ink_session_open
            and self.corner_history
        ):
            # Com detect_wake assentado, o detector dorme a 1 Hz mas a
            # tinta e a presenca de mao seguem alimentadas na cadencia
            # normal sobre o quad assentado.
            settled_due = base_should_detect
            ink_interval = self._config.ink.pass_interval_s
            if (
                not settled_due
                and ink_interval is not None
                and self._last_ink_submit_at is not None
            ):
                settled_due = (
                    packet.captured_at - self._last_ink_submit_at >= ink_interval - 1e-9
                )
            if settled_due:
                if self.hand_presence_worker is not None:
                    self.hand_presence_worker.submit(frame, packet.captured_at * 1000.0)
                ink_corners = np.median(np.stack(self.corner_history), axis=0).astype(
                    np.float32
                )
                self._last_ink_submit_at = packet.captured_at
                self._warp_and_submit_ink(
                    frame, ink_corners, captured_at=packet.captured_at
                )

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
            # Tomada de saida (bridge/STREAM.md): so a camada confirmada;
            # o publicador nunca bloqueia (latest-wins).
            self._on_ink(self._ink_publication(got))
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

    def stats(self):
        """Contadores do laco para a telemetria (issue #105).

        O app funde este dict nos tel_fields do snapshot de 1 Hz da
        SessionTelemetry; os valores saem JSON-serializaveis (int,
        float, lista de floats). Os IoUs vetados pelo piso de
        cobertura sao o dado de recalibracao do coverage_iou_floor
        em outras mesas — o limiar nao tem denominador natural para
        escalar com a resolucao como os limiares em pixels.
        """
        return {
            "coverage_iou_rejects_total": self._coverage_iou_rejects,
            "coverage_iou_reject_recent": list(self._coverage_iou_reject_recent),
            "coverage_iou_accepts_total": self._coverage_iou_accepts,
            "coverage_iou_accept_recent": list(self._coverage_iou_accept_recent),
        }
