"""Máscara de mão: onde a mão está no frame da câmera, via MediaPipe.

O canvas acumulativo (draw.ink.InkAccumulator) precisa saber quais
pixels da folha estão cobertos pela mão em cada frame: pixel coberto
é desconhecido e não ganha nem perde tinta. Quem produz essa
informação é o Hand Landmarker do MediaPipe (API tasks), rodando no
frame da câmera; a máscara sai no espaço do frame e o chamador a leva
ao espaço do papel pela mesma homografia da tinta.

A máscara de uma mão é o casco convexo dos 21 marcos, escalado de
1,30 em torno do centróide para cobrir a carne entre os dedos, unido
a um trapézio de antebraço que sai do pulso na direção oposta ao
dedo médio (o antebraço também cobre a folha e também projeta a
sombra que o detector de tinta enxerga), tudo dilatado de 15 px para
cobrir a caneta segurada e a borda difusa da mão. Medido no replay do
vídeo de desenho com limiares de confiança em 0,3: a mão de desenho é
detectada em cerca de dois terços das amostras em que aparece, e a
fração média da folha coberta fica perto de 10%.

Integração preguiçosa e opcional, no espírito do detector neural: o
mediapipe só é importado no primeiro uso, e qualquer ausência (o
pacote, o arquivo do modelo, uma falha de criação) degrada para
"sem máscara" com um aviso único, nunca para erro. O acumulador
funciona sem a máscara; ela só melhora a separação.

O arquivo do modelo (.task, binário de ~8 MB) fica fora do
repositório. O caminho chega por argumento ou pelo local convencional
models/hand_landmarker.task na raiz do repo. A fonte oficial do
arquivo é o armazenamento do Google para modelos do MediaPipe.
"""

import threading
from pathlib import Path

import cv2
import numpy as np

from .contracts import FramePacket, HandOcclusion, HandSignature

# O caminho convencional e resolvido a partir da RAIZ DO REPO, como
# o do detector neural e o do pagina->limpa; era relativo ao CWD e
# um app aberto de outro diretorio perdia a mascara de mao sem
# aviso (o modelo existia, so nao era achado). O CWD continua como
# segunda tentativa, para quem roda a partir de uma copia local.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CANDIDATES = (
    str(_REPO_ROOT / "models" / "hand_landmarker.task"),
    "models/hand_landmarker.task",
)
_MIN_CONFIDENCE = 0.3
_HULL_SCALE = 1.30
_DILATE_PX = 15
# Largura maxima do frame na deteccao de PRESENCA (assinatura):
# presenca nao precisa dos pixels, so da geometria dos marcos.
_PRESENCE_MAX_WIDTH = 640
# Teto da espera do close() por uma deteccao em voo: uma chamada
# travada no MediaPipe nao pode deixar o fechamento pendurado —
# passado o teto, _closed sinaliza e quem estava em voo fecha ao
# retornar.
_CLOSE_LOCK_TIMEOUT_S = 2.0
# Baixa luz (issue #129): o MediaPipe perde a mao em frame escuro. Abaixo
# desta luminancia media (amostra com stride) o frame ganha uma correcao
# gama leve rumo a _DARK_TARGET_MEAN antes da deteccao; frame normal nao
# paga nada alem da amostra. _DARK_MIN_GAMMA limita o realce (ruido).
_DARK_MEAN = 70.0
_DARK_TARGET_MEAN = 110.0
_DARK_MIN_GAMMA = 0.45
_DARK_SAMPLE_STRIDE = 8
# Retencao curta: ate _HOLD_FRAMES deteccoes seguidas sem mao reaproveitam
# a ultima mao vista (queda curta do detector nao libera a folha). So vale
# para o mesmo tamanho de frame; "indisponivel" nunca e retido.
_HOLD_FRAMES = 3
# A partial detection (for example, only one of two hands found) keeps
# the missing hand's last mask briefly. Full misses are already retained
# at landmark level by _retain.
_MASK_HOLD_FRAMES = 3
_MASK_HOLD_GROW_PX = 10
# Por modo de deteccao: (atributo do landmarker publicado, flag de
# indisponivel, RunningMode, avisa na falha). O VIDEO nao avisa: a
# falha dele degrada para o IMAGE e a mascara segue viva; so a falha
# do IMAGE, o modo final, anuncia a indisponibilidade.
_LANDMARKER_MODES = {
    "image": ("_landmarker", "_unavailable", "IMAGE", True),
    "video": ("_video_landmarker", "_video_unavailable", "VIDEO", False),
}


def _resolve_model_path(model_path):
    """Caminho do .task: argumento ou local convencional."""
    if model_path:
        return model_path
    for candidate in _DEFAULT_CANDIDATES:
        if Path(candidate).exists():
            return str(candidate)
    return None


def _lift_dark(frame):
    """Gama leve so em frame escuro; frame normal volta intacto."""
    sample = frame[::_DARK_SAMPLE_STRIDE, ::_DARK_SAMPLE_STRIDE]
    if sample.ndim == 3 and sample.shape[2] == 3:
        # Mesma luminancia (BT.601) do cinza do diagnostico ao vivo:
        # a media dos canais deslocaria o corte em cena com dominante.
        mean = float(
            cv2.cvtColor(np.ascontiguousarray(sample), cv2.COLOR_BGR2GRAY).mean()
        )
    else:
        mean = float(sample.mean())
    if mean >= _DARK_MEAN or mean < 1.0:
        return frame
    gamma = np.log(_DARK_TARGET_MEAN / 255.0) / np.log(mean / 255.0)
    gamma = float(np.clip(gamma, _DARK_MIN_GAMMA, 1.0))
    table = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    return cv2.LUT(frame, table)


def hand_model_available(model_path=None):
    """True se um arquivo de modelo resolve e existe neste disco."""
    path = _resolve_model_path(model_path)
    return path is not None and Path(path).exists()


def _hand_polygons(points, frame_shape):
    """Os polígonos de uma mão: casco escalado e trapézio do antebraço.

    points são os 21 marcos em pixels do frame. O trapézio sai do
    pulso (marco 0) na direção oposta ao marco 9 (base do dedo
    médio), com meia-largura de 0,75 da distância entre os marcos 5
    e 17 (a largura da palma) e comprimento de 1,5 diagonal do
    frame: o antebraço sai do quadro, e a máscara precisa ir junto.
    """
    hull = cv2.convexHull(points)
    centroid = hull.reshape(-1, 2).mean(axis=0)
    scaled = centroid + (hull.reshape(-1, 2) - centroid) * _HULL_SCALE
    polygons = [scaled.astype(np.int32)]
    wrist = points[0].astype(np.float64)
    middle_base = points[9].astype(np.float64)
    direction = wrist - middle_base
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return polygons
    direction /= norm
    normal = np.array([-direction[1], direction[0]])
    half_width = max(0.75 * float(np.linalg.norm(points[5] - points[17])), 20.0)
    frame_h, frame_w = frame_shape[:2]
    length = 1.5 * float(np.hypot(frame_w, frame_h))
    end = wrist + direction * length
    base_a = wrist + direction * (0.3 * norm) + normal * half_width * 0.8
    base_b = wrist + direction * (0.3 * norm) - normal * half_width * 0.8
    end_a = end + normal * half_width
    end_b = end - normal * half_width
    polygons.append(np.array([base_a, base_b, end_b, end_a]).astype(np.int32))
    return polygons


class HandMasker:
    """Produtor preguiçoso da máscara de mão.

    mask(frame) retorna a máscara uint8 no espaço do frame (255 =
    mão), uma máscara vazia quando nenhuma mão é detectada, ou None
    quando o recurso está indisponível (sem mediapipe, sem modelo ou
    com falha na criação). A indisponibilidade avisa uma vez e é
    permanente na sessão: não adianta tentar de novo a cada frame.

    O parametro verify e um callable (Path|str) -> None injetavel
    e nomeado OBRIGATORIO: o pipeline passa
    draw.infra.manifest.verify_model para barrar modelo divergente
    ou nao registrado antes de o MediaPipe abrir o arquivo; o None
    explicito significa que o chamador optou por nao verificar.

    O landmarker e fechado deterministicamente por close() —
    idempotente, e o objeto vira context manager — nunca por
    __del__: o __del__ do MediaPipe pode travar num landmarker
    vazado finalizado pelo GC na thread principal.
    """

    def __init__(self, model_path=None, *, verify):
        self._requested_path = model_path
        resolved = _resolve_model_path(model_path)
        if resolved is not None and verify is not None:
            # Estrito por padrao: divergente ou nao registrado levanta
            # aqui, antes de o MediaPipe abrir o arquivo.
            verify(resolved)
        self._landmarker = None
        self._video_landmarker = None
        self._unavailable = False
        self._video_unavailable = False
        self._last_video_ts = -1
        self._held_hands = None
        self._held_shape = None
        self._held_misses = 0
        self.last_detection_held = False
        self._observed_source_id = None
        self._last_hand_mask = None
        self._last_hand_count = 0
        self._mask_misses = 0
        self._warned = False
        self._closed = False
        self._lock = threading.RLock()

    def close(self):
        """Fecha os landmarkers criados; chamadas futuras degradam.

        Idempotente: a segunda chamada nao faz nada. Depois do
        fechamento mask()/presence_signature() devolvem None —
        indisponivel permanente, sem recriar o modelo por acidente.

        O fechamento adquire o lock do masker e espera uma chamada
        de deteccao em voo terminar — um landmarker nunca fecha
        enquanto outra thread o usa. Se o lock nao abre em
        _CLOSE_LOCK_TIMEOUT_S (deteccao travada no MediaPipe),
        _closed fica marcado sem o lock e quem estava em voo fecha
        os landmarkers ao retornar: o close() nao fica pendurado.
        """
        if self._closed:
            return
        if not self._lock.acquire(timeout=_CLOSE_LOCK_TIMEOUT_S):
            self._closed = True
            self._unavailable = True
            self._video_unavailable = True
            return
        try:
            self._closed = True
            self._close_landmarkers()
        finally:
            self._lock.release()

    def _close_landmarkers(self):
        """Fecha os dois landmarkers tolerando erro; idempotente."""
        for landmarker in (self._landmarker, self._video_landmarker):
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass
        self._landmarker = None
        self._video_landmarker = None
        self._unavailable = True
        self._video_unavailable = True

    def _dispose_fresh(self, landmarker):
        """Fecha um landmarker recem-criado que nao pode publicar."""
        try:
            landmarker.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _warn_once(self):
        if not self._warned:
            self._warned = True
            print(
                "Aviso: mascara de mao indisponivel (pacote mediapipe "
                "ou modelo hand_landmarker.task ausente). O canvas "
                "acumulativo roda sem ela."
            )

    def _create_landmarker(self, running_mode, path):
        try:
            import mediapipe as mp

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=running_mode,
                num_hands=2,
                min_hand_detection_confidence=_MIN_CONFIDENCE,
                min_hand_presence_confidence=_MIN_CONFIDENCE,
                min_tracking_confidence=_MIN_CONFIDENCE,
            )
            return mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception:
            return None

    def _ensure(self, kind="image"):
        """Garante o landmarker (IMAGE ou VIDEO) sob demanda.

        O modo VIDEO do MediaPipe rastreia a mao entre quadros e
        pula a deteccao de palma enquanto ela esta localizada —
        no modo IMAGE cada chamada recomeca do zero (pesquisa de
        performance de 2026-11-03, T4). Falha ao criar ou ao
        detectar marca o modo como indisponivel e a presenca cai
        para o IMAGE permanentemente: degradacao, nunca erro.
        """
        landmarker_attr, unavailable_attr, mode_name, warn = _LANDMARKER_MODES[kind]
        with self._lock:
            if getattr(self, landmarker_attr) is not None:
                return True
            if getattr(self, unavailable_attr):
                return False

            def _fail():
                setattr(self, unavailable_attr, True)
                if warn:
                    self._warn_once()
                return False

            path = _resolve_model_path(self._requested_path)
            if path is None or not Path(path).exists():
                return _fail()
            try:
                import mediapipe as mp
            except Exception:
                return _fail()
            created = self._create_landmarker(
                getattr(mp.tasks.vision.RunningMode, mode_name),
                path,
            )
            if created is None:
                return _fail()
            if self._closed:
                # O close() correu sem o lock enquanto a criacao
                # estava em voo: o recem-criado nao se publica —
                # fecha ja.
                self._dispose_fresh(created)
                setattr(self, unavailable_attr, True)
                return False
            setattr(self, landmarker_attr, created)
            return True

    def _landmarks(self, frame, max_width=None, timestamp_ms=None):
        """Marcos das maos no frame: lista de arrays (21, 2) em px
        do frame original, [] sem mao, None se indisponivel.

        Com max_width, o frame e reduzido antes da deteccao e os
        marcos (normalizados pelo MediaPipe) sao reescalados de
        volta — a geometria e a mesma, o custo de pre-processamento
        cai com os pixels.

        Com timestamp_ms (o carimbo de captura do frame), a
        deteccao roda no landmarker VIDEO com detect_for_video; o
        carimbo e forcado estritamente crescente, como a API
        exige. Sem carimbo, ou se o modo VIDEO falhar, cai no
        landmarker IMAGE de sempre.
        """
        with self._lock:
            try:
                return self._retain(
                    self._landmarks_locked(frame, max_width, timestamp_ms), frame
                )
            finally:
                # Se o close() passou sem o lock durante esta
                # chamada, o fechamento acontece aqui, na saida
                # de quem estava em voo.
                if self._closed:
                    self._close_landmarkers()

    def reset_retention(self):
        """Descarta a mao retida (troca de fonte/camera/rotacao): marcos
        de outra vista nao descrevem a nova, mesmo com o mesmo tamanho."""
        with self._lock:
            self._held_hands = None
            self._held_shape = None
            self._held_misses = 0
            self.last_detection_held = False
            self._last_hand_mask = None
            self._last_hand_count = 0
            self._mask_misses = 0

    def _retain(self, hands, frame):
        """Histerese de presenca: sem mao por ate _HOLD_FRAMES deteccoes
        seguidas, devolve a ultima mao vista (mesmo tamanho de frame)."""
        self.last_detection_held = False
        if hands is None:
            return None
        shape = frame.shape[:2]
        if hands:
            self._held_hands = hands
            self._held_shape = shape
            self._held_misses = 0
            return hands
        if self._held_hands is not None and self._held_shape == shape:
            self._held_misses += 1
            if self._held_misses <= _HOLD_FRAMES:
                self.last_detection_held = True
                return self._held_hands
        self._held_hands = None
        return hands

    def _landmarks_locked(self, frame, max_width, timestamp_ms):
        """O corpo de _landmarks, rodando sob o lock do masker."""
        use_video = timestamp_ms is not None and self._ensure("video")
        if not use_video and not self._ensure("image"):
            return None
        import mediapipe as mp

        work = frame
        if max_width and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            work = cv2.resize(
                frame,
                (max_width, int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(_lift_dark(work), cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = None
        if use_video:
            try:
                timestamp = max(int(timestamp_ms), self._last_video_ts + 1)
                result = self._video_landmarker.detect_for_video(image, timestamp)
                self._last_video_ts = timestamp
            except Exception:
                self._video_unavailable = True
                # O landmarker falho fecha antes de perder a
                # referencia: solta-lo sem close deixaria o
                # serial_dispatcher do MediaPipe vivo ate o
                # __del__ (que trava no GC). Um close quebrado
                # nao pode derrubar a degradacao para o IMAGE.
                stale = self._video_landmarker
                self._video_landmarker = None
                if stale is not None:
                    try:
                        stale.close()
                    except Exception:
                        pass
                result = None
        if result is None:
            if not self._ensure("image"):
                return None
            try:
                result = self._landmarker.detect(image)
            except Exception:
                return None
        frame_h, frame_w = frame.shape[:2]
        hands = []
        for hand in result.hand_landmarks or []:
            hands.append(
                np.array(
                    [[lm.x * frame_w, lm.y * frame_h] for lm in hand],
                    dtype=np.float32,
                )
            )
        return hands

    def mask(self, frame, timestamp_ms=None):
        """Máscara de mão atual, com retenção de quedas breves do detector.

        Full misses are held at landmark level. If only one of multiple
        hands remains detected, retain the previous union mask briefly so
        the missing hand cannot expose ink during flicker.
        """
        with self._lock:
            hands = self._landmarks(
                frame,
                max_width=_PRESENCE_MAX_WIDTH,
                timestamp_ms=timestamp_ms,
            )
            if hands is None:
                return None
            if not hands:
                # Ausência real (hold de marcos expirado): a pegada
                # retida é antiga demais para valer — reatá-la numa
                # detecção parcial futura vetaria uma região que a
                # mão já deixou.
                self._last_hand_mask = None
                self._last_hand_count = 0
                self._mask_misses = 0
                return np.zeros(frame.shape[:2], dtype=np.uint8)

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for points in hands:
                for polygon in _hand_polygons(points, frame.shape):
                    cv2.fillPoly(mask, [polygon], 255)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * _DILATE_PX + 1, 2 * _DILATE_PX + 1)
            )
            mask = cv2.dilate(mask, kernel)
            if self.last_detection_held:
                # Falha total: a histerese entregou os marcos
                # retidos e a pegada REAL cresce a cada miss — a
                # mão segue se movendo sob uma leitura que não se
                # atualiza; o footprint fixo exporia a borda dela
                # como papel à vista (Devin Review do PR #144).
                self._mask_misses += 1
                if (
                    self._last_hand_mask is not None
                    and self._last_hand_mask.shape == mask.shape
                ):
                    mask = self._merge_held(mask)
                return mask
            if (
                self._last_hand_mask is not None
                and self._last_hand_mask.shape == mask.shape
                and len(hands) < self._last_hand_count
                and self._mask_misses < _MASK_HOLD_FRAMES
            ):
                self._mask_misses += 1
                mask = self._merge_held(mask)
            else:
                self._last_hand_mask = mask.copy()
                self._last_hand_count = len(hands)
                self._mask_misses = 0
            return mask

    def _merge_held(self, mask):
        """Une a pegada retida (dilatada pelos misses) à máscara."""
        grow = _MASK_HOLD_GROW_PX * self._mask_misses
        hold_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)
        )
        return cv2.bitwise_or(mask, cv2.dilate(self._last_hand_mask, hold_kernel))

    def observe(self, packet: FramePacket, *, source_id: int) -> HandOcclusion:
        """Publica a máscara da captura com identidade e timestamp."""
        if source_id != self._observed_source_id:
            # Outra fonte/vista: marcos retidos da anterior nao valem.
            if self._observed_source_id is not None:
                self.reset_retention()
            self._observed_source_id = source_id
        mask = self.mask(
            packet.image,
            timestamp_ms=packet.captured_at * 1000.0,
        )
        return HandOcclusion(
            frame_sequence=packet.sequence,
            captured_at=packet.captured_at,
            source_id=source_id,
            occlusion_mask=None if mask is None else mask != 0,
        )

    def presence_signature(self, frame, timestamp_ms=None):
        """Assinatura de presenca sem construir mascara nenhuma.

        Devolve None (indisponivel), HandSignature(0.0, 0.0, 0.0)
        (sem mao) ou o HandSignature(area, centroide_x,
        centroide_y) em px do frame: a area e a do casco convexo
        dos 21 marcos e o centroide e a media dos marcos. E tudo o
        que o adaptador de presenca consome (ver
        hand_presence.MediaPipeHandPresence) — o caminho antigo
        pagava cvtColor + fillPoly + dilatacao eliptica de 31x31
        sobre o frame 720p inteiro so para contar pixels dessa
        mascara (pesquisa de performance de 2026-11-03, item T3).
        A deteccao roda sobre o frame reduzido a _PRESENCE_MAX_WIDTH:
        o landmarker trabalha internamente em ~192-224 px, e os
        marcos normalizados reescalam sem perda de geometria.

        A mascara cheia (mask()) continua existindo para o caminho
        de visibilidade da tinta, que precisa dos pixels.
        """
        hands = self._landmarks(
            frame,
            max_width=_PRESENCE_MAX_WIDTH,
            timestamp_ms=timestamp_ms,
        )
        if hands is None:
            return None
        if not hands:
            return HandSignature(0.0, 0.0, 0.0)
        area = 0.0
        centroids = []
        for points in hands:
            hull = cv2.convexHull(points)
            area += float(cv2.contourArea(hull))
            centroids.append(points.mean(axis=0))
        centroid = np.mean(centroids, axis=0)
        return HandSignature(area, float(centroid[0]), float(centroid[1]))
