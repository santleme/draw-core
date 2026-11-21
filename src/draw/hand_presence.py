"""Presenca de mao: adaptador MediaPipe e trabalhadora assincrona.

O adaptador devolve (presenca, moveu) a partir da assinatura da
mascara (area e centroide): e o sinal de atividade que acorda o
descanso da tinta com a mao presente, sem diff de quadro. None e
"sinal indisponivel" (sem MediaPipe ou sem modelo) e quem decide
a cadencia trata None como ativo. A trabalhadora roda a amostragem
em thread propria, latest-wins, fora do laco principal.

Extraido de draw/app.py na modularizacao de 2026-11-05.
"""

import threading

import numpy as np

from . import hand as hand_module
from .contracts import HandPresenceResult, HandSample, HandSignature

# Limiares de movimento da assinatura, calibrados nas mascaras
# reais da sessao de desenho de 2026-10-29 (hand_cam_series03,
# 83 pares consecutivos utilizaveis a 0,33 s, area mediana da
# mao ~147 mil px num quadro de 1758x968): o regime quieto da
# propria sessao de desenho fica em p5 = 6,3 px de centroide e
# 0,7% de area, p10 = 8,9 px e 1,3%; o grosso do desenho esta
# em p25 = 14,5 px e 4,2%, p50 = 23,7 px e 6,9%. O limiar fica
# entre o piso de pausa e o desenho — e e a segunda tranca: a
# primeira e a saida confirmada quieta, que o desenho ativo
# nunca deixa fechar.
_HAND_MOVE_CENTROID_PX = 12.0


_HAND_MOVE_AREA_FRACTION = 0.03
_PRESENCE_HOLD_SAMPLES = 3


def _as_hand_signature(signature):
    """Normaliza a assinatura do masker para HandSignature.

    Maskers antigos e os falsos dos testes devolvem a tupla
    (area, centroide_x, centroide_y); o adaptador aceita os dois
    formatos e entrega sempre o contrato.
    """
    if signature is None or isinstance(signature, HandSignature):
        return signature
    return HandSignature(*signature)


class MediaPipeHandPresence:
    """Presenca de mao sobre o HandMasker: True, False ou None.

    None e "sinal indisponivel" (sem mediapipe ou sem modelo), e
    quem decide a cadencia trata None como ativo: sem sinal, a
    economia nao liga e o app roda como antes.

    sample() devolve um contracts.HandSample (presenca, moveu): a
    assinatura da mascara
    (area, centroide) comparada com a da amostra anterior. E o
    sinal de atividade que acorda o descanso com a mao presente;
    present() segue existindo sobre a mesma leitura.

    close() fecha o masker detido (deterministico, idempotente —
    nada depende de __del__): o caminho de parada da
    trabalhadora de presenca chama por aqui.
    """

    def __init__(self, masker):
        self._masker = masker
        self._signature = None
        self._source_id = None
        self._missed_samples = 0

    def close(self):
        close = getattr(self._masker, "close", None)
        if callable(close):
            close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def present(self, frame):
        return self.sample(frame).present

    def sample(self, frame, timestamp_ms=None, *, source_id=0):
        if source_id != self._source_id:
            self._source_id = source_id
            self._signature = None
            self._missed_samples = 0
        # Caminho preferido: a assinatura direto dos marcos
        # (HandMasker.presence_signature), sem construir nem
        # dilatar a mascara 720p que so servia para contar pixels
        # (pesquisa de performance de 2026-11-03, T3). A mascara
        # segue como reserva para maskers sem o metodo (testes e
        # integracoes antigas). A assinatura nova mede area do
        # casco dos marcos e centroide dos marcos em vez de pixels
        # da mascara dilatada: os limiares abaixo seguem com o
        # mesmo significado (passo de centroide em px, variacao
        # relativa de area) e a direcao conservadora se mantem —
        # area de casco varia MAIS com a pose que a da mascara, e
        # variacao a mais so marca movimento (menos descanso,
        # nunca tinta perdida).
        signature_fn = getattr(self._masker, "presence_signature", None)
        if callable(signature_fn):
            try:
                signature = signature_fn(frame, timestamp_ms)
            except TypeError:
                # Maskers antigos (e os falsos dos testes) so
                # aceitam o frame: o carimbo e opcional por desenho.
                signature = signature_fn(frame)
            signature = _as_hand_signature(signature)
            if signature is None:
                return self._missing_sample(unavailable=True)
            if signature.area == 0:
                return self._missing_sample(unavailable=False)
            return self._update_signature(signature)
        mask = self._masker.mask(frame)
        if mask is None:
            return self._missing_sample(unavailable=True)
        area = int(np.count_nonzero(mask))
        if area == 0:
            return self._missing_sample(unavailable=False)
        ys, xs = np.nonzero(mask)
        signature = HandSignature(float(area), float(xs.mean()), float(ys.mean()))
        return self._update_signature(signature)

    def _missing_sample(self, *, unavailable):
        """Mantém presença curta durante misses consecutivos do landmarker."""
        if (
            self._signature is not None
            and self._missed_samples < _PRESENCE_HOLD_SAMPLES
        ):
            self._missed_samples += 1
            return HandSample(True, False)
        self._signature = None
        self._missed_samples = 0
        return HandSample(None if unavailable else False, False)

    def _update_signature(self, signature):
        previous = self._signature
        self._signature = signature
        self._missed_samples = 0
        if previous is None:
            # A mao apareceu: a transicao de presenca ja acorda
            # por set_hand_presence, e marcar movimento aqui so
            # reseta o relogio de quietude da mao — conservador.
            return HandSample(True, True)
        centroid_step = float(
            np.hypot(
                signature.centroid_x - previous.centroid_x,
                signature.centroid_y - previous.centroid_y,
            )
        )
        area_step = abs(signature.area - previous.area) / previous.area
        moved = (
            centroid_step > _HAND_MOVE_CENTROID_PX
            or area_step > _HAND_MOVE_AREA_FRACTION
        )
        return HandSample(True, moved)


def make_hand_presence(model_path, *, verify):
    """Adaptador de presenca se o modelo de mao existe, senao None.

    verify segue para o HandMasker: um callable que confere o
    arquivo do modelo, ou o None explicito de quem optou por nao
    verificar."""
    if not hand_module.hand_model_available(model_path):
        return None
    return MediaPipeHandPresence(hand_module.HandMasker(model_path, verify=verify))


class HandPresenceWorker:
    """Presenca de mao em thread propria, latest-wins (T4).

    A amostragem MediaPipe rodava SINCRONA no laco principal a
    cada resultado de deteccao: era custo de laço, teto de fps
    (pesquisa de performance de 2026-11-03). Aqui o laco so
    SUBMETE o frame (com o carimbo de captura, que alimenta o
    modo VIDEO do landmarker) e COLHE o resultado mais novo por
    poll, como ja faz com a deteccao de folha. O sinal entregue e
    o mesmo (presenca, moveu) na mesma cadencia de submissao —
    quem muda e onde o custo roda e o atraso de ate um ciclo de
    amostragem ate o set_hand_presence, irrelevante frente a
    cadencia de 250 ms da tinta e a janela de quietude de 1 s.
    """

    def __init__(self, presence):
        self._presence = presence
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._work, name="draw-hand-presence", daemon=True
        )
        self._pending = None
        self._result = None
        self._result_seq = 0
        self._stopped = False
        self._error = None

    def start(self):
        self._thread.start()
        return self

    def submit(self, frame, captured_at, *, source_id):
        """Agenda o quadro com carimbo monotônico em segundos e geração da fonte."""
        with self._condition:
            if self._stopped:
                return
            self._pending = (frame, captured_at, source_id)
            self._condition.notify_all()

    def poll(self, after_seq):
        """HandPresenceResult mais novo que after_seq, ou None."""
        with self._condition:
            if self._result is None or self._result.sequence <= after_seq:
                return None
            return self._result

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stopped
                )
                if self._stopped:
                    return
                frame, captured_at, source_id = self._pending
                self._pending = None
            try:
                sample = self._presence.sample(
                    frame, captured_at * 1000.0, source_id=source_id
                )
            except Exception as error:  # noqa: BLE001 - vai para o laco
                with self._condition:
                    self._error = error
                    self._stopped = True
                    self._condition.notify_all()
                return
            with self._condition:
                if self._stopped:
                    return
                self._result_seq += 1
                self._result = HandPresenceResult(
                    self._result_seq, captured_at, source_id, sample
                )
                self._condition.notify_all()

    @property
    def error(self):
        """A excecao que matou a thread, ou None."""
        with self._condition:
            return self._error

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        # O landmarker fecha junto com a parada: deterministico,
        # nunca pelo __del__ do MediaPipe (que trava quando o GC
        # finaliza um landmarker vazado na thread principal). Se a
        # amostragem seguir em voo alem do join (chamada travada no
        # MediaPipe), o close() do masker nao pendura: sinaliza
        # _closed e quem estava em voo fecha ao retornar.
        close = getattr(self._presence, "close", None)
        if callable(close):
            close()
