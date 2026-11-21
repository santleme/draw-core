"""Acumulador de tinta em camadas: voto entre passadas."""

from collections import deque

import cv2
import numpy as np

from . import hysteresis, identity, registration, votes
from .config import InkConfig


class LayeredInkAccumulator:
    """Tinta de uma geracao em duas camadas: provisoria e confirmada.

    Cada passada entrega o mapa de probabilidade do modelo. A
    camada provisoria e o que as passadas da janela viram acima do
    limiar e ainda nao confirmou: ela existe para a tela mostrar a
    tinta nova na primeira passada, e so para a tela. A camada
    confirmada e o canvas: um pixel so entra nela pelas regras de
    voto acima com a porta de fronteira (evidencia sub-limiar nao
    expande a borda da tinta ja confirmada), e dela nao sai
    dentro da geracao.

    Pixel provisorio que sai da janela sem confirmar e retratado:
    some da camada provisoria e entra na contagem de retratados.
    E essa contagem que mede o risco declarado da publicacao
    especulativa, o page-clean pegar mobiliario: a regua de replay
    le quanta tinta provisoria some depois.

    So a trabalhadora de tinta toca nesta classe.
    """

    def __init__(self, threshold=0.7, config=None):
        self.threshold = float(threshold)
        self._config = config if config is not None else InkConfig()
        self.reset()

    def reset(self):
        self._window = deque(maxlen=votes._INK_VOTE_WINDOW)
        self._removal_streak = None
        self._confirmed = None
        self._provisional = None
        self._provisional_ever = None
        self._retracted_ever = None
        self._low_overlap_streak = 0
        self._turn_fired = False
        self._score_buf = None
        self._hits_buf = None
        self._seen_buf = None
        self._hold = None

    def _window_stats(self):
        """(soma, hits, visto-como-tinta) da janela, sem empilhar.

        O caminho antigo fazia np.stack da janela (3 mapas float32
        de pagina inteira, ~12 MB) e somava/contava/maximizava
        sobre a pilha a cada passada — o maior transitorio do
        acumulador (pesquisa de performance de 2026-11-03, T10).
        Aqui as tres reducoes saem por acumulacao em buffers
        reusados: a soma adiciona os mesmos mapas na mesma ordem
        (resultado float identico), hits e visto sao contagens
        inteiras/booleanas exatas. Os buffers sao devolvidos para
        leitura imediata; a proxima chamada os sobrescreve.
        """
        window = list(self._window)
        shape = window[0].shape
        if self._score_buf is None or self._score_buf.shape != shape:
            self._score_buf = np.zeros(shape, dtype=np.float32)
            self._hits_buf = np.zeros(shape, dtype=np.int32)
            self._seen_buf = np.zeros(shape, dtype=bool)
        score = self._score_buf
        np.copyto(score, window[0])
        for item in window[1:]:
            np.add(score, item, out=score)
        hits = self._hits_buf
        hit = self._config.vote_hit
        hits[...] = window[0] >= hit
        for item in window[1:]:
            np.add(hits, item >= hit, out=hits)
        seen = self._seen_buf
        seen[...] = window[0] > self.threshold
        for item in window[1:]:
            np.logical_or(seen, item > self.threshold, out=seen)
        return score, hits, seen

    def add(self, prob, visible=None, *, removal_allowed=True, paper_ok=None):
        """Uma passada. paper_ok (bool, forma de prob) e a evidencia
        fotometrica de papel exposto (draw.ink.paper_evidence): onde e
        False — e por occlusion_hold_passes passadas depois — a regiao
        congela como oclusao (sem voto, sem remocao, provisoria
        parada). None mantem o comportamento sem guarda."""
        prob = np.asarray(prob, dtype=np.float32)
        if visible is not None:
            visible = np.asarray(visible, dtype=bool)
            if visible.shape != prob.shape:
                visible = None
        # Sem mascara de visibilidade nao ha evidencia negativa
        # confiavel — a guarda nao a inventa.
        had_visibility = visible is not None
        if paper_ok is not None:
            paper_ok = np.asarray(paper_ok, dtype=bool)
            if paper_ok.shape != prob.shape:
                paper_ok = None
        if self._confirmed is None or self._confirmed.shape != prob.shape:
            self._removal_streak = None
            self._confirmed = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional_ever = np.zeros(prob.shape, dtype=bool)
            self._retracted_ever = np.zeros(prob.shape, dtype=bool)
            self._window.clear()
            self._low_overlap_streak = 0
            self._turn_fired = False
        if paper_ok is not None:
            if self._hold is None or self._hold.shape != prob.shape:
                self._hold = np.zeros(prob.shape, dtype=np.uint8)
            np.subtract(self._hold, 1, out=self._hold, where=self._hold > 0)
            self._hold[~paper_ok] = self._config.occlusion_hold_passes
            frozen = ~paper_ok | (self._hold > 0)
            visible = ~frozen if visible is None else visible & ~frozen
        confirmed_before = self._confirmed > 0
        canvas_before = int(np.count_nonzero(self._confirmed))
        # Oclusao: pixel que a passada nao pode ver (a mao sobre a
        # pagina) nao entrega evidencia, nem a favor nem contra. A
        # probabilidade dele entra zerada na janela — soma zero no
        # escore, nao conta hit nem "visto como tinta" —, o
        # julgamento de virada mede a cobertura so sobre a tinta
        # confirmada visivel (ou se abstem quando quase tudo esta
        # coberto: mao segurando a pagina nao e pagina trocada) e a
        # camada provisoria congela onde nao ha visao (ver abaixo).
        evidence = prob if visible is None else np.where(visible, prob, np.float32(0.0))
        mask_now = evidence > self.threshold
        turn = False
        if canvas_before >= identity._INK_TURN_MIN_CANVAS_PX and not self._turn_fired:
            if visible is None:
                judge_turn = True
                turn_low = (
                    int(np.count_nonzero(mask_now & confirmed_before))
                    <= identity._INK_TURN_MAX_OVERLAP * canvas_before
                )
            else:
                visible_confirmed = confirmed_before & visible
                visible_px = int(np.count_nonzero(visible_confirmed))
                judge_turn = (
                    visible_px >= identity._OCCLUSION_TURN_MIN_VISIBLE_PX
                    and visible_px >= 0.5 * canvas_before
                )
                turn_low = judge_turn and (
                    int(np.count_nonzero(mask_now & visible_confirmed))
                    <= identity._INK_TURN_MAX_OVERLAP * visible_px
                )
            if judge_turn:
                if turn_low:
                    self._low_overlap_streak += 1
                    if self._low_overlap_streak >= identity._INK_TURN_PASSES:
                        turn = True
                        self._turn_fired = True
                else:
                    self._low_overlap_streak = 0
        self._window.append(evidence.copy())
        score, hits, seen_as_ink = self._window_stats()
        confirmed_rule = (score >= self._config.vote_score_strong) | (
            (hits >= votes._INK_VOTE_MIN_HITS) & (score >= self._config.vote_score)
        )
        if np.any(confirmed_rule):
            # Porta de fronteira (ver votes._INK_FRONTIER_RADIUS):
            # dentro da faixa do ombro da tinta (confirmada ou
            # vista acima do limiar nesta janela), so confirma
            # quem alguma passada viu como tinta; a soma
            # sub-limiar das copias deslocadas nao engrossa o
            # traco nem poe camada nova sobre ele.
            anchor = seen_as_ink | confirmed_before
            near_anchor = (
                cv2.dilate(
                    anchor.astype(np.uint8),
                    votes._INK_FRONTIER_KERNEL,
                )
                > 0
            )
            confirmed_rule &= seen_as_ink | ~near_anchor
        if self._config.hysteresis and np.any(confirmed_rule):
            # Poeira isolada: um candidato so confirma se a media da janela
            # e >= t_low e o componente dele tem uma ancora (media >= t_high
            # ou tinta ja confirmada). Escala da media, nao do seen.
            cfg = self._config
            # Janela ainda enchendo: passada faltante e evidencia zero.
            mean = score / votes._INK_VOTE_WINDOW
            # Tolerancia de float32: 3 x 0.9 somado em float32 e dividido por 3 da
            # 0.8999999 < 0.9, e o traco constante exatamente em t_high nunca
            # confirmava (medido na auditoria de 2026-11-19).
            eps = hysteresis.FLOAT_EPS
            strong = confirmed_rule & (mean >= cfg.t_high - eps)
            weak = confirmed_rule & ~strong & (mean >= cfg.t_low - eps)
            kept = hysteresis.keep_anchored(weak, strong | confirmed_before)
            confirmed_rule = strong | kept
        new_confirmed = confirmed_rule & ~confirmed_before
        new_confirmed_px = int(np.count_nonzero(new_confirmed))
        if new_confirmed_px:
            self._confirmed[new_confirmed] = 255
        removed_px = 0
        if self._config.removal:
            if self._removal_streak is None:
                self._removal_streak = np.zeros(prob.shape, dtype=np.uint8)
            # Sem máscara de visibilidade não há evidência negativa confiável.
            witnessed = False
            if removal_allowed and had_visibility and canvas_before:
                retained = np.count_nonzero(confirmed_before & visible & mask_now)
                witnessed = (
                    retained >= registration._REMOVAL_RETAINED_FRACTION * canvas_before
                    or retained >= registration._REMOVAL_WITNESS_FLOOR_PX
                )
            if witnessed:
                missing = (
                    confirmed_before
                    & visible
                    & np.isfinite(prob)
                    & (prob <= registration._REMOVAL_LOW_PROB)
                )
                # Com testemunha: pixel à vista e não ausente vê o
                # streak zerar (evidência positiva ou prova
                # inconclusiva); pixel ausente conta. Ocluído ou sem
                # testemunha não informa nada — o streak segura, não
                # zera: a ausência do F-10 segue precisando ser
                # persistente, mas não precisa ser em passadas
                # consecutivas nem desobstruídas (a testemunha e a
                # mão oscilam em cena real — medido: retained
                # 1,5k~13,9k alternando no sim-hand, streak nunca
                # completava).
                resettable = visible & ~missing
                self._removal_streak[resettable] = 0
                self._removal_streak[missing] = np.minimum(
                    self._removal_streak[missing].astype(np.uint16) + 1,
                    registration._REMOVAL_PASSES,
                ).astype(np.uint8)
            removed = self._removal_streak >= registration._REMOVAL_PASSES
            removed_px = int(np.count_nonzero(removed))
            if removed_px:
                self._confirmed[removed] = 0
                self._removal_streak[removed] = 0
                # Nenhuma memória de voto pode ressuscitar o traço apagado.
                for item in self._window:
                    item[removed] = 0
                self._score_buf = self._hits_buf = self._seen_buf = None
                seen_as_ink[removed] = False
                score[removed] = 0
                hits[removed] = 0
        confirmed_now = self._confirmed > 0
        provisional_now = seen_as_ink & ~confirmed_now
        if self._config.hysteresis:
            provisional_now = hysteresis.filter_provisional(
                provisional_now,
                evidence,
                confirmed_now,
                t_high=self._config.t_high,
                min_px=self._config.provisional_min_px,
            )
            if visible is not None:
                # Ancora nova no lado visivel nao revela o lado coberto: o
                # coberto so mantem o que ja era mostrado (merge abaixo).
                provisional_now = provisional_now & visible
        previous_provisional = self._provisional > 0
        if visible is not None:
            # Sem visao nao ha retracao nem estreia: o provisorio
            # coberto fica como estava. Retrair e afirmar que a
            # tinta sumiu — e quem esta sobre ela e a mao, nao o
            # apagador; sem este congelamento, toda oclusao virava
            # retracao na regua e a tinta piscava ao reaparecer.
            provisional_now = provisional_now | (previous_provisional & ~visible)
        retracted = previous_provisional & ~provisional_now & ~confirmed_now
        if np.any(retracted):
            self._retracted_ever |= retracted
        self._provisional_ever |= provisional_now
        provisional_changed = bool(np.any(provisional_now != previous_provisional))
        # bool→u8 *255: np.where+astype gastava dois temporários de
        # página inteira por passada (~1,7 ms nesta VM; ~0,04 ms
        # assim — issue #102). Buffer reusado com copyto+*= mediu
        # PIOR (~+4 ms) — provável custo de laço escalar no *= de
        # atributo; astype*255 repõe o array como sempre fez.
        self._provisional = provisional_now.astype(np.uint8) * 255
        result = {
            "confirmed": self._confirmed.copy(),
            "provisional": self._provisional.copy(),
            "confirmed_changed": new_confirmed_px > 0 or removed_px > 0,
            "provisional_changed": provisional_changed,
            "new_confirmed_px": new_confirmed_px,
            "confirmed_px": int(np.count_nonzero(self._confirmed)),
            "provisional_px": int(np.count_nonzero(self._provisional)),
            "provisional_ever_px": int(np.count_nonzero(self._provisional_ever)),
            "retracted_px_total": int(np.count_nonzero(self._retracted_ever)),
            "turn": turn,
        }
        if self._config.removal:
            result["removed_confirmed_px"] = removed_px
        return result
