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
        self._removal_stale_passes = 0
        self._confirmed = None
        self._provisional = None
        self._provisional_ever = None
        self._retracted_ever = None
        self._low_overlap_streak = 0
        self._turn_fired = False
        self._score_buf = None
        self._hits_buf = None
        self._seen_buf = None

    def restore_confirmed(self, confirmed):
        """Devolve uma copia confirmada ao canvas, unida ao que o
        acumulador ja confirmou nesta geracao.

        Usada pela recuperacao de identidade no same_page: o desenho
        foi movido para a referencia na new_generation (acumulador
        resetado) e volta intacto aqui. Quando o estado esta vazio,
        recria todas as camadas no formato da referencia — o add so
        re-inicializa via _confirmed, entao devolver so ele deixaria
        provisorio/retracao como None.
        """
        confirmed = np.asarray(confirmed, dtype=np.uint8)
        if self._confirmed is None or self._confirmed.shape != confirmed.shape:
            self._removal_streak = None
            self._removal_stale_passes = 0
            self._confirmed = np.zeros_like(confirmed)
            self._provisional = np.zeros(confirmed.shape, np.uint8)
            self._provisional_ever = np.zeros(confirmed.shape, dtype=bool)
            self._retracted_ever = np.zeros(confirmed.shape, dtype=bool)
            self._low_overlap_streak = 0
            self._turn_fired = False
        np.maximum(self._confirmed, confirmed, out=self._confirmed)

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

    def add(
        self,
        prob,
        visible=None,
        *,
        removal_allowed=True,
        valid_mask=None,
        paper_exposed=None,
    ):
        """Uma passada. valid_mask delimita o dominio geometrico
        inspecionado (guarda de borda cruzada com a pegada do
        registro); paper_exposed (bool, forma de prob) e a evidencia
        fotometrica de papel exposto (draw.ink.occlusion): onde e
        False a regiao esta sob suspeita de oclusao — a ausencia de
        tinta ali nao e observavel, e a remocao so anda onde a guarda
        provou papel. O congelamento da regiao sob oclusao fica com a
        guarda (PaperExposureGuard), fora desta camada. Nones mantem o
        comportamento sem guarda."""
        prob = np.asarray(prob, dtype=np.float32)
        if visible is not None:
            visible = np.asarray(visible, dtype=bool)
            if visible.shape != prob.shape:
                # Máscara em geometria errada é bug do chamador:
                # None devolveria a passada aberta (evidência livre sob
                # oclusão); zeros veta tudo — falha fechada dos dois
                # lados, como uma passada abstida.
                visible = np.zeros(prob.shape, dtype=bool)
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != prob.shape:
                valid_mask = None
            elif not valid_mask.any():
                # Dominio inspecionado vazio nao e "tudo falhou a
                # inspecao": a passada nao pode julgar pixel algum
                # (quad corrompido, pegada NaN), entao se abstem
                # como uma passada sem mascara — nao apaga
                # provisorio, nao conta retracao, nao vota contra.
                valid_mask = None
        if paper_exposed is not None and not isinstance(
            paper_exposed, (bool, np.bool_)
        ):
            paper_exposed = np.asarray(paper_exposed, dtype=bool)
            if paper_exposed.shape != prob.shape:
                # Máscara em geometria errada é bug do chamador — o
                # mesmo caso do visible acima: None devolveria a
                # passada ao contrato legado (voto sem prova de
                # papel, fail-open para o dreno), então congela.
                paper_exposed = False
        if self._confirmed is None or self._confirmed.shape != prob.shape:
            self._removal_streak = None
            self._removal_stale_passes = 0
            self._confirmed = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional_ever = np.zeros(prob.shape, dtype=bool)
            self._retracted_ever = np.zeros(prob.shape, dtype=bool)
            self._window.clear()
            self._low_overlap_streak = 0
            self._turn_fired = False
        confirmed_before = self._confirmed > 0
        canvas_before = int(np.count_nonzero(self._confirmed))
        # Oclusao e guarda: pixel que a passada nao pode julgar (a
        # mao sobre a pagina, ou a faixa de borda fora da area util)
        # nao entrega evidencia, nem a favor nem contra. A
        # probabilidade dele entra zerada na janela — soma zero no
        # escore, nao conta hit nem "visto como tinta" —, o
        # julgamento de virada mede a cobertura so sobre a tinta
        # confirmada no dominio inspecionado (ou se abstem quando
        # quase tudo esta fora dele: mao segurando a pagina, ou a
        # guarda alargada cobrindo a tinta, nao e pagina trocada) e
        # a camada provisoria congela onde nao ha visao (ver abaixo).
        evidence = prob if visible is None else np.where(visible, prob, np.float32(0.0))
        if valid_mask is not None:
            evidence = np.where(valid_mask, evidence, np.float32(0.0))
        mask_now = evidence > self.threshold
        # Dominio inspecionado: onde a passada pode distinguir tinta
        # de papel — nao ocluido (visible) e dentro da area util
        # (valid_mask). Sem mascara de oclusao a guarda sozinha
        # delimita esse dominio; sem nenhuma das duas nao ha
        # evidencia negativa confiavel — a virada julga o canvas
        # inteiro (legado) e a remocao se abstem.
        inspected = visible
        if inspected is None:
            inspected = valid_mask
        elif valid_mask is not None:
            inspected = inspected & valid_mask
        turn = False
        if canvas_before >= identity._INK_TURN_MIN_CANVAS_PX and not self._turn_fired:
            if inspected is None:
                judge_turn = True
                turn_low = (
                    int(np.count_nonzero(mask_now & confirmed_before))
                    <= identity._INK_TURN_MAX_OVERLAP * canvas_before
                )
            else:
                inspected_confirmed = confirmed_before & inspected
                inspected_px = int(np.count_nonzero(inspected_confirmed))
                judge_turn = (
                    inspected_px >= identity._OCCLUSION_TURN_MIN_VISIBLE_PX
                    and inspected_px >= 0.5 * canvas_before
                )
                turn_low = judge_turn and (
                    int(np.count_nonzero(mask_now & inspected_confirmed))
                    <= identity._INK_TURN_MAX_OVERLAP * inspected_px
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
        if self._removal_streak is None:
            self._removal_streak = np.zeros(prob.shape, dtype=np.uint8)
        # Assinatura barata do streak antes da atualização: se
        # nada muda na passada — pixel ocluído ou inspecionado
        # sem papel provado segura a contagem — o pendurado
        # envelhece e o worker pode devolver a cadência ao
        # probe ocioso (Devin Review do PR #144).
        streak_sig = (
            int(np.count_nonzero(self._removal_streak)),
            int(self._removal_streak.sum(dtype=np.uint64)),
        )
        # Sem dominio inspecionado nao ha evidencia negativa
        # confiavel; com a guarda ativa ela sozinha delimita
        # onde a ausencia foi de fato observada.
        witnessed = False
        # A testemunha da remocao precisa de evidencia negativa
        # real: a mascara de oclusao do chamador, ou o dominio
        # geometrico da guarda. Sem evidencia fotometrica
        # (paper_exposed=None) a remocao segue o contrato legado
        # — o chamador que quer "incerteza congela" passa mascara
        # all-False, nao None.
        if (
            removal_allowed
            and canvas_before
            and (visible is not None or valid_mask is not None)
        ):
            retained = np.count_nonzero(confirmed_before & inspected & mask_now)
            witnessed = (
                retained >= registration._REMOVAL_RETAINED_FRACTION * canvas_before
                or retained >= registration._REMOVAL_WITNESS_FLOOR_PX
            )
            missing = (
                confirmed_before
                & inspected
                & np.isfinite(prob)
                & (prob <= registration._REMOVAL_LOW_PROB)
            )
            if paper_exposed is not None:
                # Sob suspeita de oclusao a ausencia nao e
                # observavel: so conta o pixel em papel provado.
                missing &= paper_exposed
            # O reset NAO depende da testemunha: ela so governa o
            # que conta como ausencia. Pixel inspecionado e não
            # ausente vê o streak zerar onde a guarda provou papel
            # exposto; pixel visível fora do domínio inspecionado
            # (guarda que alargou por cima de uma contagem em
            # curso, borda que a pegada deixou) é prova
            # inconclusiva — não pode nem apagar o traço nem
            # segurar a trabalhadora acordada, a contagem velha
            # zera. O incremento de voto novo continua exigindo a
            # testemunha; o streak já armado é exceção abaixo.
            # Ocluído (sem visão) ou
            # inspecionado sem papel provado não informa nada — o
            # streak segura, não zera: a ausência do F-10 segue
            # precisando ser persistente, mas não precisa ser em
            # passadas consecutivas nem desobstruídas (a
            # testemunha e a mão oscilam em cena real — medido:
            # retained 1,5k~13,9k alternando no sim-hand, streak
            # nunca completava).
            # Streak já armado sob testemunha viva termina de
            # contar com a própria evidência (missing =
            # inspecionado + papel provado + prob baixa): apagar
            # o resto da tinta zera `retained` e a testemunha
            # nunca mais vive — o voto ficava congelado em 1-5,
            # a trabalhadora sondava a 1 Hz para sempre e o
            # fantasma seguia publicado no canvas (Devin Review
            # do PR #144). Sem guarda avaliada na passada o
            # caminho segue fail-closed.
            unwitnessed_votes = (
                None if paper_exposed is None else missing & (self._removal_streak > 0)
            )
            voting = missing if witnessed else unwitnessed_votes
            exposed = inspected if paper_exposed is None else inspected & paper_exposed
            # Leitura no nível do voto é evidência positiva por
            # si mesma e zera o streak mesmo onde a guarda não
            # provou papel exposto: o veto dark_ink — tinta
            # forte sobre pixel escuro nunca prova papel —
            # descreve todo pixel de traço real (prob >=
            # vote_hit e luma escura), e segurar o streak dele
            # deixava faltas espalhadas somarem até apagar a
            # tinta (Devin Review do PR #144). O veto segue
            # barrando o incremento via paper_exposed; este
            # braço só restaura o reset. A barra é a mesma do
            # observe (ink_support_threshold=vote_hit): tudo
            # que o veto consegue derrubar volta a zerar.
            seen_ink = evidence >= self._config.vote_hit
            inconclusive = ~inspected
            if visible is not None:
                inconclusive &= visible
            resettable = ~missing & (exposed | seen_ink | inconclusive)
            self._removal_streak[resettable] = 0
            if voting is not None and np.any(voting):
                self._removal_streak[voting] = np.minimum(
                    self._removal_streak[voting].astype(np.uint16) + 1,
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
        if not np.any(self._removal_streak):
            self._removal_stale_passes = 0
        elif streak_sig == (
            int(np.count_nonzero(self._removal_streak)),
            int(self._removal_streak.sum(dtype=np.uint64)),
        ):
            self._removal_stale_passes += 1
        else:
            self._removal_stale_passes = 0
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
            # Pixels fora de valid_mask (guarda de borda) nao congelam
            # provisorio velho quando a guarda alarga — o clipe de
            # valid_mask abaixo os remove de qualquer forma.
            occluded = ~visible
            provisional_now = provisional_now | (previous_provisional & occluded)
        if valid_mask is not None:
            provisional_now = provisional_now & valid_mask
        retracted = previous_provisional & ~provisional_now & ~confirmed_now
        if valid_mask is not None:
            # Pixel excluido so pela guarda (a re-ancoragem a alargou
            # sobre a faixa) nao e retracao: a tinta nao sumiu — a
            # passada deixou de inspecionar ali.
            retracted &= valid_mask
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
            "removed_confirmed_px": removed_px,
        }
        return result
