"""Ancora da folha: migracao, arbitragem e follow da geometria."""

import cv2
import numpy as np

from . import registration

# Limiares de assentamento da geometria (medidos na frente 12,
# draw-lab/live-audit-2026-11-10/12-estudo-pouso.md): passadas
# seguidas com deslocamento medio da grade sob a matriz de registro
# > _ANCHOR_MIGRATION_DISPLACEMENT_PX abrem evento; o evento fecha
# na _ANCHOR_MIGRATION_PASSES-esima passada assentada. O tremor do
# regime de registro (<= ~3 px) fica abaixo do limiar e spike de 1
# passada nao sustenta. A migracao v1 que estes limiares nomeavam
# foi removida com a flag dela (veredito da propria prova, FALHAS
# M2); os valores seguem como os limites da arbitragem e do follow.
_ANCHOR_MIGRATION_DISPLACEMENT_PX = 8.0
_ANCHOR_MIGRATION_PASSES = 3
_ANCHOR_MIGRATION_GRID = 16


# Arbitragem de ancora por conteudo (v3; frente 19, 2026-11-10;
# draw-lab/live-audit-2026-11-10/19-migracao-v2-arbitragem.md): a
# migracao da v1 foi reprovada na propria prova (FALHAS M2) porque a
# inversa do registro assume a tinta parada na camera — certo pro
# modo espurio do detector, errado pro pouso, em que a tinta anda
# com a folha. As duas hipoteses sao indistinguiveis pelo quad; a
# v3 arbitra pelo CONTEUDO: num streak de assentamento (passadas
# deslocadas > _ANCHOR_MIGRATION_DISPLACEMENT_PX, mutuamente
# consistentes a _ARB_SETTLE_PX, K = _ANCHOR_MIGRATION_PASSES),
# compara a fracao da massa da prob que cai sobre o canvas
# confirmado pre-evento na colocacao identidade (Sp, hipotese
# pagina/pouso) e na registrada (Sc, hipotese camera/espurio).
# POUSO troca a ancora SEM mover o estado; ESPURIO nao faz nada.
# Enquanto o evento esta aberto, a passada NAO vota (visible=0): a
# copia deslocada confirmava pelo voto antes de qualquer veredito
# (mecanismo que reprovou a v2 sem abstencao). Mecanismo fixo do
# preset de producao (a flag do Tier A foi removida).
_ARB_SETTLE_PX = 6.0
_ARB_DELTA = 0.05
_ARB_MIN_CANVAS_PX = 200
_ARB_MIN_PASS_PX = 200


# Mobilidade da ancora por divergencia assentada (frente 34,
# 2026-11-10; draw-lab/live-audit-2026-11-10/34-desenho-ancora-
# movel.md): a vigia v3 fechou o split errado do puxao e, ao
# fechar, expos a pergunta que nenhum mecanismo respondia — quem
# move a ancora quando ninguem divide a pagina? Sem resposta, a
# ancora congela na pose de abertura, a vigia de identidade fica
# cega a ~150 px pelo resto da geracao (F33, G1/G3) e a
# arbitragem cicla sem veredito com o canvas congelado (o
# livelock da F33). O FOLLOW e a regra do Lucas (2026-11-01) como
# mecanismo: mover o caderno = so re-registro por substituicao.
# Divergencia assentada sustentada (deslocamento da grade >
# _FOLLOW_DISPLACEMENT_PX, degrau entre quads consecutivos <=
# _FOLLOW_SETTLE_PX por _FOLLOW_PASSES passadas) sem decisao de
# identidade pendente RE-ANCORA: ancora := mediana dos quads do
# streak, estado do acumulador INTOCADO — a grade ja e a da
# folha, e o carry que deforma o estado pela homografia do raio
# da camera e o da migracao v1, reprovado no FALHAS M2. Enquanto
# o evento esta aberto a passada nao vota (a copia deslocada e o
# defeito que a abstencao da arbitragem ja fechou). Precedencia:
# a arbitragem tem o primeiro veredito de cada passada — veredito
# decisivo com acao pre-empta; spurious retem o FOLLOW pelo
# episodio inteiro; indeterminate ou ausencia de veredito
# (creep: a consistencia par-a-par nunca fecha) deixam a
# geometria decidir. Mecanismo fixo do preset de producao
# (a flag do Tier A foi removida).
_FOLLOW_DISPLACEMENT_PX = _ANCHOR_MIGRATION_DISPLACEMENT_PX
# A ancora COM o residuo so explica a pose quando a deixa (quase) parada: o
# residuo e inteiro, o erro de arredondamento fica abaixo disto. Reduzir
# o deslocamento pela metade nao explica nada — um deslize real de 14 px com
# residuo de 6 nao pode virar 8 (review do #180).
_FOLLOW_PASSES = _ANCHOR_MIGRATION_PASSES
_FOLLOW_SETTLE_PX = 6.0


def _matrix_grid_displacement(matrix, shape):
    """Deslocamento medio (px) de uma grade densa da pagina sob matrix."""
    height, width = shape
    xs = np.linspace(0.0, float(width), _ANCHOR_MIGRATION_GRID)
    ys = np.linspace(0.0, float(height), _ANCHOR_MIGRATION_GRID)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(grid, np.asarray(matrix, dtype=np.float64))
    return float(np.mean(np.linalg.norm(moved - grid, axis=-1)))


def _overlap_fraction(prob, snapshot, *, denominator_mass=None):
    """Fracao da massa de `prob` que cai no canvas confirmado.

    `denominator_mass` pode preservar a massa do mapa apos a permutacao e
    antes de uma translacao com corte. A massa anterior a permutacao nao serve
    para rotacoes de quarto de volta: `_permute_prob` redimensiona a grade e
    pode mudar sua soma. Sem um denominador explicito, o mapa ja registrado e
    normalizado pela massa que permaneceu na grade.
    """
    total = (
        float(np.sum(prob, dtype=np.float64))
        if denominator_mass is None
        else float(denominator_mass)
    )
    if total <= 0.0:
        return 0.0
    matched = float(np.sum(prob[snapshot], dtype=np.float64))
    return min(matched / total, 1.0)


def _residual_alignment_scores(raw_prob, snapshot, visible, grid_shift, anchor_roll):
    """Compare a pending residual with the same observable page evidence."""
    raw_prob = np.asarray(raw_prob, dtype=np.float32)
    snapshot = np.asarray(snapshot, dtype=bool)
    if raw_prob.ndim != 2 or snapshot.shape != raw_prob.shape:
        return None
    if not np.isfinite(raw_prob).all():
        return None
    if visible is not None:
        visible = np.asarray(visible, dtype=bool)
        if visible.shape != raw_prob.shape:
            return None
        raw_prob = np.where(visible, raw_prob, 0.0).astype(np.float32, copy=False)
    page_prob = (
        raw_prob
        if anchor_roll == 0
        else registration._permute_prob(raw_prob, -anchor_roll)
    )
    denominator_mass = float(np.sum(page_prob, dtype=np.float64))
    support_px = int(np.count_nonzero(page_prob > 0.5))
    if denominator_mass <= 0.0:
        return None
    score_without = _overlap_fraction(
        page_prob, snapshot, denominator_mass=denominator_mass
    )
    translated = registration._translate_grid(page_prob, *grid_shift)
    score_with = _overlap_fraction(
        translated, snapshot, denominator_mass=denominator_mass
    )
    return {
        "score_without_residual": score_without,
        "score_with_residual": score_with,
        "support_px": support_px,
    }


class _AnchorSteps:
    """Passos de ancora do worker: migracao, arbitragem, follow."""

    def _arb_close_event(self):
        self._arb_event_open = False
        self._arb_streak_quads = []
        self._arb_scores = []
        self._arb_snapshot = None

    def _follow_close_event(self):
        self._follow_event_open = False
        self._follow_streak = 0
        self._follow_streak_quads = []
        self._follow_hold = False

    def _pose_displacement(
        self, corners, shape, residual=(0, 0), *, residual_fresh=True
    ):
        """Deslocamento posicional e ramo escolhido, em px da grade.

        Com residuo de registro (F-11) ha duas leituras: contra a ancora SEM
        o residuo (erro de rastreio: os cantos nao andaram, so o conteudo do
        warp; o residuo nao e movimento da pagina) e contra a ancora COM ele
        (folha movida de verdade e acompanhada pelo tracker: o warp ja esta
        alinhado e o residuo so cancela o registro). A segunda so vale
        quando o residuo e fresco e EXPLICA a pose por inteiro, dentro do
        limite configurado. Residuo velho continua corrigindo o registro do
        conteudo, mas nao pode suprimir um evento geometrico (review #180)."""
        trigger_anchor = (
            self._anchor_corners
            if self._anchor_roll == 0
            else np.roll(self._anchor_corners, -self._anchor_roll, axis=0)
        )
        matrix = registration._registration_matrix(trigger_anchor, corners, shape)
        displacement = _matrix_grid_displacement(matrix, shape)
        dx, dy = residual
        if not (dx or dy):
            return displacement, "pose"
        if not residual_fresh:
            return displacement, "stale_residual"
        limit = self._config.residual_explains_px
        if displacement > limit:
            shifted = registration._shift_anchor_corners(trigger_anchor, dx, dy, shape)
            effective = _matrix_grid_displacement(
                registration._registration_matrix(shifted, corners, shape), shape
            )
            if effective <= limit:
                return effective, "fresh_residual_explains_pose"
            return displacement, "fresh_residual_does_not_explain_pose"
        return displacement, "pose_within_tolerance"

    def _arb_restart_streak(self):
        confirmed = self._accumulator._confirmed
        self._arb_snapshot = None if confirmed is None else confirmed > 0
        self._arb_streak_quads = []
        self._arb_scores = []

    def _anchor_bookkeeping(
        self, streak_quads, new_anchor, reg_anchor_median, reg_deadband
    ):
        if reg_anchor_median:
            self._anchor_samples = [quad.copy() for quad in streak_quads]
        if reg_deadband:
            self._reg_frozen_corners = new_anchor.copy()
            self._reg_deadband_over = 0

    def _anchor_arbitration_step(
        self,
        generation,
        corners,
        shape,
        identity_owned,
        reg_anchor_median,
        reg_deadband,
        raw_prob,
        registered_prob,
        visible=None,
        grid_shift=(0, 0),
        residual_fresh=True,
        reg_info=None,
        camera_denominator_mass=None,
        camera_score_available=True,
    ):
        """Avanca o evento de arbitragem; arbitra no fecho do streak.

        Chamado sob a Condition.
        Devolve (status, evento): "closed" (sem evento; a passada
        vota), "open" (evento aberto sem veredito; a passada NAO
        vota — a copia deslocada nao pode confirmar pelo voto
        enquanto a hipotese nao foi julgada) ou "verdict" (o evento
        fechou nesta passada; ela vota, re-registrada na ancora nova
        se o veredito foi pouso). Presenca perdida e identidade em
        curso fecham o evento sem veredito: a fronteira de
        geracao/identidade e a dona desses casos.
        """
        old_anchor = self._anchor_corners
        if identity_owned or corners is None or old_anchor is None:
            self._arb_close_event()
            return "closed", None
        raw_prob = np.asarray(raw_prob, dtype=np.float32)
        page_prob = raw_prob
        page_mask_valid = True
        if visible is not None:
            visible = np.asarray(visible, dtype=bool)
            if visible.shape != raw_prob.shape:
                page_prob = np.zeros_like(raw_prob)
                page_mask_valid = False
            else:
                page_prob = np.where(visible, raw_prob, 0.0).astype(
                    np.float32, copy=False
                )
        observed_prob = page_prob
        # O gatilho compara em termos POSICIONAIS: uma ancora que ja
        # carrega rotulos rolados (pouso rodado anterior) nao pode
        # fazer toda passada seguinte parecer deslocada pela
        # permutacao — o deslocamento que abre evento e o da pose,
        # nao o da rotulacao.
        displacement, displacement_source = self._pose_displacement(
            corners, shape, grid_shift, residual_fresh=residual_fresh
        )
        if reg_info is not None:
            reg_info["arbitration_pose_displacement_px"] = displacement
            reg_info["arbitration_pose_displacement_source"] = displacement_source
        if displacement <= _ANCHOR_MIGRATION_DISPLACEMENT_PX:
            self._arb_close_event()
            return "closed", None
        if not self._arb_event_open:
            self._arb_event_open = True
            self._arb_restart_streak()
        consistent = all(
            float(np.max(np.linalg.norm(corners - quad, axis=1))) <= _ARB_SETTLE_PX
            for quad in self._arb_streak_quads
        )
        if not consistent:
            # O movimento continua: a pose candidata e a mais
            # recente e o snapshot renova (o canvas nao mudou — as
            # passadas do evento nao votam).
            self._arb_restart_streak()
        self._arb_streak_quads.append(corners.copy())
        snapshot = self._arb_snapshot
        raw_total = float(np.sum(observed_prob, dtype=np.float64))
        registered_total = float(np.sum(registered_prob))
        camera_score_available = bool(camera_score_available)
        if camera_score_available:
            if camera_denominator_mass is None:
                camera_denominator_mass = registered_total
            camera_score_available = bool(
                np.isfinite(camera_denominator_mass) and camera_denominator_mass > 0.0
            )
        if snapshot is None or raw_total <= 0.0:
            score_page = 0.0
            score_page_without_residual = 0.0
            cyclic = (0.0, 0.0, 0.0)
            cyclic_without_residual = (0.0, 0.0, 0.0)
        else:
            # Com roll parado (frente 27): a evidencia da pagina e o
            # S_roll da familia ciclica, reclassificado — o warp
            # entregue ja esta em coordenadas da pagina e a unica
            # coisa errada nele e a rotacao parada (permutacao
            # pura), entao a hipotese-pagina le a prob des-rotada
            # pelo roll corrente. Medido na era 3 capturada: 0,193
            # contra 0,002 do Sc. Os ciclicos de rotacao NOVA
            # seguem zerados (segunda rotacao = abstencao).
            page_before_translation = (
                observed_prob
                if self._anchor_roll == 0
                else registration._permute_prob(observed_prob, -self._anchor_roll)
            )
            # O residuo de registro (grade da ancora) vem DEPOIS da permutacao:
            # T . Pi_k(P). A camera ja o recebe pela ancora efetiva (T . M . P).
            dx, dy = grid_shift
            page_mass_before_translation = float(
                np.sum(page_before_translation, dtype=np.float64)
            )
            score_page_without_residual = _overlap_fraction(
                page_before_translation,
                snapshot,
                denominator_mass=page_mass_before_translation,
            )
            page_prob = registration._translate_grid(page_before_translation, dx, dy)
            # Normaliza no domínio depois da permutacao/resize e antes do
            # corte da translacao. O que sai da grade conta como evidência
            # perdida; usar a massa restante premiaria uma hipótese que só
            # preserva um fragmento pequeno perfeitamente alinhado.
            score_page = _overlap_fraction(
                page_prob,
                snapshot,
                denominator_mass=page_mass_before_translation,
            )
            # Colocacoes ciclicas (frentes 23/24): a hipotese "a
            # folha assentou rodada de k x 90 graus" pontuada como a
            # medicao manda — a prob ENTREGUE permutada por -k
            # (rot90 + resize) contra o snapshot. A forma anterior
            # (registrar com os cantos rolados) marcava 0,005 onde
            # esta marca 0,23-0,27 na mesma passada real: reprovada
            # e substituida. So dentro do evento, e so a partir de
            # uma ancora posicional (roll == 0): uma segunda rotacao
            # na mesma geracao e regime nao medido — abstem.
            if self._anchor_roll != 0:
                cyclic = (0.0, 0.0, 0.0)
                cyclic_without_residual = (0.0, 0.0, 0.0)
            else:
                cyclic = []
                cyclic_without_residual = []
                for k in (1, 2, 3):
                    permuted = registration._permute_prob(observed_prob, -k)
                    moved = registration._translate_grid(permuted, dx, dy)
                    denominator_mass = float(np.sum(permuted, dtype=np.float64))
                    cyclic_without_residual.append(
                        _overlap_fraction(
                            permuted,
                            snapshot,
                            denominator_mass=denominator_mass,
                        )
                    )
                    cyclic.append(
                        _overlap_fraction(
                            moved,
                            snapshot,
                            denominator_mass=denominator_mass,
                        )
                    )
                cyclic = tuple(cyclic)
                cyclic_without_residual = tuple(cyclic_without_residual)
        # Score each hypothesis independently. A valid page candidate still
        # has evidence when the registered camera map misses the canvas; its
        # measurable camera score is then exactly zero, not a reason to erase
        # the page score as well.
        score_camera = (
            _overlap_fraction(
                registered_prob,
                snapshot,
                denominator_mass=camera_denominator_mass,
            )
            if snapshot is not None and camera_score_available
            else 0.0
        )
        pass_mass = int(np.count_nonzero(observed_prob > 0.5))
        page_score_available = page_mask_valid and raw_total > 0.0
        self._arb_scores.append(
            (
                score_page,
                score_camera,
                cyclic[0],
                cyclic[1],
                cyclic[2],
                pass_mass,
                score_page_without_residual,
                (float(camera_denominator_mass) if camera_score_available else None),
                camera_score_available,
                *cyclic_without_residual,
                page_score_available,
            )
        )
        if len(self._arb_streak_quads) < _ANCHOR_MIGRATION_PASSES:
            return "open", None
        mean_page = float(np.mean([s[0] for s in self._arb_scores]))
        mean_camera = float(np.mean([s[1] for s in self._arb_scores]))
        mean_cyclic = [
            float(np.mean([s[2 + k] for s in self._arb_scores])) for k in range(3)
        ]
        mean_mass = float(np.mean([s[5] for s in self._arb_scores]))
        mean_page_without_residual = float(np.mean([s[6] for s in self._arb_scores]))
        valid_camera_masses = [s[7] for s in self._arb_scores if s[8]]
        mean_camera_denominator_mass = (
            float(np.mean(valid_camera_masses)) if valid_camera_masses else None
        )
        all_camera_scores_available = all(s[8] for s in self._arb_scores)
        mean_cyclic_without_residual = [
            float(np.mean([s[9 + k] for s in self._arb_scores])) for k in range(3)
        ]
        all_page_scores_available = all(s[12] for s in self._arb_scores)
        canvas_px = (
            0
            if self._arb_snapshot is None
            else int(np.count_nonzero(self._arb_snapshot))
        )
        rotation_k = 0
        insufficient_evidence = (
            canvas_px < _ARB_MIN_CANVAS_PX or mean_mass < _ARB_MIN_PASS_PX
        )
        residual_pending = bool(grid_shift[0] or grid_shift[1]) and (
            canvas_px > 0 and insufficient_evidence
        )
        residual_pending_reason = (
            "insufficient_observable_ink" if residual_pending else None
        )
        if residual_pending:
            # Sem pixels observaveis que discriminem o warp antigo do cru,
            # preservar ou limpar o residuo falha em continuacoes igualmente
            # validas. Mantem a ancora e abstem ate uma leitura comparavel.
            verdict = "indeterminate"
        elif insufficient_evidence and canvas_px == 0:
            # Sem estado confirmado nao ha deslocamento anterior a preservar.
            verdict = "landing_default"
        elif not all_page_scores_available or not all_camera_scores_available:
            # A cropped camera map cannot be compared to the page candidate
            # when its full transformed support or visible page support was
            # not measurable.
            verdict = "indeterminate"
        elif insufficient_evidence:
            # Default landing is safe only for an empty canvas. Existing
            # content must not be re-anchored when this pass cannot expose
            # enough ink to compare it.
            verdict = "indeterminate"
        else:
            family = [
                ("landing", mean_page),
                ("spurious", mean_camera),
                ("landing_rotated", mean_cyclic[0]),
                ("landing_rotated", mean_cyclic[1]),
                ("landing_rotated", mean_cyclic[2]),
            ]
            order = sorted(range(len(family)), key=lambda i: family[i][1], reverse=True)
            best, second = order[0], order[1]
            if family[best][1] - family[second][1] <= _ARB_DELTA:
                verdict = "indeterminate"
                if grid_shift[0] or grid_shift[1]:
                    residual_pending = True
                    residual_pending_reason = "anchor_hypotheses_indistinguishable"
            else:
                verdict = family[best][0]
                if verdict == "landing_rotated":
                    rotation_k = best - 1
                if grid_shift[0] or grid_shift[1]:
                    if verdict == "landing":
                        score_with_residual = mean_page
                        score_without_residual = mean_page_without_residual
                    elif verdict == "landing_rotated":
                        cyclic_index = rotation_k - 1
                        score_with_residual = mean_cyclic[cyclic_index]
                        score_without_residual = mean_cyclic_without_residual[
                            cyclic_index
                        ]
                    else:
                        score_with_residual = score_without_residual = None
                    if (
                        score_with_residual is not None
                        and abs(score_with_residual - score_without_residual)
                        <= _ARB_DELTA
                    ):
                        # A strong landing against the camera can still leave
                        # the old registration residual undecidable. Do not
                        # re-anchor or vote new ink under either mapping until
                        # visible content separates the two residual states.
                        verdict = "indeterminate"
                        residual_pending = True
                        residual_pending_reason = (
                            "residual_hypotheses_indistinguishable"
                        )
        if verdict == "indeterminate" and (grid_shift[0] or grid_shift[1]):
            residual_pending = True
            if residual_pending_reason is None:
                residual_pending_reason = "anchor_hypotheses_indistinguishable"
        event = {
            "generation": generation,
            "pass_index": self._generation_passes,
            "verdict": verdict,
            "action": verdict in ("landing", "landing_default", "landing_rotated"),
            "score_page": mean_page,
            "score_page_without_residual": mean_page_without_residual,
            "score_camera": mean_camera,
            "camera_denominator_mass": mean_camera_denominator_mass,
            "camera_score_available": all_camera_scores_available,
            "page_score_available": all_page_scores_available,
            "scores_cyclic": [mean_camera, *mean_cyclic],
            "scores_cyclic_without_residual": mean_cyclic_without_residual,
            "rotation_k": rotation_k,
            "score_page_roll_compensated": self._anchor_roll != 0,
            "canvas_px": canvas_px,
            "pass_mass_px": mean_mass,
            "displacement_px": displacement,
            "pose_displacement_source": displacement_source,
            "identity_residual_fresh": bool(residual_fresh),
            "streak": len(self._arb_streak_quads),
            "identity_residual_pending": residual_pending,
            "identity_residual_pending_reason": residual_pending_reason,
        }
        self._arb_events.append(event)
        if event["action"]:
            new_anchor = registration._median_anchor(self._arb_streak_quads)
            if rotation_k:
                # Pouso com rotacao: a ancora nova carrega os rotulos
                # fisicos — os cantos da mediana re-rotulados por k,
                # a mesma permutacao que venceu a pontuacao. O estado
                # nao se move (a v3 inteira e troca de ancora).
                new_anchor = np.roll(new_anchor, rotation_k, axis=0)
            elif self._anchor_roll != 0:
                # Pouso comum (ou default) sobre a grade rolada
                # (frente 27, item 3): a mediana posicional do
                # streak rolada pelo roll parado — sem isso o par
                # ancora/roll fica incoerente e o pareamento do
                # voto e do gatilho quebra.
                new_anchor = np.roll(new_anchor, self._anchor_roll, axis=0)
            self._anchor_corners = new_anchor
            # So o pouso rodado muda a rotulacao; nos demais
            # vereditos o roll parado se preserva.
            if verdict == "landing_rotated":
                self._anchor_roll = rotation_k
            self._anchor_bookkeeping(
                self._arb_streak_quads, new_anchor, reg_anchor_median, reg_deadband
            )
            self._arb_actions += 1
        self._arb_close_event()
        return "verdict", event

    def _anchor_follow_step(
        self,
        generation,
        corners,
        shape,
        identity_owned,
        reg_anchor_median,
        reg_deadband,
        step_px,
        arb_event,
        grid_shift=(0, 0),
        residual_fresh=True,
        reg_info=None,
        arbitration_hold_reason=None,
    ):
        """Avanca o evento de divergencia assentada; re-ancora no
        fecho do streak.

        Chamado sob a Condition, DEPOIS
        do passo da arbitragem (arb_event e o veredito dela nesta
        passada, ou None). Devolve (status, evento): "closed" (sem
        evento; a passada vota), "open" (divergencia assentada em
        curso sem re-ancoragem; a passada NAO vota) ou "reanchored"
        (a ancora trocou nesta passada; ela vota, re-registrada na
        ancora nova pelo chamador). O desenho e a precedencia estao
        no comentario das constantes _FOLLOW_* e em
        34-desenho-ancora-movel.md.
        """
        old_anchor = self._anchor_corners
        if identity_owned or corners is None or old_anchor is None:
            self._follow_close_event()
            return "closed", None
        if arbitration_hold_reason is not None:
            # The geometric fallback cannot resolve an ambiguous residual or
            # replace a camera comparison whose transformed support is unknown.
            self._follow_close_event()
            if reg_info is not None:
                reg_info["follow_abstain_reason"] = arbitration_hold_reason
            return "open", None
        # Precedencia da arbitragem (veredito DESTA passada): quem
        # decidiu por conteudo nao e atravessado pela geometria.
        if arb_event is not None and arb_event["verdict"] != "indeterminate":
            if arb_event["action"]:
                # A ancora ja se moveu pelo conteudo (pouso, pouso
                # rodado ou default): a divergencia acabou.
                self._follow_close_event()
                return "closed", None
            # spurious: o conteudo decidiu que a folha nao se
            # moveu — o episodio e do juizo de conteudo e o FOLLOW
            # fica retido (sem fogo, sem abstencao propria) ate ele
            # fechar.
            self._follow_event_open = False
            self._follow_streak = 0
            self._follow_streak_quads = []
            self._follow_hold = True
            return "closed", None
        # O gatilho compara em termos POSICIONAIS, como o da
        # arbitragem: o roll parado nao e deslocamento de pose.
        displacement, displacement_source = self._pose_displacement(
            corners, shape, grid_shift, residual_fresh=residual_fresh
        )
        if reg_info is not None:
            reg_info["follow_pose_displacement_px"] = displacement
            reg_info["follow_pose_displacement_source"] = displacement_source
        if displacement <= _FOLLOW_DISPLACEMENT_PX:
            self._follow_close_event()
            return "closed", None
        if self._follow_hold:
            # Episodio retido pelo veredito spurious: a geometria
            # nao decide nem abstee enquanto o episodio durar.
            return "closed", None
        self._follow_event_open = True
        if step_px is not None and step_px <= _FOLLOW_SETTLE_PX:
            self._follow_streak += 1
            self._follow_streak_quads.append(corners.copy())
        else:
            # A folha ainda esta andando: o evento segue aberto
            # (a passada abstee), o streak recomeca.
            self._follow_streak = 0
            self._follow_streak_quads = []
        if self._follow_streak < _FOLLOW_PASSES:
            return "open", None
        new_anchor = registration._median_anchor(self._follow_streak_quads)
        if self._anchor_roll != 0:
            # Pouso comum sobre a grade rolada: a mediana
            # posicional rolada pelo roll parado (a coerencia
            # ancora/roll da arbitragem); o FOLLOW nunca cria roll.
            new_anchor = np.roll(new_anchor, self._anchor_roll, axis=0)
        closed_arb_event = self._arb_event_open
        if closed_arb_event:
            # A divergencia que a arbitragem julgava foi resolvida
            # pela geometria; ela recomeca contra a ancora nova.
            self._arb_close_event()
        self._anchor_corners = new_anchor
        self._anchor_bookkeeping(
            self._follow_streak_quads, new_anchor, reg_anchor_median, reg_deadband
        )
        self._anchor_reanchors += 1
        event = {
            "generation": generation,
            "pass_index": self._generation_passes,
            "displacement_px": displacement,
            "streak": self._follow_streak,
            "anchor_delta_px": float(
                np.max(np.linalg.norm(new_anchor - old_anchor, axis=1))
            ),
            "closed_arb_event": closed_arb_event,
        }
        self._follow_events.append(event)
        self._follow_close_event()
        return "reanchored", event
