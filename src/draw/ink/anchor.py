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

    def _pose_displacement(self, corners, shape):
        """Deslocamento posicional do quad contra a ancora, em px da grade."""
        trigger_anchor = (
            self._anchor_corners
            if self._anchor_roll == 0
            else np.roll(self._anchor_corners, -self._anchor_roll, axis=0)
        )
        matrix = registration._registration_matrix(trigger_anchor, corners, shape)
        return _matrix_grid_displacement(matrix, shape)

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
        # O gatilho compara em termos POSICIONAIS: uma ancora que ja
        # carrega rotulos rolados (pouso rodado anterior) nao pode
        # fazer toda passada seguinte parecer deslocada pela
        # permutacao — o deslocamento que abre evento e o da pose,
        # nao o da rotulacao.
        displacement = self._pose_displacement(corners, shape)
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
        raw_total = float(np.sum(raw_prob))
        registered_total = float(np.sum(registered_prob))
        if snapshot is None or raw_total <= 0.0 or registered_total <= 0.0:
            score_page = 0.0
            score_camera = 0.0
            cyclic = (0.0, 0.0, 0.0)
        else:
            # Com roll parado (frente 27): a evidencia da pagina e o
            # S_roll da familia ciclica, reclassificado — o warp
            # entregue ja esta em coordenadas da pagina e a unica
            # coisa errada nele e a rotacao parada (permutacao
            # pura), entao a hipotese-pagina le a prob des-rotada
            # pelo roll corrente. Medido na era 3 capturada: 0,193
            # contra 0,002 do Sc. Os ciclicos de rotacao NOVA
            # seguem zerados (segunda rotacao = abstencao).
            page_prob = (
                raw_prob
                if self._anchor_roll == 0
                else registration._permute_prob(raw_prob, -self._anchor_roll)
            )
            score_page = float(np.sum(page_prob * snapshot) / raw_total)
            score_camera = float(np.sum(registered_prob * snapshot) / registered_total)
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
            else:
                cyclic = tuple(
                    float(
                        np.sum(registration._permute_prob(raw_prob, -k) * snapshot)
                        / raw_total
                    )
                    for k in (1, 2, 3)
                )
        pass_mass = int(np.count_nonzero(raw_prob > 0.5))
        self._arb_scores.append(
            (score_page, score_camera, cyclic[0], cyclic[1], cyclic[2], pass_mass)
        )
        if len(self._arb_streak_quads) < _ANCHOR_MIGRATION_PASSES:
            return "open", None
        mean_page = float(np.mean([s[0] for s in self._arb_scores]))
        mean_camera = float(np.mean([s[1] for s in self._arb_scores]))
        mean_cyclic = [
            float(np.mean([s[2 + k] for s in self._arb_scores])) for k in range(3)
        ]
        mean_mass = float(np.mean([s[5] for s in self._arb_scores]))
        canvas_px = (
            0
            if self._arb_snapshot is None
            else int(np.count_nonzero(self._arb_snapshot))
        )
        rotation_k = 0
        if canvas_px < _ARB_MIN_CANVAS_PX or mean_mass < _ARB_MIN_PASS_PX:
            # Sem massa para julgar nao ha copia a fabricar nem a
            # proteger: trocar a ancora e custo zero nas duas
            # hipoteses.
            verdict = "landing_default"
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
            else:
                verdict = family[best][0]
                if verdict == "landing_rotated":
                    rotation_k = best - 1
        event = {
            "generation": generation,
            "pass_index": self._generation_passes,
            "verdict": verdict,
            "action": verdict in ("landing", "landing_default", "landing_rotated"),
            "score_page": mean_page,
            "score_camera": mean_camera,
            "scores_cyclic": [mean_camera, *mean_cyclic],
            "rotation_k": rotation_k,
            "score_page_roll_compensated": self._anchor_roll != 0,
            "canvas_px": canvas_px,
            "pass_mass_px": mean_mass,
            "displacement_px": displacement,
            "streak": len(self._arb_streak_quads),
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
        displacement = self._pose_displacement(corners, shape)
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
