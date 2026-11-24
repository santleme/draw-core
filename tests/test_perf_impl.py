"""Prova T10 da pesquisa de performance (2026-11-03).

O acumulador em camadas com buffers reusados (sem np.stack da
janela) produz, passada a passada, exatamente o mesmo canvas que a
implementacao de referencia empilhada abaixo.
"""

from collections import deque

import numpy as np

from draw.ink import accumulator as accumulator_mod
from draw.ink import identity as identity_mod
from draw.ink import votes as votes_mod

# --- T10: referencia empilhada do voto (a implementacao anterior) ---


class _ReferenceAccumulator:
    def __init__(self, threshold=0.7):
        self.threshold = float(threshold)
        self._window = deque(maxlen=votes_mod._INK_VOTE_WINDOW)
        self._confirmed = None
        self._provisional = None
        self._provisional_ever = None
        self._retracted_ever = None
        self._low_overlap_streak = 0
        self._turn_fired = False

    def add(self, prob):
        prob = np.asarray(prob, dtype=np.float32)
        if self._confirmed is None or self._confirmed.shape != prob.shape:
            self._confirmed = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional_ever = np.zeros(prob.shape, dtype=bool)
            self._retracted_ever = np.zeros(prob.shape, dtype=bool)
            self._window.clear()
            self._low_overlap_streak = 0
            self._turn_fired = False
        confirmed_before = self._confirmed > 0
        canvas_before = int(np.count_nonzero(self._confirmed))
        mask_now = prob > self.threshold
        turn = False
        if (
            canvas_before >= identity_mod._INK_TURN_MIN_CANVAS_PX
            and not self._turn_fired
        ):
            covered = int(np.count_nonzero(mask_now & confirmed_before))
            if covered <= identity_mod._INK_TURN_MAX_OVERLAP * canvas_before:
                self._low_overlap_streak += 1
                if self._low_overlap_streak >= identity_mod._INK_TURN_PASSES:
                    turn = True
                    self._turn_fired = True
            else:
                self._low_overlap_streak = 0
        self._window.append(prob.copy())
        stack = np.stack(list(self._window))
        score = np.sum(stack, axis=0, dtype=np.float32)
        hits = np.count_nonzero(stack >= votes_mod._INK_VOTE_HIT, axis=0)
        confirmed_rule = (score >= votes_mod._INK_VOTE_SCORE_STRONG) | (
            (hits >= votes_mod._INK_VOTE_MIN_HITS)
            & (score >= votes_mod._INK_VOTE_SCORE)
        )
        if np.any(confirmed_rule):
            import cv2

            seen_as_ink = np.max(stack, axis=0) > self.threshold
            anchor = seen_as_ink | confirmed_before
            near_anchor = (
                cv2.dilate(
                    anchor.astype(np.uint8),
                    votes_mod._INK_FRONTIER_KERNEL,
                )
                > 0
            )
            confirmed_rule &= seen_as_ink | ~near_anchor
        new_confirmed = confirmed_rule & ~confirmed_before
        new_confirmed_px = int(np.count_nonzero(new_confirmed))
        if new_confirmed_px:
            self._confirmed[new_confirmed] = 255
        confirmed_now = self._confirmed > 0
        seen_in_window = np.max(stack, axis=0) > self.threshold
        provisional_now = seen_in_window & ~confirmed_now
        previous_provisional = self._provisional > 0
        retracted = previous_provisional & ~provisional_now & ~confirmed_now
        if np.any(retracted):
            self._retracted_ever |= retracted
        self._provisional_ever |= provisional_now
        provisional_changed = bool(np.any(provisional_now != previous_provisional))
        self._provisional = np.where(provisional_now, 255, 0).astype(np.uint8)
        return {
            "confirmed": self._confirmed.copy(),
            "provisional": self._provisional.copy(),
            "confirmed_changed": new_confirmed_px > 0,
            "provisional_changed": provisional_changed,
            "new_confirmed_px": new_confirmed_px,
            "confirmed_px": int(np.count_nonzero(self._confirmed)),
            "provisional_px": int(np.count_nonzero(self._provisional)),
            "provisional_ever_px": int(np.count_nonzero(self._provisional_ever)),
            "retracted_px_total": int(np.count_nonzero(self._retracted_ever)),
            "turn": turn,
        }


def test_accumulator_buffered_matches_stacked_reference():
    rng = np.random.default_rng(20261103)
    shape = (60, 80)
    real = accumulator_mod.LayeredInkAccumulator()
    reference = _ReferenceAccumulator()
    # Probabilidades com massa nos pontos de decisao do voto:
    # abaixo do hit, entre hit e limiar, acima do limiar.
    choices = np.array(
        [0.0, 0.3, 0.49, 0.5, 0.55, 0.65, 0.69, 0.71, 0.8, 0.95],
        dtype=np.float32,
    )
    for step in range(40):
        prob = choices[rng.integers(0, len(choices), size=shape)]
        # Um traco estavel para o voto fechar de verdade.
        prob[10:30, 20:24] = 0.8
        got = real.add(prob)
        want = reference.add(prob)
        for key in ("confirmed", "provisional"):
            assert np.array_equal(got[key], want[key]), (step, key)
        for key in (
            "confirmed_changed",
            "provisional_changed",
            "new_confirmed_px",
            "confirmed_px",
            "provisional_px",
            "provisional_ever_px",
            "retracted_px_total",
            "turn",
        ):
            assert got[key] == want[key], (step, key)
