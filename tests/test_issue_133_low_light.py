"""Replay sintético para estabilidade de tinta em baixa luz (#133)."""

import numpy as np
import pytest

from draw import exposure
from draw.ink.accumulator import LayeredInkAccumulator
from draw.ink.config import InkConfig

H, W = 80, 120
SHADOW = (slice(10, 28), slice(40, 80))
STROKE = (slice(40, 44), slice(20, 90))
STROKE_ANCHOR = (slice(40, 44), slice(20, 25))
HAND_MARGIN = (slice(8, 72), slice(104, 120))


def _low_light_page(delta, hand_at_margin):
    gradient = np.linspace(42, 76, W, dtype=np.float32)
    page = np.repeat(gradient[None, :], H, axis=0)
    page = np.repeat(page[..., None], 3, axis=2)
    page[SHADOW] -= 10.0
    page[STROKE] = 28.0
    if hand_at_margin:
        page[HAND_MARGIN] = 235.0
    return np.clip(page + delta, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("hand_at_margin", [False, True])
def test_low_light_shadow_replay_keeps_weak_stroke_without_shadow_flicker(
    hand_at_margin,
):
    smoother = exposure.PaperGainSmoother()
    accumulator = LayeredInkAccumulator(config=InkConfig.production())
    visible = np.ones((H, W), dtype=bool)
    if hand_at_margin:
        visible[HAND_MARGIN] = False

    target_gains = []
    stable_gains = []
    outputs = []
    shadow_reference = None
    stroke_reference = None
    # Pequena oscilação temporal de exposição sobre gradiente e sombra
    # fixos; a mão branca ocupa a margem, mas não participa do p90.
    for index, delta in enumerate((0, 1, -1, 1, 0, -1, 0)):
        page = _low_light_page(delta, hand_at_margin)
        target = exposure.paper_gain(page, visible)
        gain = smoother.update(target, index / 10.0)
        target_gains.append(target)
        stable_gains.append(gain)
        ink_input = exposure.apply_gain(page, gain)
        shadow_level = float(ink_input[SHADOW].mean())
        stroke_level = float(ink_input[STROKE].mean())
        if shadow_reference is None:
            shadow_reference = shadow_level
            stroke_reference = stroke_level

        # Proxy determinístico do page-clean construído a partir dos pixels
        # efetivamente entregues à inferência, incluindo o ganho aplicado.
        prob = np.zeros((H, W), dtype=np.float32)
        prob[SHADOW] = np.clip(0.84 + (shadow_level - shadow_reference) * 0.02, 0, 1)
        prob[STROKE] = np.clip(0.75 + (stroke_reference - stroke_level) * 0.01, 0, 1)
        prob[STROKE_ANCHOR] = 0.96
        outputs.append(accumulator.add(prob, visible))

    assert max(stable_gains) - min(stable_gains) < max(target_gains) - min(target_gains)
    for output in outputs:
        assert not output["confirmed"][SHADOW].any()
        assert not output["provisional"][SHADOW].any()
    assert outputs[0]["provisional"][STROKE].all()
    assert outputs[2]["confirmed"][STROKE].all()
    if hand_at_margin:
        assert not outputs[-1]["confirmed"][HAND_MARGIN].any()
        assert not outputs[-1]["provisional"][HAND_MARGIN].any()
