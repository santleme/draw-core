"""Position-veto recovery must finish before the worker goes idle."""

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import occlusion
from draw.ink.config import InkConfig
from draw.ink.worker import PageCleanWorker


@pytest.mark.parametrize("second_edit_after", [None, 10])
def test_erase_redraw_after_hand_exit_finishes_without_another_wake(second_edit_after):
    shape = (120, 160)
    paper = np.full((*shape, 3), 220, np.uint8)
    visible = np.ones(shape, bool)
    corners = np.array([(0, 0), (160, 0), (160, 120), (0, 120)], np.float32)
    old = (slice(30, 33), slice(50, 68))
    new = (slice(40, 42), slice(50, 68))
    old_second = (slice(30, 33), slice(100, 118))
    new_second = (slice(40, 42), slice(100, 118))
    witness = (slice(80, 92), slice(8, 20))
    initial = paper.copy()
    initial[old] = initial[witness] = 60
    if second_edit_after is not None:
        initial[old_second] = 60
    edited = paper.copy()
    edited[new] = edited[witness] = 60
    if second_edit_after is not None:
        edited[old_second] = 60

    class Model:
        threshold = 0.7
        prob = np.zeros(shape, np.float32)

        def ink_prob(self, _image):
            return self.prob

    model = Model()
    model.prob[old] = model.prob[witness] = 0.95
    if second_edit_after is not None:
        model.prob[old_second] = 0.95
    now = [1000.0]
    worker = PageCleanWorker(
        model, config=InkConfig.production(pass_interval_s=0.01), clock=lambda: now[0]
    )
    generation = worker.new_generation()

    def tick(image, step=2.0):
        now[0] += step
        worker.submit(
            WarpedPage(image, generation, corners, visible, captured_at=now[0])
        )
        return worker.pump()

    # Calibrate on bare paper before learning the original marks.
    for _ in range(4):
        assert tick(paper)
    for _ in range(16):
        assert tick(initial)
    assert worker.poll(generation, 0).confirmed[old].all()
    worker.set_hand_presence(True)
    worker.set_hand_presence(False)
    model.prob = np.zeros(shape, np.float32)
    model.prob[new] = model.prob[witness] = 0.95
    if second_edit_after is not None:
        model.prob[old_second] = 0.95
    # These frames span less than a second in total. Switching to the
    # idle probe would prevent the remaining removal votes from running.
    for index in range(2 * occlusion._MARKS_HOLD_MAX + 12):
        if index == second_edit_after:
            edited[old_second] = 220
            edited[new_second] = 60
            model.prob[old_second] = 0.0
            model.prob[new_second] = 0.95
        tick(edited, step=0.011)
        if worker._paper_exposure.max_marks_recovery_age or np.any(
            worker._accumulator._removal_streak
        ):
            assert worker._pass_interval() == 0.01
    snapshot = worker.poll(generation, 0)
    assert not snapshot.confirmed[old].any(), "rest interrupted the position veto"
    assert snapshot.confirmed[new].all()
    assert snapshot.confirmed[witness].all()
    if second_edit_after is not None:
        assert not snapshot.confirmed[old_second].any()
        assert snapshot.confirmed[new_second].all()
    for _ in range(4):
        tick(edited)
    assert worker.stats()["resting"]
