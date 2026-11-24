"""Contratos de presença de mão: HandSignature e HandSample."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from draw.contracts import HandOcclusion, HandSample, HandSignature


def test_hand_signature_valid_and_frozen():
    signature = HandSignature(area=500.0, centroid_x=50.0, centroid_y=60.0)
    assert signature.area == 500.0
    with pytest.raises(FrozenInstanceError):
        signature.area = 0.0


def test_hand_signature_rejects_non_finite_and_negative():
    for bad in (float("inf"), float("nan"), "muita"):
        with pytest.raises(ValueError):
            HandSignature(area=1.0, centroid_x=bad, centroid_y=0.0)
    with pytest.raises(ValueError):
        HandSignature(area=-1.0, centroid_x=0.0, centroid_y=0.0)
    zero = HandSignature(area=0.0, centroid_x=0.0, centroid_y=0.0)
    assert zero.area == 0.0  # "sem mão" de quem respondeu


def test_hand_sample_valid_and_frozen():
    for present in (True, False, None):
        sample = HandSample(present=present, moved=False)
        assert sample.present is present
    sample = HandSample(present=True, moved=True)
    assert sample.moved is True
    with pytest.raises(FrozenInstanceError):
        sample.moved = False


def test_hand_sample_rejects_bad_fields():
    with pytest.raises(ValueError):
        HandSample(present="talvez", moved=False)
    with pytest.raises(ValueError):
        HandSample(present=True, moved=1)
    with pytest.raises(ValueError):
        HandSample(present=1, moved=False)
    with pytest.raises(ValueError):
        HandSample(present=0, moved=False)
    with pytest.raises(ValueError):
        HandSample(present=np.True_, moved=False)
    with pytest.raises(ValueError):
        HandSample(present=True, moved=np.False_)
    with pytest.raises(ValueError):
        HandOcclusion(4, 1.5, 2, present=1)


def test_hand_occlusion_keeps_empty_distinct_from_missing_and_owns_masks():
    camera_mask = np.zeros((3, 5), dtype=bool)
    observation = HandOcclusion(4, 1.5, 2, camera_mask)
    camera_mask[0, 0] = True

    assert observation.occlusion_mask is not None
    assert not observation.occlusion_mask.any()
    assert observation.orientation_mask is None
    assert not observation.occlusion_mask.flags.writeable
    assert HandOcclusion(4, 1.5, 2).occlusion_mask is None


def test_hand_occlusion_rejects_masks_on_different_camera_grids():
    with pytest.raises(ValueError, match="mesma grade"):
        HandOcclusion(
            4,
            1.5,
            2,
            np.zeros((3, 5), dtype=bool),
            np.zeros((3, 4), dtype=bool),
        )
