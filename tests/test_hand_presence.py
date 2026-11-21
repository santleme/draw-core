"""Provas do T4 da pesquisa de performance (2026-11-03):

- HandMasker.presence_signature com carimbo usa o landmarker em
  modo VIDEO (detect_for_video), com carimbos forcados
  estritamente crescentes; sem carimbo, usa o IMAGE; VIDEO
  quebrado cai para o IMAGE.
- HandPresenceWorker roda a amostragem fora do laco, latest-wins,
  e entrega (presenca, moveu) por poll.
- O adaptador repassa o carimbo ao masker.
- A parada da trabalhadora fecha o masker deterministicamente.
"""

import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from draw import hand as hand_module
from draw import hand_presence
from draw.contracts import HandPresenceResult, HandSample


def _landmarks_result():
    hand = [
        SimpleNamespace(x=0.10 + 0.01 * (i % 5), y=0.10 + 0.01 * (i // 5))
        for i in range(21)
    ]
    return SimpleNamespace(hand_landmarks=[hand])


class _StubLandmarker:
    def __init__(self, mode, log):
        self._mode = mode
        self._log = log

    def detect(self, _image):
        self._log.append(("image", self._mode))
        return _landmarks_result()

    def detect_for_video(self, _image, timestamp_ms):
        self._log.append(("video", timestamp_ms))
        return _landmarks_result()


def _install_fake_mediapipe(monkeypatch, fail_video=False):
    log = []

    class _RunningMode:
        IMAGE = "image"
        VIDEO = "video"

    def _create(options):
        mode = options.running_mode
        if fail_video and mode == _RunningMode.VIDEO:
            raise RuntimeError("sem video neste stub")
        return _StubLandmarker(mode, log)

    vision = SimpleNamespace(
        RunningMode=_RunningMode,
        HandLandmarkerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        HandLandmarker=SimpleNamespace(create_from_options=_create),
    )
    fake = types.ModuleType("mediapipe")
    fake.tasks = SimpleNamespace(
        BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        vision=vision,
    )
    fake.Image = lambda **kwargs: SimpleNamespace(**kwargs)
    fake.ImageFormat = SimpleNamespace(SRGB="srgb")
    monkeypatch.setitem(sys.modules, "mediapipe", fake)
    return log


def _masker(tmp_path):
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"stub")
    return hand_module.HandMasker(str(model), verify=None)


def test_presence_signature_uses_video_with_monotonic_ts(monkeypatch, tmp_path):
    log = _install_fake_mediapipe(monkeypatch)
    masker = _masker(tmp_path)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    signature = masker.presence_signature(frame, timestamp_ms=1000)
    assert signature is not None and signature.area > 0
    masker.presence_signature(frame, timestamp_ms=900)
    masker.presence_signature(frame)
    video_ts = [entry[1] for entry in log if entry[0] == "video"]
    assert video_ts == [1000, 1001], (
        f"carimbos do VIDEO deviam ser forcados crescentes: {log}"
    )
    assert ("image", "image") in log, "sem carimbo devia usar IMAGE"


def test_presence_signature_video_failure_falls_back_to_image(monkeypatch, tmp_path):
    log = _install_fake_mediapipe(monkeypatch, fail_video=True)
    masker = _masker(tmp_path)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    signature = masker.presence_signature(frame, timestamp_ms=1000)
    assert signature is not None and signature.area > 0
    assert ("image", "image") in log
    assert not any(entry[0] == "video" for entry in log)


class _ScriptedPresence:
    def __init__(self):
        self.calls = []

    def sample(self, frame, timestamp_ms=None, *, source_id=0):
        self.calls.append((frame.shape, timestamp_ms, source_id))
        return HandSample(True, len(self.calls) > 1)


@pytest.mark.slow
def test_hand_presence_worker_delivers_latest_result():
    presence = _ScriptedPresence()
    worker = hand_presence.HandPresenceWorker(presence).start()
    try:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        assert worker.poll(0) is None
        worker.submit(frame, 1.234, source_id=7)
        deadline = time.monotonic() + 2.0
        result = None
        while time.monotonic() < deadline:
            result = worker.poll(0)
            if result is not None:
                break
            time.sleep(0.005)
        assert result is not None
        assert result == HandPresenceResult(1, 1.234, 7, HandSample(True, False))
        assert presence.calls == [((10, 10, 3), 1234.0, 7)]
        assert worker.poll(result.sequence) is None
        assert worker.error is None
    finally:
        worker.stop()


class _TimestampedMasker:
    def __init__(self):
        self.seen = []

    def presence_signature(self, _frame, timestamp_ms=None):
        self.seen.append(timestamp_ms)
        return (500.0, 50.0, 60.0)


def test_adapter_forwards_timestamp():
    masker = _TimestampedMasker()
    adapter = hand_presence.MediaPipeHandPresence(masker)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    sample = adapter.sample(frame, timestamp_ms=777)
    assert sample.present is True
    assert masker.seen == [777]


class _ClosablePresence(_ScriptedPresence):
    def __init__(self):
        super().__init__()
        self.closed = 0

    def close(self):
        self.closed += 1


@pytest.mark.slow
def test_worker_stop_closes_the_presence_masker():
    presence = _ClosablePresence()
    worker = hand_presence.HandPresenceWorker(presence).start()
    worker.submit(np.zeros((10, 10, 3), dtype=np.uint8), 1.0, source_id=0)
    deadline = time.monotonic() + 2.0
    while worker.poll(0) is None and time.monotonic() < deadline:
        time.sleep(0.005)
    worker.stop()
    assert presence.closed == 1
    # stop() repetido e inofensivo: a idempotencia real mora no
    # HandMasker.close (ver test_hand.py).
    worker.stop()
    assert presence.closed == 2


def test_adapter_close_delegates_to_the_masker():
    class _Masker:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    masker = _Masker()
    adapter = hand_presence.MediaPipeHandPresence(masker)
    adapter.close()
    adapter.close()
    assert masker.closed == 2  # a idempotencia mora no masker


def test_source_change_resets_motion_signature():
    adapter = hand_presence.MediaPipeHandPresence(_TimestampedMasker())
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert adapter.sample(frame, 1000, source_id=0) == HandSample(True, True)
    assert adapter.sample(frame, 1100, source_id=0) == HandSample(True, False)
    assert adapter.sample(frame, 1200, source_id=1) == HandSample(True, True)


@pytest.mark.parametrize("missing", [None, (0.0, 0.0, 0.0)])
def test_presence_holds_through_three_missed_samples_then_releases(missing):
    class _FlickeringMasker:
        def __init__(self):
            self.results = [(500.0, 50.0, 60.0), missing, missing, missing, missing]

        def presence_signature(self, _frame, timestamp_ms=None):
            return self.results.pop(0)

    adapter = hand_presence.MediaPipeHandPresence(_FlickeringMasker())
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    assert adapter.sample(frame).present is True
    for _ in range(hand_module._MASK_HOLD_FRAMES):
        sample = adapter.sample(frame)
        assert sample.present is True and sample.moved is False
    expected = None if missing is None else False
    assert adapter.sample(frame).present is expected


def test_source_change_does_not_carry_presence_hysteresis():
    class _FlickeringMasker:
        def __init__(self):
            self.results = [(500.0, 50.0, 60.0), (0.0, 0.0, 0.0)]

        def presence_signature(self, _frame, timestamp_ms=None):
            return self.results.pop(0)

    adapter = hand_presence.MediaPipeHandPresence(_FlickeringMasker())
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert adapter.sample(frame, source_id=0).present is True
    assert adapter.sample(frame, source_id=1).present is False
