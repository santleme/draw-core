"""HandMasker sobre um mediapipe falso (sys.modules).

O mediapipe e extra opcional: importar draw.hand nunca depende
dele, e toda falha (pacote, arquivo, criacao) degrada para "sem
mascara". O landmarker fecha deterministicamente por close() —
idempotente e via context manager —, nunca por __del__.
"""

import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest

from draw import hand as hand_module
from draw.contracts import FramePacket


def _landmarks_result(hand_count=1):
    hands = []
    for hand_index in range(hand_count):
        hand = [
            SimpleNamespace(
                x=0.10 + 0.45 * hand_index + 0.01 * (i % 5),
                y=0.10 + 0.01 * (i // 5),
            )
            for i in range(21)
        ]
        hands.append(hand)
    return SimpleNamespace(hand_landmarks=hands)


class _StubLandmarker:
    def __init__(
        self,
        mode,
        log,
        raise_video_detect=False,
        close_raises=False,
        detect_gate=None,
        hand_count=1,
    ):
        self._mode = mode
        self._log = log
        self._raise_video_detect = raise_video_detect
        self._close_raises = close_raises
        self._detect_gate = detect_gate
        self._hand_count = hand_count
        self.images = []
        self.closed = 0

    def detect(self, image):
        self._log.append(("image", self._mode))
        self.images.append(image.data)
        self._gate()
        return _landmarks_result(self._hand_count)

    def detect_for_video(self, image, timestamp_ms):
        self._log.append(("video", timestamp_ms))
        self.images.append(image.data)
        if self._raise_video_detect:
            raise RuntimeError("detect_for_video falhou neste stub")
        self._gate()
        return _landmarks_result(self._hand_count)

    def _gate(self):
        if self._detect_gate is not None:
            entered, release = self._detect_gate
            entered.set()
            release.wait(10.0)

    def close(self):
        self.closed += 1
        if self._close_raises:
            raise RuntimeError("close falhou neste stub")


def _install_fake_mediapipe(
    monkeypatch,
    fail_video=False,
    raise_video_detect=False,
    close_raises=False,
    detect_gate=None,
    create_gate=None,
    hand_count=1,
):
    log = []
    created = []

    class _RunningMode:
        IMAGE = "image"
        VIDEO = "video"

    def _create(options):
        mode = options.running_mode
        if fail_video and mode == _RunningMode.VIDEO:
            raise RuntimeError("sem video neste stub")
        if create_gate is not None:
            entered, release = create_gate
            entered.set()
            release.wait(10.0)
        landmarker = _StubLandmarker(
            mode,
            log,
            raise_video_detect=raise_video_detect,
            close_raises=close_raises,
            detect_gate=detect_gate,
            hand_count=hand_count,
        )
        landmarker.options = options
        created.append(landmarker)
        return landmarker

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
    return log, created


def _model(tmp_path):
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"stub")
    return str(model)


def test_import_and_mask_without_mediapipe_degrade_once(tmp_path, capsys, monkeypatch):
    # O pacote e extra opcional: importar draw.hand ja aconteceu no
    # topo; sys.modules[...] = None forca o ImportError do uso.
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    masker = hand_module.HandMasker(_model(tmp_path), verify=None)
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    assert masker.mask(frame) is None
    assert masker.mask(frame) is None
    assert capsys.readouterr().out.count("mascara de mao indisponivel") == 1


def test_verify_receives_the_resolved_model_path(tmp_path):
    calls = []
    masker = hand_module.HandMasker(_model(tmp_path), verify=calls.append)
    assert calls == [masker._requested_path]


def test_verify_is_a_required_named_argument(tmp_path):
    # Integridade fail-closed: esquecer o verificador e erro de
    # chamada, nao "desligado por acidente"; o None explicito e o
    # opt-out.
    with pytest.raises(TypeError):
        hand_module.HandMasker(_model(tmp_path))


def test_loading_uses_the_explicit_model_and_image_mode(monkeypatch, tmp_path):
    _log, created = _install_fake_mediapipe(monkeypatch)
    masker = hand_module.HandMasker(_model(tmp_path), verify=None)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    mask = masker.mask(frame)
    assert len(created) == 1
    assert mask is not None and mask.shape == (80, 100)


def test_detected_hand_fills_scaled_hull_and_forearm(monkeypatch, tmp_path):
    _install_fake_mediapipe(monkeypatch)
    masker = hand_module.HandMasker(_model(tmp_path), verify=None)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    mask = masker.mask(frame)
    assert mask is not None
    # Os marcos falsos cobrem ~15x15 px no canto; o casco escalado
    # de 1,3 + trapezio do antebraco + dilatacao de 15 px cobrem
    # bastante mais.
    assert np.count_nonzero(mask) > 15 * 15


def test_close_is_idempotent_and_permanent(monkeypatch, tmp_path):
    _log, created = _install_fake_mediapipe(monkeypatch)
    masker = hand_module.HandMasker(_model(tmp_path), verify=None)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    assert masker.mask(frame) is not None
    assert len(created) == 1
    masker.close()
    masker.close()
    assert created[0].closed == 1
    # Depois do fechamento nao recria: mascara indisponivel.
    assert masker.mask(frame) is None
    assert len(created) == 1


def test_failed_video_inference_closes_the_landmarker(monkeypatch, tmp_path):
    _log, created = _install_fake_mediapipe(monkeypatch, raise_video_detect=True)
    masker = hand_module.HandMasker(_model(tmp_path), verify=None)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    signature = masker.presence_signature(frame, timestamp_ms=1000)
    # O landmarker VIDEO falho fecha antes de ser descartado; a
    # presenca cai para o landmarker IMAGE recem-criado.
    assert created[0].closed == 1
    assert len(created) == 2
    assert signature is not None


def test_failed_video_inference_tolerates_a_broken_close(monkeypatch, tmp_path):
    _install_fake_mediapipe(monkeypatch, raise_video_detect=True, close_raises=True)
    masker = hand_module.HandMasker(_model(tmp_path), verify=None)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    # O close do landmarker falho levanta, mas a degradacao para o
    # IMAGE nao pode ser derrubada por ele.
    signature = masker.presence_signature(frame, timestamp_ms=1000)
    assert signature is not None
    masker.close()


def test_close_also_closes_the_video_landmarker(monkeypatch, tmp_path):
    _log, created = _install_fake_mediapipe(monkeypatch)
    masker = hand_module.HandMasker(_model(tmp_path), verify=None)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    masker.presence_signature(frame, timestamp_ms=1000)
    masker.presence_signature(frame)
    assert len(created) == 2
    masker.close()
    assert all(landmarker.closed == 1 for landmarker in created)


def test_context_manager_closes(monkeypatch, tmp_path):
    _log, created = _install_fake_mediapipe(monkeypatch)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    with hand_module.HandMasker(_model(tmp_path), verify=None) as masker:
        masker.mask(frame)
    assert created[0].closed == 1


def test_hand_model_available_requires_a_real_file(tmp_path):
    model = tmp_path / "hand_landmarker.task"
    assert hand_module.hand_model_available(str(model)) is False
    model.write_bytes(b"stub")
    assert hand_module.hand_model_available(str(model)) is True


def test_no_model_path_and_no_default_means_unavailable(tmp_path, monkeypatch):
    # Sem argumento e sem models/hand_landmarker.task convencional:
    # resolve None e a mascara degrada sem criar nada.
    monkeypatch.setattr(
        hand_module, "_DEFAULT_CANDIDATES", (str(tmp_path / "nada.task"),)
    )
    masker = hand_module.HandMasker(verify=None)
    assert masker.mask(np.zeros((10, 10, 3), dtype=np.uint8)) is None


def test_close_does_not_free_a_landmarker_in_flight(monkeypatch, tmp_path):
    entered = threading.Event()
    release = threading.Event()
    _log, created = _install_fake_mediapipe(monkeypatch, detect_gate=(entered, release))
    monkeypatch.setattr(hand_module, "_CLOSE_LOCK_TIMEOUT_S", 0.2)
    masker = hand_module.HandMasker(_model(tmp_path), verify=None)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    done = threading.Event()

    def _call():
        masker.presence_signature(frame, timestamp_ms=1000)
        done.set()

    thread = threading.Thread(target=_call)
    thread.start()
    assert entered.wait(5.0)
    # A deteccao detem o lock: o close() estoura o timeout,
    # sinaliza _closed e NAO fecha o landmarker em uso.
    masker.close()
    assert created[0].closed == 0
    release.set()
    assert done.wait(5.0)
    thread.join(5.0)
    # Quem estava em voo fecha ao retornar.
    assert created[0].closed == 1
    # E o masker segue indisponivel depois do fechamento.
    assert masker.presence_signature(frame) is None


def test_landmarker_created_after_close_is_disposed(monkeypatch, tmp_path):
    entered = threading.Event()
    release = threading.Event()
    _log, created = _install_fake_mediapipe(monkeypatch, create_gate=(entered, release))
    monkeypatch.setattr(hand_module, "_CLOSE_LOCK_TIMEOUT_S", 0.2)
    masker = hand_module.HandMasker(_model(tmp_path), verify=None)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    done = threading.Event()

    def _call():
        masker.presence_signature(frame, timestamp_ms=1000)
        done.set()

    thread = threading.Thread(target=_call)
    thread.start()
    assert entered.wait(5.0)
    # A criacao preguicosa detem o lock: close() sinaliza _closed
    # e sai sem esperar a criacao terminar.
    masker.close()
    release.set()
    assert done.wait(5.0)
    thread.join(5.0)
    # O landmarker criado depois do _closed fecha na hora e nao
    # se publica; a volta para o IMAGE nem cria outro — _closed
    # ja marcou o masker indisponivel.
    assert len(created) == 1
    assert created[0].closed == 1


def test_missing_model_never_imports_mediapipe(tmp_path, monkeypatch):
    import builtins

    monkeypatch.setattr(
        hand_module, "_DEFAULT_CANDIDATES", (str(tmp_path / "ausente.task"),)
    )
    real_import = builtins.__import__
    attempted = []

    def guarded_import(name, *args, **kwargs):
        if name == "mediapipe" or name.startswith("mediapipe."):
            attempted.append(name)
            raise AssertionError("nao deveria importar mediapipe sem arquivo de modelo")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    masker = hand_module.HandMasker(verify=None)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    assert masker.mask(frame) is None
    assert masker.presence_signature(frame, timestamp_ms=1000) is None
    assert attempted == []


def _bare_masker():
    masker = hand_module.HandMasker.__new__(hand_module.HandMasker)
    masker._lock = threading.RLock()
    masker._held_hands = None
    masker._held_shape = None
    masker._held_misses = 0
    masker.last_detection_held = False
    masker._observed_source_id = None
    masker._last_hand_mask = None
    masker._last_hand_count = 0
    masker._mask_misses = 0
    masker._closed = False
    return masker


def test_dark_frame_is_lifted_and_normal_frame_is_untouched():
    dark = np.full((120, 160, 3), 25, np.uint8)
    lifted = hand_module._lift_dark(dark)
    assert lifted.mean() > 2 * dark.mean()
    assert lifted.dtype == np.uint8 and lifted.shape == dark.shape
    normal = np.full((120, 160, 3), 140, np.uint8)
    assert hand_module._lift_dark(normal) is normal
    black = np.zeros((120, 160, 3), np.uint8)
    assert hand_module._lift_dark(black) is black


def test_lift_dark_cost_stays_small():
    import time

    frame = np.full((720, 1280, 3), 30, np.uint8)
    hand_module._lift_dark(frame)
    start = time.perf_counter()
    for _ in range(10):
        hand_module._lift_dark(frame)
    assert (time.perf_counter() - start) / 10 < 0.02


def test_short_detection_gaps_keep_the_last_hand_then_release():
    masker = _bare_masker()
    frame = np.zeros((120, 160, 3), np.uint8)
    hands = [np.zeros((21, 2), np.float32)]
    assert masker._retain(hands, frame) is hands
    for _ in range(hand_module._HOLD_FRAMES):
        assert masker._retain([], frame) is hands
    assert masker._retain([], frame) == []
    assert masker._retain([], frame) == []


def test_retention_never_crosses_frame_sizes_or_hides_unavailable():
    masker = _bare_masker()
    hands = [np.zeros((21, 2), np.float32)]
    masker._retain(hands, np.zeros((120, 160, 3), np.uint8))
    assert masker._retain([], np.zeros((240, 320, 3), np.uint8)) == []
    masker._retain(hands, np.zeros((120, 160, 3), np.uint8))
    assert masker._retain(None, np.zeros((120, 160, 3), np.uint8)) is None


def test_a_returning_hand_resets_the_hold_budget():
    masker = _bare_masker()
    frame = np.zeros((120, 160, 3), np.uint8)
    hands = [np.zeros((21, 2), np.float32)]
    masker._retain(hands, frame)
    masker._retain([], frame)
    masker._retain([], frame)
    masker._retain(hands, frame)
    for _ in range(hand_module._HOLD_FRAMES):
        assert masker._retain([], frame) is hands


def test_reset_retention_drops_held_hand_even_for_same_size_frames():
    masker = _bare_masker()
    frame = np.zeros((120, 160, 3), np.uint8)
    hands = [np.zeros((21, 2), np.float32)]
    masker._retain(hands, frame)
    masker.reset_retention()
    assert masker._retain([], frame) == []
    assert masker.last_detection_held is False


def test_held_detection_is_flagged_and_cleared():
    masker = _bare_masker()
    masker.last_detection_held = False
    frame = np.zeros((120, 160, 3), np.uint8)
    masker._retain([np.zeros((21, 2), np.float32)], frame)
    masker._retain([], frame)
    assert masker.last_detection_held is True
    masker._retain([np.zeros((21, 2), np.float32)], frame)
    assert masker.last_detection_held is False


def test_low_light_cutoff_uses_luminance_not_channel_mean():
    # BGR (100, 40, 80): media dos canais 73, luminancia BT.601 ~59 < 70.
    frame = np.empty((120, 160, 3), np.uint8)
    frame[:] = (100, 40, 80)
    assert hand_module._lift_dark(frame) is not frame


def test_observe_keeps_image_mask_and_capture_identity(monkeypatch, tmp_path):
    log, _created = _install_fake_mediapipe(monkeypatch)
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    with hand_module.HandMasker(_model(tmp_path), verify=None) as masker:
        expected = masker.mask(frame) > 0
        result = masker.observe(FramePacket(frame, 3.0, 7), source_id=5)
    assert (result.frame_sequence, result.captured_at, result.source_id) == (7, 3.0, 5)
    np.testing.assert_array_equal(result.occlusion_mask, expected)
    assert result.orientation_mask is None
    assert result.present is None
    assert ("video", 3000) in log


def test_dark_scene_keeps_two_hand_detection_enabled(monkeypatch, tmp_path):
    _log, created = _install_fake_mediapipe(monkeypatch, hand_count=2)
    frame = np.full((720, 1280, 3), 24, np.uint8)
    with hand_module.HandMasker(_model(tmp_path), verify=None) as masker:
        mask = masker.mask(frame, timestamp_ms=1200)
        hands = masker._landmarks(frame, timestamp_ms=1300)

    assert created[0].options.num_hands == 2
    assert len(hands) == 2
    assert created[0].images[0].mean() > frame.mean() * 2
    assert np.count_nonzero(mask) > 0


def test_partial_two_hand_detection_retains_missing_hand_mask():
    masker = _bare_masker()
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    left = np.array(
        [[20 + i % 5 * 3, 20 + i // 5 * 3] for i in range(21)], dtype=np.float32
    )
    right = left.copy()
    right[:, 0] += 110
    detections = iter([[left, right], [left]])
    masker._landmarks = lambda *_args, **_kwargs: next(detections)

    both = masker.mask(frame)
    one = masker.mask(frame)
    assert np.count_nonzero(one) >= np.count_nonzero(both) * 0.8


def test_full_miss_grows_the_held_footprint():
    """Falha total do detector: a pegada retida cresce a cada miss —
    a mão segue se movendo sob a leitura antiga e a borda dela não
    pode virar papel à vista antes da liberação (Devin Review do
    PR #144)."""
    masker = _bare_masker()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    hand = np.array(
        [[40 + i % 5 * 3, 40 + i // 5 * 3] for i in range(21)],
        dtype=np.float32,
    )
    detections = iter([[hand]] + [[] for _ in range(hand_module._HOLD_FRAMES + 1)])
    masker._landmarks_locked = lambda *_a, **_k: next(detections)

    detected = masker.mask(frame)
    assert detected is not None and detected.any()
    counts = [
        np.count_nonzero(masker.mask(frame)) for _ in range(hand_module._HOLD_FRAMES)
    ]
    assert counts[0] >= np.count_nonzero(detected)
    assert list(counts) == sorted(counts) and counts[-1] > counts[0]
    released = masker.mask(frame)
    assert released is not None and not released.any()


def test_observe_resets_retention_when_the_source_changes(monkeypatch):
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    masker = _bare_masker()
    masker._lock = threading.RLock()
    masker.last_detection_held = False
    masker._observed_source_id = None
    sequence = iter([[np.zeros((21, 2), np.float32)], [], [], []])
    monkeypatch.setattr(masker, "_landmarks", lambda *_a, **_k: next(sequence))
    # mask() usa _landmarks sem retencao aqui; o que importa e o reset.
    masker._held_hands = [np.zeros((21, 2), np.float32)]
    masker._held_shape = frame.shape[:2]
    masker.observe(FramePacket(frame, 1.0, 1), source_id=1)
    masker._held_hands = [np.zeros((21, 2), np.float32)]
    masker.observe(FramePacket(frame, 2.0, 2), source_id=1)
    assert masker._held_hands is not None  # mesma fonte: mantem
    masker.observe(FramePacket(frame, 3.0, 3), source_id=2)
    assert masker._held_hands is None  # outra fonte: descartada


def test_long_absence_drops_the_held_footprint():
    """Ausência real (hold de marcos expirado) limpa a pegada retida:
    uma detecção parcial futura não a reata sobre a região que a mão
    já deixou (Devin Review do PR #144)."""
    masker = _bare_masker()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    masker._landmarks = lambda *_args, **_kwargs: []
    masker._last_hand_mask = np.full((120, 160), 255, np.uint8)
    masker._last_hand_count = 2
    masker._mask_misses = 1
    empty = masker.mask(frame)
    assert not empty.any()
    assert masker._last_hand_mask is None
    assert masker._last_hand_count == 0
    assert masker._mask_misses == 0
