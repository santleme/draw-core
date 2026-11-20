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


def _landmarks_result():
    hand = [
        SimpleNamespace(x=0.10 + 0.01 * (i % 5), y=0.10 + 0.01 * (i // 5))
        for i in range(21)
    ]
    return SimpleNamespace(hand_landmarks=[hand])


class _StubLandmarker:
    def __init__(
        self,
        mode,
        log,
        raise_video_detect=False,
        close_raises=False,
        detect_gate=None,
    ):
        self._mode = mode
        self._log = log
        self._raise_video_detect = raise_video_detect
        self._close_raises = close_raises
        self._detect_gate = detect_gate
        self.closed = 0

    def detect(self, _image):
        self._log.append(("image", self._mode))
        self._gate()
        return _landmarks_result()

    def detect_for_video(self, _image, timestamp_ms):
        self._log.append(("video", timestamp_ms))
        if self._raise_video_detect:
            raise RuntimeError("detect_for_video falhou neste stub")
        self._gate()
        return _landmarks_result()

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
        )
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
