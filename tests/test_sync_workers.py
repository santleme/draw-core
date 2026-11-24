"""lab/sync_workers.py: build_sync_pipeline (fiação) e step_sync_frame.

O builder é exercido com as peças ``draw.*`` grampeadas por
monkeypatch — os imports dentro do corpo da função existem para que
a interceptação por atributo de módulo funcione (a mesma convenção
do SyncDetector.submit). O spy do bench_ink e os pinos da paridade
dependem disso. Cobre tambem a ordenacao numerica de
``iter_frames`` (Parte 7).
"""

import cv2
import numpy as np
import pytest
from lab.sync_workers import (
    SyncBundle,
    SyncDetector,
    SyncPresence,
    build_sync_pipeline,
    iter_frames,
    step_sync_frame,
)


def _config():
    from draw.settings import PipelineConfig

    return PipelineConfig.legacy(paper_guard=True)


@pytest.fixture
def seen(monkeypatch):
    """Grampeia a construção de cada peça draw.* e registra as chamadas."""
    from draw import hand as hand_module
    from draw import hand_presence as presence_module
    from draw import page_clean as page_clean_module
    from draw import perception as perception_module
    from draw import pipeline as pipeline_module
    from draw.ink import worker as worker_module

    seen = {"calls": [], "masker_available": [], "masker_exists": True}

    class _FakePageClean:
        def __init__(self, model_path=None, *, verify):
            seen["page_clean_ctor"] = (model_path, verify)
            seen["page_clean_obj"] = self

        def load(self):
            seen["calls"].append("page_clean_load")
            return True

    class _FakeSheet:
        def __init__(self, model_path, *, verify):
            seen["sheet_ctor"] = (model_path, verify)
            seen["sheet_obj"] = self

        def load(self):
            seen["calls"].append("sheet_load")
            return True

    class _FakeMasker:
        def __init__(self, model_path=None, *, verify):
            seen["masker_ctor"] = (model_path, verify)

    class _FakeInkWorker:
        error = None

        def __init__(
            self,
            page_clean,
            hand_masker=None,
            *,
            clock=None,
            config=None,
            ink_divnorm=False,
        ):
            seen["worker_ctor"] = {
                "page_clean": page_clean,
                "hand_masker": hand_masker,
                "clock": clock,
                "config": config,
                "ink_divnorm": ink_divnorm,
            }
            seen["worker_obj"] = self

        def pump(self):
            return False

    class _FakePipeline:
        def __init__(self, **kw):
            seen["pipeline_ctor"] = kw
            seen["pipeline_obj"] = self

    def _make_presence(model_path, *, verify):
        seen["presence_ctor"] = (model_path, verify)
        return seen["presence_obj"]

    monkeypatch.setattr(page_clean_module, "PageCleanInk", _FakePageClean)
    monkeypatch.setattr(perception_module, "NeuralSheetDetector", _FakeSheet)
    monkeypatch.setattr(hand_module, "HandMasker", _FakeMasker)
    monkeypatch.setattr(worker_module, "PageCleanWorker", _FakeInkWorker)
    monkeypatch.setattr(pipeline_module, "Pipeline", _FakePipeline)
    monkeypatch.setattr(presence_module, "make_hand_presence", _make_presence)
    monkeypatch.setattr(
        hand_module,
        "hand_model_available",
        lambda m: seen["masker_available"].append(m) or seen["masker_exists"],
    )
    seen["presence_obj"] = object()
    return seen


def test_builder_wires_config_to_worker_and_pipeline(seen):
    """A mesma config alinha Pipeline e PageCleanWorker (o pin do eval)."""
    cfg = _config()
    det_worker = object()
    bundle = build_sync_pipeline(verify=None, config=cfg, detection_worker=det_worker)
    wc = seen["worker_ctor"]
    assert wc["page_clean"] is bundle.page_clean
    assert wc["hand_masker"] is None
    assert wc["clock"] is bundle.clock
    assert wc["config"] is cfg.ink
    assert wc["ink_divnorm"] == cfg.ink_divnorm
    pc = seen["pipeline_ctor"]
    assert pc["detection_worker"] is det_worker
    assert pc["ink_worker"] is bundle.ink_worker
    assert pc["hand_presence_worker"] is None
    assert pc["hand_masker"] is None
    assert pc["clock"] is bundle.clock
    assert pc["config"] is cfg
    assert bundle.pipeline is seen["pipeline_obj"]
    assert bundle.detection_worker is det_worker


def test_builder_passes_ink_model_raw_and_loads_only_what_it_builds(seen):
    """ink_model=None vai CRU ao PageCleanInk (o default é dele);
    page_clean injetado nunca é carregado pelo builder."""
    bundle = build_sync_pipeline(verify=None, config=_config())
    assert seen["page_clean_ctor"] == (None, None)
    assert seen["calls"] == ["page_clean_load"]
    assert bundle.page_clean is seen["page_clean_obj"]

    injected = object()
    bundle = build_sync_pipeline(verify=None, config=_config(), page_clean=injected)
    assert seen["calls"] == ["page_clean_load"]  # nenhum load novo
    assert bundle.page_clean is injected


def test_builder_detection_injected_worker_wins_over_everything(seen):
    """detection_worker pronto não é embrulhado e vence o detector cru."""
    worker, raw = object(), object()
    bundle = build_sync_pipeline(
        verify=None,
        config=_config(),
        detection_worker=worker,
        detector=raw,
    )
    assert bundle.detection_worker is worker


def test_builder_detection_wraps_a_raw_detector(seen):
    raw = object()
    bundle = build_sync_pipeline(
        verify=None,
        config=_config(),
        detector=raw,
        detect_ms=0.0,
        keep_history=True,
    )
    assert isinstance(bundle.detection_worker, SyncDetector)
    assert bundle.detection_worker._detector is raw
    assert bundle.detection_worker._detect_ms == 0.0
    assert bundle.detection_worker._keep_history is True


def test_builder_detection_builds_and_loads_sheet_model(seen):
    bundle = build_sync_pipeline(verify=None, config=_config(), sheet_model="seg.onnx")
    assert seen["sheet_ctor"] == ("seg.onnx", None)
    assert seen["calls"] == ["page_clean_load", "sheet_load"]
    assert isinstance(bundle.detection_worker, SyncDetector)
    assert bundle.detection_worker._detector is seen["sheet_obj"]


def test_builder_sheet_model_loads_even_when_the_worker_wins(seen):
    """Braço oráculo do sim: o binário carrega e verifica (fail-closed)
    mesmo sem alimentar a detecção."""
    worker = object()
    bundle = build_sync_pipeline(
        verify=None,
        config=_config(),
        sheet_model="seg.onnx",
        detection_worker=worker,
    )
    assert seen["sheet_ctor"] == ("seg.onnx", None)
    assert "sheet_load" in seen["calls"]
    assert bundle.detection_worker is worker


def test_builder_detection_falls_back_to_classic(seen):
    """Sem nada da detecção: SyncDetector(None) — o caminho clássico."""
    bundle = build_sync_pipeline(verify=None, config=_config())
    assert isinstance(bundle.detection_worker, SyncDetector)
    assert bundle.detection_worker._detector is None


def test_builder_no_hand_default_creates_no_hand_stack(seen):
    """_NO_HAND (o default): nem a disponibilidade do modelo é consultada."""
    bundle = build_sync_pipeline(verify=None, config=_config())
    assert bundle.hand_masker is None
    assert bundle.presence is None
    assert bundle.presence_worker is None
    assert seen["masker_available"] == []
    assert seen["pipeline_ctor"]["hand_masker"] is None
    assert seen["pipeline_ctor"]["hand_presence_worker"] is None


def test_builder_hand_none_resolves_masker_and_presence_via_default(seen):
    """hand_model=None vai SEM GATE para masker e presença — os dois
    resolvem o default convencional via hand_model_available (o
    comportamento do eval/app; derrubar a presença aqui apagava
    hand_presence_samples dos relatórios)."""
    bundle = build_sync_pipeline(verify=None, config=_config(), hand_model=None)
    assert seen["masker_available"] == [None]
    assert bundle.hand_masker is not None
    assert seen["presence_ctor"] == (None, None)
    assert bundle.presence is seen["presence_obj"]
    assert isinstance(bundle.presence_worker, SyncPresence)
    assert bundle.presence_worker.strict is False
    assert seen["pipeline_ctor"]["hand_masker"] is bundle.hand_masker
    assert seen["pipeline_ctor"]["hand_presence_worker"] is bundle.presence_worker


def test_builder_hand_unavailable_model_drops_the_masker(seen):
    seen["masker_exists"] = False
    bundle = build_sync_pipeline(
        verify=None, config=_config(), hand_model="missing.task"
    )
    assert bundle.hand_masker is None


def test_builder_hand_model_path_builds_the_full_stack(seen):
    verify = object()
    bundle = build_sync_pipeline(
        verify=verify,
        config=_config(),
        hand_model="h.task",
        presence_strict=True,
    )
    assert seen["masker_ctor"] == ("h.task", verify)
    assert seen["presence_ctor"] == ("h.task", verify)
    assert bundle.presence is seen["presence_obj"]
    assert isinstance(bundle.presence_worker, SyncPresence)
    assert bundle.presence_worker.presence is seen["presence_obj"]
    assert bundle.presence_worker.strict is True
    assert seen["pipeline_ctor"]["hand_presence_worker"] is bundle.presence_worker


def test_bundle_close_is_idempotent_and_closes_masker_then_presence():
    closed = []

    class _R:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    bundle = SyncBundle(
        clock=None,
        detection_worker=None,
        presence_worker=None,
        ink_worker=None,
        pipeline=None,
        page_clean=None,
        hand_masker=_R("masker"),
        presence=_R("presence"),
    )
    bundle.close()
    bundle.close()
    assert closed == ["masker", "presence"]


def test_bundle_close_tolerates_missing_hand_stack():
    bundle = SyncBundle(
        clock=None,
        detection_worker=None,
        presence_worker=None,
        ink_worker=None,
        pipeline=None,
        page_clean=None,
        hand_masker=None,
        presence=None,
    )
    bundle.close()
    bundle.close()


# ------------------------------------------------------------- step frame


class _Pipe:
    def __init__(self, outcome=None):
        self.outcome = outcome
        self.steps = 0
        self.collects = 0

    def step(self, packet):
        self.steps += 1
        return self.outcome

    def collect_ink(self):
        self.collects += 1
        return None


class _Worker:
    def __init__(self, pumped, error=None):
        self._pumped = pumped
        self.error = error
        self.pumps = 0

    def pump(self):
        self.pumps += 1
        return self._pumped


def test_step_sync_frame_collects_only_when_pumped():
    pipe = _Pipe()
    assert step_sync_frame(pipe, _Worker(False), object()) is False
    assert pipe.steps == 1 and pipe.collects == 0
    pipe = _Pipe()
    assert step_sync_frame(pipe, _Worker(True), object()) is True
    assert pipe.collects == 1


def test_step_sync_frame_skips_collect_when_the_step_ended():
    """Outcome não-None já encerrou: a colheita extra não roda."""
    pipe = _Pipe(outcome="ink_worker_failed")
    with pytest.raises(RuntimeError, match="encerrou"):
        step_sync_frame(pipe, _Worker(True), object())
    assert pipe.collects == 0


def test_step_sync_frame_raises_the_worker_error():
    error = ValueError("quebrou")
    with pytest.raises(RuntimeError, match="trabalhadora de tinta") as exc:
        step_sync_frame(_Pipe(), _Worker(True, error=error), object())
    assert exc.value.__cause__ is error


def test_iter_frames_ordena_diretorio_por_sufixo_numerico(tmp_path):
    """f2.png (zeros) chega antes de f10.png (255): a ordem de nome
    os invertia e o replay/simulador consumia o quadro 10 no lugar
    do 2."""
    cv2.imwrite(str(tmp_path / "f2.png"), np.zeros((4, 4, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "f10.png"), np.full((4, 4, 3), 255, np.uint8))

    frames = list(iter_frames(tmp_path, limit=10))

    assert [i for i, _frame in frames] == [0, 1]
    assert frames[0][1].max() == 0  # f2.png primeiro
    assert frames[1][1].min() == 255  # f10.png depois
