"""Testes do comparador de lab/parity_e2e (o laço ponta a ponta em si
precisa do repo legado e dos binários dos modelos)."""

import json
import os

import numpy as np
from lab.parity_e2e import compare


def _rec(i, present=True, corners=None):
    return {"frame": i, "present": present, "corners": corners}


def _mask(px):
    m = np.zeros((10, 10), bool)
    m.flat[:px] = True
    return m


def test_compare_identical():
    q = [[0, 0], [10, 0], [10, 10], [0, 10]]
    recs = [_rec(i, corners=q) for i in range(3)]
    masks = [_mask(20) for _ in range(3)]
    report = compare(recs, masks, recs, masks, iou_min=0.999, corner_tol=0.5)
    assert report["ok"] and report["frames"] == 3
    assert report["divergences"] == []


def test_compare_flags_each_kind():
    new = [_rec(0, corners=[[0, 0], [10, 0], [10, 10], [0, 10]])]
    old = [_rec(0, present=False, corners=None)]
    report = compare(new, [_mask(50)], old, [_mask(50)], iou_min=0.999, corner_tol=0.5)
    (div,) = report["divergences"]
    assert div["present"] == {"new": True, "legacy": False}
    assert "canvas_iou" not in div

    new = [_rec(1, corners=[[0, 0], [10, 0], [10, 10], [0, 10]])]
    old = [_rec(1, corners=[[0, 0], [12, 0], [10, 10], [0, 10]])]
    report = compare(new, [_mask(50)], old, [_mask(50)], iou_min=0.999, corner_tol=0.5)
    assert report["divergences"][0]["corner_max_delta_px"] == 2.0

    report = compare(
        [_rec(2)], [_mask(50)], [_rec(2)], [_mask(20)], iou_min=0.999, corner_tol=0.5
    )
    div = report["divergences"][0]
    assert div["canvas_iou"] == 0.4
    assert div["canvas_px"] == {"new": 50, "legacy": 20}


def test_compare_masks_empty_both_ok():
    report = compare(
        [_rec(0, present=False)],
        [_mask(0)],
        [_rec(0, present=False)],
        [_mask(0)],
        iou_min=0.999,
        corner_tol=0.5,
    )
    assert report["ok"]


def test_npz_keys_numeric_order(tmp_path):
    """arr_10 não pode vir antes de arr_2 — a ordem é numérica."""
    import numpy as np

    masks = [np.full((4, 4), i % 2, bool) for i in range(12)]
    np.savez_compressed(tmp_path / "c.npz", *masks)
    with np.load(tmp_path / "c.npz") as z:
        loaded = [z[k] for k in sorted(z.files, key=lambda k: int(k.split("_")[1]))]
    for i, m in enumerate(loaded):
        assert m[0, 0] == (i % 2)


def test_compare_corpus_vazio_falha():
    from lab.parity_e2e import compare

    report = compare([], [], [], [], iou_min=0.999, corner_tol=0.5)
    assert report["ok"] is False
    assert "frames" in report["error"] or "corpus" in report["error"]


def test_compare_shape_transposta_conta_divergencia():
    from lab.parity_e2e import compare

    new = [{"frame": 0, "present": True, "corners": None}]
    old = [{"frame": 0, "present": True, "corners": None}]
    nm = np.zeros((600, 800), np.uint8)
    om = np.zeros((800, 600), np.uint8)  # mesmo size, shape transposto
    report = compare(new, [nm], old, [om], iou_min=0.999, corner_tol=0.5)
    assert report["ok"] is False
    assert report["divergences"][0]["canvas_shape"]


def test_run_legacy_limpa_env_draw(monkeypatch, tmp_path):
    import subprocess

    from lab import parity_e2e

    monkeypatch.setenv("DRAW_INK_REMOVAL", "0")
    monkeypatch.setenv("DRAW_MODEL_STRICT", "0")
    monkeypatch.setenv("PATH", "/usr/bin")
    seen = {}

    def fake_run(cmd, env=None, **kw):
        seen["env"] = env
        seen["cmd"] = cmd
        (tmp_path / "telemetry.json").write_text(json.dumps({"frames": []}))
        np.savez_compressed(tmp_path / "canvas.npz")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(parity_e2e.subprocess, "run", fake_run)
    parity_e2e.run_legacy(
        tmp_path,
        tmp_path,
        tmp_path,
        frames=1,
        fps=1.0,
        sheet_model="s.onnx",
        ink_model="i.onnx",
        hand_model="h.task",
    )
    assert "DRAW_INK_REMOVAL" not in seen["env"]
    assert "DRAW_MODEL_STRICT" not in seen["env"]
    assert seen["env"]["PYTHONPATH"] == str(tmp_path)
    assert seen["env"]["PATH"] == "/usr/bin"
    assert "--hand-model" in seen["cmd"]
    assert "h.task" in seen["cmd"]


def test_compare_alinha_pelo_frame_id():
    # Mesmas contagens, lacunas diferentes: parear por posição
    # casaria o frame 2 do novo com o frame 1 do legado.
    new = [_rec(0), _rec(2)]
    old = [_rec(0), _rec(1)]
    report = compare(
        new,
        [_mask(0), _mask(0)],
        old,
        [_mask(0), _mask(0)],
        iou_min=0.999,
        corner_tol=0.5,
    )
    assert report["ok"] is False
    assert report["frames"] == 1  # só o frame 0 existiu nos dois
    missing = {(d["frame"], d["frame_missing"]) for d in report["divergences"]}
    assert missing == {(1, "new"), (2, "legacy")}


def test_compare_frame_extra_lado_unico_diverge():
    new = [_rec(0), _rec(1)]
    old = [_rec(0)]
    report = compare(
        new, [_mask(0), _mask(0)], old, [_mask(0)], iou_min=0.999, corner_tol=0.5
    )
    assert report["ok"] is False
    (div,) = report["divergences"]
    assert div == {"frame": 1, "frame_missing": "legacy"}


def _fake_legacy_repo(tmp_path):
    repo = tmp_path / "legado"
    (repo / "draw").mkdir(parents=True)
    (repo / "draw" / "loop.py").write_text("")
    return repo


def test_main_hand_measured_exige_os_dois_lados(monkeypatch, tmp_path):
    from lab import parity_e2e

    repo = _fake_legacy_repo(tmp_path)
    recs = [_rec(0, present=False)]
    masks = [_mask(0)]

    def runs(new_n, old_n):
        monkeypatch.setattr(
            parity_e2e,
            "run_new",
            lambda *a, **k: (recs, masks, new_n),
        )
        monkeypatch.setattr(
            parity_e2e,
            "run_legacy",
            lambda *a, **k: (recs, masks, old_n),
        )

    runs(4, 0)
    out = tmp_path / "um_lado"
    parity_e2e.main(
        ["--corpus", "corpus", "--out", str(out), "--legacy-repo", str(repo)]
    )
    report = json.loads((out / "parity.json").read_text())
    assert report["hand_samples"] == {"new": 4, "legacy": 0}
    assert report["hand_measured"] is False

    runs(4, 2)
    out = tmp_path / "dois_lados"
    parity_e2e.main(
        ["--corpus", "corpus", "--out", str(out), "--legacy-repo", str(repo)]
    )
    report = json.loads((out / "parity.json").read_text())
    assert report["hand_samples"] == {"new": 4, "legacy": 2}
    assert report["hand_measured"] is True


def test_driver_fecha_maskers_no_finally():
    """O import do driver seta as 8 flags validadas no topo do
    módulo — isola o os.environ para não vazar no resto da suíte."""
    import importlib

    saved = dict(os.environ)
    try:
        drv = importlib.import_module("lab.parity_legacy_driver")
    finally:
        os.environ.clear()
        os.environ.update(saved)

    class Lm:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class MaskerInterno:
        def __init__(self):
            self._landmarker = Lm()
            self._video_landmarker = Lm()

    interno = MaskerInterno()
    drv._close_hand_masker(interno)
    assert interno._landmarker.closed == 1
    assert interno._video_landmarker.closed == 1

    class MaskerPublico:
        def __init__(self):
            self.closed = 0
            self._landmarker = Lm()

        def close(self):
            self.closed += 1

    publico = MaskerPublico()
    drv._close_hand_masker(publico)
    assert publico.closed == 1
    assert publico._landmarker.closed == 0  # prefere o método público

    class LmRuim:
        def close(self):
            raise RuntimeError("finalizer ruim")

    class MaskerRuim:
        _landmarker = LmRuim()
        _video_landmarker = None

    drv._close_hand_masker(MaskerRuim())  # engole a exceção
    drv._close_hand_masker(None)


def test_run_new_pins_legacy_tier_a_config(monkeypatch, tmp_path):
    """A régua de paridade contra o SheetInkLoop legado precisa fixar a base
    legada (sem ganho, sem Detect-Wake e com InkConfig()) tanto no Pipeline
    quanto na PageCleanWorker (review #74)."""
    from lab import parity_e2e

    from draw import hand as hand_module
    from draw import page_clean as page_clean_module
    from draw import perception as perception_module
    from draw import pipeline as pipeline_module
    from draw.ink import worker as worker_module
    from draw.ink.config import InkConfig
    from draw.settings import PipelineConfig

    seen = {}

    class _NoModel:
        def __init__(self, *_a, **_kw):
            pass

        def load(self):
            pass

    class _FakeWorker:
        error = None

        def __init__(self, _pc, _hm, *, clock=None, config=None, ink_divnorm=False):
            seen["worker_config"] = config
            seen["worker_ink_divnorm"] = ink_divnorm

        def pump(self):
            return False

    class _FakePipeline:
        class tracker:
            corners = None

        clean_mask = None

        def __init__(self, **kw):
            seen["pipeline_config"] = kw.get("config")

        def step(self, _pkt):
            return None

        def collect_ink(self):
            return None

    monkeypatch.setattr(perception_module, "NeuralSheetDetector", _NoModel)
    monkeypatch.setattr(page_clean_module, "PageCleanInk", _NoModel)
    monkeypatch.setattr(hand_module, "hand_model_available", lambda _m: False)
    monkeypatch.setattr(worker_module, "PageCleanWorker", _FakeWorker)
    monkeypatch.setattr(pipeline_module, "Pipeline", _FakePipeline)
    monkeypatch.setattr(parity_e2e, "iter_frames", lambda _c, _f: iter(()))

    parity_e2e.run_new(
        "corpus",
        tmp_path,
        frames=0,
        fps=30,
        sheet_model="s.onnx",
        ink_model="i.onnx",
        hand_model=None,
        verify=None,
    )
    expected = PipelineConfig(
        ink=InkConfig(paper_guard=False), ink_auto_gain=False, detect_wake=False
    )
    assert seen["pipeline_config"] == expected
    assert seen["worker_config"] == expected.ink


def test_parity_frame_time_matches_legacy_reciprocal_rounding():
    """A 10 FPS o intervalo 8→14 cruza 0,6 s só com a fórmula do legado."""
    from lab.parity_e2e import parity_frame_time

    for fps in (10.0, 15.0, 24.0, 29.97):
        for idx in range(200):
            assert parity_frame_time(idx, fps) == idx * (1.0 / fps)
    assert parity_frame_time(14, 10.0) - parity_frame_time(8, 10.0) >= 0.6
    assert 14 / 10.0 - 8 / 10.0 < 0.6
