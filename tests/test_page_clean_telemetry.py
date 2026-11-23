"""Telemetria por etapa do _PageCleanWorker (U0 do loop v2).

A telemetria precisa de carimbos submit -> inicio da passada ->
fim da passada -> publicacao, sem mudar comportamento. Estes
testes pinam o contrato da telemetria: contadores de
submit/substituicao e duracoes derivadas coerentes por passada.
"""

import threading
import time

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import worker as worker_mod

H, W = 40, 60


class _BlockingInk:
    threshold = 0.7

    def __init__(self):
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def ink_mask(self, _warped):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            self.release.wait(timeout=5.0)
        return np.zeros((H, W), dtype=np.uint8)


@pytest.mark.slow
def test_telemetry_counts_replaced_pending_and_stamps_stages(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    fake = _BlockingInk()
    worker = worker_mod.PageCleanWorker(fake).start()
    try:
        warp = np.zeros((H, W, 3), dtype=np.uint8)
        generation = worker.new_generation()
        worker.submit(WarpedPage(warp, generation))
        assert fake.entered.wait(timeout=5.0)
        # A trabalhadora esta dentro da passada 1: o segundo submit
        # fica pendente e o terceiro o substitui (latest-wins).
        worker.submit(WarpedPage(warp, generation))
        worker.submit(WarpedPage(warp, generation))
        fake.release.set()

        version = 0
        deadline = time.monotonic() + 5.0
        snapshot = None
        while time.monotonic() < deadline:
            got = worker.poll(generation, version)
            if got is not None:
                version, _mask = got.version, got.confirmed
            snapshot = worker.telemetry_snapshot()
            if snapshot["passes"] >= 2:
                break
            time.sleep(0.005)

        assert snapshot["submits"] == 3
        assert snapshot["submits_replaced"] == 1
        assert snapshot["passes"] == 2
        # Reconciliacao com a orquestracao em camadas (PR #17):
        # publishes conta so as passadas que publicam canvas
        # confirmado novo. A mascara deste fake e vazia, entao so
        # a primeira passada publica (a abertura da geracao); a
        # segunda nao confirma nada novo e nao publica. No caminho
        # de uniao antigo, toda passada publicava e o esperado era
        # 2; o resto do contrato (submits, substituicoes, carimbos
        # por passada) segue igual.
        assert snapshot["publishes"] == 1
        assert len(snapshot["records"]) == 2
        for record in snapshot["records"]:
            assert record["generation"] == generation
            assert record["queue_wait_s"] >= 0.0
            assert record["pass_s"] >= 0.0
            assert record["submit_to_publish_s"] >= record["pass_s"]
            assert (
                record["submitted_at"]
                <= record["pass_started_at"]
                <= record["pass_ended_at"]
                <= record["published_at"]
            )
    finally:
        fake.release.set()
        worker.stop()


@pytest.mark.slow
def test_telemetry_snapshot_is_a_copy(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    fake = _BlockingInk()
    fake.release.set()
    worker = worker_mod.PageCleanWorker(fake).start()
    try:
        warp = np.zeros((H, W, 3), dtype=np.uint8)
        generation = worker.new_generation()
        worker.submit(WarpedPage(warp, generation))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if worker.telemetry_snapshot()["passes"] >= 1:
                break
            time.sleep(0.005)
        snapshot = worker.telemetry_snapshot()
        snapshot["records"].clear()
        snapshot["passes"] = 999
        fresh = worker.telemetry_snapshot()
        assert fresh["passes"] == 1
        assert len(fresh["records"]) == 1
    finally:
        worker.stop()


def test_telemetry_pass_s_includes_page_model_and_registration_work(monkeypatch):
    """Issue 8: pass_s inclui o trabalho de registro e PageModel apos a inferencia."""
    from draw.ink import page_model as pm_mod
    from draw.ink.config import InkConfig

    now = [100.0]

    class _InstantInk:
        threshold = 0.7

        def ink_mask(self, _warped):
            now[0] += 0.010
            return np.zeros((H, W), dtype=np.uint8)

    orig_observe = pm_mod.PageModel.observe

    def _timed_observe(self, *args, **kwargs):
        now[0] += 0.045
        return orig_observe(self, *args, **kwargs)

    monkeypatch.setattr(pm_mod.PageModel, "observe", _timed_observe)
    worker = worker_mod.PageCleanWorker(
        _InstantInk(),
        clock=lambda: now[0],
        config=InkConfig(page_model=True),
    )
    generation = worker.new_generation()
    warp = np.full((H, W, 3), 235, dtype=np.uint8)
    worker.submit(WarpedPage(warp, generation))
    assert worker.pump()
    record = worker.telemetry_snapshot()["records"][-1]
    assert record["pass_s"] >= 0.055 - 1e-6
    assert record["pass_s"] == pytest.approx(
        worker.stats()["pass_ms"] / 1000.0, abs=1e-4
    )
    assert (
        record["submitted_at"]
        <= record["pass_started_at"]
        <= record["pass_ended_at"]
        <= record["published_at"]
    )


_PM_STATS_KEYS = (
    "pm_seeded_frac",
    "pm_novel_px",
    "pm_suppressed_px",
    "pm_pending_px",
    "pm_unseeded_px",
)


def test_stats_exposes_page_model_fields_from_the_last_pass():
    """stats() espelha os campos pm_* do registro da ultima passada —
    e o caminho pelo qual o soak 4A/4B ve o PageModel (o app so le
    stats(), nao telemetry_snapshot)."""
    from draw.ink.config import InkConfig

    class _InstantInk:
        threshold = 0.7

        def ink_mask(self, _warped):
            return np.zeros((H, W), dtype=np.uint8)

    worker = worker_mod.PageCleanWorker(
        _InstantInk(),
        config=InkConfig(page_model=True),
    )
    # Sem passada ainda: ausente sai None, como pass_ms.
    stats = worker.stats()
    for key in _PM_STATS_KEYS:
        assert key in stats
        assert stats[key] is None
    generation = worker.new_generation()
    warp = np.full((H, W, 3), 235, dtype=np.uint8)
    worker.submit(WarpedPage(warp, generation))
    assert worker.pump()
    stats = worker.stats()
    assert isinstance(stats["pm_seeded_frac"], float)
    assert 0.0 <= stats["pm_seeded_frac"] <= 1.0
    for key in (
        "pm_novel_px",
        "pm_suppressed_px",
        "pm_pending_px",
        "pm_unseeded_px",
    ):
        assert isinstance(stats[key], int)
        assert stats[key] >= 0
    # Primeira passada: nada semeado ainda, a regiao observada
    # inteira conta como pendente.
    assert stats["pm_seeded_frac"] == 0.0
    assert 0 < stats["pm_pending_px"] <= H * W
    assert stats["pm_unseeded_px"] > 0


def test_stats_page_model_fields_are_none_without_the_model():
    """Com page_model desligado o registro nao tem pm_*: stats() emite
    None em vez de quebrar (mesma regra do pass_ms ausente)."""

    class _InstantInk:
        threshold = 0.7

        def ink_mask(self, _warped):
            return np.zeros((H, W), dtype=np.uint8)

    worker = worker_mod.PageCleanWorker(_InstantInk())
    generation = worker.new_generation()
    warp = np.full((H, W, 3), 235, dtype=np.uint8)
    worker.submit(WarpedPage(warp, generation))
    assert worker.pump()
    stats = worker.stats()
    for key in _PM_STATS_KEYS:
        assert key in stats
        assert stats[key] is None
