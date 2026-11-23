"""Relatorio de sessao: agregados e escrita do JSON por corrida."""

import json
import threading

import pytest

from draw.infra import session_report as report_module


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _FakeThread:
    def join(self, timeout=None):
        return None


def _make(tmp_path, clock):
    report = report_module.SessionReport(
        tmp_path / "session-reports",
        {"inicio_epoch": 1762214400.0, "fonte": "0"},
        now=clock,
    )
    report._started_at = clock()
    return report


def test_aggregates_fold_samples(tmp_path):
    clock = _Clock()
    report = _make(tmp_path, clock)
    report.update(
        sheet_present=False,
        state="no_sheet",
        worker_mode="idle",
        hand_present=False,
        loop_fps=10.0,
        detect_ms=30.0,
        pass_ms=100.0,
    )
    clock.t += 1.0
    report._sample_once()
    report.update(
        sheet_present=True,
        state="sheet",
        worker_mode="active",
        hand_present=True,
        loop_fps=12.0,
        detect_ms=40.0,
        pass_ms=200.0,
    )
    clock.t += 1.0
    report._sample_once()
    clock.t += 1.0
    report._sample_once()  # sem update novo: vale o retrato anterior
    report.update(
        sheet_present=False,
        state="no_sheet",
        worker_mode="idle",
        hand_present=False,
        loop_fps=8.0,
        detect_ms=20.0,
        pass_ms=50.0,
    )
    clock.t += 1.0
    report._sample_once()

    summary = report.summary()
    assert summary["amostras_s"] == 4
    assert summary["folha_presente_pct"] == 50.0
    assert summary["primeira_folha_s"] == 2.0
    assert summary["quedas_de_folha"] == 1
    assert summary["estados"] == {"no_sheet": 2, "sheet": 2}
    assert summary["modos_worker"] == {"idle": 2, "active": 2}
    assert summary["mao_presente_pct"] == 50.0
    assert summary["loop_fps_medio"] == 10.5
    assert summary["detect_ms_medio"] == 32.5
    assert summary["detect_ms_max"] == 40.0
    assert summary["pass_ms_medio"] == 137.5
    assert summary["duracao_s"] == 4.0


def test_close_writes_report_and_latest(tmp_path):
    clock = _Clock()
    report = _make(tmp_path, clock)
    report.update(sheet_present=True, state="sheet", loop_fps=9.0)
    report._sample_once()
    report._thread = _FakeThread()
    report.close()

    written = json.loads(report.path.read_text(encoding="utf-8"))
    assert written["fonte"] == "0"
    assert written["folha_presente_pct"] == 100.0
    latest = json.loads(
        (report.path.parent / "latest.json").read_text(encoding="utf-8")
    )
    assert latest == written


def test_same_second_sessions_get_distinct_paths(tmp_path):
    clock = _Clock()
    first = _make(tmp_path, clock)
    second = _make(tmp_path, clock)
    assert first.path != second.path
    assert first.path.parent == second.path.parent


def test_writes_leave_no_temporary_files(tmp_path):
    clock = _Clock()
    report = _make(tmp_path, clock)
    report.update(sheet_present=True, state="sheet")
    report._sample_once()
    report._thread = _FakeThread()
    report._write()
    report.close()
    leftovers = [p.name for p in report.path.parent.glob("*.tmp*")]
    assert leftovers == []
    latest = json.loads(
        (report.path.parent / "latest.json").read_text(encoding="utf-8")
    )
    assert latest == json.loads(report.path.read_text(encoding="utf-8"))


def test_failed_write_leaves_no_temporary_files(tmp_path, monkeypatch):
    clock = _Clock()
    report = _make(tmp_path, clock)
    report.update(sheet_present=True, state="sheet")

    def boom(self, target):
        raise OSError("rename forcado para o teste")

    monkeypatch.setattr(report_module.Path, "replace", boom)
    report._write()
    report._write()  # segunda tentativa com tag nova tambem falha
    assert report.error is not None
    leftovers = [p.name for p in report.path.parent.glob("*.tmp*")]
    assert leftovers == []


def test_concurrent_path_readers_publish_one_name(tmp_path, monkeypatch):
    """path e a escrita podem correr juntos: os dois chamadores tem de
    convergir para o mesmo nome publicado (o seq so pode ser consumido
    uma vez por relatorio)."""
    import itertools
    import threading
    import time as time_module

    clock = _Clock()
    report = _make(tmp_path, clock)

    real_seq = itertools.count()
    first_inside = threading.Event()
    release_first = threading.Event()

    class _BlockingSeq:
        def __next__(self):
            n = next(real_seq)
            if n == 0:
                first_inside.set()
                release_first.wait(timeout=5.0)
            return n

    monkeypatch.setattr(report_module, "_REPORT_SEQ", _BlockingSeq())

    seen = {}

    def read_first():
        seen["first"] = report.path

    def read_second():
        first_inside.wait(timeout=5.0)
        seen["second"] = report.path

    t1 = threading.Thread(target=read_first)
    t2 = threading.Thread(target=read_second)
    t1.start()
    t2.start()
    time_module.sleep(0.2)
    release_first.set()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert not t1.is_alive() and not t2.is_alive()
    assert seen["first"] is seen["second"]
    assert seen["first"] == report.path


def test_header_requires_inicio_epoch(tmp_path):
    """Sem o epoch do inicio o nome do arquivo nao sai: o KeyError tem
    de estourar na construcao, nao dentro da thread daemon, onde
    morreria sem deixar rastro."""
    with pytest.raises(KeyError):
        report_module.SessionReport(tmp_path / "session-reports", {"fonte": "0"})


@pytest.mark.parametrize("bad", ["ontem", object()])
def test_header_rejects_non_numeric_inicio_epoch(tmp_path, bad):
    with pytest.raises((TypeError, ValueError)):
        report_module.SessionReport(tmp_path / "session-reports", {"inicio_epoch": bad})


def test_unserializable_field_surfaces_as_error(tmp_path):
    """Um valor que json.dumps recusa (TypeError, nao OSError) parava
    o escritor sem deixar rastro: agora fica visivel em .error."""
    clock = _Clock()
    report = _make(tmp_path, clock)
    report.update(state=object())  # vira chave do histograma no resumo
    report._sample_once()
    report._write()
    assert isinstance(report.error, TypeError)
    assert not report.path.exists()


def test_writer_loop_failure_surfaces_on_error(tmp_path, monkeypatch):
    """Excecao fora do _write (no laco da thread) tambem nao pode
    morrer em silencio."""
    clock = _Clock()
    report = _make(tmp_path, clock)

    def fail(*args, **kwargs):
        raise RuntimeError("wait quebrado")

    monkeypatch.setattr(report._stopped, "wait", fail)
    report._run()  # nao propaga: a excecao vira report.error
    assert isinstance(report.error, RuntimeError)


def test_summary_waits_for_in_flight_sample(tmp_path, monkeypatch):
    """summary() le os agregados sob o mesmo cadeado de _sample_once:
    com uma amostra no ar, o resumo espera em vez de montar um
    retrato pela metade (dicts mudando durante a leitura)."""
    clock = _Clock()
    report = _make(tmp_path, clock)
    report.update(state="sheet", sheet_present=True)

    sample_holding_lock = threading.Event()
    release_sample = threading.Event()
    summary_done = threading.Event()
    blocked_once = []

    def gated_now():
        # Trava uma unica vez, ja dentro do cadeado de _sample_once.
        if not blocked_once:
            blocked_once.append(True)
            sample_holding_lock.set()
            release_sample.wait(timeout=5.0)
        return clock()

    monkeypatch.setattr(report, "_now", gated_now)

    sampler = threading.Thread(target=report._sample_once)
    sampler.start()
    assert sample_holding_lock.wait(timeout=5.0)  # amostra detem o cadeado

    def run_summary():
        report.summary()
        summary_done.set()

    summarizer = threading.Thread(target=run_summary)
    summarizer.start()
    assert not summary_done.wait(timeout=0.5)  # travada no cadeado
    release_sample.set()

    assert summary_done.wait(timeout=5.0)
    sampler.join(timeout=5.0)
    summarizer.join(timeout=5.0)
    assert not sampler.is_alive() and not summarizer.is_alive()

    summary = report.summary()
    assert summary["amostras_s"] == 1
    assert summary["estados"] == {"sheet": 1}
