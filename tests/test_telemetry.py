"""Telemetria de sessão: forma do registro, cadência e rotação.

O contrato é o que a próxima sessão ao vivo precisa: um registro
JSON por intervalo com os campos do retrato, deltas de
contadores calculados pelo escritor, e rotação com teto de
tamanho. Nada aqui abre câmera nem janela.
"""

import json
import threading
import time

import pytest

from draw.infra import telemetry as telemetry_module
from draw.infra.telemetry import SessionTelemetry


class _FakeThread:
    def join(self, timeout=None):
        return None


def _read_records(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.slow
def test_records_carry_snapshot_fields_and_counter_deltas(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    wrote = threading.Event()
    telemetry = SessionTelemetry(path, interval=0.02, _wrote=wrote).start()
    try:
        telemetry.update(
            loop_fps=12.5,
            detect_ms=36.0,
            pass_ms=96.0,
            passes_total=10,
            retracted_px_total=40,
            canvas_px=1234,
            provisional_px=56,
            state="active",
        )
        assert wrote.wait(5.0)  # primeiro registro gravado pelo escritor
        wrote.clear()
        telemetry.update(passes_total=13, retracted_px_total=40)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if path.exists() and any(
                record.get("passes_total") == 13 for record in _read_records(path)
            ):
                break
            wrote.wait(0.5)
            wrote.clear()
    finally:
        telemetry.close()
    records = _read_records(path)
    assert len(records) >= 2
    first = records[0]
    assert first["loop_fps"] == 12.5
    assert first["canvas_px"] == 1234
    assert first["provisional_px"] == 56
    assert first["state"] == "active"
    assert "t_epoch" in first
    second = next(record for record in records if record.get("passes_total") == 13)
    assert second["passes_total_delta"] == 3
    assert second["retracted_px_total_delta"] == 0


@pytest.mark.slow
def test_close_writes_the_final_snapshot(tmp_path):
    """Sair entre batidas não perde os campos do último intervalo."""
    path = tmp_path / "telemetry.jsonl"
    telemetry = SessionTelemetry(path, interval=60.0).start()
    telemetry.update(loop_fps=11.5, coverage_iou_rejects_total=3)
    telemetry.close()
    records = _read_records(path)
    assert records[-1]["loop_fps"] == 11.5
    assert records[-1]["coverage_iou_rejects_total"] == 3


@pytest.mark.slow
def test_writer_rotates_past_the_size_cap(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    telemetry = SessionTelemetry(path, interval=0.02, max_bytes=300).start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            telemetry.update(passes_total=1, blob="x" * 120)
            backup = path.with_name(path.name + ".1")
            if backup.exists():
                break
            time.sleep(0.02)
        assert backup.exists()
    finally:
        telemetry.close()


def test_unserializable_record_surfaces_as_error(tmp_path):
    """Payload que json.dumps recusa (TypeError, nao OSError) parava
    o escritor sem deixar rastro: _write devolve False e a excecao
    fica em .error — e o close() nao mascara o erro original."""
    path = tmp_path / "telemetry.jsonl"
    telemetry = SessionTelemetry(path)
    telemetry.event("ruim", payload=object())
    assert telemetry._write(telemetry._take_events()) is False
    assert isinstance(telemetry.error, TypeError)

    telemetry._thread = _FakeThread()
    telemetry.close()
    assert isinstance(telemetry.error, TypeError)
    assert _read_records(path) == []  # flush pulado: nada regravado


def test_run_loop_failure_surfaces_on_error(tmp_path, monkeypatch):
    """Excecao fora do _write (ex.: montando o retrato) tambem nao
    pode morrer em silencio na thread daemon."""
    telemetry = SessionTelemetry(tmp_path / "telemetry.jsonl")
    telemetry._stopped.set()

    def boom(self):
        raise RuntimeError("retrato quebrado")

    monkeypatch.setattr(SessionTelemetry, "_snapshot", boom)
    telemetry._run()  # nao propaga: a excecao vira telemetry.error
    assert isinstance(telemetry.error, RuntimeError)


def test_event_queue_is_bounded(tmp_path):
    """A fila tem teto: com o escritor morto ou travado, a memoria
    nao cresce sem fim — passado do limite os eventos mais antigos
    caem."""
    telemetry = SessionTelemetry(tmp_path / "telemetry.jsonl")
    limit = telemetry_module._EVENT_QUEUE_LIMIT
    for seq in range(limit + 7):
        telemetry.event("tick", seq=seq)
    records = telemetry._take_events()
    assert len(records) == limit
    assert records[0]["seq"] == 7  # os mais antigos cairam
    assert records[-1]["seq"] == limit + 6


def test_close_flushes_events_pending_when_writer_is_gone(tmp_path):
    """Escritor morto ou preso no join: os eventos enfileirados
    depois da ultima retirada ainda saem no flush final do close()."""
    path = tmp_path / "telemetry.jsonl"
    telemetry = SessionTelemetry(path, interval=60.0)
    telemetry._thread = _FakeThread()  # escritor nunca rodou
    telemetry.event("saida", seq=1)
    telemetry.close()
    assert telemetry.error is None
    assert [record["event"] for record in _read_records(path)] == ["saida"]


@pytest.mark.slow
def test_close_flushes_events_queued_after_writer_stopped(tmp_path):
    """Evento enfileirado depois do epilogo do escritor ainda sai na
    proxima chamada de close(): o flush final cobre o que sobrou."""
    path = tmp_path / "telemetry.jsonl"
    telemetry = SessionTelemetry(path, interval=60.0).start()
    telemetry.close()
    telemetry.event("depois", seq=2)
    telemetry.close()
    events = [record.get("event") for record in _read_records(path)]
    assert "depois" in events
