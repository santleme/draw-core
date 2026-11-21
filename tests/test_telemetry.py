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

from draw.infra.telemetry import SessionTelemetry


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
