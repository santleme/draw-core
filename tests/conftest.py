"""Configuração comum da suíte.

A verificação de modelos é estrita por padrão (draw/infra/manifest.py):
um ONNX fora de assets/manifest.json não carrega. As fixtures ONNX da
suíte são sintéticas e nunca serão registradas, então a suíte roda com
o opt-out explícito DRAW_MODEL_STRICT=0. Os testes que fixam o padrão
estrito (tests/test_manifest.py) removem a variável com monkeypatch.
"""

import os

import cv2
import numpy as np
import pytest

from draw.ink import registration as registration_mod
from draw.ink import worker as worker_mod


def pytest_configure():
    if "PYTEST_XDIST_WORKER" in os.environ:
        cv2.setNumThreads(1)


@pytest.fixture(autouse=True)
def _synthetic_models_are_unverified(monkeypatch):
    monkeypatch.setenv("DRAW_MODEL_STRICT", "0")


@pytest.fixture
def fast(monkeypatch):
    """Intervalos da trabalhadora em milissegundos: passadas em tempo real."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.005)
    return monkeypatch


@pytest.fixture
def spy(monkeypatch):
    """Registra (ancora, quad de registro) que cada passada entregou."""
    calls = []
    real = registration_mod._register_ink_probability

    def wrapped(
        prob,
        anchor,
        corners,
        visible=None,
        visible_cam=None,
        pass_corners=None,
        sampled=None,
    ):
        calls.append(
            (
                None if anchor is None else np.array(anchor, copy=True),
                None if corners is None else np.array(corners, copy=True),
            )
        )
        return real(
            prob,
            anchor,
            corners,
            visible,
            visible_cam=visible_cam,
            pass_corners=pass_corners,
            sampled=sampled,
        )

    monkeypatch.setattr(registration_mod, "_register_ink_probability", wrapped)
    return calls
