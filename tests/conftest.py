"""Configuração comum da suíte.

A verificação de modelos é estrita por padrão (draw/infra/manifest.py):
um ONNX fora de assets/manifest.json não carrega. As fixtures ONNX da
suíte são sintéticas e nunca serão registradas, então a suíte roda com
o opt-out explícito DRAW_MODEL_STRICT=0. Os testes que fixam o padrão
estrito (tests/test_manifest.py) removem a variável com monkeypatch.
"""

import os

import cv2
import pytest


def pytest_configure():
    if "PYTEST_XDIST_WORKER" in os.environ:
        cv2.setNumThreads(1)


@pytest.fixture(autouse=True)
def _synthetic_models_are_unverified(monkeypatch):
    monkeypatch.setenv("DRAW_MODEL_STRICT", "0")
