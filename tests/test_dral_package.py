"""Pacote draw-dral: import direto funciona e o shim draw.* devolve os
mesmos objetos.

O drawlink consome `draw.stream`/`draw.protocol` hoje; depois de
repointado ele importa `draw_dral` sem arrastar o runtime de
inferência. Estes testes travam as duas pontas: a API nova existe
standalone e a superfície antiga continua idêntica.
"""

import os
import socket
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path

import numpy as np

import draw_dral
import draw_dral.stream
import draw_dral.wire
from draw import protocol, stream

TOKEN = "t" * 32


def test_shim_devolve_os_mesmos_objetos():
    for name in (
        "InkRecord",
        "InkCanvas",
        "InkPublication",
        "InkStreamPublisher",
        "iter_records",
        "records_from_file",
        "pack_full",
        "pack_delta",
        "diff_box",
        "DEFAULT_PORT",
        "HEADER",
        "KIND_FULL",
        "KIND_DELTA",
        "MAGIC",
        "VERSION",
    ):
        assert getattr(stream, name) is getattr(draw_dral.stream, name), name
    for name in (
        "ProtocolError",
        "AuthenticationError",
        "resolve_token",
        "client_handshake",
        "server_handshake",
        "_read_exact",
        "ACCEPT",
        "HELLO",
        "TOKEN_ENV",
    ):
        assert getattr(protocol, name) is getattr(draw_dral.wire, name), name
    assert protocol._read_exact is draw_dral.wire._read_exact


def test_shim_privado_acessivel():
    # Consumidores antigos (test_stream, helper_receiver) leem privados.
    assert stream._PRIVATE_FILES == draw_dral.stream._PRIVATE_FILES
    assert protocol._STALL_WINDOWS == 3


def test_roundtrip_full_delta():
    state = {"buf": b"", "pos": 0}

    def read(n):
        chunk = state["buf"][state["pos"] : state["pos"] + n]
        state["pos"] += len(chunk)
        return chunk

    mask = np.zeros((6, 8), np.uint8)
    mask[2:4, 3:5] = 1
    state["buf"] = draw_dral.pack_full(1, 1, 123, mask, (210.0, 297.0))
    records = list(draw_dral.iter_records(read))
    assert len(records) == 1
    canvas = draw_dral.InkCanvas()
    canvas.apply(records[0])
    np.testing.assert_array_equal(canvas.mask, mask)

    mask[4, 6] = 1
    state["buf"] = draw_dral.pack_delta(1, 2, 124, mask, (4, 4, 3, 1))
    state["pos"] = 0
    canvas.apply(list(draw_dral.iter_records(read))[0])
    np.testing.assert_array_equal(canvas.mask, mask)


def test_handshake_roundtrip_socketpair():
    server, client = socket.socketpair()
    try:
        result = {}

        def serve():
            result["ok"] = draw_dral.server_handshake(server, TOKEN)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        draw_dral.client_handshake(client, TOKEN)
        thread.join(timeout=5)
        assert result["ok"] is True
    finally:
        server.close()
        client.close()


def test_pacote_nao_arrasta_runtime_pesado():
    code = (
        "import sys\n"
        "import draw_dral, draw_dral.stream, draw_dral.wire\n"
        "assert 'cv2' not in sys.modules\n"
        "assert 'onnxruntime' not in sys.modules\n"
        "assert 'draw' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_scripts_de_lab_resolvem_dral_sem_instalacao_editavel(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    scripts = {
        "diag_live.py": [
            "--source",
            "not-a-camera-source",
            "--out",
            "{out}",
            "--frames",
            "1",
            "--warmup",
            "0",
            "--spacing",
            "0",
        ],
        "desk_capture.py": [
            "--source",
            "not-a-camera-source",
            "--out",
            "{out}",
            "--max-frames",
            "1",
        ],
        "seg_scene_audit.py": [
            "--source",
            "not-a-camera-source",
            "--model",
            "unused.onnx",
            "--out",
            "{out}",
            "--allow-unverified",
        ],
    }
    child = """
import runpy, sys, types
from pathlib import Path
site_packages, script, output, *args = sys.argv[1:]
sys.path.insert(0, site_packages)  # third-party deps; -S skips editable .pth files
if script.endswith(('diag_live.py', 'seg_scene_audit.py')):
    perception = types.ModuleType('draw.perception')
    class NeuralSheetDetector:
        def __init__(self, *args, **kwargs): pass
        def load(self): pass
    perception.NeuralSheetDetector = NeuralSheetDetector
    sys.modules['draw.perception'] = perception
if script.endswith('diag_live.py'):
    page_clean = types.ModuleType('draw.page_clean')
    class PageCleanInk:
        def __init__(self, *args, threshold=0.5, **kwargs): self.threshold = threshold
        def load(self): pass
    page_clean.PageCleanInk = PageCleanInk
    sys.modules['draw.page_clean'] = page_clean
sys.argv = [script, *(arg.replace('{out}', output) for arg in args)]
try:
    runpy.run_path(script, run_name='__main__')
except (SystemExit, ValueError, RuntimeError):
    pass
assert 'draw.camera' in sys.modules
assert 'draw_dral.wire' in sys.modules
assert Path(sys.modules['draw_dral'].__file__).resolve().is_relative_to(
    Path(script).resolve().parents[1] / 'packages' / 'draw-dral' / 'src'
)
"""
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)
    for name, args in scripts.items():
        output = tmp_path / name.removesuffix(".py")
        command = [
            sys.executable,
            "-S",
            "-c",
            child,
            sysconfig.get_path("purelib"),
            str(repo / "lab" / name),
            str(output),
            *args,
        ]
        subprocess.run(
            command,
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
            env=clean_env,
        )


def test_init_reexporta_superficie():
    for name in (
        "InkCanvas",
        "InkPublication",
        "InkStreamPublisher",
        "iter_records",
        "pack_full",
    ):
        assert getattr(draw_dral, name) is getattr(draw_dral.stream, name), name
    for name in ("client_handshake", "server_handshake", "resolve_token"):
        assert getattr(draw_dral, name) is getattr(draw_dral.wire, name), name


def test_ink_publication_e_protocol_estrutural():
    # publish() aceita qualquer objeto com estes atributos; o dataclass
    # congelado do core (draw.contracts.InkPublication) os tem todos.
    import typing
    from dataclasses import fields

    from draw import contracts

    esperado = {
        "generation",
        "version",
        "confirmed",
        "captured_at",
        "anchor_corners",
        "full",
    }
    assert set(typing.get_type_hints(draw_dral.InkPublication)) == esperado
    assert {f.name for f in fields(contracts.InkPublication)} >= esperado


def test_versao_do_membro_casa_com_o_pino():
    # O contrato documentado e "mesmo commit, mesma versao": o pino
    # exato em draw-core tem de bater com o pyproject do membro e com o
    # __version__ exposto em runtime.
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    core = tomllib.loads((root / "pyproject.toml").read_text())
    member = tomllib.loads(
        (root / "packages" / "draw-dral" / "pyproject.toml").read_text()
    )
    version = member["project"]["version"]
    dep = next(d for d in core["project"]["dependencies"] if d.startswith("draw-dral"))
    assert dep == f"draw-dral=={version}"
    assert draw_dral.__version__ == version
