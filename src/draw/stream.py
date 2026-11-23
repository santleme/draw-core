"""Saída ao vivo da tinta confirmada (bridge/STREAM.md).

A implementação mora no pacote `draw-dral` (`draw_dral.stream`),
extraída para que consumidores (o drawlink) instalem só o fio, sem o
runtime de inferência. Este módulo permanece como shim de
compatibilidade: `from draw.stream import X` continua devolvendo o
mesmo objeto de `draw_dral.stream` — inclusive nomes privados, via
`__getattr__`.

Limite conhecido: o shim preserva a superfície de import e leitura,
não a escrita. Atribuir a `draw.stream.X` (ex.: monkeypatch de estado
interno) altera só este módulo, que a implementação não consulta —
quem precisa disso mira `draw_dral.stream`.
"""

import draw_dral.stream as _impl
from draw_dral.stream import (
    DEFAULT_PORT,
    FULL_FRACTION,
    HEADER,
    HEADER_SIZE,
    KIND_DELTA,
    KIND_FULL,
    MAGIC,
    MAX_CONSUMERS,
    MAX_PAYLOAD_LEN,
    MAX_PENDING_HANDSHAKES,
    MAX_PIXELS,
    MAX_SIDE,
    NO_CAPTURE_TS,
    READABLE_VERSIONS,
    SEND_TIMEOUT_S,
    VERSION,
    InkCanvas,
    InkPublication,
    InkRecord,
    InkStreamPublisher,
    diff_box,
    iter_records,
    pack_delta,
    pack_full,
    records_from_file,
)

__all__ = [
    "DEFAULT_PORT",
    "FULL_FRACTION",
    "HEADER",
    "HEADER_SIZE",
    "InkCanvas",
    "InkPublication",
    "InkRecord",
    "InkStreamPublisher",
    "KIND_DELTA",
    "KIND_FULL",
    "MAGIC",
    "MAX_CONSUMERS",
    "MAX_PAYLOAD_LEN",
    "MAX_PENDING_HANDSHAKES",
    "MAX_PIXELS",
    "MAX_SIDE",
    "NO_CAPTURE_TS",
    "READABLE_VERSIONS",
    "SEND_TIMEOUT_S",
    "VERSION",
    "diff_box",
    "iter_records",
    "pack_delta",
    "pack_full",
    "records_from_file",
]


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
