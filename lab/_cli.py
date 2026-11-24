"""Superfície de erro uniforme dos CLIs do lab.

Falhas operacionais (entrada ilegível, modelo ausente, JSON malformado,
erro do OpenCV) viram ``erro: <msg>`` em stderr + código de saída 2 — a
mesma convenção que ``parser.exit(2, ...)`` já dava a eval_capture,
desk_capture e seg_scene_audit. ``SystemExit`` do argparse,
``KeyboardInterrupt`` e qualquer erro fora de ``OPERATIONAL_ERRORS``
propagam intactos (bug de programação continua traceback).
"""

import functools
import sys

import cv2

from lab.corpus.clip import ClipError

# Erros previstos de operação de um CLI do lab: validação de entrada
# (ValueError/KeyError), falha de pipeline (RuntimeError), IO (OSError),
# OpenCV (cv2.error) e clipe/corpus malformado (ClipError).
OPERATIONAL_ERRORS = (
    ClipError,
    ValueError,
    RuntimeError,
    OSError,
    KeyError,
    cv2.error,
)


def main_errors(fn):
    """Decora ``main(argv=None) -> int``: erro operacional vira ``erro:`` + 2.

    Mesmo efeito de envolver o corpo em ``try/except OPERATIONAL_ERRORS``
    com ``parser.exit(2, f"erro: {e}")``, sem reindentar nada — o parser
    não é parâmetro porque ``parser.exit`` só imprime em stderr e levanta
    ``SystemExit``, que é o que o wrapper faz direto.
    """

    @functools.wraps(fn)
    def wrapper(argv=None):
        try:
            return fn(argv)
        except OPERATIONAL_ERRORS as error:
            print(f"erro: {error}", file=sys.stderr)
            sys.exit(2)

    return wrapper
