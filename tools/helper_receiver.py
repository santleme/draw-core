"""Receptor do protocolo do ajudante Windows, lado WSL.

O núcleo do protocolo mora em `draw.protocol` (runtime); este módulo
reexporta essa superfície e acrescenta o que é de ferramenta: o
fallback de token para DRAW_HELPER_TOKEN (o runtime não lê o
ambiente — regra do repositório), o --selfcheck e a CLI.

Implementa o contrato de bridge/PROTOCOL.md: um fluxo de registros
(cabeçalho binário de 28 bytes + payload JPEG) que chega por socket
TCP ou por arquivo gravado, sem o consumidor distinguir a fonte.
Cada frame sai como a tupla (frame_id, capture_ts_us, jpeg_bytes);
capture_ts_us é o relógio monotônico do ajudante, ancorado na
captura, e só serve para diferenças dentro da mesma sessão.

Toda conexão começa pelo handshake de token (PROTOCOL.md,
"Authentication handshake"): o token vem de DRAW_HELPER_TOKEN ou de
--token-file e nunca é impresso. O leitor limita payload_len antes
de alocar e decode_jpeg devolve None para JPEG malformado.

Uso (a partir da raiz do repositório):
  python tools/helper_receiver.py --selfcheck
  DRAW_HELPER_TOKEN=... python tools/helper_receiver.py --record \
      --host 192.168.0.10 --out sessao.drawcap [--max-frames 600] [--timeout 30]
  python tools/helper_receiver.py --dump sessao.drawcap
  DRAW_HELPER_TOKEN=... python tools/helper_receiver.py --dump \
      --host 192.168.0.10 --max-frames 5

Como biblioteca:
  from tools.helper_receiver import frames_from_file, frames_from_socket
  for frame_id, capture_ts_us, jpeg in frames_from_file("sessao.drawcap"):
      ...
"""

import argparse
import os
import secrets
import socket
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# draw.protocol reexporta o handshake de draw_dral (pacote draw-dral):
# sem instalar nada, os dois src/ entram no path.
for extra in (ROOT / "src", ROOT / "packages" / "draw-dral" / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import draw.protocol as _protocol  # noqa: E402
from draw.protocol import (  # noqa: E402
    ACCEPT,
    DEFAULT_PORT,
    HANDSHAKE_TIMEOUT_S,
    HEADER,
    HEADER_SIZE,
    HELLO,
    HELLO_MAGIC,
    HELLO_VERSION,
    MAGIC,
    MAX_FRAME_PIXELS,
    MAX_FRAME_SIDE,
    MAX_PAYLOAD_LEN,
    TOKEN_ENV,
    TOKEN_MAX_LEN,
    TOKEN_MIN_LEN,
    VERSION,
    AuthenticationError,
    ProtocolError,
    _connect,
    _iter_records,
    _read_exact,
    client_handshake,
    decode_jpeg,
    frames_from_file,
    is_complete_jpeg,
    pack_record,
    server_handshake,
)

_frames_from_socket = _protocol.frames_from_socket
_record = _protocol.record
_resolve_token = _protocol.resolve_token

# Teto de silêncio por recv pós-handshake, alinhado ao
# HelperCapture(timeout=5.0) e ao HANDSHAKE_TIMEOUT_S: o ajudante
# emite ~30 quadros/s, então 5 s sem nenhum byte = emissor morto.
# Sem o teto, o _connect devolve o socket com settimeout(None) e
# cada recv bloqueia para sempre.
DEFAULT_RECV_TIMEOUT_S = 5.0

__all__ = [
    "ACCEPT",
    "AuthenticationError",
    "DEFAULT_PORT",
    "DEFAULT_RECV_TIMEOUT_S",
    "HANDSHAKE_TIMEOUT_S",
    "HEADER",
    "HEADER_SIZE",
    "HELLO",
    "HELLO_MAGIC",
    "HELLO_VERSION",
    "MAGIC",
    "MAX_FRAME_PIXELS",
    "MAX_FRAME_SIDE",
    "MAX_PAYLOAD_LEN",
    "ProtocolError",
    "TOKEN_ENV",
    "TOKEN_MAX_LEN",
    "TOKEN_MIN_LEN",
    "VERSION",
    "_connect",
    "_iter_records",
    "_read_exact",
    "client_handshake",
    "decode_jpeg",
    "frames_from_file",
    "frames_from_socket",
    "is_complete_jpeg",
    "pack_record",
    "record",
    "resolve_token",
    "server_handshake",
]


def resolve_token(token=None, token_file=None):
    """Token como bytes: argumento, depois --token-file, depois o ambiente."""
    if token is None and token_file is None:
        token = os.environ.get(TOKEN_ENV, "")
    try:
        return _resolve_token(token, token_file)
    except AuthenticationError as error:
        raise AuthenticationError(f"{error}; defina {TOKEN_ENV}") from error


def frames_from_socket(
    host,
    port=DEFAULT_PORT,
    timeout=DEFAULT_RECV_TIMEOUT_S,
    token=None,
    max_payload=MAX_PAYLOAD_LEN,
):
    """Itera (frame_id, capture_ts_us, jpeg_bytes) do ajudante via TCP.

    Autentica com o token (argumento ou DRAW_HELPER_TOKEN) antes de ler.
    `timeout` vale por recv pós-handshake (o connect+handshake fica
    sob o mesmo teto): silêncio maior que isso sobe como TimeoutError.
    timeout=None é o opt-out da biblioteca e devolve o recv
    bloqueante; timeout<=0 é rejeitado antes de qualquer conexão.
    """
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout precisa ser um numero positivo de segundos")
    return _frames_from_socket(
        host,
        port,
        timeout=timeout,
        token=resolve_token(token),
        max_payload=max_payload,
    )


def record(
    host,
    port,
    out_path,
    max_frames=None,
    timeout=DEFAULT_RECV_TIMEOUT_S,
    token=None,
    max_payload=MAX_PAYLOAD_LEN,
):
    """Grava o fluxo do socket byte a byte num arquivo. Retorna n.

    max_frames<=0 é rejeitado antes de qualquer conexão ou resolução
    de token; timeout<=0 vem logo depois, ainda antes de conectar.
    `timeout` vale por recv pós-handshake; None mantém o recv
    bloqueante (opt-out da biblioteca).
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames precisa ser um inteiro positivo")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout precisa ser um numero positivo de segundos")
    return _record(
        host,
        port,
        out_path,
        max_frames=max_frames,
        timeout=timeout,
        token=resolve_token(token),
        max_payload=max_payload,
    )


def _synthetic_frames(n, base_id=1000, base_ts_us=5_000_000):
    """Frames JPEG sintéticos determinísticos para o selfcheck."""
    import cv2
    import numpy as np

    frames = []
    for i in range(n):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :, 0] = (i * 37) % 256
        image[:, :, 1] = np.linspace(0, 255, 320, dtype=np.uint8)[None, :]
        image[:, :, 2] = (255 - i * 11) % 256
        cv2.putText(
            image,
            str(base_id + i),
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("cv2.imencode falhou no selfcheck")
        frames.append((base_id + i, base_ts_us + i * 33_333, encoded.tobytes()))
    return frames


def _check_roundtrip(expected, received, label):
    """Confere ids, timestamps e bytes. Retorna (ok, mensagem)."""
    if len(received) != len(expected):
        return False, (f"{label}: n esperado {len(expected)}, recebido {len(received)}")
    for want, got in zip(expected, received, strict=True):
        if want != got:
            return False, (
                f"{label}: divergencia no frame_id {want[0]}: "
                f"esperado ts={want[1]} {len(want[2])} bytes, "
                f"recebido ts={got[1]} {len(got[2])} bytes"
            )
    return True, (
        f"{label}: n={len(expected)}, ids, timestamps e bytes "
        "conferem em todos os frames"
    )


def _selfcheck():
    n = 64
    expected = _synthetic_frames(n)
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selfcheck.drawcap"
        with open(path, "wb") as handle:
            for frame_id, capture_ts_us, payload in expected:
                handle.write(pack_record(frame_id, capture_ts_us, payload))

        ok, message = _check_roundtrip(
            expected, list(frames_from_file(path)), "arquivo (replay)"
        )
        print(("PASS " if ok else "FAIL ") + message)
        if not ok:
            failures.append(message)

        # Rabo truncado: os n frames bons saem, depois erro de protocolo.
        truncated = Path(tmp) / "truncado.drawcap"
        truncated.write_bytes(path.read_bytes() + b"DRA")
        seen = 0
        try:
            for _frame in frames_from_file(truncated):
                seen += 1
            failures.append(
                f"arquivo truncado: nenhum erro levantado apos os {seen} frames bons"
            )
            print(f"FAIL arquivo truncado: {seen} frames lidos sem erro")
        except ProtocolError:
            print(
                f"PASS arquivo truncado: erro de protocolo apos os "
                f"{seen} frames bons (n={n} esperado)"
            )
            if seen != n:
                failures.append(
                    f"arquivo truncado: {seen} frames antes do erro, esperados {n}"
                )

    # Caminho de socket: servidor loopback efêmero numa thread, com
    # handshake de token. A primeira conexão usa um token errado e
    # precisa ser recusada; as duas seguintes autenticam e recebem
    # os frames — uma pelo frames_from_socket, outra pelo record
    # (socket -> arquivo cru), que o selfcheck também cobre.
    token = secrets.token_urlsafe(32)
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(3)
    port = server.getsockname()[1]

    def serve():
        for _attempt in range(3):
            conn, _addr = server.accept()
            with conn:
                if not server_handshake(conn, token):
                    continue
                for frame_id, capture_ts_us, payload in expected:
                    conn.sendall(pack_record(frame_id, capture_ts_us, payload))
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        list(frames_from_socket("127.0.0.1", port, timeout=10, token="x" * 32))
        failures.append("socket: token errado foi aceito")
        print("FAIL socket: token errado foi aceito")
    except AuthenticationError:
        print("PASS socket: token errado recusado no handshake")
    received = list(frames_from_socket("127.0.0.1", port, timeout=10, token=token))
    ok, message = _check_roundtrip(expected, received, "socket (loopback)")
    print(("PASS " if ok else "FAIL ") + message)
    if not ok:
        failures.append(message)

    # record(): o caminho socket -> arquivo byte a byte. O arquivo
    # gravado é relido por frames_from_file e conferido por inteiro.
    with tempfile.TemporaryDirectory() as tmp:
        rec_path = Path(tmp) / "socket.drawcap"
        try:
            recorded = record("127.0.0.1", port, rec_path, timeout=10, token=token)
        except Exception as error:  # noqa: BLE001 - selfcheck reporta, nao levanta
            failures.append(f"record (socket->arquivo): {error}")
            print(f"FAIL record (socket->arquivo): {error}")
        else:
            ok, message = _check_roundtrip(
                expected, list(frames_from_file(rec_path)), "record (socket->arquivo)"
            )
            print(("PASS " if ok else "FAIL ") + message)
            if not ok:
                failures.append(message)
            if recorded != n:
                failures.append(f"record gravou {recorded} frames, esperados {n}")
                print(f"FAIL record: {recorded} frames gravados, esperados {n}")
    thread.join(timeout=10)

    if failures:
        print(f"SELFCHECK FALHOU: {len(failures)} problema(s).")
        return 1
    print(f"SELFCHECK OK: {4 * n} frames conferidos no total.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Receptor do protocolo do ajudante (bridge/PROTOCOL.md)"
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="gera frames sinteticos, grava, rele por arquivo e por "
        "socket loopback e confere ids, timestamps e bytes",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="grava o fluxo do socket num arquivo (--host e --out)",
    )
    parser.add_argument(
        "--dump",
        nargs="?",
        const="",
        default=None,
        metavar="ARQUIVO",
        help="lista os frames de um arquivo gravado; sem arquivo e com "
        "--host, lista os do socket",
    )
    parser.add_argument("--host", default=None, help="IP do host Windows")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--out", default=None, help="arquivo de saida")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--token-file",
        default=None,
        help=f"arquivo com o token do ajudante (padrao: ${TOKEN_ENV})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_RECV_TIMEOUT_S,
        help="teto em segundos para o silencio de cada recv "
        "pos-handshake; estourado, o emissor e dado como morto "
        "(TimeoutError). Padrao: %(default)s s",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        return _selfcheck()

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames exige um inteiro positivo")
    if args.timeout <= 0:
        parser.error("--timeout exige um numero positivo de segundos")

    if args.record:
        if not args.host or not args.out:
            parser.error("--record exige --host e --out")
        try:
            count = record(
                args.host,
                args.port,
                args.out,
                max_frames=args.max_frames,
                timeout=args.timeout,
                token=resolve_token(token_file=args.token_file),
            )
        except TimeoutError:
            print(
                f"erro: emissor parado: recv silencioso por "
                f"{args.timeout} s; registros completos ja estao em {args.out}",
                file=sys.stderr,
            )
            return 2
        except (OSError, ProtocolError) as error:
            print(f"erro: {error}", file=sys.stderr)
            return 2
        print(f"Gravados {count} frames em {args.out}.")
        return 0

    if args.dump is not None:
        count = 0
        try:
            if args.dump:
                frames = frames_from_file(args.dump)
            elif args.host:
                frames = frames_from_socket(
                    args.host,
                    args.port,
                    timeout=args.timeout,
                    token=resolve_token(token_file=args.token_file),
                )
            else:
                parser.error("--dump sem arquivo exige --host")
            for frame_id, capture_ts_us, payload in frames:
                print(
                    f"frame_id={frame_id} capture_ts_us={capture_ts_us} "
                    f"jpeg={len(payload)} bytes"
                )
                count += 1
                if args.max_frames is not None and count >= args.max_frames:
                    break
        except TimeoutError:
            print(
                f"erro: emissor parado: recv silencioso por {args.timeout} s "
                f"(apos {count} frames listados)",
                file=sys.stderr,
            )
            return 2
        except (OSError, ProtocolError) as error:
            print(f"erro: {error}", file=sys.stderr)
            return 2
        print(f"Total: {count} frames.")
        return 0

    parser.error("nada a fazer: use --selfcheck, --record ou --dump")


if __name__ == "__main__":
    sys.exit(main())
