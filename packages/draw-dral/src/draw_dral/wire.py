"""Primitivas de fio compartilhadas pelos transportes do Draw.

O handshake de autenticação (bridge/PROTOCOL.md, "Authentication
handshake") e o leitor `_read_exact` servem tanto ao fio da câmera
(registros `DRAW`, em `draw.protocol`) quanto ao fio de tinta
(registros `DRAL`, em `draw_dral.stream`).

O token entra por parâmetro ou por arquivo: a única leitura de
ambiente permitida no runtime é DRAW_MODEL_STRICT (regra do
repositório), então o app recebe --helper-token-file e o
tools/helper_receiver.py aceita DRAW_HELPER_TOKEN.
"""

import hmac
import socket
import struct
import time
from pathlib import Path

# Handshake de autenticação (PROTOCOL.md): o cliente envia
# HELLO(magic, versão, len) + token; o ajudante responde ACCEPT ou fecha.
HELLO_MAGIC = b"DRAH"
HELLO_VERSION = 1
HELLO = struct.Struct("<4sHH")
ACCEPT = b"DROK"
TOKEN_ENV = "DRAW_HELPER_TOKEN"
TOKEN_MIN_LEN = 16
TOKEN_MAX_LEN = 256
HANDSHAKE_TIMEOUT_S = 5.0


class ProtocolError(RuntimeError):
    """O fluxo violou a gramática do fio (magic, versão ou truncamento)."""


class AuthenticationError(ProtocolError):
    """Token ausente, inválido ou recusado pelo outro lado."""


def resolve_token(token=None, token_file=None):
    """Token como bytes: argumento ou arquivo (--helper-token-file).

    Sem fallback de ambiente — no runtime a única variável lida é
    DRAW_MODEL_STRICT. O fallback para DRAW_HELPER_TOKEN mora no
    wrapper de ferramentas (tools/helper_receiver.py).
    """
    if token is None and token_file:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise AuthenticationError(
                f"nao consegui ler o arquivo de token {token_file!r}: {error}"
            ) from error
    if token is None:
        token = ""
    if isinstance(token, str):
        token = token.encode("utf-8")
    if not TOKEN_MIN_LEN <= len(token) <= TOKEN_MAX_LEN:
        raise AuthenticationError(
            f"token do ajudante ausente ou fora de {TOKEN_MIN_LEN}..{TOKEN_MAX_LEN} "
            "bytes; use --helper-token-file"
        )
    return token


def client_handshake(sock, token):
    """Lado cliente: envia o token e exige ACCEPT antes de qualquer registro."""
    token = resolve_token(token)
    sock.sendall(HELLO.pack(HELLO_MAGIC, HELLO_VERSION, len(token)) + token)
    try:
        reply = _read_exact(sock.recv, len(ACCEPT))
    except (ProtocolError, OSError) as error:
        raise AuthenticationError(f"ajudante fechou o handshake: {error}") from error
    if reply != ACCEPT:
        raise AuthenticationError("ajudante recusou o token (handshake sem ACCEPT)")


def server_handshake(conn, token, timeout=HANDSHAKE_TIMEOUT_S):
    """Lado servidor: True e ACCEPT enviado se o token confere.

    Espelha bridge/windows_helper.authenticate_client (usado no
    selfcheck e nos testes). Qualquer falha devolve False sem
    responder; quem chamou fecha a conexão.
    """
    expected = resolve_token(token)
    previous = conn.gettimeout()
    conn.settimeout(timeout)
    try:
        hello = _read_exact(conn.recv, HELLO.size)
        if hello is None:
            return False
        magic, version, length = HELLO.unpack(hello)
        if magic != HELLO_MAGIC or version != HELLO_VERSION:
            return False
        if not TOKEN_MIN_LEN <= length <= TOKEN_MAX_LEN:
            return False
        offered = _read_exact(conn.recv, length)
        if offered is None or not hmac.compare_digest(offered, expected):
            return False
        conn.sendall(ACCEPT)
        return True
    except (ProtocolError, OSError):
        return False
    finally:
        conn.settimeout(previous)


def _read_exact(
    read,
    size,
    stall_retries=0,
    deadline=None,
    clock=time.monotonic,
    recv_timeout=None,
):
    """Lê exatamente size bytes via read(n) -> bytes.

    Retorna None se a fonte acabar limpa antes do primeiro byte.
    Levanta ProtocolError se acabar no meio — ou depois de
    `stall_retries` janelas de socket consecutivas sem nenhum byte,
    ou ao estourar `deadline` (relogio absoluto) com bytes parciais.

    Um socket.timeout não dessincroniza nada: os bytes já consumidos
    ficaram em `chunks` e o restante continua chegando no próximo
    recv. O timeout vale como detector de stall, não como fim de
    fluxo — por isso a tolerância é contada aqui, dentro da leitura,
    onde o fluxo ainda está íntegro (depois que a exceção sobe, o
    iterador morre e qualquer retry externo lê lixo).

    `deadline` e o prazo ABSOLUTO do registro: o timeout do socket e
    por recv, entao um frame que chega pingando byte a byte passava
    por qualquer janela — so um prazo total limita a montagem. Com
    `recv_timeout` (ex.: sock.settimeout) o bloqueio do recv e
    clampeado ao orcamento restante: um frame nunca completa depois
    do prazo.
    """
    chunks = []
    remaining = size
    stalls = 0
    while remaining > 0:
        if deadline is not None:
            budget = deadline - clock()
            if budget <= 0:
                raise ProtocolError(
                    f"registro estourou o prazo com {len(chunks)} "
                    f"chunks lidos (faltaram {remaining} de {size} bytes)"
                )
            if recv_timeout is not None:
                recv_timeout(budget)
        try:
            chunk = read(remaining)
        except (TimeoutError, socket.timeout) as error:
            stalls += 1
            if stalls <= stall_retries:
                continue
            # Sem orcamento de stall o timeout sobe intacto — quem
            # pega TimeoutError/OSError (frames_from_socket, record)
            # continua funcionando. Com orcamento esgotado o desfecho
            # e de protocolo: o emissor esta parado.
            if not stall_retries:
                raise
            raise ProtocolError(
                f"emissor parado: {stalls} janelas de socket vazias "
                f"seguidas (faltaram {remaining} de {size} bytes)"
            ) from error
        if not chunk:
            if not chunks:
                return None
            raise ProtocolError(f"fluxo truncado: faltaram {remaining} de {size} bytes")
        stalls = 0
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
