# Canais de fio do Draw

Mapa dos três protocolos binários que entram ou saem do Draw. O nome de
cada canal é o `magic` que abre o seu cabeçalho de fio.

| Canal | Direção | Transporte | Porta padrão | Ponta remota | Especificação | Implementação |
|---|---|---|---|---|---|---|
| `DRAW` | câmera → Draw (o helper escuta em 127.0.0.1; o Draw, no WSL, conecta) | TCP (`tcp://`, `drawcap://`) ou arquivo `.drawcap` gravado | 8765 | [`bridge/windows_helper.py`](../bridge/windows_helper.py) | [`bridge/PROTOCOL.md`](../bridge/PROTOCOL.md) | [`src/draw/protocol.py`](../src/draw/protocol.py) |
| `DRAL` | Draw → consumidor (o Draw escuta em 127.0.0.1; o consumidor conecta) | TCP ou arquivo (`--stream-record`, modo 0600) | 8766 | [`drawlink`](https://github.com/santleme/drawlink) (externo) | [`bridge/STREAM.md`](../bridge/STREAM.md) | [`packages/draw-dral/src/draw_dral/stream.py`](../packages/draw-dral/src/draw_dral/stream.py) (`draw.stream` é shim) |
| `HVCM` | câmera → Draw | vsock (`hvcam://`, `hvcam+reverse://`) e TCP (`hvcam+tcp://`) | 8765 | `hvcam-host.exe` | repo [santleme/hvcam](https://github.com/santleme/hvcam) (`include/protocol.h`) | [`src/draw/hvcam/`](../src/draw/hvcam/) (`_wire.py` tem magic e constantes; `_capture.py` tem o `HvcamCapture`) |

A porta padrão 8765 aparece duas vezes sem conflito prático: o `DRAW` é
TCP com o helper do Windows e o `HVCM` fala, no transporte primário,
vsock com o `hvcam-host.exe` — sockets e gramáticas diferentes. A
ressalva é o fallback `hvcam+tcp://`, que é TCP de verdade e também
usa 8765 por padrão: ele existe para cenários em que o helper não é o
par (fora do WSL), mas se os dois serviços forem levantados na mesma
máquina é preciso mudar uma das portas.

## Handshake de token compartilhado

`DRAW` e `DRAL` abrem a conexão com o mesmo handshake de token: o
cliente manda `DRAH` (HELLO + token) e o servidor responde `DROK` ou
fecha. A implementação mora em `draw_dral.wire`
([`packages/draw-dral/src/draw_dral/wire.py`](../packages/draw-dral/src/draw_dral/wire.py);
`client_handshake`, `server_handshake`, `resolve_token`) e é
reexportada por `src/draw/protocol.py`: o `draw_dral.stream` a usa
para autenticar o consumidor `DRAL`, e o `tools/helper_receiver.py`
usa cliente e servidor no selfcheck do `DRAW`. O `drawlink` implementa
o lado cliente.

## O canal HVCM é outra gramática

`HVCM` não participa do handshake de token: é o protocolo próprio do
`hvcam-host.exe` (repo santleme/hvcam, `include/protocol.h`) —
cabeçalho de 36 bytes, magic `0x4856434D`, pedido `_HVCAM_REQ = b"Q"`
(constantes em `src/draw/hvcam/_wire.py`). O transporte primário é
vsock (`hvcam://` fala com o host Windows da própria máquina;
`hvcam+reverse://` inverte quem escuta); `hvcam+tcp://host[:porta]` é
o fallback TCP para fora do WSL.
