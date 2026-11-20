# Ink stream (v1)

Saída ao vivo da tinta confirmada: o Draw publica o canvas confirmado para
um consumidor externo (o `drawlink`). É a **tomada** que isola a ponte do
laço de captura: se o consumidor cair, travar ou nunca existir, o Draw segue
digitalizando igual. Implementação: `src/draw/stream.py`.

Não confundir com o [PROTOCOL.md](PROTOCOL.md): aquele leva JPEG da câmera
(helper Windows -> WSL). Este leva tinta (Draw -> consumidor).

## Garantias de isolamento

- `publish()` só guarda a última publicação e retorna (latest-wins); a
  codificação e o envio rodam em thread própria.
- Sem consumidor, consumidor lento ou conexão caída: o laço não espera. Um
  envio que passa de 2 s derruba o consumidor.
- Desligado por padrão. Ligar: `--stream-port PORTA --stream-token-file ARQ`
  (sem env `DRAW_*`). `--stream-record ARQ` grava os mesmos bytes do socket.

## Transporte e autenticação

Igual ao PROTOCOL.md: o Draw **escuta** em `127.0.0.1` e o consumidor
**conecta**; a conexão passa pelo mesmo handshake `DRAH`/`DROK` com token
compartilhado (16..256 bytes) antes de qualquer registro. Sem TLS (canal
local). O arquivo gravado não tem handshake.

## Registro

Cabeçalho fixo de **32 bytes**, little-endian, `struct.Struct("<4sHBBIIqHHI")`:

| Offset | Size | Campo | Valor |
|-------:|-----:|-------|-------|
| 0  | 4 | `magic` | `DRAL` |
| 4  | 2 | `version` | 1 |
| 6  | 1 | `kind` | 1 = FULL, 2 = DELTA |
| 7  | 1 | `reserved` | 0 |
| 8  | 4 | `generation` | geração da folha; sobe a cada página nova |
| 12 | 4 | `ink_version` | versão da publicação dentro da geração |
| 16 | 8 | `ts_us` | relógio monotônico do Draw na **publicação** (não é carimbo de captura) |
| 24 | 2 | `width` | largura da grade da página, px |
| 26 | 2 | `height` | altura da grade, px |
| 28 | 4 | `payload_len` | `1..16 MiB` |

- **FULL**: payload = `<ff` (largura_mm, altura_mm da folha) + `zlib(máscara h*w uint8)`.
- **DELTA**: payload = `<HHHH` (x, y, w, h) + `zlib(sub-retângulo h*w)`. Só
  vale sobre um FULL anterior da **mesma geração e mesma forma**; senão é erro
  de protocolo.
- Todo consumidor novo (e todo arquivo gravado) começa por um FULL; geração
  nova também. Versões intermediárias podem ser puladas (latest-wins), e o
  DELTA é calculado contra o último estado enviado àquele destino.
- Publicação idêntica à anterior não gera registro.

O receptor valida magic, versão, tipo, `payload_len`, dimensões (<= 8192 px por
lado, <= 40 MP) **antes** de alocar, e limita a descompressão ao tamanho
declarado (sem bomba de zlib). Qualquer violação é `ProtocolError`, sem
ressincronizar. Fim limpo do arquivo/conexão encerra a iteração; cauda
truncada é erro.

## Consumidor de referência

`draw.stream.iter_records(read)` / `records_from_file(path)` entregam
`InkRecord`; `draw.stream.InkCanvas.apply(record)` mantém a máscara e devolve o
retângulo alterado (útil para vetorizar só a região suja).

## Fora da v1

Provisória (só a confirmada sai), pose/cantos, carimbos de tempo por traço,
posição de ponta de caneta, canal de controle consumidor -> Draw.
