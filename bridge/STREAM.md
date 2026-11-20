# Ink stream (v1)

Saída ao vivo da tinta confirmada: o Draw publica o canvas confirmado para
um consumidor externo (o `drawlink`). É a **tomada** que isola a ponte do
laço de captura: se o consumidor cair, travar ou nunca existir, o Draw segue
digitalizando igual. Implementação: `src/draw/stream.py`.

Não confundir com o [PROTOCOL.md](PROTOCOL.md): aquele leva JPEG da câmera
(helper Windows -> WSL). Este leva tinta (Draw -> consumidor).

## Garantias de isolamento

- `publish()` só guarda a última publicação e retorna (latest-wins); a
  codificação e o envio rodam em thread própria. Falha em um destino (disco
  cheio, socket morto) remove só aquele destino; o envio nunca para.
- Sem consumidor, consumidor lento ou conexão caída: o laço não espera. Um
  envio que passa de 2 s derruba o consumidor; consumidor que fechou é
  podado mesmo com a tinta parada.
- No encerramento limpo, a última publicação pendente ainda é despachada, e
  falha ao fechar a gravação é registrada no log (não é silenciosa).
- O consumidor só lê: o Draw trata fechar a conexão, inclusive só o lado de
  escrita (`shutdown(SHUT_WR)`), como desconexão.
- Desligado por padrão. Ligar: `--stream-port PORTA --stream-token-file ARQ`
  (sem env `DRAW_*`). `--stream-record ARQ` grava o fluxo num arquivo `0600` (recusado onde não há
  `os.fchmod`, i.e. Windows nativo: grave de dentro do WSL);
  sozinho (sem porta) não abre socket e não pede token. Não existe com
  `--markers`.
- Limites: até 8 handshakes simultâneos (o excedente é recusado sem thread) e
  8 consumidores autenticados (o mais antigo sai).

## Transporte e autenticação

Igual ao PROTOCOL.md: o Draw **escuta** em `127.0.0.1` e o consumidor
**conecta**; a conexão passa pelo mesmo handshake `DRAH`/`DROK` com token
compartilhado (16..256 bytes) antes de qualquer registro. Sem TLS (canal
local). O arquivo gravado não tem handshake.

Limite herdado do handshake do PROTOCOL v1: o consumidor envia o token **antes**
de o servidor provar quem é, então quem controlar a porta enquanto o Draw está
parado captura o token. Mesmo modelo de ameaça do PROTOCOL.md (canal local, o
token autentica o par, não cifra). Fechar isso pede um desafio-resposta, que é
versão nova do handshake, não deste stream.

## Registro

Cabeçalho fixo de **32 bytes**, little-endian, `struct.Struct("<4sHBBIIqHHI")`:

| Offset | Size | Campo | Valor |
|-------:|-----:|-------|-------|
| 0  | 4 | `magic` | `DRAL` |
| 4  | 2 | `version` | 2 (o leitor ainda aceita 1: gravações antigas) |
| 6  | 1 | `kind` | 1 = FULL, 2 = DELTA |
| 7  | 1 | `reserved` | 0 |
| 8  | 4 | `generation` | geração da folha; sobe a cada página nova |
| 12 | 4 | `ink_version` | versão da publicação dentro da geração |
| 16 | 8 | `ts_us` | relógio monotônico do Draw na **publicação** (o carimbo de **captura** vai no payload, v2) |
| 24 | 2 | `width` | largura da grade da página, px |
| 26 | 2 | `height` | altura da grade, px |
| 28 | 4 | `payload_len` | `1..16 MiB` |

- **FULL**: payload = `<ffq8f` (largura_mm, altura_mm da folha, `captured_at_us`, 4 cantos x,y do quad
  da âncora) + `zlib(máscara h*w uint8)`. `captured_at_us = -1` e cantos `NaN` = sem dado.
  (v1: só `<ff`.)
- **DELTA**: payload = `<HHHHq` (x, y, w, h, `captured_at_us`) + `zlib(sub-retângulo h*w)`. Só
  vale sobre um FULL anterior da **mesma geração e mesma forma**; senão é erro
  de protocolo.
- `generation` no fio é do publicador e monotônica: sobe a cada geração nova da
  trabalhadora e a cada **limpeza** (troca de câmera: FULL de tela vazia com
  geração nova). Não é o número interno da trabalhadora.
- Todo consumidor novo (e todo arquivo gravado) começa por um FULL; geração
  nova também. Versões intermediárias podem ser puladas (latest-wins), e o
  DELTA é calculado contra o último estado enviado àquele destino. Por isso o
  arquivo é a **mesma gramática** do socket, não os mesmos bytes: um consumidor
  tardio começa por FULL mesmo quando o arquivo gravou um DELTA.
- Limitação conhecida: a divisão de página por identidade que acontece
  **dentro** da trabalhadora (conteúdo diferente na mesma geração) não tem
  evento hoje; sai como DELTA (o canvas fica correto, mas a fronteira de
  página não é sinalizada).
- Publicação idêntica à anterior não gera registro, **exceto** quando o `InkDelta.full` da
  passada pede descontinuidade (tinta removida/migrada ou âncora movida): sai um FULL mesmo com
  máscara igual, porque a tinta mudou de lugar na câmera. O pedido sobrevive a uma publicação
  substituída antes de sair (latest-wins).
- Um DELTA cujo retângulo passa de 50% da página sai como FULL: nos clipes reais a bbox cobre
  91-97% e o DELTA economiza 2-5% (FULL em zlib: 37-55 KiB, 1,8 ms).
- `captured_at_us` usa o **mesmo relógio** do `ts_us` (monotônico do Draw): o consumidor mede
  captura -> publicação por `ts_us - captured_at_us`. É o carimbo do quadro da última passada que
  publicou tinta confirmada (inclusive a primeira, de página em branco); uma atualização só do
  provisório herda o último carimbo e a última âncora da geração, para o FULL de um consumidor
  tardio não perdê-los. `-1`/NaN só até o primeiro dado conhecido da geração. Exceção: uma
  publicação `full` (descontinuidade) vale pelo que traz, mesmo sem âncora — a recuperação de
  identidade zera a âncora sem mudar a geração, e o FULL seguinte sai com NaN em vez de herdar a
  velha (o consumidor trata NaN como "âncora desconhecida" e mantém só a escala da folha). O Draw só pede os
  metadados do `InkDelta` quando há stream (opt-in); se o delta não ficar pronto em ~5 ms (janela de
  ~1 ms entre publicar e registrar), a publicação sai sem carimbo novo (herda o anterior).
- Compatibilidade: o escritor só emite v2. Um consumidor v1 conectado ao vivo precisa atualizar
  (o `drawlink` ainda não existe); o leitor de referência aceita v1 e v2 (gravações antigas).

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

Provisória (só a confirmada sai), carimbos de tempo por traço (o `captured_at_us` é por
publicação),
posição de ponta de caneta, canal de controle consumidor -> Draw.
