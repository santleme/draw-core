# draw-dral

O fio DRAL do Draw: a gramatica do stream de tinta confirmada (Draw ->
consumidor, `bridge/STREAM.md`) mais o handshake de token compartilhado
dos dois transportes (`bridge/PROTOCOL.md`, "Authentication handshake").

Conteudo:

- `draw_dral.stream` — registros DRAL (`InkRecord`, `iter_records`,
  `records_from_file`, `pack_full`, `pack_delta`), o canvas do lado
  consumidor (`InkCanvas`) e o publicador do lado Draw
  (`InkStreamPublisher`).
- `draw_dral.wire` — erros (`ProtocolError`, `AuthenticationError`),
  resolucao de token, `client_handshake`/`server_handshake` e o
  leitor `_read_exact` compartilhado pelos dois fios.

Dependencias: `numpy` e biblioteca padrao — propositalmente sem cv2,
onnxruntime ou o restante do draw-core, para que um consumidor (o
drawlink) instale so o fio.

O transporte de camera (registros `DRAW` com JPEG, `HelperCapture`)
nao faz parte deste pacote: continua em `draw.protocol`, que reexporta
os nomes do handshake para compatibilidade.

## Instalacao

Dentro do monorepo, o uv resolve o membro pelo workspace
(`uv pip install -e .` na raiz). Fora dele — ex.: o drawlink fixando o
core num commit — o pacote sai pelo mesmo Git do draw-core, no mesmo
SHA, com `subdirectory`:

```sh
pip install "draw-dral @ git+https://github.com/santleme/draw-core@<SHA>#subdirectory=packages/draw-dral"
```

Enquanto `draw-dral` nao esta publicado num indice, quem instala
`draw-core` por Git precisa deste pino explicito: `[tool.uv.sources]`
nao acompanha a distribuicao.

Duas ressalvas de consumo: (1) o pino de SHA fixa a fonte, mas pip pode
manter uma copia ja instalada da mesma versao — atualizar pede
`--force-reinstall` ou ambiente novo; (2) o drawlink so deixa de puxar
as dependencias pesadas do draw-core quando remover seus demais
imports `draw.*` (`draw.config`, `draw.exporters`) — repointar so o fio
nao basta.
