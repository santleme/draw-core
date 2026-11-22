# Corpus de clipes reais

O corpus é a régua de condição real: clipes gravados na mesa (luz,
sombra, mão, folha movida) rotulados com o que se espera do laço,
reexecutados pelo pipeline verdadeiro a cada mudança de robustez de
tinta. Ele responde "a correção do #133 melhorou o clipe que
reproduzia o #133?" — com delta medido, não com opinião.

Os clipes moram **fora do repositório**, em `~/draw-data/` (hoje
`bench-clips/`). O `.gitignore` já proíbe versionar `captures/`,
`sessions/` e `*.drawcap`: dentro do repo ficam só a ferramenta
(`lab/corpus/`), os consumidores (`lab/replay_lab.py`,
`lab/bench_ink.py`) e esta doc. Bytes de cena nunca entram no git.

## Layout de um clipe

Um diretório por clipe, nome em kebab-case que diz a condição
(`escrita-mao-sombra`, `e5-escuro-estatico`):

```
<corpus>/escrita-mao-sombra/
    f0000.png ... f0231.png   # frames em ordem, numeracao continua
    clip.json                 # rotulo (schema abaixo)
    regions.json              # opcional: caixas no canvas final
```

A numeração contínua `f0000…` não é enfeite: os índices dos rótulos
(`frames` de `expected`) são **posições** no inventário ordenado pelo
sufixo numérico dos `f*.png` — só coincidem com o número do arquivo
quando não há lacuna. `corpus check` acusa inventário não contíguo
justamente porque lá as duas leituras divergem.

`regions.json` é `{"writing": [[x0,y0,x1,y1], ...], "ruling_only":
[[x0,y0,x1,y1], ...]}` em coordenadas do canvas retificado final; o
`bench_ink.run_clip` conta pixels de escrita × pauta pura dentro
delas. O `corpus check` exige retas inteiras, não negativas,
estritamente ordenadas (`x0 < x1`, `y0 < y1`) e dentro do canvas —
a mesma régua que o replay usa antes de contar.

## `clip.json` — schema v2

```json
{
  "version": 2,
  "scene_fps": 8.0,
  "frames": 232,
  "desc": "escrita ao vivo: mao e sombra sobre a pagina esquerda",
  "orientation": "paisagem",
  "rotate_applied": 180,
  "conditions": {
    "lighting": "sombra-movel", "surface": "folha-solta",
    "content": "escrita", "occluder": "mao",
    "motion": "estatica", "camera": "f91-mesa"
  },
  "provenance": {
    "source": "desk_capture", "captured_at": "2026-11-20",
    "tool": "python -m lab.desk_capture", "note": "repro do #133"
  },
  "expected": [{"rule": "ink_present", "min_px": 2000}],
  "issues": [133]
}
```

- `version`, `scene_fps`, `frames`, `desc` — obrigatórios.
- `orientation`, `rotate_applied` — opcionais (`run_clip` assume
  `retrato` na ausência de `orientation`).
- `conditions` — rótulos controlados da cena; é o eixo do `sweep` e
  do `report`, e o que um dia calibra o `PageModel`.
- `provenance` — de onde o clipe saiu (`source`, `captured_at`,
  `tool`, `note` opcionais).
- `expected` — regras de expectativa avaliadas no `run`:
  `no_ink` (canvas ~vazio), `ink_present` (canvas ≥ `min_px`),
  `canvas_stable` (canvas não colapsa nem zera no meio). `min_px` é
  opcional onde fizer sentido. Escopo opcional por regra:
  `frames: [a, b]` restringe a janela de quadros (posição no
  inventário ordenado pelo sufixo numérico dos `f*.png`, inclusivo —
  na captura contígua `f0000…` coincide com o número do arquivo) e
  `region: [x0,y0,x1,y1]` restringe ao
  retângulo do canvas — tinta fora da janela ou do retângulo não
  satisfaz a regra. Sem `frames`, `no_ink`/`ink_present` leem o
  estado final do canvas; com `frames`, o pico de px confirmado
  dentro da janela (`no_ink` exige pico 0, `ink_present` pico
  ≥ `min_px`). `canvas_stable` lê a série inteira (ou a janela) e
  falha se, depois da primeira tinta confirmada, o canvas confirmado
  zera ou a sessão de tinta reabre (troca de geração) — retração de
  provisório não derruba a regra.
- `issues` — números das issues do GitHub que o clipe reproduz.

A v1 é o que existe hoje nos clipes antigos: sem chave `version`,
só `{scene_fps, orientation, frames, desc, rotate_applied}`.
`corpus upgrade` reescreve para v2 pedindo os rótulos que faltam.

## `corpus.json` — manifesto do corpus

Na raiz do corpus, gerado por `corpus manifest`: sha256 de cada
clipe (mesma disciplina fail-closed de `assets/manifest.json` — o
que não confere com o hash não é o clipe que a PR mediu), contagem
de frames, `conditions` e referências de issue. `corpus verify`
confere o diretório contra o manifesto antes de qualquer `run`.

## CLI

```sh
python -m lab.corpus init <dir-do-clipe> --fps <fps>   # rotula um clipe novo (clip.json v2)
python -m lab.corpus upgrade <dir-do-clipe>            # v1 -> v2
python -m lab.corpus check <corpus>                    # valida schema, frames e regioes
python -m lab.corpus manifest <corpus>                 # (re)gera corpus.json
python -m lab.corpus verify <corpus>                   # confere sha256 contra corpus.json
python -m lab.corpus run <corpus> --bundle <nome>:<flag1,flag2> [--bundle ...]
python -m lab.corpus sweep <corpus> --grid '<campo>=<v1,v2>[;<campo>=<v3,v4>]'
python -m lab.corpus report <results.jsonl>
```

`run` executa bundles — conjuntos nomeados de flags
(`--bundle nome:flag1,flag2`, repetível; flags separadas por
vírgula) — sobre o corpus inteiro e grava `results.jsonl` (uma linha
por clipe × configuração, com as medidas do `run_clip`, o veredito
das regras `expected` e os rótulos `conditions` do clipe),
`report.md` (tabela antes/depois por clipe e por condição) e
`run.json` (baseline e bundles da corrida). `--baseline` escolhe o
bundle de comparação dos deltas; sem ele, o primeiro bundle
declarado é o baseline. `sweep` expande a grade inline
`campo=v1,v2;campo=v3` em um bundle por combinação (`--base` traz
flags comuns a todos). `report` rerrenderiza o Markdown de um
`results.jsonl` existente: usa o baseline salvo no `run.json` da
corrida (sem `run.json` ao lado, cai para o primeiro bundle dos
resultados), e `--baseline` sobrepõe explicitamente.

## A regra de trabalho

**Toda PR de robustez de tinta anexa o clipe que a reproduz.** O
fluxo:

1. Grava os quadros brutos do clipe — na mesa ou reexecutando o
   `.drawcap` da sessão em que o defeito apareceu — com
   `python -m lab.desk_capture --source <fonte> --fps <fps> --frames <dir-do-clipe>`.
   `--frames` despeja a cena de cada quadro como `fNNNN.png`, sem
   anotação e sem stride; os dumps `scene_*`/`canvas_*` e o
   `session.json` caem em `captures/` (já ignorada) como evidência
   de diagnóstico — não são material de clipe.
2. Move o diretório de quadros para o corpus em `~/draw-data/` (ou
   já aponta `--frames` direto para lá) e rotula:
   `corpus init <dir> --fps <fps-cena>` preenche o `clip.json` —
   `issues` aponta a issue, `expected` declara o comportamento
   correto.
3. `corpus check <corpus>` valida schema, contagem de quadros e
   regiões — inclusive a numeração contígua: um inventário com
   lacuna acusa aqui, porque é onde os índices de rótulo deixam de
   coincidir com os números de arquivo. O `manifest` verificado não
   carrega esse aviso.
4. `corpus manifest` registra o sha256; a PR referencia **nome do
   clipe + número da issue** na descrição.
5. Alegação de robustez exige `corpus run` antes × depois no bundle
   da issue. Fixture sintético sozinho não fecha PR de robustez —
   o simulador não tem a condição real que quebrou.

## Posição na pilha de avaliação

O corpus roda sobre `lab/replay_lab.run_replay`: o laço real do app
(`Pipeline`, detecção combinada, trabalhadora página→limpa) sob o
relógio de cena determinístico — detecção com atraso de 1 quadro,
cadência sintética. Isso cobre a classe **funcional/geométrica**;
a classe **temporal** (cadência viva da câmera, fila do driver)
continua exigindo sessão de mesa. As medidas por clipe são as de
`lab/bench_ink.run_clip` — `canvas_px`, provisório × retratado,
`cpu_s_per_scene_s`, `writing_px`/`ruling_px` quando há
`regions.json`.

Os simuladores de `docs/SIMULATOR.md` seguem sendo a camada de GT
controlado (verdade por pixel, cenários encadeados); o corpus é a
camada de delta em condição real. Nenhum dos dois substitui a mesa
para timing.

## Limites honestos

- As medidas são **deltas + regras de expectativa**, não verdade
  absoluta: `ink_present` com `min_px` diz que tinta ficou de pé,
  não que cada traço saiu correto. Para gabarito por pixel, o
  sintético.
- Telemetria do modelo de página (`pm_seeded_frac`,
  `pm_novel_px`, `pm_suppressed_px`, `pm_pending_px` nos registros
  da trabalhadora) só aparece quando a build tem o flag
  `ink.page_model=1` — hoje na `feat/page-model`; sem o flag, as
  chaves não existem e não há o que comparar.
- Clip antigo sem `conditions` não entra em fatia de relatório por
  condição — rode `corpus upgrade` em vez de inferir rótulo.

## Papel de calibração

É este harness que vai calibrar os limiares do `PageModel` —
`_NOVEL_RATIO`, `rate`, `block`, `confirm_obs` — quando existir
massa de clipes rotulados por condição. A nota de literatura que
fundamenta o modelo e lista o que está pendente de medição está em
`docs/research/nota-page-model-literatura.md` (repo
`GUZZBR1/-draw-site-`).
