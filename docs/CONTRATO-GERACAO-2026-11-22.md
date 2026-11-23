# Contrato de geração do ink worker — 2026-11-22

Nota de protocolo para `PageCleanWorker` sobre `main` (67e1167): como a
transição de geração (`new_generation`, `rezero_baseline` e a troca de
folha interna) se ordena contra o trabalho por passada que muta o
`PageModel` persistente, a referência de papel e a publicação.

Escopo: ordenação e validação de token — não altera semântica de voto,
semeadura ou exposição.

## Estado compartilhado e donos

| Estado | Dono/escritor | Leitores |
|---|---|---|
| `_generation` | `_condition` (transições) | `submit`, `_pass_commit`, `poll`, `new_generation` |
| `_page_model` (referência) | `_pass_identity` pode invalidar para `None` na fase A sob `_condition`; criação e re-zero no trecho sob `_pm_gate` | `new_generation` (snapshot), thread do worker |
| Conteúdo do `PageModel` (`_ref`, `_norm_ref`, `_observed_norm`, `_seeded`, `_unseen_hold`, `_unseen_blocked`, `_streak`, holds, chroma) | thread do worker via `exposed`/`observe`/`covered_unseeded`, dentro de `_pm_gate` | `new_generation` (`.copy()` do `observed_normalized_gray`), leituras derivadas no mesmo trecho |
| `_pm_epoch` / `_canvas_epoch` / `_pm_reset_seq` | `_pass_identity` invalida canvas/modelo na fase A sob `_condition`; re-zero e commit de epoch no trecho sob `_pm_gate` + `_condition`; `new_generation` sob ambos | `_pass_commit` (check do token), `baseline_state` |
| `_paper_ref` | `_condition` | `_pass_evidence`/guarda de papel |
| `_accumulator`, contadores, publicação | `_condition` (fase B) | `poll`, `stats` |

Tempo de vida: o `PageModel` é mantido por `(canvas_epoch, pm_reset_seq)`
— não por geração. `preserve_identity=True` atravessa a transição com a
mesma instância; a troca de folha (`keep_page=False`) e o re-zero criam
instância nova. A recriação por *epoch mismatch* (a `_pm_epoch` da
última passada difere da época corrente) acontece no início do trecho do
modelo, também dentro do gate.

Aliases mutáveis: `PageModel.observed_normalized_gray` devolve o buffer
`_observed_norm` vivo — `observe()` pode substituí-lo por uma referência
nova (`visible.all()`) ou escrevê-lo com `np.copyto` parcial. Arrays
NumPy não são profundamente imutáveis por dataclass congelada nem por
visão read-only, e NumPy/OpenCV liberam o GIL: leitura concorrente com
escrita no mesmo buffer pode observar estado inconsistente (ver
https://numpy.org/doc/stable/reference/thread_safety.html).

`WarpedPage` agora **copia** `image`/`visible`/`coverage` para buffers
próprios read-only na construção: o produtor pode reutilizar os arrays
de entrada sem alterar uma passada enfileirada. Contrato é de posse por
cópia, não de confiança na imutabilidade do produtor.

## Épocas de invalidação

- `generation`: invalida a passada inteira (voto, publicação, escrita de
  `_paper_ref`, trecho do modelo). `submit()` descarta páginas de
  geração velha na entrada; a passada em voo é descartada no check de
  token do gate.
- `pm_epoch = (canvas_epoch, pm_reset_seq)`: invalida a *vida* do modelo.
  A decisão de página nova/timeout adianta `canvas_epoch` na fase A; o
  re-zero aplicado adianta os dois componentes no commit da fase B; e
  `new_generation()` não-preserve adianta `pm_reset_seq`. O check do token
  usa ambos: `p.generation != _generation or p.pm_epoch !=
  (_canvas_epoch, _pm_reset_seq)`.

## Lock order e regiões

Ordem única: **`_pm_gate` → `_condition`**. Nenhum caminho pega
`_pm_gate` já segurando `_condition` (a fase A solta `_condition` antes
de chegar ao gate; `new_generation` pega o gate antes do lock). Sem
ciclo, sem deadlock por ordem.

`submit`, `poll`, `set_hand_presence`, `stop`, `baseline_state` e a fase
A de `_pass_commit` **não** tomam `_pm_gate`: a latência de `submit` é
contrato explícito e não espera o trecho do modelo.

Região crítica de `_pass_commit` (dentro de `_pm_gate`):

1. check do token sob `_condition` (stop + `generation`/`pm_epoch`);
2. `_pass_paper_and_page_model` — recriação por epoch, warps de
   registro, `exposed`, `covered_unseeded`, **troca do re-zero**
   (`PageModel` novo + `p.pm_epoch`/`_pm_epoch`), `observe`, `novel`/
   `appearance`/`admissible`/`pending` e derivadas;
3. lock 2b sob `_condition`: re-check do token (com `step` de re-zero),
   commit do re-zero (`_pm_rezero_requested` consumido,
   `_accumulator.reset`, `_canvas_epoch += 1`, `_pm_reset_seq += 1`),
   `_pass_vote_and_publish` e `notify_all`. O cálculo/anexo do delta vem
   depois, com `_condition` solto mas ainda sob `_pm_gate`.

A transição (`new_generation`) faz `with self._pm_gate, self._condition`
— captura de identidade (incluindo `observed_normalized_gray.copy()`),
reset de acumulador/âncoras e o incremento de `_generation`/`_pm_reset_seq`
acontecem atomicamente contra todo o trecho acima.

## Pontos de commit (linearização é por operação)

A passada **não** é uma única operação atômica — cada ponto abaixo tem o
seu commit:

- **Commit de mutação do modelo**: dentro do trecho, na escrita do
  mutador (`exposed`/`observe`/`covered_unseeded`/troca). Uma vez
  escrito, o efeito pertence ao objeto persistente — a fase B pode ainda
  rejeitar a publicação, mas não desfaz a mutação (e não precisa: o
  token era válido quando o modelo observou).
- **Commit de troca de referência (re-zero)**: a atribuição
  `self._page_model = pm` + `self._pm_epoch` no trecho, seguida do commit
  de época/canvas na fase B — **indivisíveis contra a transição** por
  estarem no mesmo gate: se a fase B rejeitasse depois da troca, o
  modelo novo ficaria com canvas velho e pedido pendente.
- **Commit de publicação (fase B)**: voto + publicação sob `_condition`
  dentro do gate — lineariza junto com o trecho contra `new_generation`.
- **Commit de fase A**: identidade, âncora e invalidações de página/canvas
  são atualizadas sob `_condition`, depois da validação inicial de
  `generation`. Se uma transição vier depois da fase A mas antes do gate,
  ela observa essas mudanças; a passada ainda é descartada antes de
  mutar o PageModel, votar ou publicar.
- **Commit de transição**: `new_generation` inteiro sob
  `gate → _condition` — lineariza entre passadas comitadas.
- **Rejeição de passada obsoleta**: `return True` no check do token —
  descarte, não commit; nada do trecho/fase B roda.

Consequência das duas ordenações possíveis:

- transição antes da fase A → a passada morre na validação inicial, sem
  efeitos;
- transição depois da fase A e antes do gate → a fase A já comitou sob
  `_condition`; o check do gate descarta o PageModel, a troca, o voto e a
  publicação da passada velha;
- transição depois → mutações do modelo **e** publicação desta passada
  comitam antes do retorno de `new_generation()` — o estado persistente
  fotografado na transição já as contém, e nada muda depois.

## Erros

Exceção em passada de geração obsoleta é telemetria
(`_telemetry_stale_errors`, exposto como `passes_stale_errors`) e a
trabalhadora segue; exceção na geração corrente seta `_error` e para a
thread — fatal. Classificação pela `p.generation` vs. `_generation` no
`except`, sob `_condition`.

## Custos medidos (grade A4 1200×850, esta máquina)

- `WarpedPage` com cópia: **~0,24 ms** por passada sobre ~5,1 MB de
  buffers (imagem uint8 + 2 máscaras) — visão seria ~0; o ownership sai
  por <0,3 ms/passada.
- Trecho sob `_pm_gate` por passada (check + `_pass_paper_and_page_model`
  + fase B): mediana **~20 ms**, p95 ~28 ms, max observado ~29 ms
  (30 passadas, conteúdo papel/tinta alternado).
- Estado mutável do `PageModel`: **~30 MB** (6 arrays float32 em grade
  cheia + 5 máscaras) — uma alternativa de copiar o modelo por geração
  pagaria isso por transição; o gate não copia nada.
- `observe()` isolado: ~0,9 ms.

`new_generation()` espera no máximo a região de uma passada em voo — da
ordem das dezenas de ms medidas acima, **não** um limite formal:
`threading.Lock` não promete ordem justa de aquisição (docs do Python) e
o trecho inclui OpenCV além de NumPy. `submit()` não toca o gate.

## Por que o gate cobre o trecho inteiro + fase B

Os três caminhos mutáveis desta base ficam todos dentro de
`_pass_paper_and_page_model` ou da fase B: a cópia de
`observed_normalized_gray` em `new_generation` corre contra `observe()`
(substituição/`copyto`); `covered_unseeded()` muta `_unseen_*`; e o
re-zero troca a referência e adianta `_pm_epoch` — com o commit de
época/canvas só na fase B. Estreitar para excluir os warps/OpenCV puros
economizaria talvez ~10 ms de espera da transição, mas exigiria partir
`_pass_paper_and_page_model` ao meio — e a troca do re-zero continuaria
exigindo fase B no mesmo gate (troca sem commit de época deixaria modelo
novo com canvas velho). O gate largo é o mínimo correto para esta
estrutura.

## Testes

`tests/test_worker_concurrency.py` — repros determinísticos com
`threading.Event` (barreiras imediatamente antes da mutação/troca, sem
sleeps): mutação de modelo persistente vs. transição, repovoamento de
`_paper_ref`, classificação de exceções nos dois lados, transição entre
fases, snapshot `observed_normalized_gray` vs. `observe()`,
`covered_unseeded()` vs. transição, e troca do re-zero nas duas
ordenações (passada antes → swap comitado; transição antes → passada
descartada sem tocar o modelo).
