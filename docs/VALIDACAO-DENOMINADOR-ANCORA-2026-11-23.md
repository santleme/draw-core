# Validação do escore de arbitragem da âncora (#180) — síntese de pesquisa

Pesquisa inicial: 2026-11-23. Implementação e validação local: 2026-11-24. Branch:
`work/180-on-main`. As seções 0–8 preservam a pesquisa feita em duas rodadas (fontes
secundárias e depois fontes primárias com Firecrawl e experimentos numéricos fora do
repositório); a seção 9 registra as alterações e medições posteriores.

**Atualização:** as seções 0–8 registram o estado e as conclusões em 23/11/2026. A seção 9
preserva snapshots de implementação intermediários de 24/11/2026; algumas de suas medições e a
primeira regra de aposentadoria do resíduo foram superadas por correções posteriores. A seção 10
é o fechamento da versão final local: inclui A/B repetido contra `0bf0177`, o teste de worker
para `unregistered` e o gate completo. Nenhum resultado sem rótulos independentes é tratado como
prova de precisão funcional.

## 0. Estado do código que esta síntese analisa

Os relatórios foram escritos contra a versão intermediária
`_total_after_clip(raw_total, before, after) = raw_total * min(sum(after)/sum(before), 1)`.
O HEAD deste branch (`2b193af`, "correct anchor residual arbitration") a **substituiu** por
`anchor._overlap_fraction(prob, snapshot)`: fração da massa de `prob` (já permutada e
transladada) que cai no snapshot, com teto em 1. Equivale à alternativa "denominador
`sum(after)`" que a pesquisa discute abaixo. Consequências, derivadas e **não medidas
contra este HEAD**:

- O escore fica em [0,1] por construção em qualquer hipótese (inclusive cíclica); o clamp
  `min(...,1.0)` já existe, então o item "clamp defensivo" está atendido.
- Sem permutação o resultado coincide com o da versão intermediária. Com `rot90`+resize o
  denominador muda por um fator de massa do resize (medido: ~0,03% médio, 0,3% máximo em
  traços; até ~11% em linhas verticais de 1 px com `INTER_LINEAR`). Os limiares da
  arbitragem (delta 0,05; escores cíclicos 0,23–0,27) foram calibrados no denominador
  antigo; recalibrar ou ao menos comparar escores antes/depois em cenas cíclicas.
- Verificado neste HEAD (2026-11-23): `tests/test_anchor_arbitration.py`,
  `test_ink_contracts.py`, `test_page_model.py`, `test_ink_registration.py` = 256 passed;
  `ruff check` limpo. A suíte completa **não** foi reexecutada neste HEAD, e o CI remoto
  não rodou (PR #180 remoto segue em `93f1c28`, base `816224f`).

## 1. Conclusões principais

1. **O erro original era real**: subtrair massa cortada (domínio pós-permutação) de um total
   do domínio bruto mistura domínios. Normalizar pela massa do próprio mapa já transformado
   (a versão atual) ou pela fração de massa sobrevivente (a intermediária) corrige isso.
2. **O teto de 1 não falha nas hipóteses cíclicas** (correção ao 1º relatório, que dizia o
   contrário). Em 1000 sementes × 4 interpolações o escore corrigido nunca passou de 1
   (máx. 0,94–0,99). O que passa de 1 é o escore **sem** correção: +0,3% em traços, até
   +11,2% em linhas verticais de 1 px (`INTER_LINEAR`); +2,2% a +52% com `INTER_NEAREST`.
   `INTER_CUBIC` gera pixels negativos (até −0,32): único caso em que o clamp é necessário.
3. **O escore não penaliza massa recortada** (simetria com o escore de câmera, que também
   divide pela massa sobrevivente). É decisão de projeto: uma hipótese com 5% da massa dentro
   da grade que acerta o snapshot pontua 1. Só os portões de recall/ganho do estimador limitam
   esse caso. Deve constar na descrição do PR.
4. **Nada na literatura valida o desenho em volta**: a tolerância de 2 px, o rival de 90% por
   eixo e a margem absoluta de 0,05 não têm fonte. A literatura só apoia princípios gerais
   (verificar hipóteses com portão antes de ranquear; margens relativas e calibradas; refino
   local após pico grosso).
5. **Maior risco**: o `identity_residual` persistido não tem metadados de frescor. Um resíduo
   velho pode "explicar" a pose por coincidência dentro de 2 px e suprimir um gatilho real.
6. **Evidência é toda sintética.** Não existe dataset público com queda e retorno do tracker,
   e a duplicação apontada pelo revisor **não foi reproduzida**.
7. **Prontidão**: #180 **não** está pronto para merge. #203 não é coberto pelas fontes
   (sem veredito de pesquisa; só CI limpo no código rebaseado).

## 2. O que as fontes primárias sustentam

| Tema | Achado | Fonte |
|---|---|---|
| `TM_CCORR_NORMED` | `Σ(T·I)/√(ΣT²·ΣI²)`: cosseno, ignora massa e recorte; em [0,1] para entradas não negativas. Caminho com máscara divide sem guarda (Inf/NaN); sem máscara devolve 0 com energia ≤ 0,5 | docs.opencv.org 4.x; `templmatch.cpp` |
| Máscara + sobreposição | Padfield restringe numerador e denominador aos pixels com as duas máscaras = 1; zera sobreposição "razoavelmente pequena" (sem número). scikit-image zera `overlap_ratio=0,3 ×` o **máximo** de sobreposição | slides CVPR 2010; fonte do scikit-image |
| `cv2.resize` | `INTER_LINEAR/CUBIC`: `src=(dst+0,5)·scale−0,5`, borda replicada; `INTER_NEAREST`: `floor(dst·scale)`; `INTER_AREA` só é média por área com ambas as escalas ≥ 1, senão é "emulada" | `resize.cpp` 4.x |
| Massa sob `rot90`+resize 240×320 | `INTER_AREA` conservou a soma (<1e-6) em todos os casos; `INTER_LINEAR` perde massa em linhas finas conforme a fase (1 px vertical: 0,888–1,112); `INTER_NEAREST` pode perder a linha inteira | experimentos (ver §5) |
| Translação inteira com recorte | `np.roll`+zeros e `warpAffine(INTER_NEAREST)` idênticos em 1000/1000; massa restante = soma da janela que fica. Translação não inteira pode duplicar coluna | experimentos |
| Teste de razão de Lowe | rejeita razão > 0,8: "eliminates 90% of the false matches while discarding less than 5% of the correct matches" (base de 40.000 pontos). Razão, não diferença absoluta; o "segundo vizinho" exclui o mesmo objeto (precedente para fundir hipóteses equivalentes). 0,8 é calibrado (Fig. 11), sem procedimento declarado | Lowe 2004 |
| ORB-SLAM | R_H > 0,45; candidatos > 75% do melhor mantidos; três candidatos consecutivos consistentes; reverificação por inliers; ORB-SLAM3 descarta/estaciona mapa após perda | arXiv 1502.00956, 2007.11898 |
| Guizar-Sicairos 2008 | todos os métodos começam com pico de FFT (upsampling 2) para "picos de magnitude similar" e refinam numa vizinhança 1,5×1,5 px: pico grosso + refino local é o padrão | Optics Letters (espelho) |
| `phaseCorrelate` | `response` = potência no centroide 5×5, ≤ 1, menor com vários picos; sem limiar oficial (diagnóstico, não confiança) | docs.opencv.org |
| Abstenção | risco–cobertura com limiar calibrado em amostra retida e limite binomial; nada sustenta 0,05 absoluto | Geifman & El-Yaniv |
| Estado obsoleto | paralelos: gating de Kalman (NIS contra χ²) e descarte de estado do ORB-SLAM3; regras de invalidação concretas são **inferência** | kalman-filter.com; ORB-SLAM3 |

## 3. Validação e estatística

- **Mutação** (Just et al., FSE 2014; 357 faltas reais): 73% acopladas a mutantes, ~10% pedem
  operador novo, 17% sem operador. Resultado limpo de mutação não cobre uma classe de bug de
  geometria. Para ~40 linhas: reverter manualmente cada elemento do fix e mostrar que ao menos
  um teste falha; `mutmut` 3.8.1 (POSIX/WSL) ou `cosmic-ray` só para mutantes de fronteira.
- **Eventos raros** (limite superior unilateral 95% com 0 eventos): n=10: 25,9%; 20: 13,9%;
  30: 9,5%; 60: 4,9%; 100: 3,0%; 299: 1,0% (regra dos três ≈ 3/n; Hanley & Lippman-Hand 1983).
  Eventos do mesmo clipe são correlacionados: 20 eventos em 4 clipes dão 13,9% (ICC 0) a 28,1%
  (0,3), 36,2% (0,5) e 52,7% (1,0). Afirmar "< 10%" pede ~60–90 eventos (12–18 clipes);
  "< 1%" é irrealista com clipes reais. Redação segura: "0/20 eventos em 4 clipes; limite
  superior ≥ 14%; não é afirmação de taxa".
- **Testes instáveis** (Luo et al. 2014; 161 commits classificados): espera assíncrona 45%,
  concorrência 20%, ordem 12%. O teste de 10 ms do `observe` se parece com espera por tempo.
  `pytest-benchmark`: `pedantic`, `--benchmark-compare-fail`. Orçamento relativo e o conselho
  sobre WSL2 são inferência. O "1,5% do Google" **não** foi verificado (blog deu 404).
- **Revisores automáticos**: nenhum estudo dá taxa de falso positivo de bot específico. Num
  estudo (arXiv 2508.18771), 42–48% dos comentários válidos de ferramentas ficam sem
  tratamento, então "não reproduzi" é evidência fraca nos dois sentidos. Tratar como
  hipótese: matriz de tentativas, teste-guarda, thread aberta até haver clipe real.
- **Hypothesis**: `arrays`, `RuleBasedStateMachine` para propriedades (escore em [0,1],
  invariância a resize de ambos os mapas, equivariância à translação) e uma máquina de estados
  queda → retorno → deslize para caçar a duplicação.

## 4. Evidência real: o que existe e o que gravar

Nenhum dataset público tem queda e retorno do tracker com folha movida e desenho. Análogos:
SmartDoc 2015 Ch.1 (150 clipes, quad por quadro, CC BY 4.0; sem regra de falha de tracking —
métrica é Jaccard por quadro); MIDV-500 (500 clipes, condições "Hand" e "Partial", licença
dos clipes não confirmada); MIDV-Holo (CC BY-SA 2.5, share-alike: revisão jurídica). VOT
reinicia 5 quadros após a falha e ignora 10 (curto prazo); longo prazo sem reinício.

Protocolo proposto (desenho da pesquisa, sem fonte): ~60 clipes estratificados por movimento
no retorno (parado; 4/8/12/14 px; deslize durante o desenho; rotação k×90°; folha levantada),
iluminação e oclusão por mão; moldura ArUco/ChArUco só offline para pose; tinta A antes e B
depois da queda; apagamento roteirizado e scan plano como verdade; controle sem o fix e sem
queda. Unidade estatística: o clipe. Fixar alvo e n antes e não parar cedo.

## 5. Experimentos (fora do repositório)

Scripts em scratchpad temporário (opencv 5.0.0, numpy 2.5.5), mapas float32 240×320, polilinhas
aleatórias de 1–5 px, 1000 sementes, `INTER_LINEAR/AREA/NEAREST/CUBIC`. Não são tinta real.

## 6. Lacunas

Corpo do artigo de Padfield (TIP 2012) não lido (só slides); fonte de `phase_cross_correlation`
deu 404; Chow 1970 e El-Yaniv & Wiener 2010 não lidos; Guizar-Sicairos via espelho
academia.edu; comportamento de não convergência do `findTransformECC` não está na doc; sem
fonte para 2 px, 90% e 0,05; Hartley & Zisserman não consultados; protocolo do LaSOT não
obtido; Firecrawl limitou taxa; afirmações sobre o código vêm do coordenador, não dos
pesquisadores; mapas sintéticos não são tinta real.

## 7. Próximos passos

Bloqueantes do #180:

1. Reexecutar a suíte completa e o CI no HEAD atual; atualizar o PR remoto (hoje em `93f1c28`
   sobre `816224f`). Push só por decisão explícita.
2. Teste de contrato que fixe a definição do escore (massa do mapa já permutado e
   transladado) e o caso `total <= 0 → 0`; incluir cíclico com `INTER_LINEAR` e linha vertical
   de 1 px.
3. Comparar escores cíclicos antes/depois da troca de denominador e recalibrar o delta 0,05 se
   preciso.
4. Frescor do resíduo persistido (evento/instante, suporte, margem, pose de referência);
   invalidar em salto de pose e reinício do tracker; exigir resíduo fresco no 2º ramo do
   gatilho de pose, com log do ramo usado.
5. `_RESIDUAL_EXPLAINS_PX` no objeto de configuração (sem flags `DRAW_*`), com testes de
   fronteira em 1,9 / 2,0 / 2,1 px.
6. Reversão manual de cada elemento do fix e do gatilho, mostrando que um teste falha em cada.
7. Descrição do PR declarando: evidência sintética, sem clipe real, 2 px e 0,05 sem fonte,
   escore que ignora massa recortada, e a regressão do gatilho de pose (apagamento) que o PR
   introduziu e corrigiu.

Desejáveis: fundir hipóteses equivalentes antes da margem e usar margem relativa; observabilidade
por eixo no estimador; testes de propriedade; 30–60 clipes reais com controle; medir o custo do
estimador em A/B no fluxo completo e rebaixar o teste absoluto de 10 ms a opt-in.

## 8. Execução local das recomendações

Atualização da mesma branch, após a síntese acima. Código e documentação ainda não foram
publicados no PR remoto.

### Escore e limiar

- Contrato adicionado para o caso extremo `rot90 + INTER_LINEAR` com linha vertical de 1 px
  na borda da grade: a soma transformada é `266,67` contra `240` original. O cálculo anterior
  pontua `1,111`; o atual, no domínio transformado, pontua `1,0`. O escore vazio continua
  sendo `0`.
- Comparação sintética de nove combinações (`k=1..3`; resíduos `(0,0)`, `(4,0)` e `(3,-5)`):
  o melhor candidato cíclico permaneceu o mesmo e nenhuma margem entre candidatos cíclicos
  cruzou `_ARB_DELTA=0,05`. A menor margem observada foi `0,381`. Os testes completos de
  pouso rotacionado também mantiveram o veredito. Não alterei o limiar: isso é apenas evidência
  sintética, e não calibração retida nem validação com câmera.
- A normalização atual continua sem penalizar massa cortada da grade. O limiar de `0,05` e o
  rival por eixo de `90%` continuam sem sustentação de fonte/calibração independente.

### Proveniência e uso geométrico do resíduo

- A estimativa agora carrega pixels casados (`support_px`), pixels observáveis, recall e margem
  de escore top-vs-runner-up. Ao persistir, registra geração, tempo monotônico, pose de
  referência e esses valores.
- O uso geométrico é invalidado por falta de pose, salto entre poses acima de `6 px`, deriva
  acumulada acima de `6 px`, suporte insuficiente ou troca de geração. Os `6 px` vêm do limiar
  já usado para assentamento; a aplicação disso como critério de frescor é uma inferência de
  engenharia, sem validação em clipe real. Uma nova geração sempre descarta o resíduo anterior.
- O resíduo obsoleto ainda pode corrigir o alinhamento do conteúdo, mas nunca reduzir o
  deslocamento usado pelos gatilhos de arbitragem/follow nem pelo portão de remoção. A
  telemetria registra frescor, motivo, idade, pose de referência, deriva e ramo geométrico.
- `residual_explains_px` ficou no fim de `InkConfig`, para preservar a ordem dos argumentos
  posicionais existentes. O default segue em `2,0 px`; os testes cobrem configurações em
  `1,9`, `2,0` e `2,1 px`, mas o valor de produção segue sem calibração empírica.

### Verificação e limites restantes

- Gate local: `ruff check .`, `ruff format --check .` e `pytest -q` passaram; `2334 passed`,
  `2 skipped`. Os testes focados de âncora, identidade, contratos e movimento deram `327 passed`,
  `1 skipped`.
- O teste de orçamento de `PaperModel.observe` é opt-in via
  `RUN_PAGE_MODEL_TIMING=1`; executei-o separadamente e passou. Isso evita um limite absoluto de
  `10 ms` sujeito a contenção no gate padrão.
- Mutações isoladas que removem o filtro de aceite dos candidatos, ignoram o frescor do resíduo
  ou restauram o limiar fixo de `2 px` fizeram seus respectivos testes falharem. O contrato do
  escore calcula explicitamente o resultado da fórmula anterior (`1,111`) e o distingue do
  atual (`1,0`).
- Não há clipe rotulado de queda/retorno do tracker nesta branch e não rodei CI remoto; a suíte
  local não substitui esses dois itens. A descrição do PR ainda precisa declarar que a evidência
  é sintética, que `2 px`, `90%` e `0,05` não têm calibração independente, que o escore ignora
  massa cortada e que a regressão de remoção após movimento físico foi corrigida.
- Testes de propriedade com Hypothesis, A/B de custo do estimador no fluxo completo e o corpus
  real de 30–60 clipes seguem como trabalho futuro. Hypothesis não é dependência do projeto.

## 9. Implementação e evidência — 2026-11-24

### Base e escopo local

- A branch `work/180-on-main` está baseada em `origin/main` no commit
  `5e53087984705423ab6061c16dc39057bf0cd6d6`; o HEAD base desta atualização é `0bf0177`
  (`fix(ink): track identity residual freshness`). O rebase não teve conflitos. As alterações
  abaixo continuam locais, sem commit ou push; o PR remoto e seu CI não foram atualizados.
- O escopo desta atualização inclui os escores da hipótese de página e da câmera, a validação
  do suporte projetado, a abstenção quando falta evidência observável, a resolução de resíduos
  pendentes e seus contratos. A revisão também encontrou e corrigiu um caminho sem cobertura
  que levantava `UnboundLocalError`. Não foram alterados `contracts.py`, dependências ou modelos.

### Escores e suporte projetado: não premiar massa cortada

O denominador da hipótese de página é a massa após eventual permutação/redimensionamento, mas
antes da translação inteira. Assim `rot90` continua comparado no domínio que produziu o
numerador, enquanto a massa que sai da grade conta como evidência perdida. A mesma regra é
usada nas hipóteses cíclicas. Os escores da página e da câmera são calculados de forma
independente: câmera sem massa registrada não apaga uma pontuação válida da hipótese da página.

O teste adversarial reproduz a falha de normalizar pela massa sobrevivente: de 10.000 unidades,
uma translação extrema deixa 100 unidades, todas sobre tinta confirmada. A fórmula antiga dá
`100/100 = 1,0`, superando um candidato rival que casa 90% (`0,9`). A nova dá `100/10.000 =
0,01`. O contrato de arbitragem testa também um caso menos extremo: 300 unidades de entrada,
90% recortadas, resultam em `score_page = 0,1`; a hipótese registrada concorrente marca
`score_camera = 0,9`, e o veredito é `spurious`.

O escore da câmera também tem denominador antes do recorte: ele re-warpa `p.prob` com a
visibilidade de origem, mede o suporte projetado completo e aplica a cobertura na geometria da
câmera. A exceção `anchor_ink` só é unida quando há máscara de cobertura; sem essa máscara, tinta
confirmada não corta o denominador. O numerador continua sendo a probabilidade registrada e
válida na grade da âncora. Se a homografia for inválida, cruzar o horizonte projetivo ou exceder
o limite de 32 vezes a área de origem, o escore da câmera fica indisponível; não há fallback para
a massa recortada. A arbitragem registra `indeterminate` e abstém. Se houver canvas confirmado,
mas massa observável abaixo do mínimo, também abstém em vez de trocar a âncora por
`landing_default`.

`_padded_warp_geometry` normaliza a escala e o sinal da homografia, projeta o suporte bilinear
`[-1, W] × [-1, H]` e o une à grade normal de destino. Isso retém o halo de interpolação nas
bordas e permite inserir a máscara do canvas no mesmo sistema de coordenadas. Contratos cobrem
matrizes projetivas finitas, homografias escalares equivalentes, impulsos nas bordas em escalas
3,1/3,9, máscaras de origem/destino e comparação com a grade de destino normal.

### Limite de memória: acumular em tiles

A primeira versão materializava todo o warp acolchoado. Na grade A4 de 850×1200, o limite de
32× permitiria 32,6 milhões de amostras; só o `float32` ocuparia cerca de 124,5 MiB. Coberturas,
máscara de tinta confirmada e a cópia temporária de `np.where` poderiam elevar o pico acima de
300 MiB. Isso era um risco de memória real para geometria extrema, embora não fosse o tamanho
observado nos clipes.

O cálculo de produção agora percorre a saída em tiles de no máximo 1024×1024 e soma com `where`,
sem criar o canvas projetado completo nem a cópia `np.where`. Um tile de probabilidade `float32`
tem no máximo 4 MiB; as máscaras temporárias são de um byte/bool por pixel. Pela contagem dos
arrays explícitos, os buffers transitórios do helper ficam na ordem de 10 MiB na grade A4, mais
temporários internos do OpenCV. Isso é um limite de alocação calculado pelo código, não uma
medição de RSS do processo. O limite total de 32× ainda limita o trabalho computacional e, se
excedido, produz abstenção.

O teste compara a soma em tiles com a referência materializada para warp projetivo, cobertura e
máscara de tinta; outro intercepta `warpPerspective` e exige que cada `dsize` respeite o tamanho
do tile. Contratos garantem invariância quando só a matriz da cobertura recebe um fator escalar
e quando a homografia da probabilidade e a homografia composta da cobertura recebem o mesmo
fator, de `±1e-300` a `±1e300`.

### Resíduo de identidade: snapshot intermediário, supersedido abaixo

O contrato novo percorre a sequência: desenho confirmado → retorno da mesma folha → estimativa
de resíduo `(0, -6)` → novo deslize físico com o tracker acompanhando → veredito `landing` →
novo traço. Quando a pontuação da página sem o resíduo é pelo menos tão alta quanto a pontuação
com o resíduo, o pouso aposenta o deslocamento antigo antes de re-registrar a passada. A
telemetria registra o deslocamento aposentado e o motivo; o teste exige que a tinta existente
não ganhe cópia deslocada e que o novo traço apareça na posição esperada. O contrato roda nos
regimes `production` e `page_model_guarded`.

Uma primeira implementação zerava o resíduo em todo `landing`. Ela falhou em 10 contratos de
regressão: oito combinações de residual `2/6 px` × eixo do deslize × regime, e dois casos de
banda escura com residual `6 px`. Esses casos mostram que o resíduo às vezes continua necessário
para compensar um viés real do warp. A regra final compara as duas pontuações e conserva o
resíduo quando ele melhora a hipótese de página. Os contratos cobrem pouso reto e rotacionado,
preservação/aposentadoria do resíduo e o novo traço após a troca de âncora; a rodada focada final
está contabilizada abaixo.

Esse snapshot usava uma regra local, sem calibração independente. Ele aposentava o resíduo em
empates; essa decisão foi posteriormente reproduzida como defeito: o estado não continha
informação para escolher entre as duas geometrias. A versão final abstém quando a diferença entre
as pontuações é `<= _ARB_DELTA` (0,05), conserva o resíduo, segura âncora e novos votos, e só
retoma quando outra observação separa as hipóteses. A regra final e seus limites estão descritos
nas subseções posteriores e no fechamento da seção 10.

### Evidência local e replay A/B do snapshot intermediário

> **Histórico, não resultado final:** o gate de 2547 testes, a tabela A/B e o custo de CPU abaixo
> pertencem ao snapshot anterior às correções finais de domínio da exposição, geometria
> `unregistered` e empate do resíduo. A comparação final foi repetida e está na seção 10. Estes
> números são mantidos para rastrear a evolução; não descrevem o head local atual.

- Cobertura focada depois das últimas mudanças de código: `tests/test_anchor_arbitration.py`,
  `tests/test_anchor_follow.py`, `tests/test_ink_registration.py`, `tests/test_ink_contracts.py`,
  `tests/test_motion_veto.py`, `tests/test_permute_prob.py` e
  `tests/test_ink_removal_registration.py`: **271 passaram**. Gate final: `ruff check .` passou,
  `ruff format --check .` passou (**221 arquivos**) e `pytest -q` terminou com **2547 passed,
  2 skipped** em 163,50 s na execução final desta worktree.
- Replay A/B no preset `producao`, mesmo corpus e mesmos pesos, comparando o worktree limpo de
  `0bf0177` com esta worktree. Os SHA-256 conferidos contra `assets/manifest.json` foram
  `draw_seg_v3_dyn.onnx` `5e2c611f…` e `page_clean_unet.onnx` `8359ff03…`.

  | Clipe | Quadros | Canvas confirmado base → atual | Provisório final base → atual | Retratado acumulado base → atual | CPU s/s base → atual* |
  |---|---:|---:|---:|---:|---:|
  | `agora-escuro-estatico` | 42 | 20.259 → 20.259 | 2.043 → 2.043 | 3.638 → 3.638 | 4,1646 → 4,6407 |
  | `e5-escuro-estatico` | 40 | 21.219 → 21.219 | 947 → 947 | 3.774 → 3.774 | 4,3426 → 4,8593 |
  | `escrita-folha-movida` | 316 | 36.851 → 36.851 | 63.768 → 63.768 | 215.229 → 215.229 | 3,5896 → 3,8079 |
  | `escrita-mao-sombra` | 232 | 48.689 → 48.556 (−133) | 48.220 → 55.352 (+7.132) | 114.900 → 124.135 (+9.235) | 2,9636 → 3,0808 |

  *A coluna de CPU nesta tabela é uma execução por versão e inclui o pipeline inteiro; não é
  benchmark isolado do helper. No clipe `escrita-folha-movida`, três execuções por versão deram
  mediana de 3,5667 CPU s/s na base (3,5536–3,5710) e 3,7915 na worktree atual (3,7720–3,8106),
  cerca de **6,3% a mais**. As três execuções preservaram os mesmos totais do canvas, passadas e
  retrações. Isso é um custo observado neste equipamento/corpus; não é uma estimativa
  universal nem um critério de aceitação de produto.
- Os replays orientados corretamente divergem no canvas final do clipe `escrita-mao-sombra`:
  **133 pixels existem só na máscara-base e zero só na atual** (IoU 0,99727), em componentes
  esparsos; a caixa na grade da página é `[38,450) × [53,651)`. O canvas final diminui de 48.689
  para 48.556 pixels. O replay não contém rótulo que diga se esses pixels eram tinta legítima ou
  tinta espúria.
- A comparação instrumentada no mesmo quadro de origem confirma a troca de decisão:
  `captured_at=28,5 s`, índice zero-based `228` do clipe (8 fps) e `pass_index=57`. A base marca
  `spurious` com `score_camera=0,164710`, `score_page=0,114558` e margem `0,050152`; o head atual
  marca `indeterminate` com `score_camera=0,091636`, `score_page=0,085170` e margem `0,006465`.
  A sequência atual reancora geometricamente nessa passada, com deslocamento de `8,963 px` e
  delta máximo da âncora de `15,082 px`; o veredito `spurious` da base segura esse `follow`. A
  instrumentação pareia por tempo de captura e índice de quadro, não só pelo contador de passadas.
  Isso confirma a mudança de decisão no mesmo instante observado, mas não prova qual decisão
  preserva melhor a tinta.
- Projetei a caixa dos pixels divergentes sobre os quadros-fonte. Em `f0150`, vários componentes
  maiores caem na área de traços visíveis da página esquerda, enquanto a mão escreve na direita;
  em `f0231`, a caixa projetada (aproximadamente x=134–685, y=339–918) sobrepõe a mão/caneta. A
  projeção tem incerteza geométrica: o quad atual difere até 5,12 px da âncora. Essa inspeção
  visual é evidência de localização, não rótulo de tinta. Os hashes dos quadros examinados são
  `f0230` `ae1e0dbb40c3e397a78e96d121cff9ab5fa2b89b193ffeae95f1e9452c8150be` e `f0231`
  `c9d6b69984f000e21fc22d918b407becc6cf5ea4da82b19ff1dacb6def177571`.
- Não há máscara independente para decidir se a redução é tinta removida corretamente ou tinta
  legítima perdida. O `clip.json` só descreve “mão e sombra se movendo sobre a página esquerda”; o
  vídeo termina com a mão sobre a região e não tem quadro posterior que a revele.
- `escrita-folha-movida` é um clipe real de 316 quadros a 8 fps, com folha em movimento e rodapé
  cortado, mas sem `conditions`, `expected`, regiões de tinta nem anotação de perda/retorno do
  tracker. `escrita-mao-sombra` também não tem rótulos ou regiões. Os dois clipes escuros têm
  `expected: no_ink`, e **ambas as versões falham** nesses veredictos: deixam 20.259 e 21.219 px
  no canvas, respectivamente. São falhas preexistentes neste A/B, iguais nas duas versões. Os
  demais totais não podem ser convertidos em recall, precisão, falsos positivos ou prova de
  ausência de duplicação.
- Os contratos sintéticos dão verdade exata para as sequências construídas e distinguem mutações
  do escore, do recorte e da aposentadoria do resíduo. Eles não medem taxa de falha na câmera,
  generalização a outras mãos/mesas, preservação de todo traço fino ou qualidade de produção.
  A regra de `docs/CORPUS.md` continua exigindo clipe rotulado para declarar correção de robustez.
- O contrato ponta a ponta de recuperação de resíduo agora exige mais que limpar as flags de hold:
  a tinta existente precisa permanecer alinhada, a telemetria precisa registrar a decisão de reter
  o resíduo e um traço novo precisa confirmar na posição certa sem cópia deslocada. Os dois casos
  focados (`production` e `page_model_guarded`) passaram. Continua sendo uma sequência sintética
  com verdade conhecida, não uma taxa de sucesso em câmera.

### Checagem geométrica complementar: SmartDoc-2015

- Rodei `datasheet001`, sample oficial com 214 quadros 1920×1080 e quadriláteros rotulados no XML.
  O laço registra a detecção consumida com um quadro de atraso; por isso comparei `det_quad` com o
  índice `det_frame` e o `quad` confirmado com o índice do registro. Em 109 detecções consumidas
  com quadrilátero, o erro médio por canto da detecção foi `4,49 px` (mediana `4,52`, p95 `8,18`,
  máximo `11,50`). O `quad` confirmado teve erro médio `74,01 px` (mediana `57,84`, p95 `181,72`,
  máximo `195,51`).
- Nos registros 56–115, o `quad` confirmado permaneceu idêntico por 60 quadros; o maior
  deslocamento de canto no XML entre os extremos foi `132,77 px`. No registro 70, a detecção
  consumida do quadro 69 acompanha o XML, enquanto o `quad` confirmado ainda mostra a posição
  anterior. A base e o head atual deram registros exatamente iguais nos 214 quadros; canvas
  `16.816 px`, retrações `377.457 px` e 51 passadas também são iguais. Assim, o sample não mede
  melhora do #180; ele identifica uma investigação separada sobre a confirmação de movimentos
  grandes no `SheetTracker`.
- A pausa longa em saltos grandes é deliberada no `SheetTracker`. O replay avança o tempo de cena
  virtualmente e executa o pipeline assíncrono em laboratório; os 60 quadros não equivalem a uma
  medição de latência numa captura ao vivo. A confirmar com reprodução em tempo real e, depois,
  câmera real antes de classificar como defeito ou mudar essa política.
- SmartDoc rotula a pose da página, não tinta manuscrita, saída da mão, perda/retorno do tracker,
  apagamento ou duplicação. Os três vídeos da amostra (214, 176 e 198 quadros) são documentos
  impressos. O adaptador temporário marcou `no_ink` para permitir a execução, mas esse rótulo não
  foi fornecido pelos mantenedores e não é ground truth de tinta. Os agregados de base/head são
  iguais nos três vídeos e não valem como precisão, recall ou taxa de falso positivo.
- A busca ampla via Firecrawl e Alexandria não encontrou um dataset público de escrita em vídeo
  que reúna máscaras independentes de tinta, visibilidade/saída da mão e perda/retorno do tracker.
  Um artigo de 2026 relata 13.507 quadros anotados, mas informa que os vídeos não podem ser
  publicados por privacidade e permissões; não é possível usá-los como holdout reproduzível
  ([artigo e disponibilidade dos dados](https://arxiv.org/html/2606.02342v1)).

### Pesquisa externa aplicada ao protocolo

- A documentação oficial do OpenCV especifica que `warpPerspective` recebe uma grade de saída
  com tamanho `dsize`, usa amostragem inversa salvo `WARP_INVERSE_MAP` e preenche fora da imagem
  segundo o `borderMode`. Isso sustenta o mecanismo do recorte por `dsize` e a necessidade de
  acolchoar a grade para medir massa projetada; não sustenta a função de escore, a regra de
  aposentadoria ou os limiares deste projeto ([transformações geométricas, OpenCV 4.x](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)).
- Karthik, Moudgil e Gandhi descrevem avaliação de rastreamento longo por redetecção,
  recuperação e continuidade/reliabilidade, em sequências com falhas e reaquisição; isso
  sustenta medir episódios de queda e retorno, não apenas sobreposição em quadros isolados.
  É uma orientação geral de avaliação de tracking, não valida o estimador deste projeto
  ([artigo](https://arxiv.org/abs/1910.12273)).
- ITTO discute avaliação de rastreamento de pontos em vídeo longo e evidencia por que frequência
  e deslocamento de reaparição importam. É análogo metodológico e não mede tinta, papel ou o
  comportamento de `PageCleanWorker` ([artigo](https://arxiv.org/abs/2510.19819)).
- MIDV-500 fornece vídeo de documentos e quadriláteros por quadro, útil para testar detecção e
  pose; não rotula persistência de tinta, duplicação ou apagamento. TAP-Vid fornece trajetórias
  e visibilidade de pontos, útil para instrumentar perda/retorno, mas também não substitui o
  corpus funcional do desenho ([MIDV-500](https://arxiv.org/abs/1807.05786),
  [TAP-Vid](https://arxiv.org/abs/2211.03726)).
- Com zero eventos em `n` ensaios Bernoulli independentes, o limite superior unilateral de 95%
  é aproximadamente `3/n`. Frames do mesmo clipe não são ensaios independentes; a unidade deve
  ser episódio/clipe, com sessões separadas e condições previamente definidas. A regra dos três
  não prova independência e não transforma um corpus pequeno em garantia de segurança
  ([Hanley & Lippman-Hand, 1983](https://pubmed.ncbi.nlm.nih.gov/6827763/)).

### Próxima validação que fecha a lacuna

1. Estender `escrita-mao-sombra` com quadros após a saída da mão e rotular a tinta visível numa
   máscara independente, incluindo os 133 pixels divergentes; acrescentar um clipe controlado
   com perda/retorno do tracker, resíduo mensurável, deslize posterior, escrita e apagamento.
   Rotular pose e visibilidade por quadro; episódios, não frames, são a unidade independente.
   Não havia dispositivo `/dev/video*` nesta sessão para gravar a extensão do clipe.
2. No holdout, medir se a classificação `spurious`/`indeterminate` perto de `_ARB_DELTA` e o
   `follow` subsequente preservam tinta e permitem apagar. Neste evento, a margem da base foi
   0,050152 e a atual 0,006465. Sem rótulos independentes, não há base para
   recalibrar o delta nem para alterar a regra geométrica.
3. O A/B final sem rótulos está registrado na seção 10 e atualiza o snapshot histórico acima.
   Depois de rotular os clipes, medir deslocamento falso/duplicação, recall de tinta, retenção de
   apagamento, abstenção e atraso até confirmação. Os replays medem diferenças entre versões,
   mas não provam melhoria funcional. Julgar o custo observado contra um orçamento de CPU
   definido pelo produto, não contra um limite inferido destes clipes.
4. Executar o CI remoto no head publicado. A implementação continua local, sem commit/push; por
   isso ainda não existe evidência de CI remoto.
5. Investigar o trecho congelado do `SheetTracker` separadamente: primeiro medir em replay com
   relógio de captura e processamento em tempo real; depois comparar com câmera. Só então decidir
   se o custo deliberado para saltos acima de 100 px precisa de ajuste.

### Fontes do dataset geométrico

- A página dos mantenedores descreve 120 vídeos, cerca de 24 mil quadros, documentos A4 em quatro
  fundos e quadriláteros por quadro. O registro Zenodo disponibiliza o sample sob CC BY 4.0. Isso
  justifica seu uso para pose da página, não para rótulos de tinta
  ([SmartDoc-2015, mantenedores](https://smartdoc.univ-lr.fr/smartdoc-2015-challenge-1/smartdoc-2015-challenge-1-dataset/),
  [registro Zenodo](https://zenodo.org/records/1230218)).

### Revisão remota atualizada — 2026-11-24

- O PR #180 continua aberto e mergeável no GitHub, mas o head remoto ainda é `93f1c28`; o status
  de Devin Review está pendente. A consulta atual da API lista 9 threads: 5 resolvidas e 4 ainda
  abertas. As abertas são a duplicação após deslize, a remoção bloqueada após retorno, a seleção
  de candidatos do resíduo e a massa dos escores cíclicos. Portanto, a resposta anterior aos dois
  primeiros comentários não equivale a resolução do review.
- O worktree atual já filtra candidatos antes da ordenação. Fortaleci
  `test_identity_residual_ranks_only_candidates_that_pass_acceptance`: o caso força um candidato
  reprovado com escore bruto maior e um candidato aprovado com escore menor; a estimativa precisa
  escolher exatamente `(0, -5)`. Isso prova a regra de seleção local, não sua calibração em vídeo.
- Para a rotação, cada escore cíclico agora usa como denominador a massa **depois de
  `_permute_prob` e antes da translação**, o mesmo domínio da hipótese medida. O contrato novo
  percorre arbitragem: a massa aumenta de `240` para `266,67`, a translação corta `0,83`, a fórmula
  de domínio misto daria `>1`, e o escore correto é `265,83 / 266,67 = 0,996875`.
- Os dois contratos direcionados passaram, assim como os arquivos completos
  `tests/test_page_model.py` e `tests/test_anchor_arbitration.py` (**215 passed, 1 skipped**).
  O contrato `test_residual_is_retired_after_a_confirmed_raw_page_landing` reproduz o exemplo do
  comentário de duplicação: retorno com residual `(0, -6)`, depois deslize físico adicional de
  12 px com warp bruto inalterado, seguido de pouso, retirada do residual, preservação do canvas e
  traço novo sem cópia deslocada. Em execução direcionada, esse contrato, os três casos de
  remoção após retorno, os três casos de deslize, a seleção de candidato e o escore cíclico deram
  **16 passed**. Essa é evidência sintética do caminho reproduzido; não confirma ocorrência ou
  frequência em câmera.
- Uma reprodução diferencial encontrou e corrigiu mais um defeito real na worktree: com 400 px
  de canvas, página bruta perfeitamente alinhada e mapa da câmera vazio, mas mensurável com
  denominador `400`, o `HEAD` limpo `0bf0177` zerava `score_page` por causa de `registered_total`
  e terminava em `indeterminate`. Com a mudança local, a página pontua `1,0`, a câmera pontua
  `0,0` e a arbitragem escolhe `landing`; `test_empty_registered_map_does_not_erase_valid_page_score`
  fixa esse resultado. No controle com denominador de câmera zero, a página ainda pontua `1,0`,
  mas o evento permanece `indeterminate`, pois a hipótese da câmera não é mensurável. O teste
  pareado passa no head corrigido e a reprodução isolada no `HEAD` limpo devolve `score_page=0`,
  `score_camera=0`, `indeterminate`.
- A busca local não encontrou arquivos `.mp4`, `.mov`, `.avi`, `.mkv` ou `.webm` em `draw-data`;
  os clipes de replay estão como sequências PNG. Elas permitem replay determinístico, mas não
  acrescentam captura após a saída da mão nem rótulos independentes de tinta. Também não havia
  dispositivo `/dev/video*` nesta sessão.
- As mudanças locais continuam sem commit/push. Assim, embora o gate completo local esteja verde
  (**2547 passed, 2 skipped**), o CI remoto segue no head antigo e as quatro threads abertas ainda
  não foram atualizadas nem resolvidas no GitHub.

### Correção do domínio da máscara fotométrica — 2026-11-24

- Uma reprodução independente encontrou outra incompatibilidade entre o escore de câmera e seu
  denominador. `camera_prob` é primeiro warpado com `INTER_LINEAR` e depois filtrado pela máscara
  `ExposureAssessment.visible` warpada com `INTER_NEAREST`; `_camera_probability_denominator`
  aplicava a mesma máscara à probabilidade antes do warp. Na fronteira subpixel, essas operações
  não conservam a mesma massa.
- O contrato `test_camera_denominator_applies_exposure_visibility_after_the_warp` fixa um caso
  mínimo: translação de 0,5 px, probabilidade em duas colunas contíguas e visibilidade fotométrica
  terminando entre elas. Antes da correção, o numerador de referência era **30** e o denominador
  **20**; o novo caminho mantém **30** nos dois. A reprodução foi executada primeiro contra o
  código anterior e falhou com `20 != 30`.
- `registration._sum_probability_warp_padded` agora aceita máscaras adicionais no domínio de
  saída. O denominador mantém `p.visible` como máscara de origem, transforma a máscara
  fotométrica pela mesma matriz do warp e a intersecta depois do warp. A exceção de tinta
  ancorada continua abrindo apenas o gate de cobertura; ela não reabre pixels negados pela
  exposição. A máscara fotométrica ausente (`None`) não altera o cálculo.
- Os contratos direcionados de warp e denominador passam após a mudança: **24 passed, 41
  deselected** em `tests/test_anchor_arbitration.py`. Esse resultado comprova a igualdade no caso
  reproduzido e preserva os casos existentes de massa cortada, cobertura, escala, limite de
  memória e exceção de âncora. O gate completo desta versão ainda precisa ser executado; não há
  CI remoto porque não houve push.

### Geometria da máscara no ramo `unregistered` e empate do resíduo — 2026-11-24

- No ramo de movimento `unregistered`, `camera_prob` conserva `camera_corners`, mas o mapa que
  vota passa a usar `registration_corners = anchor`. O numerador estava transformando
  `ExposureAssessment.visible` com `registration_corners`; o denominador usa a matriz de câmera
  original. Assim, a máscara do numerador e o mapa medido pelo denominador podiam ficar em grades
  diferentes. O worker agora transforma a exposição usando `camera_corners`, que é a geometria
  que produziu `camera_prob`; a máscara do voto continua usando `registration_corners`.
- `test_camera_exposure_mask_keeps_the_camera_hypothesis_geometry` fixa o estado que resulta da
  escolha `unregistered`: grade 100×100, quad de câmera deslocado 10 px, probabilidade em
  `[5:45, 40:50]` e exposição em `x < 50`. A máscara com a geometria errada zera o numerador;
  com a geometria da câmera, numerador e denominador são ambos **400**. É um contrato de estado
  sintético, não uma captura com o gate de movimento ativo.
- Outra reprodução state-level encontrou empate entre a hipótese de pouso com resíduo e sem
  resíduo. Barras horizontais a cada 6 px, resíduo `(0, -6)` e pose a 14 px davam
  `score_page=1`, `score_page_without_residual=1` e `score_camera=1/3`. O código anterior aceitava
  `landing`, aposentava o resíduo no `>=` e colocava um novo traço seis pixels fora da posição
  esperada. Não era ambiguidade entre pouso e câmera: era ambiguidade sobre qual correção de
  coordenadas preservava a página.
- A arbitragem agora abstém quando o candidato vencedor depende de duas leituras do resíduo cuja
  diferença absoluta é `<= _ARB_DELTA` (0,05). O mesmo critério cobre o candidato cíclico. O hold
  impede a troca de âncora e o voto de traços novos até que conteúdo observável separe as leituras;
  o resíduo só é aposentado quando a hipótese sem resíduo vence por mais que 0,05. Contratos
  pareados cobrem empate normal e cíclico, abstinência de traço novo e resolução posterior: ao
  aparecer a marca antiga distintiva, a arbitragem aceita o `landing`, mantém o resíduo e projeta
  um novo traço de `y=176:182` para `y=170:176`. É uma sequência state-level com estado de resíduo
  semeado, não uma captura integral. Esse limite 0,05 é a margem já existente da arbitragem,
  não um valor calibrado em corpus rotulado.
- Os quatro contratos novos direcionados passam (**4 passed, 64 deselected**). A suíte dos
  arquivos de arbitragem, contratos de tinta, PageModel e gate de movimento passa (**331 passed,
  1 skipped**). O gate completo final está registrado na seção 10. A revisão é sintética e
  state-level; não existe ainda
  uma captura pública reproduzível que percorra detecção do resíduo, observabilidade periódica,
  empate e pouso numa única sequência de câmera.

## 10. Fechamento local e evidência disponível — 2026-11-24

- O corpus externo passou por `python -m lab.corpus verify`. Rodei o preset `producao` de
  `lab/bench_ink.py` sequencialmente na base limpa `0bf0177` e na worktree final, com os mesmos
  clipes e pesos validados por `verify_model`. Nenhum dos oito resultados relatou erro.

  | Clipe | Quadros | Canvas base → final | Provisório mediano base → final | Retraído base → final | CPU s/s base → final* |
  |---|---:|---:|---:|---:|---:|
  | `agora-escuro-estatico` | 42 | 20.259 → 20.259 | 2.036 → 2.036 | 3.638 → 3.638 | 4,22 → 4,30 |
  | `e5-escuro-estatico` | 40 | 21.219 → 21.219 | 1.002 → 1.002 | 3.774 → 3.774 | 4,38 → 4,53 |
  | `escrita-folha-movida` | 316 | 36.851 → 36.851 | 61.793 → 61.793 | 215.229 → 215.229 | 3,66 → 3,91 |
  | `escrita-mao-sombra` | 232 | 48.689 → 48.556 (−133) | 42.868 → 42.868 | 114.900 → 124.135 (+9.235) | 3,00 → 3,18 |

  *Uma execução por versão, CPU-segundos por segundo de cena do pipeline completo. No clipe
  `escrita-folha-movida`, três execuções por versão deram medianas de **3,74 s/s** na base
  (3,68–3,80) e **3,90 s/s** na worktree final (3,80–3,93), diferença observada de **+4,3%**;
  as faixas se sobrepõem e o n é pequeno. Canvas, provisório e retrações foram iguais nas três
  execuções. Em `escrita-mao-sombra`, a diferença de 133 px e as 9.235 px adicionais retraídas
  não podem ser classificadas como acerto/erro sem rótulos independentes. Os dois clipes
  `escuro-estatico` continuam violando `no_ink` com os mesmos canvases grandes em ambas as
  versões. Esses dados confirmam diferença de comportamento e custo neste corpus, não melhora
  funcional geral.
- Teste do caminho real `unregistered`: o novo contrato chama
  `_pass_register_motion_coverage`, força a hipótese pelo `MotionGate` e verifica que o worker
  troca a geometria de voto para a âncora enquanto conserva `camera_corners` para pontuar a
  câmera; numerador e denominador ficam em 400 px. A mutação que usa `registration_corners` na
  máscara fotométrica fez o teste falhar com numerador **0**. Isso prova que o contrato discrimina
  o defeito de geometria no estágio do worker; o replay de câmera ainda não selecionou
  `unregistered`.
- Gate final nesta worktree após adicionar esse contrato: `ruff check .` passou,
  `ruff format --check .` confirmou **221 arquivos já formatados**, e `pytest -q` terminou com
  **2552 passed, 2 skipped em 160,52 s**. `git diff --check` também passou.
- Os testes mostram que os casos sintéticos reproduzidos deixam de inverter o denominador ou
  aposentar um resíduo ainda ambíguo. Não provam a taxa de ocorrência nem melhoria funcional em
  câmera. Não há rótulo independente por pixel, e o empate foi construído com o resíduo já
  semeado; não se percorreu nesse caso a sequência inteira de captura, estimação, ocultação
  periódica e retorno.
- Replay adicional da captura real `s4_escrita_1117_1800.drawcap`, com os modelos ONNX conferidos
  por `verify_model` e `PipelineConfig()` de produção: **900/900** frames decodificados, **215**
  passadas em 42,08 s offline. A guarda fotométrica forneceu máscara em 206 passadas. O gate de
  movimento registrou 75 `accept`, 100 `correct`, 40 `veto` e **0 `unregistered`**; não estimou
  resíduos. As 60 decisões de âncora foram 13 `spurious` e 47 `indeterminate`, sem `landing`.
  Isso prova que o replay completo roda com o código e que há uso real da máscara fotométrica;
  não exercita a correção `unregistered`, o empate sintético nem mede acerto de tinta. `pass_ms`
  foi 0,0 porque o replay injeta um relógio de cena síncrono; esse campo não mede tempo de CPU.
- Segundo replay integral: `s3_geometria_1117_1754.drawcap`, também com 900/900 quadros, 420
  passadas de tinta, 772 detecções e os mesmos modelos verificados e preset de produção. Na base
  limpa `0bf0177`, o gate de movimento contou 400 `accept` e 5 `veto`; na worktree final contou
  389 `accept`, 6 `correct` e 7 `veto`. Ambos fizeram três tentativas de estimar resíduo, todas
  `(0, 0)`, e nenhum escolheu `unregistered`; portanto essa captura não valida nem exercita a
  correção do ramo `unregistered` ou o tratamento de resíduo não nulo. As decisões de âncora
  mudaram de 33 `indeterminate`, 6 `spurious`, 1 `landing_default` e 1 `landing_rotated` na base
  para 12 `indeterminate` e 1 `landing_rotated` na worktree. Isso é uma diferença observada, não
  uma melhora demonstrada: não há rótulos independentes para classificar essas decisões.
- O canvas final desse replay tinha 37.014 px na base e 54.084 px na worktree. Comparando as
  máscaras sem alinhamento adicional, havia 392 px só na base e 17.462 só na worktree (IoU
  0,672). Esse IoU não mede qualidade: os quads finais também divergiram bastante (base:
  `[[110.634,479.063],[161.367,206.672],[629.475,92.452],[634.379,479.063]]`; worktree:
  `[[57.419,443.093],[276.183,125.557],[638.070,223.669],[638.750,510.014]]`), então as máscaras
  não estão na mesma coordenada física da página. A inspeção visual das projeções sobre o último
  quadro mostrou pixels vermelhos fora da escrita visível nos dois resultados; sem anotação por
  pixel e registro independente da folha, não é possível separar tinta recuperada, fantasma e
  erro de pose. Esse replay é uma evidência de integração e de mudança de estado, não de acurácia.
- A revisão das fontes técnicas foi feita com Firecrawl sobre a documentação oficial do OpenCV
  4.13: `warpPerspective` define a matriz, o tamanho da saída `dsize`, a interpolação e o modo/
  valor de borda; isso sustenta o mecanismo de recorte e o acolchoamento da grade, não a
  calibração do escore. A busca na biblioteca Alexandria não encontrou um provedor técnico
  específico para OpenCV/homografia; usei a documentação primária diretamente. Fonte:
  [OpenCV, Geometric Image Transformations](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html).
- GitHub foi consultado em leitura: PR #180 segue aberto, head remoto `93f1c28`, mergeável,
  com 4 threads de revisão ainda não resolvidas. A implementação e este relatório foram
  consolidados em commit local, ainda não enviado; portanto o CI remoto não cobre este código.
- Conferência do commit do relatório informado: `4d6f924` existe, mas a consulta local de
  `git branch --contains 4d6f924` retorna apenas `backup/work-180-d7ded1e`; a branch ativa
  `work/180-on-main`, baseada em `0bf0177`, não contém esse commit. Não movi refs nem reescrevi o
  histórico. O arquivo desta worktree inclui o conteúdo de pesquisa e as atualizações locais.
