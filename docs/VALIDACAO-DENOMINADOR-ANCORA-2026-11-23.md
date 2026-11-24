# Validação do escore de arbitragem da âncora (#180) — síntese de pesquisa

Data: 2026-11-23. Branch: `work/180-on-main`. Condensa dois relatórios de pesquisa
(duas rodadas: fontes secundárias e depois fontes primárias com Firecrawl e experimentos
numéricos fora do repositório). Este documento é só pesquisa e recomendações; não altera código.

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
