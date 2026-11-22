# Nota de literatura — modelo persistente de página limpa (`PageModel`)

> **Estado:** nota de trabalho. Não faz parte da coleção numerada de propostas desta pasta.
> **Escopo:** levantamento de fontes primárias sobre os mecanismos do
> `src/draw/ink/page_model.py` (referência por pixel na grade da âncora,
> semeadura após N observações limpas, rastreio lento onde há papel,
> `novel` assimétrico, portão `admissible`). Só fontes primárias:
> papers, textos de patente, documentação oficial e código-fonte de
> primeira parte. Afirmações que não puderam ser confirmadas em fonte
> primária estão marcadas como **[não verificado]**.

## Introdução

A fila de robustez #129–#133 foi diagnosticada como um único defeito:
cada passada decide sozinha, sem memória do que é a página limpa.
O `PageModel` implementa o remédio clássico — uma imagem de referência
persistente da superfície, na grade retificada da âncora:

- `ref` (float32 por pixel): luminância de referência, semeada pixel a
  pixel quando o bloco de 32 px foi "visto" (≥50% visível) por
  `confirm_obs=3` passadas consecutivas; rastreada com `rate=0.125`
  apenas onde o chamador afirma papel (`worker.py` `pm.observe` passa
  `paper = visible & (prob <= _REMOVAL_LOW_PROB)`).
- `novel(gray)` = semeado **e** `ref - gray > novel_ratio · ref`
  (contraste de Weber, `novel_ratio=0,30` calibrado nas capturas
  reais): só o que **escureceu** contra a referência conta como tinta
  nova.
- `admissible(visible)` = `seeded & visible`: pixel nunca observado
  limpo ("pendente") não entrega evidência.

As perguntas desta nota: (1) quem fez isso antes e como; (2) a
convenção de atualização exponencial da referência; (3) alternativas
de invariância a iluminação; (4) a semântica "pendente até primeira
observação limpa" e se produtos comerciais resolvem o problema de
outra forma; (5) o que a documentação oficial diz sobre falha do
detector de mão em baixa luz; (6) trabalho adjacente.

---

## 1. Referência temporal de "superfície limpa" em captura de escrita por câmera

### 1.1 O sistema que o `page_model.py` cita — MSR RTWCS (fonte primária completa)

He, L.-W. & Zhang, Z., *Real-Time Whiteboard Capture and Processing
Using a Video Camera for Teleconferencing*, Microsoft Research
Technical Report **MSR-TR-2004-91**, set. 2004.
[PDF da Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2004-91.pdf) ·
[página da publicação](https://www.microsoft.com/en-us/research/publication/real-time-whiteboard-capture-and-processing-using-a-video-camera-for-teleconferencing/).
Versão de periódico (autoria inclui Zicheng Liu): He, Liu & Zhang,
*IEEE Transactions on Multimedia* 8(2), 2006,
[doi:10.1109/TMM.2006.886385](https://doi.org/10.1109/tmm.2006.886385).

O TR detalha exatamente o padrão que o `PageModel` implementa, com
escolhas de engenharia muito próximas:

- **Grade por blocos.** Cada quadro é dividido em *cells* de **16×16
  px** ("roughly the same as what we expect the size of a single
  character"). O `PageModel` usa blocos de **32 px** — a mesma ordem;
  a justificativa deles (granularidade de caractere vs. custo) é a
  mesma do nosso comentário "fino o bastante para a mão não
  contaminar, grosso o bastante para <1 ms". (TR §4.1)
- **"Visto por N frames" antes de entrar no modelo.** Cada célula tem
  um contador de **idade**: só células estacionárias por mais de
  **4 quadros (~0,5 s a 7,5 Hz)** viram candidatas a fundo. É o mesmo
  mecanismo do nosso `confirm_obs=3` em blocos (0,45–0,75 s). (TR §4.1.2)
- **Fundo = o que é mais claro.** A cor de fundo da célula é a média
  do **percentil 90** de luminância, porque "the ink absorbs the
  incident light, the luminance of the whiteboard pixels is higher
  than pen stroke pixels". A assimetria do nosso `novel` (só o mais
  escuro é conteúdo novo) é exatamente essa física. (TR §4.1.3)
- **Atualização só onde não é fundo/outlier.** As cores candidatas a
  fundo passam por um ajuste **least-median-of-squares de um plano
  global** que descarta células com cores discrepantes (as que só têm
  traço ou primeiro plano), e só as restantes alimentam o modelo —
  "we have to identify those cells to prevent them from contaminating
  the whiteboard color model". É o análogo do nosso `paper` mask: o
  chamador exclui evidência de tinta antes do rastreio. (TR §4.1.4)
- **Update exponencial (Kalman).** O modelo por célula segue
  `C_i = C_i + K·(O_i − C_i)` — a mesma forma do nosso
  `ref += rate·(obs − ref)`, com ganho `K` adaptativo em vez de
  constante. A incerteza `P_i` **começa em ∞ "to indicate no
  observation is available"**. (TR §4.1.4)
- **Região nunca observada não produz nada.** O modelo propaga cores
  para células ocluídas a partir dos vizinhos **com incerteza inflada**,
  "a hole of size N generally takes N/2 frames to get filled", e os
  valores preenchidos "are quickly supplanted by the observed values
  once they become available" — mas "the bookshelf area ... is never
  filled", e "if a region of the whiteboard is never exposed to the
  camera, our system will obviously not be able to figure out the
  content in that region". Nossa escolha (abster-se em vez de propagar)
  é a versão conservadora dessa mesma semântica. (TR §4.1.4, §5)
- **Classificação tríade.** Célula = fundo | primeiro plano
  (distância euclidiana ao modelo > 4 níveis de brilho) | traço novo.
  Filtro espacial: células de fundo adjacentes a primeiro plano viram
  primeiro plano — "delays the recognition of strokes by a few frames,
  but it prevents some parts of the foreground object from being
  classified as strokes — a far worse situation". Mesma prioridade do
  nosso desenho (falso negativo de tinta é barato; ocluidor lido como
  tinta é o bug #129). (TR §4.1.5–4.1.6)
- **Conteúdo impresso é referência.** No modo de captura de documento
  impresso (câmera apontada para baixo, suporte "gooseneck"), basta um
  alinhamento por homografia quadro-a-quadro — "only a minor
  algorithmic change was needed" — e o conteúdo já existente entra na
  captura inicial como estado base, não como "tinta nova". Sustenta a
  decisão #132 (pauta impressa pertence à referência). (TR §4.3)
- **Falha conhecida idêntica à nossa.** "If a person stands perfectly
  still in front of the whiteboard for an extended period, our system
  would not be able to determine that it is a person." Oclusão
  estática é o caso de falha documentado — o nosso `pending` é a
  resposta estrutural a ele. (TR §5)

### 1.2 A patente do mesmo sistema

**US7426297B2**, *System and method for real-time whiteboard capture
and processing*, inventores Zhengyou Zhang e Liwei He, cessionária
Microsoft; concedida em 16/09/2008, expirada por falta de anuidade
(registro de 2024). [Google Patents](https://patents.google.com/patent/US7426297B2/en).
A reivindicação 1 descreve literalmente o pipeline: "dividing each of
said images into cells of cell images; ... computing the background
color of the cell images which represent the color of blank
whiteboard; updating a whiteboard background model by integrating the
color of the blank whiteboard computed from the cell images;
classifying the cell images into foreground or whiteboard cells using
the whiteboard color model". O corpo da patente repete o TR
(células 16×16, percentil 90, plano least-median-squares, Kalman com
`P_i=∞` inicial, propagação a vizinhos com Δ de incerteza, refresh de
células a cada 30 s). Nada no `PageModel` excede o que essa patente
expirada descreve — e mesmo assim ela está **expirada**, o que é
relevante se um dia houver preocupação de IP.

### 1.3 "Whiteboard It!" — o projeto irmão de quadro único

Zhang & He, *Whiteboard It! Convert Whiteboard Content into an
Electronic Document*, MSR, ago. 2002.
[PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/Whiteboard-It.pdf) ·
[página](https://www.microsoft.com/en-us/research/publication/whiteboard-it).
**Atenção à diferença:** Whiteboard It! é pipeline de **foto única**
(localizar quadrilátero → retificar → white-balance do fundo →
"inkização" vetorial), sem modelo temporal — é o ancestral do Office
Lens, não do RTWCS. Serve como contraste: mostra o que a abordagem
sem memória consegue (limpeza estática) e o que não consegue
(oclusores, traços novos no tempo).

### 1.4 Logitech Scribe — o caso comercial

[logitech.com/scribe](https://www.logitech.com/en-gb/products/video-conferencing/room-solutions/scribe.html)
(página de produto, fonte primária para as alegações de marketing):
"Scribe's built-in AI delivers a transparency effect, allowing
participants to see 'through' the presenter for an unobstructed view
of the whiteboard"; "automatically enhances the color and contrast of
dry erase markers"; "uses image segmentation to recognize ... sticky
notes". O [white paper oficial](https://www.logitech.com/content/dam/logitech/en/video-collaboration/pdf/whitepaper-enabling-whiteboard-collaboration-with-logitech-scribe.pdf)
descreve: correção de perspectiva, remoção de brilho, realce de tinta
e "magically 'remove' the person at the board". O [Q&A oficial do
webinar](https://info.logitech.com/rs/201-WGH-889/images/062221_ScribeQandA.pdf)
afirma: "The AI is optimized for many different lighting scenarios,
including removing glare and minimizing shadows."
**O que a fonte primária não diz:** nenhum detalhe de algoritmo — a
"remoção do apresentador" é certamente um modelo temporal de conteúdo
a la RTWCS (a "transparência" só pode mostrar o que a câmera já viu),
mas isso é inferência nossa, não documentação: **[não verificado o
mecanismo interno]**.

### 1.5 CleanPage — burst alinhado + mediana por pixel

Courtney, J., *CleanPage: Fast and Clean Document and Whiteboard
Capture*, Journal of Imaging 6(10):102, 2020.
[PMC8321167](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321167/) ·
[doi:10.3390/jimaging6100102](https://doi.org/10.3390/jimaging6100102).
Em vez de um quadro, captura uma sequência curta; alinha por
features + homografia RANSAC; e **mescla por mediana por pixel ao
longo da sequência** em cada canal — a mediana temporal é o
estimador "limpo" que rejeita transientes. É a prova independente de
que "referência por pixel construída de múltiplas observações
registradas" é o mecanismo canônico de documento-por-câmera — no
nosso caso, contínua e com portões (visto/papel) em vez de burst
curto supervisionado.

---

## 2. Convenção de atualização do fundo: média corrente gated por "não-frente"

**A regra "atualizar devagar e só onde não é primeiro plano" é padrão
de fábrica da literatura de background subtraction.**

- **Média corrente exponencial.** Wren et al., *Pfinder: real-time
  tracking of the human body*, IEEE TPAMI 19(7):780–785, 1997,
  [doi:10.1109/34.598236](https://doi.org/10.1109/34.598236) — modelo
  por pixel de média/variância atualizado por média corrente
  (running average), o ancestral de `ref += rate·(obs − ref)`.
- **Stauffer & Grimson** formalizam com mistura de Gaussianas por
  pixel: pesos atualizados por `ω = (1−α)ω + α·M`, e "1/α defines the
  time constant which determines the speed at which the
  distribution's parameters change". Pixel que não bate em nenhuma
  Gaussiana de fundo "is considered foreground until there is a
  Gaussian that includes them with sufficient, consistent evidence" —
  ou seja, a evidência consistente precede a absorção, e o update de
  cada componente é gated pelo match. (C. Stauffer & W.E.L. Grimson,
  *Adaptive background mixture models for real-time tracking*, CVPR
  1999, [ieeexplore 784637](https://ieeexplore.ieee.org/document/784637/) /
  [PDF MIT](http://www.ai.mit.edu/projects/vsam/Publications/stauffer_cvpr98_track.pdf);
  aplicação em *Learning patterns of activity...*, IEEE TPAMI 22(8),
  [doi:10.1109/34.868677](https://doi.org/10.1109/34.868677).)
- **OpenCV MOG2 (código-fonte de primeira parte).**
  [`bgfg_gaussmix2.cpp`](https://github.com/opencv/opencv/blob/4.x/modules/video/src/bgfg_gaussmix2.cpp):
  - L.106: `static const int defaultHistory2 = 500; // Learning rate; alpha = 1/defaultHistory2`
    → α automático em regime ≈ **1/500 = 0,002 por quadro**.
  - L.916: `learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min(2*nframes, history)`
    → ramp-up inicial mais rápido, convergindo para 1/história.
  - Header [`background_segm.hpp`](https://github.com/opencv/opencv/blob/4.x/modules/video/include/opencv2/video/background_segm.hpp):
    `learningRate` "0 means that the background model is not updated
    at all, 1 means that the background model is completely
    reinitialized"; e a sobrecarga
    `apply(image, knownForegroundMask, fgmask, learningRate)` permite
    ao chamador **marcar primeiro plano conhecido para excluí-lo do
    update** — o equivalente API-oficial do nosso `paper` mask.
  - O parâmetro `fCT = 0.05` ("the number of samples needed to accept
    to prove the component exists") é o mecanismo de "N amostras
    consistentes antes de aceitar" dentro do algoritmo canônico.
- **A revisão Piccardi** fecha a convenção: "the background image
  must be a representation of the scene with no moving objects and
  must be kept regularly updated so as to adapt to the varying
  luminance conditions" — modelo limpo + atualização contínua são os
  dois pilares. (M. Piccardi, *Background subtraction techniques: a
  review*, IEEE SMC 2004,
  [doi:10.1109/icsmc.2004.1400815](https://doi.org/10.1109/icsmc.2004.1400815).)
- **Matsushita et al.** aplicam a mesma ideia à iluminação: decompõem
  a sequência em imagens de refletância (conteúdo) e de iluminação
  (sombra), "time-dependent", e normalizam cada quadro **dividindo
  pela estimativa de iluminação por pixel** — é a razão-imagem, não o
  delta absoluto, que elimina sombra. (Y. Matsushita, K. Nishino,
  K. Ikeuchi, M. Sakauchi, *Illumination Normalization with
  Time-dependent Intrinsic Images for Video Surveillance*, IEEE
  TPAMI 26(10):1336–1347, 2004,
  [doi:10.1109/TPAMI.2004.86](https://doi.org/10.1109/tpami.2004.86) ·
  PMID 17076411.)

**E os 0,125 por passada?** Em constante de tempo: ~8 passadas para
~63% de absorção ≈ **1,2–2,0 s** a 4–6,7 passadas/s. O MOG2 default
em regime é ~16,7 s (0,002 × 30 fps). Ou seja, nosso rate adapta
**~8–14× mais rápido** em tempo de relógio. É defensável — porque
(a) o update é gated por `paper` (o perigo do α alto é engolir
primeiro plano, o que a máscara impede por construção); (b) a
"passada" já é uma unidade temporal maior que o quadro bruto; (c) o
fenômeno alvo (nuvem/lâmpada, #133) é mais rápido que o "scene
change" da vigilância. Mas o **valor** 0,125 não tem respaldo
direto — a literatura sustenta o mecanismo (α pequeno gated por
não-frente, τ=1/α como constante de tempo), não a constante. RTWCS,
o caso mais próximo, usa ganho de Kalman adaptativo e refresh
explícito a cada 30 s.

---

## 3. Aparência de documento invariante a iluminação/sombra

### 3.1 Estimação de campo de shading (a alternativa "por imagem")

- **Jung, Hasan & Kim, *Water-Filling***, ACCV 2018,
  [arXiv:1904.09763](https://arxiv.org/abs/1904.09763): o documento
  vira uma superfície topográfica de **luminância**; um processo de
  "imersão" (equação de difusão iterativa) estima o mapa de shading;
  a reconstrução usa modelo Lambertiano. Razoável em sombras fracas/
  médias; segundo Wang & Chen (abaixo), degrada em sombra forte.
- **Wang & Chen, *Local Water-Filling***, Sensors 20(23):6929, 2020,
  [PMC7730344](https://pmc.ncbi.nlm.nih.gov/articles/PMC7730344/) ·
  [doi:10.3390/s20236929](https://doi.org/10.3390/s20236929):
  separa explicitamente **umbra e penumbra** — umbra = bacia, penumbra
  = crista entre platô e bacia; umbra tratada via Retinex, penumbra
  por "binarized water-filling". Ponto-chave para nós: "many shadow
  points belong to shadow boundaries and they are very similar with
  surrounding texts" — **a penumbra (sombra gradiente) é o caso
  difícil**, exatamente onde um delta absoluto em luminância crua
  confunde sombra com tinta.
- **Das et al., *Intrinsic Decomposition of Document Images
  In-the-Wild***, BMVC 2020, [arXiv:2011.14447](https://arxiv.org/abs/2011.14447):
  decomposição aprendida refletância/shading auto-supervisionada;
  como pré-processo de OCR, −26% de CER.
- **SD7K**: correção necessária — o dataset SD7K é de **Li, Chen, Pun
  & Cun**, *High-Resolution Document Shadow Removal via A Large-Scale
  Real-World Dataset and a Frequency-Aware Shadow Erasing Net*, ICCV
  2023, [arXiv:2308.14221](https://arxiv.org/abs/2308.14221)
  (7.620 pares reais 2462×3699; 6479/760 treino/teste;
  [repo oficial](https://github.com/CXH-Research/DocShadow-SD7K)).
  **Kligler et al. é outro trabalho** — *Document Enhancement Using
  Visibility Detection*, CVPR 2018,
  [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2018/html/Kligler_Document_Enhancement_Using_CVPR_2018_paper.html)
  (imagem→point cloud 3D→detecção de visibilidade; dataset próprio de
  ~300 imagens). A atribuição "SD7K de Kligler" está errada na ordem
  original desta pesquisa.

### 3.2 Razão (não delta) como teste de sombra — a evidência direta

- **OpenCV MOG2, `detectShadows`** (default true): "A shadow is
  detected if pixel is a darker version of the background. The shadow
  threshold (Tau)... Tau = 0.5 means that if a pixel is more than
  twice darker then it is not shadow"
  ([`background_segm.hpp`](https://github.com/opencv/opencv/blob/4.x/modules/video/include/opencv2/video/background_segm.hpp)).
  O teste canônico de sombra é **razão contra a referência**, não
  diferença absoluta — e ele já precisa da referência temporal que o
  `PageModel` mantém.
- **Prati, Mikic, Trivedi & Cucchiara**, *Detecting Moving Shadows:
  Algorithms and Evaluation*, IEEE TPAMI 25(7):918–923, 2003,
  [doi:10.1109/TPAMI.2003.1206520](https://doi.org/10.1109/tpami.2003.1206520)
  (o paper que o comentário do OpenCV cita): a família vencedora usa
  "similar chromaticity but lower brightness than the background
  model" — distorção de brilho (razão) + invariância de cromaticidade.
- **Homomórfico/Retinex**: a normalização por divisão
  (I / passa-baixa(I)) já existe no repo como
  `photometry.division_normalize`
  (`src/draw/photometry.py:31`, gaussiana de k ≈ 10% do lado, alvo
  235) e é aplicada ao warp em `worker.py:1214-1219` — é o membro
  homomórfico da família Retinex (Land & McCann, *Lightness and
  retinex theory*, JOSA 61(1), 1971,
  [doi:10.1364/JOSA.61.000001](https://doi.org/10.1364/JOSA.61.000001)).
- **Matsushita 2004** (acima): a normalização certa é I/L por pixel —
  razão contra a referência de iluminação.

### 3.3 Resposta para o `novel_delta`

- **Razão/delta vs. referência temporal é proxy reconhecido** — o
  RTWCS classifica por distância ao modelo (4 níveis de brilho), o
  MOG2 e Prati por razão, Matsushita por quociente.
- **Delta absoluto em luminância crua é a forma fraca.** Sob um
  **gradiente de sombra móvel** (penumbra varrendo a página), o
  pixel escurece suavemente e cruza qualquer delta fixo → falso
  "novel". A literatura de sombra em documento trata penumbra como
  entidade própria (Wang & Chen) e os detectores canônicos usam
  razão (MOG2 Tau, Prati) ou campo de shading (Jung, Das).
- **Caminhos com respaldo**: (a) testar `novel` em razão
  (`gray < ref·τ`, com τ tipo 0,5–0,8 como o Tau do MOG2) em vez de
  delta absoluto; (b) alimentar `observe`/`novel` com o warp já
  normalizado por `division_normalize`, movendo `novel_delta` para o
  domínio normalizado onde a sombra gradiente já foi parcialmente
  removida; (c) combinar os dois. Qual delas segura sombra gradiente
  sem engolir traco claro é **medida do corpus**, não decisão de
  literatura.

---

## 4. "Pendente até a primeira observação limpa"

### 4.1 Precedente direto

- **RTWCS**: `P_i` inicial = ∞ "no observation is available";
  células nunca observadas só recebem valor por propagação com
  incerteza inflada e são suplantadas pela observação real;
  inicialização de ~5 s inclui "initialize the whiteboard color
  model, and capture the content already existing" — existe uma fase
  de baseline explícita, e o conteúdo já presente entra como base
  (não como tinta nova). A nossa inovação relativa é tratar a
  ausência de baseline como **estado por pixel** (pendente) em vez de
  fase global — necessária porque no nosso cenário a mão cobre o
  caderno logo na abertura (#130). Ingredientes: idade>4 quadros
  (deles) + incerteza sem observação (deles) + "o que já estava lá é
  base" (deles) — combinados de forma que não encontramos verbatim na
  literatura, mas cada um existe separadamente.
- **MOG2 `fCT`**: componente precisa de "samples needed to accept to
  prove the component exists" — a ideia de "só aceita após evidência
  consistente" é nativa.
- **Piccardi** (acima): "a representation of the scene with no moving
  objects" — a premissa de baseline limpo é a definição mesma de
  background image.
- **Conclusão:** "pixel abstém-se até ser visto limpo N vezes" é uma
  composição nossa de peças todas documentadas; não encontramos na
  literatura de captura de documento/quadro um portão formal com esse
  nome, e a semântica exata "primeira observação limpa = o que estava
  lá vira página (incluindo pauta impressa)" é a nossa extensão do
  "content already existing" do RTWCS.

### 4.2 As canetas inteligentes contornam o problema — instrumentando

Verificação em páginas oficiais:

- **Livescribe (Anoto)**: a caneta tem câmera IR lendo o padrão de
  micropontos Anoto no papel — "the camera located at the tip of the
  smartpen takes a continuous series of ... dot photographs ... 75
  snapshots per second"; exige papel pontilhado.
  [FAQ oficial](https://www.livescribe.com/en-us/faq/online_help/Maps/Common/CRef_SP_About_c_about-livescribe-dot-paper.html).
- **Neo smartpen (NeoLab)**: idem, códigos "Ncode" — "An infrared
  camera under the Neo Smartpen nib reads the coordinates of a
  microscopic code (Ncode) printed on the paper. ... Unfortunately,
  regular notebook doesn't work".
  [shop.neosmartpen.com](https://shop.neosmartpen.com/pages/product-service).
- **Nuwa Pen**: "TRIDENT camera system ... three tiny cameras near
  the tip" — funciona em **qualquer papel**, mas a instrumentação está
  na **caneta** (e é o que eles anunciam: "the only pen that copies
  your handwriting from any paper without using a special surface or
  additional tool"). [nuwapen.com](https://nuwapen.com/pages/faq) e
  [suporte oficial](https://support.nuwapen.com/articles/9410927-how-does-the-nuwa-pen-work).
- **Equil Smartmarker**: receptor ultrassônico + **capa** ("sleeve")
  que veste marcador comum de quadro branco — instrumenta o
  instrumento, não a página.
  [brochura técnica](https://assets.touchboards.com/assets/1/26/SmartMarker_TechnicalBrochure.pdf).
- **Rocketbook**: escaneamento pós-hoc pelo app
  ([getrocketbook.com](https://getrocketbook.com/pages/how-rocketbook-works),
  [help](https://rocketbookhelp.zendesk.com/hc/en-us/articles/360002984014-How-to-Scan-with-the-Rocketbook-App))
  — não é captura em tempo real.

**Validação da afirmação:** "nenhum produto comercial faz
caneta-comum + papel-comum via câmera externa" se **sustenta com uma
ressalva**: a Logitech Scribe faz exatamente isso para **marcador de
quadro branco** (câmera externa, marcador comum, remoção de
apresentador por IA). Para **caneta esferográfica/lápis sobre papel
de caderno**, todos os produtos encontrados instrumentam a caneta
(Livescribe, Neo, Nuwa), o papel (dot pattern), ou o processo
(Rocketbook pós-hoc). O espaço "câmera no teto + caneta comum + papel
comum" é, até onde as fontes primárias alcançam, território de
pesquisa (RTWCS para documento impresso; ver §6).

---

## 5. Oclusão de mão em baixa luz

### 5.1 O que a fonte primária do MediaPipe documenta

*Model Card — MediaPipe Hands (Lite/Full), out. 2021*
([PDF oficial Google, storage.googleapis.com](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf)),
e a [página oficial do Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
(o `hand_landmarker.task` empacota detector de palma 192×192 +
regressor 224×224, ~30K imagens reais + sintéticas):

- **Uso pretendido**: "Predicting landmarks within the crop of
  prominently displayed hands in images or videos captured by a
  smartphone camera" — AR mobile. Nosso cenário (câmera de mesa,
  mão segurando caneta, escrita) já é adjacent-out-of-domain.
- **Fora de escopo, textualmente**: "Predicting hand landmarks with
  gloves or **occlusions. For example when the hand is holding
  objects**" — a mão segurando a caneta é declaradamente fora do
  escopo do modelo.
- **Iluminação**: não há requisito mínimo documentado; há a
  confissão oposta — "The model has **not been tested** in
  'in-the-wild' smartphone camera conditions, including low-end
  devices, **low light, motion blur** etc., that can affect
  performance" e "its quality can degrade in extreme conditions".
- **Válido apenas com score**: landmarks "should only be considered
  valid when the presence score is higher than a threshold".

**Conclusão para o #129:** o próprio Google não garante o detector
sob baixa luz nem sob oclusão por caneta — tratar a máscara de mão
por quadro como testemunha falível e congelar/registrar em vez de
apagar é a resposta correta, e é o que `admissible`/`pending` fazem
estruturalmente.

### 5.2 O que os sistemas de captura fazem quando o detector de oclusor falha

- **RTWCS**: nenhum detector de oclusor dedicado — a oclusão é
  *inferida* do próprio modelo (célula difere do fundo e não casa
  como traço → primeiro plano). Células de primeiro plano **não são
  processadas** (a evidência anterior fica) e o depósito de saída
  **retém** o conteúdo já visto — por isso o participante remoto "can
  see the contents behind the presenter" (TR §5). A retenção é a
  regra; o caso de falha documentado (pessoa imóvel) é aceito como
  limitação, não corrigido por histerese — eles preferem dilatar a
  região de primeiro plano a risco de "classify foreground as
  strokes — a far worse situation".
- Ou seja: a literatura não oferece "histerese temporal de máscara de
  ocluidor" como técnica nomeada; oferece **o modelo persistente
  como amortecedor** — o que reforça que a correção do #129 é o
  `PageModel`, não mais uma guarda na máscara por quadro.

---

## 6. Trabalho adjacente

- **Trajetória de escrita a partir de imagem de câmera**:
  *inkGround*, [arXiv:2610.38720](https://arxiv.org/abs/2610.38720)
  (out. 2026): benchmark de recuperação de trajetória de traços a
  partir de **páginas capturadas por câmera** (537 imagens, ~163 mil
  traços) com pseudo-rótulos via modelo offline→online; mede VLMs.
  Adjacente: trabalha com imagem final de página fotografada, não
  vídeo — mostra que "página via câmera" virou benchmark recente.
- **Vídeo de cima para análise de escrita**: *Detecting Pen-In-Air
  States from Video*, [arXiv:2606.02342](https://arxiv.org/abs/2606.02342)
  (2026): prova de conceito de **vídeo top-view** de escrita,
  tracking de ponta de caneta (YOLO) + features cinemáticas,
  F₂ até 0,805 para estados pen-up — a direção "câmera sobre papel"
  está ativa, ainda em escala piloto.
- **O ancestral de tudo**: Wellner, *Interacting with paper on the
  DigitalDesk*, CACM 36(7):87–96, 1993,
  [doi:10.1145/159544.159630](https://doi.org/10.1145/159544.159630)
  — câmera + projetor sobre a mesa, lê documentos e responde a
  caneta/dedo; o "digitizer desk" como conceito é de 1993.
- **Remoção de pauta/linha impressa** (relevante para #132):
  - Dalitz, Droettboom, Pranzas & Fujinaga, *A Comparative Study of
    Staff Removal Algorithms*, IEEE TPAMI 30(5):753–766, 2008,
    [doi:10.1109/TPAMI.2007.70749](https://doi.org/10.1109/tpami.2007.70749)
    — remover linhas de pauta de partituras é um **subcampo inteiro**
    (o problema de separar traço sobreposto à linha é o cerne).
  - Chen, Lopresti & Nagy, *Conservative preprocessing of document
    images*, IJDAR 19(4), 2016,
    [doi:10.1007/s10032-016-0273-3](https://doi.org/10.1007/s10032-016-0273-3)
    — evidência direta a favor da nossa decisão: **manter** as linhas
    de pauta detectadas em vez de removê-las (e depois remendar traços
    quebrados) reduziu o erro de identificação de escritor em 4,5%;
    o preprocessamento "conservador" ganha por não destruir
    informação.
  - *Ruling Line Removal in Handwritten Page Images*, ICPR 2010,
    [doi:10.1109/icpr.2010.663](https://doi.org/10.1109/icpr.2010.663)
    — o problema existe como literatura própria exatamente porque
    remover linha impressa sem quebrar traço é difícil.
  - **Leitura:** tratar conteúdo impresso como pertencente à
    referência (o que a primeira observação limpa viu é página)
    **contorna** um problema que a literatura confirma ser difícil e
    destrutivo quando atacado por remoção — decisão sólida.

---

## O que a literatura sustenta / o que fica por nossa conta

| Decisão do `PageModel` | O que a literatura sustenta | O que é nosso medir |
|---|---|---|
| Referência persistente por pixel na grade retificada | Canônico: RTWCS (Kalman por célula), CleanPage (mediana por pixel em burst alinhado), toda a família BG-subtraction (Piccardi, Stauffer-Grimson, MOG2). | — |
| Semear só após `confirm_obs=3` observações vistas | Idêntico ao "age > 4 quadros" do RTWCS (~0,5 s) e ao `fCT` do MOG2; a janela de 0,45–0,75 s é a mesma ordem. | Confirmar no corpus que 3 cobre "mão parada breve" sem atrasar demais a primeira tinta. |
| Pixel pendente não entrega evidência | "Nunca observado" é `P_i=∞` no RTWCS; região nunca exposta "obviously" sem conteúdo; abstinência > propagação (nós somos os conservadores). | Telemetria de `seeded_fraction`: quanto da folha fica pendente na prática e por quanto tempo. |
| `rate=0.125` gated por `paper` | O mecanismo é padrão (running average gated por não-frente: Wren, MOG2 `knownForegroundMask`, RTWCS outlier-gated Kalman). | O **valor**: ~1,6 s de constante de tempo é ~10× mais rápido que o default MOG2; defensável pelos portões, mas a deriva-alvo (lâmpada/nuvem) vs. jitter de exposição precisa ser medida no corpus. |
| `novel` assimétrico (só mais escuro) | Mesma física do RTWCS ("ink absorbs light" → fundo = percentil 90). Apagar/revelar papel clarear e não contar é corolário seguro. | Confirmar que canetas claras (grafite claro, marca-texto) ainda cruzam o delta no corpus. |
| `novel_delta=24` absoluto em luminância crua | **Ponto fraco frente à literatura**: sombra é testada por razão (MOG2 Tau=0,5; Prati "lower brightness, similar chromaticity") ou por campo de shading (Jung, Das, Wang & Chen — penumbra = caso difícil explícito). Delta absoluto cruza penumbra móvel como tinta. | Decidir entre delta absoluto, razão (`gray < ref·τ`), ou `novel` sobre warp já `division_normalize` — calibração com clipes reais de sombra gradiente (#133). |
| `block=32` px | Mesma ordem dos 16×16 do RTWCS ("tamanho de um caractere"). | Medir vazamento na aresta mão/papel e custo real. |
| Conteúdo impresso na referência (#132) | Forte: RTWCS captura conteúdo existente como base; Chen/Lopresti/Nagy — manter pauta > remover; Dalitz — remoção é campo difícil. | — |
| "Freeze por falta de testemunha" p/ mão em baixa luz (#129) | Forte: MediaPipe declara oclusão+objeto fora de escopo e baixa luz não testada; RTWCS retém último estado sob oclusão. | Medir tempo de recuperação quando a mão sai em luz ruim. |
| "Nenhum produto comercial faz caneta+papel via câmera" | Sustentado **com ressalva**: Logitech Scribe faz isso para marcador de quadro; para caneta/papel todos instrumentam a caneta (Livescribe, Neo, Nuwa) ou o papel. | — |

### Itens marcados como não verificados

- Mecanismo interno de "remoção do apresentador" da Logitech Scribe —
  alegação de marketing sem algoritmo publicado; a inferência "modelo
  temporal de conteúdo" é nossa.
- Não foi localizada, na literatura de captura de
  quadro/documento/matting, uma técnica nomeada de "histerese de
  máscara de ocluidor" distinta do modelo de fundo persistente — o
  amortecedor documentado é o próprio modelo, como implementamos.
- A semântica exata "primeira observação limpa congela o conteúdo
  presente como página" (incluindo pauta/régua) não aparece verbatim;
  é composição nossa de "P_i=∞ até observar" + "content already
  existing" do RTWCS.
