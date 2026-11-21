# Continuidade de identidade de páginas em sessões de desenho

## Resumo

Manter a identidade de uma página ao longo de uma sessão é diferente de perceber que uma folha está presente em cada observação. A página pode permanecer, deslocar-se, sair temporariamente da cena e retornar; outras folhas podem ser visualmente semelhantes ou diferentes. Decisões inadequadas podem associar desenhos de páginas distintas ou interromper a continuidade da mesma página. Este artigo propõe uma taxonomia de eventos para avaliar essas decisões em um sistema de desenho assistido por câmera. A classificação separa seis eventos observáveis, três decisões possíveis — continuar, criar uma identidade distinta ou adiar — e as consequências de fusão e separação indevidas. O protocolo usa sequências anotadas, variações qualitativas na duração da ausência e métricas de reidentificação correta, fusão, separação e abstenção. A avaliação mantém a implementação como caixa-preta e compara decisões visíveis com referência independente. Tabelas e diagramas definem os eventos e custos, sem apresentar desempenho medido. Como não há resultados experimentais, o artigo não afirma que qualquer decisão seja acertada em um sistema existente nem descreve o comportamento interno. A proposta oferece uma base para coletar e relatar continuidade, incerteza e erros de identidade em sessões de desenho.

**Palavras-chave:** identidade de página; rastreamento; continuidade; incerteza; avaliação

## Abstract

Maintaining a page’s identity throughout a session differs from detecting that a sheet is present in each observation. A page may remain, move, leave the scene temporarily, and return; other sheets may look similar or different. Inadequate decisions can associate drawings from distinct pages or interrupt continuity for the same page. This article proposes an event taxonomy for evaluating these decisions in a camera-assisted drawing system. The classification separates six observable events, three possible decisions—continue, create a distinct identity, or defer—and the consequences of incorrect merging and splitting. The protocol uses annotated sequences, qualitative variations in absence duration, and measures for correct re-identification, merging, splitting, and abstention. Evaluation treats the implementation as a black box and compares visible decisions with an independent reference. Tables and diagrams define events and costs without reporting measured performance. Since no experimental results are available, the article does not claim that any decision is correct in an existing system or define a waiting time. The proposal offers a basis for collecting and reporting continuity, uncertainty, and identity errors in drawing sessions.

**Keywords:** page identity; tracking; continuity; uncertainty; evaluation

## 1. Introdução

Uma sessão de desenho pode envolver uma única folha que se move ou várias folhas parecidas usadas em sequência. A câmera pode deixar de observar a página por causa de deslocamento, enquadramento, interrupção da atividade ou troca de suporte. Quando uma folha volta a aparecer, uma decisão precisa preservar a continuidade quando se trata da mesma página e evitar misturar conteúdo quando é outra. A situação parece simples enquanto a página está sempre visível, mas torna-se ambígua diante de ausência e semelhança visual.

É útil distinguir detecção da página e manutenção de identidade. A primeira pergunta se há uma folha observável em determinado momento. A segunda relaciona observações ao longo da sessão e decide se pertencem ao mesmo objeto de trabalho. Uma avaliação que conte apenas presença pode classificar como sucesso uma sequência que associa marcas de folhas diferentes. Outra que privilegie cautela pode separar repetidamente observações da mesma página, fragmentando a história do desenho. Ambos os erros têm custos distintos para continuidade, integridade e compreensão do conteúdo.

O presente artigo trata o problema como avaliação externa de um sistema de desenho assistido por câmera. São registrados eventos observáveis e decisões comunicadas ou inferíveis a partir da saída visível. Não se descreve como a identidade é calculada, qual evidência interna a sustenta ou por quanto tempo uma decisão é adiada. A pergunta é: **como caracterizar a continuidade de identidade de páginas diante de deslocamento, ausência, retorno e aparição de folhas visualmente semelhantes?** A pergunta inclui tanto acerto quanto incerteza, pois adiar uma decisão pode ser preferível a fundir páginas distintas, dependendo da tarefa.

O artigo oferece três contribuições. Primeiro, transforma a expressão genérica “taxonomia de eventos” em uma classificação explícita com seis eventos, decisões possíveis e consequências. Segundo, separa fusão indevida, separação indevida, reidentificação correta e decisão adiada. Terceiro, apresenta um protocolo de sequências anotadas que relaciona esses desfechos à duração observável de ausência, preservando a implementação como caixa-preta. Essas contribuições definem o que observar; não constituem evidência de que um sistema atual resolva os eventos.

A seção 2 revisa rastreamento, reidentificação, recuperação visual e decisão sob incerteza. A seção 3 apresenta a taxonomia e explicita os custos de continuar, criar uma identidade distinta ou adiar. A seção 4 descreve sequências, anotação, métricas e análise. A seção 5 reserva estruturas para resultados futuros. A seção 6 discute interpretações possíveis e relação com a literatura. A seção 7 delimita validade e dados, e a seção 8 conclui indicando a evidência necessária.

A unidade principal é uma transição entre observações anotadas dentro de uma sessão. Cada transição recebe uma descrição do estado anterior, do evento visível e da relação de identidade conhecida pela referência. A referência pode ser definida por marcação física discreta, anotação da sessão ou outro procedimento independente, desde que não use a resposta que está sendo avaliada como verdade. Os avaliadores registram casos ambíguos em vez de decidir retrospectivamente a favor de uma interpretação conveniente.

A relevância da questão não se limita a um registro técnico. Se duas folhas forem unidas, marcas podem aparecer sob o contexto errado e o usuário pode não conseguir recuperar qual desenho pertence a qual suporte. Se a mesma folha for tratada como nova a cada retorno, o histórico se fragmenta e a tarefa de retomada pode ficar confusa. A decisão adiada também tem custo: pode exigir que a pessoa aguarde ou escolha uma ação adicional. Esses impactos devem ser examinados sem presumir que todos os erros tenham a mesma gravidade.

Páginas visualmente semelhantes constituem um caso importante, mas não o único. Duas folhas em branco podem parecer iguais; dois desenhos podem compartilhar padrões; uma página diferente pode conter parte do mesmo conteúdo se houve cópia. A aparência sozinha não define identidade física. Por isso, o protocolo distingue verdade de referência e sinais observáveis, e limita a conclusão ao modo como o estudo estabeleceu essa verdade. A classificação oferece uma linguagem de avaliação, não uma teoria universal de identidade.

## 2. Trabalhos relacionados

### 2.1 Rastreamento e reidentificação

Rastreamento de objetos relaciona observações de um mesmo alvo ao longo de uma sequência. Em vídeos com múltiplos objetos, a tarefa inclui detecção, associação entre quadros e tratamento de entradas ou saídas do campo visual. A literatura distingue erros de localização e erros de identidade, pois um alvo pode permanecer visível enquanto sua associação muda. Métricas de rastreamento multiobjeto medem aspectos diferentes: correspondência correta, fragmentação da trajetória, associação incorreta e consistência de identidade [1–6]. Essas distinções são úteis para avaliar folhas, desde que a unidade e a verdade de referência sejam adaptadas a uma sessão de desenho.

Reidentificação visual costuma perguntar se um objeto reaparece em outra imagem ou após uma interrupção. Estudos em pessoas, veículos e objetos de cena oferecem estratégias de criação de conjuntos de dados, anotação de trajetórias e comparação de ocorrências [7–12]. O domínio da página difere: papel pode ser pouco texturizado, folhas podem ser idênticas e o conteúdo desenhado muda durante o uso. Portanto, resultados de pessoas ou veículos não informam diretamente a dificuldade de reidentificar uma página. O valor transferível é conceitual: explicitar quando a trajetória foi interrompida, quais instâncias poderiam ser confundidas e como erros de associação são contados.

Trabalhos sobre rastreamento também tratam oclusões, reaparecimento, mudança de pose, movimento de câmera e entrada de novos alvos. Nem todo desaparecimento significa que o objeto deixou de existir; pode ter sido apenas ocultado ou perdido pelo enquadramento. Uma sequência anotada deve registrar essa diferença sempre que a referência permita. Na tarefa de desenho, a folha pode ser removida e trazida de volta, mas um corte na gravação ou uma página momentaneamente coberta também cria um intervalo sem observação. O protocolo mantém essas causas separadas quando conhecidas e registra “causa desconhecida” quando não há evidência.

### 2.2 Recuperação de páginas e imagens

Recuperação de imagens busca localizar conteúdo visualmente relacionado em uma coleção. Sistemas de recuperação de documentos e páginas manuscritas estudam semelhança de layout, texto, marcas e regiões. Esse campo contribui com formas de formar pares positivos e negativos e com a necessidade de testar conteúdo repetido. Ainda assim, recuperar uma imagem semelhante não equivale a confirmar que a mesma folha física retornou. Uma cópia, uma página reimpressa ou dois suportes parecidos podem compartilhar conteúdo sem serem o mesmo objeto. A referência física precisa ser definida independentemente [21–25].

Reconhecimento de lugar e localização visual abordam o retorno a uma cena após mudanças de ponto de vista, iluminação ou tempo. Esses problemas ajudam a pensar em aparências que variam para um mesmo local e em ambientes distintos que parecem iguais. A página muda de pose e conteúdo, mas mantém uma relação física contínua com o suporte. A comparação com localização visual serve para organizar variação e ambiguidade, não para importar uma métrica sem adaptação.

Benchmarks de documentos fotografados e coleções de imagens de páginas mostram como sequências e anotação podem apoiar comparação. Os conjuntos variam em número de páginas, tipo de escrita, repetição de conteúdo, captura e referência disponível. A análise precisa declarar se a identidade se refere à folha física, ao documento lógico ou à versão do desenho. Neste artigo, identidade designa o suporte físico usado na sessão; conteúdo é a informação desenhada sobre ele. Essa escolha impede que cópias idênticas sejam automaticamente consideradas a mesma folha.

### 2.3 Decisão sob incerteza

Decisões de classificação podem permitir uma opção de abstenção quando evidência insuficiente não sustenta uma escolha. Estudos sobre classificação seletiva, calibração e decisão com custos assimétricos mostram por que acertar, errar e não decidir são desfechos diferentes. Em uma interação, adiar pode conservar segurança, mas interromper o fluxo ou transferir esforço ao usuário. A literatura oferece modelos para comparar custo de erro e cobertura da decisão; a aplicação ao retorno de uma página exige definir o custo de cada consequência no contexto da tarefa.

Gestão de incerteza também depende da unidade temporal. Uma decisão feita em uma observação pode ser revista quando há novas evidências, enquanto uma escolha apresentada como final pode afetar conteúdo subsequente. A avaliação registra momento do evento e decisão visível, mas não infere regras internas. Se a interface mostrar que o estado está pendente, isso pode ser contado como abstenção; se não houver sinal observável, o estudo descreve apenas o comportamento que pode ser anotado.

A lacuna é integrar a taxonomia de eventos de uma página com decisões e custos explícitos em uma sessão de desenho. Rastreamento fornece noções de fragmentação e associação; recuperação de imagens aborda semelhança; estudos de decisão tratam incerteza e abstenção. Nenhuma dessas linhas, isoladamente, define a diferença entre fundir duas folhas e separar a mesma folha na tarefa proposta. O artigo combina esses componentes para permitir comparação sem assumir como a identidade é calculada.

Métricas de identidade têm compromissos distintos. Uma associação pode ser correta por quadro e ainda fragmentar uma trajetória longa; uma contagem de trajetórias pode ocultar períodos curtos de fusão. Benchmarks multiobjeto e estudos de associação temporal mostram por que correspondência, continuidade e troca de identidade precisam de definições explícitas [4–13]. Para páginas, o denominador deve excluir intervalos sem referência física e informar a frequência de cada evento, pois as sequências raras podem dominar a incerteza de uma métrica agregada.

Trabalhos de recuperação de conteúdo e de reidentificação de pessoas oferecem analogias úteis, mas seus alvos diferem do suporte em papel. A aparência de uma pessoa pode mudar com roupa, pose e câmera; uma página muda com o desenho, dobra, iluminação e posição. Estudos de características locais e correspondência espacial ajudam a contextualizar variação visual, enquanto pesquisas de duplicatas documentais lembram que similaridade não equivale a identidade [14–26]. O protocolo precisa declarar se mede o mesmo suporte físico ou o mesmo conteúdo lógico.

## 3. Enquadramento do problema

Identidade é definida aqui como continuidade do mesmo suporte físico ao longo da sessão. A aparência e o conteúdo podem variar sem que a folha deixe de ser a mesma; de modo inverso, outra folha pode parecer muito semelhante. O protocolo mantém a anotação de identidade separada da decisão avaliada. Assim, a referência informa qual suporte está presente, enquanto a resposta externa indica se a continuidade foi preservada, uma nova identidade foi iniciada ou a decisão ficou pendente.

A taxonomia contém seis eventos. **Permanece** descreve a página visível sem transição relevante na identidade física. **Desloca** registra mudança de posição ou pose enquanto a mesma folha permanece na sessão. **Some** indica que a folha deixa de ser observada por um intervalo, sem concluir que foi descartada. **Volta** é a reaparição da mesma folha física após ausência. **Aparece outra parecida** descreve a entrada de um suporte distinto com aparência semelhante à página anterior. **Aparece outra diferente** descreve a entrada de um suporte visualmente distinguível. A causa de ausência e o grau de semelhança são anotados separadamente quando a referência permite.

As decisões possíveis são **continuar**, **nova** e **adiar**. Continuar mantém a relação com a identidade anterior. Nova inicia uma identidade distinta. Adiar registra que a evidência observável foi considerada insuficiente para escolher entre as duas relações. Esses rótulos descrevem decisão externa; não revelam como ela foi produzida. Quando a interface não explicita a decisão, avaliadores podem derivá-la da saída visível apenas se a regra de anotação estiver definida e puder ser aplicada sem reconstruir detalhes internos.

Uma decisão correta depende do evento e da referência. Para uma página que permanece ou se desloca, continuar preserva a trajetória. Após desaparecimento e retorno da mesma folha, continuar é uma reidentificação correta. Se aparece outra folha, iniciar uma identidade nova evita fusão. Uma decisão adiada não é automaticamente correta nem errada: ela representa abstenção, cujo custo precisa ser relatado. Na análise, abstenções são separadas dos erros de fusão e separação e também se examina se aparecem concentradas em eventos difíceis.

**Tabela 1.** Taxonomia de eventos de página, decisões possíveis e consequências de continuar, criar uma identidade distinta ou adiar uma decisão incerta, considerando a referência física independente da saída observada na sessão.

| Evento de referência | Continuar | Nova identidade | Adiar |
|---|---|---|---|
| Permanece | Continuidade correta | Separação indevida da mesma folha | Abstenção sem necessidade aparente |
| Desloca | Continuidade correta | Fragmentação da trajetória da mesma folha | Interrupção temporária para análise |
| Some | Estado de continuidade não observável durante a ausência | Nova identidade prematura se a folha retorna | Abstenção até nova observação |
| Volta a mesma folha | Reidentificação correta | Separação indevida da mesma folha | Continuidade não resolvida |
| Aparece outra parecida | Fusão indevida de folhas distintas | Nova identidade correta | Abstenção diante de semelhança |
| Aparece outra diferente | Fusão indevida de folhas distintas | Nova identidade correta | Abstenção evitável se distinção é clara |

O custo é assimétrico e depende da aplicação. Em uma sessão que preserva cada página como unidade de trabalho, uma fusão pode atribuir marcas ao suporte errado; em uma tarefa que espera continuidade, uma separação pode exigir recuperação manual. A equipe declara qual consequência tem prioridade e mantém as métricas independentes para que outro estudo possa aplicar custos diferentes. Nenhuma regra universal transforma os dois erros em uma única penalidade.

Uma **fusão indevida** ocorre quando observações de folhas físicas distintas são associadas a uma única identidade. Seu custo pode incluir mistura de marcas, atribuição errada de histórico e confusão para retomar a sessão. Uma **separação indevida** ocorre quando a mesma folha recebe identidades distintas após deslocamento ou retorno. Seu custo inclui fragmentação do histórico e perda de continuidade. Nenhum desses erros deve ser resumido apenas como “identidade incorreta”, pois suas consequências e frequências podem diferir.

A taxonomia também distingue o evento incerto da decisão adiada. A anotação de referência pode estar ambígua porque a gravação não permite saber se a folha voltou ou se outra semelhante entrou. Nesse caso, a transição é marcada como incerta e não se atribui decisão correta ou errada sem evidência. Já a decisão adiada é uma resposta observada diante de um evento cuja identidade de referência é conhecida. Essa diferença evita tratar incerteza do avaliador como abstenção do sistema.

Similaridade visual é graduada de forma qualitativa e separada da identidade física. As categorias “parecida” e “diferente” devem ter exemplos no manual, mas não pressupõem medida interna de semelhança. Se a equipe não conseguir distinguir as classes de modo consistente, pode relatar aparência por anotação independente e fazer análise de sensibilidade. A categoria não se baseia em conteúdo textual confidencial nem em características internas não visíveis ao pesquisador.

A decisão pode ter consequências além do rótulo. Uma fusão pode acrescentar marcas de uma página ao histórico de outra, enquanto uma separação pode exigir que a pessoa recupere manualmente o desenho anterior. A avaliação anota efeitos observáveis no conteúdo ou na interação, mas não presume que todo erro provoque perda irreversível. O custo depende do que a sessão permite corrigir, do contexto de uso e da expectativa da pessoa. Esses efeitos são medidos como desfechos secundários, sem substituir a classificação de identidade.

A Figura 1 representa uma sessão com permanência, deslocamento, ausência, aparição de página semelhante e retorno. A ordem no esquema é apenas ilustrativa; eventos reais podem ocorrer em outra sequência. A Figura 2 mostra eventos, decisões e classes de consequência sem indicar cálculo interno. A Figura 3 reserva uma forma ilustrativa para examinar erro segundo duração observável da ausência. Nenhuma figura informa processamento interno ou desempenho medido.

![](figures/03-timeline-identidade.svg)

**Figura 1.** Linha do tempo conceitual com permanência, deslocamento, ausência, retorno e observação de páginas semelhantes, sem indicar regra interna ou processamento; os marcos são eventos observáveis; elaboração própria.

## 4. Protocolo de avaliação proposto

O protocolo utiliza sequências anotadas de sessões com uma ou mais páginas. Cada sessão contém períodos de permanência e deslocamento, ausências com causa conhecida quando possível, retorno da mesma folha e entrada de folhas distintas. Condições com páginas visualmente semelhantes são incluídas se fizerem parte da pergunta, sem presumir que toda semelhança produza ambiguidade. A duração observável da ausência varia entre sessões ou blocos para permitir análise da associação com erros, mas o estudo não descreve parâmetros internos.

Um roteiro de captura descreve a tarefa, troca física de página, enquadramento e forma de marcar a referência. A verdade de identidade pode ser estabelecida por observador que acompanha a sessão, por marca externa não visível à saída avaliada ou por registro independente. A marcação não pode depender da decisão observada. Se o suporte sai do quadro e sua identidade deixa de ser conhecida, o episódio é rotulado como indeterminado. As gravações são avaliadas por anotadores treinados que não veem a resposta agregada da condição sempre que possível.

O conjunto de condições cruza evento de referência, semelhança entre páginas, duração qualitativa da ausência e estado de movimento. A matriz deve priorizar cobertura balanceada em vez de produzir todas as combinações, que podem ser numerosas. A ordem é alternada ou randomizada para reduzir confusão com aprendizado e fadiga. Sessões espontâneas e instruídas são identificadas separadamente; uma situação induzida não é tratada como equivalente a um retorno natural.

As métricas têm denominadores explícitos. **Reidentificação correta** é o número de retornos da mesma folha associados à identidade de referência, dividido pelos retornos cuja identidade pode ser anotada. **Fusão indevida** conta eventos em que folhas distintas compartilham uma identidade, com denominador de aparições de folhas novas conhecidas. **Separação indevida** conta novas identidades atribuídas à mesma folha, usando transições em que a continuidade física foi confirmada. **Abstenção** reporta decisões adiadas por tipo de evento. A unidade pode ser evento, trajetória ou sessão; o relatório apresenta a unidade de forma consistente.

**Tabela 2.** Métricas para reidentificação correta, fusão indevida e separação indevida, com denominadores declarados e tratamento separado de abstenção e casos sem referência confiável, facilitando comparação entre sessões avaliadas.

| Métrica | Definição | Denominador | Interpretação |
|---|---|---|---|
| Reidentificação correta | Retorno da mesma folha associado à identidade de referência | Retornos com referência conhecida | Continuidade preservada após ausência |
| Fusão indevida | Folhas distintas associadas à mesma identidade | Aparições distintas com referência conhecida | Mistura de históricos de folhas |
| Separação indevida | Mesma folha representada como identidades diferentes | Transições de continuidade confirmada | Fragmentação do histórico |
| Abstenção | Decisão explicitamente adiada | Eventos com decisão observável | Incerteza transferida ou preservada |
| Caso indeterminado | Referência não permite identificar a folha | Eventos sem referência confiável | Excluído de taxa de acerto e reportado à parte |

A análise apresenta uma matriz de confusão entre decisão e referência, complementada por frequências de fusão, separação e abstenção. Para trajetórias, são relatadas fragmentações e associações incorretas por sessão, sem tratar cada quadro correlacionado como observação independente. Resultados são agregados por participante e sessão quando estes agrupamentos existem. Intervalos de incerteza refletem a unidade real de amostragem. Casos indeterminados permanecem visíveis no relatório e não são removidos silenciosamente.

A associação entre duração de ausência e erro é explorada por categorias predefinidas e por distribuição contínua quando a marca temporal puder ser anotada. A análise não presume relação crescente, nem transforma uma tendência ilustrada em hipótese confirmada. Se a amostra for pequena em certas categorias, isso será reportado como cobertura limitada. A comparação entre semelhança visual e erro considera também ordem de páginas e contexto de tarefa, sem inferir que a semelhança tenha sido usada por qualquer componente interno.

O protocolo pode avaliar custo de interação de uma decisão adiada, como interrupção observável ou solicitação de ação adicional, desde que a saída permita medir esse efeito. Uma entrevista posterior pode perguntar se a continuidade percebida correspondeu à tarefa, mas essa percepção não substitui a referência física. Métricas técnicas e relatos de participantes são analisados separadamente e relacionados apenas com método declarado.

O manual de anotação define evento, identidade física, semelhança, início e fim da ausência e evidência para uma decisão visível. Uma amostra das sequências é anotada por duas pessoas para examinar concordância. Divergências são preservadas até adjudicação e usadas para identificar categorias ambíguas. A equipe reporta dados faltantes, cortes na gravação e alterações de tarefa. A Figura 1 oferece uma linha temporal conceitual; a Figura 2 resume a relação externa entre evento, decisão e custo; e a Figura 3 apresenta apenas uma forma potencial de gráfico.

O plano pré-registra quais eventos contam como elegíveis, como a sessão é encerrada e como sequências incompletas são tratadas. Uma página que sai do campo sem referência posterior não pode ser avaliada como retorno correto ou incorreto. Uma falha de captura que apaga o intervalo inteiro é distinta de uma saída física da página. Categorias que não possam ser determinadas são mantidas visíveis no fluxo de exclusões, para que a incerteza da coleta não seja confundida com uma decisão do sistema.

O protocolo ainda separa qualidade da associação e impacto na atividade. Uma matriz de confusão caracteriza correspondência; uma tarefa ou entrevista pode caracterizar efeitos sobre retomada e compreensão. Quando ambos forem coletados, são analisados em camadas: primeiro a relação com a referência física, depois a experiência relatada. A associação entre os dois não demonstra causa e pode depender de familiaridade, expectativa ou possibilidade de corrigir o erro.

![](figures/03-decisoes-identidade.svg)

**Figura 2.** Diagrama conceitual que relaciona eventos observados a decisões possíveis e custos de associação ou separação; as setas não representam processamento interno na implementação avaliada; elaboração própria.

![](figures/03-erro-ausencia.svg)

**Figura 3.** Curva ilustrativa da relação entre duração de ausência e erro de identidade, sem escala numérica. A forma esperada, não é resultado, mas apenas esquema conceitual; elaboração própria.

## 5. Estrutura dos resultados

Esta seção reserva o formato de apresentação para uma avaliação futura. Nenhuma sequência ou decisão observada foi fornecida, portanto não há taxas de reidentificação, fusão, separação ou abstenção. Os campos marcados como `[PENDENTE-DADOS]` serão preenchidos somente depois da anotação independente e da revisão dos casos indeterminados. As tabelas descrevem o que deverá ser reportado, não resultados previstos.

O relato começa com a composição da amostra: sessões, páginas físicas distintas, retornos, aparições novas e eventos de referência por classe. Categorias sem ocorrência serão identificadas como não observadas, em vez de receberem taxa nula. A duração de ausência será mostrada por distribuição de eventos, com referência à unidade temporal e à regra de anotação usada. Casos incertos da referência serão contados separadamente para evitar que a cobertura pareça maior do que a informação disponível.

**Tabela 3.** Estrutura para relatar eventos anotados e decisões observadas, mantendo campos vazios até que as sequências tenham sido coletadas e revisadas por avaliadores independentes, sem qualquer resultado observado.

| Evento | Casos com referência | Decisões observadas | Observações de anotação |
|---|---|---|---|
| Permanência ou deslocamento | [PENDENTE-DADOS] | [PENDENTE-DADOS] | [PENDENTE-DADOS] |
| Ausência e retorno da mesma folha | [PENDENTE-DADOS] | [PENDENTE-DADOS] | [PENDENTE-DADOS] |
| Aparição de outra folha | [PENDENTE-DADOS] | [PENDENTE-DADOS] | [PENDENTE-DADOS] |
| Referência indeterminada | [PENDENTE-DADOS] | Não aplicável | [PENDENTE-DADOS] |

Os resultados principais deverão mostrar reidentificação correta, fusão indevida, separação indevida e abstenção com seus denominadores. A matriz entre evento e decisão facilita identificar se erros se concentram em retorno, semelhança ou mudança de página. Para sequências com várias transições, a fragmentação por sessão complementará a contagem por evento. A incerteza será reportada de acordo com o agrupamento por pessoa ou sessão; observações correlacionadas não serão contadas como se fossem participantes distintos.

Uma figura futura poderá relacionar erro com duração observável da ausência, desde que a amostra cubra diferentes intervalos e o desenho não confunda duração com identidade de página ou condição de captura. A Figura 3 é apenas uma forma ilustrativa e não implica relação crescente. Se o padrão empírico não for monotônico, a visualização deverá refletir isso. Resultados sem recuperação de identidade serão descritos como tal, sem atribuir uma regra interna para explicar o caso.

## 6. Discussão

Um predomínio de reidentificações corretas após retorno indicaria continuidade nas sequências e condições avaliadas, mas não demonstraria que a identidade foi preservada em qualquer cenário. A leitura dependeria da variedade de páginas, da duração observada de ausência, da taxa de páginas semelhantes e da qualidade da referência. Uma sequência composta apenas por folhas visualmente distintas testaria um problema mais simples que o encontro com páginas semelhantes. A composição da amostra deve acompanhar qualquer resumo de desempenho.

Se fusões indevidas se concentrarem quando outra página parecida aparece, o achado apontaria sensibilidade da associação à ambiguidade visual nas condições observadas. Ainda assim, a interpretação deve considerar ordem de captura, conteúdo, pose e anotação física. O resultado não revelaria qual atributo visual foi usado, nem autorizaria recomendar uma solução interna. Ele poderia justificar novos estudos com pares de páginas controlados ou com variações naturais de conteúdo.

Se separações indevidas forem mais comuns após retorno da mesma folha, a continuidade da sessão estaria sendo fragmentada em algumas condições. A consequência pode ser mais ou menos grave conforme a pessoa consiga recuperar o desenho anterior por ação explícita. A análise deve distinguir fragmentação técnica da percepção de perda por participantes. Uma decisão de iniciar nova identidade pode ser prudente em casos incertos e ainda assim impor custo de retomada; por isso, as métricas técnicas e de interação são reportadas em conjunto, sem serem confundidas.

Abstenções podem reduzir fusões e separações, mas também interromper a tarefa ou exigir uma confirmação. Um resultado com menos erros e mais decisões adiadas não é automaticamente superior nem inferior. A comparação depende de custo, frequência e possibilidade de recuperação, aspectos relacionados ao uso. Estudos de decisão seletiva e incerteza motivam relatar cobertura junto com erro, em vez de ocultar abstenções numa taxa agregada [13, 20]. A equipe deve descrever o que a pessoa vê e faz após a decisão pendente, se isso for observável.

Uma associação entre duração da ausência e erro pode ser plausível, mas não é inevitável. Um período maior pode coincidir com troca de página, mudança de atividade ou alterações no ambiente. Uma ausência breve pode ocorrer justamente durante uma troca intencional de suporte. O protocolo anota esses eventos para evitar interpretar a duração isoladamente. Se não houver cobertura suficiente por categoria, a discussão deverá limitar-se à descrição, sem extrapolar uma curva para tempos ausentes.

Rastreamento multiobjeto oferece conceitos como fragmentação de trajetória, troca de identidade e associação correta. Tais conceitos ajudam a separar erros que uma medida simples de presença não detectaria. No entanto, métricas de benchmark são definidas para quadros e alvos com condições próprias. Este protocolo adota as distinções conceituais e recalcula denominadores com a página física e a sessão como unidades relevantes. A transferência direta de rankings ou valores de outros domínios seria injustificada.

A recuperação visual pode relacionar páginas ou imagens parecidas, mas identidade física não se reduz à similaridade de conteúdo. Uma cópia do mesmo desenho em outra folha representa outra instância para este artigo. Esse recorte deve ser explícito, sobretudo quando bancos de documentos tratam versões ou páginas lógicas como identidade. Se uma aplicação definir a identidade pelo conteúdo e não pelo suporte, será necessário mudar a referência e interpretar os custos de outra maneira.

Os resultados possíveis também precisam ser vistos sob a perspectiva de uso. Uma fusão pode confundir autoria ou histórico, enquanto uma separação pode fragmentar o desenho e exigir navegação manual. A importância depende de tarefa, reversibilidade e possibilidade de correção. Estudos com participantes poderiam avaliar se decisões pendentes são compreensíveis e se erros afetam a confiança. Esse estudo posterior não deve inferir experiência a partir de contagens técnicas, pois uma ocorrência rara pode ter impacto alto e uma ocorrência frequente pode ser facilmente corrigida.

Por fim, não se define uma duração aceitável de ausência. A literatura de rastreamento pode sugerir condições e medidas de comparação, mas o intervalo adequado depende de velocidade de uso, interação e custo de cada decisão. A contribuição desta proposta é garantir que ausência, retorno e identidade sejam descritos antes de apresentar um julgamento de desempenho. Padrões e recomendações só podem emergir de dados e tarefas observados.

## 7. Limitações e ameaças à validade

A validade de construção depende da definição de identidade física e dos eventos anotados. A classificação “página parecida” é subjetiva se não houver instruções claras; “retorno” pode ser confundido com outra instância sem marca externa; e “ausência” pode decorrer de corte da gravação em vez de movimento real. O protocolo separa esses casos e usa uma categoria indeterminada, mas a anotação continua dependente da qualidade e da continuidade do registro.

As métricas podem privilegiar um tipo de erro. Uma taxa agregada de acerto pode parecer alta mesmo quando fusões têm consequências severas ou quando abstenções se concentram em condições frequentes. Contagens com denominadores diferentes não devem ser comparadas sem explicação. A equipe escolhe previamente desfechos principais, reporta frequências brutas e incerteza e apresenta a distribuição por evento. A análise de fragmentação por sessão complementa a medida por transição, mas não a substitui.

A validade interna é ameaçada por coocorrência entre ausência e troca de página, semelhança e conteúdo, ou pose e deslocamento. Uma manipulação controlada pode isolar parte desses fatores, porém reduzir a naturalidade da tarefa. Uma observação espontânea preserva uso, mas não sustenta a mesma inferência causal. O desenho precisa dizer qual questão pretende responder e limitar a linguagem das conclusões ao tipo de comparação realizada.

A validade externa depende de tipos de folha, desenho, duração da sessão, forma de captura e diversidade de participantes. Páginas em branco, folhas com conteúdo denso e desenhos em andamento podem apresentar desafios diferentes. Uma avaliação feita em mesa estável não representa automaticamente outras superfícies ou contextos. O estudo deve descrever a população e o ambiente e evitar a afirmação de que a taxonomia cobre todos os usos.

A qualidade da referência pode ser a principal fonte de erro. Uma marca física ou acompanhamento presencial ajuda a estabelecer identidade, mas pode falhar se folhas forem trocadas fora do registro. Anotadores podem divergir sobre a mesma página quando a sequência tem cortes ou enquadramento parcial. Concordância e adjudicação precisam ser reportadas; sequências sem verdade confiável são excluídas das métricas de acerto e apresentadas como indeterminadas. Uma referência imperfeita limita também a aparente precisão do sistema avaliado.

Há ainda dependência temporal. Quadros próximos da mesma sessão compartilham contexto e não equivalem a observações independentes. Ignorar o agrupamento pode estreitar artificialmente a incerteza. Uma amostra desigual, com muitas sequências de poucas pessoas, pode aparentar grande quantidade sem representar diversidade real. A análise deve explicitar unidade experimental, unidade de anotação e agrupamento estatístico.

Finalmente, o protocolo de caixa-preta mede saídas externas e não explica como a continuidade é produzida. Uma associação errada pode resultar de vários fatores que a avaliação não separa. Essa restrição preserva a possibilidade de comparação sem divulgar detalhes internos, mas reduz diagnóstico e proposta de correção. Conclusões sobre mecanismo ou cálculo de identidade não são sustentadas por este desenho.

## 8. Conclusão

Este artigo propõe uma taxonomia para avaliar a identidade de páginas ao longo de sessões de desenho. Seis eventos — permanência, deslocamento, ausência, retorno da mesma folha, aparição de outra semelhante e aparição de outra diferente — são relacionados às decisões de continuar, iniciar uma identidade distinta ou adiar. A classificação separa reidentificação correta, fusão indevida, separação indevida, abstenção e referência indeterminada.

O protocolo usa sequências anotadas e referência física independente. Ele compara decisões observáveis com a identidade conhecida, descreve a duração de ausência como condição externa e mantém a implementação como caixa-preta. Tabelas e figuras organizam eventos e custos sem apresentar dados. Resultados futuros deverão mostrar denominadores, incerteza, agrupamento e consequências para a tarefa.

Ainda é necessária uma avaliação com folhas variadas, eventos de retorno e páginas visualmente semelhantes. A validação da anotação e a análise dos custos de decisão são essenciais antes de qualquer recomendação. Até então, a relação entre ausência, semelhança e erro permanece `[PENDENTE-DADOS]`.

## Declarações

- Uso de IA: texto redigido com apoio de IA generativa, com revisão e responsabilidade dos autores; referências conferidas por DOI.
- Dados: não há dados experimentais nesta proposta; sequências e resultados futuros serão descritos após coleta e validação.
- Código: não há código de análise de resultados nesta versão; materiais futuros serão identificados por versão.
- Conflito de interesses: os autores deverão declarar qualquer conflito pertinente antes da submissão.
- Ética: estudos com participantes e gravação de sessões serão submetidos à avaliação ética aplicável antes do recrutamento.
- Contribuição dos autores: a versão final deverá explicitar as contribuições individuais dos autores.

## Referências

[1] Kuhn, H. W. The Hungarian Method for the Assignment Problem. Naval Research Logistics Quarterly, 1955. https://doi.org/10.1002/nav.3800020109
[2] Reid, D. An Algorithm for Tracking Multiple Targets. IEEE Transactions on Automatic Control, 1979. https://doi.org/10.1109/TAC.1979.1102177
[3] Blackman, S. S. Multiple Hypothesis Tracking for Multiple Target Tracking. IEEE Aerospace and Electronic Systems Magazine, 2004. https://doi.org/10.1109/MAES.2004.1263228
[4] Joo, S.-W.; Chellappa, R. A Multiple-Hypothesis Approach for Multiobject Visual Tracking. IEEE Transactions on Image Processing, 2007. https://doi.org/10.1109/TIP.2007.906254
[5] Cox, I. J.; Hingorani, S. L. An Efficient Implementation of Reid’s Multiple Hypothesis Tracking Algorithm and Its Evaluation for the Purpose of Visual Tracking. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1996. https://doi.org/10.1109/34.481539
[6] Bernardin, K.; Stiefelhagen, R. Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. EURASIP Journal on Image and Video Processing, 2008. https://doi.org/10.1155/2008/246309
[7] Dendorfer, P. et al. MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking. International Journal of Computer Vision, 2021. https://doi.org/10.1007/s11263-020-01393-0
[8] Ristani, E. et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. European Conference on Computer Vision Workshops, 2016. https://doi.org/10.1007/978-3-319-48881-3_2
[9] Luiten, J. et al. HOTA: A Higher Order Metric for Evaluating Multi-object Tracking. International Journal of Computer Vision, 2021. https://doi.org/10.1007/s11263-020-01375-2
[10] Bewley, A. et al. Simple Online and Realtime Tracking. International Conference on Image Processing, 2016. https://doi.org/10.1109/ICIP.2016.7533003
[11] Wojke, N.; Bewley, A.; Paulus, D. Simple Online and Realtime Tracking with a Deep Association Metric. International Conference on Image Processing, 2017. https://doi.org/10.1109/ICIP.2017.8296962
[12] Zhang, Y. et al. ByteTrack: Multi-object Tracking by Associating Every Detection Box. European Conference on Computer Vision, 2022. https://doi.org/10.1007/978-3-031-20047-2_1
[13] Vo, B.-N.; Vo, B.-T.; Phung, D. Labeled Random Finite Sets and the Bayes Multi-Target Tracking Filter. IEEE Transactions on Signal Processing, 2014. https://doi.org/10.1109/TSP.2014.2364014
[14] Bedagkar-Gala, A.; Shah, S. K. A Survey of Approaches and Trends in Person Re-Identification. Image and Vision Computing, 2014. https://doi.org/10.1016/j.imavis.2014.02.001
[15] Ye, M. et al. Deep Learning for Person Re-Identification: A Survey and Outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. https://doi.org/10.1109/TPAMI.2021.3054775
[16] Zheng, L. et al. Scalable Person Re-Identification: A Benchmark. IEEE/CVF International Conference on Computer Vision, 2015. https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html
[17] Zheng, L. et al. Person Re-identification in the Wild. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017. https://doi.org/10.1109/CVPR.2017.357
[18] Gou, M. et al. DukeMTMC4ReID: A Large-Scale Multi-camera Person Re-identification Dataset. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2017. https://doi.org/10.1109/CVPRW.2017.185
[19] Zheng, L. et al. MARS: A Video Benchmark for Large-Scale Person Re-Identification. European Conference on Computer Vision, 2016. https://doi.org/10.1007/978-3-319-46466-4_52
[20] Yu, T. et al. Robust Person Re-Identification by Modelling Feature Uncertainty. IEEE/CVF International Conference on Computer Vision, 2019. https://doi.org/10.1109/ICCV.2019.00064
[21] Lowe, D. G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 2004. https://doi.org/10.1023/B:VISI.0000029664.99615.94
[22] Bay, H. et al. Speeded-Up Robust Features (SURF). Computer Vision and Image Understanding, 2008. https://doi.org/10.1016/j.cviu.2007.09.014
[23] Sivic, J.; Zisserman, A. Video Google: A Text Retrieval Approach to Object Matching in Videos. IEEE/CVF International Conference on Computer Vision, 2003. https://doi.org/10.1109/ICCV.2003.1238663
[24] Philbin, J. et al. Object Retrieval with Large Vocabularies and Fast Spatial Matching. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2007. https://doi.org/10.1109/CVPR.2007.383172
[25] Conrad, J. G.; Schriber, C. P. Managing déjà vu: Collection Building for the Identification of Nonidentical Duplicate Documents. Journal of the American Society for Information Science and Technology, 2006. https://doi.org/10.1002/asi.20363
[26] Dicle, C.; Camps, O. I.; Sznaier, M. The Way They Move: Tracking Multiple Targets with Similar Appearance. IEEE/CVF International Conference on Computer Vision, 2013. https://openaccess.thecvf.com/content_iccv_2013/papers/Dicle_The_Way_They_2013_ICCV_paper.pdf
