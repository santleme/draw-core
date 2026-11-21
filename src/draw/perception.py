"""Detector neural da folha: segmentação por ONNX atrás do contrato
de draw.paper.detect_sheet_with_confidence.

O modelo da fase 2 (SegFormer-B0 exportado para ONNX) recebe o frame
esmagado para a resolução de entrada, normalizado com média e desvio
do ImageNet, e devolve logits de máscara em dois canais e um logit
de presença. A resolução segue o arquivo: um ONNX de formato fixo
impõe o tamanho dele (512x512 no export original); um ONNX dinâmico
(eixos H/W simbólicos) roda a 384x384, o ponto medido no Ryzen 7
7700X do Lucas em 2026-10-29 — session.run de 122 para 58 ms, IoU de
máscara de 0,984 contra o 512, nenhum veredito de presença trocado
nos frames dele fora as três frames da caixa branca, que a 384
deixam de ser falso positivo. A sessão roda com 4 threads intra-op,
também medido: o padrão do ONNX Runtime marcava 248 ms no mesmo
tensor, 4 threads marcam 122 ms e 16 threads pioram para 210 ms
(draw-lab/perf-ryzen-2026-10-29/REPORT.md). A
presença decide se há papel. A máscara decide onde: o quadrilátero da
folha sai do maior contorno da máscara limiarizada, com a mesma escada
de epsilon do detector clássico, um reajuste de cada aresta como reta
sobre os pontos do contorno e refino subpixel nos cantos dentro do
quadro.

Quando a escada de epsilon não fecha sobre o contorno, o
quadrilátero é o retângulo de área mínima do contorno, sem reajuste
nem refino: medido nos frames reais em que a escada falha, os dois
passos degradam essa reserva (os números ficam em _min_area_quad). A
reserva só existe onde antes o retorno era nenhum quadrilátero;
onde a escada fecha, o caminho é o mesmo de sempre.

O reajuste de arestas existe para o caso comum de uso: a página
cortada pela borda do quadro. Ali o vértice da aproximação mora sobre
a borda e não é o canto da página. A reta ajustada sobre os pontos
visíveis da aresta projeta o canto para fora do quadro, que é onde a
página continua. Pontos do contorno colados na borda do frame ficam
fora do ajuste: ali quem cortou a máscara foi o quadro, não a página.
Cada canto reajustado só é aceito se ficar a no máximo
_FIT_MAX_DISPLACE_PX do vértice da aproximação; sem essa trava, uma
aresta curta ou contaminada projeta cantos para longe demais.

Aresta cuja cobertura de pontos é curta (a mão cobre a maior parte
dela) não ganha reta ajustada: dentro de um quadro só, os pontos
visíveis dela são a borda do braço, não a aresta, e a reta do
fragmento projeta os cantos a centenas de pixels (o teto acima veta
a correção e o canto congela no vértice da aproximação, que nesses
quadros mora sobre a borda do quadro). Quem resolve é a memória de
arestas do detector: a aresta coberta herda a última reta confiável
dela mesma, e só sem herança vale a reta travada na direção do
segmento. Os cantos vizinhos de uma aresta travada ou herdada não
passam pelo teto: a projeção já nasce limitada.

A sessão ONNX é preguiçosa e o onnxruntime só é importado no primeiro
uso. Importar este módulo nunca quebra em ambiente sem a dependência;
usar o detector sem ela levanta um erro explícito.

O arquivo do modelo fica fora do repositório (binário de 15 MB). O
caminho chega sempre por argumento explícito ao detector.
"""

import inspect
import time
from pathlib import Path
from typing import NamedTuple, Tuple

import cv2
import numpy as np

from . import geometry as _geometry

_INPUT_SIZE = 384
_ORT_INTRA_THREADS = 3
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_NORM_SCALE = (1.0 / (255.0 * _STD)).astype(np.float32)
_NORM_BIAS = (-_MEAN / _STD).astype(np.float32)
_MEMORY_MIN_MASK_IOU = 0.7
_MEMORY_MASK_STRIDE = 4
_PAPER_CLASS = 1
_HAND_CLASS = 2
_INK_CLASS = 3
_SEGMENTATION_PRESENCE_MIN_RATIO = 0.01
_SEGMENTATION_ABSENT_LOGIT = -10.0
_SEGMENTATION_PRESENT_LOGIT = 1.0
_TIMING_PHASES = ("preprocess", "session_run", "postprocess")
_TIMING_LOG_EVERY = 100


# Portão do caminho clássico (medido em 2026-10-30 no laboratório,
# draw-lab/fp-gate-2026-10-30/): com um modelo neural carregado, um
# logit de presença até este valor é um "não há papel" forte o bastante
# para o clássico não poder declarar folha por cima dele (o incidente
# da caixa branca: a v2 media logit ≈ −10,9 na caixa e o banner vinha
# do clássico). Nos frames reais medidos (44 positivos por modelo: os
# 43 do frames-dataset e o frame_real), o menor logit de positivo foi
# +7,38 (v3 dinâmico); nos 6 negativos do dataset, o maior logit foi
# −6,98 (v3). O limiar fica 10,4 abaixo do positivo mais fraco e 4,0
# acima do negativo mais forte: veta o clássico em 18/18 corridas
# (3 modelos × 6 negativos) sobre os negativos do dataset e não custa
# nenhum positivo medido. Ele NÃO cobre a caixa branca na v3 a 512
# (logit +0,41 a +0,98: a própria neural afirma papel, fraco, e o
# quadrilátero fecha sobre ~3,5% de máscara) nem os logits −0,006 a
# −0,382 da v3 dinâmica na caixa: nenhum dos dois é "fortemente
# negativo", e baixar o limiar para perto de zero desligaria a reserva
# clássica justamente no cenário de generalização, a folha real que a
# neural subdispara num setup novo.
PRESENCE_VETO_LOGIT = -3.0


def _frame_to_tensor(frame, size):
    """Pré-processo ÚNICO frame→tensor do detector de folha.

    Contrato: resize INTER_LINEAR para (largura, altura) de `size`
    (par (altura, largura) do contrato do modelo), BGR→RGB, /255 e
    normalização ImageNet (média/desvio acima). Treino
    (training/train.py) e produção leem o MESMO contrato; qualquer
    ferramenta externa (bancada de julgamento, sondas) deve importar
    esta função em vez de reimplementar o pré-processo — a divergência
    silenciosa BGR-sem-normalização da bancada (2026-11-08) é o motivo
    desta extração. Reduzir ANTES de converter: o resize é linear por
    canal e a conversão é uma permutação exata de canais; as operações
    comutam e a conversão roda sobre a imagem já reduzida.
    """
    input_h, input_w = size
    small = cv2.resize(
        frame,
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = (resized * _NORM_SCALE + _NORM_BIAS).transpose(2, 0, 1)
    return tensor[None].astype(np.float32)


def _load_session(model_path, *, verify, intra_threads=None):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime nao esta instalado; o detector neural precisa "
            "dele. Instale com: pip install onnxruntime"
        ) from error
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"modelo ONNX nao encontrado: {path}. Passe o caminho do arquivo .onnx."
        )
    # Integridade obrigatoria como argumento: o pipeline injeta
    # draw.infra.manifest.verify_model como `verify`; verify=None é o
    # opt-out explicito do chamador, nada muda.
    if verify is not None:
        verify(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = (
        intra_threads
        if intra_threads is not None and intra_threads > 0
        else _ORT_INTRA_THREADS
    )
    _disable_spin(options)
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _disable_spin(options):
    """Desliga o busy-wait (spin) dos pools desta sessao.

    O app roda duas sessoes ORT + MediaPipe + OpenCV no mesmo
    processo; com spin ligado (o padrao do ORT), as threads de uma
    sessao giram em espera ativa e roubam CPU da sessao que esta
    trabalhando — em pipelines multi-modelo medidos por terceiros,
    detectores rodaram 2-3x mais lentos que isolados com spin
    ligado. O estudo de performance de 2026-11-03 apontou o spin
    como suspeito do piso de ~2,5 nucleos sem explicacao no regime
    sem folha. O runtime migrou sem o religar: a sessao sempre
    roda com spin desabilitado, o padrao do legado.
    """
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")


class _SessionContract(NamedTuple):
    """O que o arquivo ONNX diz sobre si mesmo.

    input_name é o nome que o feed do session.run precisa usar —
    nunca um literal no código. size é a resolução quadrada de
    entrada, já resolvida (o dinâmico cai em _INPUT_SIZE). outputs
    são os nomes das saídas, na ordem do arquivo, para não
    reconstruir a lista a cada frame.
    """

    input_name: str
    size: Tuple[int, int]
    outputs: Tuple[str, ...]


def _session_contract(session):
    """Lê e valida o contrato de entrada/saída de uma sessão ONNX.

    O adaptador antigo mandava o tensor no feed com o nome literal
    "image" e lia só shape[2], presumindo quadrado. Um export com
    entrada "input" quebrava no primeiro frame (não na carga), e um
    export retangular era esmagado para quadrado sem aviso. Aqui o
    nome vem do arquivo e a resolução é validada na primeira
    chamada, com erro explícito.
    """
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(
            "o modelo ONNX precisa de exatamente uma entrada de imagem; "
            f"este arquivo declara {[spec.name for spec in inputs]}"
        )
    spec = inputs[0]
    shape = list(spec.shape)
    if len(shape) != 4:
        raise RuntimeError(
            "a entrada do modelo precisa ser [N, C, H, W]; o arquivo "
            f"declara {spec.name} com forma {shape}"
        )
    height, width = shape[2], shape[3]
    fixed = [
        dimension
        for dimension in (height, width)
        if isinstance(dimension, int) and dimension > 0
    ]
    if len(fixed) == 2:
        if fixed[0] != fixed[1]:
            raise RuntimeError(
                "entrada retangular não é suportada: o pré-processamento "
                "esmaga o frame para uma resolução quadrada e deformaria "
                f"a imagem em silêncio (o arquivo declara {height}x{width})"
            )
        size = (fixed[0], fixed[1])
    elif len(fixed) == 0:
        size = (_INPUT_SIZE, _INPUT_SIZE)
    else:
        raise RuntimeError(
            "entrada com altura fixa e largura dinâmica (ou o inverso) "
            f"não é suportada; o arquivo declara {shape}"
        )
    return _SessionContract(
        input_name=spec.name,
        size=size,
        outputs=tuple(output.name for output in session.get_outputs()),
    )


class NeuralSheetDetector:
    """Detector neural com sessão própria.

    __call__(frame) devolve (cantos, veredito, máscara): o mesmo trio
    da interface dos detectores do harness de avaliação. O veredito é
    "paper" quando o logit de presença é positivo e o quadrilátero
    fecha; fora isso, (None, None, máscara). A máscara é uint8
    0/255 no tamanho do frame, limiarizada em 0,5 como na avaliação
    da fase 2. return_mask=False evita materializar essa máscara;
    o ajuste e a memória de arestas usam sempre a grade nativa.

    O detector guarda as retas confiáveis das arestas entre chamadas
    (a memória de arestas): quando uma aresta perde cobertura numa
    chamada (a mão sobre ela), a reta herdada estima essa aresta até
    ela voltar a ser visível. A herança só vale em cena contínua: a
    máscara da chamada precisa ter IoU de pelo menos
    _MEMORY_MIN_MASK_IOU com a da chamada anterior, medido nas
    máscaras reduzidas por _MEMORY_MASK_STRIDE. Cena nova recomeça
    sem herança.

    O atributo last_presence_logit guarda o sinal de presença da
    última chamada a predict (None antes da primeira e após um frame
    vazio): nos modelos legados é o logit treinado; no SegFormer de
    quatro classes é um sinal derivado da área segmentada. O app usa-o
    para decidir se a reserva clássica pode falar.
    """

    def __init__(
        self,
        model_path,
        *,
        verify,
        intra_threads=None,
        timing=False,
        spread_mode="auto",
    ):
        """model_path é obrigatório; verify é um callable
        Callable[[Path | str], None] | None obrigatório chamado sobre
        o arquivo antes de abrir a sessão (o pipeline injeta
        draw.infra.manifest.verify_model; verify=None é o opt-out
        explícito). intra_threads é o teto intra-op da sessão ORT
        (None = 3, o valor medido na mesa). timing=True liga o
        cronômetro de fases (get_timing_stats). spread_mode controla a
        união de duas páginas: auto aplica o limite de calha de 10%,
        always ignora esse limite e never desativa a união."""
        _geometry._validate_spread_mode(spread_mode)
        self.model_path = str(model_path)
        self._timing_enabled = bool(timing)
        self._verify = verify
        self._intra_threads = intra_threads
        self._spread_mode = spread_mode
        self._session = None
        self._contract = None
        self._contract_session = None
        self._edge_lines = None
        self._memory_mask = None
        self._memory_geometry = None
        self.last_presence_logit = None
        # Cobertura de folha da última previsão (0/255, resolução da
        # máscara devolvida): classes de folha sob qualquer ocupante
        # — papel, tinta E mão. A mão fica dentro para a cobertura
        # não virar buraco permanente onde a mão paira; quem segura
        # o voto sob a mão é a visibilidade, não a cobertura.
        self.last_coverage_mask = None
        self.last_hand_mask = None
        self._timing_stats = {
            phase: {"count": 0, "sum_ms": 0.0, "max_ms": 0.0}
            for phase in _TIMING_PHASES
        }

    def get_timing_stats(self):
        """Snapshot em ms das previsões completas com timing=True.

        Ative o parâmetro ao construir o detector. Carga da sessão,
        leitura do contrato e fit_quad não entram nas três fases.
        """
        return {
            phase: {
                **stats,
                "mean_ms": stats["sum_ms"] / stats["count"] if stats["count"] else 0.0,
            }
            for phase, stats in self._timing_stats.items()
        }

    def _record_timing(self, durations):
        for phase, seconds in zip(_TIMING_PHASES, durations, strict=True):
            stats = self._timing_stats[phase]
            milliseconds = seconds * 1000.0
            stats["count"] += 1
            stats["sum_ms"] += milliseconds
            stats["max_ms"] = max(stats["max_ms"], milliseconds)
        count = self._timing_stats["session_run"]["count"]
        if count % _TIMING_LOG_EVERY == 0:
            summary = " | ".join(
                f"{phase}: mean={stats['mean_ms']:.3f} ms max={stats['max_ms']:.3f} ms"
                for phase, stats in self.get_timing_stats().items()
            )
            print(f"timing frames={count} | {summary}")

    def _get_session(self):
        if self._session is None:
            self._session = _load_session(
                self.model_path,
                verify=self._verify,
                intra_threads=self._intra_threads,
            )
        return self._session

    def contract(self):
        """O contrato do arquivo carregado (nome da entrada, resolução).

        É lido uma vez por sessão. Um teste que injeta a sessão em
        _session (sem passar pelo disco) continua funcionando: o
        contrato é resolvido da sessão em uso, não do caminho.
        """
        session = self._get_session()
        if self._contract is None or self._contract_session is not session:
            self._contract = _session_contract(session)
            self._contract_session = session
        return self._contract

    def load(self):
        """Carrega a sessão agora, para erro de modelo falhar cedo."""
        self._get_session()

    def _fallback_lines(self, mask):
        """As retas herdáveis para esta máscara, ou None sem continuidade."""
        if self._edge_lines is None or self._memory_mask is None:
            return None
        small = mask[::_MEMORY_MASK_STRIDE, ::_MEMORY_MASK_STRIDE] > 0
        if small.shape != self._memory_mask.shape:
            return None
        inter = int((small & self._memory_mask).sum())
        union = int((small | self._memory_mask).sum())
        if union == 0 or inter / union < _MEMORY_MIN_MASK_IOU:
            return None
        return self._edge_lines

    def _fit(self, mask, frame=None, *, scale_x=1.0, scale_y=1.0):
        """fit_quad com a memória de arestas do detector.

        A guarda de plausibilidade (fit_plausible) fecha o caminho:
        quadrilátero recusado devolve None e não alimenta a memória
        de arestas, para a geometria selvagem não virar herança.
        """
        geometry = (mask.shape, scale_x, scale_y)
        if geometry != self._memory_geometry:
            self._edge_lines = None
            self._memory_mask = None
            self._memory_is_spread = False
            self._memory_spread_area = 0.0
            self._memory_geometry = geometry
        contour, is_spread, contour_area = _geometry._sheet_contour_with_spread(
            mask, spread_mode=getattr(self, "_spread_mode", "auto")
        )
        prev_spread = getattr(self, "_memory_is_spread", False)
        prev_area = getattr(self, "_memory_spread_area", 0.0)
        # Invalida a memoria de arestas apenas na transicao pagina unica ->
        # caderno aberto ou quando o caderno esta EXPANDINDO para fora (nunca
        # quando encolhe sob oclusao de mao, para preservar a guarda de outlier).
        spread_expanded = (is_spread and not prev_spread) or (
            is_spread and prev_area > 0 and contour_area > prev_area * 1.05
        )
        fallback = None if spread_expanded else self._fallback_lines(mask)
        # Dublês de teste antigos nao aceitam `contour`; decide pela assinatura
        # em vez de capturar TypeError (que mascararia erros reais do ajuste).
        impl = _geometry._fit_quad_impl
        if "contour" in inspect.signature(impl).parameters:
            corners, lines, trusted = impl(
                mask, frame, fallback, scale_x, scale_y, contour=contour
            )
        else:
            corners, lines, trusted = impl(mask, frame, fallback, scale_x, scale_y)
        ref_shape = (
            frame.shape
            if frame is not None
            else (mask.shape[0] * scale_y, mask.shape[1] * scale_x)
        )
        if corners is not None and not _geometry.fit_plausible(corners, ref_shape):
            return None
        if lines is not None:
            # Aresta sem medida confiável neste quadro grava ausente:
            # copiar a linha velha faria o próximo quadro tratá-la
            # como histórico contínuo, e uma folha nova herdaria as
            # arestas da anterior. A linha antiga só sobrevive onde
            # a herança deste quadro foi válida (fallback presente).
            stored = [None, None, None, None]
            for i in range(4):
                if trusted[i]:
                    stored[i] = lines[i]
                elif (
                    fallback is not None
                    and fallback[i] is not None
                    and self._edge_lines is not None
                ):
                    stored[i] = self._edge_lines[i]
            self._edge_lines = stored
        elif fallback is None:
            # Ajuste sem retas (ex.: _min_area_quad) em quadro sem heranca:
            # descarta retas antigas para nao parear a nova mascara com linhas
            # pre-expansao no proximo quadro.
            self._edge_lines = None
        self._memory_mask = mask[::_MEMORY_MASK_STRIDE, ::_MEMORY_MASK_STRIDE] > 0
        self._memory_is_spread = is_spread
        if not is_spread:
            self._memory_spread_area = 0.0
        else:
            # Guarda o pico de area do caderno aberto apenas quando houve retas
            # medidas e a guarda de outlier nao pode ter travado nenhuma aresta.
            if lines is not None and (
                fallback is None or (trusted is not None and all(trusted))
            ):
                self._memory_spread_area = max(prev_area, contour_area)
        return corners

    def predict(self, frame, *, native=False):
        """Roda o modelo. Retorna (sinal de presença, máscara 0/255).

        native=True mantém a resolução de saída do modelo, sem resize.
        O padrão preserva a máscara full-res para diagnóstico/avaliação.
        O contrato legado tem uma máscara binária e um logit de presença.
        O SegFormer de quatro classes exporta apenas logits semânticos;
        nesse caso, papel e tinta formam a região da folha (a mão continua
        excluída) e a própria área prevista fornece um sinal de presença.

        O nome da entrada e a lista de saídas vêm do arquivo
        (contract()), nunca de literais: um export com entrada
        "input" ou com uma saída auxiliar a mais deixa de quebrar no
        primeiro frame. Um mask_logits de quatro canais é lido pelo
        argmax (papel + tinta), não mais como fundo x classe 1.
        """
        session = self._get_session()
        contract = self.contract()
        preprocess_start = time.perf_counter() if self._timing_enabled else None
        frame_h, frame_w = frame.shape[:2]
        tensor = _frame_to_tensor(frame, contract.size)
        output_names = list(contract.outputs)
        feed = {contract.input_name: tensor}
        if self._timing_enabled:
            preprocess_end = time.perf_counter()
        outputs = session.run(output_names, feed)
        if self._timing_enabled:
            inference_end = time.perf_counter()
        by_name = dict(zip(output_names, outputs, strict=True))

        if "mask_logits" in by_name and "presence_logit" in by_name:
            logits = by_name["mask_logits"]
            presence = by_name["presence_logit"]
            if logits.ndim != 4:
                raise ValueError(
                    "saída 'mask_logits' precisa ter forma [N, C, H, W]; "
                    f"veio {logits.shape}"
                )
            channels = int(logits.shape[1])
            if channels == 2:
                # canais do contrato legado: 0 = fundo, 1 = papel.
                # prob(papel) > 0,5 <=> exp_papel > exp_fundo <=>
                # logit_papel > logit_fundo: o softmax de dois canais
                # que existia aqui dava a mesma máscara e custava um
                # exp sobre 2 x lado x lado por frame.
                mask_small = (logits[0, 1] > logits[0, 0]).astype(np.uint8)
                coverage_small = mask_small
                hand_small = None
            elif channels == 4:
                # Export híbrido: máscara semântica de quatro classes
                # com cabeça de presença treinada. Antes, esta forma
                # era lida em silêncio como fundo x classe 1 (máscara
                # errada, sem erro nenhum).
                classes = np.argmax(logits[0], axis=0)
                mask_small = (
                    (classes == _PAPER_CLASS) | (classes == _INK_CLASS)
                ).astype(np.uint8)
                # Cobertura de voto: tudo que não é fundo — papel,
                # tinta e mão sobre a folha entram, mesa não entra.
                coverage_small = (classes != 0).astype(np.uint8)
                hand_small = (classes == _HAND_CLASS).astype(np.uint8)
            else:
                raise ValueError(
                    "saída 'mask_logits' precisa ter 2 canais "
                    "(fundo, papel) ou 4 (fundo, papel, mão, tinta); "
                    f"veio {channels}"
                )
            self.last_presence_logit = float(presence[0])
        elif "logits" in by_name:
            logits = by_name["logits"]
            if logits.ndim != 4 or logits.shape[1] != 4:
                raise ValueError(
                    "saída 'logits' precisa ter forma [N, 4, H, W] "
                    "(fundo, papel, mão, tinta)"
                )
            classes = np.argmax(logits[0], axis=0)
            page = (classes == _PAPER_CLASS) | (classes == _INK_CLASS)
            page_ratio = float(np.mean(page))
            mask_small = page.astype(np.uint8)
            coverage_small = (classes != 0).astype(np.uint8)
            hand_small = (classes == _HAND_CLASS).astype(np.uint8)
            self.last_presence_logit = (
                _SEGMENTATION_PRESENT_LOGIT
                if page_ratio >= _SEGMENTATION_PRESENCE_MIN_RATIO
                else _SEGMENTATION_ABSENT_LOGIT
            )
        else:
            raise ValueError(
                "saídas ONNX incompatíveis: esperava 'mask_logits' e "
                "'presence_logit' ou 'logits' com quatro classes; "
                f"recebi {output_names}"
            )
        mask = mask_small * 255
        # Cobertura para o voto de tinta: o que o
        # modelo diz não ser fundo, com buracos internos fechados —
        # uma mão que ele não reconheceu pairando sobre a folha não
        # pode secar a tinta debaixo dela. Fica na resolução nativa:
        # quem consome (draw/app.py) faz o upsample no mesmo ponto
        # em que a máscara de chamada ganha a resolução do frame.
        self.last_coverage_mask = _geometry.fill_interior_holes(coverage_small) * 255
        # Oclusão semântica: a classe mão vira segundo sinal do
        # canal visible do voto de tinta.
        # Resolução nativa da máscara — quem consome faz o upsample
        # no mesmo ponto da cobertura.
        self.last_hand_mask = None if hand_small is None else hand_small * 255
        if not native:
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        if self._timing_enabled:
            postprocess_end = time.perf_counter()
            self._record_timing(
                (
                    preprocess_end - preprocess_start,
                    inference_end - preprocess_end,
                    postprocess_end - inference_end,
                )
            )
        return self.last_presence_logit, mask

    def __call__(self, frame, *, return_mask=True):
        if frame is None or frame.size == 0:
            self.last_presence_logit = None
            return None, None, None
        presence, mask = self.predict(frame, native=True)
        frame_h, frame_w = frame.shape[:2]
        mask_h, mask_w = mask.shape
        corners = None
        if presence > 0:
            corners = self._fit(
                mask, frame, scale_x=frame_w / mask_w, scale_y=frame_h / mask_h
            )
        verdict = "paper" if corners is not None else None
        if return_mask:
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = None
        return corners, verdict, mask

    def detect_sheet_with_confidence(self, frame):
        """O par (cantos, veredito) do contrato de draw.paper."""
        corners, verdict, _ = self(frame, return_mask=False)
        return corners, verdict
