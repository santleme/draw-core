"""Photometric guard for page regions darkened by hands or shadows."""

import cv2
import numpy as np

from ..contracts import ExposureAssessment

_REFERENCE_TILE_PX = 24
_PAGE_REFERENCE_PERCENTILE = 70.0
_MIN_DARKENING = 5.0
_DARKENING_FRACTION = 0.20
_CHROMA_CHANGE = 0.20
_VISIBLE_TILE_FRACTION = 0.85
# Um objeto claro ainda pode ter a luma e o croma do papel. Compare a
# variação relativa local com a textura de papel aprendida por tile.
_TEXTURE_CHANGE = 0.015
_TEXTURE_MIN_PAPER_FRACTION = 0.6
# Fração de tinta lida pelo modelo num tile cuja referência foi
# formada em papel: marca nova sobre papel é oclusão suspeita, não
# textura aprendida.
_TEXTURE_NEW_INK_FRACTION = 0.01
_TEXTURE_NEW_INK_MAX_FRACTION = 0.18
# Fração de pixels de tinta em LUGARES NOVOS que já veta o tile: a
# régua com ticks menores que o traço derrubava a fração total e o
# veto por fração não disparava — comparar o total não vê que a tinta
# mudou de lugar. Marcas novas fora da máscara aprendida (dilatada
# para o jitter de borda) são tinta do oclusor, não apagamento
# parcial (Devin Review do PR #144).
_TEXTURE_NEW_MARK_FRACTION = 0.02
_TEXTURE_MARK_DILATE = 3
# A textura aprendida num tile de borda de lâmpada vem de um degrau
# de luma: a maioria dos pixels fica abaixo da média do tile (o lado
# escuro domina). Textura de impressão é o contrário — marcas
# esparsas escuras sobre papel claro, minoria abaixo da média. O
# resgate do mismatch pelo deslocamento de luz só vale para textura
# de degrau (Devin Review do PR #144).
_TEXTURE_STEP_DARK_SHARE = 0.6
# Um deslocamento fotométrico da página inteira (luz da sala caindo,
# auto-exposição da câmera, balanço de branco) move a quase totalidade
# dos tiles de uma vez — não é oclusão local. Quando a parcela anômala
# dos tiles avaliáveis chega a _GLOBAL_SHIFT_FRACTION as referências
# cedem ao nível novo, em vez de congelar a página num único bloco:
# a referência que só sobe não distingue uma queda global de >20% de
# um occluder, e sem esta saída a página inteira virava veto
# (Devin Review do PR #144).
_GLOBAL_SHIFT_FRACTION = 0.8
_GLOBAL_SHIFT_MIN_TILES = 4
_GLOBAL_SHIFT_MIN_RATIO = 0.25
_GLOBAL_SHIFT_MAX_RATIO = 4.0
# Bloco mínimo que veta adições: componente conexo de tiles anômalos
# com os dois lados >= 3 e área >= 9 (um traço fino não é oclusor).
_BLOCK_MIN_TILE_SIDE = 3
_BLOCK_MIN_TILE_AREA = 9
# Piso de luma no denominador do croma: pixels quase pretos produzem
# croma infinito; 20 é o valor embutido no teto da referência.
_CHROMA_LUMA_FLOOR = 20.0
# O croma medido é canal / grayscale (máx. 255/20), não canal /
# canal-máximo: iluminação colorida legítima passa de 2 — papel
# neutro sob luz azul marca ~2.5 no canal B. Um teto de 2 na
# referência deixava um resíduo >= _CHROMA_CHANGE e a página
# inteira ficava vetada enquanto a luz durasse (Devin Review do
# PR #144).
_CHROMA_REFERENCE_MAX = 255.0 / _CHROMA_LUMA_FLOOR
# Recuperação por preenchimento: um bloco escuro que o modelo lê como
# tinta, passada após passada e com a mão fora do tile, é conteúdo da
# página — não oclusor. Até _FILL_RECOVERY_PASSES seguidas o veto de
# adição vale como sempre; depois ele cai só para o voto de adição.
# A remoção segue exigindo papel exposto, que o preenchimento nunca
# produz: o tile continua anômalo para o segundo veredito.
_FILL_INK_PIXEL_THRESHOLD = 0.7
_FILL_INK_TILE_SUPPORT = 0.15
_FILL_INK_TILE_MEAN = 0.25
# Semente densa do componente: o teste 2-D mede o retângulo do
# componente, e duas faixas finas se encontrando num canto (o "L" da
# sombra) passavam por preenchimento. A semente exige um blob de
# suporte que ocupe bem o PRÓPRIO retângulo: um traço ou contorno
# fino de tinta enche o bbox (contorno de 4px ~34%, traço ~100%),
# enquanto o L oco das faixas de sombra fica em ~18%. Faixas finas
# só podem ESTENDER a partir de uma semente, não semear sozinhas
# (Devin Review do PR #144).
_FILL_SEED_MIN_DIM_PX = 4
_FILL_SEED_MIN_FILL = 0.28
# Fração da dimensão do bbox que uma corrida contínua precisa ter
# para contar como banda. O que veta o L de canto não é o tamanho
# da corrida — a faixa de 72 px cruza o bbox inteiro — é o número
# de grupos disjuntos: 1 por eixo no L, 3 no zigzag multi-volta.
_SEED_PARALLEL_RUN = 0.8
# Caducidade do veto por posição: marcas novas persistentes em tile
# que segue provando papel (luma e textura da referência) são edição
# legítima — apagou e redesenhou — não capa de oclusor. Sem o prazo o
# traço antigo nunca terminaria a remoção; o buraco residual aceito é
# a capa imóvel fotometricamente idêntica ao papel (Devin Review do
# PR #144). 12 passadas = 2x a janela de remoção.
_MARKS_HOLD_MAX = 12
_FILL_RECOVERY_PASSES = 4
# Piso absoluto de papel exposto: a absorção de deslocamento global
# segue a iluminação, mas uma página mediana abaixo deste nível não
# pode ser papel — a lente tapada/luz apagada é um oclusor, não um
# novo ponto de referência. Sem o piso, um frame escuro inteiro era
# adotado como baseline e a tinta confirmada era apagada.
_MIN_PAPER_LUMA = 24.0
# Papel que acabou de sair de anomalia fica em recuperação por este
# número de passadas: a evidência voltou mas ainda não estabilizou —
# uma anomalia persistente nunca entra em recuperação (espelha o hold
# de `paper_ok` do guarda anterior, agora por tile).
_RECOVERY_HOLD_PASSES = 4
# Teto do tempo morto da espera da trabalhadora por recuperação
# pendente (streak de preenchimento ou hold de recuperação): cobre os
# dois orçamentos — hoje coincidem em 4, mas pertencem a mecanismos
# diferentes. O teto não corre sobre o pendente global direto: a
# trabalhadora soma a idade do item pendente mais velho
# (max_recovery_age) como carência, então o limite real é o teto
# mais a idade dele — e como idade pendente é sempre menor que este
# valor, a espera total nunca passa de 2*_RECOVERY_WAIT_PASSES - 1.
_RECOVERY_WAIT_PASSES = max(_FILL_RECOVERY_PASSES, _RECOVERY_HOLD_PASSES)


def _has_dense_pixel_seed(region):
    """Whether a pixel-support region contains a raster-sized dense blob."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        region.astype(np.uint8), connectivity=8
    )
    pixel_corners = np.array(
        [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)], dtype=np.float32
    )
    for blob_index in range(1, count):
        blob_x, blob_y, blob_width, blob_height, blob_area = stats[blob_index]
        if min(blob_width, blob_height) < _FILL_SEED_MIN_DIM_PX:
            continue
        blob = (
            labels[blob_y : blob_y + blob_height, blob_x : blob_x + blob_width]
            == blob_index
        )
        contours, _ = cv2.findContours(
            blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        hull = cv2.convexHull(contours[0]).reshape(-1, 2).astype(np.float32)
        # Keep the original center rectangle as the calibrated density
        # denominator; the 28% threshold was tuned against that convention.
        (_, _), (center_w, center_h), _ = cv2.minAreaRect(hull)
        # For the minimum-size gate, treat each raster pixel as its unit
        # square. Expanding hull vertices gives the outer extent without
        # materializing four points for every pixel.
        corners = (hull[:, None, :] + pixel_corners).reshape(-1, 2)
        (_, _), (pixel_w, pixel_h), _ = cv2.minAreaRect(corners)
        if (
            min(pixel_w, pixel_h) >= _FILL_SEED_MIN_DIM_PX
            and blob_area >= _FILL_SEED_MIN_FILL * center_w * center_h
            and _closed_or_solid_seed(blob, blob_area, hull)
        ):
            return True
    return False


def _closed_or_solid_seed(blob, blob_area, hull):
    """A forma, não a densidade, separa a semente real da miragem.

    Encurtando as faixas, o L de canto atinge a mesma fração do bbox
    de um contorno fechado ou de um traço sólido (~28%), e a medida
    no retângulo de centros decide sozinha por uma borda de um
    pixel. O L é côncavo e aberto — deixa mais da metade do próprio
    hull vazia; o traço ou o blob sólido preenche a maior parte do
    hull; e o contorno fino, que também deixa o hull vazio, fecha um
    buraco. Qualquer um dos dois basta — e quando nenhum dos dois se
    aplica, o traço multi-volta ainda se distingue da borda de
    sombra pelo número de bandas paralelas disjuntas (Devin Review
    do PR #144)."""
    if blob_area >= 0.5 * cv2.contourArea(hull):
        return True
    # O bbox é justo, então o blob toca as quatro bordas: o rótulo 0
    # (o próprio blob) sempre aparece na borda do mapa de rótulos do
    # fundo, e qualquer rótulo de fundo ausente da borda é um buraco
    # fechado pelo contorno.
    labels = cv2.connectedComponents((~blob).astype(np.uint8), connectivity=4)[1]
    edge = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    if np.any(~np.isin(labels, edge)):
        return True
    return _parallel_band_seed(blob)


def _parallel_band_seed(blob):
    """Duas ou mais bandas longas disjuntas num mesmo eixo.

    A banda única e o canto em L seguem ambíguos com a borda da
    sombra — ficam vetados. Faixas paralelas separadas por papel
    são a evidência de traço multi-volta (o zigzag do review, três
    grupos na horizontal). O critério aceita um falso positivo
    residual — um artefato em pente (vão entre dedos, persiana,
    sombra de régua) também semeia — porque o custo é adiar o veto
    de adição num bloco que o modelo insiste em ler como tinta,
    enquanto o falso negativo congelava o traço real para sempre
    (Devin Review do PR #144)."""
    for axis in range(2):
        grid = blob if axis == 0 else blob.T
        limit = grid.shape[1] * _SEED_PARALLEL_RUN
        long_run = np.zeros(grid.shape[0], dtype=bool)
        for index, row in enumerate(grid):
            padded = np.concatenate(([False], row, [False]))
            diff = np.diff(padded.astype(np.int8))
            starts = np.flatnonzero(diff == 1)
            if starts.size:
                ends = np.flatnonzero(diff == -1)
                long_run[index] = (ends - starts).max() >= limit
        edges = np.diff(np.concatenate(([False], long_run, [False])).astype(np.int8))
        if np.count_nonzero(edges == 1) >= 2:
            return True
    return False


class PaperExposureGuard:
    """Track the brightest observed paper level and reject darkened tiles.

    The page-clean model can report a confirmed stroke as absent when a
    hand or shadow covers it. This guard uses the warped camera image, not
    the hand model, and turns such regions into unknown evidence. Its
    brightness reference only moves upward, so a brief shadow cannot become
    the new baseline — except when nearly the whole visible page changes
    at once, which is a global illumination shift, not an occluder: then
    the references follow the scene instead of blocking the page. A coarse
    chroma reference also rejects objects whose brightness resembles the
    paper. The low-resolution references keep the per-pass cost small and
    make fine pen strokes too small to trigger the guard.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._reference = None
        self._color_reference = None
        self._page_reference = None
        self._page_color_reference = None
        self._texture_reference = None
        self._ink_streak = None
        self._fill_adopted = None
        self._last_fill_adoption = False
        self._recovering = None
        self._was_anomalous = None
        self._was_texture_anomaly = None
        self._texture_ink_mask = None
        self._texture_dark_share = None
        self._marks_hold_age = None
        self._marks_pending_age = 0
        self._occluded_confirmed = None
        self._anomalous_px = None
        self._suspect_ink = None

    @property
    def pending_fill_recovery(self):
        """Streak de evidência de tinta contando num bloco (ainda não
        adotado): a recuperação por preenchimento tem trabalho pendente
        e precisa de passadas limpas chegando — a porta de descanso não
        pode ler essas passadas como quietude, senão a trabalhadora
        dorme no meio da contagem e o preenchimento nunca vota (Devin
        Review do PR #144)."""
        return self._ink_streak is not None and bool(
            np.any((self._ink_streak > 0) & (self._ink_streak < _FILL_RECOVERY_PASSES))
        )

    @property
    def last_fill_adoption(self):
        """True apenas na passada em que um bloco cruzou o limiar de
        adoção: é uma borda, não um estado — a trabalhadora rearma a
        janela de confirmação uma vez por preenchimento adotado, sem
        segurar o orçamento enquanto o bloco seguir adotado."""
        return self._last_fill_adoption

    @property
    def recovering(self):
        """Algum tile saiu de anomalia e ainda está no hold de recuperação.

        Espelha o `recovering_px` do guarda anterior: dispara quando a
        evidência fotométrica VOLTA (sombra que sai, mão que levanta),
        não enquanto a anomalia persiste — uma borda escura permanente
        nunca entra em recuperação e deixa a trabalhadora descansar.
        """
        return self._recovering is not None and bool(np.any(self._recovering))

    @property
    def recovery_pending(self):
        """Recuperação com trabalho pendente (streak ou hold ativo)."""
        return self.pending_fill_recovery or self.recovering

    @property
    def anomalous_px(self):
        """Tiles anômalos de luma/croma na resolução de pixel da
        última passada observada, ou None antes da primeira. Sob eles
        a evidência de retorno à aparência semeada não abre remoção —
        a régua escura que imita a guia fica vetada pelo tile
        (Devin Review do PR #144)."""
        return self._anomalous_px

    @property
    def suspect_ink(self):
        """Tinta lida fora das posições aprendidas e das isentas por
        confirmação sob anomalia: candidata a marca de oclusor. Tile
        denso demais para amostrar papel (<60%) nunca aprende as
        próprias marcas — a impressão semeada ali segue lendo tinta e
        cabe ao chamador descontá-la antes de vetar (Devin Review do
        PR #144). None antes da primeira observação."""
        return self._suspect_ink

    @property
    def max_marks_recovery_age(self):
        """Oldest clean position veto still waiting for observations.

        Dirty or no-longer-suspect tiles reset their age to zero. Keep
        this countdown separate from the shorter fill recovery budget.
        """
        return self._marks_pending_age

    @property
    def max_recovery_age(self):
        """Idade do item pendente mais velho, em passadas; 0 sem pendente.

        `recovery_pending` é global: um streak que nasce na passada em
        que outro morre emenda os dois numa espera só, e a trabalhadora
        cobrava do preenchimento novo as passadas gastas no que morreu
        — a contagem fechava antes da leitura que o adota (Devin Review
        do PR #144). A idade devolve a régua por item: o maior streak
        ainda contando (o próprio valor) ou o hold decorrido há mais
        tempo (_RECOVERY_HOLD_PASSES - restante). Todo item pendente é
        mais novo que _RECOVERY_WAIT_PASSES, então a carência que a
        trabalhadora soma ao teto da espera é limitada por construção.
        """
        age = 0
        if self._ink_streak is not None:
            streaks = self._ink_streak[
                (self._ink_streak > 0) & (self._ink_streak < _FILL_RECOVERY_PASSES)
            ]
            if streaks.size:
                age = int(streaks.max())
        if self._recovering is not None:
            holds = self._recovering[self._recovering > 0]
            if holds.size:
                age = max(age, _RECOVERY_HOLD_PASSES - int(holds.min()))
        return age

    def _darkened(self, known, current):
        """(darkened, expected): luma por tile contra a referência."""
        expected = np.where(known, self._reference, self._page_reference or 0.0)
        drop = np.maximum(_MIN_DARKENING, expected * _DARKENING_FRACTION)
        return current < expected - drop, expected

    def _color_anomaly(self, known, chroma, shape):
        """Tiles com croma longe da referência; zeros sem evidência."""
        color_anomaly = np.zeros(shape, dtype=bool)
        if chroma is None or self._page_color_reference is None:
            return color_anomaly
        expected_color = (
            np.where(
                known[..., None],
                self._color_reference,
                self._page_color_reference,
            )
            if self._color_reference is not None
            else np.broadcast_to(self._page_color_reference, chroma.shape)
        )
        return np.linalg.norm(chroma - expected_color, axis=2) > _CHROMA_CHANGE

    def observe(
        self,
        image,
        visible=None,
        ink_prob=None,
        ink_support_threshold=None,
        confirmed_ink=None,
    ):
        """Return separate masks for safe additions and paper exposure.

        Only uint8 camera images carry reliable photometric evidence. An
        unsupported image fails closed for erasure at the accumulator
        boundary, while preserving existing callers that provide only ink
        probabilities. A thin mark can change paper luminance without
        looking like an object block: it remains available for drawing,
        but its pixels still cannot prove that prior ink disappeared.

        `ink_prob` (o mapa de probabilidade desta passada, na mesma
        grade) alimenta a recuperação por preenchimento: um bloco que o
        modelo lê como tinta por _FILL_RECOVERY_PASSES passadas, sempre
        com o tile à vista, deixa de vetar adições. Sem o mapa o veto é
        permanente, como antes. `ink_support_threshold` é o nível de
        prob em que um pixel conta como tinta forte; o chamador passa o
        `vote_hit` da sua configuração para que a guarda some exatamente
        a evidência que a camada de voto consegue confirmar — com o
        piso fixo de 0,7 um preenchimento sustentado em [vote_hit, 0,7)
        confirmava no voto mas nunca chegava a votar (Devin Review do
        PR #144). None mantém o piso histórico.
        """
        # Borda de adoção dura uma observação: qualquer retorno —
        # precoce ou completo — apaga a do ciclo anterior antes de
        # recomputar, senão uma passada sem imagem válida a repetia
        # e a trabalhadora rearmava a janela para sempre.
        self._last_fill_adoption = False
        self._marks_pending_age = 0
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            return None
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
            return None

        height, width = image.shape[:2]
        grid_width = max(1, (width + _REFERENCE_TILE_PX - 1) // _REFERENCE_TILE_PX)
        grid_height = max(1, (height + _REFERENCE_TILE_PX - 1) // _REFERENCE_TILE_PX)
        size = (grid_width, grid_height)
        reduced = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if image.ndim == 3:
            current = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY).astype(np.float32)
            color = reduced.astype(np.float32)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            current = reduced.astype(np.float32)
            color = None
            gray = image.astype(np.float32)
        texture_weights = np.ones((height, width), dtype=np.float32)
        support_level = (
            _FILL_INK_PIXEL_THRESHOLD
            if ink_support_threshold is None
            else float(ink_support_threshold)
        )
        if ink_prob is not None:
            ink_prob = np.asarray(ink_prob, dtype=np.float32)
            if ink_prob.shape == (height, width):
                texture_weights = (
                    np.isfinite(ink_prob) & (ink_prob < support_level)
                ).astype(np.float32)
            else:
                texture_weights.fill(0.0)
        paper_fraction = cv2.resize(texture_weights, size, interpolation=cv2.INTER_AREA)
        ink_fraction = 1.0 - paper_fraction
        sampled_mean = cv2.resize(
            gray * texture_weights, size, interpolation=cv2.INTER_AREA
        )
        sampled_mean_square = cv2.resize(
            gray * gray * texture_weights, size, interpolation=cv2.INTER_AREA
        )
        texture_mean = np.divide(
            sampled_mean,
            paper_fraction,
            out=np.zeros_like(sampled_mean),
            where=paper_fraction > 0.0,
        )
        texture_mean_square = np.divide(
            sampled_mean_square,
            paper_fraction,
            out=np.zeros_like(sampled_mean_square),
            where=paper_fraction > 0.0,
        )
        texture_valid = paper_fraction >= _TEXTURE_MIN_PAPER_FRACTION
        texture = np.sqrt(
            np.maximum(texture_mean_square - texture_mean * texture_mean, 0.0)
        ) / np.maximum(texture_mean, _CHROMA_LUMA_FLOOR)
        # Parcela dos pixels amostrados abaixo da média do tile: degrau
        # de luma (lâmpada) tem o lado escuro dominante; impressão é
        # marca escura esparsa sobre fundo claro.
        tile_mean_px = cv2.resize(
            texture_mean, (width, height), interpolation=cv2.INTER_NEAREST
        )
        dark_share = np.divide(
            cv2.resize(
                ((gray < tile_mean_px) & (texture_weights > 0)).astype(np.float32),
                size,
                interpolation=cv2.INTER_AREA,
            ),
            paper_fraction,
            out=np.zeros_like(texture_mean),
            where=paper_fraction > 0.0,
        )

        if visible is None:
            updateable = np.ones(current.shape, dtype=bool)
        else:
            visible = np.asarray(visible, dtype=bool)
            if visible.shape != (height, width):
                # Máscara em geometria errada é bug do chamador: sem
                # veredito confiável devolve None (remoção congela no
                # acumulador) em vez de congelar a referência e ainda
                # varrer os streaks/adotados de preenchimento.
                return None
            if np.all(visible):
                updateable = np.ones(current.shape, dtype=bool)
            else:
                fraction = cv2.resize(
                    visible.astype(np.float32), size, interpolation=cv2.INTER_AREA
                )
                updateable = fraction >= _VISIBLE_TILE_FRACTION

        if confirmed_ink is not None:
            confirmed_ink = np.asarray(confirmed_ink, dtype=bool)
            if confirmed_ink.shape != (height, width):
                return None

        if self._reference is None or self._reference.shape != current.shape:
            self._reference = np.full(current.shape, np.nan, dtype=np.float32)
            self._color_reference = (
                None
                if color is None
                else np.full((*current.shape, 3), np.nan, dtype=np.float32)
            )
            self._page_reference = None
            self._page_color_reference = None
            self._texture_reference = np.full(current.shape, np.nan, dtype=np.float32)
            self._texture_learned_clean = np.zeros(current.shape, dtype=bool)
            self._texture_ink_baseline = np.full(
                current.shape, np.nan, dtype=np.float32
            )
            self._texture_ink_mask = np.zeros((height, width), dtype=bool)
            self._texture_dark_share = np.full(current.shape, np.nan, dtype=np.float32)
            self._was_texture_anomaly = np.zeros(current.shape, dtype=bool)
            self._marks_hold_age = np.zeros(current.shape, dtype=np.uint8)
            self._ink_streak = np.zeros(current.shape, dtype=np.uint8)
            self._fill_adopted = np.zeros(current.shape, dtype=bool)
            self._recovering = np.zeros(current.shape, dtype=np.uint8)
            self._was_anomalous = np.zeros(current.shape, dtype=bool)
            self._occluded_confirmed = np.zeros((height, width), dtype=bool)

        known = np.isfinite(self._reference)
        candidates = current[updateable]
        # Invariante explícita: sem tiles à vista não há estimativa —
        # o deslocamento global só roda com shifted >= 4, o que já
        # implica candidates não-vazio, mas o nome fica definido sempre.
        page_reference = 0.0
        if candidates.size:
            page_reference = float(
                np.percentile(candidates, _PAGE_REFERENCE_PERCENTILE)
            )
            if self._page_reference is None:
                self._page_reference = page_reference
            else:
                self._page_reference = max(self._page_reference, page_reference)

        # A tile with no clean history is compared to the page's robust
        # brightness estimate. This catches a hand already present on the
        # first sampled frame, before its darkness can become its baseline.
        darkened, expected = self._darkened(known, current)
        chroma = None
        if color is not None:
            chroma = color / np.maximum(current[..., None], _CHROMA_LUMA_FLOOR)
            color_tiles = updateable & ~darkened
            if np.any(color_tiles):
                self._page_color_reference = np.median(
                    chroma[color_tiles], axis=0
                ).astype(np.float32)
        color_anomaly = self._color_anomaly(known, chroma, current.shape)

        texture_known = np.isfinite(self._texture_reference)
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = np.roll(self._texture_reference, (dy, dx), axis=(0, 1))
                if dy < 0:
                    neighbor[-1, :] = np.nan
                elif dy > 0:
                    neighbor[0, :] = np.nan
                if dx < 0:
                    neighbor[:, -1] = np.nan
                elif dx > 0:
                    neighbor[:, 0] = np.nan
                neighbors.append(neighbor)
        neighbor_values = np.stack(neighbors)
        neighbor_valid = np.isfinite(neighbor_values)
        neighbor_count = neighbor_valid.sum(axis=0)
        ordered_neighbors = np.sort(
            np.where(neighbor_valid, neighbor_values, np.inf), axis=0
        )
        lower = np.take_along_axis(
            ordered_neighbors,
            np.maximum((neighbor_count - 1) // 2, 0)[None],
            axis=0,
        )[0]
        upper = np.take_along_axis(
            ordered_neighbors, (neighbor_count // 2)[None], axis=0
        )[0]
        neighbor_reference = np.where(neighbor_count > 0, (lower + upper) * 0.5, 0.0)
        texture_reference = np.where(
            texture_known, self._texture_reference, neighbor_reference
        )
        texture_comparable = texture_known | (neighbor_count >= 2)
        texture_mismatch = texture_comparable & (
            np.abs(texture - texture_reference)
            > np.maximum(_TEXTURE_CHANGE, texture_reference * 0.5)
        )
        # A textura atual combina com a mediana dos vizinhos: usada
        # só no rebaseline de tiles JÁ escuros demais para o nível
        # novo — uma superfície que cai junto com a página nunca é
        # anomalous e não passa por aqui (Devin Review do PR #144).
        local_texture_match = (neighbor_count >= 2) & (
            np.abs(texture - neighbor_reference)
            <= np.maximum(_TEXTURE_CHANGE, neighbor_reference * 0.5)
        )
        # Tinta nova sobre tile aprendida limpa: as marcas de um
        # oclusor podem ser lidas como tinta e sair da amostra
        # ponderada — a régua passava com textura de papel e a falta
        # apagava o traço sob ela. O veto compara com a fração de
        # tinta gravada quando a referência foi formada: a régua
        # soma tinta sobre uma baseline limpa e veta; a metade
        # apagada de um traço retido fica ABAIXO da baseline e os
        # pixels limpos votam a remoção — tinta confirmada por voto
        # não vira papel, e tinta retida não é "nova" (Devin Review
        # do PR #144). Cobertura insuficiente barra só o aprendizado
        # (texture_valid ali), não o veto por pixel.
        # Tinta em lugares novos: a máscara aprendida dilatada cobre o
        # jitter de borda do traço retido; pixels de tinta fora dela
        # são marcas do oclusor (o tick da régua) mesmo quando a fração
        # total CAI — cobertura não é o teste, posição é (Devin Review
        # do PR #144).
        ink_mask = texture_weights == 0.0
        learned_marks = cv2.dilate(
            self._texture_ink_mask.astype(np.uint8),
            np.ones((2 * _TEXTURE_MARK_DILATE + 1,) * 2, dtype=np.uint8),
        )
        # Tinta confirmada sob tile anômalo é conteúdo provado sob
        # oclusão, nunca marca de oclusor: o traço que viveu sob tile
        # denso demais para aprender a própria tinta virava "marca
        # nova" para sempre quando a anomalia caía — a remoção
        # congelava no veto por posição (merge do #161 no PR #144).
        # A isenção NÃO vale para tinta confirmada em tile limpo: o
        # tick da régua confirmado por voto se isentaria do próprio
        # veto e liberaria a remoção do traço sob ele — a cobertura
        # confirmada só conta nas passadas em que o tile era a
        # anomalia (atualizada no fim desta passada).
        confirmed_cover = np.zeros((height, width), dtype=bool)
        if confirmed_ink is not None:
            confirmed_cover = (
                cv2.dilate(
                    confirmed_ink.astype(np.uint8),
                    np.ones((2 * _TEXTURE_MARK_DILATE + 1,) * 2, dtype=np.uint8),
                )
                > 0
            )
        exempt = self._occluded_confirmed
        suspect_ink = ink_mask & (learned_marks == 0) & ~exempt
        new_marks_fraction = cv2.resize(
            suspect_ink.astype(np.float32),
            size,
            interpolation=cv2.INTER_AREA,
        )
        # A fração acima da baseline compara a tinta que a tinta
        # confirmada sob anomalia não explica: sem o desconto, o traço
        # sobrevivente de uma anomalia segurava o veto de fração
        # sozinho. Marcas novas confirmadas em tile limpo seguem
        # contando — o veto delas caduca pelo caminho do marks_hold.
        excess_fraction = cv2.resize(
            (ink_mask & ~exempt).astype(np.float32),
            size,
            interpolation=cv2.INTER_AREA,
        )
        # Marcas novas persistentes em tile que SEGUE provando papel
        # (luma e textura da referência intactas) não são capa de
        # oclusor — são edição: o usuário apagou e redesenhou. O veto
        # por posição caduca em _MARKS_HOLD_MAX passadas seguidas, senão
        # o traço antigo nunca completaria a remoção ao lado da marca
        # nova. O contador só anda em tile limpo — superfície estranha
        # (anomalia, mismatch, textura incomparável) segura o veto
        # sem prazo (Devin Review do PR #144).
        paper_clean = (
            texture_comparable & ~texture_mismatch & ~darkened & ~color_anomaly
        )
        marks_fire = (
            texture_known
            & self._texture_learned_clean
            & (new_marks_fraction > _TEXTURE_NEW_MARK_FRACTION)
            & (ink_fraction <= _TEXTURE_NEW_INK_MAX_FRACTION)
        )
        self._marks_hold_age = np.where(
            marks_fire & paper_clean,
            np.minimum(self._marks_hold_age.astype(np.uint16) + 1, 255),
            0,
        ).astype(np.uint8)
        marks_hold = marks_fire & ~(
            paper_clean & (self._marks_hold_age >= _MARKS_HOLD_MAX)
        )
        if confirmed_ink is not None:
            # New additions alone need no extra removal observations. Only
            # keep sampling tiles where confirmed ink is currently missing.
            missing = (
                cv2.resize(
                    (confirmed_ink & ~ink_mask).astype(np.float32),
                    size,
                    interpolation=cv2.INTER_AREA,
                )
                > 0
            )
            pending_ages = self._marks_hold_age[marks_hold & paper_clean & missing]
            if pending_ages.size:
                self._marks_pending_age = int(pending_ages.max())
        new_ink = (
            texture_known
            & self._texture_learned_clean
            & (
                (
                    (
                        excess_fraction
                        > self._texture_ink_baseline + _TEXTURE_NEW_INK_FRACTION
                    )
                    | marks_hold
                )
                & (ink_fraction <= _TEXTURE_NEW_INK_MAX_FRACTION)
            )
        )
        texture_anomaly = texture_mismatch | new_ink

        anomalous = darkened | color_anomaly
        flattened = np.zeros(current.shape, dtype=bool)
        updateable_px = int(np.count_nonzero(updateable))
        shifted = int(np.count_nonzero(anomalous & updateable))
        if (
            shifted >= _GLOBAL_SHIFT_MIN_TILES
            and shifted >= (_GLOBAL_SHIFT_FRACTION * updateable_px)
            and page_reference >= _MIN_PAPER_LUMA
        ):
            # Quase toda a página à vista mudou junta: é a iluminação
            # global, não um oclusor. A razão medida nos tiles que
            # mudaram juntos escala TODA a referência conhecida —
            # inclusive a dos tiles escondidos pela mão: ela ficava
            # clara demais e, ao reaparecer no nível novo, virava um
            # bloco escuro permanente. E a referência de página cede
            # ao percentil medido, em vez de ficar presa no máximo
            # histórico (Devin Review do PR #144).
            measured = anomalous & updateable & known
            if np.any(measured):
                # expected ⊆ self._reference para tiles conhecidos:
                # lê a referência direto, sem recomputar o grid.
                ratio = float(np.median(current[measured] / self._reference[measured]))
            elif self._page_reference:
                ratio = page_reference / self._page_reference
            else:
                ratio = 1.0
            ratio = float(
                np.clip(ratio, _GLOBAL_SHIFT_MIN_RATIO, _GLOBAL_SHIFT_MAX_RATIO)
            )
            self._reference[known] *= np.float32(ratio)
            self._page_reference = page_reference
            if self._color_reference is not None:
                # Aplica o deslocamento de croma comum; usar o valor
                # bruto por tile ensinaria junto uma sombra colorida.
                color_known = known & np.all(np.isfinite(self._color_reference), axis=2)
                color_tiles = updateable & color_known
                if np.any(color_tiles):
                    color_shift = np.median(
                        chroma[color_tiles] - self._color_reference[color_tiles], axis=0
                    )
                    self._color_reference[known] = np.clip(
                        self._color_reference[known] + color_shift,
                        0.0,
                        _CHROMA_REFERENCE_MAX,
                    )
                self._page_color_reference = np.median(
                    chroma[updateable], axis=0
                ).astype(np.float32)

            # Absorve apenas o componente comum. Um tile muito mais
            # escuro/colorido que o resto continua anômalo e não prova
            # papel exposto — inclusive sob a sombra durante a mudança.
            darkened, expected = self._darkened(known, current)
            color_anomaly = self._color_anomaly(known, chroma, current.shape)
            anomalous = darkened | color_anomaly

            # A lâmpada direcional apaga e a página nivela no brilho
            # do lado escuro: a razão mediana escala junto o gradiente
            # antigo e a faixa antes mais clara continuava "escura"
            # para sempre — área residual pequena demais para reabrir
            # a absorção e excluída do refresh, que só corrige tiles
            # sem anomalia. O tile à vista que já combina com o nível
            # NOVO da página não é oclusor: reancora a referência no
            # medido e limpa a anomalia. Um oclusor real segue abaixo
            # do nível da página e permanece vetado, assim como o tile
            # marcado só pelo croma (Devin Review do PR #144).
            page_drop = max(_MIN_DARKENING, page_reference * _DARKENING_FRACTION)
            # Reancora tiles cujo brilho mudou junto com a página
            # apenas quando a textura volta a combinar com a
            # referência — do próprio tile, ou dos vizinhos quando o
            # tile já era desproporcionalmente claro: a textura dele
            # era o gradiente da lâmpada, não a folha, e some quando
            # a luz nivela. Uma capa lisa sobre tile impresso cai
            # JUNTO com a página — não é `anomalous` por luma e nem
            # chega aqui: o veto de textura fica em `paper_grid` e
            # segura a remoção sob a capa (Devin Review do PR #144).
            flattened |= (
                updateable
                & anomalous
                & ~color_anomaly
                & ~new_ink
                & (~texture_anomaly | local_texture_match)
                & (current >= page_reference - page_drop)
            )
            # O tile de borda de lâmpada aprendeu textura de gradiente:
            # quando a luz nivela a luma escala junto com a página
            # (não é `anomalous`) mas a textura some — mismatch que o
            # próprio deslocamento criou. Sem resgate ele vetava a
            # remoção para sempre: `flattened` exigia `anomalous` e o
            # refresh exige `~texture_anomaly`. Só a anomalia FRESCA
            # resgata — a capa lisa já era anômala antes do shift e
            # segue vetada (Devin Review do PR #144).
            flattened |= (
                updateable
                & ~anomalous
                & texture_mismatch
                & ~new_ink
                & ~self._was_texture_anomaly
                & (self._texture_dark_share >= _TEXTURE_STEP_DARK_SHARE)
                & (current >= page_reference - page_drop)
            )
            if np.any(flattened):
                self._reference[flattened] = current[flattened]
                if self._color_reference is not None:
                    self._color_reference[flattened] = chroma[flattened]
                # A tile já validado como parte do novo nível global da
                # folha também atualiza sua textura: a iluminação anterior
                # podia criar variação local que some quando a luz nivela.
                self._texture_reference[flattened] = texture[flattened]
                # A baseline de tinta só DESCE no reancoramento: um
                # oclusor que aparece junto com a mudança de luz não
                # grava suas marcas como o normal do tile — sem o
                # mínimo, a régua que chega com o escurecimento
                # global herdaria baseline suja e o veto de tinta
                # nova morreria na mesma passada (Devin Review do
                # PR #144).
                self._texture_ink_baseline[flattened] = np.fmin(
                    self._texture_ink_baseline[flattened], ink_fraction[flattened]
                )
                texture_anomaly[flattened] = False
                anomalous &= ~flattened

        # Recuperação por preenchimento: o streak conta as passadas
        # seguidas em que o tile anômalo, à vista, é lido como tinta
        # pelo modelo. A mão sobre o tile (updateable False) não
        # deixa a contagem andar — a adoção só acontece com a mão
        # fora do caminho ou quando a cena não tem máscara alguma.
        supported = np.zeros(current.shape, dtype=bool)
        pixel_support = None
        # ink_prob já é float32 ndarray aqui (convertido acima, na
        # amostra de textura); a checagem de grade decide o suporte.
        if ink_prob is not None and ink_prob.shape == (height, width):
            pixel_support = ink_prob >= support_level
        if pixel_support is not None and np.any(anomalous):
            support_fraction = cv2.resize(
                pixel_support.astype(np.float32), size, interpolation=cv2.INTER_AREA
            )
            support_mean = cv2.resize(ink_prob, size, interpolation=cv2.INTER_AREA)
            # A média do tile no nível do hit conta evidência densa
            # sustentada que o voto configurado confirmaria (um bloco
            # preenchido a 0,68 sob Sketch) mesmo sem todos os pixels
            # no hit (Devin Review do PR #144).
            tile_support = (
                anomalous
                & updateable
                & (
                    (
                        (support_fraction >= _FILL_INK_TILE_SUPPORT)
                        & (support_mean >= _FILL_INK_TILE_MEAN)
                    )
                    | (support_mean >= support_level)
                )
            )
            # Só um suporte 2-D pode semear a recuperação: o veto de
            # adição só existe para manchas que cobrem >= 2 tiles em
            # ambas as dimensões, então uma faixa de uma linha de tile —
            # a borda de sombra que o modelo lê como tinta — não tinha
            # nada a recuperar e não pode ser adotada como fill
            # (Devin Review do PR #144).
            #
            # A semente é medida no blob de pixels, não na fração por
            # tile: um traço sólido cruzando a fronteira de dois tiles
            # lê ~30% em cada um — indistinguível por fração de uma
            # faixa fina de sombra. O que separa é a forma: a semente
            # precisa de um blob preenchendo _FILL_SEED_MIN_FILL do
            # próprio bbox com >= _FILL_SEED_MIN_DIM_PX de lado menor.
            # O L das faixas de sombra é oco (~18% do bbox) e não semeia;
            # o contorno fino de tinta (~34%) e o traço sólido semeiam.
            if np.any(tile_support):
                tile_h = height / size[1]
                tile_w = width / size[0]
                count, labels, tile_stats, _ = cv2.connectedComponentsWithStats(
                    tile_support.astype(np.uint8), connectivity=8
                )
                for index in range(1, count):
                    x, y, component_width, component_height, _ = tile_stats[index]
                    if component_width < 2 or component_height < 2:
                        continue
                    region = pixel_support[
                        int(y * tile_h) : int(np.ceil((y + component_height) * tile_h)),
                        int(x * tile_w) : int(np.ceil((x + component_width) * tile_w)),
                    ]
                    if not np.any(region):
                        continue
                    # A diagonal solid stroke fills its oriented rectangle;
                    # the hollow L-shaped shadow remains sparse in any angle.
                    if _has_dense_pixel_seed(region):
                        component = labels[
                            y : y + component_height, x : x + component_width
                        ]
                        supported[
                            y : y + component_height, x : x + component_width
                        ] |= component == index
        self._ink_streak[supported] = np.minimum(
            self._ink_streak[supported].astype(np.uint16) + 1, 255
        ).astype(np.uint8)
        self._ink_streak[~supported] = 0
        # Quem perdeu o suporte desaprende a adoção: se a evidência
        # voltar e o tile readotar, a borda dispara de novo — a
        # trabalhadora rearma a janela de confirmação por borda, não
        # por estado retido (Devin Review do PR #144).
        self._fill_adopted &= self._ink_streak > 0
        adopted = self._ink_streak >= _FILL_RECOVERY_PASSES
        self._last_fill_adoption = bool(np.any(adopted & ~self._fill_adopted))
        self._fill_adopted |= adopted

        # The removal test uses every anomalous tile. For additions, require
        # a two-dimensional block before suppressing pixels; a thin dark pen
        # stroke is not an object or a shadow boundary.
        block = np.zeros(anomalous.shape, dtype=np.uint8)
        if np.any(anomalous):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                anomalous.astype(np.uint8), connectivity=8
            )
            for label in range(1, count):
                _x, _y, block_width, block_height, area = stats[label]
                if (
                    block_width >= _BLOCK_MIN_TILE_SIDE
                    and block_height >= _BLOCK_MIN_TILE_SIDE
                    and area >= _BLOCK_MIN_TILE_AREA
                ):
                    block[labels == label] = 1
        if np.any(adopted):
            block[adopted] = 0
        if np.any(block):
            block = cv2.dilate(block, np.ones((3, 3), dtype=np.uint8))

        initialize = updateable & ~known & ~anomalous
        self._reference[initialize] = current[initialize]
        if self._color_reference is not None:
            self._color_reference[initialize] = chroma[initialize]
        refresh = updateable & known & ~anomalous
        np.maximum(self._reference, current, out=self._reference, where=refresh)
        texture_initialize = (
            updateable & texture_valid & ~texture_known & ~anomalous & ~texture_anomaly
        )
        self._texture_reference[texture_initialize] = texture[texture_initialize]
        texture_refresh = (
            updateable & texture_valid & texture_known & ~anomalous & ~texture_anomaly
        )
        self._texture_reference[texture_refresh] = texture[texture_refresh]
        # Referência formada com fundo visível bastante era papel:
        # tinta nova sobre ela é anomalia, não "sempre foi assim". A
        # baseline grava a fração de tinta presente quando a
        # referência (re)aprende — a régua marcada soma tinta sobre
        # a baseline limpa, e o apagamento parcial de um traço
        # retido fica abaixo dela e prova papel nos pixels limpos
        # (Devin Review do PR #144).
        texture_learned = texture_initialize | texture_refresh
        self._texture_learned_clean[texture_learned] = True
        # Tiles reancorados no shift já escreveram baseline em mínimo
        # acima: regravar com a fração corrente SUBIRIA o piso quando o
        # oclusor traz marcas junto com a mudança de luz (Devin Review
        # do PR #144).
        self._texture_ink_baseline[texture_learned & ~flattened] = ink_fraction[
            texture_learned & ~flattened
        ]
        # A máscara aprendida guarda ONDE a tinta estava quando a
        # referência aprendeu — marca em lugar novo é do oclusor mesmo
        # com a fração total menor (régua com tick fino).
        learned_pixels = (
            cv2.resize(
                texture_learned.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        self._texture_ink_mask[learned_pixels] = ink_mask[learned_pixels]
        self._texture_dark_share[texture_learned] = dark_share[texture_learned]

        # A clean first sample can seed and use its own tile immediately;
        # anomalously dark tiles remain unknown and cannot authorize votes.
        # initialize ⊆ updateable, então a grade é "conhecido ou à vista,
        # sem anomalia".
        paper_grid = (known | updateable) & ~anomalous & ~texture_anomaly
        paper_exposed = (
            True
            if paper_grid.all()
            else cv2.resize(
                paper_grid.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        if pixel_support is not None or confirmed_ink is not None:
            expected_pixels = cv2.resize(
                expected, (width, height), interpolation=cv2.INTER_NEAREST
            )
            pixel_drop = np.maximum(
                _MIN_DARKENING, expected_pixels * _DARKENING_FRACTION
            )
            # A tile can look like paper on average even when its edge
            # contains a solid dark fill. Strong ink pixels over dark
            # camera pixels never prove exposed paper, including tiles
            # that are only partially covered by the fill.
            dark_ink = np.zeros((height, width), dtype=bool)
            if pixel_support is not None:
                dark_ink |= pixel_support & (gray < expected_pixels - pixel_drop)
            if confirmed_ink is not None:
                # A confirmed stroke remains non-paper if the model
                # misses it in this pass. Cleared pixels return to the
                # paper luminance and can vote for removal independently.
                dark_ink |= confirmed_ink & (gray < expected_pixels - pixel_drop)
            if np.any(dark_ink):
                if isinstance(paper_exposed, (bool, np.bool_)):
                    paper_exposed = np.full((height, width), paper_exposed, dtype=bool)
                paper_exposed[dark_ink] = False
        unblocked = None
        if np.any(block):
            unblocked = (
                cv2.resize(
                    (block == 0).astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
        if pixel_support is not None and np.any(adopted):
            adopted_pixels = (
                cv2.resize(
                    adopted.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
            recovered = adopted_pixels & pixel_support
            # Borda de preenchimento parcial: a média de prob do tile
            # de borda dilui com os pixels de papel e fica abaixo do
            # piso — o miolo adota e a borda, que é o mesmo traço,
            # ficava vetada para sempre. O piso continua valendo para
            # SEMEAR a adoção; depois dela o veto cai para o
            # componente conexo de tinta escura forte ancorado no
            # adotado — que atravessa as tiles de borda vizinhas. A
            # faixa fina isolada dentro de uma sombra não tem âncora
            # adotada e segue bloqueada (Devin Review do PR #144).
            if np.any(recovered) and np.any(dark_ink):
                # A ponte só atravessa pixels realmente à vista:
                # a mão cobrindo o fosso entre o preenchimento e
                # uma borda de sombra quebrava a conexão pelo mapa
                # cru e a borda votava junto — tinta fantasma
                # (Devin Review do PR #144).
                # A extensão do fill exige leitura de tinta nesta
                # passada. Tinta confirmada mas perdida pelo modelo
                # continua vetando a remoção acima, porém não pode
                # conectar o fill a uma sombra e liberar falsos pixels
                # para adição (Devin Review do PR #144).
                connectable = pixel_support & (gray < expected_pixels - pixel_drop)
                if visible is not None:
                    connectable &= visible
                _count, fill_labels = cv2.connectedComponents(
                    connectable.astype(np.uint8), connectivity=8
                )
                reached = np.unique(fill_labels[recovered])
                reached = reached[reached != 0]
                if reached.size:
                    recovered = recovered | np.isin(fill_labels, reached)
            if np.any(recovered):
                if unblocked is None:
                    unblocked = np.ones((height, width), dtype=bool)
                unblocked[recovered] = True
        # Hold de recuperação: decai um por passada, zera onde a
        # anomalia voltou, e o tile que saiu de anomalia recarrega o
        # hold (a evidência voltando fica aberta _RECOVERY_HOLD_PASSES).
        np.subtract(
            self._recovering, 1, out=self._recovering, where=self._recovering > 0
        )
        self._recovering[anomalous] = 0
        self._recovering[self._was_anomalous & ~anomalous] = _RECOVERY_HOLD_PASSES
        # Snapshot da tinta confirmada enquanto o tile era anômalo:
        # só ela é isenta dos vetos de marca nova na próxima passada.
        # A anomalia tem que ser explicada pela própria tinta —
        # fração acima do teto do veto de marcas, onde o veto nem
        # opera. Uma sombra sobre a régua (anomalia de luma com
        # fração de tinta baixa) NÃO grava: os ticks confirmados
        # continuariam isentos depois que ela saísse e o veto por
        # posição morreria junto com ela (Devin Review do PR #144).
        # Pixels que saíram do canvas perdem a isenção junto com a
        # confirmação.
        occluded_px = (
            cv2.resize(
                (anomalous & (ink_fraction > _TEXTURE_NEW_INK_MAX_FRACTION)).astype(
                    np.uint8
                ),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        self._occluded_confirmed &= confirmed_cover
        self._occluded_confirmed[occluded_px] = confirmed_cover[occluded_px]
        self._anomalous_px = (
            cv2.resize(
                anomalous.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        self._suspect_ink = suspect_ink
        self._was_anomalous = anomalous.copy()
        self._was_texture_anomaly = texture_anomaly.copy()
        return ExposureAssessment(
            visible=unblocked,
            paper_exposed=paper_exposed,
        )
