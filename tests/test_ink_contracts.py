"""Contratos caixa-preta do motor de tinta (entrada -> saida).

Cada teste dirige uma PageCleanWorker de verdade apenas pela API
publica: `new_generation`, `set_hand_presence`, `submit(WarpedPage)`,
`pump` com relogio injetado, `poll`/`poll_display` para as saidas e
`stats`/`telemetry_snapshot` para contagens. Nenhum atributo privado,
constante interna ou helper privado e lido — se um remendo interno
for reescrito ou deletado, estes testes nao tocam nele; so o que o
produto mostra (tinta confirmada e provisoria exibida) e observado.

A cena e sintetica na grade da pagina retificada: a imagem entregue
no WarpedPage e o warp que a camera teria produzido; `visible` e
`coverage` vivem na grade da camera (o quad da pagina dentro de um
frame maior) como no caminho real — a trabalhadora os retifica.

O modelo de tinta e um _ScriptInk injetado por construtor: devolve
probabilidade pelo conteudo do warp (forte onde esta escuro como
tinta, fraco onde esta cinza como pauta/vinco/lapis/mao), o mesmo
papel que o page_clean ONNX tem na passada real.
"""

import dataclasses

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink.config import InkConfig
from draw.ink.worker import PageCleanWorker

# Grade da pagina retificada e do frame de camera. O lado curto da
# grade fica >= 200 px de proposito: abaixo disso a faixa de guarda
# automatica de borda nao se arma (ver resolve_border_guard_px).
H, W = 240, 320
CAM_H, CAM_W = 280, 360
OX, OY = 20, 20
Q0 = np.array(
    [(OX, OY), (OX + W, OY), (OX + W, OY + H), (OX, OY + H)],
    dtype=np.float32,
)

PAPER = 222.0
PAPER_DIM = 150.0
INK = 40.0
HAND_GRAY = 130.0
TABLE_GRAY = 45.0
RULING_GRAY = 185.0
CREASE_GRAY = 168.0
PENCIL_GRAY = 140.0

# Passo do relogio injetado entre passadas: com a mao presente a
# cadencia e a do preset (0,15 s); sem mao (ou sem sinal) a sonda
# ociosa exige ~1,0 s.
_DT_ACTIVE = 0.2
_DT_IDLE = 1.1


class _Clock:
    """Relogio monotonico deterministico injetado na trabalhadora."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


class _ScriptInk:
    """Modelo de tinta falso: probabilidade pelo conteudo do warp.

    Escuro de traco (>~100) vira 0,95; cinzas de pauta, vinco,
    lapis e pele de mao (100..204) viram 0,45 — incerteza como a do
    modelo real sobre conteudo impresso; claro vira 0.
    """

    def ink_prob(self, warped):
        gray = warped if warped.ndim == 2 else warped.mean(axis=2)
        prob = np.zeros(gray.shape, dtype=np.float32)
        prob[gray < 100] = 0.95
        prob[(gray >= 100) & (gray < 205)] = 0.45
        return prob


def _bgr(gray):
    return np.stack([gray] * 3, axis=-1).astype(np.uint8)


class _Scene:
    """Cena sintetica: conteudo da pagina + estado de camera.

    `page` e a luminancia da folha (com tinta, pauta, vinco, mesa
    invadindo). `corners` e o quad que retificou o warp, na grade da
    camera. `hand` marca a oclusao na grade da pagina; `coverage`
    marca onde o quad cobre papel de verdade.
    """

    def __init__(self, paper=PAPER):
        self.page = np.full((H, W), paper, dtype=np.float32)
        self.corners = Q0.copy()
        self.hand = np.zeros((H, W), dtype=bool)
        self.table = np.zeros((H, W), dtype=bool)
        self.send_coverage = True

    def draw(self, rect, gray=INK):
        y0, y1, x0, x1 = rect
        self.page[y0:y1, x0:x1] = gray

    def hand_on(self, rect):
        y0, y1, x0, x1 = rect
        self.hand[y0:y1, x0:x1] = True

    def hand_off(self):
        self.hand[:] = False

    def jitter(self, dx, dy):
        """Quad desloca (dx, dy) na camera; o warp traz a banda de
        mesa que o deslize admitiu na borda oposta."""
        self.corners = Q0 + np.array([dx, dy], dtype=np.float32)
        shifted = np.full_like(self.page, TABLE_GRAY)
        y0, y1 = max(0, -dy), min(H, H - dy)
        x0, x1 = max(0, -dx), min(W, W - dx)
        if y1 > y0 and x1 > x0:
            shifted[y0:y1, x0:x1] = self.page[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        self.table = shifted == TABLE_GRAY
        return shifted

    def warped(self):
        img = self.page.copy()
        img[self.table] = TABLE_GRAY
        img[self.hand] = HAND_GRAY
        return img

    def cam_mask(self, page_mask, fill):
        """Mascara de grade de pagina colada no quad atual da camera."""
        mask = np.full((CAM_H, CAM_W), fill, dtype=bool)
        ox = int(round(float(self.corners[0][0])))
        oy = int(round(float(self.corners[0][1])))
        mask[oy : oy + H, ox : ox + W] = page_mask
        return mask

    def page_frame(self, generation):
        visible = self.cam_mask(~self.hand, fill=True)
        coverage = (
            self.cam_mask(~self.table, fill=False) if self.send_coverage else None
        )
        return WarpedPage(
            _bgr(self.warped()), generation, self.corners, visible, coverage
        )


# ------------------------------------------------------------------
# Dirigir a trabalhadora: so API publica.
# ------------------------------------------------------------------


def _worker(config):
    clock = _Clock()
    return clock, PageCleanWorker(_ScriptInk(), clock=clock, config=config)


def _drive(worker, clock, page, dt=_DT_ACTIVE):
    """Uma submissao + avanco de relogio + passada (se a cadencia deixar)."""
    worker.submit(page)
    clock.advance(dt)
    return worker.pump()


def _passes(worker, clock, generation, scene, n, dt=_DT_ACTIVE):
    """n ciclos submit/pump sobre a cena corrente."""
    for _ in range(n):
        _drive(worker, clock, scene.page_frame(generation), dt)


def _confirmed(worker, generation):
    snap = worker.poll(generation, 0)
    if snap is None:
        return np.zeros((H, W), dtype=np.uint8)
    return np.asarray(snap.confirmed)


def _display(worker, generation):
    snap = worker.poll_display(generation, 0)
    if snap is None:
        return (
            np.zeros((H, W), dtype=np.uint8),
            np.zeros((H, W), dtype=bool),
            0,
        )
    return np.asarray(snap.confirmed), np.asarray(snap.provisional), snap.version


def _rect_mask(rect, shape=(H, W)):
    y0, y1, x0, x1 = rect
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _write_stroke(worker, clock, generation, scene, stroke):
    """Traço escrito com a mao presente e revelado na saida dela."""
    scene.draw(stroke)
    worker.set_hand_presence(True, moved=True)
    _passes(worker, clock, generation, scene, 4)
    worker.set_hand_presence(False)
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)


def _dilate(mask, px):
    out = mask.copy()
    for _ in range(px):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


# Regimes sob medicao. "production" e o preset do PipelineConfig —
# desde o PR que promoveu o PageModel, ele ja carrega o modelo
# persistente com os remendos #129-#132 desligados. "legacy_patches"
# reconstrui a pilha anterior (pauta por DP, guarda de papel, guarda
# de borda auto, sem PageModel): os mesmos 8 contratos precisam
# sustentar-se nos dois regimes para a troca ser segura.
PRODUCTION = InkConfig.production()
LEGACY_PATCHES = dataclasses.replace(
    PRODUCTION,
    page_model=False,
    ruling_removal=True,
    paper_guard=True,
    border_guard_px=None,
)

REGIMES = {
    "production": PRODUCTION,
    "legacy_patches": LEGACY_PATCHES,
}


@pytest.fixture(params=list(REGIMES), ids=list(REGIMES))
def regime(request):
    return request.param


@pytest.fixture
def ink(regime):
    clock, worker = _worker(REGIMES[regime])
    yield worker, clock
    try:
        worker.stop()
    except RuntimeError:
        # Modo pump(): a thread nunca foi iniciada e stop() tenta o
        # join do mesmo jeito — sem thread para esperar, nada a fazer.
        pass


# ------------------------------------------------------------------
# Contrato 1: mao sobre traco confirmado preserva o desenho.
# ------------------------------------------------------------------


def test_hand_over_confirmed_stroke_preserves_it(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)

    stroke = (90, 96, 60, 260)  # traco real confirmado (linha ~6 px)
    _write_stroke(worker, clock, generation, scene, stroke)
    want = _rect_mask(stroke)
    got = _confirmed(worker, generation) > 0
    assert np.count_nonzero(got & want) >= 0.95 * np.count_nonzero(want)

    # A mao estaciona sobre o traco (mascara correta): sem evidencia,
    # a regiao congela — nem apaga nem escreve.
    scene.hand_on((60, 140, 100, 170))
    worker.set_hand_presence(True)
    _passes(worker, clock, generation, scene, 4)
    got = _confirmed(worker, generation) > 0
    assert np.count_nonzero(got & want) >= 0.98 * np.count_nonzero(want)

    # Falha do masker: a mao continua sobre o traco, mas a mascara
    # oscila — dois quadros reportam a regiao toda visivel e o warp
    # traz os pixels da mao (cinza-pele) por cima do traco.
    scene.hand_off()
    masked = scene.page.copy()
    masked[60:140, 100:170] = HAND_GRAY
    for _ in range(2):
        worker.submit(WarpedPage(_bgr(masked), generation, scene.corners, None, None))
        clock.advance(_DT_ACTIVE)
        worker.pump()
    got = _confirmed(worker, generation) > 0
    assert np.count_nonzero(got & want) >= 0.95 * np.count_nonzero(want)

    # Mao sai de vez: o traco segue intacto e nada novo apareceu.
    _passes(worker, clock, generation, scene, 3, dt=_DT_IDLE)
    got = _confirmed(worker, generation) > 0
    assert np.count_nonzero(got & want) >= 0.95 * np.count_nonzero(want)
    assert np.count_nonzero(got & ~_dilate(want, 3)) == 0


# ------------------------------------------------------------------
# Contrato 2: conteudo fraco pre-existente (pauta, vinco, lapis,
# impresso cinza) nao vaza para a saida. Marcas fortes pre-existentes
# estao fora do escopo — suprimi-las e o trabalho do #130/PageModel,
# medido separadamente; para o voto elas sao tinta legitima.
# ------------------------------------------------------------------


def test_preexisting_weak_content_does_not_leak(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    for y in range(30, H - 20, 24):  # pauta impressa
        scene.draw((y, y + 2, 10, W - 10), RULING_GRAY)
    scene.draw((0, H, 150, 162), CREASE_GRAY)  # vinco/lombada
    scene.draw((180, 184, 200, 280), PENCIL_GRAY)  # desenho a lapis antigo
    _passes(worker, clock, generation, scene, 8, dt=_DT_IDLE)
    confirmed, provisional, _version = _display(worker, generation)
    assert np.count_nonzero(confirmed) == 0
    assert np.count_nonzero(provisional) == 0


# ------------------------------------------------------------------
# Contrato 3: traco novo apos estabilizacao aparece limpo, inclusive
# cruzando pauta.
# ------------------------------------------------------------------


def test_new_stroke_after_settling_confirms_clean(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    for y in range(30, H - 20, 24):
        scene.draw((y, y + 2, 10, W - 10), RULING_GRAY)
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)

    stroke = (90, 96, 60, 260)  # cruza duas linhas de pauta
    _write_stroke(worker, clock, generation, scene, stroke)
    want = _rect_mask(stroke)
    got = _confirmed(worker, generation) > 0
    assert np.count_nonzero(got & want) >= 0.95 * np.count_nonzero(want)
    extra = got & ~_dilate(want, 2)
    assert np.count_nonzero(extra) <= 0.02 * np.count_nonzero(want)


# ------------------------------------------------------------------
# Contrato 4: jitter de 1-3 px na borda contra fundo escuro nao gera
# tinta nas margens.
# ------------------------------------------------------------------


def test_border_jitter_over_dark_table_makes_no_ink(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    scene.send_coverage = False  # sem mascara seg: so o warp fala
    moves = [(1, 0), (2, 0), (3, 1), (1, 2), (0, 3), (-2, 1), (-3, 0), (-1, -2)]
    for dx, dy in moves:
        img = scene.jitter(dx, dy)
        worker.submit(WarpedPage(_bgr(img), generation, scene.corners, None, None))
        clock.advance(_DT_IDLE)
        worker.pump()
    got = _confirmed(worker, generation) > 0
    margin = np.zeros((H, W), dtype=bool)
    margin[:8, :] = margin[-8:, :] = margin[:, :8] = margin[:, -8:] = True
    assert np.count_nonzero(got & margin) == 0
    assert np.count_nonzero(got) == 0


# ------------------------------------------------------------------
# Contrato 5: cena parada em baixa luz nao gera churn de saida.
# ------------------------------------------------------------------


def test_dim_static_scene_does_not_churn(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene(paper=PAPER_DIM)
    versions = set()
    for i in range(10):
        scene.page[:] = PAPER_DIM + (4 if i % 2 else -4)  # flicker de exposicao
        _drive(worker, clock, scene.page_frame(generation), _DT_IDLE)
        confirmed, provisional, version = _display(worker, generation)
        versions.add(version)
        assert np.count_nonzero(confirmed) == 0
    confirmed, provisional, _ = _display(worker, generation)
    assert np.count_nonzero(confirmed) == 0
    assert np.count_nonzero(provisional) == 0
    assert len(versions) <= 3  # assentada: poucas publicacoes, nenhuma tinta


# ------------------------------------------------------------------
# Contrato 6: borracha remove o traco sem gerar tinta falsa.
# ------------------------------------------------------------------


def test_eraser_removes_stroke_without_false_ink(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)

    # Dois tracos: o desenho inteiro segue em cena depois da borracha,
    # entao a passada tem testemunha de que a folha ainda e a mesma.
    stroke_a = (90, 96, 60, 260)
    stroke_b = (150, 156, 60, 200)
    _write_stroke(worker, clock, generation, scene, stroke_a)
    _write_stroke(worker, clock, generation, scene, stroke_b)
    want_a = _rect_mask(stroke_a)
    want_b = _rect_mask(stroke_b)
    want = want_a | want_b
    assert np.count_nonzero(_confirmed(worker, generation) & want) > 0

    # A borracha apaga um terco do traco A sob a mao: a mao cobre a
    # regiao enquanto os pixels voltam ao nivel do papel; depois sai.
    y0, y1, x0, x1 = stroke_a
    cut = x0 + (x1 - x0) // 3
    scene.hand_on(stroke_a)
    worker.set_hand_presence(True, moved=True)
    _passes(worker, clock, generation, scene, 2)
    scene.page[y0:y1, x0:cut] = PAPER
    scene.hand_on((y0, y1, x0, cut))
    _passes(worker, clock, generation, scene, 2)
    scene.hand_off()
    worker.set_hand_presence(False)
    _passes(worker, clock, generation, scene, 8, dt=_DT_IDLE)

    got = _confirmed(worker, generation) > 0
    erased = want_a.copy()
    erased[:, cut:] = False
    kept = want & ~erased
    assert np.count_nonzero(got & erased) <= 0.05 * np.count_nonzero(erased)
    assert np.count_nonzero(got & ~_dilate(want, 3)) == 0
    assert np.count_nonzero(got & kept) >= 0.90 * np.count_nonzero(kept)


# ------------------------------------------------------------------
# Contrato 7: deslize leve da folha nao duplica tracos.
# ------------------------------------------------------------------


def test_light_sheet_slide_does_not_duplicate_ink(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)

    stroke = (90, 96, 60, 260)
    _write_stroke(worker, clock, generation, scene, stroke)
    before = _confirmed(worker, generation) > 0
    original_px = np.count_nonzero(before)
    assert original_px > 0

    # A folha desliza ~8 px e o quad acompanha com erro de rastreio:
    # o conteudo do warp anda o residuo (-2, -1, 0 px).
    for dx, residual in ((8, -2), (8, -1), (8, 0)):
        scene.corners = Q0 + np.array([dx, 0], dtype=np.float32)
        img = np.full_like(scene.page, TABLE_GRAY)
        if residual < 0:
            img[:, :residual] = scene.page[:, -residual:]
        elif residual > 0:
            img[:, residual:] = scene.page[:, :-residual]
        else:
            img[:] = scene.page
        scene.table = img == TABLE_GRAY
        worker.submit(
            WarpedPage(
                _bgr(img),
                generation,
                scene.corners,
                scene.cam_mask(~scene.hand, fill=True),
                scene.cam_mask(~scene.table, fill=False),
            )
        )
        clock.advance(_DT_ACTIVE)
        worker.pump()
    _passes(worker, clock, generation, scene, 3, dt=_DT_IDLE)

    got = _confirmed(worker, generation) > 0
    total = np.count_nonzero(got)
    assert total >= 0.75 * original_px
    assert total <= 1.4 * original_px
    # Uma copia deslocada apareceria como segundo componente ou como
    # excesso de pixels: o total acima ja barra a duplicacao limpa.


# ------------------------------------------------------------------
# Contrato 8: traco lido uma vez antes da mao sair termina de
# confirmar nas passadas pos-saida, sem a mao voltar.
# ------------------------------------------------------------------


def test_stroke_seen_once_before_hand_exit_finishes_confirming(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)

    # O traco e desenhado um instante antes da mao levantar: so uma
    # leitura com a mao presente; o resto da janela de votos acontece
    # nas passadas que a saida re-arma (review #74/#122).
    stroke = (90, 96, 60, 260)
    scene.draw(stroke)
    worker.set_hand_presence(True, moved=True)
    _passes(worker, clock, generation, scene, 1)
    worker.set_hand_presence(False)
    _passes(worker, clock, generation, scene, 8, dt=_DT_IDLE)

    want = _rect_mask(stroke)
    got = _confirmed(worker, generation) > 0
    assert np.count_nonzero(got & want) >= 0.95 * np.count_nonzero(want)
    extra = got & ~_dilate(want, 2)
    assert np.count_nonzero(extra) <= 0.02 * np.count_nonzero(want)
