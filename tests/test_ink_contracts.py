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

O driver embute a garantia que os contratos precisam: um quadro que
o teste narra como evento (deslize, mascara falha, jitter) tem que
virar passada (`must=True`), e a leitura exige canvas publicado —
sem isso a trabalhadora poderia dormir sobre o evento e a checagem
lira o canvas de antes dele.
"""

import dataclasses

import cv2
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
OCCLUDER_GRAY = 110.0

# Linhas de pauta impressa na cena (fonte unica: os testes que as
# desenham e o traco que as cruza leem daqui).
RULE_YS = range(30, H - 20, 24)

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
    modelo real sobre conteudo impresso; claro vira 0. A faixa
    105..119 e o objeto escuro que nao e tinta (manga, sombra densa):
    escuro demais para a fotometria chamar de papel, escuro demais
    para o modelo chamar de tinta — prob 0,05.
    """

    def ink_prob(self, warped):
        gray = warped if warped.ndim == 2 else warped.mean(axis=2)
        prob = np.zeros(gray.shape, dtype=np.float32)
        prob[gray < 100] = 0.95
        prob[(gray >= 105) & (gray < 120)] = 0.05
        prob[(gray >= 120) & (gray < 205)] = 0.45
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

    def shifted(self, quad_delta, content_shift):
        """Quad anda `quad_delta` na camera; o conteudo do warp anda
        `content_shift` na grade da pagina (o erro de rastreio) e a
        banda descoberta vira mesa escura."""
        self.corners = Q0 + np.asarray(quad_delta, dtype=np.float32)
        dx, dy = int(content_shift[0]), int(content_shift[1])
        img = np.full_like(self.page, TABLE_GRAY)
        y0, y1 = max(0, dy), min(H, H + dy)
        x0, x1 = max(0, dx), min(W, W + dx)
        if y1 > y0 and x1 > x0:
            img[y0:y1, x0:x1] = self.page[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        self.table = img == TABLE_GRAY
        return img

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

    def frame(self, generation, img=None, *, bare=False):
        """WarpedPage da cena (ou de `img`, se dada) nas mascaras
        padrao: `visible` e `coverage` colados no quad da camera.
        bare=True omite as duas — a passada so tem o warp para ler."""
        if img is None:
            img = self.warped()
        visible = coverage = None
        if not bare:
            visible = self.cam_mask(~self.hand, fill=True)
            coverage = (
                self.cam_mask(~self.table, fill=False) if self.send_coverage else None
            )
        return WarpedPage(_bgr(img), generation, self.corners, visible, coverage)


# ------------------------------------------------------------------
# Dirigir a trabalhadora: so API publica.
# ------------------------------------------------------------------


def _worker(config):
    clock = _Clock()
    return clock, PageCleanWorker(_ScriptInk(), clock=clock, config=config)


def _feed(worker, clock, page, dt=_DT_ACTIVE, *, must=True):
    """Submete `page`, avanca o relogio e roda uma passada se a
    cadencia deixar. must=True exige que o quadro vire passada —
    um evento narrado engolido pelo descanso deixaria o teste ler
    o canvas de antes dele."""
    worker.submit(page)
    clock.advance(dt)
    ran = worker.pump()
    if must:
        assert ran, "quadro submetido nao virou passada"
    return ran


def _passes(worker, clock, generation, scene, n, dt=_DT_ACTIVE, *, must=False):
    """n ciclos submit/pump sobre a cena corrente; devolve quais
    quadros viraram passada (o descanso pode pular os ociosos)."""
    return [
        _feed(worker, clock, scene.frame(generation), dt, must=must) for _ in range(n)
    ]


def _confirmed(worker, generation):
    snap = worker.poll(generation, 0)
    assert snap is not None, "trabalhadora nunca publicou canvas"
    return np.asarray(snap.confirmed)


def _display(worker, generation):
    snap = worker.poll_display(generation, 0)
    assert snap is not None, "trabalhadora nunca publicou canvas"
    return np.asarray(snap.confirmed), np.asarray(snap.provisional), snap.version


def _stroke_mask(scene):
    """Mascara do que foi desenhado: scene.page so carrega conteudo
    desenhado — mesa e mao vivem no warp, e os cinzas fracos
    (pauta, vinco, lapis) ficam todos >= 100."""
    return scene.page < 100


def _write_stroke(worker, clock, generation, scene, stroke):
    """Traco escrito com a mao presente e revelado na saida dela."""
    scene.draw(stroke)
    worker.set_hand_presence(True, moved=True)
    _passes(worker, clock, generation, scene, 4)
    worker.set_hand_presence(False)
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)


def _dilate(mask, px):
    return cv2.dilate(
        mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    ).astype(bool)


def _frac(got, want):
    """Fracao de `want` presente em `got` (cobertura do esperado)."""
    return np.count_nonzero(got & want) / np.count_nonzero(want)


def _beyond(got, want, px):
    """Pixels de `got` fora da vizinhanca de `want` (vazamento)."""
    return np.count_nonzero(got & ~_dilate(want, px))


# Regimes sob medicao. "production" e o preset do PipelineConfig.
# "page_model_guarded" e o candidato PageModel que os contratos 9
# (oclusor sem mascara) e 4 (faixa de mesa sustentada) mediram como
# necessario: o modelo subsumiu o removedor de pauta, mas NAO a
# guarda fotometrica (pm.exposed devolve tudo-True antes da
# sementeira; sem paper_ok um oclusor escuro de prob baixa apaga
# traco confirmado por voto de remocao) nem o recuo de borda
# (sem mascara de cobertura, uma faixa de mesa escura dentro do
# quad confirma como linha de margem). A variante sem guardas
# (paper_guard=False, border_guard_px=0) falha nos dois contratos.
PRODUCTION = InkConfig.production()
PAGE_MODEL_GUARDED = dataclasses.replace(
    PRODUCTION,
    page_model=True,
    ruling_removal=False,
)

REGIMES = {
    "production": PRODUCTION,
    "page_model_guarded": PAGE_MODEL_GUARDED,
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

    _write_stroke(worker, clock, generation, scene, (90, 96, 60, 260))
    want = _stroke_mask(scene)
    got = _confirmed(worker, generation) > 0
    assert _frac(got, want) >= 0.95

    # A mao estaciona sobre o traco (mascara correta): sem evidencia,
    # a regiao congela — nem apaga nem escreve.
    scene.hand_on((60, 140, 100, 170))
    worker.set_hand_presence(True)
    _passes(worker, clock, generation, scene, 4)
    got = _confirmed(worker, generation) > 0
    assert _frac(got, want) >= 0.98

    # Falha do masker: a mao segue trabalhando sobre o traco, mas a
    # mascara oscila — dois quadros reportam a regiao toda visivel e
    # o warp traz os pixels da mao (cinza-pele) por cima do traco.
    scene.hand_off()
    worker.set_hand_presence(True, moved=True)
    masked = scene.page.copy()
    masked[60:140, 100:170] = HAND_GRAY
    for _ in range(2):
        _feed(worker, clock, scene.frame(generation, img=masked, bare=True))
    got = _confirmed(worker, generation) > 0
    assert _frac(got, want) >= 0.95

    # Mao sai de vez: sem o sinal a porta de ociosidade le a mao
    # ainda presente e parada, e os quadros limpos nunca rodam.
    worker.set_hand_presence(False)
    ran = _passes(worker, clock, generation, scene, 3, dt=_DT_IDLE)
    assert any(ran), "nenhuma passada rodou depois da saida da mao"
    got = _confirmed(worker, generation) > 0
    assert _frac(got, want) >= 0.95
    assert _beyond(got, want, 3) == 0


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
    # A pauta entra FORTE (prob 0.95 como no v2 sobre impresso escuro):
    # acima da barra do voto, ela so nao vaza se o mecanismo proprio de
    # cada regime agir — remocao de pauta em production, referencia do
    # PageModel em page_model_solo. Os cinzas fracos (vinco, lapis)
    # ficam abaixo do piso comum do voto e medem o resultado do
    # produto, nao qual guarda agiu.
    for y in RULE_YS:
        scene.draw((y, y + 2, 10, W - 10), INK)
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
    for y in RULE_YS:
        scene.draw((y, y + 2, 10, W - 10), RULING_GRAY)
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)

    # Traco vertical cruzando as pautas 78-79 e 102-103 (RULE_YS).
    _write_stroke(worker, clock, generation, scene, (76, 106, 157, 163))
    want = _stroke_mask(scene)
    got = _confirmed(worker, generation) > 0
    assert _frac(got, want) >= 0.95
    assert _beyond(got, want, 2) <= 0.02 * np.count_nonzero(want)


# ------------------------------------------------------------------
# Contrato 4: jitter de 1-3 px na borda contra fundo escuro nao gera
# tinta nas margens.
# ------------------------------------------------------------------


def test_border_jitter_over_dark_table_makes_no_ink(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    scene.send_coverage = False  # sem mascara seg: so o warp fala
    # Papel limpo assenta primeiro: a faixa de mesa so e admitida
    # como evidencia se a referencia (e o dominio do voto) ja
    # cobrirem a borda — e o cenario do review do #162.
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE, must=True)
    moves = [(1, 0), (2, 0), (3, 1), (1, 2), (0, 3), (-2, 1), (-3, 0), (-1, -2)]
    for dx, dy in moves:
        # O quad desliza (dx, dy); o conteudo escorrega o oposto e a
        # borda recolhida admite mesa escura.
        img = scene.shifted((dx, dy), (-dx, -dy))
        _feed(worker, clock, scene.frame(generation, img=img, bare=True), _DT_IDLE)
    # O jitter alterna a faixa e ela nunca junta votos. Um deslize
    # sustentado na mesma direcao com o quad parado (rastreio nao
    # acompanha) deixa a MESMA faixa de mesa exposta dentro da folha
    # por toda a janela de votos — sem o recuo projetado automatico
    # ela confirmaria como linha de margem.
    for _ in range(5):
        img = scene.shifted((0, 0), (5, 0))
        _feed(worker, clock, scene.frame(generation, img=img), _DT_IDLE)
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
        _feed(worker, clock, scene.frame(generation), _DT_IDLE)
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
    want_a = _stroke_mask(scene)
    _write_stroke(worker, clock, generation, scene, stroke_b)
    want_b = _stroke_mask(scene) & ~want_a
    want = want_a | want_b
    # Cada traco precisa estar confirmado por si: se o alvo da
    # borracha nunca tivesse confirmado, o fim do teste passaria
    # sem remover tinta alguma.
    got0 = _confirmed(worker, generation) > 0
    assert _frac(got0, want_a) >= 0.9
    assert _frac(got0, want_b) >= 0.9

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
    assert _beyond(got, want, 3) == 0
    assert _frac(got, kept) >= 0.90


# ------------------------------------------------------------------
# Contrato 7: deslize leve da folha nao duplica tracos.
# ------------------------------------------------------------------


def test_light_sheet_slide_does_not_duplicate_ink(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)

    _write_stroke(worker, clock, generation, scene, (90, 96, 60, 260))
    before = _confirmed(worker, generation) > 0
    original_px = np.count_nonzero(before)
    assert original_px > 0

    # A folha desliza ~8 px e o quad acompanha com erro de rastreio:
    # o conteudo do warp anda o residuo (-2, -1, 0 px). A mao que
    # arrasta a folha mantem a trabalhadora acordada — 8 px fica
    # abaixo do gatilho geometrico de despertar e, em descanso, todo
    # quadro submetido e ignorado. Uma copia confirmada no meio do
    # deslize poderia sumir ate o assentamento: a cada quadro o
    # canvas nao pode ter tinta alem do envelope do residuo.
    worker.set_hand_presence(True, moved=True)
    for residual in (-2, -1, 0):
        img = scene.shifted((8, 0), (residual, 0))
        _feed(worker, clock, scene.frame(generation, img=img))
        conf, _prov, _v = _display(worker, generation)
        assert _beyond(conf > 0, before, 4) == 0, (
            f"tinta transitoria no quadro de residuo {residual}"
        )
    worker.set_hand_presence(False)
    _passes(worker, clock, generation, scene, 3, dt=_DT_IDLE)

    got = _confirmed(worker, generation) > 0
    total = np.count_nonzero(got)
    assert total >= 0.75 * original_px
    assert total <= 1.15 * original_px
    # Eco mais fino que o residuo merge no proprio traco (o total
    # acima limita); fora da vizinhanca nao pode haver copia.
    assert _beyond(got, before, 2) == 0


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
    scene.draw((90, 96, 60, 260))
    worker.set_hand_presence(True, moved=True)
    _passes(worker, clock, generation, scene, 1, must=True)
    want = _stroke_mask(scene)
    # Uma leitura so nao confirma a janela de votos: se o traco ja
    # estivesse confirmado, a fase pos-saida seria vacua.
    got = _confirmed(worker, generation) > 0
    assert _frac(got, want) < 0.5, "traco confirmou antes da saida da mao"
    worker.set_hand_presence(False)
    ran = _passes(worker, clock, generation, scene, 8, dt=_DT_IDLE)
    assert any(ran), "nenhuma passada rodou depois da saida da mao"

    got = _confirmed(worker, generation) > 0
    assert _frac(got, want) >= 0.95
    assert _beyond(got, want, 2) <= 0.02 * np.count_nonzero(want)


# Contrato 9: objeto escuro sem mascara sobre traco confirmado nao o
# apaga — a guarda fotometrica congela a regiao para o voto de
# remocao, com ou sem deteccao de mao. (O contrato 8, traco visto
# uma vez antes da mao sair, chega por outro PR.)
# ------------------------------------------------------------------


def test_unmasked_dark_occluder_does_not_erase_confirmed_ink(ink):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE, must=True)

    # Tres tracos: com um deles coberto, o restante continua visivel
    # e satisfaz a testemunha de remocao (a folha ainda e a mesma) —
    # e o cenario exato em que o voto de remocao agiria.
    stroke_a = (90, 96, 60, 260)
    _write_stroke(worker, clock, generation, scene, stroke_a)
    want_a = _stroke_mask(scene)
    _write_stroke(worker, clock, generation, scene, (150, 156, 60, 200))
    _write_stroke(worker, clock, generation, scene, (200, 206, 60, 260))
    want = _stroke_mask(scene)
    got0 = _confirmed(worker, generation) > 0
    assert _frac(got0, want) >= 0.9
    cover = np.zeros((H, W), dtype=bool)
    cover[70:110, 40:180] = True
    covered = want_a & cover
    # O trecho que vai ser coberto parte confirmado inteiro — se a
    # leitura falhar em mais de uns pixels, o bug nao e do oclusor.
    assert _frac(got0, covered) >= 0.97

    # Uma manga/objeto escuro pousa sobre parte do traco A e a
    # mascara de mao o erra por completo: o warp traz o objeto
    # (cinza ~110, prob de tinta baixa) e a mascara reporta tudo
    # visivel. Sem a guarda fotometrica o escuro conta como papel
    # exposto e o voto de remocao apaga o traco fisico em seis
    # passadas — sem borracha alguma na cena.
    clock.advance(2.1)  # elegibilidade de remocao apos a confirmacao
    occluded = scene.page.copy()
    occluded[cover] = OCCLUDER_GRAY
    for _ in range(8):
        # A mao que segura o objeto continua trabalhando — parada
        # ela poria a trabalhadora em descanso e os quadros nao
        # rodariam, deixando o teste ler o canvas anterior.
        worker.set_hand_presence(True, moved=True)
        _feed(worker, clock, scene.frame(generation, img=occluded))

    got = _confirmed(worker, generation) > 0
    assert _frac(got, covered) >= 0.97, "traco sob objeto escuro sumiu"
    assert _frac(got, want) >= 0.9
    assert _beyond(got, want, 3) == 0


# ------------------------------------------------------------------
# Contrato 10: objeto escuro na abertura nao semeia como referencia
# nem gera tinta — pm.exposed devolve tudo-True antes da primeira
# sementeira, entao so paper_ok impede que o bloco conte observacao.
# ------------------------------------------------------------------


def test_startup_occluder_never_seeds_nor_inks(ink, regime):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    cover = np.zeros((H, W), dtype=bool)
    cover[60:120, 60:200] = True
    occluded = scene.page.copy()
    occluded[cover] = OCCLUDER_GRAY
    # O objeto esta la desde a abertura e a mascara de mao o erra.
    # Se a guarda estiver desligada, os blocos sobre o oclusor
    # semeiam com o tom dele (110) — a referencia nasce envenenada
    # e um traco de ~90 cairia dentro da razao para sempre.
    for _ in range(6):
        worker.set_hand_presence(True, moved=True)
        _feed(worker, clock, scene.frame(generation, img=occluded))
    worker.set_hand_presence(False)

    pm = getattr(worker, "_page_model", None)
    if REGIMES[regime].page_model:
        # Com o PageModel ligado o modelo EXISTE: se sumir, a asserção de
        # semeadura nao pode sumir junto em silencio.
        assert pm is not None, f"{regime}: sem PageModel apos a abertura"
    if pm is not None:
        assert np.count_nonzero(pm.seeded & cover) == 0, (
            "oclusor sem mascara virou referencia de pagina"
        )
    assert np.count_nonzero(_confirmed(worker, generation)) == 0

    # Revelado, o papel semeia com o tom real e um traco de 90 —
    # mais escuro que o papel, mais claro que o oclusor — confirma
    # normalmente (a banda que a referencia envenenada suprimiria).
    _passes(worker, clock, generation, scene, 6, dt=_DT_IDLE)
    scene.page[75:81, 80:150] = 90.0
    worker.set_hand_presence(True, moved=True)
    _passes(worker, clock, generation, scene, 4)
    worker.set_hand_presence(False)
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)
    want = _stroke_mask(scene)
    got = _confirmed(worker, generation) > 0
    assert _frac(got, want) >= 0.9


# Contratos 8-9 (F-11): a mesma folha devolvida depois de uma queda do
# tracker. O erro de cantos entre geracoes chega ao motor como residuo
# de registro: o conteudo da pagina retificada volta deslocado na
# grade. O produto deve (a) reconhecer a mesma folha, (b) devolver o
# desenho confirmado e (c) nao engrossar nem duplicar os tracos.
#
# Nenhum atributo privado: new_generation(preserve_identity=True),
# submit/pump e poll.
# ------------------------------------------------------------------

_F11_STROKES = ((50, 56, 40, 280), (90, 96, 40, 280), (130, 136, 40, 280))


def _f11_draw_session(worker, clock, residual):
    """Escreve tres tracos, derruba, devolve a folha deslocada de
    `residual` px. Devolve (gen2, mascara esperada, confirmado antes)."""
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)
    for stroke in _F11_STROKES:
        _write_stroke(worker, clock, generation, scene, stroke)
    before = _confirmed(worker, generation) > 0

    returned = _Scene()
    for y0, y1, x0, x1 in _F11_STROKES:
        returned.draw((y0 + residual, y1 + residual, x0, x1))
    gen2 = worker.new_generation(preserve_identity=True)
    _passes(worker, clock, gen2, returned, 6, dt=_DT_IDLE)
    return gen2, before


@pytest.mark.parametrize("residual", [0, 2, 6])
def test_sheet_returned_after_drop_keeps_the_drawing(ink, residual):
    worker, clock = ink
    gen2, before = _f11_draw_session(worker, clock, residual)
    assert np.count_nonzero(before) > 0
    got = _confirmed(worker, gen2) > 0
    # (b) o desenho confirmado volta
    assert np.count_nonzero(got & before) >= 0.95 * np.count_nonzero(before)
    # (c) sem engrossar nem duplicar: nada alem de 1 px do que ja existia
    assert np.count_nonzero(got & ~_dilate(before, 1)) == 0


# ------------------------------------------------------------------
# Contrato 9 (F-11): a folha devolvida com erro de rastreio E cobertura
# parcial. O conteudo do warp anda `residual` px, a faixa que o quad nao
# cobre vira mesa escura (o modelo a le como tinta se a cobertura nao
# acompanhar a evidencia) e parte da folha segue descoberta. O desenho
# confirmado — inclusive o traco escondido na regiao descoberta e o
# encostado nela — volta intacto, sem fantasma e sem tinta de mesa.
# ------------------------------------------------------------------

_F11_EDGE_STROKES = (
    (30, 36, 40, 280),
    (90, 96, 40, 280),
    (188, 194, 40, 280),  # encostado na regiao descoberta
    (210, 216, 40, 280),  # escondido nela
)
_F11_UNCOVERED_FROM = 200  # linha da pagina onde a cobertura acaba


@pytest.mark.parametrize("residual", [2, 6])
def test_sheet_returned_with_tracking_error_and_partial_coverage(ink, residual):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)
    for stroke in _F11_EDGE_STROKES:
        _write_stroke(worker, clock, generation, scene, stroke)
    before = _confirmed(worker, generation) > 0
    assert np.count_nonzero(before) > 0

    returned = _Scene()
    for stroke in _F11_EDGE_STROKES:
        returned.draw(stroke)
    img = returned.shifted((0, 0), (0, -residual))
    # Faixa descoberta (oclusao/sombra): mesa escura e sem cobertura.
    uncovered = np.zeros(img.shape, dtype=bool)
    uncovered[_F11_UNCOVERED_FROM - residual :, :] = True
    img[uncovered] = TABLE_GRAY
    returned.table = returned.table | uncovered
    gen2 = worker.new_generation(preserve_identity=True)
    for _ in range(6):
        _feed(worker, clock, returned.frame(gen2, img), dt=_DT_IDLE, must=False)

    got = _confirmed(worker, gen2) > 0
    assert np.count_nonzero(got & before) >= 0.95 * np.count_nonzero(before)
    assert np.count_nonzero(got & ~_dilate(before, 1)) == 0


def test_slide_after_a_return_with_residual_keeps_the_drawing(ink):
    """O residuo corrigido persiste na geracao: um deslize posterior (o
    tracker acompanha, o conteudo do warp nao anda) segue limpo, como no
    caso sem residuo. Cobre a arbitragem/follow de ancora ligadas."""
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)
    for stroke in _F11_STROKES:
        _write_stroke(worker, clock, generation, scene, stroke)
    before = _confirmed(worker, generation) > 0

    returned = _Scene()
    for y0, y1, x0, x1 in _F11_STROKES:
        returned.draw((y0 + 6, y1 + 6, x0, x1))
    gen2 = worker.new_generation(preserve_identity=True)
    _passes(worker, clock, gen2, returned, 6, dt=_DT_IDLE)
    returned.corners = Q0 + np.float32([5, 3])
    for _ in range(10):
        _feed(worker, clock, returned.frame(gen2), dt=_DT_IDLE, must=False)

    got = _confirmed(worker, gen2) > 0
    assert np.count_nonzero(got & before) >= 0.95 * np.count_nonzero(before)
    assert np.count_nonzero(got & ~_dilate(before, 1)) == 0


@pytest.mark.parametrize("residual", [2, 6])
def test_new_stroke_after_a_return_with_residual_confirms_in_place(ink, residual):
    """Depois da devolucao com erro de rastreio, tinta NOVA precisa
    confirmar no lugar certo da grade do canvas. A evidencia nova passa
    pela cobertura, pelo cinza do PageModel e pela pegada; este contrato e
    sensivel ao CONJUNTO desses mapas na grade corrigida (um mutante que
    desalinha os tres de uma vez o reprova), sem isolar qual deles. O
    desenho antigo, restaurado, continua intacto e esconde a falha — so um
    traco novo a expoe."""
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)
    for stroke in _F11_STROKES:
        _write_stroke(worker, clock, generation, scene, stroke)
    old = _stroke_mask(scene)

    returned = _Scene()
    for y0, y1, x0, x1 in _F11_STROKES:
        returned.draw((y0 + residual, y1 + residual, x0, x1))
    gen2 = worker.new_generation(preserve_identity=True)
    _passes(worker, clock, gen2, returned, 6, dt=_DT_IDLE)

    new = (170, 176, 40, 280)  # lugar vazio da pagina, na grade do canvas
    truth = _Scene()
    truth.draw(new)
    want = _stroke_mask(truth)
    # A folha anda `residual` px na grade do warp: o traco novo tambem.
    _write_stroke(
        worker, clock, gen2, returned, (170 + residual, 176 + residual, 40, 280)
    )
    got = _confirmed(worker, gen2) > 0
    assert _frac(got, want) >= 0.9
    assert _beyond(got, want | old, 1) == 0


@pytest.mark.parametrize(
    "table_shift",
    [None, 9, 11],  # sem mesa (controle) e com banda escura fora da folha
)
def test_dark_uncovered_band_does_not_steer_the_residual(ink, table_shift):
    worker, clock = ink
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)
    for stroke in _F11_STROKES:
        _write_stroke(worker, clock, generation, scene, stroke)
    drawn = _stroke_mask(scene)
    returned = _Scene()
    for y0, y1, x0, x1 in _F11_STROKES:
        returned.draw((y0 + 6, y1 + 6, x0, x1))
        if table_shift:  # banda escura fora da folha, sem cobertura
            returned.table[y0 + table_shift : y1 + table_shift + 6, x0:x1] = True
    gen2 = worker.new_generation(preserve_identity=True)
    _passes(worker, clock, gen2, returned, 8, dt=_DT_IDLE)

    got = _confirmed(worker, gen2) > 0
    assert _frac(got, drawn) >= 0.95
    assert _beyond(got, drawn, 1) == 0


def _return_with_residual(worker, clock, residual, strokes=_F11_STROKES):
    generation = worker.new_generation()
    scene = _Scene()
    _passes(worker, clock, generation, scene, 4, dt=_DT_IDLE)
    for stroke in strokes:
        _write_stroke(worker, clock, generation, scene, stroke)
    drawn = _stroke_mask(scene)
    returned = _Scene()
    for y0, y1, x0, x1 in strokes:
        returned.draw((y0 + residual, y1 + residual, x0, x1))
    gen2 = worker.new_generation(preserve_identity=True)
    _passes(worker, clock, gen2, returned, 6, dt=_DT_IDLE)
    return gen2, drawn, returned


@pytest.mark.parametrize(
    "residual",
    [6, 9, 12],  # 6 funcionava; 9 e 12 passam do limite de 8 px do portao
)
def test_eraser_removes_a_stroke_after_a_return_with_residual(ink, residual):
    worker, clock = ink
    stroke_a, stroke_b = (90, 96, 60, 260), (150, 156, 60, 200)
    gen2, drawn, returned = _return_with_residual(
        worker, clock, residual, (stroke_a, stroke_b)
    )
    want_a = np.zeros_like(drawn)
    want_a[stroke_a[0] : stroke_a[1], stroke_a[2] : stroke_a[3]] = True
    y0, y1, x0, x1 = stroke_a
    cut = x0 + (x1 - x0) // 3
    ya, yb = y0 + residual, y1 + residual
    returned.hand_on((ya, yb, x0, x1))
    worker.set_hand_presence(True, moved=True)
    _passes(worker, clock, gen2, returned, 2)
    returned.page[ya:yb, x0:cut] = PAPER
    returned.hand_on((ya, yb, x0, cut))
    _passes(worker, clock, gen2, returned, 2)
    returned.hand_off()
    worker.set_hand_presence(False)
    _passes(worker, clock, gen2, returned, 8, dt=_DT_IDLE)

    got = _confirmed(worker, gen2) > 0
    erased = want_a.copy()
    erased[:, cut:] = False
    assert np.count_nonzero(got & erased) <= 0.05 * np.count_nonzero(erased)
    assert _frac(got, drawn & ~erased) >= 0.90  # a testemunha segue confirmada
