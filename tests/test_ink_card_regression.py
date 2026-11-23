"""Rede de regressão: um cartao claro sobre tinta confirmada nao a apaga.

Testes de WORKER (fluxo completo), nao de componente: afirmam so o que o
produto mostra — a tinta confirmada — e nao qual mecanismo interno a
protege. Nada aqui espera que `PageModel.exposed()` rejeite o cartao: nessa
camada a mascara pode abrir sobre ele, e a tinta ainda assim e preservada
porque o worker combina a exposicao com outras evidencias de papel.

Cada cenario e uma sequencia propria, sem estado compartilhado:

- A: o cartao ja esta sobre a tinta no instante em que a luz fica quente
  (as referencias locais ainda sao as da luz antiga);
- B: o cartao chega DEPOIS de 40 passadas quentes. Medido: as referencias do
  PAPEL ja convergiram (razao B/R 1,00 -> 0,50); as de BAIXO dos tracos nao
  adaptam nunca (1,00 mesmo apos 160 passadas), porque pixel de tinta nao
  atualiza a referencia — e sao justamente as celulas que o cartao cobre;
- C: o cartao cobre os 4 tracos POR INTEIRO, com um grupo de tracos
  testemunha FORA dele. E uma CARACTERIZACAO: sem testemunha a remocao nem
  tem evidencia (o resultado nao provaria nada); com ela, um traco
  totalmente substituido por area com cor de papel e indistinguivel de uma
  borracha NESTE CENARIO SINTETICO e para as evidencias que o worker
  observa; um cartao real pode trazer bordas, textura, reflexo ou movimento
  que este ensaio nao modela. Mesma semantica do contrato da borracha;
- D: controle, cartao neutro (razao B/R diferente do papel quente);

Registro estavel (quad constante, sem sinais de mao) e passadas suficientes
para o protocolo de remocao concluir (2x `_REMOVAL_PASSES`).

Escopo da evidencia: cartoes ESTATICOS e sinteticos, no harness de relogio
injetado (cada quadro vira uma passada a 1,1 s, sem sinais de mao). Nao
cobre lampada real, balanco de branco automatico da camera, cadencia real
do worker nem outros oclusores.

Discriminacao medida (nao fixada nos testes): A falha no head 7a6a9f1 do
#175; A, B e D perdem 100 px com `paper_guard=False` (a protecao observada
depende da guarda de papel, hipotese a confirmar isolando as mascaras);
`occlusion_hold_passes=0` nao muda nenhum cenario. Um cenario sem mudanca de
cor foi descartado: passava na main antiga e com a guarda desligada, ou seja,
nao discriminava neste harness.
"""

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import registration as registration_mod
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig

H, W = 96, 128
Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)

PAPER = 235.0
# Quatro tracos de 100 px (5x20): a tinta que o cartao cobre.
COVERED = [(15, 20, 10, 30), (35, 40, 10, 30), (55, 60, 10, 30), (75, 80, 10, 30)]
# Tres tracos de 400 px (5x80) fora do cartao: testemunha da remocao
# (1200 px de 1600 retidos = 75% >= _REMOVAL_RETAINED_FRACTION de 0,65).
WITNESS = [(15, 20, 40, 120), (35, 40, 40, 120), (55, 60, 40, 120)]

NEUTRAL_INK = (40.0, 40.0, 40.0)
WARM_INK = (20.0, 40.0, 40.0)  # a lampada quente tambem tinge a tinta
WARM_CARD = (122.0, 245.0, 245.0)  # claro, com a cor do papel quente
NEUTRAL_CARD = (245.0, 245.0, 245.0)  # claro, razao B/R diferente da quente

HOLD_PASSES = 2 * registration_mod._REMOVAL_PASSES
ADAPT_PASSES = 40


class _ScriptInk:
    """Tinta falsa guiada pelo conteudo do proprio warp (escuro = tinta)."""

    threshold = 0.7

    def ink_prob(self, warped):
        gray = warped if warped.ndim == 2 else warped.mean(axis=2)
        return (gray < 100).astype(np.float32) * 0.95


def _paper(warm):
    page = np.full((H, W, 3), PAPER, dtype=np.float32)
    if warm:
        page[..., 0] *= 0.5  # a lampada quente derruba o azul
    return page


def _with(page, strokes, color, cols=None):
    img = page.copy()
    for y0, y1, x0, x1 in strokes:
        end = x1 if cols is None else x0 + cols
        img[y0:y1, x0:end] = color
    return img


class _Clock:
    """Relogio deterministico injetado na trabalhadora (modo pump)."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


# Passo do relogio entre passadas: sem sinal de mao, a sonda ociosa exige
# ~1,0 s para a passada rodar.
_DT_IDLE = 1.1


class _Run:
    def __init__(self, config):
        self.clock = _Clock()
        self.worker = worker_mod.PageCleanWorker(
            _ScriptInk(), clock=self.clock, config=config
        )
        self.generation = self.worker.new_generation()

    def drive(self, img, n=1):
        for _ in range(n):
            frame = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
            self.worker.submit(WarpedPage(frame, self.generation, Q0))
            self.clock.advance(_DT_IDLE)
            assert self.worker.pump(), "quadro submetido nao virou passada"

    def confirmed(self):
        snap = self.worker.poll(self.generation, 0)
        return np.zeros((H, W), dtype=bool) if snap is None else snap.confirmed > 0

    def stop(self):
        try:
            self.worker.stop()
        except RuntimeError:
            # Modo pump(): a thread nunca foi iniciada.
            pass


def _stroke_present(confirmed, strokes, what):
    """Cada traco alvo tem tinta confirmada (>= 90% da area), logo ANTES do
    cartao. Sem isso 'a tinta sobreviveu' seria vacuo: nao havia tinta."""
    for y0, y1, x0, x1 in strokes:
        area = (y1 - y0) * (x1 - x0)
        got = np.count_nonzero(confirmed[y0:y1, x0:x1])
        assert got >= 0.9 * area, f"{what} ({y0},{x0}): {got}/{area} px antes do cartao"


def _paper_refs_converged(run):
    """Precondicao do cenario: depois da adaptacao, a razao B/R que o
    PageModel guarda para o PAPEL (longe dos tracos) esta na luz quente.
    Le estado interno de proposito — e uma garantia sobre a montagem do
    cenario, nao sobre o comportamento afirmado."""
    model = run.worker._page_model
    assert model is not None, "sem PageModel: o cenario de adaptacao nao se aplica"
    far = (slice(85, H), slice(0, W))  # linhas sem nenhum traco
    ratio = float(model._blue_ref[far].mean() / model._red_ref[far].mean())
    assert ratio == pytest.approx(0.5, abs=0.05), (
        f"referencias nao adaptaram: {ratio:.3f}"
    )


def scenario(name, config=None):
    """Roda um cenario e devolve (confirmado antes do cartao, depois)."""
    config = config or InkConfig(page_model=True)
    run = _Run(config)
    try:
        normal, warm = _paper(False), _paper(True)
        covered_ink = COVERED
        run.drive(normal, 7)  # a referencia do PageModel semeia
        if name == "C":
            ink = _with(_with(normal, COVERED, NEUTRAL_INK), WITNESS, NEUTRAL_INK)
        else:
            ink = _with(normal, COVERED, NEUTRAL_INK)
        run.drive(ink, 10)
        before = run.confirmed()
        _stroke_present(before, COVERED, "traco coberto")
        if name == "C":
            _stroke_present(before, WITNESS, "testemunha")

        if name == "A":  # luz quente + cartao ao mesmo tempo
            card = _with(_with(warm, covered_ink, WARM_INK), covered_ink, WARM_CARD, 5)
            run.drive(card, HOLD_PASSES)
        elif name in ("B", "D", "C"):  # adapta primeiro, cartao depois
            adapted = _with(warm, covered_ink, WARM_INK)
            if name == "C":
                adapted = _with(adapted, WITNESS, WARM_INK)
            run.drive(adapted, ADAPT_PASSES)
            _paper_refs_converged(run)
            before = run.confirmed()
            _stroke_present(before, covered_ink, "traco coberto apos a adaptacao")
            if name == "C":
                _stroke_present(before, WITNESS, "testemunha apos a adaptacao")
            color = NEUTRAL_CARD if name == "D" else WARM_CARD
            cols = 20 if name == "C" else 5
            run.drive(_with(adapted, covered_ink, color, cols), HOLD_PASSES)
        else:
            raise ValueError(name)
        return before, run.confirmed()
    finally:
        run.stop()


def test_card_present_at_the_color_change_keeps_the_ink():
    before, after = scenario("A")
    assert np.array_equal(after, before)


def test_card_after_40_warm_passes_keeps_the_ink():
    before, after = scenario("B")
    assert np.array_equal(after, before)


def test_card_fully_over_strokes_with_a_witness_reads_as_erasure():
    """Caracterizacao, nao promessa de protecao: ver a docstring do modulo."""
    before, after = scenario("C")
    witness = np.zeros_like(before)
    covered = np.zeros_like(before)
    for y0, y1, x0, x1 in WITNESS:
        witness[y0:y1, x0:x1] = True
    for y0, y1, x0, x1 in COVERED:
        covered[y0:y1, x0:x1] = True
    assert np.array_equal(after & witness, before & witness)  # testemunha intacta
    lost = np.count_nonzero(before & covered & ~after)
    assert lost >= 0.5 * np.count_nonzero(before & covered)  # lida como apagamento


def test_neutral_card_control_keeps_the_ink():
    before, after = scenario("D")
    assert np.array_equal(after, before)
