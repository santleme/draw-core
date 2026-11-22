"""PageModel: a referencia persistente da pagina limpa.

O mecanismo classico de captura por camera fixa (classificar cada
pixel contra uma imagem de fundo mantida no tempo) respondendo as
cinco falhas da fila de robustez de 2026-11-21:

- regiao nunca vista limpa fica pendente e nao entrega evidencia
  (a extensao do "quem nao ve, se abstem" para a vida da geracao);
- o que estava na pagina quando o pixel foi visto pela primeira
  vez semeia a referencia — conteudo antigo, pauta e vinco nunca
  viram tinta nova (a semantica do baseline do #130);
- a referencia rastreia devagar onde ha papel — deriva de
  iluminacao (#133) afunda nela, nao no voto;
- traco nunca afunda na referencia — a mascara `paper` o exclui,
  e so o que ESCURECEU contra a referencia e conteudo novo, entao
  borracha nunca le como tinta.
"""

import threading
import time

import numpy as np
import pytest

from draw.contracts import WarpedPage
from draw.ink import identity as identity_mod
from draw.ink import page_model as pm_mod
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig

H, W = 96, 128
PAPER = 235.0
INK = 40.0


def _paper(shape=(H, W)):
    return np.full(shape, PAPER, dtype=np.float32)


def test_pending_until_confirm_observations():
    model = pm_mod.PageModel((H, W), confirm_obs=3)
    gray = _paper()
    for _ in range(2):
        model.observe(gray)
    assert np.count_nonzero(model.seeded) == 0
    assert np.count_nonzero(model.admissible()) == 0
    model.observe(gray)
    assert np.all(model.seeded)
    assert np.all(model.admissible())


def test_occluded_block_never_seeds_and_revealed_content_is_not_novel():
    model = pm_mod.PageModel((H, W), confirm_obs=2)
    gray = _paper()
    gray[32:64, 32:64] = INK  # desenho antigo sob a mao
    covered = np.ones((H, W), dtype=bool)
    covered[32:64, 32:64] = False
    for _ in range(5):
        model.observe(gray, visible=covered)
    # O bloco central nunca foi visto: pendente, sem evidencia.
    assert not np.any(model.seeded[32:64, 32:64])
    assert not np.any(model.admissible()[32:64, 32:64])
    # A mao sai: a primeira observacao limpa semeia COM o desenho —
    # ele e pagina, nao tinta nova (o "pending" do #130).
    for _ in range(2):
        model.observe(gray)
    assert np.all(model.seeded)
    assert not np.any(model.novel(gray))
    assert not np.any(model.pending())


def test_seeded_content_stays_not_novel_but_new_stroke_is():
    model = pm_mod.PageModel((H, W), confirm_obs=2)
    gray = _paper()
    gray[10:20, 10:20] = INK  # conteudo presente desde a semeadura
    for _ in range(2):
        model.observe(gray)
    assert not np.any(model.novel(gray))
    # Traco novo, mais escuro que a referencia: conteudo novo.
    after = gray.copy()
    after[50:60, 50:60] = INK
    novel = model.novel(after)
    assert np.all(novel[50:60, 50:60])
    assert np.count_nonzero(novel) == 100


def test_erasure_is_not_novel():
    model = pm_mod.PageModel((H, W), confirm_obs=1)
    gray = _paper()
    gray[10:20, 10:20] = INK
    model.observe(gray)
    erased = _paper()
    # Pixel mais CLARO que a referencia e mudanca, mas nao tinta:
    # borracha e mao saindo nao podem escrever.
    assert not np.any(model.novel(erased))


def test_tracking_absorbs_shading_drift():
    model = pm_mod.PageModel((H, W), confirm_obs=1, rate=0.5)
    gray = _paper()
    model.observe(gray, paper=np.ones((H, W), dtype=bool))
    # Sombra chegando devagar: a referencia segue o papel e o pixel
    # nunca vira "novo" — a oscilacao do #133 afunda na referencia.
    for level in (220, 200, 180):
        dark = np.full((H, W), float(level), dtype=np.float32)
        model.observe(dark, paper=np.ones((H, W), dtype=bool))
    for _ in range(8):
        model.observe(
            np.full((H, W), 180.0, dtype=np.float32),
            paper=np.ones((H, W), dtype=bool),
        )
    assert not np.any(model.novel(np.full((H, W), 180.0, np.float32)))
    np.testing.assert_allclose(model.reference, 180.0, atol=1.0)


def test_ink_never_sinks_into_reference():
    model = pm_mod.PageModel((H, W), confirm_obs=1, rate=0.5)
    model.observe(_paper(), paper=np.ones((H, W), dtype=bool))
    stroked = _paper()
    stroked[40:50, 40:50] = INK
    paper = np.ones((H, W), dtype=bool)
    paper[40:50, 40:50] = False  # a trabalhadora exclui evidencia de tinta
    for _ in range(40):
        model.observe(stroked, paper=paper)
    # Sem a mascara `paper` a referencia engoliria o traco e ele
    # deixaria de ser novo — o afundamento que a exclusao impede.
    assert np.all(model.novel(stroked)[40:50, 40:50])
    np.testing.assert_allclose(model.reference[40:50, 40:50], PAPER)


def test_ragged_grid_seeds_partial_border_blocks():
    model = pm_mod.PageModel((70, 100), confirm_obs=2)
    gray = _paper((70, 100))
    for _ in range(2):
        model.observe(gray)
    assert np.all(model.seeded)


def test_partial_visibility_block_threshold():
    model = pm_mod.PageModel((H, W), confirm_obs=1, block=32)
    visible = np.zeros((H, W), dtype=bool)
    visible[:32, :16] = True  # metade de um bloco de 32: visto
    visible[:32, 32:48] = True
    model.observe(_paper(), visible=visible)
    # O bloco [0:32, 0:32] tem 50% visivel -> semeado; o que ele
    # semeou foi so a metade vista — a metade coberta segue pendente.
    assert np.all(model.seeded[:32, :16])
    assert not np.any(model.seeded[:32, 16:32])
    assert np.all(model.seeded[:32, 32:48])


def test_novel_and_admissible_need_reference():
    model = pm_mod.PageModel((H, W), confirm_obs=1)
    gray = _paper()
    assert not np.any(model.novel(gray))
    assert not np.any(model.admissible())
    assert np.all(model.pending())


def test_shape_mismatch_raises():
    model = pm_mod.PageModel((H, W))
    with pytest.raises(ValueError):
        model.observe(np.zeros((H + 1, W), dtype=np.float32))


@pytest.mark.slow
def test_observe_budget_on_a4_grid():
    model = pm_mod.PageModel((1200, 850), confirm_obs=3)
    gray = _paper((1200, 850))
    visible = np.ones((1200, 850), dtype=bool)
    paper = np.ones((1200, 850), dtype=bool)
    model.observe(gray, visible=visible, paper=paper)  # aquece
    start = time.monotonic()
    for _ in range(20):
        model.observe(gray, visible=visible, paper=paper)
    per_pass_ms = (time.monotonic() - start) / 20 * 1000
    assert per_pass_ms < 10.0


class _ScriptInk:
    """Tinta falsa guiada pelo conteudo do proprio warp.

    ink_prob devolve alto onde a imagem esta escura — o par
    (imagem, prob) da passada fica consistente como na passada real.
    """

    threshold = 0.7

    def ink_prob(self, warped):
        gray = warped if warped.ndim == 2 else warped.mean(axis=2)
        return (gray < 100).astype(np.float32) * 0.95


def _bgr(gray):
    return np.stack([gray] * 3, axis=-1).astype(np.uint8)


Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)


def _drive(worker, generation, image, visible=None):
    before = worker.stats()["passes_total"]
    worker.submit(WarpedPage(_bgr(image), generation, Q0, visible))
    deadline = time.monotonic() + 5.0
    while worker.stats()["passes_total"] <= before:
        if time.monotonic() > deadline:
            raise AssertionError("a passada nao executou")
        time.sleep(0.001)


def _confirmed(worker, generation):
    snap = worker.poll(generation, 0)
    return np.zeros((H, W), dtype=np.uint8) if snap is None else snap.confirmed


def _worker(config):
    return worker_mod.PageCleanWorker(
        _ScriptInk(),
        config=config,
    ).start()


def test_page_model_suppresses_pre_existing_content(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        image[10:20, 10:20] = INK  # desenho antigo ja na folha
        for _ in range(7):
            _drive(worker, generation, image)
        canvas = _confirmed(worker, generation)
        # Semeado com o desenho dentro: nunca foi conteudo novo.
        assert np.count_nonzero(canvas) == 0
        # Controle na mesma cena sem o modelo: o voto confirma.
        worker2 = _worker(InkConfig())
        try:
            gen2 = worker2.new_generation()
            for _ in range(7):
                _drive(worker2, gen2, image)
            assert np.count_nonzero(_confirmed(worker2, gen2)) == 100
        finally:
            worker2.stop()
    finally:
        worker.stop()


def test_page_model_confirms_new_stroke_after_seeding(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, generation, image)
        image[50:60, 50:60] = INK  # traco novo apos a referencia
        for _ in range(4):
            _drive(worker, generation, image)
        canvas = _confirmed(worker, generation)
        assert np.count_nonzero(canvas) == 100
    finally:
        worker.stop()


def test_page_model_pending_region_freezes_then_seeds_content(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        image[40:60, 40:60] = INK  # conteudo sob a mao desde o inicio
        covered = np.ones((H, W), dtype=bool)
        covered[32:64, 32:64] = False
        for _ in range(5):
            _drive(worker, generation, image, visible=covered)
        assert np.count_nonzero(_confirmed(worker, generation)) == 0
        # Mao fora: o bloco ainda esta pendente — conteudo revelado
        # nao vota ate semear; quando semeia, semeia COM o conteudo.
        for _ in range(6):
            _drive(worker, generation, image)
        assert np.count_nonzero(_confirmed(worker, generation)) == 0
    finally:
        worker.stop()


def test_page_model_seeds_the_full_grid(monkeypatch):
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, generation, image)
        assert worker._page_model is not None
        assert worker._page_model.seeded_fraction == pytest.approx(1.0)
    finally:
        worker.stop()


def _drive_page(worker, generation, page):
    before = worker.stats()["passes_total"]
    worker.submit(page)
    deadline = time.monotonic() + 5.0
    while worker.stats()["passes_total"] <= before:
        if time.monotonic() > deadline:
            raise AssertionError("a passada nao executou")
        time.sleep(0.001)


def test_page_model_reacquires_same_sheet_with_baseline_and_handwriting(monkeypatch):
    """Issue 1: marcas de baseline suprimidas do canvas nao derrubam containment."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        # 900 px de conteudo pre-existente (faixas finas para passar na guarda 32x32)
        image[10:20, 10:55] = INK
        image[24:34, 10:55] = INK
        for _ in range(3):
            _drive(worker, gen1, image)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # 100 px de escrita nova confirmada (containment sem baseline = 100/1000 = 10% < 50%)
        image[60:70, 80:90] = INK
        for _ in range(4):
            _drive(worker, gen1, image)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 100
        pm_before = worker._page_model

        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, image)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert events[-1]["recall"] == pytest.approx(1.0)
        assert events[-1]["containment"] == pytest.approx(1.0)
        confirmed = _confirmed(worker, gen2)
        assert np.count_nonzero(confirmed) == 100
        assert np.all(confirmed[60:70, 80:90] > 0)
        assert not np.any(confirmed[10:34, 10:55] > 0)
        assert worker._page_model is pm_before
    finally:
        worker.stop()


def test_page_model_reacquisition_control_rejects_different_sheet(monkeypatch):
    """Issue 1 (controle): folha diferente (mesmo com a mesma pauta impressa) e rejeitada."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        image[10:20, 10:55] = INK
        image[24:34, 10:55] = INK
        for _ in range(3):
            _drive(worker, gen1, image)
        image[60:70, 80:90] = INK
        for _ in range(4):
            _drive(worker, gen1, image)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 100
        pm_before = worker._page_model

        # Outra folha que compartilha os mesmos 900 px de baseline mas tem escrita diferente
        other = np.full((H, W), PAPER, dtype=np.float32)
        other[10:20, 10:55] = INK
        other[24:34, 10:55] = INK
        other[75:85, 15:65] = INK
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive(worker, gen2, other)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
        assert np.count_nonzero(_confirmed(worker, gen2)) == 0
        assert worker._page_model is not pm_before
    finally:
        worker.stop()


def test_page_model_reacquires_small_stroke_with_printed_baseline_and_visible_mask(
    monkeypatch,
):
    """R1767-1774: referencia confirmada < 32 px com baseline impresso e visible nao defere."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        image = np.full((H, W), PAPER, dtype=np.float32)
        # 900 px de baseline impresso
        image[10:20, 10:55] = INK
        image[24:34, 10:55] = INK
        for _ in range(3):
            _drive(worker, gen1, image, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Ponto manuscrito de 20 px (< _IDENTITY_MIN_VISIBLE_PX = 32)
        image[60:64, 80:85] = INK
        for _ in range(4):
            _drive(worker, gen1, image, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 20

        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, image, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert events[-1]["deferred"] is False
        assert np.count_nonzero(_confirmed(worker, gen2)[60:64, 80:85]) == 20
    finally:
        worker.stop()


def test_page_model_clean_replacement_with_adjacent_print_rejects_old_tick(
    monkeypatch,
):
    """R1805-1812: copia limpa com traco impresso colado ao tick manuscrito nao herda o tick."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        form = np.full((H, W), PAPER, dtype=np.float32)
        # Traco impresso [20:25, 50:55] (25 px) + pauta impressa
        form[20:25, 50:55] = INK
        form[10:15, 10:40] = INK
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Tick manuscrito de 10 px [20:25, 55:57] colado ao traco impresso [20:25, 50:55]
        with_tick = form.copy()
        with_tick[20:25, 55:57] = INK
        for _ in range(4):
            _drive(worker, gen1, with_tick, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 10

        # Caso 1: a mesma folha com o tick retorna. A pauta impressa e
        # tao escura quanto o tick (INK=40): o pixel escuro na posicao
        # salva pode ser a pauta ou o traco — a passada nao distingue
        # por aparencia. O veredito honesto e INCONCLUIDO com hold:
        # nao vota contra, nao herda para copia identica e protege a
        # referencia para sempre — repetir a passada nunca decide.
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(6):
            _drive(worker, gen2, with_tick, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "deferred_occlusion"
        assert np.count_nonzero(worker._identity_reference[20:25, 55:57]) == 10

        # Caso 1b: a mesma folha retorna com residuo de 1 px para a esquerda
        # (impresso em [20:25, 49:54], tick em [20:25, 54:56]) — mesma
        # ambiguidade fotometrica: protegido, nao destruido.
        shifted_with_tick = np.full((H, W), PAPER, dtype=np.float32)
        shifted_with_tick[20:25, 49:54] = INK
        shifted_with_tick[10:15, 9:39] = INK
        shifted_with_tick[20:25, 54:56] = INK
        gen2b = worker.new_generation(preserve_identity=True)
        for _ in range(6):
            _drive(worker, gen2b, shifted_with_tick, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "deferred_occlusion"
        assert np.count_nonzero(worker._identity_reference[20:25, 55:57]) == 10

        # Caso 2: copia limpa do formulario (sem o tick manuscrito, mesmo deslocada 1 px)
        # entra -> new_page limpa o tick
        shifted_form = np.full((H, W), PAPER, dtype=np.float32)
        shifted_form[20:25, 49:54] = INK
        shifted_form[10:15, 9:39] = INK
        gen3 = worker.new_generation(preserve_identity=True)
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive(worker, gen3, shifted_form, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
        assert np.count_nonzero(_confirmed(worker, gen3)) == 0
    finally:
        worker.stop()


def test_page_model_reacquires_ink_traced_over_printed_guide_line(monkeypatch):
    """R1798-1802: tinta desenhada sobre pauta impressa nao derruba recall na reaquisicao."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        # Pauta impressa com luminancia 95 (acima do limiar de tinta do stub, mas mais clara que INK=40)
        PRINTED_GUIDE = 95.0
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[30:36, 20:90] = PRINTED_GUIDE
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Usuario traca por cima da pauta impressa (70% sobre o baseline, 30% fora)
        traced = form.copy()
        traced[31:35, 30:65] = INK  # 140 px sobre a pauta impressa
        traced[36:38, 40:60] = INK  # 40 px saindo da pauta
        for _ in range(4):
            _drive(worker, gen1, traced, visible=vis_all)
        confirmed_before = int(np.count_nonzero(_confirmed(worker, gen1)))
        assert confirmed_before == 180

        # A mesma folha retorna -> same_page preserva toda a tinta confirmada
        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, traced, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2))) == confirmed_before
    finally:
        worker.stop()


def test_page_model_reacquires_visible_tick_beside_occluded_printed_mark(monkeypatch):
    """R240-244: marca impressa oculta pela mao nao pode se alinhar sobre o tick visivel."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        form = np.full((H, W), PAPER, dtype=np.float32)
        # Marca impressa de 2 colunas [20:25, 53:55]
        form[20:25, 53:55] = INK
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Tick manuscrito de 2 colunas [20:25, 55:57] imediatamente a direita da marca impressa
        with_tick = form.copy()
        with_tick[20:25, 55:57] = INK
        for _ in range(4):
            _drive(worker, gen1, with_tick, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 10

        # Na reaquisicao, a mao cobre a marca impressa [20:25, 53:55] mas deixa o
        # tick visivel. O tick e tao escuro quanto a pauta (INK=40): escuro na
        # posicao salva sem escurecer a referencia local e INCONCLUSIVO — o
        # hold protege a referencia; repetir a passada nunca condena.
        vis_hand = np.ones((H, W), dtype=bool)
        vis_hand[18:27, 45:55] = False
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(6):
            _drive(worker, gen2, with_tick, visible=vis_hand)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "deferred_occlusion"
        assert np.count_nonzero(worker._identity_reference[20:25, 55:57]) == 10
    finally:
        worker.stop()


def test_page_model_reacquires_partially_covered_stroke_crossing_printed_rule(
    monkeypatch,
):
    """R1847-1858: traco de 60 px (10 sobre pauta, 50 fora) com 20 px cobertos pela mao (30 fora visiveis) nao defere."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        PRINTED_GUIDE = 95.0
        form = np.full((H, W), PAPER, dtype=np.float32)
        # Pauta impressa horizontal nas linhas 30..32
        form[30:32, 10:100] = PRINTED_GUIDE
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Traco vertical de 60 px (12 linhas x 5 colunas: linhas 28..40, colunas 50..55)
        # 10 px cruzam a pauta [30:32, 50:55] e 50 px ficam fora da pauta
        with_stroke = form.copy()
        with_stroke[28:40, 50:55] = INK
        for _ in range(4):
            _drive(worker, gen1, with_stroke, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 60

        # Na reaquisicao, a mao cobre as ultimas 4 linhas do traco [36:40, 50:55] (20 px fora da pauta),
        # restando 40 px visiveis do traco (10 sobre a pauta + 30 fora da pauta < 32).
        vis_partial = np.ones((H, W), dtype=bool)
        vis_partial[36:45, 45:60] = False
        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, with_stroke, visible=vis_partial)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert events[-1]["deferred"] is False
        assert np.count_nonzero(_confirmed(worker, gen2)[28:40, 50:55]) == 60
    finally:
        worker.stop()


@pytest.mark.slow
def test_identity_recovery_budget_on_a4_grid(monkeypatch):
    """worker.py:236 & test_page_model.py:575: avaliacao de identidade A4 com marcas em cantos opostos < 10 ms e fora do lock do worker."""
    shape = (1200, 850)
    ref = np.zeros(shape, dtype=np.uint8)
    ref[400:420, 300:350] = 255
    baseline = np.zeros(shape, dtype=bool)
    # Marcas impressas em cantos opostos cobrindo quase toda a folha A4
    baseline[10:15, 10:200] = True
    baseline[200:205, 100:700] = True
    baseline[1180:1185, 650:840] = True
    candidate = np.zeros(shape, dtype=np.float32)
    candidate[11:16, 11:201] = 0.95
    candidate[201:206, 101:701] = 0.95
    candidate[1181:1186, 651:841] = 0.95
    candidate[401:421, 301:351] = 0.95
    visible = np.ones(shape, dtype=bool)
    worker_mod._evaluate_page_identity(
        ref, baseline, candidate, visible, None, None, False, 0.7
    )
    start = time.monotonic()
    for _ in range(20):
        res = worker_mod._evaluate_page_identity(
            ref, baseline, candidate, visible, None, None, False, 0.7
        )
    per_pass_ms = (time.monotonic() - start) / 20 * 1000
    assert res[0] is True
    # O dominio ambiguo (fotometria local) e o guard-full adicionaram
    # ~1 ms ao custo base de ~8 ms medido em 739180e — o bound pega
    # regressao de ordem de grandeza (varredura float inteira,
    # dilatacao global), nao ruido de ~15% sob carga.
    assert per_pass_ms < 15.0

    # Verifica na trabalhadora real que _evaluate_page_identity roda fora de _condition
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    orig_eval = worker_mod._evaluate_page_identity

    def _hooked_eval(*args, **kwargs):
        entered.set()
        release.wait(timeout=5.0)
        return orig_eval(*args, **kwargs)

    monkeypatch.setattr(worker_mod, "_evaluate_page_identity", _hooked_eval)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[10:15, 10:40] = INK
        for _ in range(3):
            _drive(worker, gen1, form)
        with_ink = form.copy()
        with_ink[50:60, 50:60] = INK
        for _ in range(4):
            _drive(worker, gen1, with_ink)

        gen2 = worker.new_generation(preserve_identity=True)
        worker.submit(WarpedPage(_bgr(with_ink), gen2, Q0))
        assert entered.wait(timeout=5.0)

        t0 = time.monotonic()
        worker.submit(WarpedPage(_bgr(with_ink), gen2, Q0))
        elapsed = time.monotonic() - t0
        assert elapsed < 0.2
        assert not release.is_set()
        release.set()

        _drive(worker, gen2, with_ink)
        assert worker.error is None
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
    finally:
        release.set()
        worker.stop()


def test_page_model_reacquires_repeated_printed_rules_over_4096_px_with_local_residual(
    monkeypatch,
):
    """R238-241: >4096 px de pautas impressas 100x2 com residuo local de 1 px nao escolhe shift errado."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    h_grid, w_grid = 240, 200
    q_grid = np.array(
        [(0, 0), (w_grid, 0), (w_grid, h_grid), (0, h_grid)], dtype=np.float32
    )
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((h_grid, w_grid), dtype=bool)
        form = np.full((h_grid, w_grid), PAPER, dtype=np.float32)
        # 30 pautas horizontais de 2x100 px = 6000 px (> 4096 px)
        for idx in range(30):
            r0 = 10 + idx * 6
            form[r0 : r0 + 2, 10:110] = INK
        for _ in range(3):
            _drive_page(
                worker,
                gen1,
                WarpedPage(_bgr(form), gen1, q_grid, visible=vis_all),
            )
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Marca manuscrita separada de 30 px (6x5)
        with_mark = form.copy()
        with_mark[20:26, 150:155] = INK
        for _ in range(4):
            _drive_page(
                worker,
                gen1,
                WarpedPage(_bgr(with_mark), gen1, q_grid, visible=vis_all),
            )
        assert np.count_nonzero(_confirmed(worker, gen1)) == 30

        # Na reaquisicao, as ultimas 5 pautas (idx 25..29) tem residuo de 1 px para a direita,
        # enquanto as outras 25 permanecem no lugar.
        returning = np.full((h_grid, w_grid), PAPER, dtype=np.float32)
        for idx in range(30):
            r0 = 10 + idx * 6
            c0 = 10 + (1 if idx >= 25 else 0)
            returning[r0 : r0 + 2, c0 : c0 + 100] = INK
        returning[20:26, 150:155] = INK

        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive_page(
                worker,
                gen2,
                WarpedPage(_bgr(returning), gen2, q_grid, visible=vis_all),
            )
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert np.count_nonzero(_confirmed(worker, gen2)[20:26, 150:155]) == 30
    finally:
        worker.stop()


def test_page_model_reference_stays_within_camera_range_after_large_brightness_jump():
    """page_model.py:170: salto grande de brilho rejeitado por _estimate_gain nao estoura 255."""
    model = pm_mod.PageModel((H, W), confirm_obs=1, rate=0.5)
    dim = np.full((H, W), 90.0, dtype=np.float32)
    paper_mask = np.ones((H, W), dtype=bool)
    model.observe(dim, paper=paper_mask)
    assert float(model.reference.max()) == pytest.approx(90.0)

    # Salto de 90 -> 245 (gain_p90 = 2.72 > 2.0, rejeitado pelo _estimate_gain)
    bright = np.full((H, W), 245.0, dtype=np.float32)
    for _ in range(10):
        model.observe(bright, paper=paper_mask)
        assert float(model.reference.max()) <= 255.0
    np.testing.assert_allclose(model.reference, 245.0, atol=1.0)


def test_page_model_supports_consecutive_cornerless_submissions(monkeypatch):
    """Issue 2: submissoes consecutivas sem cantos usam identidade sem crashar."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        image[10:20, 10:20] = INK
        for _ in range(3):
            _drive_page(worker, generation, WarpedPage(_bgr(image), generation))
        assert worker.error is None
        assert worker.last_registration is None
        assert worker._page_model is not None
        assert worker._page_model.seeded_fraction == pytest.approx(1.0)
        assert np.count_nonzero(_confirmed(worker, generation)) == 0

        image[50:60, 50:60] = INK
        for _ in range(4):
            _drive_page(worker, generation, WarpedPage(_bgr(image), generation))
        assert worker.error is None
        confirmed = _confirmed(worker, generation)
        assert np.count_nonzero(confirmed) == 100
        assert np.all(confirmed[50:60, 50:60] > 0)
    finally:
        worker.stop()


def test_page_model_handles_cornered_then_cornerless_submission(monkeypatch):
    """Issue 2: passada com cantos seguida de passada sem cantos nao derruba worker."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        _drive_page(worker, generation, WarpedPage(_bgr(image), generation, Q0))
        _drive_page(worker, generation, WarpedPage(_bgr(image), generation, None))
        _drive_page(worker, generation, WarpedPage(_bgr(image), generation, None))
        assert worker.error is None
        assert worker.last_registration is None
        assert worker._page_model.seeded_fraction == pytest.approx(1.0)
    finally:
        worker.stop()


def test_page_model_seeding_remaining_ignores_hidden_blocks():
    """Issue 4: seeding_remaining conta apenas blocos expostos em semeadura ativa."""
    model = pm_mod.PageModel((H, W), confirm_obs=3)
    gray = _paper()
    vis = np.ones((H, W), dtype=bool)
    vis[0:32, 0:64] = False
    for _ in range(3):
        model.observe(gray, visible=vis)
    assert model.seeding_remaining == 0

    # Expõe [0:32, 0:32] mantendo [0:32, 32:64] oculto (streak=0)
    vis[0:32, 0:32] = True
    model.observe(gray, visible=vis)
    assert model.seeding_remaining == 2
    model.observe(gray, visible=vis)
    assert model.seeding_remaining == 1
    model.observe(gray, visible=vis)
    assert model.seeding_remaining == 0
    assert np.all(model.seeded[0:32, 0:32])
    assert not np.any(model.seeded[0:32, 32:64])


def test_page_model_hand_exit_seeds_exposed_blocks_before_rest_and_allows_writing(
    monkeypatch,
):
    """Issue 4: saida da mao mantem passadas suficientes para semear bloco recem-exposto."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.05)
    now = [1000.0]
    worker = worker_mod.PageCleanWorker(
        _ScriptInk(),
        config=InkConfig(page_model=True),
        clock=lambda: now[0],
    )
    generation = worker.new_generation()
    image = np.full((H, W), PAPER, dtype=np.float32)
    covered = np.ones((H, W), dtype=bool)
    covered[32:64, 32:64] = False

    worker.set_hand_presence(True)
    for _ in range(4):
        worker.submit(WarpedPage(_bgr(image), generation, Q0, covered))
        assert worker.pump()
        now[0] += 0.05
    assert not np.any(worker._page_model.seeded[32:64, 32:64])

    # Mao sai e folha fica limpa: deve semear o bloco recem-exposto antes de descansar
    worker.set_hand_presence(False)
    for _ in range(10):
        worker.submit(WarpedPage(_bgr(image), generation, Q0))
        worker.pump()
        now[0] += 0.06
        if worker.stats()["resting"]:
            break
    assert worker.stats()["resting"]
    assert np.all(worker._page_model.seeded[32:64, 32:64])

    # Pausa ociosa de 10 s: nenhuma passada roda em descanso
    passes_at_rest = worker.stats()["passes_total"]
    for _ in range(10):
        worker.submit(WarpedPage(_bgr(image), generation, Q0))
        assert not worker.pump()
        now[0] += 1.0
    assert worker.stats()["passes_total"] == passes_at_rest

    # Escrita subsequente no bloco recem-semeado confirma normalmente
    worker.set_hand_presence(True, moved=True)
    image[40:50, 40:50] = INK
    for _ in range(4):
        worker.submit(WarpedPage(_bgr(image), generation, Q0))
        assert worker.pump()
        now[0] += 0.05
    snap = worker.poll(generation, 0)
    assert snap is not None
    assert np.count_nonzero(snap.confirmed[40:50, 40:50]) == 100


def test_page_model_rests_when_block_remains_hidden(monkeypatch):
    """Issue 4: bloco que permanece oculto nao impede o descanso total."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.05)
    now = [1000.0]
    worker = worker_mod.PageCleanWorker(
        _ScriptInk(),
        config=InkConfig(page_model=True),
        clock=lambda: now[0],
    )
    generation = worker.new_generation()
    image = np.full((H, W), PAPER, dtype=np.float32)
    covered = np.ones((H, W), dtype=bool)
    covered[0:32, 0:32] = False
    covered[32:64, 32:64] = False

    worker.set_hand_presence(True)
    for _ in range(4):
        worker.submit(WarpedPage(_bgr(image), generation, Q0, covered))
        assert worker.pump()
        now[0] += 0.05

    # Mao sai, [32:64, 32:64] exposto mas [0:32, 0:32] segue oculto
    still_hidden = np.ones((H, W), dtype=bool)
    still_hidden[0:32, 0:32] = False
    worker.set_hand_presence(False)
    for _ in range(12):
        worker.submit(WarpedPage(_bgr(image), generation, Q0, still_hidden))
        worker.pump()
        now[0] += 0.06
        if worker.stats()["resting"]:
            break
    assert worker.stats()["resting"]
    assert np.all(worker._page_model.seeded[32:64, 32:64])
    assert not np.any(worker._page_model.seeded[0:32, 0:32])


def test_page_model_resets_on_identity_timeout_within_same_generation(monkeypatch):
    """Issue 5: quarantine_timeout reseta o PageModel junto do acumulador."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, gen1, image)
        image[10:20, 10:20] = INK
        for _ in range(4):
            _drive(worker, gen1, image)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 100

        gen2 = worker.new_generation(preserve_identity=True)
        occluded = np.ones((H, W), dtype=bool)
        occluded[0:32, 0:32] = False
        for _ in range(identity_mod._PAGE_IDENTITY_MAX_DEFERRALS):
            _drive(worker, gen2, image, visible=occluded)
        assert (
            worker.telemetry_snapshot()["page_identity_events"][-1]["decision"]
            == "deferred_occlusion"
        )
        assert worker._page_model is not None

        # Passada que estoura a quarentena: reseta acumulador e PageModel
        _drive(worker, gen2, image, visible=occluded)
        assert (
            worker.telemetry_snapshot()["page_identity_events"][-1]["decision"]
            == "quarantine_timeout"
        )
        assert worker._page_model is None

        # Nova folha na mesma geracao: marca pre-existente semeia e traco novo confirma
        new_sheet = np.full((H, W), PAPER, dtype=np.float32)
        new_sheet[50:60, 50:60] = INK
        for _ in range(3):
            _drive(worker, gen2, new_sheet)
        assert np.count_nonzero(_confirmed(worker, gen2)) == 0
        new_sheet[10:20, 10:20] = INK
        for _ in range(4):
            _drive(worker, gen2, new_sheet)
        assert np.count_nonzero(_confirmed(worker, gen2)[10:20, 10:20]) == 100
    finally:
        worker.stop()


def test_page_model_resets_on_new_page_after_deferred_occlusion(monkeypatch):
    """Issue 5: new_page apos deferred_occlusion descarta a referencia da ancora velha."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, gen1, image)
        image[10:20, 10:20] = INK
        for _ in range(4):
            _drive(worker, gen1, image)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 100

        gen2 = worker.new_generation(preserve_identity=True)
        occluded = np.ones((H, W), dtype=bool)
        occluded[0:32, 0:32] = False
        _drive(worker, gen2, image, visible=occluded)
        old_pm = worker._page_model
        assert old_pm is not None

        # Folha diferente visivel: 3 passadas de consenso -> new_page
        other = np.full((H, W), PAPER, dtype=np.float32)
        other[60:70, 60:70] = INK
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive(worker, gen2, other)
        assert (
            worker.telemetry_snapshot()["page_identity_events"][-1]["decision"]
            == "new_page"
        )
        assert worker._page_model is not old_pm
        # Mais 2 passadas terminam de semear a folha nova com [60:70, 60:70] como baseline
        for _ in range(2):
            _drive(worker, gen2, other)
        assert np.count_nonzero(_confirmed(worker, gen2)) == 0
        # Traco novo desenhado em [10:20, 10:20] (onde havia tinta na pagina velha) confirma
        other[10:20, 10:20] = INK
        for _ in range(4):
            _drive(worker, gen2, other)
        assert np.count_nonzero(_confirmed(worker, gen2)[10:20, 10:20]) == 100
    finally:
        worker.stop()


def test_page_model_clean_copy_does_not_inherit_hidden_sketch(monkeypatch):
    """Review PR #145: folha limpa do mesmo formulario com a pauta
    impressa vizinha do rascunho nao restaura o desenho velho — a
    dilatacao do recall nao pode contar a pele da pauta como escrita."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        form = np.full((H, W), PAPER, dtype=np.float32)
        # Marca impressa curta [40:42, 47:60], 2 px acima do rascunho.
        form[40:42, 47:60] = 95.0
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Rascunho manuscrito de 15 px logo abaixo da marca.
        with_dash = form.copy()
        with_dash[44:47, 50:55] = INK
        for _ in range(7):
            _drive(worker, gen1, with_dash, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 15

        # Copia limpa: mesma pauta, um pouco mais escura — a pele de
        # 1 px fora da mascara aprendida entra em `seen`, a 2-3 px da
        # posicao salva do rascunho (dentro do raio do recall).
        clean_copy = np.full((H, W), PAPER, dtype=np.float32)
        clean_copy[40:43, 47:60] = 95.0
        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, clean_copy, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] != "same_page"
        assert np.count_nonzero(_confirmed(worker, gen2)) == 0

        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive(worker, gen2, clean_copy, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
        confirmed = _confirmed(worker, gen2)
        assert np.count_nonzero(confirmed[44:47, 50:55]) == 0
    finally:
        worker.stop()


def test_page_model_reacquires_small_stroke_on_blank_paper(monkeypatch):
    """Review PR #145: marca de 20 px em folha SEM pauta (baseline
    capturado vazio) restaura — o cap do piso depende do rastreio
    de baseline estar ativo, nao de haver pixels impressos."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, gen1, image, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Ponto manuscrito de 20 px (< _IDENTITY_MIN_VISIBLE_PX = 32)
        image[60:64, 80:85] = INK
        for _ in range(4):
            _drive(worker, gen1, image, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 20

        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, image, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert events[-1]["deferred"] is False
        assert np.count_nonzero(_confirmed(worker, gen2)[60:64, 80:85]) == 20
    finally:
        worker.stop()


def test_page_model_reacquires_thin_stroke_shifted_with_print(monkeypatch):
    """Review PR #145: traco de 1 px colado a pauta que desceu 1 px
    junto com ela nao vira folha nova — o deslocamento medido na
    pauta desloca o quadro da escrita salva tambem."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        form = np.full((H, W), PAPER, dtype=np.float32)
        # Pauta impressa de 2 px [40:42, 10:100].
        form[40:42, 10:100] = 95.0
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Traco manuscrito de 1 px 1 px abaixo da pauta [43, 40:60].
        with_stroke = form.copy()
        with_stroke[43, 40:60] = INK
        for _ in range(7):
            _drive(worker, gen1, with_stroke, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 20

        # A MESMA folha volta com pauta e traco deslocados 1 px para
        # baixo: o traco em 44 cai na margem dilatada da pauta sem
        # tocar a posicao salva — sem o quadro deslocado ele era
        # descartado e a folha votava como nova.
        shifted = np.full((H, W), PAPER, dtype=np.float32)
        shifted[41:43, 10:100] = 95.0
        shifted[44, 40:60] = INK
        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, shifted, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert np.count_nonzero(_confirmed(worker, gen2)[43, 40:60]) == 20

        # Controle: copia limpa deslocada do mesmo formulario (sem o
        # traco) NAO herda a escrita — o quadro deslocado admite so
        # re-observacao, e na copia nao ha traco onde ele espera.
        clean_shifted = np.full((H, W), PAPER, dtype=np.float32)
        clean_shifted[41:43, 10:100] = 95.0
        gen3 = worker.new_generation(preserve_identity=True)
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive(worker, gen3, clean_shifted, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
        assert np.count_nonzero(_confirmed(worker, gen3)[43:45, 40:60]) == 0
    finally:
        worker.stop()


def test_page_model_occlusion_band_enters_hold_and_restores(monkeypatch):
    """Review PR #145: mao cobrindo a maior parte da escrita que
    cruza pauta entra em hold de oclusao em vez de queimar a
    quarentena — o desconto ambiguo vale no gate precoce de
    evidencia, nao no piso estrito de restauracao."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        form = np.full((H, W), PAPER, dtype=np.float32)
        # Pauta impressa de 4 px [30:34, 10:100].
        form[30:34, 10:100] = 95.0
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Traco vertical de 220 px [28:50, 50:60]: 40 px sobre a
        # pauta (ambiguos) e 180 px julgaveis.
        with_stroke = form.copy()
        with_stroke[28:50, 50:60] = INK
        for _ in range(7):
            _drive(worker, gen1, with_stroke, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 220

        # Mao cobre as linhas >= 35 da regiao do traco: restam 20 px
        # julgaveis visiveis [28:30] e 40 px ambiguos [30:34]. O gate
        # precoce descontado deixa o frame chegar ao match, que segura
        # a decisao em hold sem queimar a quarentena.
        vis_hand = np.ones((H, W), dtype=bool)
        vis_hand[35:, 45:65] = False
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(identity_mod._PAGE_IDENTITY_MAX_DEFERRALS + 2):
            _drive(worker, gen2, with_stroke, visible=vis_hand)
        events = worker.telemetry_snapshot()["page_identity_events"]
        # Alem do orcamento de quarentena (5): com hold a referencia
        # segue pendente — sem ele o sexto frame a destruia.
        assert events[-1]["decision"] == "deferred_occlusion"
        assert events[-1]["quarantine_frames"] == 0

        # Mao sai: a folha inteira visivel restaura o desenho exato.
        _drive(worker, gen2, with_stroke, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert np.count_nonzero(_confirmed(worker, gen2)[28:50, 50:60]) == 220
    finally:
        worker.stop()


def test_page_model_baseline_footprint_without_border_guard(monkeypatch):
    """Review PR #145: com a guarda de borda desligada, o padding do
    registro nao semeia a referencia de pagina limpa — a pegada do
    modelo nao depende da guarda."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True, border_guard_px=0, paper_guard=False))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        # 1a passada estabelece a ancora no quad inteiro.
        _drive(worker, generation, image)

        # Folha recuada 6 px a direita e 4 px para baixo: o warp deixa
        # padding (zeros) nas linhas < 4 e colunas < 6 da ancora. Sem
        # mascara de visibilidade, sem guarda de borda e sem a guarda
        # de papel (que ja rejeitaria o preto do padding), a pegada e
        # a unica coisa que separa "nao observei" de "observei vazio".
        shifted_quad = Q0 + np.float32([[6, 4], [6, 4], [6, 4], [6, 4]])
        for _ in range(6):
            _drive_page(
                worker,
                generation,
                WarpedPage(_bgr(image), generation, shifted_quad),
            )
        model = worker._page_model
        assert model is not None
        assert not np.any(model.seeded[:3, :4])
        assert np.all(model.seeded[32:64, 32:64])
    finally:
        worker.stop()


def test_page_model_does_not_seed_synthetic_fill_when_coverage_expands(monkeypatch):
    """Issue 6: preenchimento sintetico fora de coverage nao semeia baseline."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        real_sheet = np.full((H, W), PAPER, dtype=np.float32)
        real_sheet[10:20, 10:20] = INK  # marca pre-existente na faixa esquerda

        # Cobertura inicial exclui as primeiras 32 colunas (preenchidas com cor de papel)
        cov_clipped = np.ones((H, W), dtype=bool)
        cov_clipped[:, :32] = False
        filled = real_sheet.copy()
        filled[:, :32] = PAPER
        for _ in range(4):
            _drive_page(
                worker,
                generation,
                WarpedPage(_bgr(filled), generation, Q0, coverage=cov_clipped),
            )
        assert not np.any(worker._page_model.seeded[:, :32])
        assert np.all(worker._page_model.seeded[:, 32:])
        assert np.count_nonzero(_confirmed(worker, generation)) == 0

        # Cobertura expande e revela a marca pre-existente real: semeia como fundo
        cov_full = np.ones((H, W), dtype=bool)
        for _ in range(4):
            _drive_page(
                worker,
                generation,
                WarpedPage(_bgr(real_sheet), generation, Q0, coverage=cov_full),
            )
        assert np.all(worker._page_model.seeded)
        assert np.count_nonzero(_confirmed(worker, generation)) == 0

        # Traco novo na faixa recem-semeada confirma normalmente
        real_sheet[50:60, 10:20] = INK
        for _ in range(4):
            _drive_page(
                worker,
                generation,
                WarpedPage(_bgr(real_sheet), generation, Q0, coverage=cov_full),
            )
        confirmed = _confirmed(worker, generation)
        assert np.count_nonzero(confirmed) == 100
        assert np.all(confirmed[50:60, 10:20] > 0)
    finally:
        worker.stop()


def test_page_model_coverage_drop_preserves_confirmed_ink_reconfirmation(monkeypatch):
    """Issue 6: queda de coverage sobre tinta ja confirmada nao corrompe ref nem bloqueia voto."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        cov_full = np.ones((H, W), dtype=bool)
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive_page(
                worker,
                generation,
                WarpedPage(_bgr(image), generation, Q0, coverage=cov_full),
            )
        image[50:60, 50:60] = INK
        for _ in range(4):
            _drive_page(
                worker,
                generation,
                WarpedPage(_bgr(image), generation, Q0, coverage=cov_full),
            )
        assert np.count_nonzero(_confirmed(worker, generation)[50:60, 50:60]) == 100

        # Queda semantica de coverage sobre o traco confirmado
        cov_drop = np.ones((H, W), dtype=bool)
        cov_drop[45:65, 45:65] = False
        for _ in range(4):
            _drive_page(
                worker,
                generation,
                WarpedPage(_bgr(image), generation, Q0, coverage=cov_drop),
            )
        assert np.count_nonzero(_confirmed(worker, generation)[50:60, 50:60]) == 100
    finally:
        worker.stop()


def test_page_model_missed_hand_over_seeded_paper_is_blocked(monkeypatch):
    """Issue 7: mao nao mascarada sobre papel ja semeado nao vota nem puxa ref."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, generation, image)
        assert worker._page_model.seeded_fraction == pytest.approx(1.0)

        # Mao escura/quente 48x48 (visible=True) com vincos onde _ScriptInk da 0.95
        hand_bgr = _bgr(image)
        hand_bgr[24:72, 40:88] = (90, 110, 150)
        hand_bgr[42:52, 55:65] = (30, 35, 45)
        for _ in range(5):
            _drive_page(worker, generation, WarpedPage(hand_bgr, generation, Q0))
        assert np.count_nonzero(_confirmed(worker, generation)) == 0
        np.testing.assert_allclose(
            worker._page_model.reference[24:72, 40:88], PAPER, atol=1e-3
        )
    finally:
        worker.stop()


def test_page_model_missed_hand_preserves_existing_confirmed_ink(monkeypatch):
    """Issue 7: mao nao mascarada sobre traco confirmado nao o apaga nem corrompe ref."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True, removal=True))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, generation, image)
        image[44:52, 54:64] = INK
        for _ in range(4):
            _drive(worker, generation, image)
        assert np.count_nonzero(_confirmed(worker, generation)[44:52, 54:64]) == 80

        # Palma sem mascara (gray ~120 -> prob=0 no _ScriptInk) cobrindo o traco
        hand_bgr = _bgr(image)
        hand_bgr[24:72, 40:88] = (110, 120, 140)
        for _ in range(10):
            _drive_page(worker, generation, WarpedPage(hand_bgr, generation, Q0))
        assert np.count_nonzero(_confirmed(worker, generation)[44:52, 54:64]) == 80
    finally:
        worker.stop()


def test_page_model_occlusion_hold_blocks_recovery_edge_shadow(monkeypatch):
    """Issue 7 & R351-357: occlusion_hold_passes bloqueia todas as N passadas de recuperacao."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True, occlusion_hold_passes=2))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, generation, image)

        # 1 passada com mao nao mascarada -> arma hold=2
        hand_bgr = _bgr(image)
        hand_bgr[24:72, 40:88] = (80, 90, 110)
        _drive_page(worker, generation, WarpedPage(hand_bgr, generation, Q0))

        # Nas 2 passadas seguintes (dentro do hold=2), residuo escuro recebe ZERO votos.
        # Uma 3a passada entrega apenas o 1o voto elegivel (insuficiente para confirmar,
        # pois o voto forte exige 2 leituras).
        residue = image.copy()
        residue[42:52, 55:65] = INK
        for _ in range(3):
            _drive(worker, generation, residue)
        assert np.count_nonzero(_confirmed(worker, generation)) == 0

        # A 4a passada (2o voto elegivel apos o hold=2) confirma o traco
        _drive(worker, generation, residue)
        assert np.count_nonzero(_confirmed(worker, generation)[42:52, 55:65]) == 100
    finally:
        worker.stop()


def test_page_model_exposed_hold_counts_exact_recovery_passes():
    """R351-357 & page_model.py:417: hold conta apenas passadas de recuperacao VISIVEIS."""
    for hold_n in (1, 3):
        model = pm_mod.PageModel((H, W), confirm_obs=1)
        paper = _paper()
        model.observe(paper)
        occluded = paper.copy()
        occluded[24:72, 40:88] = 80.0
        exp0 = model.exposed(occluded, hold_passes=hold_n)
        assert not exp0[48, 64]
        assert model.recovering_px == 0

        # Quadros ocultos pela mascara visible=False nao consomem o hold
        hidden_vis = np.zeros((H, W), dtype=bool)
        for _ in range(hold_n + 2):
            model.exposed(paper, visible=hidden_vis, hold_passes=hold_n)
            assert model.recovering_px == 0
            assert not model.hold_mask[48, 64]

        for step in range(hold_n):
            exp_rec = model.exposed(paper, hold_passes=hold_n)
            assert not exp_rec[48, 64], (
                f"hold={hold_n} liberou cedo na passada {step + 1}"
            )
            assert model.recovering_px > 0
            assert model.hold_mask[48, 64]

        exp_done = model.exposed(paper, hold_passes=hold_n)
        assert exp_done[48, 64]
        assert model.recovering_px == 0
        assert not model.hold_mask[48, 64]


def test_page_model_does_not_seed_under_startup_shadow_rejected_by_paper_guard(
    monkeypatch,
):
    """R2285-2293: sombra local na abertura rejeitada por paper_ok nao semeia baseline."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True, occlusion_hold_passes=1))
    try:
        generation = worker.new_generation()
        real_sheet = np.full((H, W), PAPER, dtype=np.float32)
        real_sheet[40:50, 40:50] = INK  # marca impressa pre-existente

        # Sombra local [32:64, 32:64] em nivel 110 (paper_ok False contra papel 235),
        # mas sem mascara de mao (visible=None) e marca pre-existente a 40.
        shadowed = real_sheet.copy()
        shadowed[32:64, 32:64] = 110.0
        shadowed[40:50, 40:50] = INK
        for _ in range(4):
            _drive(worker, generation, shadowed)
        assert not np.any(worker._page_model.seeded[32:64, 32:64])
        assert np.count_nonzero(_confirmed(worker, generation)) == 0

        # Sombra sai: hold esgota e o bloco semeia COM a marca pre-existente como fundo
        for _ in range(5):
            _drive(worker, generation, real_sheet)
        assert np.all(worker._page_model.seeded[32:64, 32:64])
        assert np.count_nonzero(_confirmed(worker, generation)) == 0

        # Traco novo na mesma regiao confirma normalmente
        real_sheet[52:60, 40:50] = INK
        for _ in range(4):
            _drive(worker, generation, real_sheet)
        assert np.count_nonzero(_confirmed(worker, generation)[52:60, 40:50]) == 80
    finally:
        worker.stop()


def test_page_model_adapts_to_sustained_dimming_and_captures_new_stroke(monkeypatch):
    """R307-311 & R369-371: marca impressa semeada continua suprimida sob dimming e relighting."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        bright = np.full((H, W), PAPER, dtype=np.float32)
        # Marca impressa estreita em nivel 95 (abaixo de 100 -> _ScriptInk da 0.95)
        bright[14:22, 14:44] = 95.0
        for _ in range(3):
            _drive(worker, generation, bright)
        assert worker._page_model.seeded_fraction == pytest.approx(1.0)
        assert np.count_nonzero(_confirmed(worker, generation)) == 0

        # Iluminacao cai de 235 para 170: papel le 170 e marca impressa cai de 95 para 69
        # (razao de queda 26/95 = 0,27 < novel_ratio=0,30 — proporcionalmente inalterada),
        # junto de um traco novo a 40 em [50:60, 80:90].
        dimmed = np.full((H, W), 170.0, dtype=np.float32)
        dimmed[14:22, 14:44] = 69.0
        dimmed[50:60, 80:90] = INK
        dimmed_with_occluder = dimmed.copy()
        dimmed_with_occluder[0:36, 70:106] = 70.0
        exp = worker._page_model.exposed(
            dimmed_with_occluder, bgr=_bgr(dimmed_with_occluder)
        )
        assert exp[55, 85]
        assert not exp[10, 85]

        # Dimming sustentado por 10 passadas: marca impressa segue suprimida, traco novo confirma
        for _ in range(10):
            _drive(worker, generation, dimmed)
        confirmed = _confirmed(worker, generation)
        assert np.count_nonzero(confirmed[50:60, 80:90]) == 100
        assert not np.any(confirmed[14:22, 14:44] > 0)
        assert float(worker._page_model.reference[80, 100]) < PAPER - 15.0

        # Luz original retorna (235 papel, 95 marca impressa): marca impressa segue suprimida
        relit = bright.copy()
        relit[50:60, 80:90] = INK
        for _ in range(6):
            _drive(worker, generation, relit)
        confirmed_relit = _confirmed(worker, generation)
        assert np.count_nonzero(confirmed_relit[50:60, 80:90]) == 100
        assert not np.any(confirmed_relit[14:22, 14:44] > 0)
    finally:
        worker.stop()


def test_page_model_hand_exit_holds_until_uncovered_stroke_confirms_before_rest(
    monkeypatch,
):
    """R2392-2404: hold do PageModel estende a janela pos-saida ate confirmar traco."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.05)
    now = [1000.0]
    worker = worker_mod.PageCleanWorker(
        _ScriptInk(),
        config=InkConfig(page_model=True, occlusion_hold_passes=2),
        clock=lambda: now[0],
    )
    generation = worker.new_generation()
    # Folha com papel a 200 e patch [24:72, 40:88] mais claro a 250:
    # uma sombra local a 180 no patch cai abaixo de 0.78 * 250 = 195 (aciona hold
    # do PageModel), mas fica acima de 0.78 * 200 = 156 (paper_ok global = True).
    base = np.full((H, W), 200.0, dtype=np.float32)
    base[24:72, 40:88] = 250.0
    worker.set_hand_presence(True)
    for _ in range(3):
        worker.submit(WarpedPage(_bgr(base), generation, Q0))
        assert worker.pump()
        now[0] += 0.05

    shadow = base.copy()
    shadow[24:72, 40:88] = 180.0
    worker.submit(WarpedPage(_bgr(shadow), generation, Q0))
    assert worker.pump()
    now[0] += 0.05

    # Mao sai, sombra limpa e revela traco novo em [42:52, 55:65]:
    # as 2 passadas de hold do PageModel nao podem esgotar a confirmacao antes do descanso.
    uncovered = base.copy()
    uncovered[42:52, 55:65] = INK
    worker.set_hand_presence(False)
    for _ in range(12):
        worker.submit(WarpedPage(_bgr(uncovered), generation, Q0))
        worker.pump()
        now[0] += 0.06
        if worker.stats()["resting"]:
            break
    assert worker.stats()["resting"]
    snap = worker.poll(generation, 0)
    assert snap is not None
    assert np.count_nonzero(snap.confirmed[42:52, 55:65]) == 100

    # Controle: oclusao permanente (~raw_exposed) ainda atinge descanso finito
    worker.set_hand_presence(True, moved=True)
    worker.submit(WarpedPage(_bgr(shadow), generation, Q0))
    assert worker.pump()
    now[0] += 0.05
    worker.set_hand_presence(False)
    for _ in range(12):
        worker.submit(WarpedPage(_bgr(shadow), generation, Q0))
        worker.pump()
        now[0] += 0.06
        if worker.stats()["resting"]:
            break
    assert worker.stats()["resting"]


def test_page_model_coverage_drop_preserves_identity_baseline_on_reacquisition(
    monkeypatch,
):
    """R2316-2325: queda de coverage sobre baseline nao apaga _pm_baseline_ink."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        cov_full = np.ones((H, W), dtype=bool)
        image = np.full((H, W), PAPER, dtype=np.float32)
        # 900 px de baseline na margem esquerda
        image[10:20, 10:55] = INK
        image[24:34, 10:55] = INK
        for _ in range(3):
            _drive_page(
                worker,
                gen1,
                WarpedPage(_bgr(image), gen1, Q0, coverage=cov_full),
            )
        # 100 px de escrita confirmada a direita
        image[60:70, 80:90] = INK
        for _ in range(4):
            _drive_page(
                worker,
                gen1,
                WarpedPage(_bgr(image), gen1, Q0, coverage=cov_full),
            )
        assert np.count_nonzero(_confirmed(worker, gen1)) == 100

        # Coverage perde a margem esquerda antes da perda de tracking
        cov_right_only = np.ones((H, W), dtype=bool)
        cov_right_only[:, :60] = False
        filled = image.copy()
        filled[:, :60] = PAPER
        for _ in range(3):
            _drive_page(
                worker,
                gen1,
                WarpedPage(_bgr(filled), gen1, Q0, coverage=cov_right_only),
            )

        # Folha retorna com coverage cheia: baseline preservado garante same_page
        gen2 = worker.new_generation(preserve_identity=True)
        _drive_page(
            worker,
            gen2,
            WarpedPage(_bgr(image), gen2, Q0, coverage=cov_full),
        )
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert np.count_nonzero(_confirmed(worker, gen2)[60:70, 80:90]) == 100
    finally:
        worker.stop()


def test_page_model_work_does_not_block_submit(monkeypatch):
    """Issue 8: PageModel roda fora de _condition e nao bloqueia submit()."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    orig_observe = pm_mod.PageModel.observe
    calls = [0]

    def _slow_observe(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            entered.set()
            release.wait(timeout=5.0)
        return orig_observe(self, *args, **kwargs)

    monkeypatch.setattr(pm_mod.PageModel, "observe", _slow_observe)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        worker.submit(WarpedPage(_bgr(image), generation, Q0))
        assert entered.wait(timeout=5.0)

        t0 = time.monotonic()
        worker.submit(WarpedPage(_bgr(image), generation, Q0))
        elapsed = time.monotonic() - t0
        assert elapsed < 0.2
        assert not release.is_set()
        release.set()

        _drive(worker, generation, image)
        assert worker.error is None
    finally:
        release.set()
        worker.stop()


def test_page_model_generation_change_mid_pass_discards_safely(monkeypatch):
    """Issue 8: troca de geracao enquanto PageModel roda fora do lock descarta passada velha."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    orig_observe = pm_mod.PageModel.observe
    armed = [False]

    def _hooked_observe(self, *args, **kwargs):
        res = orig_observe(self, *args, **kwargs)
        if armed[0]:
            armed[0] = False
            entered.set()
            release.wait(timeout=5.0)
        return res

    monkeypatch.setattr(pm_mod.PageModel, "observe", _hooked_observe)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, gen1, image)

        stroked = image.copy()
        stroked[50:60, 50:60] = INK
        _drive(worker, gen1, stroked)
        armed[0] = True
        worker.submit(WarpedPage(_bgr(stroked), gen1, Q0))
        assert entered.wait(timeout=5.0)

        gen2 = worker.new_generation()
        release.set()
        for _ in range(3):
            _drive(worker, gen2, image)
        assert worker.error is None
        assert np.count_nonzero(_confirmed(worker, gen2)) == 0
    finally:
        release.set()
        worker.stop()


def test_page_model_counters_exclude_zero_and_hidden_pixels(monkeypatch):
    """Issue 9: pm_suppressed_px e pm_novel_px contam apenas votos no dominio visivel."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        image[10:20, 10:20] = INK  # 100 px baseline pre-existente
        for _ in range(3):
            _drive(worker, generation, image)

        # Confirma 50 px em [30:35, 10:20]
        image[30:35, 10:20] = INK
        for _ in range(4):
            _drive(worker, generation, image)
        assert np.count_nonzero(_confirmed(worker, generation)) == 50

        # Nova passada com:
        # - 100 px baseline visivel [10:20, 10:20] (suprimido)
        # - 50 px ja confirmado [30:35, 10:20]
        # - 80 px novo visivel [50:58, 10:20] (novel)
        # - 200 px escuro mas OCULTO [70:80, 10:30] (visible=False)
        image[50:58, 10:20] = INK
        image[70:80, 10:30] = INK
        vis = np.ones((H, W), dtype=bool)
        vis[70:80, 10:30] = False
        _drive(worker, generation, image, visible=vis)
        record = worker.telemetry_snapshot()["records"][-1]
        assert record["pm_suppressed_px"] == 100
        assert record["pm_novel_px"] == 80
        assert record["mask_px"] == 80 + 50
    finally:
        worker.stop()


def test_page_model_counters_under_non_identity_warp_border(monkeypatch):
    """Issue 9: borda borderValue=0 fora do warp registrado nao infla pm_novel_px."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True, anchor_follow=False))
    try:
        generation = worker.new_generation()
        image = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, generation, image)

        shifted_quad = Q0 + np.float32([[6, 4], [6, 4], [6, 4], [6, 4]])
        vis_cam = np.ones((H, W), dtype=bool)
        _drive_page(
            worker,
            generation,
            WarpedPage(_bgr(image), generation, shifted_quad, visible=vis_cam),
        )
        record = worker.telemetry_snapshot()["records"][-1]
        assert record["pm_novel_px"] == 0
        assert record["pm_suppressed_px"] == 0
    finally:
        worker.stop()


def test_page_model_reacquires_stroke_drawn_entirely_on_printed_rule(monkeypatch):
    """R457-461: toda a tinta salva sobre a pauta (ramo degenerado) — a
    mesma folha restaura pela evidencia de escrita que escurece a pauta,
    e a copia limpa da mesma pauta nao a herda."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        form = np.full((H, W), PAPER, dtype=np.float32)
        # Regra impressa de 6 px de altura
        form[40:46, 10:110] = 95.0
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Usuario traca um segmento de 40 px INTEIRO dentro da regra:
        # a tinta salva fica 100% sobre o baseline impresso.
        traced = form.copy()
        traced[42:44, 40:60] = INK
        for _ in range(6):
            _drive(worker, gen1, traced, visible=vis_all)
        confirmed_before = int(np.count_nonzero(_confirmed(worker, gen1)))
        assert confirmed_before >= 30

        # A mesma folha retorna: o traco escurece a regra e restaura.
        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, traced, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2))) == confirmed_before

        # Copia limpa com a MESMA regra (sem o traco): as posicoes salvas
        # mostram a pauta esperada (95), e o nivel salvo da tinta era 40
        # — a aparencia de pauta na posicao PROVA que a tinta nao esta
        # ali (um traco de 40 escureceria o lugar). Ausencia real, nao
        # ambiguidade: mismatch e new_page, sem herdar o desenho.
        gen3 = worker.new_generation(preserve_identity=True)
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive(worker, gen3, form, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
        assert np.count_nonzero(_confirmed(worker, gen3)) == 0
    finally:
        worker.stop()


def test_page_model_clean_copy_thicker_print_edge_does_not_inherit(monkeypatch):
    """R643-646: copia limpa cuja marca impressa engorda 1 px para cima
    da posicao deslocada do traco salvo nao restaura o desenho; a mesma
    folha com o traco de verdade deslocado restaura."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    vis_all = np.ones((H, W), dtype=bool)

    def _seeded_worker():
        w = _worker(InkConfig(page_model=True))
        gen1 = w.new_generation()
        form = np.full((H, W), PAPER, dtype=np.float32)
        # Pauta longa na linha 30 + marca impressa curta na linha 60
        form[30, 10:110] = 95.0
        form[60, 40:60] = 95.0
        for _ in range(3):
            _drive(w, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(w, gen1)) == 0
        # Traco manuscrito de 20 px logo abaixo da marca impressa
        with_dash = form.copy()
        with_dash[61, 45:65] = INK
        for _ in range(4):
            _drive(w, gen1, with_dash, visible=vis_all)
        assert np.count_nonzero(_confirmed(w, gen1)) == 20
        return w

    # Controle positivo: a MESMA folha deslocada 1 px — a pauta
    # mede o deslocamento e o traco deslocado escurece alem dela.
    worker = _seeded_worker()
    try:
        shifted_same = np.full((H, W), PAPER, dtype=np.float32)
        shifted_same[31, 10:110] = 95.0
        shifted_same[61, 40:60] = 95.0
        shifted_same[62, 45:65] = INK
        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, shifted_same, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        # O canvas restaurado volta na posicao SALVA (grade ancora):
        # o traco deslocado para 62 registra de volta sobre o 61.
        assert np.count_nonzero(_confirmed(worker, gen2)[61, 45:65]) == 20
    finally:
        worker.stop()

    # Copia limpa deslocada 1 px com a marca impressa APARECENDO
    # 2 px de altura — a linha extra cai exatamente sobre a
    # posicao deslocada do traco: posicao nao basta, a aparencia
    # e de pauta e nada e admitido.
    worker2 = _seeded_worker()
    try:
        clean_copy = np.full((H, W), PAPER, dtype=np.float32)
        clean_copy[31, 10:110] = 95.0
        clean_copy[61:63, 40:60] = 95.0
        gen2 = worker2.new_generation(preserve_identity=True)
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive(worker2, gen2, clean_copy, visible=vis_all)
        events = worker2.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
        assert np.count_nonzero(_confirmed(worker2, gen2)) == 0
    finally:
        worker2.stop()


def test_page_model_occluded_mismatch_in_discount_band_does_not_vote(
    monkeypatch,
):
    """R530-535: fatia clara com mismatch abaixo do piso SEM desconto
    adia em vez de votar — a mesma folha ocluida com tinta nova visivel
    sobrevive a 3 passadas cobertas e restaura ao descobrir."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[30:34, 10:100] = 95.0
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Traco de 220 px cruzando a pauta (40 px sobre ela, 180 fora)
        stroked = form.copy()
        stroked[28:50, 50:60] = INK
        for _ in range(6):
            _drive(worker, gen1, stroked, visible=vis_all)
        confirmed_before = int(np.count_nonzero(_confirmed(worker, gen1)))
        assert confirmed_before >= 200

        # Retorno com a base do traco coberta pela mao (20 px julgaveis
        # visiveis + 40 ambiguos sobre a pauta) e tinta NOVA de 30 px
        # exposta longe do traco — a fatia clara falha containment, mas
        # esta abaixo do piso de avaliacao inteiro: adia, nao vota.
        vis_hand = np.ones((H, W), dtype=bool)
        vis_hand[34:52, 40:80] = False
        added = stroked.copy()
        added[10:15, 10:16] = INK
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(3):
            _drive(worker, gen2, added, visible=vis_hand)
        events = worker.telemetry_snapshot()["page_identity_events"]
        decisions = [e["decision"] for e in events[-3:]]
        assert "new_page" not in decisions
        assert "quarantine_timeout" not in decisions
        assert worker._identity_reference is not None

        # Mao sai: o traco inteiro reaparece e a folha confirma —
        # o desenho salvo inteiro volta (a tinta nova confirma junto,
        # e tinta real na mesma folha).
        _drive(worker, gen2, added, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        restored = _confirmed(worker, gen2)
        assert np.count_nonzero(restored[28:50, 50:60]) == 220
    finally:
        worker.stop()


def test_page_model_first_baseline_pass_after_switch_uses_own_geometry(
    monkeypatch,
):
    """R3032: apos troca de folha a passada vota na geometria propria —
    a pegada da ancora velha nao pode podar a primeira observacao de
    baseline da folha nova."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    observed = []
    orig_observe = pm_mod.PageModel.observe

    def _spy(self, gray, **kwargs):
        observed.append(
            (
                id(self),
                None if kwargs.get("visible") is None else kwargs["visible"].copy(),
            )
        )
        return orig_observe(self, gray, **kwargs)

    monkeypatch.setattr(pm_mod.PageModel, "observe", _spy)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        sheet_a = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, gen1, sheet_a)
        stroked_a = sheet_a.copy()
        stroked_a[24:27, 20:90] = INK
        stroked_a[40:43, 30:80] = INK
        for _ in range(6):
            _drive(worker, gen1, stroked_a)
        assert np.count_nonzero(_confirmed(worker, gen1)) >= 100

        # Folha B em quad deslocado: a passada que decide new_page vota
        # na geometria propria — a pegada (ancora velha -> quad
        # deslocado) nao cobre a grade inteira e NAO pode entrar na
        # mascara de observacao do modelo novo.
        sheet_b = np.full((H, W), PAPER, dtype=np.float32)
        sheet_b[60:63, 10:80] = INK
        shifted_quad = Q0 + np.float32([[14, 0]] * 4)
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive_page(
                worker,
                gen2,
                WarpedPage(_bgr(sheet_b), gen2, shifted_quad, visible=None),
            )
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"

        pm = worker._page_model
        first_call = next(vis for pm_id, vis in observed if pm_id == id(pm))
        assert first_call is None or np.all(first_call)
    finally:
        worker.stop()


def test_page_model_pencil_stroke_beside_printed_rule_reacquires(monkeypatch):
    """R673-681: traco lapis (70) colado a regra impressa (95) e julgado
    contra o papel LOCAL embaixo dele (235), nao contra a pauta —
    a mesma folha restaura."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[30, 20:100] = 95.0
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Traco de lapis em 70 uma linha abaixo da regra impressa:
        # a comparacao antiga com a pauta (0.7 * 95 = 66.5) apagava
        # o traco; o local (papel 235 - 70 > delta) prova escrita.
        penciled = form.copy()
        penciled[31, 40:70] = 70.0
        for _ in range(6):
            _drive(worker, gen1, penciled, visible=vis_all)
        confirmed_before = int(np.count_nonzero(_confirmed(worker, gen1)))
        assert confirmed_before >= 20

        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, penciled, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert np.count_nonzero(_confirmed(worker, gen2)[31, 40:70]) == 30
    finally:
        worker.stop()


def test_page_model_single_traced_mark_clean_replacement_rejects(monkeypatch):
    """R665-681: marca impressa unica totalmente tracada — a copia
    limpa nao herda porque a pauta vista na posicao salva prova a
    ausencia do nivel de tinta guardado (40 != 95)."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        vis_all = np.ones((H, W), dtype=bool)
        form = np.full((H, W), PAPER, dtype=np.float32)
        # A UNICA marca impressa da folha: 80 px de regra cinza
        form[40:44, 40:60] = 95.0
        for _ in range(3):
            _drive(worker, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(worker, gen1)) == 0

        # Traco manuscrito cobrindo a marca INTEIRA (80 px): nao sobra
        # amostra de pauta limpa fora da posicao salva — o caso exato
        # em que a mediana global ficava sem referencia e a posicao
        # sozinha herdava o desenho na copia.
        traced = form.copy()
        traced[40:44, 40:60] = INK
        for _ in range(6):
            _drive(worker, gen1, traced, visible=vis_all)
        confirmed_before = int(np.count_nonzero(_confirmed(worker, gen1)))
        assert confirmed_before >= 60

        # Mesma folha: o traco escurece a marca (40 vs 95 esperado) -> same_page
        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, traced, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2))) == confirmed_before

        # Copia limpa com a MESMA marca (sem o traco): as posicoes
        # salvas mostram 95 = esperado; a tinta salva era 40 —
        # distinguivel -> ausencia real -> new_page.
        gen3 = worker.new_generation(preserve_identity=True)
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive(worker, gen3, form, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
        assert np.count_nonzero(_confirmed(worker, gen3)) == 0
    finally:
        worker.stop()


def test_page_model_two_print_tones_darker_mark_does_not_inherit(monkeypatch):
    """R665-681: formulario com regras claras (110) e marca escura (65):
    na copia deslocada, a marca de 65 na posicao salva prova ausencia
    da tinta de nivel 40 — o mediano global antigo a admitiria."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    vis_all = np.ones((H, W), dtype=bool)

    def _seeded_worker():
        w = _worker(InkConfig(page_model=True))
        gen1 = w.new_generation()
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[30, 10:110] = 110.0  # regra clara
        form[60, 40:60] = 65.0  # marca impressa escura
        for _ in range(3):
            _drive(w, gen1, form, visible=vis_all)
        assert np.count_nonzero(_confirmed(w, gen1)) == 0
        # Traco manuscrito (40) logo abaixo da marca escura
        marked = form.copy()
        marked[61, 40:60] = INK
        for _ in range(6):
            _drive(w, gen1, marked, visible=vis_all)
        assert np.count_nonzero(_confirmed(w, gen1)) == 20
        return w

    # Copia deslocada 1 px para baixo: a marca de 65 ocupa as linhas
    # 61-62 — a linha 61 e a posicao salva do traco. Escuro na posicao,
    # mas o nivel esperado (65) difere do nivel salvo (40) > delta:
    # ausencia provada -> new_page, nao heranca.
    worker = _seeded_worker()
    try:
        gen2 = worker.new_generation(preserve_identity=True)
        copy_shifted = np.full((H, W), PAPER, dtype=np.float32)
        copy_shifted[31, 10:110] = 110.0
        copy_shifted[61:63, 40:60] = 65.0
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive(worker, gen2, copy_shifted, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
        assert np.count_nonzero(_confirmed(worker, gen2)) == 0
    finally:
        worker.stop()

    # Mesma folha deslocada: o traco real aterrissa na linha 62 e
    # escurece a posicao deslocada (40 vs 65 esperado) -> same_page.
    worker = _seeded_worker()
    try:
        gen2 = worker.new_generation(preserve_identity=True)
        same_shifted = np.full((H, W), PAPER, dtype=np.float32)
        same_shifted[31, 10:110] = 110.0
        same_shifted[61, 40:60] = 65.0
        same_shifted[62, 40:60] = INK
        for _ in range(identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES):
            _drive(worker, gen2, same_shifted, visible=vis_all)
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert np.count_nonzero(_confirmed(worker, gen2)[62, 40:60]) == 20
    finally:
        worker.stop()


def _saved_sheet(worker, form, written):
    gen1 = worker.new_generation()
    vis_all = np.ones((H, W), dtype=bool)
    for _ in range(3):
        _drive(worker, gen1, form, visible=vis_all)
    assert np.count_nonzero(_confirmed(worker, gen1)) == 0
    for _ in range(6):
        _drive(worker, gen1, written, visible=vis_all)
    return gen1, int(np.count_nonzero(_confirmed(worker, gen1)))


def test_page_model_mixed_traced_and_freehand_same_sheet_reacquires(monkeypatch):
    """worker.py:635: traco sobre a pauta + escrita livre — a mesma folha
    restaura; o containment nao perde a parte tracada da referencia."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[30:34, 10:110] = 95.0
        written = form.copy()
        written[30:34, 20:100] = INK  # tracado: 320 px sobre a pauta
        written[60:64, 40:60] = INK  # livre: 80 px
        gen1, before = _saved_sheet(worker, form, written)
        assert before >= 380

        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, written, visible=np.ones((H, W), dtype=bool))
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2))) == before
    finally:
        worker.stop()


def test_page_model_mixed_form_clean_copy_rejects_without_hold(monkeypatch):
    """worker.py:877: copia limpa inteiramente visivel de um formulario
    com traco sobre a pauta vota new_page desde a primeira passada — o
    traco salvo sobre a pauta nao conta como escondido (sem hold)."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[30:34, 10:110] = 95.0
        written = form.copy()
        written[30:34, 20:100] = INK
        written[60:62, 40:50] = INK  # livre pequeno: 20 px
        _saved_sheet(worker, form, written)

        gen2 = worker.new_generation(preserve_identity=True)
        frames = identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES
        for _ in range(frames):
            _drive(worker, gen2, form, visible=np.ones((H, W), dtype=bool))
        events = worker.telemetry_snapshot()["page_identity_events"]
        decisions = [e["decision"] for e in events[-frames:]]
        assert "deferred_occlusion" not in decisions
        assert decisions[-1] == "new_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2))) == 0
    finally:
        worker.stop()


def test_page_model_light_pencil_beside_dark_print_reacquires(monkeypatch):
    """worker.py:556: lapis (70) colado a regra escura (40) nao escurece a
    pior aparencia vizinha, mas o tom salvo difere do papel embaixo
    dele — reaparecer prova presenca; a copia limpa prova ausencia."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[30, 20:100] = INK
        penciled = form.copy()
        penciled[31, 40:70] = 70.0
        _, before = _saved_sheet(worker, form, penciled)
        assert before >= 20

        gen2 = worker.new_generation(preserve_identity=True)
        _drive(worker, gen2, penciled, visible=np.ones((H, W), dtype=bool))
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2))) == before

        gen3 = worker.new_generation(preserve_identity=True)
        frames = identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES
        for _ in range(frames):
            _drive(worker, gen3, form, visible=np.ones((H, W), dtype=bool))
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
        assert int(np.count_nonzero(_confirmed(worker, gen3))) == 0
    finally:
        worker.stop()


def test_page_model_hand_over_stroke_keeps_saved_ink_tone(monkeypatch):
    """worker.py:1365: a mao cobrindo o traco na ultima passada antes da
    troca nao vira o tom salvo — a mesma folha restaura pelo tom real."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[30, 20:100] = INK
        penciled = form.copy()
        penciled[31, 40:70] = 70.0
        gen1, before = _saved_sheet(worker, form, penciled)
        assert before >= 20

        hand = penciled.copy()
        hand[25:45, 35:75] = 200.0
        vis_hand = np.ones((H, W), dtype=bool)
        vis_hand[25:45, 35:75] = False
        _drive(worker, gen1, hand, visible=vis_hand)
        assert int(np.count_nonzero(_confirmed(worker, gen1))) == before

        gen2 = worker.new_generation(preserve_identity=True)
        ref = worker._identity_gray_ref
        assert ref is not None
        assert np.all(np.abs(ref[31, 40:70] - 70.0) <= 2.0)
        _drive(worker, gen2, penciled, visible=np.ones((H, W), dtype=bool))
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2))) == before
    finally:
        worker.stop()


def _ambiguous_identity_fixture(shape=(H, W)):
    """Referencia salva inteira sobre a pauta, com tom salvo igual ao da
    pauta (80 sobre 95): nenhuma passada distingue tinta de pauta."""
    model = pm_mod.PageModel(shape)
    form = np.full(shape, PAPER, dtype=np.float32)
    form[30:34, 10:110] = 95.0
    vis = np.ones(shape, dtype=bool)
    for _ in range(4):
        model.observe(form, visible=vis, paper=vis)
    assert model.seeded_fraction == 1.0
    ref = np.zeros(shape, dtype=np.uint8)
    ref[30:34, 20:100] = 255
    baseline = np.zeros(shape, dtype=bool)
    baseline[30:34, 10:110] = True
    gray_ref = np.full(shape, np.nan, dtype=np.float32)
    gray_ref[30:34, 20:100] = 80.0
    return model, form, ref, baseline, gray_ref


def _identity_candidate(gray):
    return ((PAPER - gray) / PAPER).astype(np.float32)


def test_identity_ambiguous_saved_ink_with_unrelated_new_stroke_holds():
    """worker.py:778: tinta nova longe da tinta salva ambigua nao prova
    que ela sumiu — inconclusivo com hold, nao new_page."""
    model, form, ref, baseline, gray_ref = _ambiguous_identity_fixture()
    gray = form.copy()
    gray[30:34, 20:100] = 80.0
    gray[60:66, 40:60] = INK  # tinta nova sem relacao com a salva
    vis = np.ones((H, W), dtype=bool)
    res = worker_mod._evaluate_page_identity(
        ref,
        baseline,
        _identity_candidate(gray),
        vis,
        None,
        None,
        False,
        0.2,
        identity_gray=gray,
        identity_model=model,
        identity_gray_ref=gray_ref,
    )
    assert res[0] is False
    assert res[5] is True and res[6] == "occlusion"


def test_identity_coincidental_fragment_does_not_restore_ambiguous_canvas():
    """worker.py:772: dominio julgavel vazio por ambiguidade (tinta toda
    sobre a pauta) e um fragmento escuro coincidente no quadro deslocado:
    6 px nao restauram os 160 salvos numa folha com outro desenho."""
    model, form, ref, baseline, gray_ref = _ambiguous_identity_fixture()
    ref[:] = 0
    ref[31:33, 20:100] = 255
    gray = np.full((H, W), PAPER, dtype=np.float32)
    gray[31:35, 10:110] = 95.0  # mesma pauta, folha 1 px abaixo
    gray[33, 20:26] = INK  # fragmento no quadro deslocado
    gray[60:80, 30:90] = INK  # o desenho da folha substituta
    res = worker_mod._evaluate_page_identity(
        ref,
        baseline,
        _identity_candidate(gray),
        np.ones((H, W), dtype=bool),
        None,
        None,
        False,
        0.2,
        identity_gray=gray,
        identity_model=model,
        identity_gray_ref=gray_ref,
    )
    assert res[0] is False
    assert res[1] is not None and res[1] < identity_mod._PAGE_IDENTITY_MIN_RECALL
    # Tinta salva toda a vista: adiamento sem hold, queima a quarentena.
    assert res[5] is True and res[7] is False


def test_page_model_clean_copy_darker_print_does_not_inherit(monkeypatch):
    """page_model.py:744: copia limpa com a pauta um pouco mais escura (85
    contra 95) fica a menos de `novel_delta` da escrita salva (65), mas
    segue mais perto da pauta — nao herda o canvas salvo."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[30:34, 10:110] = 95.0
        written = form.copy()
        written[30:34, 20:100] = 65.0
        _, before = _saved_sheet(worker, form, written)
        assert before > 0

        darker_copy = form.copy()
        darker_copy[30:34, 10:110] = 85.0
        vis = np.ones((H, W), dtype=bool)
        frames = identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES
        bound = identity_mod._PAGE_IDENTITY_MAX_DEFERRALS + frames + 1
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(bound):
            _drive(worker, gen2, darker_copy, visible=vis)
        events = worker.telemetry_snapshot()["page_identity_events"]
        decisions = [e["decision"] for e in events]
        assert "same_page" not in decisions
        # Termina (nao fica adiada) e publica o canvas da folha nova.
        assert decisions[-1] == "new_page"
        assert worker.poll(gen2, 0) is not None
        assert int(np.count_nonzero(_confirmed(worker, gen2))) == 0
    finally:
        worker.stop()


def test_saved_tone_evidence_shifted_match_uses_saved_frame():
    """page_model.py:761: folha 1 px abaixo, lapis inalterado e uma regra
    do mesmo tom logo alem da vizinhanca valida: a regra (grade salva)
    nao pode empatar com o match exato do traco deslocado."""
    model = pm_mod.PageModel((H, W))
    form = np.full((H, W), PAPER, dtype=np.float32)
    form[36, 10:110] = 90.0
    vis = np.ones((H, W), dtype=bool)
    for _ in range(4):
        model.observe(form, visible=vis, paper=vis)
    assert model.seeded_fraction == 1.0
    levels = np.full((H, W), np.nan, dtype=np.float32)
    levels[31, 40:70] = 90.0
    at = np.zeros((H, W), dtype=bool)
    at[31, 40:70] = True
    gray = np.full((H, W), PAPER, dtype=np.float32)
    gray[37, 10:110] = 90.0  # a regra, deslocada com a folha
    gray[32, 40:70] = 90.0  # o lapis, deslocado com a folha
    distinct, matched = model.saved_tone_evidence(
        gray, levels, at, dy=1, dx=0, near=3, visible=vis
    )
    assert distinct[31, 40:70].all()
    assert matched[32, 40:70].all()


def test_saved_tone_evidence_unshifted_match_checks_aligned_print():
    """page_model.py:766: regra um pouco mais escura (85 contra 95) da
    copia limpa cai sobre o traco salvo (65) com a folha a -3 linhas: o
    candidato no lugar salvo tambem consulta a pauta no quadro medido."""
    model = pm_mod.PageModel((H, W))
    form = np.full((H, W), PAPER, dtype=np.float32)
    form[34:36, 10:110] = 95.0
    vis = np.ones((H, W), dtype=bool)
    for _ in range(4):
        model.observe(form, visible=vis, paper=vis)
    levels = np.full((H, W), np.nan, dtype=np.float32)
    levels[30, 40:70] = 65.0
    at = np.zeros((H, W), dtype=bool)
    at[30, 40:70] = True
    gray = np.full((H, W), PAPER, dtype=np.float32)
    gray[30:32, 10:110] = 85.0  # a regra da copia limpa, 4 linhas acima
    distinct, matched = model.saved_tone_evidence(
        gray, levels, at, dy=-3, dx=0, near=3, visible=vis
    )
    assert distinct[30, 40:70].all()
    assert not matched.any()


def test_saved_tone_evidence_unshifted_match_ignores_stationary_rule():
    """page_model.py:769: o deslocamento global (-1) projeta a regra
    vizinha (95) para perto do traco clareado (65 -> 85), mas a regra
    ficou no lugar: sem evidencia local de que a pauta andou, o quadro
    medido nao veta o traco que tambem ficou."""
    model = pm_mod.PageModel((H, W))
    form = np.full((H, W), PAPER, dtype=np.float32)
    form[34, 10:110] = 95.0
    vis = np.ones((H, W), dtype=bool)
    for _ in range(4):
        model.observe(form, visible=vis, paper=vis)
    levels = np.full((H, W), np.nan, dtype=np.float32)
    levels[30, 25:105] = 65.0
    at = np.zeros((H, W), dtype=bool)
    at[30, 25:105] = True
    gray = form.copy()
    gray[30, 25:105] = 85.0
    distinct, matched = model.saved_tone_evidence(
        gray, levels, at, dy=-1, dx=0, near=3, visible=vis
    )
    assert distinct[30, 25:105].all()
    assert matched[30, 25:105].all()


def test_saved_tone_evidence_wide_shifted_band_keeps_clean_veto():
    """page_model.py:814: faixa impressa larga (95, linhas 34-47) de uma
    copia limpa 4 linhas acima (85, 30-43) com alinhamento medindo -3: a
    referencia vizinha ve 85 nas duas amostras — nao prova que ficou, a
    explicacao limpa (85 ~ 95) segue vetando o traco salvo (65)."""
    model = pm_mod.PageModel((H, W))
    form = np.full((H, W), PAPER, dtype=np.float32)
    form[34:48, 10:110] = 95.0
    vis = np.ones((H, W), dtype=bool)
    for _ in range(4):
        model.observe(form, visible=vis, paper=vis)
    levels = np.full((H, W), np.nan, dtype=np.float32)
    levels[30, 25:105] = 65.0
    at = np.zeros((H, W), dtype=bool)
    at[30, 25:105] = True
    gray = np.full((H, W), PAPER, dtype=np.float32)
    gray[30:44, 10:110] = 85.0
    distinct, matched = model.saved_tone_evidence(
        gray, levels, at, dy=-3, dx=0, near=3, visible=vis
    )
    assert distinct[30, 25:105].all()
    assert not matched.any()


def test_page_model_wide_shifted_band_clean_copy_rejects(monkeypatch):
    """page_model.py:814: copia limpa com faixa larga 4 linhas acima
    cobrindo o traco salvo nao herda o canvas pela recuperacao."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        form = np.full((H, W), PAPER, dtype=np.float32)
        form[34:48, 10:110] = 95.0
        form[66:68, 10:110] = 95.0
        form[82:84, 10:110] = 95.0
        written = form.copy()
        written[29:31, 25:105] = 65.0
        _, before = _saved_sheet(worker, form, written)
        assert before >= 120

        copy = np.full((H, W), PAPER, dtype=np.float32)
        copy[30:44, 10:110] = 85.0
        copy[62:64, 10:110] = 85.0
        copy[78:80, 10:110] = 85.0
        vis = np.ones((H, W), dtype=bool)
        frames = identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES
        bound = identity_mod._PAGE_IDENTITY_MAX_DEFERRALS + frames + 1
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(bound):
            _drive(worker, gen2, copy, visible=vis)
        decisions = [
            e["decision"] for e in worker.telemetry_snapshot()["page_identity_events"]
        ]
        assert "same_page" not in decisions
        assert decisions[-1] == "new_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2))) == 0
    finally:
        worker.stop()


def test_saved_tone_evidence_hidden_projected_rule_stays_ambiguous():
    """page_model.py:818: regra parada (95, linhas 34-35), alinhamento
    medindo -2 e a mao sobre a posicao projetada (32-33): o veto depende
    de amostra escondida — o traco (65 -> 85) nao e distinto nem casado."""
    model = pm_mod.PageModel((H, W))
    form = np.full((H, W), PAPER, dtype=np.float32)
    form[34:36, 10:110] = 95.0
    vis = np.ones((H, W), dtype=bool)
    for _ in range(4):
        model.observe(form, visible=vis, paper=vis)
    levels = np.full((H, W), np.nan, dtype=np.float32)
    levels[29:31, 25:105] = 65.0
    at = np.zeros((H, W), dtype=bool)
    at[29:31, 25:105] = True
    gray = form.copy()
    gray[29:31, 25:105] = 85.0
    hand = vis.copy()
    hand[32:34, :] = False
    distinct, matched, unsure, occluded = model.saved_tone_evidence(
        gray, levels, at, dy=-2, dx=0, near=3, visible=hand, with_unsure=True
    )
    assert not distinct[29:31, 25:105].any()
    assert not matched.any()
    assert unsure[29:31, 25:105].all()
    assert occluded[29:31, 25:105].all()
    # Mesmo veto, mas a amostra esta fora da pegada do registro (pose
    # recortada, sem mao): incerto, nao ocluido.
    clip = vis.copy()
    clip[32:, :] = False
    distinct, matched, unsure, occluded = model.saved_tone_evidence(
        gray,
        levels,
        at,
        dy=-2,
        dx=0,
        near=3,
        visible=clip,
        with_unsure=True,
        sampled=clip,
    )
    assert not distinct[29:31, 25:105].any()
    assert unsure[29:31, 25:105].all()
    assert not occluded.any()
    # A mao sai: a regra provadamente ficou e o traco reaparece.
    distinct, matched = model.saved_tone_evidence(
        gray, levels, at, dy=-2, dx=0, near=3, visible=vis
    )
    assert distinct[29:31, 25:105].all()
    assert matched[29:31, 25:105].all()


def test_page_model_hidden_projected_rule_does_not_discard_canvas(monkeypatch):
    """page_model.py:818: a mao cobre a posicao projetada da regra vizinha
    durante todo o consenso de troca: a mesma folha nao vota new_page e
    restaura quando a mao sai."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        form = _ruled_form()
        written = form.copy()
        written[29:31, 25:105] = 65.0
        _, before = _saved_sheet(worker, form, written)
        assert before >= 120

        returned = np.full((H, W), PAPER, dtype=np.float32)
        returned[34:36, 10:110] = 95.0
        returned[64:66, 10:110] = 95.0
        returned[80:82, 10:110] = 95.0
        returned[29:31, 25:105] = 85.0
        vis = np.ones((H, W), dtype=bool)
        hand = vis.copy()
        hand[32:34, :] = False
        # Mais passadas cobertas que o prazo de adiamentos: o traco
        # bloqueado pela amostra escondida segura o hold.
        passes = (
            max(
                identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES,
                identity_mod._PAGE_IDENTITY_MAX_DEFERRALS,
            )
            + 2
        )
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(passes):
            _drive(worker, gen2, returned, visible=hand)
        decisions = [
            e["decision"] for e in worker.telemetry_snapshot()["page_identity_events"]
        ]
        assert "new_page" not in decisions
        assert "quarantine_timeout" not in decisions
        for _ in range(2):
            _drive(worker, gen2, returned, visible=vis)
        decisions = [
            e["decision"] for e in worker.telemetry_snapshot()["page_identity_events"]
        ]
        assert "new_page" not in decisions
        assert decisions[-1] == "same_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2)[29:31, 25:105])) == 160
    finally:
        worker.stop()


def test_page_model_hidden_projected_rule_fragment_does_not_restore(monkeypatch):
    """page_model.py:807: folha substituta com so a marca pequena (40 px)
    da salva e banda impressa onde estava o traco longo (160 px), com a
    mao na posicao projetada da regra vizinha: o traco conta como
    escondido, nao some do denominador — 40 de 200 nao restaura."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        form = _ruled_form()
        written = form.copy()
        written[29:31, 25:105] = 65.0
        written[50:52, 40:60] = INK
        _, before = _saved_sheet(worker, form, written)
        assert before >= 160

        replacement = np.full((H, W), PAPER, dtype=np.float32)
        replacement[34:36, 10:110] = 95.0
        replacement[64:66, 10:110] = 95.0
        replacement[80:82, 10:110] = 95.0
        replacement[29:31, 25:105] = 85.0
        replacement[50:52, 40:60] = INK
        hand = np.ones((H, W), dtype=bool)
        hand[32:34, :] = False
        frames = identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(frames + 1):
            _drive(worker, gen2, replacement, visible=hand)
        decisions = [
            e["decision"] for e in worker.telemetry_snapshot()["page_identity_events"]
        ]
        assert decisions and "same_page" not in decisions
    finally:
        worker.stop()


def _clipped_rule_scene(freehand=False):
    """Folha pautada (regras 4-5 e 34-35) de volta 2 linhas acima: a
    regra 2-3 visivel estabelece o deslocamento, a projetada 32-33
    decide o traco salvo 29-30 (65 salvo, 85 visto). `freehand` soma
    um traco livre de 300 px na margem direita (linhas 8-27, colunas
    112-126), fora das regras. `run(visible, sampled)` zera a
    prob e poe papel (255) fora de `sampled`, como o registro."""
    model = pm_mod.PageModel((H, W))
    form = np.full((H, W), PAPER, dtype=np.float32)
    form[4:6, 10:110] = 95.0
    form[34:36, 10:110] = 95.0
    vis = np.ones((H, W), dtype=bool)
    for _ in range(4):
        model.observe(form, visible=vis, paper=vis)
    baseline = form < 200
    ref = np.zeros((H, W), dtype=np.uint8)
    ref[29:31, 25:105] = 255
    gray_ref = np.full((H, W), np.nan, dtype=np.float32)
    gray_ref[29:31, 25:105] = 65.0
    gray = np.full((H, W), PAPER, dtype=np.float32)
    gray[2:4, 10:110] = 95.0
    gray[32:34, 10:110] = 95.0
    gray[29:31, 25:105] = 85.0
    if freehand:
        ref[8:28, 112:127] = 255
        gray_ref[8:28, 112:127] = INK
        gray[8:28, 112:127] = INK

    def run(visible, sampled, hold_seen=False):
        seen = gray.copy()
        seen[~sampled] = 255.0
        candidate = _identity_candidate(seen)
        candidate[~sampled] = 0.0
        return worker_mod._evaluate_page_identity(
            ref,
            baseline,
            candidate,
            visible,
            sampled,
            None,
            hold_seen,
            0.2,
            identity_gray=seen,
            identity_model=model,
            identity_gray_ref=gray_ref,
        )

    return run, vis


def test_identity_hidden_projected_rule_holds_only_under_the_hand():
    """worker.py:804: o mesmo traco salvo e a mesma regra projetada,
    escondida uma vez pela mao (hold) e outra pelo recorte do registro
    (adia sem hold, sujeito ao timeout — nem herdando hold anterior)."""
    run, vis = _clipped_rule_scene()
    hand = vis.copy()
    hand[32:34, :] = False
    covered = run(hand, vis)
    assert covered[5] is True and covered[6] == "occlusion"
    assert covered[7] is True
    clip = vis.copy()
    clip[32:, :] = False
    for hold_seen in (False, True):
        clipped = run(clip, clip, hold_seen)
        assert clipped[0] is False and clipped[5] is True
        assert clipped[7] is False


def test_identity_clipped_rule_hand_on_shifted_stroke_keeps_hold():
    """worker.py:830: recorte fixo; a posicao salva (29-30) visivel e
    inconclusiva, a mao cobre so a posicao deslocada (27-28) dentro da
    pegada: o hold anterior persiste; sem a mao, o recorte sozinho nao
    segura. O hold vem de um quadro anterior com a mao sobre a regra
    projetada ainda dentro da pegada."""
    run, vis = _clipped_rule_scene()
    on_rule = vis.copy()
    on_rule[32:34, :] = False
    first = run(on_rule, vis)
    assert first[6] == "occlusion" and first[7] is True
    clip = vis.copy()
    clip[32:, :] = False
    hand = clip.copy()
    hand[27:29, :] = False
    covered = run(hand, clip, first[7])
    assert covered[5] is True and covered[6] == "occlusion"
    assert covered[7] is True
    lifted = run(clip, clip, covered[7])
    assert lifted[5] is True and lifted[7] is False


def test_identity_clipped_rule_and_hand_covered_stroke_hold():
    """worker.py:812/823: o recorte bloqueia o traco da borda (160 px) e
    a mao cobre o traco livre (300 px) dentro da pegada: descobri-lo
    basta para o piso, entao adia com hold; a mao sai e a folha e a
    mesma. Se a folha anda ao tirar a mao e o traco livre sai da
    pegada, sobra so o veto geometrico: o hold anterior nao persiste."""
    run, vis = _clipped_rule_scene(freehand=True)
    clip = vis.copy()
    clip[32:, :] = False
    hand = clip.copy()
    hand[:, 111:] = False
    covered = run(hand, clip)
    assert covered[0] is False and covered[5] is True
    assert covered[6] == "occlusion" and covered[7] is True
    lifted = run(clip, clip, covered[7])
    assert lifted[0] is True
    # A folha anda ao tirar a mao: a borda direita da pegada (um lado
    # do quadrilatero registrado) passa por dentro da margem.
    moved_out = clip.copy()
    moved_out[:, 111:] = False
    stalled = run(moved_out, moved_out, covered[7])
    assert stalled[0] is False and stalled[5] is True
    assert stalled[6] == "occlusion" and stalled[7] is False


def _ruled_form():
    form = np.full((H, W), PAPER, dtype=np.float32)
    form[34:36, 10:110] = 95.0
    form[66:68, 10:110] = 95.0
    form[82:84, 10:110] = 95.0
    return form


def test_page_model_stationary_rule_keeps_stroke_under_global_shift(monkeypatch):
    """page_model.py:769: as regras distantes sobem 1 px e dirigem o
    alinhamento; a regra vizinha e o traco clareado ficam no lugar —
    a mesma folha restaura, nao vota new_page."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        form = _ruled_form()
        written = form.copy()
        written[29:31, 25:105] = 65.0
        _, before = _saved_sheet(worker, form, written)
        assert before >= 120

        returned = np.full((H, W), PAPER, dtype=np.float32)
        returned[34:36, 10:110] = 95.0
        returned[65:67, 10:110] = 95.0
        returned[81:83, 10:110] = 95.0
        returned[29:31, 25:105] = 85.0
        vis = np.ones((H, W), dtype=bool)
        frames = identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(frames + 1):
            _drive(worker, gen2, returned, visible=vis)
        decisions = [
            e["decision"] for e in worker.telemetry_snapshot()["page_identity_events"]
        ]
        assert "new_page" not in decisions
        assert decisions[-1] == "same_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2)[29:31, 25:105])) == 160
    finally:
        worker.stop()


def test_page_model_shifted_clean_copy_rule_over_stroke_rejects(monkeypatch):
    """test_page_model.py:2371: copia limpa um pouco mais escura (85)
    volta 4 px acima; o alinhamento mede o deslocamento sozinho e a
    regra que cai sobre o traco salvo (65) nao herda o canvas."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    worker = _worker(InkConfig(page_model=True))
    try:
        form = _ruled_form()
        written = form.copy()
        written[30:32, 25:105] = 65.0
        _, before = _saved_sheet(worker, form, written)
        assert before >= 120

        copy = np.full((H, W), PAPER, dtype=np.float32)
        copy[30:32, 10:110] = 85.0
        copy[62:64, 10:110] = 85.0
        copy[78:80, 10:110] = 85.0
        vis = np.ones((H, W), dtype=bool)
        frames = identity_mod._PAGE_IDENTITY_SWITCH_CONSENSUS_FRAMES
        bound = identity_mod._PAGE_IDENTITY_MAX_DEFERRALS + frames + 1
        gen2 = worker.new_generation(preserve_identity=True)
        for _ in range(bound):
            _drive(worker, gen2, copy, visible=vis)
        decisions = [
            e["decision"] for e in worker.telemetry_snapshot()["page_identity_events"]
        ]
        assert "same_page" not in decisions
        assert decisions[-1] == "new_page"
        assert int(np.count_nonzero(_confirmed(worker, gen2))) == 0
    finally:
        worker.stop()


def test_align_identity_baseline_removes_prior_halo_across_band_edge():
    """worker.py:367: tinta salva logo FORA da banda de busca ainda projeta
    o halo dilatado dela para dentro — nao pode puxar o ajuste."""
    shape = (80, 120)
    kernel = np.ones((identity_mod._IDENTITY_DILATION_KERNEL,) * 2, np.uint8)
    radius = identity_mod._IDENTITY_DILATION_KERNEL // 2
    baseline = np.zeros(shape, dtype=bool)
    baseline[40, 20:100] = True
    prior = np.zeros(shape, dtype=bool)
    prior[36, 20:100] = True  # 1 px fora da banda (40 - radius = 37)
    seen = np.zeros(shape, dtype=bool)
    seen[40, 20:40] = True  # pauta vista no lugar, parcialmente
    seen[38, 20:100] = True  # pele do traco salvo, dentro do halo
    _, dy, dx = worker_mod._align_identity_baseline(
        baseline, seen, prior, radius, kernel
    )
    assert (dy, dx) == (0, 0)


@pytest.mark.slow
def test_identity_recovery_photometric_budget_on_a4_grid():
    """test_page_model.py:613: o caminho completo (cinza, modelo semeado,
    tom salvo, alinhamento e comparacao local) cabe no orcamento A4."""
    shape = (1200, 850)
    form = np.full(shape, PAPER, dtype=np.float32)
    form[10:15, 10:200] = 95.0
    form[200:205, 100:700] = 95.0
    form[1180:1185, 650:840] = 95.0
    vis = np.ones(shape, dtype=bool)
    model = pm_mod.PageModel(shape)
    for _ in range(4):
        model.observe(form, visible=vis, paper=vis)
    baseline = form < 200
    written = form.copy()
    written[200:205, 300:400] = INK  # tracado sobre a pauta
    written[400:420, 300:350] = INK  # livre
    written[1181:1185, 700:760] = 75.0  # lapis sobre a pauta
    ref = np.zeros(shape, dtype=np.uint8)
    ref[written < form - 10] = 255
    gray_ref = np.full(shape, np.nan, dtype=np.float32)
    gray_ref[ref > 0] = written[ref > 0]
    gray = np.roll(written, (1, 1), axis=(0, 1))
    candidate = _identity_candidate(gray)

    def run():
        return worker_mod._evaluate_page_identity(
            ref,
            baseline,
            candidate,
            vis,
            None,
            None,
            False,
            0.2,
            identity_gray=gray,
            identity_model=model,
            identity_gray_ref=gray_ref,
        )

    run()
    start = time.monotonic()
    for _ in range(20):
        res = run()
    per_pass_ms = (time.monotonic() - start) / 20 * 1000
    assert res[0] is True
    # ~19 ms medidos contra ~9-15 ms do caminho reduzido na mesma
    # maquina (antes desta revisao: 40-60 ms); o bound pega regressao
    # de ordem de grandeza, nao ruido sob carga.
    assert per_pass_ms < 30.0, per_pass_ms


def test_novel_and_returned_stay_disjoint_over_dark_print():
    """Review #174: pauta escura (ref ~24), traco que a escurece dentro
    de _RETURN_TOL (|diff| <= 8) mas acima da razao de Weber era lido
    como tinta nova E como papel devolvido — e a trabalhadora apagava
    o traco visivel pelo caminho de remocao."""
    model = pm_mod.PageModel((H, W), confirm_obs=1)
    dark = _paper()
    dark[40:44, 20:80] = 24.0  # pauta escura semeada
    model.observe(dark)

    traced = dark.copy()
    traced[40:44, 30:60] = 16.0  # traco: 33% de contraste, diff=8
    novel_mask, returned_mask = model.appearance(traced)
    assert novel_mask[42, 45]
    assert not returned_mask[42, 45]
    assert not model.returned(traced)[42, 45]

    # Restaurar a pauta de verdade continua contando como devolvida.
    assert model.returned(dark)[42, 45]


def test_novel_discriminates_absolute_delta_at_both_ends():
    """Caso de contraste local (sem dimming global): onde as duas
    regras divergem. ref 180 -> 150 cai 30 absoluto (delta vazaria)
    mas so 17% (razao suprime); ref 60 -> 40 cai so 20 absoluto
    (delta suprimiria) mas 33% (razao detecta)."""
    model = pm_mod.PageModel((H, W), confirm_obs=1)
    scene = _paper()
    scene[40:46, 20:60] = 180.0  # papel acinzentado
    scene[60:66, 20:60] = 60.0  # pauta escura
    model.observe(scene)

    obs = scene.copy()
    obs[40:46, 30:50] = 150.0  # sombra local suave sobre o cinza
    obs[60:66, 30:50] = 40.0  # traco sobre a pauta escura
    novel_mask, _ = model.appearance(obs)
    assert not novel_mask[43, 40], "sombra 17% virou tinta nova"
    assert novel_mask[63, 40], "traco 33% sobre pauta escura nao foi visto"


def test_identity_darker_admits_what_novel_admits():
    """Review #174: darker_than_reference precisa enxergar o traco que
    novel() admitiu — senao a reaquisicao segura a recuperacao para
    sempre com o desenho salvo a vista."""
    model = pm_mod.PageModel((H, W), confirm_obs=1)
    dark = _paper()
    dark[40:44, 20:80] = 60.0
    model.observe(dark)

    traced = dark.copy()
    traced[40:44, 30:60] = 40.0  # 33% > 30%; gap absoluto 20 < 24
    assert model.novel(traced)[42, 45]
    darker, _ = model.darker_than_reference(traced)
    assert darker[42, 45]
