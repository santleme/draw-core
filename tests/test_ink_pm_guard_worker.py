"""Contratos do PageModel pelo worker real (issue #129): oclusor sem mascara.

O PageModel substituiu a guarda fotometrica independente: `pm.exposed`
so libera papel exposto para votar e apagar — objeto escuro, sombra ou
objeto com a luminancia do papel mas o cromatismo errado congelam a
regiao, com ou sem a mascara do MediaPipe. Os tracos dos cenarios sao
faixas finas desenhadas NA IMAGEM (8 px a cada 16 px): como um traco
de caneta real, mais fino que a janela de 32 px da guarda, cada janela
continua vendo papel no pico e o traco e "novel" contra a referencia.
"""

import time

import numpy as np

from draw.contracts import WarpedPage
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig

H, W = 256, 384
Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)
INK_TONE = (40, 42, 44)
HAND_TONE = (60, 62, 64)


def _stripe_mask(rows, cols, period=16, width=8):
    """Faixas finas dentro de um retangulo: tinta "novel" para o modelo
    sem nunca preencher uma janela inteira da guarda fotometrica."""
    mask = np.zeros((H, W), bool)
    for r0 in range(rows.start, rows.stop - width + 1, period):
        mask[r0 : r0 + width, cols] = True
    return mask


WITNESS = _stripe_mask(slice(16, 112), slice(16, 160))  # ~6,9k px: passa o piso
COVERED = _stripe_mask(slice(144, 208), slice(224, 336))
HAND = (slice(128, 224), slice(208, 352))  # oclusor solido sobre COVERED


class _Scripted:
    threshold = 0.7

    def __init__(self):
        self.content = np.zeros((H, W), np.float32)

    def ink_prob(self, _warped):
        return self.content


def _paper():
    image = np.empty((H, W, 3), np.uint8)
    image[:] = (200, 205, 210)
    return image


def _stroked():
    """Folha com os dois tracos desenhados de verdade na imagem."""
    image = _paper()
    image[WITNESS] = INK_TONE
    image[COVERED] = INK_TONE
    return image


def _prob(*, covered_ink=True, extra=None):
    prob = np.zeros((H, W), np.float32)
    prob[WITNESS] = 0.95
    if covered_ink:
        prob[COVERED] = 0.95
    if extra is not None:
        prob[extra] = 0.95
    return prob


def _pass(worker, generation, image, visible=None):
    before = worker.stats()["passes_total"]
    if visible is None:
        visible = np.ones((H, W), bool)
    worker.submit(WarpedPage(image, generation, Q0, visible))
    deadline = time.monotonic() + 5.0
    while worker.stats()["passes_total"] <= before:
        assert time.monotonic() < deadline, "a passada nao executou"
        time.sleep(0.001)


def _run(monkeypatch, config):
    """Semeia a referencia na folha limpa e depois confirma os tracos."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _Scripted()
    worker = worker_mod.PageCleanWorker(ink, config=config).start()
    generation = worker.new_generation()
    for _ in range(4):  # semeia: a referencia e a folha limpa
        _pass(worker, generation, _paper())
    ink.content = _prob()
    for _ in range(6):  # os tracos aparecem e confirmam
        _pass(worker, generation, _stroked())
    confirmed = worker.poll(generation, 0).confirmed
    assert confirmed[COVERED].all()
    assert confirmed[WITNESS].all()
    return worker, ink, generation


def _no_idle_probe(monkeypatch):
    """Sem sonda ociosa: so as passadas submetidas votam (sem corrida)."""
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 3600.0)
    time.sleep(0.1)


def _pass_until(worker, generation, image, done, limit):
    """Submete ate done(snapshot) ou o limite: parar a tempo, porque a
    trabalhadora descansa de verdade quando nao ha mais o que votar."""
    for _ in range(limit):
        if done(worker.poll(generation, 0)):
            return
        _pass(worker, generation, image)


def _hand_image():
    image = _stroked()
    image[HAND] = HAND_TONE  # mao/sombra escura que o MediaPipe nao viu
    return image


def _occluder_image(tone):
    image = _stroked()
    image[HAND] = tone
    return image


def test_dark_occluder_high_prob_neither_confirms_nor_erases(monkeypatch):
    """Oclusor escuro sem mascara com o modelo lendo tinta ALTA sob ele:
    nenhum pixel novo confirma dentro do objeto e o traco coberto nao
    sai — nem quando a probabilidade cai depois."""
    worker, ink, generation = _run(monkeypatch, InkConfig(page_model=True))
    try:
        before = worker.poll(generation, 0).confirmed.copy()
        ink.content = _prob()  # o modelo "ve" tinta ate sob o objeto
        for _ in range(12):
            _pass(worker, generation, _hand_image())
        snap = worker.poll(generation, 0)
        assert not np.any(snap.confirmed[HAND] & ~before[HAND])
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[WITNESS].all()
        # Nem quando a probabilidade do traco coberto cai a zero.
        ink.content = _prob(covered_ink=False)
        for _ in range(8):
            _pass(worker, generation, _hand_image())
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
        # O objeto sai: a regiao destrava e tinta nova confirma.
        new = _stripe_mask(slice(224, 240), slice(16, 120))
        ink.content = _prob(extra=new)
        image = _stroked()
        image[new] = INK_TONE
        _pass_until(
            worker,
            generation,
            image,
            lambda snap: snap.confirmed[new].all(),
            16,
        )
        assert worker.poll(generation, 0).confirmed[new].all()
    finally:
        worker.stop()


def test_control_without_page_model_erases_under_the_same_occluder(monkeypatch):
    """Controle: sem PageModel nada congela — a mesma sombra apaga."""
    worker, ink, generation = _run(monkeypatch, InkConfig())
    try:
        ink.content = _prob(covered_ink=False)
        for _ in range(14):
            _pass(worker, generation, _hand_image())
        assert not worker.poll(generation, 0).confirmed[COVERED].any()
    finally:
        worker.stop()


def test_paper_luma_wrong_chroma_occluder_neither_confirms_nor_erases(
    monkeypatch,
):
    """Oclusor com a luminancia do papel mas B/R fora do cromatismo da
    referencia (a barragem cromatica do #129): luma engana, croma nao."""
    worker, ink, generation = _run(monkeypatch, InkConfig(page_model=True))
    try:
        before = worker.poll(generation, 0).confirmed.copy()
        chroma_hand = _occluder_image((140, 205, 215))  # luma de papel, B/R baixo
        ink.content = _prob()
        for _ in range(12):
            _pass(worker, generation, chroma_hand)
        snap = worker.poll(generation, 0)
        assert not np.any(snap.confirmed[HAND] & ~before[HAND])
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[WITNESS].all()
        ink.content = _prob(covered_ink=False)
        for _ in range(8):
            _pass(worker, generation, chroma_hand)
        assert worker.poll(generation, 0).confirmed[COVERED].all()
    finally:
        worker.stop()


def test_startup_occluder_seeds_then_recovers_after_leaving(monkeypatch):
    """Oclusor presente desde a abertura e semeado como parte da pagina:
    nao cria tinta enquanto cobre e, ao sair, a regiao libera e confirma
    traco novo (o hold de oclusao finito, nao congelamento eterno)."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    config = InkConfig(page_model=True)
    ink = _Scripted()
    worker = worker_mod.PageCleanWorker(ink, config=config).start()
    generation = worker.new_generation()
    occl = (slice(128, 224), slice(16, 176))
    occluded = _paper()
    occluded[occl] = HAND_TONE
    try:
        for _ in range(7):  # semeia com o objeto em cima da folha
            _pass(worker, generation, occluded)
        assert worker._page_model._seeded[occl].all()
        assert not worker.poll(generation, 0).confirmed.any()
        # Saiu: o papel reaparece e o hold de oclusao libera a regiao.
        for _ in range(config.occlusion_hold_passes + 2):
            _pass(worker, generation, _paper())
        # Traco novo na regiao recuperada confirma.
        mask = _stripe_mask(slice(140, 188), slice(32, 120))
        ink.content = np.zeros((H, W), np.float32)
        ink.content[mask] = 0.95
        image = _paper()
        image[mask] = INK_TONE
        _pass_until(
            worker,
            generation,
            image,
            lambda snap: snap.confirmed[mask].all(),
            14,
        )
        assert worker.poll(generation, 0).confirmed[mask].all()
    finally:
        worker.stop()


def test_paper_really_erased_still_removes_with_page_model(monkeypatch):
    """O modelo nao segura tinta que sumiu de verdade: papel exposto
    limpo remove o traco confirmado."""
    worker, ink, generation = _run(monkeypatch, InkConfig(page_model=True))
    try:
        ink.content = _prob(covered_ink=False)
        for _ in range(14):
            _pass(worker, generation, _paper())
        snap = worker.poll(generation, 0)
        assert not snap.confirmed[COVERED].any()
        assert snap.confirmed[WITNESS].all()
    finally:
        worker.stop()


def test_black_frame_after_a_reference_freezes_instead_of_erasing(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig(page_model=True))
    try:
        ink.content = _prob(covered_ink=False)
        black = np.zeros((H, W, 3), np.uint8)
        for _ in range(14):
            _pass(worker, generation, black)
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
    finally:
        worker.stop()


def test_post_exit_confirm_window_is_rearmed_while_a_shadow_recovers(monkeypatch):
    worker, ink, generation = _run(
        monkeypatch,
        InkConfig(post_exit_full_window=True, page_model=True),
    )
    try:
        _no_idle_probe(monkeypatch)
        worker.set_hand_presence(True)
        for _ in range(2):
            _pass(worker, generation, _hand_image())
        worker.set_hand_presence(False)
        window = worker._post_exit_confirm_passes()
        assert window > 1
        _pass(worker, generation, _stroked())  # sombra saiu: papel em recuperacao
        assert worker._confirm_passes == window
        assert worker._confirm_masked_skips >= 1
    finally:
        worker.stop()


def test_permanent_border_does_not_rearm_the_window(monkeypatch):
    worker, ink, generation = _run(
        monkeypatch,
        InkConfig(post_exit_full_window=True, page_model=True),
    )
    try:
        worker.set_hand_presence(True)
        worker.set_hand_presence(False)
        image = _stroked()
        image[:, :64] = HAND_TONE  # borda escura permanente, sem tinta nela
        for _ in range(2):
            _pass(worker, generation, image)
        assert worker._confirm_masked_skips == 0
    finally:
        worker.stop()


def _mature(monkeypatch, config, *, idle_probe=False):
    worker, ink, generation = _run(monkeypatch, config)
    for _ in range(14):  # pagina madura e parada: o descanso esta ao alcance
        _pass(worker, generation, _stroked())
    if not idle_probe:
        _no_idle_probe(monkeypatch)
    return worker, ink, generation


def test_stroke_hidden_on_the_first_post_exit_pass_still_confirms(monkeypatch):
    """Sombra ainda cobre o traco na 1a passada pos-saida (modelo ve 0,0) e
    some na seguinte: a janela cheia de votos precisa sobrar apos o hold."""
    config = InkConfig.production()
    worker, ink, generation = _mature(monkeypatch, config)
    try:
        worker.set_hand_presence(True)
        late = (slice(224, 240), slice(16, 120))
        shadow = _stroked()
        shadow[(slice(208, 256), slice(0, 160))] = HAND_TONE
        for _ in range(3):
            _pass(worker, generation, shadow)
        worker.set_hand_presence(False)
        ink.content = _prob()  # o modelo ainda nao ve o traco sob a sombra
        _pass(worker, generation, shadow)
        ink.content = _prob(extra=late)
        revealed = _stroked()
        revealed[late] = INK_TONE
        _pass_until(
            worker,
            generation,
            revealed,
            lambda snap: snap.confirmed[late].all(),
            config.occlusion_hold_passes + 8,
        )
        assert worker.poll(generation, 0).confirmed[late].all()
    finally:
        worker.stop()


def test_stroke_erased_under_a_shadow_is_removed_after_hand_exit(monkeypatch):
    """Apagou um traco confirmado sob a sombra da mao: ao levantar a mao, o
    hold barra a remocao, mas o descanso nao pode chegar antes dela."""
    config = InkConfig.production()
    worker, ink, generation = _mature(monkeypatch, config, idle_probe=True)
    try:
        worker.set_hand_presence(True)
        ink.content = _prob(covered_ink=False)
        for _ in range(3):
            _pass(worker, generation, _hand_image())
        worker.set_hand_presence(False)
        # Sombra ainda persiste em parte da janela pos-saida antes de limpar.
        for _ in range(2):
            _pass(worker, generation, _hand_image())
        _pass_until(
            worker,
            generation,
            _stroked(),
            lambda snap: not snap.confirmed[COVERED].any(),
            config.occlusion_hold_passes + 12,
        )
        snap = worker.poll(generation, 0)
        assert not snap.confirmed[COVERED].any()
        assert snap.confirmed[WITNESS].all()
        _pass_until(
            worker,
            generation,
            _stroked(),
            lambda _snap: worker.stats()["resting"],
            6,
        )
        assert worker.stats()["resting"] is True
    finally:
        worker.stop()


def test_lingering_shadow_across_all_post_exit_passes_still_confirms_and_border_rests(
    monkeypatch,
):
    """Sombra nao mascarada persiste por todas as passadas pos-saida
    enquanto o modelo detecta o traco: sob o oclusor o modelo se abstem
    (nada elegivel — o descanso e permitido), mas na liberacao o
    recovering re-arma a janela cheia e o traco confirma; uma borda
    escura permanente ainda deixa o worker descansar."""
    config = InkConfig.production()
    worker, ink, generation = _mature(monkeypatch, config, idle_probe=True)
    try:
        late = (slice(224, 240), slice(16, 120))
        border = _stroked()
        border[:, 336:] = HAND_TONE  # borda escura permanente
        shadow = border.copy()
        shadow[(slice(208, 256), slice(0, 160))] = HAND_TONE

        worker.set_hand_presence(True)
        ink.content = _prob(extra=late)
        for _ in range(2):
            _pass(worker, generation, shadow)

        worker.set_hand_presence(False)
        for _ in range(worker._post_exit_confirm_passes()):
            _pass(worker, generation, shadow)

        assert not worker.poll(generation, 0).confirmed[late].any()

        # A sombra saiu junto com um evento real (a mao passa de novo):
        # acordado, o recovering re-arma a janela e o traco confirma.
        worker.set_hand_presence(True)
        worker.set_hand_presence(False)
        revealed = border.copy()
        revealed[late] = INK_TONE
        _pass_until(
            worker,
            generation,
            revealed,
            lambda snap: snap.confirmed[late].all(),
            config.occlusion_hold_passes + worker._post_exit_confirm_passes() + 8,
        )
        assert worker.poll(generation, 0).confirmed[late].all()

        _pass_until(
            worker,
            generation,
            border,
            lambda _snap: worker.stats()["resting"],
            6,
        )
        assert worker.stats()["resting"] is True
    finally:
        worker.stop()
