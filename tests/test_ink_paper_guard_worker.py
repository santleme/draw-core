"""Guarda fotometrica pelo worker real (issue #129): mao sem mascara."""

import threading
import time

import numpy as np

from draw.contracts import WarpedPage
from draw.ink import occlusion, registration, votes
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig

H, W = 256, 384
Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)
WITNESS = (slice(16, 112), slice(16, 160))  # ~13,8k px: passa o piso
COVERED = (slice(144, 208), slice(224, 336))
HAND = (slice(128, 224), slice(208, 352))


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
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _Scripted()
    worker = worker_mod.PageCleanWorker(ink, config=config).start()
    generation = worker.new_generation()
    ink.content = _prob()
    for _ in range(5):
        _pass(worker, generation, _paper())
    confirmed = worker.poll(generation, 0).confirmed
    assert confirmed[COVERED].all()
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
    image = _paper()
    image[HAND] = (60, 62, 64)  # mao/sombra escura que o MediaPipe nao viu
    return image


def test_unmasked_dark_hand_never_erases_confirmed_stroke(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig())
    try:
        ink.content = _prob(covered_ink=False)
        for _ in range(14):
            _pass(worker, generation, _hand_image())
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[WITNESS].all()
        # Mao sai, o traco continua la e tinta nova confirma.
        new = (slice(224, 240), slice(16, 120))
        ink.content = _prob(extra=new)
        for _ in range(12):
            _pass(worker, generation, _paper())
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[new].all()
    finally:
        worker.stop()


def test_control_without_guard_erases_under_the_same_hand(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig(paper_guard=False))
    try:
        ink.content = _prob(covered_ink=False)
        for _ in range(14):
            _pass(worker, generation, _hand_image())
        assert not worker.poll(generation, 0).confirmed[COVERED].any()
    finally:
        worker.stop()


def test_legacy_mode_holds_an_armed_streak_when_the_witness_dies(monkeypatch):
    """Regressao (Devin Review do PR #144): o modo legado (paper_guard
    desligado) entregava paper_exposed=True ao acumulador — lido como
    "avaliacao fotometrica aprovada" — e um streak de remocao armado sob
    testemunha viva drenava sozinho quando o modelo perdia o resto da
    tinta (sombra cobrindo tudo), apagando traco confirmado sem nenhuma
    prova de papel. Sem avaliacao o opt-out entrega None e o voto
    pendurado segura (contrato legado), nao completa."""
    worker, ink, generation = _run(monkeypatch, InkConfig(paper_guard=False))
    try:
        # Arma o streak: COVERED some da leitura e a testemunha segue —
        # no modo legado isso ja basta para contar ausencias.
        ink.content = _prob(covered_ink=False)
        for _ in range(3):
            _pass(worker, generation, _paper())
        streak = worker._accumulator._removal_streak
        armed = int(streak[COVERED].max())
        assert armed > 0
        # A sombra espalha e o modelo perde o desenho inteiro: sem
        # testemunha, o streak armado congela em vez de drenar — com
        # paper_exposed=True ele completava a remocao no escuro. O pin
        # no valor armado pega dreno parcial que um teto frouxo deixaria
        # passar.
        ink.content = np.zeros((H, W), np.float32)
        for _ in range(registration._REMOVAL_PASSES + 4):
            _pass(worker, generation, _paper())
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[WITNESS].all()
        assert int(worker._accumulator._removal_streak[COVERED].max()) == armed
        # A sombra sai com a tinta intacta: a leitura volta e o streak
        # pendurado zera — nada fica armado esperando a proxima falta
        # (o canvas intacto sozinho nao prova isso: um streak parado
        # em 1-5 retomaria a contagem na proxima falta).
        ink.content = _prob()
        for _ in range(3):
            _pass(worker, generation, _paper())
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[WITNESS].all()
        assert not worker._accumulator._removal_streak.any()
    finally:
        worker.stop()


def test_paper_really_erased_still_removes_with_guard(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig())
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
    worker, ink, generation = _run(monkeypatch, InkConfig())
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
    worker, ink, generation = _run(monkeypatch, InkConfig(post_exit_full_window=True))
    try:
        _no_idle_probe(monkeypatch)
        worker.set_hand_presence(True)
        for _ in range(2):
            _pass(worker, generation, _hand_image())
        worker.set_hand_presence(False)
        window = worker._post_exit_confirm_passes()
        assert window > 1
        _pass(worker, generation, _paper())  # sombra saiu: papel em recuperacao
        assert worker._confirm_passes == window
        assert worker._confirm_masked_skips >= 1
    finally:
        worker.stop()


def test_permanent_border_does_not_rearm_the_window(monkeypatch):
    worker, ink, generation = _run(monkeypatch, InkConfig(post_exit_full_window=True))
    try:
        worker.set_hand_presence(True)
        worker.set_hand_presence(False)
        image = _paper()
        image[:, 336:] = (60, 62, 64)  # borda escura permanente, sem tinta nela
        for _ in range(2):
            _pass(worker, generation, image)
        assert worker._confirm_masked_skips == 0
    finally:
        worker.stop()


def _mature(monkeypatch, config, *, idle_probe=False):
    worker, ink, generation = _run(monkeypatch, config)
    for _ in range(14):  # pagina madura e parada: o descanso esta ao alcance
        _pass(worker, generation, _paper())
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
        shadow = _paper()
        shadow[(slice(208, 256), slice(0, 160))] = (60, 62, 64)
        for _ in range(3):
            _pass(worker, generation, shadow)
        worker.set_hand_presence(False)
        ink.content = _prob()  # o modelo ainda nao ve o traco sob a sombra
        _pass(worker, generation, shadow)
        ink.content = _prob(extra=late)
        _pass_until(
            worker,
            generation,
            _paper(),
            lambda snap: snap.confirmed[late].all(),
            occlusion._FILL_RECOVERY_PASSES + 8,
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
            _paper(),
            lambda snap: not snap.confirmed[COVERED].any(),
            occlusion._FILL_RECOVERY_PASSES + 12,
        )
        snap = worker.poll(generation, 0)
        assert not snap.confirmed[COVERED].any()
        assert snap.confirmed[WITNESS].all()
        _pass_until(
            worker,
            generation,
            _paper(),
            lambda _snap: worker.stats()["resting"],
            6,
        )
        assert worker.stats()["resting"] is True
    finally:
        worker.stop()


def test_lingering_shadow_across_all_post_exit_passes_still_confirms_and_border_rests(
    monkeypatch,
):
    """Sombra nao mascarada persiste por todas as 3 passadas pos-saida
    enquanto o modelo detecta o traco: o orcamento congelado mantem o worker
    acordado ate limpar e confirmar, e uma borda escura permanente ainda deixa
    o worker descansar."""
    config = InkConfig.production()
    worker, ink, generation = _mature(monkeypatch, config, idle_probe=True)
    try:
        border_paper = _paper()
        border_paper[:, 336:] = (60, 62, 64)
        late = (slice(224, 240), slice(16, 120))
        shadow = border_paper.copy()
        # >= 3 linhas de tile (~24px): abaixo disso a guarda trata a sombra
        # como traco fino e nao veta adicoes.
        shadow[(slice(176, 256), slice(0, 160))] = (60, 62, 64)

        worker.set_hand_presence(True)
        for _ in range(2):
            _pass(worker, generation, shadow)

        worker.set_hand_presence(False)
        # O modelo so ve o traco nas passadas pos-saida: 3 leituras sob a
        # sombra ainda nao chegam a _FILL_RECOVERY_PASSES, entao o bloco
        # nao e adotado e o traco continua sem confirmar.
        ink.content = _prob(extra=late)
        for _ in range(worker._post_exit_confirm_passes()):
            _pass(worker, generation, shadow)

        assert not worker.poll(generation, 0).confirmed[late].any()
        assert not worker.stats()["resting"]

        _pass_until(
            worker,
            generation,
            border_paper,
            lambda snap: snap.confirmed[late].all(),
            occlusion._FILL_RECOVERY_PASSES + 8,
        )
        assert worker.poll(generation, 0).confirmed[late].all()

        _pass_until(
            worker,
            generation,
            border_paper,
            lambda _snap: worker.stats()["resting"],
            6,
        )
        assert worker.stats()["resting"] is True
    finally:
        worker.stop()


def test_stationary_hand_samples_do_not_cancel_the_adopted_fill_vote(
    monkeypatch,
):
    """Regressao (Devin Review do PR #144): saida e volta da mao, que
    fica parada segurando a folha. O laco entrega uma amostra de
    presenca por quadro — nao so nas transicoes — e cada amostra True
    repetida zerava o _confirm_passes rearmado pela adocao do
    preenchimento: a trabalhadora descansava no meio do voto e o bloco
    adotado nunca confirmava. So a transicao que encerra o episodio de
    mao fora pode cancelar a janela."""
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 3600.0)
    monkeypatch.setattr(worker_mod, "_HAND_REST_STILL_SECONDS", 0.3)
    now = [1000.0]
    ink = _Scripted()
    config = InkConfig.production(pass_interval_s=0.1)
    worker = worker_mod.PageCleanWorker(ink, config=config, clock=lambda: now[0])
    generation = worker.new_generation()
    visible = np.ones((H, W), bool)

    def _tick(image):
        """Um quadro do laco: submit + pump na cadencia configurada."""
        worker.submit(WarpedPage(image, generation, Q0, visible, captured_at=now[0]))
        now[0] += 0.11
        return worker.pump()

    # Pagina madura sem sinal de mao: so a testemunha confirma — o
    # COVERED nasce depois, sob o bloco escuro.
    ink.content = _prob(covered_ink=False)
    for _ in range(6):
        assert _tick(_paper())
    assert worker.poll(generation, 0).confirmed[WITNESS].all()

    # A sequencia completa de saida: a mao entra, sai (armando a
    # janela pos-saida) e volta ja parada — a transicao de volta
    # cancela a janela pos-saida, como antes.
    worker.set_hand_presence(True)
    assert _tick(_paper())
    worker.set_hand_presence(False)
    assert _tick(_paper())
    worker.set_hand_presence(True)

    # Bloco escuro lido como tinta (a sombra que o MediaPipe perdeu):
    # _FILL_RECOVERY_PASSES de suporte adotam o preenchimento e a
    # borda de adocao rearma a janela de confirmacao — com a mao
    # presente o tempo todo, repetida a cada tick como no laco.
    ink.content = _prob()
    for _ in range(occlusion._FILL_RECOVERY_PASSES):
        worker.set_hand_presence(True)
        assert _tick(_hand_image())
        worker.set_hand_presence(True)
    assert worker._paper_exposure._fill_adopted.any()
    assert not worker.poll(generation, 0).confirmed[COVERED].all()

    # A amostra repetida com a mao parada nao pode zerar o orcamento
    # rearmado pela adocao.
    worker.set_hand_presence(True)
    assert worker._confirm_passes == votes._INK_VOTE_WINDOW

    # As passadas seguintes votam o preenchimento adotado ate ele
    # confirmar; o descanso com a mao quieta so pode entrar depois —
    # pump() devolver False antes disso e o bug dormindo no voto.
    confirmed = False
    limit = occlusion._FILL_RECOVERY_PASSES + votes._INK_VOTE_WINDOW + 2
    for _ in range(limit):
        worker.set_hand_presence(True)
        ran = _tick(_hand_image())
        worker.set_hand_presence(True)
        confirmed = worker.poll(generation, 0).confirmed[COVERED].all()
        if confirmed:
            break
        assert ran, "descansou antes do preenchimento adotado confirmar"
    assert confirmed, "o preenchimento adotado nao confirmou"

    # Confirmado, o descanso com a mao parada segue valendo: a janela
    # gasta nao segura a trabalhadora acordada para sempre.
    for _ in range(worker_mod._REST_QUIET_PASSES + 2):
        _tick(_hand_image())
    assert worker.stats()["resting"]


def test_fill_streak_born_as_another_dies_still_reaches_adoption(monkeypatch):
    """Regressão (Devin Review do PR #144): a sombra gasta 3
    passadas de streak pós-saída e morre; o preenchimento que só
    então fica à vista herdava a espera dos dois — a porta
    fechava na 4a passada do novo e a trabalhadora dormia antes
    da leitura que adota o bloco. A carência por idade deixa o
    streak tardio completar o horizonte dele."""
    config = InkConfig.production(pass_interval_s=0.02)
    worker, ink, generation = _mature(monkeypatch, config, idle_probe=True)
    try:
        # Blocos 3x4+ tiles, longe da testemunha e do traco base.
        fill = (slice(144, 216), slice(24, 120))  # tiles 6-8 x 1-4
        shadow = (slice(24, 96), slice(240, 336))  # tiles 1-3 x 10-13
        scene = _paper()
        scene[fill] = (60, 62, 64)
        scene[shadow] = (60, 62, 64)
        masked = np.ones((H, W), bool)
        masked[fill] = False  # a mão ainda cobre o preenchimento

        worker.set_hand_presence(True)
        for _ in range(2):
            _pass(worker, generation, scene, masked)
        worker.set_hand_presence(False)

        # Pós-saída 1-3: a sombra lê como tinta enquanto a máscara
        # segue sobre o preenchimento — os três skips se vão sem
        # que o streak dele comece a contar.
        prob = _prob()
        prob[shadow] = 0.95
        prob[fill] = 0.95
        ink.content = prob
        for _ in range(3):
            _pass(worker, generation, scene, masked)

        # Passada 4: a sombra perde o suporte e o preenchimento fica
        # limpo — o streak dele nasce com a espera já gasta.
        prob = _prob()
        prob[fill] = 0.95
        ink.content = prob
        for _ in range(3):
            _pass(worker, generation, scene)

        # Sob o teto global antigo a porta já teria fechado: sem a
        # carência a trabalhadora descansa aqui e o preenchimento
        # nunca chega à 4a leitura.
        assert not worker.stats()["resting"]
        assert not worker.poll(generation, 0).confirmed[fill].any()

        _pass_until(
            worker,
            generation,
            scene,
            lambda snap: snap.confirmed[fill].all(),
            occlusion._FILL_RECOVERY_PASSES + 8,
        )
        assert worker.poll(generation, 0).confirmed[fill].all()

        # Adotado e confirmado, o pendente some e o descanso volta a
        # fechar normalmente.
        _pass_until(
            worker,
            generation,
            scene,
            lambda _snap: worker.stats()["resting"],
            6,
        )
        assert worker.stats()["resting"] is True
    finally:
        worker.stop()


def test_alternating_fill_support_still_hits_the_wait_bound(monkeypatch):
    """Vaivém sem adoção continua limitado: o suporte alternando
    entre dois blocos mantém o pendente aceso sem borda de adoção;
    a espera tem que fechar e a trabalhadora descansar."""
    config = InkConfig.production(pass_interval_s=0.02)
    worker, ink, generation = _mature(monkeypatch, config, idle_probe=True)
    try:
        shadow_a = (slice(24, 96), slice(240, 336))  # tiles 1-3 x 10-13
        shadow_b = (slice(168, 240), slice(144, 216))  # tiles 7-9 x 6-8
        scene = _paper()
        scene[shadow_a] = (60, 62, 64)
        scene[shadow_b] = (60, 62, 64)
        prob_a = _prob()
        prob_a[shadow_a] = 0.95
        prob_b = _prob()
        prob_b[shadow_b] = 0.95

        worker.set_hand_presence(True)
        _pass(worker, generation, scene)
        worker.set_hand_presence(False)
        for index in range(occlusion._RECOVERY_WAIT_PASSES * 3):
            if worker.stats()["resting"]:
                break
            ink.content = prob_a if index % 2 == 0 else prob_b
            _pass(worker, generation, scene)
        # O suporte nunca sustenta um streak até a adoção, mas a
        # espera estendida pela carência tem teto: descansa.
        assert worker.stats()["resting"] is True
        assert worker._fill_recovery_waited >= occlusion._RECOVERY_WAIT_PASSES
        snap = worker.poll(generation, 0)
        assert not snap.confirmed[shadow_a].any()
        assert not snap.confirmed[shadow_b].any()
    finally:
        worker.stop()


def test_mid_pass_generation_swap_writes_to_the_orphan_guard(monkeypatch):
    """new_generation durante a inferência troca a guarda; o observe
    da passada velha escreve na instância órfã — estado de página
    velha não semeia referência na página nova."""
    worker, ink, generation = _run(monkeypatch, InkConfig())
    observed = threading.Event()
    try:
        old_guard = worker._paper_exposure
        real_observe = old_guard.observe

        def poisoned(*args, **kwargs):
            try:
                return real_observe(*args, **kwargs)
            finally:
                worker.new_generation()
                observed.set()

        old_guard.observe = poisoned
        worker.submit(WarpedPage(_paper(), generation, Q0, np.ones((H, W), bool)))
        assert observed.wait(5.0)
        assert worker._paper_exposure is not old_guard
        assert worker._paper_exposure._reference is None
    finally:
        worker.stop()
