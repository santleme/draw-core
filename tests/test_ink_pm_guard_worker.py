"""Contratos do PageModel pelo worker real (issue #129): oclusor sem mascara.

O PageModel substituiu a guarda fotometrica independente: `pm.exposed`
so libera papel exposto para votar e apagar — objeto escuro, sombra ou
objeto com a luminancia do papel mas o cromatismo errado congelam a
regiao, com ou sem a mascara do MediaPipe. Os tracos dos cenarios sao
faixas finas desenhadas NA IMAGEM (8 px a cada 16 px): como um traco
de caneta real, mais fino que a janela de 32 px da guarda, cada janela
continua vendo papel no pico e o traco e "novel" contra a referencia.
"""

import dataclasses
import threading
import time

import numpy as np

from draw.contracts import WarpedPage
from draw.ink import occlusion, registration, votes
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


def _run(monkeypatch, config, *, image=_stroked):
    """Semeia a referencia na folha limpa e depois confirma os tracos.

    Sem o PageModel os tracos podem existir so na leitura do modelo
    (`image=_paper`): como no PR #144, a guarda de tiles nao veta o
    que nao e anomalo — e a pagina madura nao carrega tiles em
    recuperacao que queimariam a espera da fase medida."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _Scripted()
    worker = worker_mod.PageCleanWorker(ink, config=config).start()
    generation = worker.new_generation()
    for _ in range(4):  # semeia: a referencia e a folha limpa
        _pass(worker, generation, _paper())
    ink.content = _prob()
    for _ in range(6):  # os tracos aparecem e confirmam
        _pass(worker, generation, image())
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


def test_control_without_page_model_holds_under_the_same_occluder(monkeypatch):
    """Controle: sem PageModel a guarda de tiles (permanente) segura a
    remocao sob a mesma sombra — o congelamento fotometrico nao depende
    do modelo — e a regiao destrava quando o objeto sai."""
    worker, ink, generation = _run(monkeypatch, InkConfig())
    try:
        ink.content = _prob(covered_ink=False)
        for _ in range(14):
            _pass(worker, generation, _hand_image())
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[WITNESS].all()
        # O objeto sai: a regiao libera e tinta nova confirma —
        # congelamento, nunca aniquilacao.
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
        for _ in range(7):  # o objeto cobre a folha desde a abertura
            _pass(worker, generation, occluded)
        # A regiao coberta nao semeia: a guarda de tiles a veta antes
        # do PageModel observar — ela espera como covered_unseeded em
        # vez de virar referencia (o "fantasma" de inicializacao).
        assert not worker._page_model._seeded[occl].any()
        assert not worker.poll(generation, 0).confirmed.any()
        # Saiu: o papel reaparece e o hold de oclusao libera a regiao
        # para o modelo semear.
        for _ in range(config.occlusion_hold_passes + 2):
            _pass(worker, generation, _paper())
        assert worker._page_model._seeded[occl].all()
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


def test_armed_streak_holds_while_the_guard_proves_no_paper(monkeypatch):
    """O caso do review do #144 sem o modo legado: com a guarda sempre
    ligada, um streak de remocao armado sob testemunha viva nao drena
    quando a cena escurece por inteiro — nada prova papel exposto, a
    ausencia nao e observavel e a contagem pendurada segura. Sem a
    prova fotometrica o opt-out antigo lia "avaliacao aprovada" e
    completava a remocao no escuro."""
    worker, ink, generation = _run(monkeypatch, InkConfig())
    try:
        # Arma o streak: COVERED some da leitura e a testemunha segue.
        ink.content = _prob(covered_ink=False)
        for _ in range(3):
            _pass(worker, generation, _paper())
        streak = worker._accumulator._removal_streak
        armed = int(streak[COVERED].max())
        assert armed > 0
        # Escuro total: a guarda nao prova papel, o streak armado
        # congela em vez de drenar. O pin no valor armado pega dreno
        # parcial que um teto frouxo deixaria passar.
        ink.content = np.zeros((H, W), np.float32)
        dark = np.zeros((H, W, 3), np.uint8)
        for _ in range(registration._REMOVAL_PASSES + 4):
            _pass(worker, generation, dark)
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[WITNESS].all()
        assert int(worker._accumulator._removal_streak[COVERED].max()) == armed
        # A cena volta limpa com a leitura de tinta intacta: papel
        # provado em todo pixel e nada ausente — o streak pendurado
        # zera, nada fica armado esperando a proxima falta (o canvas
        # intacto sozinho nao prova isso: um streak parado em 1-5
        # retomaria a contagem na proxima falta).
        ink.content = _prob()
        for _ in range(3):
            _pass(worker, generation, _paper())
        snap = worker.poll(generation, 0)
        assert snap.confirmed[COVERED].all()
        assert snap.confirmed[WITNESS].all()
        assert not worker._accumulator._removal_streak.any()
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


def _mature(monkeypatch, config, *, idle_probe=False, image=_stroked):
    worker, ink, generation = _run(monkeypatch, config, image=image)
    for _ in range(14):  # pagina madura e parada: o descanso esta ao alcance
        _pass(worker, generation, image())
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
        # O papel revelado e limpo (o traco foi apagado sob a sombra):
        # so papel provado drena o streak — com traco fotometrico na
        # imagem a prova de papel falha na faixa e nada sai.
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
    """Sombra nao mascarada persiste por todas as passadas pos-saida
    enquanto o modelo detecta o traco: o orcamento congelado (hold do
    PageModel sobre o traco em suspenso mais a espera de recuperacao
    da guarda de tiles) mantem o worker acordado ate a sombra limpar
    e o traco confirmar; uma borda escura permanente ainda deixa o
    worker descansar."""
    config = InkConfig.production()
    worker, ink, generation = _mature(monkeypatch, config, idle_probe=True)
    try:
        # O traço tardio precisa estar DESENHADO na imagem: com o
        # PageModel ativo, tinta so na leitura do modelo (sem
        # novidade fotometrica) e suprimida antes do voto.
        border = _stroked()
        border[:, 336:] = HAND_TONE  # borda escura permanente
        late = (slice(224, 240), slice(16, 120))
        shadow = border.copy()
        # >= 3 linhas de tile (~24px): abaixo disso a guarda trata a sombra
        # como traco fino e nao veta adicoes.
        shadow[(slice(176, 256), slice(0, 160))] = HAND_TONE

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

        revealed = border.copy()
        revealed[late] = INK_TONE
        _pass_until(
            worker,
            generation,
            revealed,
            lambda snap: snap.confirmed[late].all(),
            occlusion._FILL_RECOVERY_PASSES
            + config.occlusion_hold_passes
            + worker._post_exit_confirm_passes()
            + 8,
        )
        assert worker.poll(generation, 0).confirmed[late].all()

        _pass_until(
            worker,
            generation,
            border,
            lambda _snap: worker.stats()["resting"],
            occlusion._RECOVERY_WAIT_PASSES + worker._post_exit_confirm_passes() + 8,
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
    mao fora pode cancelar a janela. Sem o PageModel: sob ele o bloco
    adotado segue vetado como oclusor pelo pm.exposed (o contrato do
    modelo manda no fim) — a adocao que vota ate confirmar e a do
    caminho sem o modelo, na mesma janela de voto estrito do
    production onde a regressao foi medida."""
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 3600.0)
    monkeypatch.setattr(worker_mod, "_HAND_REST_STILL_SECONDS", 0.3)
    now = [1000.0]
    ink = _Scripted()
    config = InkConfig.strict_vote(
        0.85,
        hysteresis=True,
        motion_veto=True,
        pass_interval_s=0.1,
        border_guard_px=None,
    )
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
    streak tardio completar o horizonte dele. Sem o PageModel: sob
    ele o bloco adotado segue vetado como oclusor pelo pm.exposed.
    A janela pos-saida precisa ser a de voto estrito (3), a mesma
    do production onde a regressao foi medida."""
    config = InkConfig.strict_vote(
        0.85,
        hysteresis=True,
        motion_veto=True,
        pass_interval_s=0.02,
        border_guard_px=None,
    )
    worker, ink, generation = _mature(
        monkeypatch, config, idle_probe=True, image=_paper
    )
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
    a espera tem que fechar e a trabalhadora descansar. Sem o
    PageModel: a espera medida aqui e a da guarda de tiles, na
    mesma janela de voto estrito do production onde foi medida."""
    config = InkConfig.strict_vote(
        0.85,
        hysteresis=True,
        motion_veto=True,
        pass_interval_s=0.02,
        border_guard_px=None,
    )
    worker, ink, generation = _mature(
        monkeypatch, config, idle_probe=True, image=_paper
    )
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


def test_startup_small_shadow_does_not_seed_and_hide_later_stroke(monkeypatch):
    """Regressao PR #189: sombra parada 48x48 (2x2 tiles) na abertura
    nao dispara o veto de bloco (3x3), mas a guarda fotometrica do
    PageModel barra a semeadura do interior escuro para nao esconder
    o traco desenhado logo apos revelar o papel."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _Scripted()
    config = dataclasses.replace(InkConfig.production(), pass_interval_s=0.02)
    worker = worker_mod.PageCleanWorker(ink, config=config).start()
    generation = worker.new_generation()
    try:
        shadow = (slice(48, 96), slice(48, 96))  # 2x2 tiles (24 px/tile)
        shadowed = _paper()
        shadowed[shadow] = (70, 70, 70)
        for _ in range(3):
            _pass(worker, generation, shadowed)
        assert not worker._page_model.seeded[64:80, 64:80].any()

        # Sombra sai: papel limpo semeia no tom real, e traco a luma 90 confirma.
        clean = _paper()
        for _ in range(3):
            _pass(worker, generation, clean)
        assert worker._page_model.seeded[shadow].all()

        stroke = (slice(64, 72), slice(60, 84))
        scene = clean.copy()
        scene[stroke] = (90, 90, 90)
        prob = _prob()
        prob[stroke] = 0.95
        ink.content = prob
        _pass_until(
            worker,
            generation,
            scene,
            lambda snap: snap.confirmed[stroke].all(),
            6,
        )
        assert worker.poll(generation, 0).confirmed[stroke].all()
    finally:
        worker.stop()


def test_sketch_worker_confirms_faint_graphite_between_20_and_30_contrast(monkeypatch):
    """Regressao PR #189: na trabalhadora completa, sketch()
    (page_model_novel_ratio=0.20) confirma grafite com ~25% de
    contraste de Weber enquanto razao 0.30 o suprime."""
    stroke = (slice(120, 128), slice(60, 140))

    def _run_contrast(novel_ratio):
        cfg = dataclasses.replace(
            InkConfig.sketch(),
            pass_interval_s=0.02,
            page_model_novel_ratio=novel_ratio,
        )
        worker, ink, generation = _mature(monkeypatch, cfg)
        try:
            worker.set_hand_presence(True)
            scene = _stroked()
            # ~25% abaixo do papel (~205 -> ~154): entre 20% e 30% de contraste.
            scene[stroke] = (154, 154, 154)
            prob = _prob()
            prob[stroke] = 0.95
            ink.content = prob
            for _ in range(4):
                _pass(worker, generation, scene)
            return worker.poll(generation, 0).confirmed[stroke].copy()
        finally:
            worker.stop()

    assert _run_contrast(0.20).all()
    assert not _run_contrast(0.30).any()


def test_uneven_lighting_at_startup_seeds_darker_paper_and_captures_stroke(monkeypatch):
    """Regressao PR #189 (Devin Review): faixa clara (235) em 25% da folha
    nao pode fixar o corte de abertura acima do papel mais escuro (180)
    nos outros 75%; traco fino desenhado no interior escuro precisa confirmar."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker_mod, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _Scripted()
    config = dataclasses.replace(InkConfig.production(), pass_interval_s=0.02)
    worker = worker_mod.PageCleanWorker(ink, config=config).start()
    generation = worker.new_generation()
    try:
        uneven = np.full((H, W, 3), 180, dtype=np.uint8)
        uneven[:, : W // 4] = (235, 235, 235)
        for _ in range(3):
            _pass(worker, generation, uneven)
        assert worker._page_model.seeded[:, W // 2 :].all()

        stroke = (slice(120, 128), slice(W // 2 + 20, W // 2 + 100))
        scene = uneven.copy()
        scene[stroke] = (80, 80, 80)
        prob = _prob()
        prob[stroke] = 0.95
        ink.content = prob
        _pass_until(
            worker,
            generation,
            scene,
            lambda snap: snap.confirmed[stroke].all(),
            6,
        )
        assert worker.poll(generation, 0).confirmed[stroke].all()
    finally:
        worker.stop()
