"""Contrato de geracao do ink worker: repros deterministicos.

Interleavings escolhidos com threading.Event — nao exaustivos, mas
deterministicos para os schedules conhecidos. As barreiras param a
passada nos limites relevantes, sem depender de sleeps ou cadencia real.

O teste do PageModel verifica a linearizacao desta implementacao:
new_generation() e a passada disputam o gate; a transicao fotografa o
estado persistente quando retorna, e ele nao pode mudar depois.
"""

import threading

import numpy as np

from draw.contracts import WarpedPage
from draw.ink import page_model as pm_mod
from draw.ink import paper_evidence as pe_mod
from draw.ink import worker as worker_mod
from draw.ink.config import InkConfig

H, W = 96, 128
PAPER = 235.0
INK = 40.0

Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)


class _ScriptInk:
    """Tinta falsa guiada pelo conteudo do proprio warp."""

    threshold = 0.7

    def ink_prob(self, warped):
        gray = warped if warped.ndim == 2 else warped.mean(axis=2)
        return (gray < 100).astype(np.float32) * 0.95


def _bgr(gray):
    return np.stack([gray] * 3, axis=-1).astype(np.uint8)


def _worker(config):
    return worker_mod.PageCleanWorker(_ScriptInk(), config=config).start()


def _drive(worker, generation, image, visible=None):
    before = worker.stats()["passes_total"]
    worker.submit(WarpedPage(_bgr(image), generation, Q0, visible))
    with worker._condition:
        assert worker._condition.wait_for(
            lambda: worker._passes_total > before, timeout=5.0
        ), "a passada nao executou"


def _confirmed(worker, generation):
    snap = worker.poll(generation, 0)
    return np.zeros((H, W), dtype=np.uint8) if snap is None else snap.confirmed


def _model_state(model):
    """Copia todo o estado, incluindo arrays de evidencia e caches."""
    return {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in vars(model).items()
    }


def _assert_same_model_state(at_flip, after):
    assert set(at_flip) == set(after)
    for name, before in at_flip.items():
        now = after[name]
        if isinstance(before, np.ndarray):
            np.testing.assert_equal(now, before, err_msg=name)
        else:
            assert now == before, name


class _ObservedGate:
    """Lock de teste que confirma que a transicao tentou tomar o gate."""

    def __init__(self, thread_name):
        self._lock = threading.Lock()
        self._thread_name = thread_name
        self.attempted = threading.Event()

    def __enter__(self):
        if threading.current_thread().name == self._thread_name:
            self.attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


def test_stale_pass_cannot_mutate_preserved_model(monkeypatch):
    """preserve_identity mantem o PageModel: depois que new_generation()
    retorna, nenhuma mutacao com token velho pode aterrissar no objeto
    que a geracao nova vai usar.

    A passada velha e parada na ENTRADA de exposed() — antes do primeiro
    mutador — com uma mascara parcialmente visivel para que exposed()
    (hold) e observe() (streak, _last_visible) mudem o estado de forma
    detectavel. A transicao roda em outra thread e fotografa o estado
    no instante do seu retorno. A barreira confirma que tentou tomar o
    gate antes de liberar os mutadores.
    """
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    orig_exposed = pm_mod.PageModel.exposed
    orig_observe = pm_mod.PageModel.observe
    armed = [False]

    def _gated_exposed(self, *args, **kwargs):
        if armed[0]:
            armed[0] = False
            entered.set()
            release.wait(timeout=5.0)
        return orig_exposed(self, *args, **kwargs)

    def _watched_observe(self, *args, **kwargs):
        try:
            return orig_observe(self, *args, **kwargs)
        finally:
            done.set()

    monkeypatch.setattr(pm_mod.PageModel, "exposed", _gated_exposed)
    monkeypatch.setattr(pm_mod.PageModel, "observe", _watched_observe)
    worker = _worker(InkConfig(page_model=True))
    gate = _ObservedGate("generation-flip")
    worker._pm_gate = gate
    try:
        gen1 = worker.new_generation()
        paper = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, gen1, paper)
        model = worker._page_model
        assert model is not None
        covered = np.ones((H, W), dtype=bool)
        covered[32:64, 32:64] = False
        done.clear()  # ja disparou nas passadas de semeadura
        armed[0] = True
        worker.submit(WarpedPage(_bgr(paper), gen1, Q0, covered))
        assert entered.wait(timeout=5.0)
        flip_done = threading.Event()
        at_flip = {}

        def _flip():
            worker.new_generation(preserve_identity=True)
            at_flip["state"] = _model_state(model)
            flip_done.set()

        flip_thread = threading.Thread(target=_flip, name="generation-flip")
        flip_thread.start()
        assert gate.attempted.wait(timeout=5.0)
        assert not flip_done.is_set()
        release.set()
        assert done.wait(timeout=5.0)
        assert flip_done.wait(timeout=5.0)
        flip_thread.join(timeout=5.0)
        assert not flip_thread.is_alive()
        after = _model_state(model)
        _assert_same_model_state(at_flip["state"], after)
        assert worker._page_model is model
        assert worker.error is None
    finally:
        release.set()
        worker.stop()


def test_stale_pass_cannot_repopulate_paper_ref(monkeypatch):
    """A escrita de _paper_ref e o reset de new_generation() disputam:
    a passada velha calcula a referencia fora do lock e grava o
    marcador depois de a geracao nova ja te-lo apagado."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    anchored = threading.Event()
    orig_ref = pe_mod.paper_reference
    orig_anchor = worker_mod.PageCleanWorker._pass_anchor_state
    armed = [False]

    def _gated_reference(image, visible=None):
        result = orig_ref(image, visible)
        if armed[0]:
            armed[0] = False
            entered.set()
            release.wait(timeout=5.0)
        return result

    def _anchored(self, p):
        # _pass_evidence (e a escrita do marcador) ja terminou aqui.
        anchored.set()
        return orig_anchor(self, p)

    monkeypatch.setattr(pe_mod, "paper_reference", _gated_reference)
    monkeypatch.setattr(worker_mod.PageCleanWorker, "_pass_anchor_state", _anchored)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        paper = np.full((H, W), PAPER, dtype=np.float32)
        _drive(worker, gen1, paper)
        assert worker._paper_ref is not None
        anchored.clear()  # a 1a passada tambem ancora
        armed[0] = True
        worker.submit(WarpedPage(_bgr(paper), gen1, Q0))
        assert entered.wait(timeout=5.0)
        worker.new_generation()
        release.set()
        assert anchored.wait(timeout=5.0)
        assert worker._paper_ref is None
    finally:
        release.set()
        worker.stop()


def test_stale_pass_exception_does_not_stop_worker(monkeypatch):
    """Erro local de um job ja descartado nao pode derrubar a geracao
    nova: o except classifica pela geracao da passada — obsoleta conta
    na telemetria e segue; falha da geracao corrente para o worker."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    orig_ref = pe_mod.paper_reference
    armed = [False]

    def _boom_reference(image, visible=None):
        if armed[0]:
            armed[0] = False
            entered.set()
            release.wait(timeout=5.0)
            raise RuntimeError("falha local da passada obsoleta")
        return orig_ref(image, visible)

    monkeypatch.setattr(pe_mod, "paper_reference", _boom_reference)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        paper = np.full((H, W), PAPER, dtype=np.float32)
        _drive(worker, gen1, paper)
        armed[0] = True
        worker.submit(WarpedPage(_bgr(paper), gen1, Q0))
        assert entered.wait(timeout=5.0)
        gen2 = worker.new_generation()
        release.set()
        # O except da passada velha roda na thread do worker: espera o
        # desfecho — erro registrado (comportamento antigo) ou o
        # contador de erros obsoletos (contrato).
        with worker._condition:
            assert worker._condition.wait_for(
                lambda: (
                    worker.error is not None
                    or getattr(worker, "_telemetry_stale_errors", 0) >= 1
                ),
                timeout=5.0,
            )
        assert worker.error is None
        assert worker._telemetry_stale_errors >= 1
        for _ in range(2):
            _drive(worker, gen2, paper)
        assert worker.error is None
    finally:
        release.set()
        worker.stop()


def test_current_pass_exception_stops_worker(monkeypatch):
    """Lado fatal da classificacao: excecao na geracao corrente continua
    parando o worker — so o erro de job obsoleto e descartavel."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    orig_ref = pe_mod.paper_reference
    armed = [False]

    def _boom_reference(image, visible=None):
        if armed[0]:
            armed[0] = False
            raise RuntimeError("falha da geracao corrente")
        return orig_ref(image, visible)

    monkeypatch.setattr(pe_mod, "paper_reference", _boom_reference)
    worker = _worker(InkConfig(page_model=True))
    try:
        gen1 = worker.new_generation()
        paper = np.full((H, W), PAPER, dtype=np.float32)
        _drive(worker, gen1, paper)
        armed[0] = True
        worker.submit(WarpedPage(_bgr(paper), gen1, Q0))
        with worker._condition:
            assert worker._condition.wait_for(
                lambda: worker.error is not None, timeout=5.0
            )
        assert isinstance(worker.error, RuntimeError)
        assert worker._telemetry_stale_errors == 0
    finally:
        worker.stop()


def test_generation_flip_between_phase_a_and_phase_b(monkeypatch):
    """Contrato (caracterizacao, verde hoje e depois do fencing): a
    transicao no meio do trecho captura um estado coerente — a
    referencia de identidade preservada carrega o desenho confirmado
    no instante efetivo da transicao."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    orig_pm = worker_mod.PageCleanWorker._pass_paper_and_page_model
    armed = [False]

    def _gated_pm(self, p):
        if armed[0]:
            armed[0] = False
            entered.set()
            release.wait(timeout=5.0)
        try:
            return orig_pm(self, p)
        finally:
            done.set()

    monkeypatch.setattr(
        worker_mod.PageCleanWorker, "_pass_paper_and_page_model", _gated_pm
    )
    worker = _worker(InkConfig(page_model=True))
    gate = _ObservedGate("generation-flip")
    worker._pm_gate = gate
    try:
        gen1 = worker.new_generation()
        paper = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, gen1, paper)
        stroked = paper.copy()
        stroked[50:60, 50:60] = INK
        for _ in range(4):
            _drive(worker, gen1, stroked)
        assert np.count_nonzero(_confirmed(worker, gen1)[50:60, 50:60]) == 100
        done.clear()  # dispara em toda passada; so a armada interessa
        armed[0] = True
        worker.submit(WarpedPage(_bgr(stroked), gen1, Q0))
        assert entered.wait(timeout=5.0)
        flip_done = threading.Event()

        def _flip():
            worker.new_generation(preserve_identity=True)
            flip_done.set()

        flip_thread = threading.Thread(target=_flip, name="generation-flip")
        flip_thread.start()
        assert gate.attempted.wait(timeout=5.0)
        assert not flip_done.is_set()
        release.set()
        assert done.wait(timeout=5.0)
        assert flip_done.wait(timeout=5.0)
        flip_thread.join(timeout=5.0)
        assert not flip_thread.is_alive()
        ref = worker._identity_reference
        assert ref is not None
        assert np.count_nonzero(ref[50:60, 50:60]) == 100
        assert worker.error is None
    finally:
        release.set()
        worker.stop()


def test_identity_gray_snapshot_ordered_with_observe(monkeypatch):
    """new_generation(preserve_identity=True) copia
    observed_normalized_gray sob o gate; observe() substitui o buffer
    ou grava nele com np.copyto dentro do trecho do modelo. As duas
    operacoes disputam o mesmo gate: a copia nao pode intercalar com
    a escrita — ou a transicao comita antes da passada (e o token a
    descarta) ou o snapshot lineariza depois do observe inteiro.

    A passada armada para na ENTRADA de observe() — imediatamente
    antes da escrita no buffer — e a transicao tem que esperar o gate.
    Depois do desfecho, _identity_gray_ref e uma copia do buffer ja
    observado por esta passada, nao uma mistura dos dois estados.
    """
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    orig_observe = pm_mod.PageModel.observe
    armed = [False]

    def _gated_observe(self, *args, **kwargs):
        if armed[0]:
            armed[0] = False
            entered.set()
            release.wait(timeout=5.0)
        try:
            return orig_observe(self, *args, **kwargs)
        finally:
            done.set()

    monkeypatch.setattr(pm_mod.PageModel, "observe", _gated_observe)
    worker = _worker(InkConfig(page_model=True))
    gate = _ObservedGate("generation-flip")
    worker._pm_gate = gate
    try:
        gen1 = worker.new_generation()
        paper = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, gen1, paper)
        model = worker._page_model
        assert model is not None
        done.clear()  # ja disparou nas passadas de semeadura
        darker = np.full((H, W), 200.0, dtype=np.float32)
        armed[0] = True
        worker.submit(WarpedPage(_bgr(darker), gen1, Q0))
        assert entered.wait(timeout=5.0)
        flip_done = threading.Event()

        def _flip():
            worker.new_generation(preserve_identity=True)
            flip_done.set()

        flip_thread = threading.Thread(target=_flip, name="generation-flip")
        flip_thread.start()
        assert gate.attempted.wait(timeout=5.0)
        assert not flip_done.is_set()
        release.set()
        assert done.wait(timeout=5.0)
        assert flip_done.wait(timeout=5.0)
        flip_thread.join(timeout=5.0)
        assert not flip_thread.is_alive()
        ref = worker._identity_gray_ref
        assert ref is not None
        # A copia aconteceu depois do observe inteiro: igual ao buffer
        # do modelo no desfecho, pixel a pixel (NaN inclusos).
        np.testing.assert_array_equal(ref, model.observed_normalized_gray)
        assert worker.error is None
    finally:
        release.set()
        worker.stop()


def test_covered_unseeded_mutation_ordered_with_transition(monkeypatch):
    """covered_unseeded() escreve _unseen_hold/_unseen_blocked na
    instancia persistente do PageModel — com preserve_identity o objeto
    atravessa a transicao, entao uma mutacao de token velho ficaria
    guardada no modelo da geracao nova. O gate ordena: a mutacao
    comita antes do retorno de new_generation() e nada muda depois.

    `blocked` e o dominio observavel menos o que a guarda fotometrica
    vetou — e so pixels ainda sem referencia (`~_seeded`) contam. A
    regiao [32:64,32:64] fica coberta na semeadura (nunca observada,
    nunca semeada) e a passada armada a expoe com conteudo escuro:
    a guarda veta, `blocked` fica nao vazio e `entered` escreve o
    hold — a mutacao e detectavel no estado fotografado.

    A barreira para a passada na ENTRADA de covered_unseeded() —
    imediatamente antes da escrita — e a transicao espera o gate.
    """
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    orig_covered = pm_mod.PageModel.covered_unseeded
    armed = [False]

    def _gated_covered(self, blocked, *, hold_passes):
        if armed[0]:
            armed[0] = False
            entered.set()
            release.wait(timeout=5.0)
        try:
            return orig_covered(self, blocked, hold_passes=hold_passes)
        finally:
            done.set()

    monkeypatch.setattr(pm_mod.PageModel, "covered_unseeded", _gated_covered)
    worker = _worker(InkConfig(page_model=True))
    gate = _ObservedGate("generation-flip")
    worker._pm_gate = gate
    try:
        gen1 = worker.new_generation()
        paper = np.full((H, W), PAPER, dtype=np.float32)
        covered = np.ones((H, W), dtype=bool)
        covered[32:64, 32:64] = False
        for _ in range(3):
            _drive(worker, gen1, paper, visible=covered)
        model = worker._page_model
        assert model is not None
        darkened = paper.copy()
        darkened[32:64, 32:64] = INK
        done.clear()
        armed[0] = True
        worker.submit(WarpedPage(_bgr(darkened), gen1, Q0))
        assert entered.wait(timeout=5.0)
        flip_done = threading.Event()
        at_flip = {}

        def _flip():
            worker.new_generation(preserve_identity=True)
            at_flip["state"] = _model_state(model)
            flip_done.set()

        flip_thread = threading.Thread(target=_flip, name="generation-flip")
        flip_thread.start()
        assert gate.attempted.wait(timeout=5.0)
        assert not flip_done.is_set()
        release.set()
        assert done.wait(timeout=5.0)
        assert flip_done.wait(timeout=5.0)
        flip_thread.join(timeout=5.0)
        assert not flip_thread.is_alive()
        # A mutacao desta passada comitou antes do retorno da
        # transicao: hold e blocked da regiao vetada ja estavam
        # registrados quando new_generation() devolveu.
        assert np.any(at_flip["state"]["_unseen_blocked"])
        assert np.any(at_flip["state"]["_unseen_hold"])
        after = _model_state(model)
        _assert_same_model_state(at_flip["state"], after)
        assert worker._page_model is model
        assert worker.error is None
    finally:
        release.set()
        worker.stop()


def test_rezero_model_swap_ordered_with_transition(monkeypatch):
    """O re-zero cria e atribui um PageModel novo e adianta _pm_epoch
    fora de _condition: a fase B rejeitaria o voto depois, mas a troca
    ja feita nao se desfaz. O gate ordena a troca contra a transicao —
    a barreira para a passada imediatamente antes da construcao do
    modelo novo, dentro do trecho."""
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    orig_ctor = pm_mod.PageModel
    constructed = {}
    armed = [False]

    def _gated_ctor(shape):
        if armed[0]:
            armed[0] = False
            entered.set()
            release.wait(timeout=5.0)
        pm = orig_ctor(shape)
        constructed["pm"] = pm
        return pm

    monkeypatch.setattr(pm_mod, "PageModel", _gated_ctor)
    worker = _worker(InkConfig(page_model=True))
    gate = _ObservedGate("generation-flip")
    worker._pm_gate = gate
    try:
        gen1 = worker.new_generation()
        paper = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, gen1, paper)
        old_model = worker._page_model
        assert old_model is not None
        seq0 = worker._pm_reset_seq
        epoch0 = worker._canvas_epoch
        assert worker.rezero_baseline()
        before = worker.stats()["passes_total"]
        armed[0] = True
        worker.submit(WarpedPage(_bgr(paper), gen1, Q0))
        assert entered.wait(timeout=5.0)
        flip_done = threading.Event()

        def _flip():
            worker.new_generation(preserve_identity=True)
            flip_done.set()

        flip_thread = threading.Thread(target=_flip, name="generation-flip")
        flip_thread.start()
        assert gate.attempted.wait(timeout=5.0)
        assert not flip_done.is_set()
        release.set()
        with worker._condition:
            assert worker._condition.wait_for(
                lambda: worker._passes_total > before, timeout=5.0
            )
        assert flip_done.wait(timeout=5.0)
        flip_thread.join(timeout=5.0)
        assert not flip_thread.is_alive()
        # Troca + epoca comitaram atomicamente antes da transicao: o
        # modelo vivo e o construido pelo re-zero desta passada, e o
        # epoch e exatamente um passo a frente (re-zero; a transicao
        # com preserve_identity nao mexe no sequenciador).
        assert constructed["pm"] is worker._page_model
        assert worker._page_model is not old_model
        assert worker._pm_reset_seq == seq0 + 1
        assert worker._canvas_epoch == epoch0 + 1
        assert worker._pm_epoch == (worker._canvas_epoch, worker._pm_reset_seq)
        assert worker.error is None
    finally:
        release.set()
        worker.stop()


def test_stale_pass_cannot_swap_model(monkeypatch):
    """O token e validado na mesma secao critica que autoriza a troca:
    a passada com geracao velha e descartada na entrada do trecho do
    modelo — nunca constroi nem atribui o PageModel do re-zero.

    submit() rejeita geracao velha na entrada, entao a passada tem que
    estar em voo: para no FIM da fase A (ainda sob _condition, com
    p.pm_rezero ja armado — a troca estava autorizada quando o token
    era valido); a transicao toma o gate e espera _condition — ao
    soltar, ela comita antes de a passada chegar ao check do token.
    A troca nao-preserve cancela o pedido de re-zero junto com a
    geracao velha: e a cancela pelo motivo certo, nao pela passada.
    """
    monkeypatch.setattr(worker_mod, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    committed = threading.Event()
    pm_called = threading.Event()
    orig_prepare = worker_mod.PageCleanWorker._pass_prepare_vote
    orig_commit = worker_mod.PageCleanWorker._pass_commit
    orig_pm = worker_mod.PageCleanWorker._pass_paper_and_page_model
    armed = [False]

    def _gated_prepare(self, p):
        orig_prepare(self, p)
        if armed[0]:
            # p.pm_rezero e p.pm_epoch ja foram ligados nesta passada.
            armed[0] = False
            entered.set()
            release.wait(timeout=5.0)

    def _watched_commit(self, p):
        try:
            return orig_commit(self, p)
        finally:
            committed.set()

    def _spy_pm(self, p):
        pm_called.set()
        return orig_pm(self, p)

    monkeypatch.setattr(
        worker_mod.PageCleanWorker, "_pass_prepare_vote", _gated_prepare
    )
    monkeypatch.setattr(worker_mod.PageCleanWorker, "_pass_commit", _watched_commit)
    monkeypatch.setattr(
        worker_mod.PageCleanWorker, "_pass_paper_and_page_model", _spy_pm
    )
    worker = _worker(InkConfig(page_model=True))
    gate = _ObservedGate("generation-flip")
    worker._pm_gate = gate
    try:
        gen1 = worker.new_generation()
        paper = np.full((H, W), PAPER, dtype=np.float32)
        for _ in range(3):
            _drive(worker, gen1, paper)
        model = worker._page_model
        assert model is not None
        seq0 = worker._pm_reset_seq
        assert worker.rezero_baseline()
        committed.clear()
        pm_called.clear()
        armed[0] = True
        worker.submit(WarpedPage(_bgr(paper), gen1, Q0))
        assert entered.wait(timeout=5.0)
        flip_done = threading.Event()
        flipped = {}

        def _flip():
            flipped["gen"] = worker.new_generation()
            flip_done.set()

        flip_thread = threading.Thread(target=_flip, name="generation-flip")
        flip_thread.start()
        assert gate.attempted.wait(timeout=5.0)
        assert not flip_done.is_set()
        release.set()
        assert flip_done.wait(timeout=5.0)
        flip_thread.join(timeout=5.0)
        assert committed.wait(timeout=5.0)
        # O check do token descartou a passada obsoleta na entrada do
        # trecho: o modelo velho segue vivo. A transicao nao-preserve
        # cancelou o pedido (folha nova nao herda recaptura pendente).
        assert not pm_called.is_set()
        assert worker._page_model is model
        assert not worker._pm_rezero_requested
        assert worker._pm_reset_seq == seq0 + 1
        # Um pedido NOVO na geracao nova aplica normalmente. A 1a
        # passada da geracao e a de identidade: zera canvas e modelo
        # (epoch mismatch recria na sequencia) e consumiria qualquer
        # pedido pendente — o pedido precisa vir depois dela.
        _drive(worker, flipped["gen"], paper)
        assert worker._page_model is not model
        assert worker.rezero_baseline()
        _drive(worker, flipped["gen"], paper)
        assert worker._pm_reset_seq == seq0 + 2
        assert worker.error is None
    finally:
        release.set()
        worker.stop()
