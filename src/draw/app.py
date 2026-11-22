"""Interface de producao: janela, teclas e montagem do pipeline.

Uso:
  python -m draw                                  # webcam local (indice 0)
  python -m draw --check                          # valida modelos e camera e sai
  python -m draw --fullscreen                     # tela cheia, sem a moldura
  python -m draw --markers                        # folha-modelo impressa
  python -m draw --neural modelo.onnx             # outro detector de folha
  python -m draw --page-clean modelo.onnx         # outro modelo de tinta
  python -m draw --hand-model modelo.task         # outro modelo de mao
  python -m draw --source 1                       # outra webcam local
  python -m draw --source http://192.168.0.10:8080/video   # camera via rede
  python -m draw --source sessao.drawcap          # gravacao do ajudante
  python -m draw --source tcp://192.168.0.10      # ajudante ao vivo
      --helper-token-file token.txt               # token do ajudante
  python -m draw --source hvcam://                # webcam do Windows via vsock
  python -m draw --source hvcam+reverse://        # conexao sem registro/admin
  python -m draw --source hvcam+tcp://10.0.0.2    # idem, fallback TCP

Sem nenhuma flag, o caminho e o de producao: folha pelo detector
neural (models/draw_seg_v3_dyn.onnx, com o classico de reserva) e
tinta pelo modelo pagina->limpa (models/page_clean_unet.onnx), em
duas camadas na trabalhadora: a provisoria aparece na tela na
primeira passada e a confirmada, que e o canvas dos consumidores,
entra pelo voto entre passadas. Os binarios moram em models/ na
raiz do repositorio e sao conferidos contra assets/manifest.json
antes de abrir (sha256 + tamanho); --allow-unverified rebaixa a
recusa para aviso.

A tinta usa por padrao o preset de producao (InkConfig.production):
ganho digital, voto 0,85, PageModel (pauta, desenho antigo e sombra
suprimidos pela referencia da pagina), histerese, veto por
movimento, passadas a 0,15 s e Detect-Wake. --legacy-ink volta ao
comportamento anterior; --ink-vote e --ink-interval ajustam os numeros.

A fonte via rede (MJPEG/RTSP) e o caminho recomendado no WSL. O WSL nao
enxerga webcams USB do Windows por padrao, mas enxerga streams de rede.
A alternativa nativa e a fonte hvcam://, que puxa os quadros da webcam
do Windows por Hyper-V socket (hvcam-host.exe, santleme/hvcam) — sem
IP, sem firewall e sem usbipd.
Para evitar o registro de servico do Windows, rode o Draw com
`--source hvcam+reverse://` e inicie `hvcam-host.exe --reverse` no
Windows (a abertura aguarda ate 35 s pela conexao inicial do host).
Assim todo o processamento (e o ML futuro) fica no Linux. A camera so
envia os frames. Ela pode ser um celular, outro computador ou uma camera IP.

Teclas: s = salvar o canvas como PNG, f = capturar um frame cru,
g = ligar ou desligar a captura automatica de frames, q/Esc = sair.
"""

import argparse
import dataclasses
import itertools
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import camera as camera_module
from . import config
from . import paper as paper_module
from . import perception as neural_module
from . import ui as ui_module
from . import view as view_module
from .auto_orient import AutoPageOrientation, AutoRotationDetector
from .camera import (
    LatestFrameCapture,
    PngWriter,
    open_capture,
    parse_source,
)
from .hand import HandMasker, hand_model_available
from .hand_presence import HandPresenceWorker, make_hand_presence
from .infra import manifest as manifest_module
from .infra import session_report as session_report_module
from .infra import telemetry as telemetry_module
from .infra.safe_io import ensure_private_dir
from .ink import render as ink_render
from .ink.config import InkConfig
from .ink.worker import PageCleanWorker
from .page_clean import PageCleanInk
from .perception import NeuralSheetDetector
from .pipeline import INK_WORKER_FAILED, Pipeline
from .selfcheck import (
    _gui_backend_compiled as _gui_backend_compiled,
)
from .selfcheck import (
    _require_gui_backend,
    run_check,
)
from .settings import PipelineConfig
from .tracker import SheetDetectionWorker, _corners_json

# Tempo que o aviso de fonte morta fica na janela antes da saida.
# Sem ele, a morte da fonte e um congelamento silencioso. O waitKey
# devolve antes se qualquer tecla for apertada.
_DEAD_NOTICE_MS = 3000

# Teto do pool paralelo do OpenCV. Sem teto, o pool abre com as
# CPUs logicas do host (16 no Ryzen) e vira o terceiro disputante
# dos nucleos fisicos, junto dos pools ORT do detector e da tinta
# (oversubscription medida na mesa do Lucas, 2026-11-02: CPU
# desenhando 550,8% com os pools 4+8+16; teto combinado 2+4 no
# ORT cortou para 289,8%, passada igual).
_OPENCV_THREAD_CAP = 4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SHEET_MODEL = _REPO_ROOT / "models" / "draw_seg_v3_dyn.onnx"

_CLASSIC_TRACE_VERDICT = {"paper": "corners", "blank_maybe": "blank_maybe"}


def _apply_opencv_thread_cap():
    """Limita o pool do OpenCV ao teto combinado. Devolve o teto."""
    cap = max(1, min(_OPENCV_THREAD_CAP, os.cpu_count() or _OPENCV_THREAD_CAP))
    cv2.setNumThreads(cap)
    return cap


def _detect_sheet_combined_traced(frame, neural_detector):
    """Detecção do modo sem marcadores, com o rastro da aquisição.

    Neural primário, clássico de reserva — a decisão do legado,
    linha a linha. Sem detector neural, é o clássico puro. Com
    detector, ele decide primeiro; fechando o quadrilátero, vale o
    veredito dele. Não fechando, a reserva clássica só fala quando a
    neural não negou a folha com força: logit de presença até
    PRESENCE_VETO_LOGIT devolve (None, None) direto. Logit negativo
    fraco (acima do limiar) não veta: a reserva segue valendo para a
    folha real que a neural subdispara num setup novo.

    Terceiro desfecho: se nenhum quadrilátero fechou mas o logit de
    presença é positivo, devolve (None, "paper"): presença sem
    posição. O rastreador lê isso como "a folha está lá, segura o
    quadrilátero anterior e renova o relógio de perda".

    O terceiro elemento da saída é o rastro (presença, fonte do
    desfecho, vereditos parciais) que alimenta o evento de aquisição
    da telemetria. O quarto são os extras: sempre um dict (mesmo
    vazio) para que uma detecção sem máscaras LIMPE a cobertura e a
    máscara de mão anteriores no resultado publicado.
    """
    trace = {
        "neural_ran": neural_detector is not None,
        "presence_logit": None,
        "veto_logit": None,
        "veto_margin": None,
        "neural_corners": None,
        "neural_verdict": None,
        "fit_reject": None,
        "classic_ran": False,
        "classic_verdict": None,
        "classic_corners": None,
        "scene_stats": None,
    }
    extras = {}

    def _done(corners, verdict, source):
        trace["source"] = source
        trace["combined_corners"] = _corners_json(corners)
        trace["combined_verdict"] = verdict
        # extras sai sempre (mesmo vazio): é o que deixa a detecção
        # clássica ou sem máscara LIMPAR a cobertura anterior no
        # laço, em vez de deixá-la defasada.
        return corners, verdict, trace, extras

    def _classic():
        corners, verdict = paper_module.detect_sheet_with_confidence(frame)
        trace["classic_ran"] = True
        trace["classic_verdict"] = (
            _CLASSIC_TRACE_VERDICT.get(verdict, verdict)
            if corners is not None
            else None
        )
        trace["classic_corners"] = _corners_json(corners)
        return corners, verdict

    if neural_detector is not None:
        trace["veto_logit"] = float(neural_module.PRESENCE_VETO_LOGIT)
        # A detecção pede a máscara sempre: a cobertura vai no
        # extras para o laço retificar com a homografia do warp de
        # tinta; a máscara de mão alimenta a oclusão semântica.
        corners, verdict, mask = neural_detector(frame, return_mask=True)
        # Sinais de qualidade de cena da segmentação (issue #106): o
        # rastro os leva à telemetria em TODOS os desfechos neurais —
        # a cena estranha sem quadrilátero (veto, presença sem
        # posição) é exatamente o caso medido lá. getattr com default:
        # dublês de teste antigos não têm o atributo.
        trace["scene_stats"] = getattr(neural_detector, "last_scene_stats", None)
        if corners is not None and mask is not None:
            # Máscara quase vazia não é evidência de folha: sem o
            # piso, uma detecção ruim transformaria a cobertura num
            # voto sempre-zero e secaria a tinta inteira. A
            # cobertura é papel+tinta+mão — só o fundo fica fora.
            coverage_mask = getattr(neural_detector, "last_coverage_mask", None)
            if (
                coverage_mask is not None
                and np.count_nonzero(coverage_mask) > coverage_mask.size * 0.05
            ):
                frame_h, frame_w = frame.shape[:2]
                if coverage_mask.shape != (frame_h, frame_w):
                    coverage_mask = cv2.resize(
                        coverage_mask,
                        (frame_w, frame_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                extras["coverage_mask"] = coverage_mask
        # Máscara semântica de mão (mesma remessa, mesmo frame):
        # segundo sinal de oclusão para o canal visible do voto de
        # tinta. Sobe em TODO desfecho neural — o fit de cantos pode
        # falhar numa cena onde a segmentação ainda produziu a
        # evidência, e o resultado clássico precisa dela para
        # renovar a oclusão vencida e destravar a tinta (review do
        # PR #121). hand_mask ausente continua significando
        # "evidência indisponível", distinto de "sem mão".
        hand_mask = getattr(neural_detector, "last_hand_mask", None)
        if hand_mask is not None:
            frame_h, frame_w = frame.shape[:2]
            if hand_mask.shape != (frame_h, frame_w):
                hand_mask = cv2.resize(
                    hand_mask,
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            extras["hand_mask"] = hand_mask
        logit = neural_detector.last_presence_logit
        trace["presence_logit"] = None if logit is None else float(logit)
        # A distância do logit ao limiar é o dado que a recalibração
        # do veto em outras mesas precisa (issue #105); hoje o rastro
        # só mostra source:"veto".
        trace["veto_margin"] = (
            None if logit is None else float(logit - neural_module.PRESENCE_VETO_LOGIT)
        )
        # getattr: dublês de teste podem não definir last_fit_reject.
        trace["fit_reject"] = getattr(neural_detector, "last_fit_reject", None)
        trace["neural_corners"] = _corners_json(corners)
        trace["neural_verdict"] = verdict
        if corners is not None:
            return _done(corners, verdict, "neural")
        if logit is not None and logit <= neural_module.PRESENCE_VETO_LOGIT:
            return _done(None, None, "veto")
        corners, verdict = _classic()
        if corners is None and logit is not None and logit > 0:
            return _done(None, "paper", "presence_only")
        return _done(corners, verdict, "classic" if corners is not None else "none")
    corners, verdict = _classic()
    return _done(corners, verdict, "classic" if corners is not None else "none")


_SNAPSHOT_SEQ = itertools.count()


def _save_stamp():
    """Nome único por processo: epoch em ms + contador monotônico.

    Segundos inteiros colidem quando o usuário salva duas vezes no
    mesmo segundo; milissegundos ainda colidem na fila do PngWriter —
    o contador zera a chance sem depender de resolução de relógio.
    """
    return f"{time.time_ns() // 1_000_000}_{next(_SNAPSHOT_SEQ)}"


def _report_format_suggestion(telemetry, suggestion):
    telemetry.event("format_suggestion", **suggestion)
    ratio = suggestion["ratio"]
    ratio_text = "indisponivel" if ratio is None else f"{ratio:.4f}"
    label = suggestion["suggestion"]
    if label == "serie_a":
        label = (
            "serie A (proporcao 0,71) — tamanho nao estimado; "
            "A4 assumido se nada for escolhido"
        )
    elif label is None:
        label = f"abstencao ({suggestion['reason']})"
    print(f"Sugestao de formato: proporcao {ratio_text}; {label}.")


def _ignore_ink(_publication):
    pass


def _is_official_model_target(requested, official_name):
    """True quando `requested` e None ou aponta para models/<official_name>."""
    if requested is None:
        return True
    official = manifest_module.DEFAULT_MODELS_DIR / official_name
    req_path = Path(requested)
    try:
        return os.path.abspath(req_path) == os.path.abspath(official)
    except OSError:
        return False


def _provision_official_models(neural_model, page_clean_model, hand_model):
    """Baixa individualmente cada modelo oficial ausente pedido pela sessao.

    Falha no download de um modelo opcional (ex.: folha ou mao) nao aborta
    a tentativa de baixar o modelo obrigatorio de tinta (`page_clean_unet.onnx`).
    """
    for requested, official_name in (
        (page_clean_model, "page_clean_unet.onnx"),
        (neural_model, "draw_seg_v3_dyn.onnx"),
        (hand_model, "hand_landmarker.task"),
    ):
        if not _is_official_model_target(requested, official_name):
            continue
        try:
            manifest_module.ensure_models([official_name])
        except RuntimeError as error:
            print(f"Aviso: download automatico de {official_name} falhou ({error}).")


def run(
    source=0,
    save_dir="captures",
    markers_mode=False,
    neural_model=None,
    hand_model=None,
    page_clean_model=None,
    helper_token_file=None,
    fullscreen=False,
    rotation=0,
    pipeline_config=None,
    stream_port=None,
    stream_token_file=None,
    stream_record=None,
    auto_rotate=False,
    auto_orientation=False,
    spread_mode_explicit=False,
):
    spread_mode = "auto" if pipeline_config is None else pipeline_config.spread_mode
    spread_mode_explicit = spread_mode_explicit or spread_mode != "auto"
    if markers_mode and spread_mode_explicit:
        raise ValueError(
            "spread_mode nao combina com --markers (requer detector neural)"
        )

    _apply_opencv_thread_cap()
    _require_gui_backend()
    ui_module._open_window(fullscreen)

    # Recursos que o finally derruba; None = ainda nao iniciado.
    # Depois do probe de GUI, TODO o setup roda dentro do try: uma
    # falha em qualquer ponto (pasta sem permissao, modelo, camera)
    # desce pelo finally liberando o que ja subiu.
    ink_worker = None
    hand_masker = None
    hand_presence_worker = None
    detection_worker = None
    frame_source = None
    png_writer = None
    telemetry = None
    session_report = None
    ink_stream = None
    pipeline = None
    try:
        if not markers_mode:
            _provision_official_models(neural_model, page_clean_model, hand_model)
        neural_detector = None
        if neural_model is None and not markers_mode and _DEFAULT_SHEET_MODEL.exists():
            neural_model = str(_DEFAULT_SHEET_MODEL)
        if neural_model is not None and not markers_mode:
            neural_detector = NeuralSheetDetector(
                neural_model,
                verify=manifest_module.verify_model,
                spread_mode=spread_mode,
            )
            neural_detector.load()
            print(
                f"Detector neural ativo: {neural_model}. "
                "O classico cobre os frames em que o neural nao ve folha."
            )
        elif not markers_mode:
            print(
                "Aviso: sem detector neural de folha (modelo nao "
                "encontrado); usando so o detector classico, que nao "
                "enxerga a folha em varias mesas. Baixe os modelos "
                "oficiais pela Release do repositorio."
            )
            if spread_mode_explicit:
                print(
                    f"Aviso: spread_mode={spread_mode!r} nao aplicado; "
                    "ele requer o detector neural de folha."
                )

        page_clean = None
        if not markers_mode:
            page_clean = PageCleanInk(
                page_clean_model, verify=manifest_module.verify_model
            )
            try:
                page_clean.load()
                print(
                    f"Tinta pagina->limpa ativa: {page_clean.model_path} "
                    f"(limiar {page_clean.threshold}). Canvas por uniao "
                    "simples das mascaras."
                )
            except (RuntimeError, FileNotFoundError) as error:
                raise RuntimeError(
                    f"O modelo pagina->limpa nao carregou ({error}); sem "
                    "ele nao ha tinta no caminho novo. Baixe os modelos "
                    "oficiais pela Release do repositorio."
                ) from error

        ink_worker = None
        hand_masker = None
        if page_clean is not None:
            hand_masker = (
                HandMasker(hand_model, verify=manifest_module.verify_model)
                if hand_model_available(hand_model)
                else None
            )
            ink_worker = PageCleanWorker(
                page_clean,
                hand_masker,
                config=None if pipeline_config is None else pipeline_config.ink,
                ink_divnorm=(
                    False if pipeline_config is None else pipeline_config.ink_divnorm
                ),
            ).start()
            print(
                "Tinta em thread propria: o laco principal nao espera "
                "o modelo (passadas com cadencia de ~250 ms)."
            )

        clean_hand_presence = None
        hand_presence_worker = None
        if page_clean is not None and not markers_mode:
            clean_hand_presence = make_hand_presence(
                hand_model, verify=manifest_module.verify_model
            )
            if clean_hand_presence is not None:
                print(
                    "Porta de ociosidade pela mao ativa no caminho "
                    "pagina->limpa: sem mao, a tinta sonda ate a saida "
                    "confirmada assentar, descansa sem passadas e a "
                    "geometria da tinta congela."
                )
                # Amostragem fora do laco (T4): o laco submete e colhe.
                hand_presence_worker = HandPresenceWorker(clean_hand_presence).start()

        def detect_sheet(frame):
            return _detect_sheet_combined_traced(frame, neural_detector)

        capture = open_capture(
            parse_source(source), helper_token_file=helper_token_file
        )
        frame_source = LatestFrameCapture(capture, rotation=rotation).start()
        detection_worker = None
        if not markers_mode:
            # coverage=True na trabalhadora: os extras da deteccao
            # (mascara de cobertura e de mao) sempre sao publicados — a
            # mascara de mao alimenta a oclusao semantica mesmo quando
            # a cobertura esta desligada no pipeline.
            detection_worker = SheetDetectionWorker(detect_sheet, coverage=True).start()
        png_writer = PngWriter()
        # Capturas, telemetria e relatórios são privados: a pasta ganha
        # um .gitignore "*" qualquer que seja o --save-dir.
        save_path = ensure_private_dir(save_dir)
        telemetry = telemetry_module.SessionTelemetry(
            save_path / "telemetry.jsonl"
        ).start()
        paper_spec = config.current()
        format_metadata = {
            "paper_format": paper_spec.name,
            "paper_orientation": paper_spec.orientation,
            "page_grid": [paper_spec.width, paper_spec.height],
            "mm_per_px": config.mm_per_px(),
        }
        telemetry.update(**format_metadata)
        telemetry.event("session_open", **format_metadata)
        print(
            f"Formato da sessao: {paper_spec.name} ({paper_spec.orientation}), "
            f"grade {paper_spec.width}x{paper_spec.height}, "
            f"{config.mm_per_px():.6f} mm/px."
        )
        print(
            "Telemetria de sessao ligada: 1 registro/s em "
            f"{save_path / 'telemetry.jsonl'} (arquivo local e rotativo)."
        )

        # Relatorio de sessao: um JSON por corrida com cabecalho
        # (modelos com sha, abertura da camera, porta da mao) e os
        # agregados da sessao, para uma corrida que falha deixar dito
        # o que rodou — a telemetria bruta sozinha nao diz.
        def _model_header(path):
            if path is None:
                return None
            try:
                return {
                    "arquivo": str(path),
                    "sha256": manifest_module.sha256_of(path),
                }
            except OSError:
                return {"arquivo": str(path), "sha256": None}

        session_report = session_report_module.SessionReport(
            save_path / "session-reports",
            {
                "inicio_epoch": time.time(),
                **format_metadata,
                "fonte": str(source),
                "modo_marcadores": bool(markers_mode),
                "modelo_folha": _model_header(
                    neural_model if neural_detector is not None else None
                ),
                "modelo_tinta": _model_header(
                    page_clean.model_path if page_clean is not None else None
                ),
                "tinta_modo": (
                    "page-clean" if page_clean is not None else "marcadores"
                ),
                "mao_disponivel": clean_hand_presence is not None,
                "camera_abertura": camera_module._LAST_OPEN_PROBE,
            },
        ).start()
        print(
            "Relatorio de sessao ligado: resumo desta corrida em "
            f"{session_report.path} (atualizado a cada 15 s e no fim)."
        )
        if markers_mode:
            print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
        else:
            print("Draw v0 rodando. Mostre uma folha para a camera.")
        print(
            "Teclas: s = salvar canvas, f = capturar frame, "
            "g = captura automatica (desligada), q/Esc = sair."
        )

        frames_path = save_path / "frames"
        frames_saved = 0
        auto_capture = False
        loop_frames = 0
        tel_window_start = time.monotonic()
        tel_window_frames = 0
        tel_loop_fps = 0.0
        tel_detect_ms = None
        tel_sheet_present = False

        def save_raw_frame(img):
            frames_path.mkdir(parents=True, exist_ok=True)
            out = frames_path / f"frame_{int(time.time() * 1000)}.png"

            def _saved(path):
                nonlocal frames_saved
                frames_saved += 1
                print(f"Frame salvo em {path} (total: {frames_saved}).")

            png_writer.save(out, img, _saved)

        def handle_capture_key(key, img):
            nonlocal auto_capture
            if key == ord("f"):
                save_raw_frame(img)
            elif key == ord("g"):
                auto_capture = not auto_capture
                if auto_capture:
                    print(
                        "Captura automatica ligada: um frame a cada "
                        f"10 quadros vai para {frames_path}."
                    )
                else:
                    print("Captura automatica desligada.")

        format_suggester = paper_module.FormatSuggester()

        def save_canvas_snapshot(fallback_panel=None):
            """Salva a camada confirmada (nunca a provisoria).

            Com folha em cena, `fallback_panel` e a folha retificada
            quando ainda nao ha tinta confirmada. Sem folha, salva
            so o canvas confirmado retido (canvas_persist) — a folha
            em branco e a provisoria nunca viram arquivo.
            """
            if pipeline is not None and pipeline.clean_mask is not None:
                # O snapshot e um consumidor de canvas: ele salva a
                # camada confirmada, nunca o painel com a tinta
                # provisoria.
                save_image = ink_render.render_canvas(pipeline.clean_mask)
            elif fallback_panel is not None:
                save_image = fallback_panel
            else:
                print("Nada a salvar: sem canvas confirmado.")
                return
            save_path.mkdir(parents=True, exist_ok=True)
            out = save_path / f"canvas_{_save_stamp()}.png"
            png_writer.save(
                out, save_image, lambda path: print(f"Canvas salvo em {path}")
            )

        def save_page_snapshot(prefix, mask):
            save_path.mkdir(parents=True, exist_ok=True)
            snapshot_path = save_path / f"{prefix}_{_save_stamp()}.png"
            png_writer.save(
                snapshot_path,
                ink_render.render_canvas(mask),
                lambda path: print(f"Pagina anterior salva em {path}"),
            )

        def emit_acquisition(record):
            telemetry.event("acquisition", **record)

        tracker = None
        if not markers_mode:
            if stream_port is not None or stream_record is not None:
                from .stream import InkStreamPublisher

                ink_stream = InkStreamPublisher(
                    port=stream_port,
                    token_file=stream_token_file,
                    paper_mm=config.paper_mm(),
                    record_path=stream_record,
                )
                if ink_stream.port is not None:
                    print(f"Stream de tinta em 127.0.0.1:{ink_stream.port}")
            pipeline = Pipeline(
                detection_worker=detection_worker,
                ink_worker=ink_worker,
                hand_presence_worker=hand_presence_worker,
                hand_masker=hand_masker,
                format_suggester=format_suggester,
                on_format_suggestion=lambda suggestion: _report_format_suggestion(
                    telemetry, suggestion
                ),
                on_acquisition=emit_acquisition,
                on_ink=ink_stream.publish if ink_stream is not None else _ignore_ink,
                ink_meta=ink_stream is not None,
                config=pipeline_config,
            )
            tracker = pipeline.tracker

        last_seq = 0
        last_view = None
        frame = None
        # Cache da ultima folha retificada do modo marcadores: sem
        # pipeline de tinta nao ha mascara confirmada, entao o 's'
        # durante um stall so tem o que salvar com este fallback.
        last_marker_warped = None
        # Ultimo quadro da camera ja composto: a interface (pisca, botoes,
        # mouse, janela) segue viva com ele enquanto a camera nao entrega
        # quadro novo, e uma travada da camera fica so dentro da camera.
        ui_inputs = None
        ui_signature = None

        ui_module._camera_menu["current"] = parse_source(source)
        base_rotation = rotation
        seq_offset = 0
        reset_source = False
        preserve_ink_on_reset = False
        auto_rotator = (
            AutoRotationDetector() if (auto_rotate and not markers_mode) else None
        )
        auto_orienter = (
            AutoPageOrientation(initial=config.current().orientation)
            if (auto_orientation and not markers_mode)
            else None
        )
        last_auto_rot_det_seq = 0
        switch = {"thread": None, "result": None}

        def start_switch(target):
            ui_module._camera_menu["busy"] = target

            def work():
                try:
                    # Nova fonte abre com a rotacao base da CLI (nao herda
                    # o flip automatico inferido para a camera anterior).
                    candidate = ui_module._open_ready_source(
                        target, helper_token_file, rotation=base_rotation
                    )
                    switch["result"] = (target, candidate, None)
                except Exception as error:
                    switch["result"] = (target, None, error)

            switch["thread"] = threading.Thread(target=work, daemon=True)
            switch["thread"].start()

        def service_camera_switch():
            nonlocal frame_source, last_seq, seq_offset, reset_source, rotation
            target = ui_module._camera_menu["request"]
            if target is not None and switch["thread"] is None:
                ui_module._camera_menu["request"] = None
                start_switch(target)
            result = switch["result"]
            if result is None:
                return
            switch["result"], switch["thread"] = None, None
            target, candidate, error = result
            ui_module._camera_menu["busy"] = None
            if error is not None:
                print(f"Nao consegui abrir {ui_module._camera_label(target)}: {error}")
                ui_module._camera_menu["notice"] = target
                ui_module._camera_menu["notice_until"] = time.monotonic() + 2.5
                return
            old, frame_source = frame_source, candidate
            old.stop(force=False)
            seq_offset += last_seq
            last_seq = 0
            rotation = base_rotation
            if auto_rotator is not None:
                auto_rotator.reset()
            if auto_orienter is not None:
                auto_orienter.reset_streak()
            reset_source = True
            ui_module._camera_menu["current"] = target
            print(f"Camera trocada para {ui_module._camera_label(target)}.")

        def render_ui(status, canvas):
            nonlocal last_view, ui_signature
            ui = ui_module._ui_state()
            ui_module._camera_menu["cam_shape"] = status.shape
            ui_signature = ui_module._ui_signature(ui)
            # O resize do painel e cacheado por publicacao: geracao +
            # versao identificam o conteudo do canvas acumulado (a
            # versao reinicia por geracao). Canvas sem publicacao
            # propria (blank, warp de marcadores) vai sem token.
            canvas_version = (
                (pipeline.clean_generation, pipeline.clean_display_version)
                if pipeline is not None and canvas is pipeline.accumulated_canvas
                else None
            )
            last_view = view_module.compose_view(
                status,
                canvas,
                size=ui["size"],
                rec_on=ui["rec_on"],
                buttons=ui["buttons"],
                pressed=ui["pressed"],
                wordmark=ui["wordmark"],
                camera_menu=ui["camera"],
                settings_menu=ui["settings"],
                hud=ui["hud"],
                canvas_version=canvas_version,
            )
            ui_module._show(last_view)

        while True:
            service_camera_switch()
            packet = frame_source.wait_next(
                last_seq, timeout=ui_module._ui_wait_seconds()
            )
            if packet is None and not frame_source.dead:
                # Fonte viva sem quadro novo na janela: a leitura pode
                # estar dentro de um stall TCP prolongado. Antes do
                # primeiro quadro, mostra o aviso de espera (a janela ja
                # existe, mas fica vazia); depois dele, a interface
                # segue viva com o ultimo quadro. As teclas sao
                # atendidas sem realimentar o quadro velho no pipeline.
                if ui_inputs is None:
                    ui_inputs = (ui_module._waiting_view(), view_module.blank_sheet())
                    render_ui(*ui_inputs)
                elif ui_module._ui_signature(ui_module._ui_state()) != ui_signature:
                    render_ui(*ui_inputs)
                key = ui_module._next_key()
                if key == ord("s"):
                    # Marcadores nao tem mascara confirmada: o salvavel
                    # visivel na tela de espera e a folha retificada.
                    save_canvas_snapshot(last_marker_warped if markers_mode else None)
                if frame is not None:
                    handle_capture_key(key, frame)
                if key in (ord("q"), 27):
                    break
                continue
            if packet is None:
                # A thread de captura declarou a fonte morta (falhas
                # seguidas de leitura ou excecao no read) e wait_next
                # acordou sem quadro. O aviso vai para o console e para
                # a janela, sobre a ultima vista, e fica _DEAD_NOTICE_MS
                # ou ate uma tecla: so o console seria um congelamento
                # silencioso para quem esta olhando a janela.
                print(
                    "A camera abriu, mas parou de entregar quadros. "
                    "Feche outros programas usando a camera e tente de novo."
                )
                ui_module._show(view_module.dead_source_view(last_view))
                cv2.waitKey(_DEAD_NOTICE_MS)
                break
            frame = packet.image
            last_seq = packet.sequence
            ui_module._hud["captured_at"] = packet.captured_at
            if seq_offset:
                packet = dataclasses.replace(packet, sequence=last_seq + seq_offset)
            if reset_source:
                reset_source = False
                if pipeline is not None:
                    if preserve_ink_on_reset:
                        # Correcao automatica (180°/retrato-paisagem) nao e troca
                        # de camera: a tinta confirmada nao pode sumir do Save.
                        # A grade antiga nao serve na nova orientacao, entao
                        # guarda um snapshot da pagina antes de limpar.
                        kept = pipeline.clean_mask
                        if kept is not None and bool(np.any(kept)):
                            save_page_snapshot("pagina_reorientada", kept)
                    pipeline.reset_source(packet.captured_at)
                preserve_ink_on_reset = False

            loop_frames += 1
            tel_window_frames += 1
            tel_now = time.monotonic()
            if tel_now - tel_window_start >= 1.0:
                tel_loop_fps = round(
                    tel_window_frames / (tel_now - tel_window_start), 2
                )
                ui_module._hud["fps"] = tel_loop_fps
                tel_window_start = tel_now
                tel_window_frames = 0
            tel_fields = {
                "loop_fps": tel_loop_fps,
                "detect_ms": tel_detect_ms,
                "sheet_present": tel_sheet_present,
                "state": "sheet" if tel_sheet_present else "no_sheet",
            }
            tel_fields.update(frame_source.stats())
            if ink_worker is not None:
                tel_fields.update(ink_worker.stats())
            if detection_worker is not None:
                tel_fields.update(detection_worker.stats())
            # Contadores do laço — rejeições do piso de cobertura —
            # entram no retrato de 1 Hz.
            if pipeline is not None:
                tel_fields.update(pipeline.stats())
            telemetry.update(**tel_fields)
            session_report.update(**tel_fields)
            if auto_capture and loop_frames % 10 == 0:
                save_raw_frame(frame)

            if markers_mode:
                warped, markers = paper_module.warp_paper(frame)
                sheet_present = warped is not None
                tel_sheet_present = sheet_present
                if sheet_present:
                    suggestion = format_suggester.observe(
                        [markers[i] for i in config.MARKER_IDS],
                        frame.shape,
                        sheet_boundary=False,
                    )
                    if suggestion is not None:
                        _report_format_suggestion(telemetry, suggestion)
            else:
                markers = {}
                warped = None
                loop_outcome = pipeline.step(packet)
                tel_detect_ms = pipeline.detect_ms
                sheet_present = pipeline.sheet_present
                tel_sheet_present = sheet_present
                if (
                    auto_rotator is not None
                    and pipeline.last_det_seq != last_auto_rot_det_seq
                ):
                    last_auto_rot_det_seq = pipeline.last_det_seq
                    # A máscara de mão acompanha qualquer desfecho
                    # neural — veto e "none" incluídos — mas a
                    # rotação por borda só faz sentido quando a
                    # máscara veio de detecção que fechou cantos:
                    # mãos no setup da mesa ou num veto na janela de
                    # remoção queimariam o único flip sem folha de
                    # evidência. O gate vive em
                    # Pipeline.rotation_hand_mask (review PR #123).
                    rotation_mask = pipeline.rotation_hand_mask
                    if rotation_mask is not None and auto_rotator.observe_mask(
                        rotation_mask
                    ):
                        rotation = 180 if frame_source.rotation == 0 else 0
                        frame_source.set_rotation(rotation)
                        reset_source = True
                        preserve_ink_on_reset = True
                        if auto_orienter is not None:
                            auto_orienter.reset_streak()
                        print(
                            "Camera invertida detectada automaticamente pela mao; "
                            f"girando entrada em {rotation}°."
                        )
                if (
                    auto_orienter is not None
                    and not reset_source
                    and sheet_present
                    and tracker.corners is not None
                ):
                    new_orient = auto_orienter.observe(tracker.corners, frame.shape)
                    if new_orient is not None:
                        config.apply_format(config.current().name, new_orient)
                        paper_spec = config.current()
                        if ink_stream is not None:
                            ink_stream.set_paper_mm(paper_spec.paper_mm())
                        reset_source = True
                        preserve_ink_on_reset = True
                        print(
                            "Orientacao da folha ajustada automaticamente para "
                            f"{new_orient} ({paper_spec.width}x{paper_spec.height})."
                        )
                if loop_outcome == INK_WORKER_FAILED:
                    ui_module._show(
                        view_module.error_view(
                            last_view,
                            "A tinta pagina->limpa falhou",
                            [
                                "O canvas parou de atualizar.",
                                "Veja a mensagem no terminal e reinicie o app.",
                            ],
                        ),
                    )
                    cv2.waitKey(_DEAD_NOTICE_MS)
                    break

            status = frame.copy()
            if markers_mode:
                for x, y in markers.values():
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
            elif tracker.corners is not None:
                view_corners = (
                    pipeline.display_corners
                    if pipeline.display_corners is not None
                    else tracker.corners
                )
                outline = view_corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(status, [outline], True, (0, 255, 0), 2)
                for x, y in view_corners:
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
            if not sheet_present:
                if markers_mode:
                    last_marker_warped = None  # quadro processado sem folha
                    status_text = f"Mostre a folha-modelo ({len(markers)}/4 marcadores)"
                else:
                    status_text = "Mostre uma folha para a camera"
                cv2.putText(
                    status,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                panel = (
                    pipeline.accumulated_canvas
                    if pipeline is not None and pipeline.accumulated_canvas is not None
                    else view_module.blank_sheet()
                )
                render_ui(status, panel)
                ui_inputs = (status, panel)
                key = ui_module._next_key()
                if key == ord("s"):
                    # Sem folha, so o canvas confirmado retido salva.
                    save_canvas_snapshot()
                handle_capture_key(key, frame)
                if key in (ord("q"), 27):
                    break
                continue
            else:
                if not markers_mode:
                    if tracker.verdict == "blank_maybe":
                        status_text = "Folha detectada (baixa confianca)"
                        status_color = (0, 255, 255)
                    else:
                        status_text = "Folha detectada"
                        status_color = (0, 255, 0)
                    cv2.putText(
                        status,
                        status_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        status_color,
                        2,
                    )
                if markers_mode:
                    # A heuristica de tinta do legado nao migrou: no
                    # modo de marcadores o painel mostra a folha
                    # retificada como esta — e e ela que o 's' salva,
                    # inclusive num stall (cache retido do ultimo warp).
                    canvas = warped
                    last_marker_warped = warped
                elif pipeline.accumulated_canvas is not None:
                    canvas = pipeline.accumulated_canvas
                else:
                    # Tinta assincrona: ate a trabalhadora publicar a
                    # primeira mascara desta folha, o painel e a folha
                    # em branco. Rodar o modelo aqui, sincrono, era o
                    # congelamento que a trabalhadora existe para tirar.
                    canvas = view_module.blank_sheet()
                render_ui(status, canvas)
                ui_inputs = (status, canvas)

                key = ui_module._next_key()
                if key == ord("s"):
                    save_canvas_snapshot(canvas)
                handle_capture_key(key, frame)
                if key in (ord("q"), 27):
                    break
                continue

    finally:
        # Epilogo sempre roda: captura liberada, trabalhadora de
        # tinta parada, fila de PNG drenada e janela destruida
        # mesmo quando o laco morre por excecao. Sem o finally, um
        # erro no meio do laco vazava a camera aberta e a thread
        # daemon ficava lendo o dispositivo ate o processo sair.
        # Os `is not None` cobrem falha de setup antes de cada
        # recurso subir: o try abrange do primeiro load ao ultimo loop.
        if frame_source is not None:
            frame_source.stop()
        if detection_worker is not None:
            detection_worker.stop()
            # Aquisicoes que nenhum quadro do rastreador consumiu
            # (latest-wins ou fim de sessao) tambem viram evento.
            if telemetry is not None:
                for trace in detection_worker.take_traces():
                    telemetry.event(
                        "acquisition",
                        **{
                            **trace,
                            "frame_index": None,
                            "consumed_by_tracker": False,
                            "tracker_before": None,
                            "tracker_after": None,
                            "tracker_event": None,
                        },
                    )
        if hand_presence_worker is not None:
            hand_presence_worker.stop()
        if ink_worker is not None:
            ink_worker.stop()
        if ink_stream is not None:
            ink_stream.close()
        if hand_masker is not None:
            # O masker da trabalhadora fecha aqui, depois do join:
            # determinístico, nunca pelo __del__ do MediaPipe.
            hand_masker.close()
        if telemetry is not None:
            # O laço entrega os contadores antes do step de cada
            # quadro — as rejeições do último frame ainda entram
            # no retrato final (issue #105).
            if pipeline is not None:
                telemetry.update(**pipeline.stats())
            telemetry.close()
        if session_report is not None:
            session_report.close()
        if png_writer is not None:
            png_writer.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Build headless ou GUI indisponivel: nao ha o que destruir.
            pass


def main():
    parser = argparse.ArgumentParser(description="Draw, papel para o digital")
    parser.add_argument(
        "--source",
        default="0",
        help="indice da webcam local (0, 1...), URL do stream "
        "(ex.: http://192.168.0.10:8080/video), gravacao .drawcap, "
        "stream tcp://host[:porta] do ajudante ou camera do Windows "
        "via hvcam://, hvcam+reverse:// (com hvcam-host --reverse) "
        "ou hvcam+tcp://host. Padrao: 0",
    )
    parser.add_argument(
        "--helper-token-file",
        default=None,
        metavar="ARQUIVO",
        help="arquivo com o token do ajudante Windows para "
        "--source tcp:// (bridge/PROTOCOL.md). Sem ele, fontes "
        "tcp:// sao recusadas no handshake",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=None,
        metavar="PORTA",
        help="publica a tinta confirmada em 127.0.0.1:PORTA para o drawlink "
        "(bridge/STREAM.md; exige --stream-token-file). Desligado por padrao",
    )
    parser.add_argument(
        "--stream-token-file",
        default=None,
        metavar="ARQUIVO",
        help="arquivo com o token do stream de tinta (16..256 bytes)",
    )
    parser.add_argument(
        "--stream-record",
        default=None,
        metavar="ARQUIVO",
        help="grava o stream de tinta (mesma gramatica do socket, arquivo "
        "privado 0600) para replay",
    )
    parser.add_argument(
        "--markers",
        action="store_true",
        help="usa a folha-modelo impressa com marcadores ArUco",
    )
    parser.add_argument(
        "--neural",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="detector neural (ONNX) como primario, com o classico de "
        "reserva. Sem a flag, o padrao e models/draw_seg_v3_dyn.onnx "
        "quando o arquivo existe",
    )
    parser.add_argument(
        "--page-clean",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="modelo pagina->limpa (ONNX), a fonte da tinta. Sem "
        "caminho, usa models/page_clean_unet.onnx",
    )
    parser.add_argument(
        "--hand-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.task",
        help="modelo de mao (MediaPipe Hand Landmarker) para a porta "
        "de ociosidade e a oclusao no voto de tinta. Sem caminho, "
        "usa models/hand_landmarker.task quando o arquivo existe",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="valida modelos e camera e sai, sem abrir janela",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="abre em tela cheia, sem a moldura da janela (liga e desliga "
        "tambem na engrenagem)",
    )
    parser.add_argument("--save-dir", default="captures")
    parser.add_argument(
        "--rotate",
        type=int,
        choices=camera_module.INPUT_ROTATIONS,
        default=None,
        help=(
            "gira o quadro da camera na captura (180 = webcam invertida na "
            "mesa); sem a flag, detecta e gira automaticamente quando a "
            "mascara semantica de mao esta disponivel"
        ),
    )
    parser.add_argument(
        "--orientation",
        choices=config.ORIENTATIONS,
        default=None,
        help=(
            "orientacao da grade da pagina: paisagem para caderno aberto ou "
            "folha deitada; sem a flag, adapta automaticamente entre retrato e paisagem"
        ),
    )
    parser.add_argument(
        "--spread-mode",
        choices=("auto", "always", "never"),
        default=None,
        help=(
            "união de duas páginas: auto limita a calha a 10%%; always aceita "
            "calhas maiores quando tamanho e alinhamento combinam; never "
            "mantém os contornos separados. Requer detector neural de folha; "
            "não combina com --markers"
        ),
    )
    parser.add_argument(
        "--ink-vote",
        type=float,
        default=None,
        metavar="NIVEL",
        help=(
            "voto estrito da tinta ao vivo (InkConfig.strict_vote), 0,634-0,99: "
            "derruba a pauta que o page_clean v2 deixa passar, ao custo de "
            "afinar o traco (medido: 0,85 e o ponto equilibrado). So com a "
            "imagem bem exposta: com papel escuro quebra as letras"
        ),
    )
    parser.add_argument(
        "--ink-interval",
        type=float,
        default=None,
        metavar="SEGUNDOS",
        help=(
            "intervalo minimo entre passadas ativas da tinta "
            "(InkConfig.pass_interval_s; substitui o piso de 0,25 s, mantendo "
            "sonda ociosa e descanso). Com o page_clean INT8, 0,15 mediu 6,7 "
            "passadas/s em ~3 nucleos (simulado, sem a mao)"
        ),
    )
    parser.add_argument(
        "--legacy-ink",
        action="store_true",
        help=(
            "volta ao comportamento anterior ao preset de producao: voto "
            "legado, sem ganho, pauta, histerese, veto por movimento nem "
            "Detect-Wake, passadas a 0,25 s (InkConfig() e PipelineConfig "
            "sem os mecanismos); para comparar e depurar"
        ),
    )
    parser.add_argument(
        "--preset",
        choices=("sketch", "ruled", "technical"),
        default=None,
        help=(
            "perfil de produto da tinta ao vivo: sketch (esboço/lápis — "
            "fidelidade máxima: voto legado, sem remoção de pauta), ruled "
            "(caderno pautado — o preset de produção atual) ou technical "
            "(técnico/CAD — voto 0,92, vetores mais limpos para DXF). "
            "--ink-vote não combina (o perfil fixa o nível); --ink-interval "
            "segue como ajuste sobre o perfil"
        ),
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="carrega modelos divergentes ou fora do manifesto com "
        "aviso, em vez de recusar (equivale a DRAW_MODEL_STRICT=0)",
    )
    parser.add_argument(
        "--format",
        choices=list(config.FORMATS),
        default=None,
        help=(
            "formato da folha (padrao: a4); em retrato, a menos que combinado "
            "com --orientation paisagem"
        ),
    )
    args = parser.parse_args()
    if args.allow_unverified:
        manifest_module.allow_unverified()
        print(
            "AVISO: --allow-unverified ativo; modelos divergentes ou fora do "
            "manifesto serao carregados apenas com aviso."
        )
    try:
        orientation = "retrato" if args.orientation is None else args.orientation
        if args.markers and orientation != "retrato":
            parser.error(
                "--orientation paisagem nao combina com --markers (folha-modelo A4 em pe)"
            )
        config.apply_format(args.format or "a4", orientation)
    except ValueError as error:
        parser.error(str(error))
    # Validado ANTES do --check: o mesmo nivel invalido impede a sessao ao
    # vivo, entao o preflight nao pode sair com sucesso (review do PR #53).
    legacy_ink = args.legacy_ink or args.markers
    for flag, given in (
        ("--ink-vote", args.ink_vote is not None),
        ("--ink-interval", args.ink_interval is not None),
        ("--legacy-ink", args.legacy_ink),
        ("--preset", args.preset is not None),
    ):
        if args.markers and given:
            parser.error(
                f"{flag} nao combina com --markers (sem tinta no modo marcadores)"
            )
    if args.preset is not None:
        if args.legacy_ink:
            parser.error("--preset nao combina com --legacy-ink")
        if args.ink_vote is not None:
            parser.error(
                "--ink-vote nao combina com --preset (o perfil fixa o nivel de voto)"
            )
    try:
        if legacy_ink:
            ink_config = InkConfig()
            if args.ink_vote is not None:
                ink_config = InkConfig.strict_vote(args.ink_vote)
            if args.ink_interval is not None:
                ink_config = dataclasses.replace(
                    ink_config, pass_interval_s=args.ink_interval
                )
        else:
            if args.preset == "sketch":
                ink_config = InkConfig.sketch()
            elif args.preset == "technical":
                ink_config = InkConfig.technical()
            else:
                options = {}
                if args.ink_vote is not None:
                    options["vote_level"] = args.ink_vote
                if args.ink_interval is not None:
                    options["pass_interval_s"] = args.ink_interval
                ink_config = InkConfig.production(**options)
            if args.preset is not None:
                overrides = {}
                if args.ink_interval is not None:
                    overrides["pass_interval_s"] = args.ink_interval
                if overrides:
                    ink_config = dataclasses.replace(ink_config, **overrides)
    except ValueError as error:
        parser.error(str(error))
    # Marcadores nao tem detector neural nem tinta: preset legado, sem erro.
    if args.markers and args.spread_mode is not None:
        parser.error("--spread-mode nao combina com --markers (requer detector neural)")
    pipeline_config = PipelineConfig(
        ink=ink_config,
        ink_auto_gain=not legacy_ink,
        detect_wake=not legacy_ink,
        spread_mode="auto" if args.spread_mode is None else args.spread_mode,
    )
    if args.stream_port is not None and not args.stream_token_file:
        parser.error("--stream-port exige --stream-token-file")
    if args.markers and (args.stream_port is not None or args.stream_record):
        parser.error("--stream-port/--stream-record nao existem com --markers")
    if args.check:
        parsed = parse_source(args.source)
        sys.exit(run_check(parsed, helper_token_file=args.helper_token_file))
    neural_model = args.neural
    page_clean_model = args.page_clean
    hand_model = args.hand_model
    if args.markers and (
        neural_model is not None
        or page_clean_model is not None
        or hand_model is not None
    ):
        parser.error("--neural/--page-clean/--hand-model nao combinam com --markers")
    if neural_model is not None and not neural_model:
        parser.error("--neural sem caminho: passe o arquivo .onnx")
    if page_clean_model is not None and not page_clean_model:
        page_clean_model = None  # padrao: models/page_clean_unet.onnx
    if hand_model is not None and not hand_model:
        hand_model = None  # padrao: models/hand_landmarker.task
    rotation = 0 if args.rotate is None else args.rotate
    auto_rotate = args.rotate is None and not args.markers
    auto_orientation = args.orientation is None and not args.markers
    try:
        run(
            source=args.source,
            save_dir=args.save_dir,
            markers_mode=args.markers,
            neural_model=neural_model,
            hand_model=hand_model,
            page_clean_model=page_clean_model,
            helper_token_file=args.helper_token_file,
            fullscreen=args.fullscreen,
            rotation=rotation,
            auto_rotate=auto_rotate,
            auto_orientation=auto_orientation,
            spread_mode_explicit=args.spread_mode is not None,
            pipeline_config=pipeline_config,
            stream_port=args.stream_port,
            stream_token_file=args.stream_token_file,
            stream_record=args.stream_record,
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
