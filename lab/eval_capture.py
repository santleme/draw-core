"""Avalia canvas vs GT ou dirige o laço real do Draw sem interface.

O GT entra apenas depois da captura: não alimenta detector, warp ou tinta.
O modo headless usa tempo de vídeo, detecção síncrona e pump() da worker
real; mede qualidade funcional offline, não latência/throughput de webcam.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_DEFAULT_SHEET_MODEL = (
    Path(__file__).resolve().parents[1] / "models" / "draw_seg_v3_dyn.onnx"
)


def _sheet_model_path(explicit=None):
    # Mesma precedência de draw.app.run: argumento > instalado.
    if explicit is not None:
        return str(explicit)
    if _DEFAULT_SHEET_MODEL.exists():
        return str(_DEFAULT_SHEET_MODEL)
    return None


def pixel_metrics(prediction, truth):
    """Comparação exata, sem dilatação, tolerância ou alinhamento por conteúdo.

    Denominador vazio vale 1 (nenhum erro desse tipo); IoU de duas
    máscaras vazias vale 1. As contagens tornam a convenção auditável.
    """
    prediction, truth = np.asarray(prediction) > 0, np.asarray(truth) > 0
    if prediction.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("predição e GT precisam ser máscaras 2D do mesmo tamanho")
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_pixels": tp + fp,
        "gt_pixels": tp + fn,
    }


def compare_masks(prediction, truth, stroke_masks=(), *, stroke_recall=0.5):
    """Relatório por movimento de caneta: captured, partial, lost, unscorable.

    Recall é calculado em toda a geometria de cada traço, inclusive
    interseções; interseções podem ser explicadas por mais de um traço.
    Não é uma inferência causal de qual passada viu qual traço.
    """
    if not math.isfinite(stroke_recall) or not 0 < stroke_recall <= 1:
        raise ValueError("stroke_recall deve estar em (0, 1]")
    result = pixel_metrics(prediction, truth)
    result["stroke_recall_threshold"] = stroke_recall
    result["strokes"] = []
    for index, mask in enumerate(stroke_masks):
        metrics = pixel_metrics(prediction, mask)
        status = (
            "unscorable"
            if metrics["gt_pixels"] == 0
            else "captured"
            if metrics["recall"] >= stroke_recall
            else "partial"
            if metrics["tp"]
            else "lost"
        )
        result["strokes"].append(
            {
                "index": index,
                "status": status,
                "recall": metrics["recall"],
                "captured_pixels": metrics["tp"],
                "gt_pixels": metrics["gt_pixels"],
            }
        )
    result["stroke_counts"] = {
        status: sum(s["status"] == status for s in result["strokes"])
        for status in ("captured", "partial", "lost", "unscorable")
    }
    return result


class _VideoClock:
    now = 0.0

    def __call__(self):
        return self.now


class _SynchronousDetector:
    """Só adapta a execução; a decisão é a mesma de draw.app."""

    error = None

    def __init__(self, neural):
        from draw.app import _detect_sheet_combined_traced

        self.detect = lambda frame: _detect_sheet_combined_traced(frame, neural)
        self.result = None
        self.sequence = 0

    def submit(self, frame, captured_at, frame_index=None):
        from draw.contracts import DetectionResult
        from draw.tracker import _gate_info_from_trace

        outcome = self.detect(frame)
        corners, verdict, trace = outcome[0], outcome[1], outcome[2]
        extras = outcome[3] if len(outcome) > 3 else None
        self.sequence += 1
        self.result = DetectionResult(
            sequence=self.sequence,
            corners=corners,
            verdict=verdict,
            captured_at=captured_at,
            detect_ms=0.0,
            gate_info=_gate_info_from_trace(trace),
            coverage_mask=(extras.get("coverage_mask") if extras else None),
            hand_mask=(extras.get("hand_mask") if extras else None),
        )

    def poll(self, after_seq):
        if self.result is not None and self.result.sequence > after_seq:
            return self.result
        return None

    def take_traces(self, upto_result_seq=None):
        return []


class _SynchronousHandPresence:
    """Mesmo protocolo da worker de mão, com amostragem real síncrona."""

    def __init__(self, presence):
        self.presence = presence
        self.result = None
        self.sequence = 0
        self.available_samples = 0
        self.error = None

    def submit(self, frame, timestamp_ms):
        try:
            sample = self.presence.sample(frame, timestamp_ms)
        except Exception as error:
            # Como a worker do app: o laço degrada para sinal indisponível.
            self.error = error
            return
        self.sequence += 1
        self.available_samples += int(sample.present is not None)
        self.result = (self.sequence, sample.present, sample.moved)

    def poll(self, after_seq):
        if self.result is not None and self.result[0] > after_seq:
            return self.result
        return None


def run_pipeline(
    capture,
    *,
    fps=None,
    neural_model=None,
    page_clean_model=None,
    hand_model=None,
    on_frame=None,
    verify="manifest",
    pipeline_config=None,
):
    """Consome VideoCapture ou SyntheticCapture via Pipeline real.

    Não inicia threads, não descarta frames, não lê propriedades de GT da
    fonte. Captura continua pertencendo ao chamador, que deve chamar
    release(). ``on_frame(pipeline, worker, frame_index)`` opcional roda
    depois do pump/collect de cada quadro — a auditoria por fase usa isso
    para fotografar clean_mask/provisional/âncora no fim de cada fase.

    verify é o callable de integridade dos modelos: o padrão "manifest"
    injeta draw.infra.manifest.verify_model (sha256 + tamanho contra
    assets/manifest.json); None desliga, para bancadas e testes com
    modelos fora do manifesto. ``pipeline_config`` (padrão: base legada
    sem ganho, sem Detect-Wake e com InkConfig(), como em sim_writer;
    ``desk_capture.main`` injeta ``PipelineConfig()``) alinha o Pipeline
    e a PageCleanWorker no mesmo preset.
    """
    from draw import config, hand
    from draw.contracts import FramePacket
    from draw.hand_presence import make_hand_presence
    from draw.infra.manifest import verify_model
    from draw.ink.config import InkConfig
    from draw.ink.worker import PageCleanWorker
    from draw.page_clean import PageCleanInk
    from draw.perception import NeuralSheetDetector
    from draw.pipeline import Pipeline
    from draw.settings import PipelineConfig

    if verify == "manifest":
        verify = verify_model
    if pipeline_config is None:
        pipeline_config = PipelineConfig(
            ink=InkConfig(), ink_auto_gain=False, detect_wake=False
        )

    if not capture.isOpened():
        raise ValueError("não foi possível abrir a fonte de vídeo")
    fps = float(fps if fps is not None else capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("vídeo sem FPS válido; informe --fps")
    neural_model = _sheet_model_path(neural_model)
    neural = NeuralSheetDetector(neural_model, verify=verify) if neural_model else None
    if neural is not None:
        neural.load()
    clock = _VideoClock()
    hand_presence_worker = None
    page_clean = PageCleanInk(page_clean_model, verify=verify)
    page_clean.load()
    ecc_hand_masker = (
        hand.HandMasker(hand_model, verify=verify)
        if hand.hand_model_available(hand_model)
        else None
    )
    visibility_configured = ecc_hand_masker is not None
    worker = PageCleanWorker(
        page_clean, ecc_hand_masker, clock=clock, config=pipeline_config.ink
    )
    presence = make_hand_presence(hand_model, verify=verify)
    if presence is not None:
        hand_presence_worker = _SynchronousHandPresence(presence)
    detector = _SynchronousDetector(neural)
    loop = Pipeline(
        detection_worker=detector,
        ink_worker=worker,
        hand_presence_worker=hand_presence_worker,
        hand_masker=ecc_hand_masker,
        clock=clock,
        log=lambda *_args: None,
        config=pipeline_config,
    )
    frame_index = 0
    anchors = []
    registrations = []
    registration_history = []
    submit_frames = []
    canvas_epoch = worker.canvas_epoch
    _submit = worker.submit

    def _tagged_submit(page):
        # O frame que originou o job consumido: a pose verdadeira
        # da passada mora no bundle GT sob este índice — latest-
        # wins garante que o pendente é sempre o último submit.
        submit_frames.append(frame_index)
        return _submit(page)

    worker.submit = _tagged_submit
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            clock.now = frame_index / fps
            outcome = loop.step(
                FramePacket(
                    image=frame, captured_at=clock.now, sequence=frame_index + 1
                )
            )
            passes_before = worker.stats()["passes_total"]
            pumped = worker.pump()
            if pumped and outcome is None:
                outcome = loop.collect_ink()
            if worker.canvas_epoch != canvas_epoch:
                # O acumulador foi zerado (timeout de identidade
                # ou página nova): os registros anteriores
                # projetavam tinta que não existe mais — saem de
                # ``registrations``, mas ficam na history marcados
                # pela época: checkpoints anteriores ao reset ainda
                # precisam da geometria que produziu o canvas deles.
                registrations.clear()
                canvas_epoch = worker.canvas_epoch
            if pumped and worker.stats()["passes_total"] != passes_before:
                # A geometria de CADA passada que votou: a tinta
                # cai na grade por X = P(âncora)·P(reg)⁻¹·P(sub)·H,
                # com H a pose verdadeira da folha no frame —
                # P(sub)·H leva a página ao warp que o modelo viu
                # e o registro à âncora. Passadas abstidas vêm com
                # last_registration None e não projetam GT.
                registration = worker.last_registration
                if registration is not None:
                    anchor, reg_corners, sub_corners = registration
                    if anchor is None:
                        anchor = reg_corners
                    if anchor is not None:
                        anchor = np.asarray(anchor, np.float32)
                        anchors.append(anchor.copy())
                        record = {
                            "frame": (submit_frames[-1] if submit_frames else None),
                            "epoch": worker.canvas_epoch,
                            "anchor": anchor.copy(),
                            "quad": (
                                None
                                if reg_corners is None
                                else np.asarray(reg_corners, np.float32).copy()
                            ),
                            "submitted": (
                                None
                                if sub_corners is None
                                else np.asarray(sub_corners, np.float32).copy()
                            ),
                        }
                        registrations.append(record)
                        registration_history.append(record)
            if worker.error is not None:
                raise RuntimeError("worker de tinta falhou") from worker.error
            if outcome is not None:
                raise RuntimeError(f"laço do Draw encerrou: {outcome}")
            if on_frame is not None:
                on_frame(loop, worker, frame_index)
            frame_index += 1
    finally:
        # Os landmarkers (MediaPipe) fecham aqui, não pelo __del__:
        # num vídeo longo eles vazavam até o fim do processo.
        if ecc_hand_masker is not None:
            ecc_hand_masker.close()
        if presence is not None:
            presence.close()
    if frame_index == 0:
        raise ValueError("fonte de vídeo não contém frames decodificáveis")
    mask = None if loop.clean_mask is None else loop.clean_mask.copy()
    corners = worker.anchor_corners
    passes = worker.stats()["passes_total"]
    return {
        "mask": mask,
        "anchors": anchors,
        "registrations": registrations,
        "registration_history": registration_history,
        "corners": None if corners is None else np.asarray(corners).copy(),
        "frames": frame_index,
        "fps": fps,
        "ink_passes": passes,
        "detections": detector.sequence,
        "status": "ok" if mask is not None else "no_canvas",
        "mode": "page_clean",
        "sheet_detector": "neural+classic" if neural is not None else "classic",
        "canvas_shape": [config.PAPER_H, config.PAPER_W],
        "scheduler": "synchronous_video_clock",
        "hand_visibility_configured": visibility_configured,
        "hand_presence_samples": (
            hand_presence_worker.sequence if hand_presence_worker is not None else 0
        ),
        "hand_presence_available_samples": (
            hand_presence_worker.available_samples
            if hand_presence_worker is not None
            else 0
        ),
    }


def load_canvas(path, kind=None):
    path = Path(path)
    kind = kind or ("mask" if path.suffix.lower() == ".npy" else "image")
    data = (
        np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npy"
        else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    )
    if data is None:
        raise ValueError(f"não foi possível ler {path}")
    if data.ndim == 3 and kind == "image":
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("canvas precisa ser uma imagem/máscara 2D finita")
    if kind == "image":
        return data < 128
    return data >= 0.5 if np.issubdtype(data.dtype, np.floating) else data > 0


def evaluate_canvas(
    prediction,
    gt_path,
    *,
    canvas_space="page",
    corners=None,
    anchors=None,
    registrations=None,
    stroke_recall=0.5,
):
    """Alinha GT por homografia conhecida, nunca por ajuste ao desenho.

    page sem corners pressupõe retificação ideal (bundle necessário).
    `registrations` recebe o registro de CADA passada do worker que
    votou — dicts com "anchor", "quad" (quad usado no registro),
    "submitted" (quad que retificou o warp) e "frame" (índice no
    vídeo): a grade do canvas é indexada pela retificação da âncora,
    mas cada evidência entra pela cadeia
    P(âncora)·P(quad)⁻¹·P(sub)·H da SUA passada, com H a pose
    verdadeira da folha naquele frame (campo "quads" do bundle;
    sem ele o quad submetido a estima e o termo colapsa na escala
    página→grade). O GT é projetado pela mesma cadeia e unido sobre
    as passadas. `anchors`/`corners` isolados e tuplas (âncora,
    quad) viram registros sem submetido/frame — retificação
    canônica é a leitura honesta de um canvas salvo. scene exige a
    mesma grade de câmera que ink_full. NPY simples não tem traços.
    """
    from draw.paper import perspective_matrix

    gt_path = Path(gt_path)
    bundle = None
    if gt_path.suffix.lower() == ".npz":
        with np.load(gt_path, allow_pickle=False) as archive:
            bundle = {key: archive[key] for key in archive.files}
        if int(bundle["version"]) != 1:
            raise ValueError("versão de GT não suportada")
        # A avaliação do fim do vídeo pede a tinta no estado final
        # (pós-borracha/redesenho): ink_full guarda TODOS os traços e
        # só serve à métrica por traço — usar como verdade puniria um
        # pipeline que apagou corretamente.
        truth = bundle["ink_final"] if "ink_final" in bundle else bundle["ink_full"]
    else:
        truth = np.load(gt_path, allow_pickle=False)
    if truth.ndim != 2:
        raise ValueError("ink_full precisa ser máscara 2D")
    # Sem canvas adquirido: não inventa geometria; todo o GT é perdido.
    if prediction is None:
        prediction = np.zeros_like(truth)
        canvas_space = "scene"
    prediction = np.asarray(prediction)
    if prediction.ndim != 2:
        raise ValueError("predição precisa ser uma máscara 2D")
    height, width = prediction.shape
    # Pose final da folha: a avaliação é do estado em que o vídeo
    # termina — move/lift/return fazem a pose inicial divergir da
    # âncora detectada no fim. page_to_scene_final existe sempre;
    # os mapas densos só existem quando a folha curva.
    page_to_scene_key = (
        "page_to_scene_final"
        if bundle is not None and "page_to_scene_final" in bundle
        else "page_to_scene"
    )
    map_tag = (
        "final_" if bundle is not None and "scene_to_page_final_x" in bundle else ""
    )
    scene_map = (
        (
            bundle[f"scene_to_page_{map_tag}x"],
            bundle[f"scene_to_page_{map_tag}y"],
        )
        if bundle is not None and f"scene_to_page_{map_tag}x" in bundle
        else None
    )
    page_truth = (
        (
            bundle["page_ink_final"]
            if "page_ink_final" in bundle
            else bundle["page_ink_full"]
        )
        if bundle is not None
        else None
    )

    def project(mask, matrix, page_matrix):
        """Projeta máscara de página para a grade do canvas/cena.

        Folha curvada exporta o mapa denso cena→página da sua pose —
        homografia não descreve a dobra: página→cena vira remap e
        cena→canvas segue a âncora detectada. Sem âncora (ideal_page)
        ou sem mapa, o caminho é a homografia de sempre.
        """
        if scene_map is not None and (canvas_space == "scene" or matrix is not None):
            scene = cv2.remap(
                mask,
                scene_map[0],
                scene_map[1],
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            if canvas_space == "scene":
                return scene
            return cv2.warpPerspective(
                scene, matrix, (width, height), flags=cv2.INTER_NEAREST
            )
        return cv2.warpPerspective(
            mask, page_matrix, (width, height), flags=cv2.INTER_NEAREST
        )

    page_matrix = None
    alignment = "scene"
    if canvas_space == "scene":
        if prediction.shape != truth.shape:
            raise ValueError("canvas da cena e ink_full têm grades diferentes")
        if bundle is not None:
            # Em scene o GT ja vem na grade da camera; page_matrix
            # so serve aos tracos individuais (strokes()).
            page_matrix = bundle[page_to_scene_key]
            stroke_mats = [None]
            stroke_page_mats = [page_matrix]
    elif canvas_space == "page":
        # Registros das passadas que VOTARAM: a tinta confirmada sob
        # a passada i ficou na posição X_i·p da grade, com
        #   X_i = P(âncora_i)·P(reg_i)⁻¹ · P(sub_i)·H_i
        # — a matriz de registro daquela passada (a mesma do worker)
        # sobre o mapa página→warp-submetido. H_i é a pose
        # verdadeira da folha no frame da passada (bundle["quads"]);
        # sem ela o quad submetido a estima e P(sub)·H colapsa na
        # escala página→grade. Registros/tuplas antigas sem
        # submetido/frame e âncoras soltas seguem o caminho
        # estimado, como antes.
        records = []
        for record in registrations or ():
            if isinstance(record, dict):
                records.append(
                    (
                        record.get("anchor"),
                        record.get("quad"),
                        record.get("submitted"),
                        record.get("frame"),
                    )
                )
            else:
                anchor, reg = record
                records.append((anchor, reg, None, None))
        if not records:
            for anchor in anchors or ():
                records.append((anchor, anchor, None, None))
            if corners is not None:
                records.append((corners, corners, None, None))
        true_quads = (
            np.asarray(bundle["quads"], np.float32)
            if bundle is not None and "quads" in bundle
            else None
        )
        # Dedup por bytes: vídeo estático repete a mesma geometria
        # a cada passada e uma projeção por repetição seria
        # desperdício — o frame só distingue quando a pose
        # verdadeira do bundle entra na projeção.
        seen = {}
        for anchor, reg, sub, frame in records:
            key = (
                None if anchor is None else np.asarray(anchor, np.float32).tobytes(),
                None if reg is None else np.asarray(reg, np.float32).tobytes(),
                None if sub is None else np.asarray(sub, np.float32).tobytes(),
                frame if true_quads is not None else None,
            )
            seen.setdefault(key, (anchor, reg, sub, frame))
        records = list(seen.values())
        if records:
            destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            truth_src = page_truth if bundle is not None else truth
            page_to_grid = np.diag(
                [
                    width / truth_src.shape[1],
                    height / truth_src.shape[0],
                    1.0,
                ]
            )
            page_corners_px = np.float32(
                [
                    [0, 0],
                    [truth_src.shape[1], 0],
                    [truth_src.shape[1], truth_src.shape[0]],
                    [0, truth_src.shape[0]],
                ]
            )
            projections = []
            stroke_mats = []
            stroke_page_mats = []
            for anchor, reg, sub, frame in records:
                for quad in (anchor, reg, sub):
                    if quad is not None and (
                        np.asarray(quad).shape != (4, 2) or not np.isfinite(quad).all()
                    ):
                        raise ValueError(
                            "corners precisa ter formato (4, 2) e valores finitos"
                        )
                if anchor is None or reg is None:
                    # Sem um lado do registro a evidência entra na
                    # própria geometria — a grade é a da passada.
                    reg_matrix = np.eye(3, dtype=np.float64)
                else:
                    reg_matrix = perspective_matrix(
                        anchor, destination
                    ) @ np.linalg.inv(perspective_matrix(reg, destination))
                if (
                    sub is not None
                    and true_quads is not None
                    and frame is not None
                    and 0 <= frame < len(true_quads)
                    and np.asarray(true_quads[frame]).shape == (4, 2)
                    and np.isfinite(true_quads[frame]).all()
                ):
                    # Pose verdadeira do frame: a página entra no
                    # warp pelo quad SUBMETIDO aplicado sobre a
                    # pose real — viés de detecção vira medida.
                    input_map = perspective_matrix(
                        sub, destination
                    ) @ perspective_matrix(page_corners_px, true_quads[frame])
                else:
                    # Estimativa pelo quad submetido: P(sub)·H
                    # colapsa na escala página→grade.
                    input_map = page_to_grid
                page_matrix = reg_matrix @ input_map
                stroke_mats.append(None)
                stroke_page_mats.append(page_matrix)
                projections.append(
                    cv2.warpPerspective(
                        truth_src,
                        page_matrix,
                        (width, height),
                        flags=cv2.INTER_NEAREST,
                    )
                )
            truth = (
                projections[0]
                if len(projections) == 1
                else np.maximum.reduce(projections)
            )
            alignment = "per_frame_anchors" if len(records) > 1 else "detected_anchor"
        elif bundle is not None:
            ph, pw = bundle["page_ink_full"].shape
            page_matrix = np.diag([width / pw, height / ph, 1.0])
            stroke_mats = [None]
            stroke_page_mats = [page_matrix]
            truth = project(page_truth, None, page_matrix)
            alignment = "ideal_page"
        else:
            raise ValueError("GT NPY em espaço de página exige --corners; use o bundle")
    else:
        raise ValueError("canvas_space precisa ser page ou scene")

    def strokes():
        if bundle is None:
            return
        shape = bundle["page_ink_full"].shape
        points, offsets = bundle["points"], bundle["offsets"]
        for index, thickness in enumerate(bundle["thickness"]):
            mask = np.zeros(shape, np.uint8)
            geo = points[offsets[index] : offsets[index + 1]].astype(np.int32)
            cv2.polylines(mask, [geo], False, 255, int(thickness))
            # Apenas a parte do traço que sobrevive ao estado final
            # conta: apagado pela borracha não é predição esperada.
            if page_truth is not None:
                mask = mask & page_truth
            projected = [
                project(mask, m, pm)
                for m, pm in zip(stroke_mats, stroke_page_mats, strict=True)
            ]
            yield (
                projected[0] if len(projected) == 1 else np.maximum.reduce(projected)
            )

    report = compare_masks(prediction, truth, strokes(), stroke_recall=stroke_recall)
    report.update(
        alignment=alignment,
        canvas_space=canvas_space,
        shape=[height, width],
        per_stroke_available=bundle is not None,
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--canvas", type=Path)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--save-canvas", type=Path, help="máscara .npy + .corners.npy")
    parser.add_argument("--canvas-space", choices=("page", "scene"), default="page")
    parser.add_argument("--canvas-kind", choices=("mask", "image"))
    parser.add_argument("--corners", type=Path, help="âncora real (4,2), .npy")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--stroke-recall", type=float, default=0.5)
    parser.add_argument("--neural-model")
    parser.add_argument("--page-clean-model")
    parser.add_argument(
        "--hand-model", help="modelo .task de mão (opcional, como no app)"
    )
    args = parser.parse_args(argv)
    try:
        pipeline = None
        registrations = None
        corners = (
            None if args.corners is None else np.load(args.corners, allow_pickle=False)
        )
        if args.video:
            if args.corners or args.canvas_space != "page":
                raise ValueError("--video usa a âncora detectada, em espaço page")
            capture = cv2.VideoCapture(str(args.video))
            try:
                # Adaptadores opcionais do app podem emitir avisos. Mantém
                # stdout como JSON, inclusive sem MediaPipe instalado.
                with contextlib.redirect_stdout(sys.stderr):
                    pipeline = run_pipeline(
                        capture,
                        fps=args.fps,
                        neural_model=args.neural_model,
                        page_clean_model=args.page_clean_model,
                        hand_model=args.hand_model,
                    )
            finally:
                capture.release()
            prediction = pipeline.pop("mask")
            corners = pipeline.pop("corners")
            anchors = pipeline.pop("anchors")
            registrations = pipeline.pop("registrations")
            history = pipeline.pop("registration_history", [])
            pipeline["anchor_corners"] = None if corners is None else corners.tolist()
            pipeline["anchor_count"] = len(anchors)
            pipeline["registration_count"] = len(registrations)
            pipeline["registration_history_count"] = len(history)
        else:
            prediction = load_canvas(args.canvas, args.canvas_kind)
        report = evaluate_canvas(
            prediction,
            args.gt,
            canvas_space=args.canvas_space,
            corners=corners,
            registrations=registrations,
            stroke_recall=args.stroke_recall,
        )
        if pipeline is not None:
            report["pipeline"] = pipeline
        if args.save_canvas:
            if prediction is None:
                raise ValueError("nenhum canvas adquirido para salvar")
            if args.save_canvas.suffix.lower() != ".npy":
                raise ValueError("--save-canvas exige extensão .npy")
            args.save_canvas.parent.mkdir(parents=True, exist_ok=True)
            # astype(uint8) ANTES do *255 truncava probabilidades a 0/1;
            # binariza primeiro, depois escala para 0/255.
            np.save(
                args.save_canvas,
                (np.asarray(prediction) > 0).astype(np.uint8) * 255,
            )
            if corners is not None:
                np.save(args.save_canvas.with_suffix(".corners.npy"), corners)
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as error:
        parser.exit(2, f"erro: {error}\n")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
