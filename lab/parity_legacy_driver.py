"""Lado legado do teste de paridade ponta a ponta.

Roda DENTRO do repo legado (PYTHONPATH apontado para ele): o pacote
`draw` deste processo é o antigo, com draw.loop.SheetInkLoop e os
adaptadores síncronos no protocolo de tuplas que ele espera. O lado
novo roda em lab/parity_e2e.py no draw-core, sobre o mesmo corpus e
os mesmos binários de modelo.

As 8 flags do conjunto validado (MIGRATION_SPEC, Tier A — commit
legado a95eec8 / docs/SIMULATOR.md) são ligadas AQUI, porque no
legado toda flag de comportamento nasce desligada e o default não é
a configuração validada; no draw-core elas são o default de
PipelineConfig/InkConfig.

Saída: out/telemetry.json (corners/present por frame + hand_samples,
amostras de presença com sinal real — zero denuncia ambiente sem
mediapipe/modelo) + out/canvas.npz (clean_mask>0 de cada frame). Com
--hand-model resolvendo um arquivo, masker e presença entram nos dois
lados como na produção. Uso:

  PYTHONPATH=/caminho/do/draw python lab/parity_legacy_driver.py \
      --corpus <dir|video.mp4> --out <dir> [--frames N] [--fps 2.0] \
      [--hand-model models/hand_landmarker.task]
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

VALIDATED_FLAGS = (
    "DRAW_INK_FOLLOW_MOVE",
    "DRAW_SHEET_COVERAGE",
    "DRAW_CANVAS_PERSIST",
    "DRAW_INK_SKIP_STALE",
    "DRAW_INK_REMOVAL",
    "DRAW_SEG_HAND_OCCLUSION",
    "DRAW_ANCHOR_ARBITRATION",
    "DRAW_ANCHOR_FOLLOW",
)
for _flag in VALIDATED_FLAGS:
    os.environ[_flag] = "1"


def _close_hand_masker(masker):
    """Fecha os landmarkers MediaPipe de um HandMasker.

    Sem fechamento explícito os finalizers do MediaPipe podem
    deadlockar na saída do processo e deixar o run_legacy esperando
    para sempre. O HandMasker legado não expõe close(): usa o método
    público quando existe e, caso contrário, fecha os dois landmarkers
    internos diretamente.
    """
    if masker is None:
        return
    close = getattr(masker, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
        return
    for name in ("_landmarker", "_video_landmarker"):
        landmarker = getattr(masker, name, None)
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass


def frame_sort_key(path):
    """Espelho de lab.corpus.clip.frame_sort_key — manter em sync.

    O driver é auto-contido (roda sob o PYTHONPATH do repo legado),
    então a chave se duplica aqui: a paridade alinha os dois lados
    pelo índice do frame, e o índice só significa o mesmo quadro se
    a ordem do diretório for a mesma. Sem ela, f10000.png (formato
    f%04d da captura, após o quadro 9999) pareava com f1001.png do
    lado novo.
    """
    digits = re.search(r"(\d+)$", path.stem)
    if digits is not None:
        return (0, int(digits.group(1)), path.name)
    return (1, 0, path.name)


def iter_frames(source, limit):
    """Frames de um diretório de imagens ou de um vídeo."""
    path = Path(source)
    if path.is_dir():
        files = sorted(
            (
                p
                for p in path.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            ),
            key=frame_sort_key,
        )
        for i, f in enumerate(files):
            if i >= limit:
                break
            frame = cv2.imread(str(f))
            if frame is not None:
                yield i, frame
        return
    capture = cv2.VideoCapture(str(path))
    try:
        index = 0
        while index < limit:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        capture.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--sheet-model", default="models/draw_seg_v3_dyn.onnx")
    ap.add_argument("--ink-model", default="models/page_clean_unet.onnx")
    ap.add_argument("--hand-model", default="models/hand_landmarker.task")
    args = ap.parse_args(argv)

    from draw.ink_layers import _PageCleanWorker
    from draw.loop import SheetInkLoop
    from draw.neural import NeuralSheetDetector
    from draw.sheet_track import _gate_info_from_trace

    from draw import hand as hand_module
    from draw import hand_presence as hand_presence_module
    from draw.app import _detect_sheet_combined_traced
    from draw.page_clean import PageCleanInk

    detector = NeuralSheetDetector(args.sheet_model)
    detector.load()
    page_clean = PageCleanInk(args.ink_model)
    page_clean.load()
    # Mão como na produção: masker do page-clean + presença do laço.
    # Sem o arquivo (ou sem mediapipe instalada) os dois lados rodam
    # sem o recurso e a telemetria marca a mão como não medida.
    hand_masker = (
        hand_module.HandMasker(args.hand_model)
        if hand_module.hand_model_available(args.hand_model)
        else None
    )
    presence_masker = (
        hand_module.HandMasker(args.hand_model)
        if hand_module.hand_model_available(args.hand_model)
        else None
    )
    presence = (
        hand_presence_module._MediaPipeHandPresence(presence_masker)
        if presence_masker is not None
        else None
    )

    class Clock:
        def __init__(self, dt):
            self.dt = dt
            self.t = 0.0

        def __call__(self):
            return self.t

    clock = Clock(1.0 / args.fps)

    class SyncDet:
        """Protocolo de tuplas do laço legado, execução síncrona."""

        def __init__(self):
            self._result = None
            self._seq = 0
            self.error = None
            self.traces = []

        def submit(self, frame, captured_at, frame_index=None):
            outcome = _detect_sheet_combined_traced(frame, detector)
            corners, verdict, trace = outcome[0], outcome[1], outcome[2]
            extras = outcome[3] if len(outcome) > 3 else None
            self._seq += 1
            result = (self._seq, corners, verdict, captured_at, 1.0)
            gate = _gate_info_from_trace(trace)
            if gate is not None:
                result = result + (gate,)
            if extras is not None:
                if gate is None:
                    result = result + (None,)
                result = result + (extras,)
            self._result = result
            self.traces.append(trace)

        def poll(self, after_seq):
            if self._result is None or self._result[0] <= after_seq:
                return None
            return self._result

        def take_traces(self, upto_result_seq=None):
            return []

    class SyncPresence:
        """Presença síncrona no contrato (seq, presença, moveu)."""

        def __init__(self, presence):
            self._presence = presence
            self._result = None
            self._seq = 0
            self.error = None
            self.real_samples = 0

        def submit(self, frame, timestamp_ms):
            self._seq += 1
            sample = self._presence.sample(frame, timestamp_ms)
            present, moved = (
                (sample.present, sample.moved) if hasattr(sample, "present") else sample
            )
            self._result = (self._seq, present, moved)
            if present is not None:
                self.real_samples += 1

        def poll(self, after_seq):
            if self._result is None or self._result[0] <= after_seq:
                return None
            return self._result

    det = SyncDet()
    presence_worker = SyncPresence(presence) if presence is not None else None
    worker = _PageCleanWorker(page_clean, hand_masker, clock=clock)
    loop = SheetInkLoop(
        detection_worker=det,
        ink_worker=worker,
        hand_presence_worker=presence_worker,
        hand_masker=hand_masker,
        clock=clock,
        log=lambda *_a: None,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records, canvases = [], []
    try:
        for idx, frame in iter_frames(args.corpus, args.frames):
            clock.t = idx * clock.dt
            loop.step(frame, clock.t * 1000.0)
            worker.pump()
            loop.collect_ink()
            if worker.error is not None:
                raise RuntimeError(
                    "trabalhadora de tinta legada falhou"
                ) from worker.error
            records.append(
                {
                    "frame": idx,
                    "present": loop.tracker.corners is not None,
                    "corners": (
                        None
                        if loop.tracker.corners is None
                        else np.asarray(loop.tracker.corners, float).tolist()
                    ),
                }
            )
            canvases.append(
                np.zeros((1, 1), bool)
                if loop.clean_mask is None
                else (np.asarray(loop.clean_mask) > 0)
            )
        hand_samples = presence_worker.real_samples if presence_worker else 0
        (out / "telemetry.json").write_text(
            json.dumps(
                {
                    "frames": records,
                    "flags": sorted(VALIDATED_FLAGS),
                    "hand_samples": hand_samples,
                },
                indent=1,
            )
        )
        np.savez_compressed(out / "canvas.npz", *canvases)
        print(json.dumps({"frames": len(records), "out": str(out)}))
    finally:
        _close_hand_masker(hand_masker)
        _close_hand_masker(presence_masker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
