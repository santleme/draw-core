"""Driver headless de mesa: fonte ao vivo -> Pipeline real -> dumps.

Port do diag_live_desk.py do legado para o lab do draw-core. A fonte
ao vivo — webcam local, hvcam://, .drawcap/tcp:// do ajudante ou
camera via rede — alimenta o mesmo Pipeline sincrono do eval/replay,
sem threads e com relogio de video. A cada --every quadros grava
scene_NNNNN.png (quadro + ancora detectada) e canvas_NNNNN.png
(mascara confirmada, 255 = tinta) no diretorio de saida, com uma
linha de metricas por dump no stdout. Ao sair grava session.json
com o resumo da corrida e, com mascara, canvas_final.png.

Saida padrao: captures/desk-<AAAAMMDD-HHMMSS>/ (captures/ esta no
.gitignore). Encerra com Ctrl+C ou --max-frames.

Uso:
  python -m lab.desk_capture                              # webcam local (indice 0)
  python -m lab.desk_capture --source hvcam:// --fps 30
  python -m lab.desk_capture --source sessao.drawcap --fps 30 --out /tmp/desk
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.eval_capture import run_pipeline  # noqa: E402


class _TappedCapture:
    """Envelope que memoriza o ultimo quadro lido e limita a corrida.

    run_pipeline nao entrega o quadro ao on_frame — o driver precisa
    dele para o scene_*.png. Alem disso conta os quadros lidos e mede
    o fps real de chegada: numa fonte ao vivo isso expoe camera
    travada ou hvcam aquem do negociado, que o fps nominal esconde.
    """

    def __init__(self, capture, max_frames=None):
        self._capture = capture
        self._max_frames = max_frames
        self.last_frame = None
        self.frames_read = 0
        self._t0 = None

    def isOpened(self):
        return self._capture.isOpened()

    def get(self, prop):
        # Fontes nao-cv2 (hvcam, ajudante) nao expõe get(): devolver
        # 0.0 faz o run_pipeline pedir --fps em vez de quebrar em
        # AttributeError.
        get = getattr(self._capture, "get", None)
        return get(prop) if get is not None else 0.0

    def read(self):
        if self._max_frames is not None and self.frames_read >= self._max_frames:
            return False, None
        ok, frame = self._capture.read()
        if not ok:
            return False, None
        if self._t0 is None:
            self._t0 = time.monotonic()
        self.frames_read += 1
        self.last_frame = frame
        return True, frame

    def real_fps(self):
        """Quadros lidos por segundo desde o primeiro — medido, nao nominal."""
        if not self.frames_read or self._t0 is None:
            return 0.0
        elapsed = time.monotonic() - self._t0
        return self.frames_read / elapsed if elapsed > 0 else 0.0

    def release(self):
        return self._capture.release()


class _DumpEvery:
    """on_frame do run_pipeline: scene/canvas a cada `every` quadros.

    Mesma leitura do diag_live_desk: a cena sai com a ancora do
    worker desenhada; o canvas sai so quando ha mascara confirmada.
    A linha de metricas traz presenca de ancora, tinta confirmada,
    passadas, sequencia do detector, miss do rastreador, ultimo
    evento, epoca do acumulador e o fps real medido na entrada.
    """

    def __init__(self, out_dir, tap, every=15, log=print, now=time.monotonic):
        if every < 1:
            raise ValueError("--every precisa ser >= 1")
        self.out_dir = Path(out_dir)
        self.tap = tap
        self.every = every
        self.log = log
        self.now = now
        self.t0 = now()

    def __call__(self, loop, worker, frame_index):
        if frame_index % self.every:
            return
        stamp = f"{frame_index:05d}"
        corners = worker.anchor_corners
        vis = self.tap.last_frame.copy()
        if corners is not None:
            quad = np.asarray(corners, np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [quad], True, (0, 255, 0), 2)
        cv2.imwrite(str(self.out_dir / f"scene_{stamp}.png"), vis)
        mask = loop.clean_mask
        conf = 0
        if mask is not None:
            confirmed = np.asarray(mask) > 0
            conf = int(confirmed.sum())
            cv2.imwrite(
                str(self.out_dir / f"canvas_{stamp}.png"),
                confirmed.astype(np.uint8) * 255,
            )
        self.log(
            f"f{frame_index:5d} t={self.now() - self.t0:5.1f}s "
            f"anchor={'sim' if corners is not None else 'NAO'} "
            f"conf={conf} passes={worker.stats()['passes_total']} "
            f"det={getattr(loop.detection_worker, 'sequence', None)} "
            f"miss={getattr(loop.tracker, '_frames_since_seen', None)} "
            f"evt={loop.last_tracker_event} "
            f"epoch={worker.canvas_epoch} fps={self.tap.real_fps():.1f}"
        )


def run(
    capture,
    *,
    out_dir,
    every=15,
    fps=None,
    max_frames=None,
    neural_model=None,
    page_clean_model=None,
    hand_model=None,
    verify="manifest",
    log=print,
):
    """Dirige o Pipeline real sobre a fonte ate ela fechar ou Ctrl+C.

    A captura continua do chamador (release() fica fora daqui).
    Devolve o resumo de run_pipeline, ou None quando interrompido;
    em ambos os casos session.json fica no out_dir — os campos
    volumosos (mask, anchors, registrations) saem como contagens.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tap = _TappedCapture(capture, max_frames=max_frames)
    dump = _DumpEvery(out_dir, tap, every=every, log=log)
    log(f"saida em {out_dir}")
    result = None
    interrupted = False
    try:
        result = run_pipeline(
            tap,
            fps=fps,
            neural_model=neural_model,
            page_clean_model=page_clean_model,
            hand_model=hand_model,
            on_frame=dump,
            verify=verify,
        )
    except KeyboardInterrupt:
        interrupted = True
    if result is None:
        summary = {"status": "interrupted", "frames": tap.frames_read}
    else:
        summary = {
            key: value
            for key, value in result.items()
            if key
            not in (
                "mask",
                "anchors",
                "registrations",
                "registration_history",
                "corners",
            )
        }
        summary["anchor_count"] = len(result["anchors"])
        summary["registration_count"] = len(result["registrations"])
        summary["registration_history_count"] = len(result["registration_history"])
        summary["anchor_corners"] = (
            None
            if result["corners"] is None
            else np.asarray(result["corners"]).tolist()
        )
        if result["mask"] is not None:
            cv2.imwrite(
                str(out_dir / "canvas_final.png"),
                (np.asarray(result["mask"]) > 0).astype(np.uint8) * 255,
            )
    summary["interrupted"] = interrupted
    serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "session.json").write_text(serialized, encoding="utf-8")
    return result


def _default_out_dir():
    stamp = time.strftime("desk-%Y%m%d-%H%M%S", time.localtime())
    return Path("captures") / stamp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="0",
        help="indice de webcam, .drawcap, tcp://, hvcam:// ou URL (default 0)",
    )
    parser.add_argument(
        "--every", type=int, default=15, help="intervalo de dump em quadros"
    )
    parser.add_argument(
        "--out", type=Path, help="diretorio de saida (default captures/desk-<ts>)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="fps nominal; exigido por fontes sem get() (hvcam, ajudante)",
    )
    parser.add_argument(
        "--max-frames", type=int, help="encerra a corrida apos N quadros"
    )
    parser.add_argument("--neural-model")
    parser.add_argument("--page-clean-model")
    parser.add_argument(
        "--hand-model", help="modelo .task de mao (opcional, como no app)"
    )
    parser.add_argument(
        "--helper-token-file", type=Path, help="token do ajudante (fontes tcp://)"
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="desliga a verificacao sha256 dos modelos",
    )
    args = parser.parse_args(argv)
    if args.every < 1:
        parser.error("--every precisa ser >= 1")

    from draw.camera import open_capture, parse_source

    out_dir = args.out or _default_out_dir()
    source = parse_source(args.source)
    try:
        capture = open_capture(source, helper_token_file=args.helper_token_file)
        try:
            result = run(
                capture,
                out_dir=out_dir,
                every=args.every,
                fps=args.fps,
                max_frames=args.max_frames,
                neural_model=args.neural_model,
                page_clean_model=args.page_clean_model,
                hand_model=args.hand_model,
                verify=None if args.allow_unverified else "manifest",
            )
        finally:
            capture.release()
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as error:
        parser.exit(2, f"erro: {error}\n")
    if result is None:
        print(f"interrompido; dumps parciais em {out_dir}")
    else:
        print(
            f"fim: {result['frames']} quadros, {result['ink_passes']} passadas, "
            f"{result['detections']} deteccoes, status={result['status']}, "
            f"canvas_shape={result['canvas_shape']} -> {out_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
