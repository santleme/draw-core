"""Driver headless de mesa: fonte ao vivo -> Pipeline real -> dumps.

Port do diag_live_desk.py do legado para o lab do draw-core. A fonte
ao vivo — webcam local, hvcam://, .drawcap/tcp:// do ajudante ou
camera via rede — alimenta o mesmo Pipeline sincrono do eval/replay,
sem threads e com relogio de video. A cada --every quadros grava
scene_NNNNN.png (quadro + ancora detectada) e canvas_NNNNN.png
(mascara confirmada, 255 = tinta) no diretorio de saida, com uma
linha de metricas por dump no stdout. Ao sair grava session.json
com o resumo da corrida e, com mascara, canvas_final.png — inclusive
na saida por Ctrl+C, usando a ultima mascara publicada.

Saida padrao: captures/desk-<AAAAMMDD-HHMMSS>/ (captures/ esta no
.gitignore). Um --out que ja contem artefatos de outra corrida e
recusado — sessoes nao se misturam e evidencia anterior nao e
apagada. Fonte ao vivo tolera ate _LIVE_READ_RETRIES leituras falhas
seguidas (mesmo orcamento do app); esgotado, o desfecho e erro —
timeout de grab nunca encerra a corrida como "ok". Encerra com
Ctrl+C ou --max-frames.

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

# Mesmo orcamento do app antes de declarar a fonte morta
# (_CAPTURE_MAX_FAILURES em draw.camera).
_LIVE_READ_RETRIES = 15


def _write_png(path, image):
    """imwrite checado: False silencioso vira OSError nomeando o destino."""
    if not cv2.imwrite(str(path), image):
        raise OSError(f"falha ao gravar {path}")


def _live_retries(source):
    """Fontes ao vivo toleram leituras falhas transitorias; arquivo
    (.drawcap, video) termina de verdade no primeiro False."""
    if isinstance(source, int):
        return _LIVE_READ_RETRIES
    return 0 if Path(str(source)).is_file() else _LIVE_READ_RETRIES


class _TappedCapture:
    """Envelope que memoriza o ultimo quadro lido e limita a corrida.

    run_pipeline nao entrega o quadro ao on_frame — o driver precisa
    dele para o scene_*.png. Conta os quadros lidos e mede a taxa de
    ENTREGA da fonte (tempo dentro de read(), sem o trabalho do
    pipeline): uma queda ali e fonte travada ou hvcam aquem do
    negociado, nao processamento. Em fonte viva, uma leitura falha e
    transitória — o grab seguinte pode vir —, entao read() retenta
    ate `live_retries` falhas seguidas e so entao desiste, com erro,
    nunca fingindo fim de fluxo.
    """

    def __init__(self, capture, max_frames=None, live_retries=0, now=time.monotonic):
        self._capture = capture
        self._max_frames = max_frames
        self._live_retries = live_retries
        self._now = now
        self.last_frame = None
        self.frames_read = 0
        self._io_seconds = 0.0

    def isOpened(self):
        return self._capture.isOpened()

    def get(self, prop):
        # Fontes nao-cv2 (hvcam, ajudante) nao expõe get(): devolver
        # 0.0 faz o run_pipeline pedir --fps em vez de quebrar em
        # AttributeError.
        get = getattr(self._capture, "get", None)
        return get(prop) if get is not None else 0.0

    def _read_once(self):
        started = self._now()
        ok, frame = self._capture.read()
        self._io_seconds += self._now() - started
        return ok, frame

    def read(self):
        if self._max_frames is not None and self.frames_read >= self._max_frames:
            return False, None
        ok, frame = self._read_once()
        failures = 0 if ok else 1
        while not ok and failures < self._live_retries:
            ok, frame = self._read_once()
            failures += 0 if ok else 1
        if not ok:
            if self._live_retries:
                raise RuntimeError(
                    f"fonte ao vivo parou de entregar: {failures} leituras "
                    f"falhas seguidas apos {self.frames_read} quadros"
                )
            return False, None
        self.frames_read += 1
        self.last_frame = frame
        return True, frame

    def real_fps(self):
        """Quadros entregues por segundo de read() — a taxa da fonte.

        O tempo do pipeline entre leituras (inferencia, imwrite) fica
        fora do denominador; o que se mede e so o que a fonte demora
        para entregar.
        """
        if not self.frames_read or self._io_seconds <= 0:
            return 0.0
        return self.frames_read / self._io_seconds

    def release(self):
        return self._capture.release()


class _DumpEvery:
    """on_frame do run_pipeline: scene/canvas a cada `every` quadros.

    Mesma leitura do diag_live_desk: a cena sai com a ancora do
    worker desenhada; o canvas sai so quando ha mascara confirmada.
    A linha de metricas traz presenca de ancora, tinta confirmada,
    passadas, sequencia do detector, miss do rastreador, ultimo
    evento, epoca do acumulador e o fps real medido na entrada.
    `last_mask` retem a mascara mais recente a CADA quadro — o dump
    e periodico, mas o Ctrl+C pode cair entre dois e a tinta
    confirmada depois do ultimo nao pode se perder.
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
        self.last_mask = None

    def __call__(self, loop, worker, frame_index):
        self.last_mask = loop.clean_mask
        if frame_index % self.every:
            return
        stamp = f"{frame_index:05d}"
        corners = worker.anchor_corners
        vis = self.tap.last_frame.copy()
        if corners is not None:
            quad = np.asarray(corners, np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [quad], True, (0, 255, 0), 2)
        _write_png(self.out_dir / f"scene_{stamp}.png", vis)
        mask = loop.clean_mask
        conf = 0
        if mask is not None:
            confirmed = np.asarray(mask) > 0
            conf = int(confirmed.sum())
            _write_png(
                self.out_dir / f"canvas_{stamp}.png",
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
    live_retries=0,
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
    Recusa um out_dir que ja tenha artefatos de outra corrida.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if _desk_artifacts(out_dir):
        raise ValueError(
            f"{out_dir} ja contem artefatos de captura de mesa; escolha "
            "outro --out (a evidencia anterior nao e apagada)"
        )
    tap = _TappedCapture(capture, max_frames=max_frames, live_retries=live_retries)
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
        final_mask = dump.last_mask
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
        final_mask = result["mask"]
    if final_mask is not None:
        _write_png(
            out_dir / "canvas_final.png",
            (np.asarray(final_mask) > 0).astype(np.uint8) * 255,
        )
    summary["canvas_final"] = final_mask is not None
    summary["interrupted"] = interrupted
    serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "session.json").write_text(serialized, encoding="utf-8")
    return result


def _desk_artifacts(out_dir):
    """Algum arquivo que uma corrida anterior do desk_capture deixou."""
    return (out_dir / "session.json").exists() or any(
        path
        for pattern in ("scene_*.png", "canvas_*.png")
        for path in out_dir.glob(pattern)
    )


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
                live_retries=_live_retries(source),
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
