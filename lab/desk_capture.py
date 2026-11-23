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

--frames <dir> grava a cena BRUTA de cada quadro como fNNNN.png, sem
anotacao e sem stride — e o caminho de captura para clipes do corpus
(docs/CORPUS.md): os dumps scene_/canvas_ sao evidencia de
diagnostico, nao material de replay.

Uso:
  python -m lab.desk_capture                              # webcam local (indice 0)
  python -m lab.desk_capture --source hvcam:// --fps 30
  python -m lab.desk_capture --source sessao.drawcap --fps 30 --out /tmp/desk
  python -m lab.desk_capture --source sessao.drawcap --fps 30 --frames clipe/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.eval_capture import run_pipeline  # noqa: E402

# Mesmo orcamento do app antes de declarar a fonte morta
# (_CAPTURE_MAX_FAILURES em draw.camera).
_LIVE_READ_RETRIES = 15


class LiveSourceExhausted(RuntimeError):
    """Fonte ao vivo esgotou o orcamento de releituras.

    Distinta de RuntimeError para a finalizacao da corrida gravar a
    evidencia parcial (session.json + canvas_final) ANTES do erro
    subir — a CLI ainda sai 2, mas a tinta confirmada nao se perde.
    """


def _write_png(path, image):
    """imwrite checado: False silencioso vira OSError nomeando o destino."""
    if not cv2.imwrite(str(path), image):
        raise OSError(f"falha ao gravar {path}")


def _live_retries(source):
    """Fontes ao vivo toleram leituras falhas transitorias; arquivo
    (.drawcap, video) termina de verdade no primeiro False. `file://`
    e so outra grafia de caminho local — EOF igualmente real."""
    if isinstance(source, int):
        return _LIVE_READ_RETRIES
    text = str(source)
    if text.lower().startswith("file://"):
        text = unquote(urlparse(text).path)
    return 0 if Path(text).is_file() else _LIVE_READ_RETRIES


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
        self._first_return = None
        self._last_return = None
        self._src_first_ts = None
        self._src_last_ts = None
        self._src_frames = 0
        self._src_dropped = 0
        self._src_drop_base = 0

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
        ok, frame = self._read_once()
        failures = 0 if ok else 1
        while not ok and failures < self._live_retries:
            ok, frame = self._read_once()
            failures += 0 if ok else 1
        if not ok:
            if self._live_retries:
                raise LiveSourceExhausted(
                    f"fonte ao vivo parou de entregar: {failures} leituras "
                    f"falhas seguidas apos {self.frames_read} quadros"
                )
            return False, None
        self.frames_read += 1
        self.last_frame = frame
        stamp = self._now()
        if self._first_return is None:
            self._first_return = stamp
        self._last_return = stamp
        src_ts = getattr(self._capture, "last_capture_ts_ms", None)
        if src_ts is not None:
            if self._src_first_ts is None:
                self._src_first_ts = src_ts
                # Baseline: saltos anteriores ao primeiro quadro
                # utilizavel (JPEGs indecodificaveis na conexao)
                # ficam FORA da janela da taxa.
                self._src_drop_base = getattr(self._capture, "source_dropped", 0)
            self._src_last_ts = src_ts
            self._src_frames += 1
            # Quadros emitidos = entregues + saltos de seq que a fonte
            # contabilizou — sem isto, frames perdidos no caminho
            # derrubam a taxa medida mesmo com a camera a pleno.
            self._src_dropped = (
                getattr(self._capture, "source_dropped", 0) - self._src_drop_base
            )
        return True, frame

    def _read_once(self):
        ok, frame = self._capture.read()
        return ok, frame

    def real_fps(self):
        """Cadencia observada na torneira: quadros/espacamento entre retornos.

        Mede o ritmo com que os quadros cruzam o read(), nao o tempo
        dentro dele — uma leitura instantanea de backend com buffer
        nao infla o numero. Quando o pipeline e o gargalo, o numero
        mostra a cadencia efetiva que a corrida sustentou.
        """
        if (
            self._first_return is None
            or self._last_return is None
            or self._last_return <= self._first_return
        ):
            return 0.0
        return (self.frames_read - 1) / (self._last_return - self._first_return)

    def src_fps(self):
        """Cadencia do emissor (hvcam, ajudante), quando a fonte a expõe.

        Os carimbos last_capture_ts_ms sao do lado que captura —
        imunes ao ritmo do pipeline e a buffers. None quando a fonte
        nao publica o carimbo (VideoCapture comum).
        """
        if (
            self._src_first_ts is None
            or self._src_last_ts is None
            or self._src_last_ts <= self._src_first_ts
        ):
            return None
        return (
            (self._src_frames + self._src_dropped - 1)
            * 1000.0
            / (self._src_last_ts - self._src_first_ts)
        )

    def release(self):
        return self._capture.release()


class _DumpEvery:
    """on_frame do run_pipeline: scene/canvas a cada `every` quadros.

    Mesma leitura do diag_live_desk: a cena sai com a ancora do
    worker desenhada; o canvas sai so quando ha mascara confirmada.
    A linha de metricas traz presenca de ancora, tinta confirmada,
    passadas, sequencia do detector, miss do rastreador, ultimo
    evento, epoca do acumulador e o fps real medido na entrada.
    `loop` fica retido: a mascara que a finalizacao grava e a VIVA
    do pipeline (a que estiver publicada na hora do Ctrl+C), nao o
    retrato do ultimo quadro processado — uma sessao reaberta pode
    te-la zerado depois do ultimo dump.
    """

    def __init__(
        self, out_dir, tap, every=15, log=print, now=time.monotonic, frames_dir=None
    ):
        if every < 1:
            raise ValueError("--every precisa ser >= 1")
        self.out_dir = Path(out_dir)
        self.tap = tap
        self.every = every
        self.log = log
        self.now = now
        self.t0 = now()
        self.loop = None
        self.frames_dir = None if frames_dir is None else Path(frames_dir)

    def __call__(self, loop, worker, frame_index):
        self.loop = loop
        if self.frames_dir is not None:
            # Quadro bruto, contiguo e sem anotacao — o fNNNN.png do clipe.
            _write_png(self.frames_dir / f"f{frame_index:04d}.png", self.tap.last_frame)
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
        src = self.tap.src_fps()
        src_text = f" src={src:.1f}" if src is not None else ""
        self.log(
            f"f{frame_index:5d} t={self.now() - self.t0:5.1f}s "
            f"anchor={'sim' if corners is not None else 'NAO'} "
            f"conf={conf} passes={worker.stats()['passes_total']} "
            f"det={getattr(loop.detection_worker, 'sequence', None)} "
            f"miss={getattr(loop.tracker, '_frames_since_seen', None)} "
            f"evt={loop.last_tracker_event} "
            f"epoch={worker.canvas_epoch} fps={self.tap.real_fps():.1f}{src_text}"
        )


def run(
    capture,
    *,
    out_dir,
    every=15,
    fps=None,
    max_frames=None,
    live_retries=0,
    frames_dir=None,
    neural_model=None,
    page_clean_model=None,
    hand_model=None,
    verify="manifest",
    pipeline_config=None,
    log=print,
):
    """Dirige o Pipeline real sobre a fonte ate ela fechar ou Ctrl+C.

    A captura continua do chamador (release() fica fora daqui).
    Devolve o resumo de run_pipeline, ou None quando interrompido;
    em ambos os casos session.json fica no out_dir — os campos
    volumosos (mask, anchors, registrations) saem como contagens.
    Recusa um out_dir que ja tenha artefatos de outra corrida e um
    frames_dir ja povoado com f*.png.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if _desk_artifacts(out_dir):
        raise ValueError(
            f"{out_dir} ja contem artefatos de captura de mesa; escolha "
            "outro --out (a evidencia anterior nao e apagada)"
        )
    if frames_dir is not None:
        frames_dir = Path(frames_dir)
        if frames_dir.is_dir() and any(frames_dir.glob("f*.png")):
            raise ValueError(
                f"{frames_dir} ja contem quadros f*.png; escolha outro "
                "--frames (capturas nao se misturam)"
            )
        frames_dir.mkdir(parents=True, exist_ok=True)
    tap = _TappedCapture(capture, max_frames=max_frames, live_retries=live_retries)
    dump = _DumpEvery(out_dir, tap, every=every, log=log, frames_dir=frames_dir)
    log(f"saida em {out_dir}")
    if frames_dir is not None:
        log(f"quadros brutos em {frames_dir}")
    result = None
    interrupted = False
    failure = None
    try:
        result = run_pipeline(
            tap,
            fps=fps,
            neural_model=neural_model,
            page_clean_model=page_clean_model,
            hand_model=hand_model,
            on_frame=dump,
            verify=verify,
            pipeline_config=pipeline_config,
        )
    except KeyboardInterrupt:
        interrupted = True
    except LiveSourceExhausted as error:
        failure = error
    if result is None:
        summary = {
            "status": "interrupted" if interrupted else "source_failed",
            "frames": tap.frames_read,
        }
        if failure is not None:
            summary["error"] = str(failure)
        # A mascara VIVA do pipeline — publicada ou limpa depois do
        # ultimo quadro fechado — e a verdade do canvas final.
        final_mask = dump.loop.clean_mask if dump.loop is not None else None
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
    if failure is not None:
        raise failure
    return result


# Os dumps do desk usam sufixo estritamente numerico (>=5 digitos):
# scene_settled_42.png do falsify e de outra ferramenta e nao colide.
_DESK_DUMP_RE = re.compile(r"(scene|canvas)_\d{5,}\.png")


def _desk_artifacts(out_dir):
    """Algum arquivo que uma corrida anterior do desk_capture deixou.

    Confere NOMES via iterdir, nao exists(): um symlink pendurado
    chamado canvas_final.png tem exists()==False e, sem a guarda,
    redirecionaria a escrita do canvas final para fora do --out.
    """
    names = {path.name for path in out_dir.iterdir()}
    return (
        "session.json" in names
        or "canvas_final.png" in names
        or any(_DESK_DUMP_RE.fullmatch(name) for name in names)
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
    parser.add_argument(
        "--frames",
        type=Path,
        metavar="DIR",
        help="grava a cena bruta de cada quadro como DIR/fNNNN.png (clipe do corpus)",
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
    parser.add_argument(
        "--legacy-ink",
        action="store_true",
        help="usa a configuracao legada da tinta (sem ganho, sem Detect-Wake, InkConfig())",
    )
    args = parser.parse_args(argv)
    if args.every < 1:
        parser.error("--every precisa ser >= 1")

    from draw.camera import open_capture, parse_source
    from draw.ink.config import InkConfig
    from draw.settings import PipelineConfig

    pipeline_config = (
        PipelineConfig(ink=InkConfig(), ink_auto_gain=False, detect_wake=False)
        if args.legacy_ink
        else PipelineConfig()
    )
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
                frames_dir=args.frames,
                neural_model=args.neural_model,
                page_clean_model=args.page_clean_model,
                hand_model=args.hand_model,
                verify=None if args.allow_unverified else "manifest",
                pipeline_config=pipeline_config,
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
