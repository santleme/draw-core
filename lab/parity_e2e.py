"""Paridade ponta a ponta: Pipeline do draw-core vs SheetInkLoop legado.

Roda o mesmo corpus (diretório de frames ou MP4) pelos dois laços, com
os MESMOS binários de modelo e adaptadores síncronos equivalentes, e
compara por frame: presença da folha, cantos do rastreador e a máscara
de canvas confirmada. O laço legado roda num subprocesso dentro do
repo antigo (--legacy-repo; os pacotes `draw` dos dois lados colidem
num mesmo processo) pelo driver lab/parity_legacy_driver.py, que liga
as 8 flags do conjunto validado — o default do legado não é a
configuração validada; no draw-core elas já são o default.

Diferença com a paridade de workers falsos (tests/test_pipeline*): aqui
detecção neural, rastreador, warp, trabalhadora de tinta e acumulador
são os reais dos dois lados — só a execução é adaptada (síncrona,
relógio de cena) nos dois.

Com --hand-model resolvendo um arquivo (default: o do repo legado),
masker e presença de mão entram nos dois lados como na produção —
sem ele os caminhos dependentes de mão ficam sem paridade medida e
o relatório marca isso (hand_samples zero).

Uso:
  python lab/parity_e2e.py --corpus <dir|video.mp4> --out <dir> \
      [--legacy-repo ~/draw] [--frames 120] [--fps 2.0] \
      [--iou-min 0.999] [--corner-tol-px 0.5]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from lab.sync_workers import (  # noqa: E402
    StepClock,
    SyncDetector,
    SyncPresence,
    iter_frames,
)


def parity_frame_time(idx, fps):
    """Instante do quadro, com a MESMA fórmula do driver legado.

    O parity_legacy_driver.py (auto-contido, não muda) calcula
    `idx * (1.0 / fps)`. Dividir `idx / fps` é equivalente em
    matemática mas arredonda diferente: a 10 FPS, 1.4 - 0.8 dá
    0.5999999999999999 (abaixo do limiar de 0,6 s do tracker) contra
    0.6000000000000001 na multiplicação — um quadro de presença
    divergente reprova a paridade sem que os detectores discordem.
    """
    return idx * (1.0 / fps)


def run_new(
    corpus, out_dir, *, frames, fps, sheet_model, ink_model, hand_model, verify
):
    """Lado novo: Pipeline com adaptadores síncronos, defaults de produção.

    Com hand_model resolvendo um arquivo, masker e presença entram como
    na produção (draw.app) — sem eles os caminhos dependentes de mão
    ficam sem paridade medida e o relatório diz isso.
    """
    from draw import hand as hand_module
    from draw.contracts import FramePacket
    from draw.hand_presence import make_hand_presence
    from draw.ink.config import InkConfig
    from draw.ink.worker import PageCleanWorker
    from draw.page_clean import PageCleanInk
    from draw.perception import NeuralSheetDetector
    from draw.pipeline import Pipeline
    from draw.settings import PipelineConfig

    detector = NeuralSheetDetector(str(sheet_model), verify=verify)
    detector.load()
    page_clean = PageCleanInk(str(ink_model), verify=verify)
    page_clean.load()
    hand_masker = (
        hand_module.HandMasker(hand_model, verify=verify)
        if hand_module.hand_model_available(hand_model)
        else None
    )
    presence = make_hand_presence(hand_model, verify=verify) if hand_model else None

    clock = StepClock()
    det = SyncDetector(detector)
    presence_worker = (
        SyncPresence(presence, strict=True) if presence is not None else None
    )
    # A régua de paridade compara contra o SheetInkLoop legado (Tier A, sem
    # ganho, sem Detect-Wake e com voto/cadência legados): fixa a mesma base
    # em Pipeline e PageCleanWorker para não misturar o preset de produção.
    legacy_config = PipelineConfig(
        ink=InkConfig(), ink_auto_gain=False, detect_wake=False
    )
    worker = PageCleanWorker(
        page_clean,
        hand_masker,
        clock=clock,
        config=legacy_config.ink,
        ink_divnorm=legacy_config.ink_divnorm,
    )
    loop = Pipeline(
        detection_worker=det,
        ink_worker=worker,
        hand_presence_worker=presence_worker,
        hand_masker=hand_masker,
        clock=clock,
        log=lambda *_a: None,
        config=legacy_config,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, canvases = [], []
    try:
        for idx, frame in iter_frames(corpus, frames):
            clock.t = parity_frame_time(idx, fps)
            loop.step(FramePacket(image=frame, captured_at=clock.t, sequence=idx + 1))
            worker.pump()
            loop.collect_ink()
            if worker.error is not None:
                raise RuntimeError("trabalhadora de tinta falhou") from worker.error
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
    finally:
        # Os landmarkers fecham aqui, não pelo __del__.
        if hand_masker is not None:
            hand_masker.close()
        if presence is not None:
            presence.close()
    hand_samples = presence_worker.real_samples if presence_worker else 0
    (out_dir / "telemetry.json").write_text(
        json.dumps({"frames": records, "hand_samples": hand_samples}, indent=1)
    )
    np.savez_compressed(out_dir / "canvas.npz", *canvases)
    return records, canvases, hand_samples


def run_legacy(
    legacy_repo, corpus, out_dir, *, frames, fps, sheet_model, ink_model, hand_model
):
    """Lado legado: subprocesso com PYTHONPATH=<repo legado>.

    DRAW_* herdado NÃO passa: uma flag de Tier B/C (ou um
    DRAW_MODEL_STRICT=0) no ambiente do chamador alteraria a linha
    de base do legado e fabricaria paridade falsa. O driver seta as
    8 flags validadas ele mesmo.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("DRAW_")}
    env["PYTHONPATH"] = str(legacy_repo)
    driver = _REPO_ROOT / "lab" / "parity_legacy_driver.py"
    cmd = [
        sys.executable,
        str(driver),
        "--corpus",
        str(corpus),
        "--out",
        str(out_dir),
        "--frames",
        str(frames),
        "--fps",
        str(fps),
        "--sheet-model",
        str(sheet_model),
        "--ink-model",
        str(ink_model),
        "--hand-model",
        str(hand_model),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit("driver legado falhou")
    out_dir = Path(out_dir)
    telemetry = json.loads((out_dir / "telemetry.json").read_text())
    with np.load(out_dir / "canvas.npz") as archive:
        # As chaves sao arr_0..arr_N: ordem numerica, nao lexica.
        canvases = [
            archive[key]
            for key in sorted(archive.files, key=lambda k: int(k.split("_")[1]))
        ]
    return telemetry["frames"], canvases, telemetry.get("hand_samples", 0)


def compare(new_records, new_masks, old_records, old_masks, *, iou_min, corner_tol):
    """Relatório de divergências por frame, alinhado pelo índice da fonte.

    Cada registro guarda o índice do frame de origem — com lacunas
    (imagem ilegível pulada), parear por posição compararia entradas
    diferentes. O alinhamento é pelo identificador: frame só num lado
    é divergência reportada, nunca silêncio.
    """
    new_by_frame = {
        record["frame"]: (record, mask)
        for record, mask in zip(new_records, new_masks, strict=True)
    }
    old_by_frame = {
        record["frame"]: (record, mask)
        for record, mask in zip(old_records, old_masks, strict=True)
    }
    report = {"frames": 0, "divergences": []}
    for frame_id in sorted(set(new_by_frame) | set(old_by_frame)):
        pair_new = new_by_frame.get(frame_id)
        pair_old = old_by_frame.get(frame_id)
        if pair_new is None or pair_old is None:
            report["divergences"].append(
                {
                    "frame": frame_id,
                    "frame_missing": "new" if pair_new is None else "legacy",
                }
            )
            continue
        (new, nm), (old, om) = pair_new, pair_old
        report["frames"] += 1
        issues = {}
        if new["present"] != old["present"]:
            issues["present"] = {"new": new["present"], "legacy": old["present"]}
        if new["corners"] is not None and old["corners"] is not None:
            delta = float(
                np.linalg.norm(
                    np.asarray(new["corners"]) - np.asarray(old["corners"]), axis=1
                ).max()
            )
            if delta > corner_tol:
                issues["corner_max_delta_px"] = delta
        if nm.shape == om.shape:
            union = int((nm | om).sum())
            iou = 1.0 if union == 0 else float((nm & om).sum() / union)
            if iou < iou_min:
                issues["canvas_iou"] = round(iou, 6)
                issues["canvas_px"] = {"new": int(nm.sum()), "legacy": int(om.sum())}
        else:
            # Qualquer shape divergente conta — transpostas (800,600)
            # vs (600,800) têm o mesmo size e escapavam das duas
            # checagens anteriores.
            issues["canvas_shape"] = {"new": list(nm.shape), "legacy": list(om.shape)}
        if issues:
            report["divergences"].append({"frame": frame_id, **issues})
    report["ok"] = bool(report["frames"]) and not report["divergences"]
    if not report["frames"]:
        report["error"] = "corpus sem frames legíveis — nada comparado"
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--legacy-repo", default=str(Path.home() / "draw"))
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--sheet-model")
    ap.add_argument("--ink-model")
    ap.add_argument(
        "--hand-model",
        help=(
            "modelo de mão (hand_landmarker.task); o default é o do "
            "repo legado. Sem arquivo resolvendo, os dois lados rodam "
            "sem masker/presença e o relatório marca a mão como não medida"
        ),
    )
    ap.add_argument("--iou-min", type=float, default=0.999)
    ap.add_argument("--corner-tol-px", type=float, default=0.5)
    ap.add_argument(
        "--allow-unverified",
        action="store_true",
        help="desliga a verificação do manifesto no lado novo (modelos fora dele)",
    )
    args = ap.parse_args(argv)

    legacy_repo = Path(args.legacy_repo)
    if not (legacy_repo / "draw" / "loop.py").exists():
        raise SystemExit(f"repo legado não encontrado em {legacy_repo}")
    legacy_models = legacy_repo / "models"
    sheet_model = Path(
        args.sheet_model or legacy_models / "draw_seg_v3_dyn.onnx"
    ).resolve()
    ink_model = Path(args.ink_model or legacy_models / "page_clean_unet.onnx").resolve()
    hand_model = Path(
        args.hand_model or legacy_models / "hand_landmarker.task"
    ).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.allow_unverified:
        verify = None
    else:
        from draw.infra.manifest import verify_model

        verify = verify_model
    new_records, new_masks, new_hand_samples = run_new(
        args.corpus,
        out / "new",
        frames=args.frames,
        fps=args.fps,
        sheet_model=sheet_model,
        ink_model=ink_model,
        hand_model=hand_model,
        verify=verify,
    )
    # Corpus vazio/ilegível: falha explícita antes de gastar a
    # passada legada — o comparador também recusa zero frames.
    if not new_records:
        raise SystemExit("corpus sem frames legíveis — nada a comparar")
    old_records, old_masks, old_hand_samples = run_legacy(
        legacy_repo,
        args.corpus,
        out / "legacy",
        frames=args.frames,
        fps=args.fps,
        sheet_model=sheet_model,
        ink_model=ink_model,
        hand_model=hand_model,
    )
    report = compare(
        new_records,
        new_masks,
        old_records,
        old_masks,
        iou_min=args.iou_min,
        corner_tol=args.corner_tol_px,
    )
    report["corpus"] = str(args.corpus)
    report["sheet_model"] = str(sheet_model)
    report["ink_model"] = str(ink_model)
    report["hand_model"] = str(hand_model)
    # Só conta como medida quando os DOIS lados produziram amostras
    # reais de presença: com um lado a zero (mediapipe/modelo ausente
    # ou sinal indisponível) o caminho de mão desse lado ficou ligado
    # mas nada foi medido — um "ok" não cobre a mão nele.
    report["hand_samples"] = {"new": new_hand_samples, "legacy": old_hand_samples}
    report["hand_measured"] = bool(new_hand_samples and old_hand_samples)
    (out / "parity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
