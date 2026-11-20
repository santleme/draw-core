"""Gera o page_clean INT8 de pagina inteira a partir do fp32 (receita do models-v3).

Receita medida em 2026-11-20 (draw-lab/bench-3b/RESULTADO.md, Ryzen 7
7700X, 4 threads, ONNX Runtime CPU):

  fp32 mosaico de 256 (models-v2)     ~234 ms  holdout-v1 IoU 0,8700
  INT8 pagina inteira (esta receita)  ~93 ms  holdout-v1 IoU 0,8812

1. Quantizacao estatica QDQ, pesos QInt8 por canal, ativacoes QUInt8,
   calibracao MinMax sobre blocos de 256 de paginas retificadas reais.
   MinMax guarda as saidas de todos os blocos (pico medido: 454 MB
   com 20 paginas); Entropy/Percentile guardam histogramas e pesam
   muito mais.
2. Altura e largura da entrada e da saida viram dinamicas: o
   draw.page_clean le isso na carga e roda a pagina inteira numa
   chamada (padding ate multiplo de 8, recorte exato na volta).

Calibracao: as paginas retificadas que o page_clean recebe ao vivo,
capturadas no replay dos clipes da mesa (bench-clips, preset
ganho+pauta+histerese), 10 por clipe espalhadas pelo clipe inteiro;
os indices pares calibram (os impares ficam para a bancada de tempo).
Esses clipes sao gravacoes proprias da camera do autor — disjuntos do
holdout-v1 (school-notebooks-test) e das paginas em branco do portao
de FP (school-notebooks train/val): sem hash em comum e
similaridade maxima 0,46 contra 0,92 entre paginas do proprio holdout.

Uso:
  python lab/quantize_page_clean.py --src page_clean_unet_fp32.onnx \\
      --clips ~/draw-data/bench-clips --pages-cache /tmp/pc-pages \\
      --out models/page_clean_unet.onnx
  (ou --pages DIR com os .npz ja capturados)
"""

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PATCH = 256
_PAGES_PER_CLIP = 10
_PRESET = "ganho+pauta+histerese"


def capture_pages(clips_dir, out_dir, src_model):
    """Replay de cada clipe; guarda as paginas que chegam ao page_clean."""
    import lab.bench_ink as bi
    from draw.infra.manifest import verify_model
    from draw.page_clean import PageCleanInk
    from draw.perception import NeuralSheetDetector

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = _REPO_ROOT / "models"
    written = []
    pages = []
    original = PageCleanInk.ink_prob

    def recording(self, warped):
        pages.append(warped.copy())
        return original(self, warped)

    PageCleanInk.ink_prob = recording
    try:
        ink = PageCleanInk(str(src_model), verify=None)
        ink.load()
        for clip in sorted(Path(clips_dir).expanduser().iterdir()):
            if not (clip / "clip.json").exists():
                continue
            pages.clear()
            detector = NeuralSheetDetector(
                str(models / "draw_seg_v3_dyn.onnx"), verify=verify_model
            )
            detector.load()
            bi.run_clip(
                clip,
                bi.preset_flags(_PRESET),
                detector=detector,
                page_clean=ink,
                verify=None,
            )
            # Espalhadas pelo clipe inteiro: pages[::step][:10] ficava so
            # com o comeco dos clipes longos (19 passadas -> as 10 primeiras).
            count = min(_PAGES_PER_CLIP, len(pages))
            index = np.unique(np.linspace(0, len(pages) - 1, count).round().astype(int))
            pick = [pages[i] for i in index]
            path = out_dir / f"{clip.name}.npz"
            np.savez_compressed(path, *pick)
            written.append(path)
            print(f"{clip.name}: {len(pages)} passadas -> {len(pick)} paginas")
    finally:
        PageCleanInk.ink_prob = original
    # So os arquivos desta captura: um cache reaproveitado pode ter
    # clipes que sairam do corpus.
    return written


def load_pages(paths):
    """Paginas de uma lista de .npz, ou de todos os .npz de uma pasta."""
    if isinstance(paths, (str, Path)):
        paths = sorted(Path(paths).glob("*.npz"))
    pages = []
    for path in paths:
        data = np.load(path)
        pages += [data[k] for k in data.files]
    if not pages:
        raise SystemExit("nenhuma pagina .npz de calibracao")
    return pages


def tiles(page):
    h, w = page.shape[:2]
    out = []
    for y in range(0, h, _PATCH):
        for x in range(0, w, _PATCH):
            t = page[y : y + _PATCH, x : x + _PATCH]
            t = np.pad(t, [(0, _PATCH - t.shape[0]), (0, _PATCH - t.shape[1]), (0, 0)])
            out.append(t.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    return out


def dynamic_side(src, dst):
    """Altura e largura dinamicas na entrada e na saida (pagina inteira)."""
    import onnx

    model = onnx.load(str(src))
    for value in list(model.graph.input) + list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        for index, name in ((2, "height"), (3, "width")):
            dims[index].ClearField("dim_value")
            dims[index].dim_param = name
    # As formas intermediarias inferidas para 256 deixariam de valer.
    del model.graph.value_info[:]
    # O quant_pre_process declara dominios que nenhum no usa
    # (ai.onnx.ml 5, com.microsoft...). O ONNX Runtime 1.17 — o piso do
    # pyproject — recusa o arquivo pelo ai.onnx.ml 5 (review do PR #66).
    used = {node.domain for node in model.graph.node}
    keep = [op for op in model.opset_import if op.domain in used or op.domain == ""]
    del model.opset_import[:]
    model.opset_import.extend(keep)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def quantize(src, pages, dst):
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    calibration = pages[::2]

    class Reader(CalibrationDataReader):
        # Um bloco por vez: MinMax e incremental, nada fica acumulado.
        def __init__(self):
            self._feeds = ({"image": t} for p in calibration for t in tiles(p))

        def get_next(self):
            return next(self._feeds, None)

    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        fixed = Path(tmp) / "int8_256.onnx"
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        quantize_static(
            str(pre),
            str(fixed),
            Reader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            # Sem CalibMaxIntermediateOutputs: no ORT 1.23 ela LIMPA o
            # buffer ao encher sem calcular os limites (so o ultimo lote
            # calibrava) e no 1.17 chama um metodo que nao existe (review
            # do PR #66). O MinMax guarda as saidas de todos os blocos:
            # pico medido abaixo, rode com teto de memoria (ulimit -v).
        )
        dynamic_side(fixed, dst)
    return len(calibration)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="page_clean fp32 (models-v2)")
    parser.add_argument("--out", required=True, help="ONNX INT8 de pagina inteira")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pages", help="pasta com .npz de paginas ja capturadas")
    source.add_argument("--clips", help="pasta dos bench-clips (captura por replay)")
    parser.add_argument(
        "--pages-cache", help="onde gravar as paginas capturadas com --clips"
    )
    args = parser.parse_args(argv)

    if args.pages:
        pages = load_pages(args.pages)
    elif args.pages_cache:
        pages = load_pages(capture_pages(args.clips, args.pages_cache, args.src))
    else:
        with tempfile.TemporaryDirectory(prefix="pc-pages-") as cache:
            pages = load_pages(capture_pages(args.clips, cache, args.src))
    used = quantize(args.src, pages, args.out)
    data = Path(args.out).read_bytes()
    print(
        f"{args.out}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}"
        f" ({used} paginas de calibracao de {len(pages)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
