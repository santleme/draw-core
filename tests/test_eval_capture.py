"""lab/eval_capture.py: métricas puras, avaliação por bundle e o laço real.

O laço dirigido (run_pipeline) roda com modelos ONNX de teste gerados
na hora (onnx_fixtures) e detector clássico grampeado — o valor é
exercitar Pipeline + PageCleanWorker reais sem thread e sem câmera.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab import eval_capture as ec  # noqa: E402

onnx_fixtures = pytest.importorskip("onnx_fixtures")  # noqa: E402


def test_pixel_metrics_perfeito():
    mask = np.zeros((10, 10), bool)
    mask[2:5, 2:5] = True
    got = ec.pixel_metrics(mask, mask.copy())
    assert got["iou"] == 1.0 and got["precision"] == 1.0 and got["recall"] == 1.0
    assert got["tp"] == 9 and got["fp"] == 0 and got["fn"] == 0


def test_pixel_metrics_disjunto_e_vazio():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[:2] = True
    b[8:] = True
    got = ec.pixel_metrics(a, b)
    assert got["iou"] == 0.0 and got["recall"] == 0.0
    assert (
        ec.pixel_metrics(np.zeros((10, 10), bool), np.zeros((10, 10), bool))["iou"]
        == 1.0
    )
    with pytest.raises(ValueError):
        ec.pixel_metrics(np.zeros((4, 4), bool), np.zeros((5, 5), bool))


def test_compare_masks_status_por_traco():
    truth = np.zeros((20, 20), bool)
    truth[2:4, 2:4] = True
    pred = np.zeros((20, 20), bool)
    pred[2:4, 2:4] = True  # captured
    s1 = np.zeros((20, 20), bool)
    s1[2:4, 2:4] = True  # captured
    s2 = np.zeros((20, 20), bool)
    s2[10:14, 10:14] = True  # lost
    s3 = np.zeros((20, 20), bool)
    s3[16:18, 16:18] = True
    pred[16, 16] = True  # partial (recall < 0.5)
    report = ec.compare_masks(pred, truth, [s1, s2, s3])
    statuses = [s["status"] for s in report["strokes"]]
    assert statuses == ["captured", "lost", "partial"]
    assert report["stroke_counts"]["captured"] == 1
    with pytest.raises(ValueError):
        ec.compare_masks(pred, truth, [], stroke_recall=0)


def _write_bundle(path, **fields):
    base = {
        "version": np.array(1),
        "ink_full": np.zeros((40, 40), np.uint8),
        "page_ink_full": np.zeros((30, 30), np.uint8),
        "page_to_scene": np.eye(3, dtype=np.float64),
        "points": np.zeros((0, 2), np.int32),
        "offsets": np.zeros(1, np.int64),
        "thickness": np.zeros(0, np.int64),
    }
    base.update(fields)
    np.savez(path, **base)


def test_evaluate_canvas_prefere_ink_final_ao_ink_full(tmp_path):
    """O GT de fim de vídeo é a tinta sobrevivente, não a tinta total."""
    ink_full = np.zeros((40, 40), np.uint8)
    ink_full[:10] = 1  # apagado depois: só em ink_full
    ink_final = np.zeros((40, 40), np.uint8)
    ink_final[30:] = 1  # o que sobrevive
    bundle = tmp_path / "gt.npz"
    _write_bundle(bundle, ink_full=ink_full, ink_final=ink_final)
    pred = np.zeros((40, 40), bool)
    pred[30:] = True  # acertou só o estado final
    report = ec.evaluate_canvas(pred, bundle, canvas_space="scene")
    assert report["recall"] == 1.0 and report["precision"] == 1.0


def test_evaluate_canvas_sem_ink_final_cai_no_ink_full(tmp_path):
    ink_full = np.zeros((40, 40), np.uint8)
    ink_full[5:8] = 1
    bundle = tmp_path / "gt.npz"
    _write_bundle(bundle, ink_full=ink_full)
    pred = np.zeros((40, 40), bool)
    report = ec.evaluate_canvas(pred, bundle, canvas_space="scene")
    assert report["recall"] == 0.0 and report["gt_pixels"] == 120


def test_evaluate_canvas_npy_sem_cantos_exige_bundle(tmp_path):
    np.save(tmp_path / "gt.npy", np.zeros((30, 30), np.uint8))
    with pytest.raises(ValueError, match="corners|bundle"):
        ec.evaluate_canvas(
            np.zeros((30, 30), bool), tmp_path / "gt.npy", canvas_space="page"
        )


def test_load_canvas_npy_e_imagem(tmp_path):
    np.save(tmp_path / "m.npy", np.array([[0, 255], [255, 0]], np.uint8))
    got = ec.load_canvas(tmp_path / "m.npy")
    assert got.dtype == bool and got[0, 1] and got[1, 0]
    import cv2

    cv2.imwrite(str(tmp_path / "i.png"), np.array([[200, 10], [10, 200]], np.uint8))
    got = ec.load_canvas(tmp_path / "i.png", kind="image")
    assert got[1, 0] and got[0, 1] and not got[0, 0]


class _FakeNeural:
    def __init__(self):
        self.last_presence_logit = 5.0
        self.last_coverage_mask = np.full((120, 160), 255, np.uint8)
        self.last_hand_mask = np.zeros((120, 160), np.uint8)

    def __call__(self, frame, *, return_mask=True):
        corners = np.array(
            [[10, 10], [150, 10], [150, 110], [10, 110]], dtype=np.float32
        )
        return corners, "paper", np.ones(frame.shape[:2], np.uint8)


def test_detector_sincrono_publica_contrato():
    det = ec._SynchronousDetector(_FakeNeural())
    frame = np.zeros((120, 160, 3), np.uint8)
    det.submit(frame, 1.5, 7)
    result = det.poll(0)
    assert result.sequence == 1
    assert result.verdict == "paper"
    assert result.captured_at == 1.5
    assert result.coverage_mask is not None and result.coverage_mask.any()
    assert result.hand_mask is not None
    assert det.poll(0) is result  # mesma sequência não republica
    assert det.poll(1) is None


class _FakeCapture:
    """VideoCapture de mentira: folha clara com um traço escuro."""

    def __init__(self, frames, corners=(20, 10, 200, 150)):
        self._frames = frames
        self._index = 0
        self._corners = corners

    def isOpened(self):
        return True

    def get(self, prop):
        return 10.0

    def read(self):
        if self._index >= self._frames:
            return False, None
        self._index += 1
        frame = np.zeros((240, 320, 3), np.uint8)
        x0, y0, x1, y1 = self._corners
        frame[y0:y1, x0:x1] = 220  # folha
        frame[60:80, 60:160] = 20  # "tinta" escura dentro da folha
        return True, frame

    def release(self):
        pass


def test_run_pipeline_dirige_pipeline_real(tmp_path, monkeypatch):
    """Laço de ponta a ponta sem thread: detecção grampeada + worker ONNX."""

    from draw import paper as paper_module

    onnx_path = onnx_fixtures.page_clean_model(
        tmp_path / "pc.onnx", patch=256, dynamic_batch=True
    )
    # neural_model=None cai no default instalado (models/draw_seg_v3_dyn.onnx)
    # quando ele existe — e aí a detecção grampeada é ignorada. Aponta o
    # default para um caminho ausente: o laço segue pelo clássico que o
    # monkeypatch controla, com ou sem binário real no repo.
    monkeypatch.setattr(ec, "_DEFAULT_SHEET_MODEL", tmp_path / "missing.onnx")
    corners = np.array([[20, 10], [200, 10], [200, 150], [20, 150]], dtype=np.float32)
    monkeypatch.setattr(
        paper_module,
        "detect_sheet_with_confidence",
        lambda frame: (corners.copy(), "paper"),
    )
    capture = _FakeCapture(frames=40)
    result = ec.run_pipeline(
        capture,
        fps=10,
        neural_model=None,
        page_clean_model=str(onnx_path),
        hand_model="__none__",
        verify=None,
    )
    assert result["frames"] == 40
    assert result["mode"] == "page_clean"
    assert result["detections"] > 0
    assert result["ink_passes"] is not None and result["ink_passes"] > 0
    assert result["status"] == "ok"
    # O modelo didático (logit = luminância) deve ter confirmado a
    # região escura como tinta.
    assert result["mask"] is not None and result["mask"].any()
    # Cada passada que votou deixa o registro (âncora, quad de
    # registro, quad submetido, frame) que a avaliação usa para
    # projetar o GT.
    assert result["registrations"]
    for record in result["registrations"]:
        assert np.asarray(record["anchor"]).shape == (4, 2)
        assert record["quad"] is None or np.asarray(record["quad"]).shape == (4, 2)
        assert record["submitted"] is None or np.asarray(record["submitted"]).shape == (
            4,
            2,
        )
        assert record["frame"] is None or 0 <= record["frame"] < result["frames"]
    # História epochada: sem reset no caminho feliz ela coincide com
    # a lista corrente e todo registro carrega a época do acumulador.
    assert len(result["registration_history"]) == len(result["registrations"])
    assert all(
        current is past
        for current, past in zip(
            result["registrations"], result["registration_history"], strict=True
        )
    )
    assert all(record["epoch"] >= 1 for record in result["registration_history"])


def test_run_pipeline_rejeita_captura_fechada(tmp_path):
    class _Closed:
        def isOpened(self):
            return False

    with pytest.raises(ValueError, match="fonte de vídeo"):
        ec.run_pipeline(_Closed(), fps=10, page_clean_model="x.onnx", verify=None)


def test_run_pipeline_rejeita_fps_invalido(tmp_path):
    class _NoFps(_FakeCapture):
        def get(self, prop):
            return 0.0

    with pytest.raises(ValueError, match="FPS"):
        ec.run_pipeline(_NoFps(1), page_clean_model="x.onnx", verify=None)


def test_evaluate_canvas_projeta_gt_pelo_registro_da_passada(tmp_path):
    # Folha andou +4 com a âncora presa na pose inicial: o registro
    # da passada deposita a tinta deslocada +4 na grade (a grade é
    # a vista da câmera pela âncora velha — a folha anda sob ela).
    # O GT tem de ser projetado pelo par (âncora, quad) da passada —
    # a pose final ou a âncora sozinha erram o lugar.
    from lab.eval_capture import evaluate_canvas

    h, w = 40, 60
    truth = np.zeros((h, w), np.uint8)
    truth[5:15, 5:15] = 1
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, truth)

    a0 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    c1 = np.float32([[4, 0], [w + 4, 0], [w + 4, h], [4, h]])
    prediction = np.zeros((h, w), np.uint8)
    prediction[5:15, 9:19] = 1  # a tinta registrou deslocada +4

    final_pose = evaluate_canvas(prediction, gt_path, canvas_space="page", corners=c1)
    registered = evaluate_canvas(
        prediction, gt_path, canvas_space="page", registrations=[(a0, c1)]
    )
    assert registered["recall"] == 1.0
    assert registered["iou"] > final_pose["iou"]


def test_evaluate_canvas_registros_se_unem_sobre_passadas(tmp_path):
    # Dois pares de registro (âncora parada, quad andou): a tinta
    # entrou na grade nas duas colocações — a união cobre as duas.
    from lab.eval_capture import evaluate_canvas

    h, w = 40, 60
    truth = np.zeros((h, w), np.uint8)
    truth[5:15, 5:15] = 1
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, truth)

    a0 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    c1 = np.float32([[4, 0], [w + 4, 0], [w + 4, h], [4, h]])
    prediction = np.zeros((h, w), np.uint8)
    prediction[5:15, 5:19] = 1  # canônica (a0,a0) + deslocada (a0,c1)

    single = evaluate_canvas(
        prediction,
        gt_path,
        canvas_space="page",
        registrations=[(a0, a0)],
    )
    multi = evaluate_canvas(
        prediction,
        gt_path,
        canvas_space="page",
        registrations=[(a0, a0), (a0, c1)],
    )
    assert multi["alignment"] == "per_frame_anchors"
    assert multi["iou"] > single["iou"]


def test_evaluate_canvas_bundle_projeta_pelo_registro(tmp_path):
    # Bundle em espaço de página + folha que andou sob a âncora
    # velha: a projeção segue o par de registro, não a pose final.
    h, w = 40, 60
    page_gt = np.zeros((h, w), np.uint8)
    page_gt[5:15, 5:15] = 1
    page_gt[20:30, 20:30] = 1
    bundle_path = tmp_path / "gt.npz"
    _write_bundle(
        bundle_path,
        page_ink_full=page_gt,
        page_to_scene=np.eye(3, dtype=np.float64),
    )
    a0 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    c1 = np.float32([[4, 0], [w + 4, 0], [w + 4, h], [4, h]])
    prediction = np.zeros((h, w), np.uint8)
    prediction[5:15, 9:19] = 1  # traço 1 registrou +4
    prediction[20:30, 24:34] = 1  # traço 2 também

    registered = ec.evaluate_canvas(
        prediction,
        bundle_path,
        canvas_space="page",
        registrations=[(a0, c1)],
    )
    assert registered["recall"] == 1.0
    assert registered["iou"] == 1.0


def test_evaluate_canvas_projecao_igual_ao_registro_do_worker(tmp_path):
    # Invariante do conserto: a projeção do GT na grade armazenada
    # é EXATAMENTE a matriz que _register_ink_probability aplica à
    # evidência — aqui com página 30×30 ≠ grade 40×60, para a
    # composição página→grade→âncora ficar distinguível.
    import cv2

    from draw.ink import registration as reg

    h, w = 40, 60
    ph, pw = 30, 30
    page_gt = np.zeros((ph, pw), np.uint8)
    page_gt[5:15, 5:15] = 1
    page_gt[20:25, 20:28] = 1
    bundle_path = tmp_path / "gt.npz"
    _write_bundle(
        bundle_path,
        page_ink_full=page_gt,
        page_to_scene=np.eye(3, dtype=np.float64),
    )
    a0 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    c1 = np.float32([[4, 0], [w + 4, 0], [w + 4, h], [4, h]])

    # O que o worker produziria: página retificada na grade e
    # registrada sob (âncora=a0, quad da passada=c1).
    truth_grid = cv2.warpPerspective(
        page_gt.astype(np.float32),
        np.diag([w / pw, h / ph, 1.0]),
        (w, h),
        flags=cv2.INTER_NEAREST,
    )
    stored = reg._register_ink_probability(truth_grid, a0, c1)[0] > 0

    report = ec.evaluate_canvas(
        stored,
        bundle_path,
        canvas_space="page",
        registrations=[(a0, c1)],
    )
    assert report["iou"] == 1.0
    assert report["recall"] == 1.0


def test_evaluate_canvas_usa_pose_verdadeira_do_bundle(tmp_path):
    # Detector enviesado: a folha está parada na pose canônica, mas
    # o quad submetido empurra a borda esquerda +4 — o warp que o
    # modelo viu tem a tinta deslocada −4, e o GT tem de comparar
    # na geometria distorcida, não na canônica.
    import cv2

    h, w = 40, 60
    page_gt = np.zeros((h, w), np.uint8)
    page_gt[5:15, 5:15] = 1
    true_pose = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    sub_biased = np.float32([[4, 0], [w + 4, 0], [w + 4, h], [4, h]])
    bundle_path = tmp_path / "gt.npz"
    _write_bundle(
        bundle_path,
        page_ink_full=page_gt,
        page_to_scene=np.eye(3, dtype=np.float64),
        quads=true_pose[None],
    )

    destination = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    from draw.paper import perspective_matrix

    # O que o modelo viu e o worker armazenou: página na pose
    # verdadeira retificada pelo quad enviesado (âncora = reg = sub).
    input_map = perspective_matrix(sub_biased, destination) @ perspective_matrix(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]), true_pose
    )
    stored = (
        cv2.warpPerspective(
            page_gt.astype(np.float32), input_map, (w, h), flags=cv2.INTER_NEAREST
        )
        > 0
    )

    record = {
        "frame": 0,
        "anchor": sub_biased,
        "quad": sub_biased,
        "submitted": sub_biased,
    }
    report = ec.evaluate_canvas(
        stored, bundle_path, canvas_space="page", registrations=[record]
    )
    assert report["iou"] == 1.0
    assert report["recall"] == 1.0

    # Controle: sem a pose verdadeira no bundle a mesma passada
    # cai na leitura canônica e o viés é cobrado como erro.
    _write_bundle(
        bundle_path,
        page_ink_full=page_gt,
        page_to_scene=np.eye(3, dtype=np.float64),
    )
    canonical = ec.evaluate_canvas(
        stored, bundle_path, canvas_space="page", registrations=[record]
    )
    assert canonical["iou"] < 1.0


def test_run_pipeline_alinha_config_entre_pipeline_e_worker(tmp_path, monkeypatch):
    """Sem pipeline_config explícito, eval_capture usa a base legada em ambos;
    com pipeline_config explícito, repassa config.ink à PageCleanWorker."""
    from draw import pipeline as pipeline_module
    from draw.ink import worker as worker_module
    from draw.ink.config import InkConfig
    from draw.settings import PipelineConfig

    onnx_path = onnx_fixtures.page_clean_model(
        tmp_path / "pc.onnx", patch=256, dynamic_batch=True
    )
    monkeypatch.setattr(ec, "_DEFAULT_SHEET_MODEL", tmp_path / "missing.onnx")
    seen = []
    real_worker = worker_module.PageCleanWorker
    real_pipeline = pipeline_module.Pipeline

    def spy_worker(*a, **kw):
        seen.append(("worker", kw.get("config")))
        return real_worker(*a, **kw)

    def spy_pipeline(*a, **kw):
        seen.append(("pipeline", kw.get("config")))
        return real_pipeline(*a, **kw)

    monkeypatch.setattr(worker_module, "PageCleanWorker", spy_worker)
    monkeypatch.setattr(pipeline_module, "Pipeline", spy_pipeline)

    ec.run_pipeline(
        _FakeCapture(frames=1),
        fps=10,
        neural_model=None,
        page_clean_model=str(onnx_path),
        hand_model="__none__",
        verify=None,
    )
    legacy = PipelineConfig(ink=InkConfig(), ink_auto_gain=False, detect_wake=False)
    assert seen == [("worker", legacy.ink), ("pipeline", legacy)]

    seen.clear()
    prod = PipelineConfig()
    ec.run_pipeline(
        _FakeCapture(frames=1),
        fps=10,
        neural_model=None,
        page_clean_model=str(onnx_path),
        hand_model="__none__",
        verify=None,
        pipeline_config=prod,
    )
    assert seen == [("worker", prod.ink), ("pipeline", prod)]
