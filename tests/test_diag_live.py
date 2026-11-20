"""Ferramenta de diagnóstico ao vivo (lab/diag_live.py): partes puras."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from lab import diag_live

from draw import config


class _FakeClock:
    def __init__(self, start=100.0):
        self.now = start

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_quad_aspect_is_width_over_height():
    quad = [[0, 0], [200, 0], [200, 100], [0, 100]]
    assert diag_live.quad_aspect(quad) == pytest.approx(2.0)
    tall = [[0, 0], [100, 0], [100, 300], [0, 300]]
    assert diag_live.quad_aspect(tall) == pytest.approx(1 / 3)


def test_aspect_grid_keeps_quad_proportion_on_the_long_side():
    wide = [[0, 0], [300, 0], [300, 100], [0, 100]]  # 3:1
    assert diag_live.aspect_grid(wide, long_side=900) == (900, 300)
    tall = [[0, 0], [100, 0], [100, 300], [0, 300]]
    assert diag_live.aspect_grid(tall, long_side=900) == (300, 900)
    config.apply_format("a4")
    assert max(diag_live.aspect_grid(wide)) == max(config.PAPER_W, config.PAPER_H)


def test_luminance_splits_quad_left_right():
    frame = np.full((100, 200, 3), 200, np.uint8)
    frame[:, :100] = 40  # metade esquerda escura
    quad = [[10, 10], [190, 10], [190, 90], [10, 90]]
    stats = diag_live.luminance_stats(frame, quad)
    assert stats["quad_left_mean"] < 60 < 180 < stats["quad_right_mean"]
    assert stats["frame_mean"] == pytest.approx(120, abs=1)
    assert "quad_left_mean" not in diag_live.luminance_stats(frame)


def test_rotate_frame_180_moves_the_corner():
    frame = np.full((40, 60, 3), 230, np.uint8)
    frame[2:8, 3:9] = 20
    out = diag_live.rotate_frame(frame, 180)
    assert out[32:38, 51:57].mean() < 60
    assert diag_live.rotate_frame(frame, 0) is frame


def test_warp_to_returns_requested_grid():
    frame = np.zeros((120, 160, 3), np.uint8)
    quad = [[10, 10], [150, 10], [150, 110], [10, 110]]
    assert diag_live.warp_to(frame, quad, (70, 50)).shape == (50, 70, 3)


def test_diagnose_frame_writes_artifacts_and_summary(tmp_path):
    """Roda o fluxo inteiro com detector e tinta de mentira."""
    config.apply_format("a4")
    frame = np.full((240, 320, 3), 200, np.uint8)
    quad = np.float32([[20, 20], [300, 20], [300, 220], [20, 220]])

    class _Detector:
        last_hand_mask = None

        def __call__(self, image, *, return_mask=True):
            return quad, "paper", np.full(image.shape[:2], 255, np.uint8)

    class _Ink:
        threshold = 0.9

        def ink_prob(self, warped):
            return np.full(warped.shape[:2], 0.95, np.float32)

    record = diag_live.diagnose_frame(frame, _Detector(), _Ink(), tmp_path, 0)
    for name in (
        "000_raw.png",
        "000_seg.png",
        "000_warp.png",
        "000_warp_aspect.png",
        "000_prob.png",
        "000_mask.png",
        "000_mask_aspect.png",
    ):
        assert cv2.imread(str(tmp_path / name)) is not None, name
    assert record["verdict"] == "paper"
    assert record["quad_aspect"] == pytest.approx(280 / 200)
    assert record["ink_px"]["warp"]["0.85"] > 0
    json.dumps(record)  # serializável para o summary.json


def test_diagnose_frame_without_quad_records_luminance_only(tmp_path):
    class _Detector:
        last_hand_mask = None

        def __call__(self, image, *, return_mask=True):
            return None, None, None

    record = diag_live.diagnose_frame(
        np.full((60, 80, 3), 100, np.uint8), _Detector(), object(), tmp_path, 3
    )
    assert record["corners"] is None
    assert record["luminance"]["frame_mean"] == pytest.approx(100)
    assert (tmp_path / "003_raw.png").exists()


# --- review do PR #51 ---------------------------------------------------

_QUAD = np.float32([[20, 20], [300, 20], [300, 220], [20, 220]])


class _FakeDetector:
    def __init__(self, *args, **kwargs):
        self.last_hand_mask = None

    def load(self):
        pass

    def __call__(self, image, *, return_mask=True):
        return _QUAD, "paper", np.full(image.shape[:2], 255, np.uint8)


class _FakeInk:
    threshold = 0.9

    def __init__(self, *args, **kwargs):
        pass

    def load(self):
        pass

    def ink_prob(self, warped):
        return np.full(warped.shape[:2], 0.95, np.float32)


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr("draw.perception.NeuralSheetDetector", _FakeDetector)
    monkeypatch.setattr("draw.page_clean.PageCleanInk", _FakeInk)
    yield
    config.apply_format("a4")


def _png_dir(folder, names, bad=()):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = folder / name
        if name in bad:
            path.write_bytes(b"isto nao e um png")
        else:
            cv2.imwrite(str(path), np.full((240, 320, 3), 150, np.uint8))
    return folder


class _Capture:
    """Fonte de mentira: `good` leituras boas e depois só falhas."""

    def __init__(self, good):
        self.good = good
        self.released = 0

    def isOpened(self):
        return True

    def read(self):
        if self.good > 0:
            self.good -= 1
            return True, np.zeros((8, 8, 3), np.uint8)
        return False, None

    def release(self):
        self.released += 1


def test_dead_source_is_bounded_and_the_capture_released():
    """hvcam devolve False em stream caído: sem limite o laço não
    terminava e o resumo nunca saía (review #51)."""
    clk = _FakeClock()
    cap = _Capture(good=4)  # 1 abre a janela de FPS; 3 viram quadros
    got = []
    with pytest.raises(diag_live.SourceExhausted, match="sem quadro"):
        for frame, _fps in diag_live._frames_from_source(
            "hvcam://",
            6,
            0.0,
            0.0,
            stall_s=0.3,
            opener=lambda _s: cap,
            fps_burst_s=0.0,
            clock=clk,
            sleep=clk.sleep,
        ):
            got.append(frame)
    assert len(got) == 3
    assert cap.released == 1


def test_source_opens_through_the_app_open_capture():
    """Índice de câmera passa por parse_source/open_capture do app, com o
    MJPEG+720p do _configure_local_capture (review #51)."""
    seen = []

    def opener(target):
        seen.append(target)
        return _Capture(good=2)

    clk = _FakeClock()
    list(
        diag_live._frames_from_source(
            "0", 1, 0.0, 0.0, opener=opener, fps_burst_s=0.0, clock=clk, sleep=clk.sleep
        )
    )
    list(
        diag_live._frames_from_source(
            "hvcam://",
            1,
            0.0,
            0.0,
            opener=opener,
            fps_burst_s=0.0,
            clock=clk,
            sleep=clk.sleep,
        )
    )
    assert seen == [0, "hvcam://"]  # '0' vira índice; URL segue como texto


def test_warmup_zero_still_reports_a_real_fps():
    clk = _FakeClock()

    class _Ticking(_Capture):
        def read(self):
            clk.sleep(0.005)
            return super().read()

    cap = _Ticking(good=1_000_000)
    fps = [
        f
        for _frame, f in diag_live._frames_from_source(
            "0",
            3,
            0.0,
            0.0,
            opener=lambda _s: cap,
            fps_burst_s=0.05,
            clock=clk,
            sleep=clk.sleep,
        )
    ]
    assert len(fps) == 3 and all(v > 0 for v in fps)


def test_main_keeps_a_partial_summary_when_the_source_dies(
    monkeypatch, tmp_path, fake_models
):
    def dying(*_args, **_kwargs):
        for _ in range(2):
            yield np.full((240, 320, 3), 150, np.uint8), 30.0
        raise diag_live.SourceExhausted("fonte parou de entregar: 10.0 s sem quadro")

    monkeypatch.setattr(diag_live, "_frames_from_source", dying)
    out = tmp_path / "out"
    code = diag_live.main(["--source", "hvcam://", "--out", str(out), "--frames", "6"])
    assert code == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] is False and "sem quadro" in summary["error"]
    assert len(summary["frames"]) == 2
    assert (out / "001_raw.png").exists()


def test_main_refuses_a_folder_that_already_has_a_diagnostic(
    tmp_path, fake_models, capsys
):
    """Reusar --out misturava execuções: o warp antigo ficava ao lado do
    raw novo (review #51)."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "000_raw.png").write_bytes(b"antigo")
    src = _png_dir(tmp_path / "in", ["a.png"])
    assert diag_live.main(["--input", str(src), "--out", str(out)]) == 2
    assert (out / "000_raw.png").read_bytes() == b"antigo"  # nada sobrescrito
    assert "já tem um diagnóstico" in capsys.readouterr().err
    (out / "000_raw.png").unlink()
    (out / "summary.json").write_text("{}")
    assert diag_live.main(["--input", str(src), "--out", str(out)]) == 2


def test_failed_png_write_names_the_output(tmp_path):
    (tmp_path / "000_raw.png").mkdir()  # imwrite devolve False, sem exceção
    with pytest.raises(OSError, match="000_raw.png"):
        diag_live.diagnose_frame(
            np.full((240, 320, 3), 150, np.uint8),
            _FakeDetector(),
            _FakeInk(),
            tmp_path,
            0,
        )


def test_dir_replay_skips_corrupt_pngs_and_limits_by_good_decodes(tmp_path, capsys):
    folder = _png_dir(
        tmp_path / "in", ["001.png", "002.png", "003.png"], bad=("001.png",)
    )
    frames = list(diag_live._frames_from_dir(folder, 2))
    assert len(frames) == 2  # 002 e 003: o corrompido não gasta o limite
    assert "001.png ilegível" in capsys.readouterr().err


def test_main_rejects_a_missing_input_folder(tmp_path, fake_models, capsys):
    code = diag_live.main(
        ["--input", str(tmp_path / "nao-existe"), "--out", str(tmp_path / "o")]
    )
    assert code == 2
    assert "não é uma pasta" in capsys.readouterr().err
    assert not (tmp_path / "o" / "summary.json").exists()


def test_main_fails_when_the_folder_has_no_readable_frame(tmp_path, fake_models):
    folder = _png_dir(tmp_path / "in", ["a.png"], bad=("a.png",))
    out = tmp_path / "out"
    assert diag_live.main(["--input", str(folder), "--out", str(out)]) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] is False and summary["frames"] == []


def test_format_flag_sets_the_session_grid(tmp_path, fake_models):
    """Carta é 929x1200: o diagnóstico usava sempre a grade A4 (review #51)."""
    src = _png_dir(tmp_path / "in", ["a.png"])
    out = tmp_path / "out"
    assert (
        diag_live.main(["--input", str(src), "--out", str(out), "--format", "carta"])
        == 0
    )
    record = json.loads((out / "summary.json").read_text())["frames"][0]
    assert record["grids"]["warp"] == [929, 1200]
    assert cv2.imread(str(out / "000_warp.png")).shape[:2] == (1200, 929)


def test_ink_fraction_is_comparable_across_grids(tmp_path):
    """A contagem bruta de duas grades de tamanhos diferentes não
    compara; a fração sim (review #51)."""
    config.apply_format("a4")
    record = diag_live.diagnose_frame(
        np.full((240, 320, 3), 150, np.uint8), _FakeDetector(), _FakeInk(), tmp_path, 0
    )
    px = record["ink_px"]
    assert px["warp"]["0.5"] != px["warp_aspect"]["0.5"]  # grades diferentes
    assert record["ink_frac"]["warp"]["0.5"] == pytest.approx(1.0)
    assert record["ink_frac"]["warp_aspect"]["0.5"] == pytest.approx(1.0)


# --- segunda rodada do review do PR #51 ---------------------------------


def test_empty_input_path_is_rejected_and_never_opens_a_camera(
    monkeypatch, tmp_path, fake_models, capsys
):
    """--input "" caía no ramo da webcam (verdade falsa) e Path("") é a
    pasta atual: agora é erro de uso (review #51)."""

    def boom(*_a, **_k):
        raise AssertionError("não devia abrir a câmera")

    monkeypatch.setattr(diag_live, "_frames_from_source", boom)
    assert diag_live.main(["--input", "", "--out", str(tmp_path / "o")]) == 2
    assert diag_live.main(["--input", "  ", "--out", str(tmp_path / "o")]) == 2
    assert "não é uma pasta" in capsys.readouterr().err


def test_read_fps_is_not_diluted_by_diagnosis_time():
    """O gerador fica suspenso durante o diagnóstico; o FPS vem de uma
    rajada contínua de leituras, não de leituras ÷ tempo total."""
    clk = _FakeClock()

    class _Slow(_Capture):
        def read(self):
            clk.sleep(0.01)  # ~100 leituras/s
            return super().read()

    cap = _Slow(good=10_000)
    seen = []
    for _frame, fps in diag_live._frames_from_source(
        "0", 3, 0.0, 0.0, opener=lambda _s: cap, clock=clk, sleep=clk.sleep
    ):
        seen.append(fps)
        clk.sleep(0.4)  # "diagnóstico" demorado entre os quadros
    assert len(seen) == 3 and len(set(seen)) == 1  # medido uma vez
    assert seen[0] > 50  # sem diluir: ~95/s, não ~20/s


def test_png_write_failure_keeps_completed_frames(monkeypatch, tmp_path, fake_models):
    """Falha ao gravar o PNG do 2º quadro: o 1º segue no resumo parcial
    (review #51)."""
    real = diag_live._write_png

    def flaky(path, image):
        if Path(path).name.startswith("001_"):
            raise OSError(f"falha ao gravar {path}")
        return real(path, image)

    monkeypatch.setattr(diag_live, "_write_png", flaky)
    src = _png_dir(tmp_path / "in", ["a.png", "b.png", "c.png"])
    out = tmp_path / "out"
    assert diag_live.main(["--input", str(src), "--out", str(out)]) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] is False and "001_" in summary["error"]
    assert len(summary["frames"]) == 1


@pytest.mark.parametrize(
    "kept", ["000_mask.png", "000_warp.png", "003_prob.png", "001_seg.png"]
)
def test_output_guard_catches_any_retained_artifact(tmp_path, fake_models, kept):
    """Operador guardou só as máscaras/warps de uma rodada: a pasta
    continua sendo uma rodada anterior (review #51)."""
    out = tmp_path / "out"
    out.mkdir()
    (out / kept).write_bytes(b"x")
    src = _png_dir(tmp_path / "in", ["a.png"])
    assert diag_live.main(["--input", str(src), "--out", str(out)]) == 2
    assert (out / kept).read_bytes() == b"x"


# --- terceira rodada do review do PR #51 --------------------------------


@pytest.mark.parametrize("kept", ["1000_raw.png", "10000_mask.png", "0001_warp.png"])
def test_output_guard_catches_four_plus_digit_frame_prefixes(
    tmp_path, fake_models, kept
):
    """Rodadas de 1000+ quadros usam 4+ dígitos; a guarda só via 3 e a
    rodada seguinte sobrescrevia o PNG guardado (review #51)."""
    out = tmp_path / "out"
    out.mkdir()
    (out / kept).write_bytes(b"x")
    src = _png_dir(tmp_path / "in", ["a.png"])
    assert diag_live.main(["--input", str(src), "--out", str(out)]) == 2
    assert (out / kept).read_bytes() == b"x"


def test_unrelated_png_names_do_not_trip_the_guard(tmp_path, fake_models):
    out = tmp_path / "out"
    out.mkdir()
    (out / "notas.png").write_bytes(b"x")
    (out / "12_x.png").write_bytes(b"x")  # 2 dígitos: não é artefato nosso
    src = _png_dir(tmp_path / "in", ["a.png"])
    assert diag_live.main(["--input", str(src), "--out", str(out)]) == 0


def test_fps_window_opens_at_the_first_good_read():
    """Com --warmup 0 o 1º read dá timeout (> a janela de FPS) e depois a
    câmera entrega normalmente: o FPS não pode ficar 0 (review #51)."""
    clk = _FakeClock()

    class _SlowStart(_Capture):
        started = False

        def read(self):
            if not self.started:
                self.started = True
                clk.sleep(0.2)  # timeout inicial, maior que a janela
                return False, None
            clk.sleep(0.002)
            return super().read()

    cap = _SlowStart(good=10_000)
    fps = [
        f
        for _frame, f in diag_live._frames_from_source(
            "0",
            2,
            0.0,
            0.0,
            stall_s=5.0,
            opener=lambda _s: cap,
            fps_burst_s=0.05,
            clock=clk,
            sleep=clk.sleep,
        )
    ]
    assert len(fps) == 2 and fps[0] > 50


def test_summary_write_failure_is_a_clear_error_not_a_traceback(
    monkeypatch, tmp_path, fake_models, capsys
):
    """Volume cheio no resumo: erro nomeando o arquivo, saída 1, e a linha
    de resumo do terminal ainda sai (review #51)."""
    real = Path.write_text

    def full_disk(self, *args, **kwargs):
        if self.name == "summary.json":
            raise OSError(28, "No space left on device")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", full_disk)
    src = _png_dir(tmp_path / "in", ["a.png"])
    code = diag_live.main(["--input", str(src), "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 1
    assert "summary.json" in captured.err and "No space left" in captured.err
    assert "1 quadros" in captured.out


# --- quarta rodada do review do PR #51 + --orientation -----------------


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "abc"])
@pytest.mark.parametrize("flag", ["--warmup", "--spacing"])
def test_non_finite_or_negative_seconds_are_a_usage_error(
    monkeypatch, tmp_path, fake_models, bad, flag, capsys
):
    """--warmup nan nunca satisfazia o `>=` do aquecimento: a ferramenta
    travava com a câmera funcionando (review #51). Recusa no parser, sem
    abrir a câmera."""
    opened = []
    monkeypatch.setattr(
        diag_live, "_frames_from_source", lambda *a, **k: opened.append(a)
    )
    with pytest.raises(SystemExit) as error:
        diag_live.main(["--source", "0", flag, bad, "--out", str(tmp_path / "o")])
    assert error.value.code == 2
    assert opened == []
    assert "finito" in capsys.readouterr().err or bad == "abc"


@pytest.mark.parametrize(
    "argv", [["--stall-timeout", "0"], ["--stall-timeout", "nan"], ["--frames", "0"]]
)
def test_other_numeric_options_are_validated(tmp_path, fake_models, argv):
    with pytest.raises(SystemExit) as error:
        diag_live.main(["--source", "0", *argv, "--out", str(tmp_path / "o")])
    assert error.value.code == 2


def test_partial_summary_hint_only_when_it_was_written(
    monkeypatch, tmp_path, fake_models, capsys
):
    """Falhou o diagnóstico E a escrita do resumo: não apontar para um
    "resumo parcial" que não existe (review #51)."""
    real = Path.write_text

    def full_disk(self, *args, **kwargs):
        if self.name == "summary.json":
            raise OSError(28, "No space left on device")
        return real(self, *args, **kwargs)

    def dying(*_a, **_k):
        yield np.full((240, 320, 3), 150, np.uint8), 30.0
        raise diag_live.SourceExhausted("fonte parou de entregar")

    monkeypatch.setattr(Path, "write_text", full_disk)
    monkeypatch.setattr(diag_live, "_frames_from_source", dying)
    code = diag_live.main(["--source", "0", "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 1
    assert "resumo parcial em" not in err
    assert "não pôde ser gravado" in err and "No space left" in err


def test_orientation_flag_deita_a_grade_do_warp(tmp_path, fake_models):
    src = _png_dir(tmp_path / "in", ["a.png"])
    out = tmp_path / "out"
    assert (
        diag_live.main(
            ["--input", str(src), "--out", str(out), "--orientation", "paisagem"]
        )
        == 0
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["orientation"] == "paisagem"
    assert summary["frames"][0]["grids"]["warp"] == [1200, 850]
    assert cv2.imread(str(out / "000_warp.png")).shape[:2] == (850, 1200)
