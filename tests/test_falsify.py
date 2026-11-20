"""Contratos do falsificador: amostragem, mutação e placar."""

import numpy as np
import pytest
from lab.falsify import _BOUNDS, mutate_config, sample_config, score_report
from lab.sim_source import SyntheticCapture

_FEATURE_KEYS = (
    "hand",
    "erase_stroke",
    "erase_rect",
    "redraw_text",
    "move",
    "lift",
    "return_pose",
    "grab_hand",
    "rest_hand_seconds",
    "sweep_seconds",
    "auto_exposure",
    "auto_white_balance",
    "motion_blur",
    "hand_shadow",
    "jpeg_quality",
    "ink_sheen",
    "page_curl",
)


def _select_covering_indices(configs):
    """Amostra determinística cobrindo categorias, booleans e extremos."""
    selected: set[int] = set()
    for key in _FEATURE_KEYS:
        seen_vals: set[bool] = set()
        for idx, cfg in enumerate(configs):
            params = cfg["params"]
            if key not in params:
                continue
            val = params[key]
            if isinstance(val, bool):
                if val not in seen_vals:
                    selected.add(idx)
                    seen_vals.add(val)
                if len(seen_vals) == 2:
                    break
            else:
                selected.add(idx)
                break
    for key in _BOUNDS:
        present = [
            (float(cfg["params"][key]), idx)
            for idx, cfg in enumerate(configs)
            if key in cfg["params"]
        ]
        if present:
            selected.add(min(present)[1])
            selected.add(max(present)[1])
    for key in ("erase_rect", "lift", "move", "return_pose"):
        present = []
        for idx, cfg in enumerate(configs):
            if key in cfg["params"]:
                val = cfg["params"][key]
                metric = (
                    float(val[2] * val[3])
                    if key == "erase_rect"
                    else float(np.linalg.norm(val))
                )
                present.append((metric, idx))
        if present:
            selected.add(min(present)[1])
            selected.add(max(present)[1])
    return sorted(selected)


def test_sampled_configs_build_valid_captures():
    rng = np.random.default_rng(11)
    configs = [sample_config(rng, index) for index in range(40)]
    indices = sorted(set(range(10)) | set(_select_covering_indices(configs)))
    for idx in indices:
        cfg = configs[idx]
        capture = SyntheticCapture(cfg["text"], seed=cfg["seed"], **cfg["params"])
        assert capture.frame_count > 0
        ok, frame = capture.read()
        assert ok and frame.shape == (512, 512, 3)
        capture.release()


def test_mutation_stays_valid_and_near_parent():
    rng = np.random.default_rng(12)
    parent = sample_config(rng, 0)
    children = [mutate_config(parent, rng) for _ in range(40)]
    selected: set[int] = set(range(8))
    seen_mutated: set[str] = set()
    for idx, child in enumerate(children):
        for key, value in child["params"].items():
            if key not in seen_mutated and parent["params"].get(key) != value:
                seen_mutated.add(key)
                selected.add(idx)
    for idx in sorted(selected):
        child = children[idx]
        capture = SyntheticCapture(child["text"], seed=child["seed"], **child["params"])
        assert capture.frame_count > 0
        capture.release()


def test_mutation_never_leaves_parameter_contract():
    """Mutação que escapasse do contrato do SyntheticCapture viraria
    score=-1 por ERRO — não por falha do pipeline. Cada campo mutável
    tem faixa própria."""
    rng = np.random.default_rng(42)
    children = []
    for index in range(15):
        parent = sample_config(rng, index)
        for _ in range(25):
            child = mutate_config(parent, rng)
            params = child["params"]
            for key, value in params.items():
                if key in ("ink_sheen", "page_curl"):
                    assert 0.0 <= value <= 1.0, (key, value)
                if key == "jpeg_quality":
                    assert 10 <= value <= 100
                if key == "hand_scale":
                    assert 0 < value <= 4
                if key == "fps":
                    assert 0 < value <= 120
                if key == "erase_rect":
                    x, y, w, h = value
                    assert x >= 0 and y >= 0 and w >= 1 and h >= 1
                    assert x + w <= 360 and y + h <= 510
                if key in ("move", "return_pose", "lift"):
                    assert all(np.isfinite(v) for v in value)
                if key == "lift":
                    assert float(np.linalg.norm(value)) >= 1.0
            children.append(child)

    boundary_configs = [
        {
            "text": "ab",
            "seed": 1,
            "params": {
                "fps": 4.0,
                "stroke_seconds": 0.05,
                "warmup_seconds": 0.0,
                "settle_seconds": 0.2,
                "hold_seconds": 0.0,
                "erase_rect": (0, 0, 1, 1),
                "erase_seconds": 0.1,
                "move": (-600.0, -600.0, -180.0),
                "move_seconds": 0.1,
                "lift": (1.0, 0.0),
                "absent_seconds": 0.1,
                "lift_seconds": 0.1,
                "return_seconds": 0.1,
                "return_pose": (-600.0, -600.0, -180.0),
                "rest_hand_seconds": 0.0,
                "sweep_seconds": 0.0,
                "height": 6.0,
                "hand": False,
                "grab_hand": False,
                "auto_exposure": False,
                "auto_white_balance": False,
                "motion_blur": False,
                "hand_shadow": False,
                "hand_scale": 0.3,
                "ink_sheen": 0.0,
                "page_curl": 0.0,
                "jpeg_quality": 10,
            },
        },
        {
            "text": "ab",
            "seed": 2,
            "params": {
                "fps": 60.0,
                "stroke_seconds": 0.8,
                "warmup_seconds": 0.4,
                "settle_seconds": 0.5,
                "hold_seconds": 0.5,
                "erase_rect": (359, 509, 1, 1),
                "erase_seconds": 0.5,
                "redraw_text": "ab",
                "move": (600.0, 600.0, 180.0),
                "move_seconds": 0.5,
                "lift": (400.0, 400.0),
                "absent_seconds": 0.5,
                "lift_seconds": 0.5,
                "return_seconds": 0.5,
                "return_pose": (600.0, 600.0, 180.0),
                "rest_hand_seconds": 0.5,
                "sweep_seconds": 0.5,
                "height": 60.0,
                "hand": True,
                "grab_hand": True,
                "auto_exposure": True,
                "auto_white_balance": True,
                "motion_blur": True,
                "hand_shadow": True,
                "hand_scale": 4.0,
                "ink_sheen": 1.0,
                "page_curl": 1.0,
                "jpeg_quality": 100,
            },
        },
    ]
    sample_indices = _select_covering_indices(children)
    for cfg in [children[i] for i in sample_indices] + boundary_configs:
        capture = SyntheticCapture(cfg["text"], seed=cfg["seed"], **cfg["params"])
        assert capture.frame_count > 0
        capture.release()


def test_run_trial_pipeline_crash_is_failing_outcome(tmp_path, monkeypatch):
    """Crash dentro da auditoria é desfecho falho do pipeline, não
    cenário inválido: score máximo (elegível a elite/mutação), erro
    registrado — e sem keep o vídeo gerado continua sem dono."""
    import lab.falsify as fz

    def fake_export(capture, output):
        output.write_bytes(b"x")
        output.with_suffix(".gt.npz").write_bytes(b"x")
        output.with_suffix(".json").write_bytes(b"x")
        output.with_name(output.stem + "_ink_full.npy").write_bytes(b"x")
        return {"frames": 1}

    monkeypatch.setattr(fz, "export_video", fake_export)
    monkeypatch.setattr(
        fz,
        "audit",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("auditoria falhou")),
    )
    cfg = {"text": "ab", "seed": 1, "params": {"fps": 8, "settle_seconds": 0.3}}
    result = fz.run_trial(cfg, tmp_path, "bad", models={}, keep_artifacts=False)
    assert result["score"] == fz.PIPELINE_CRASH_SCORE
    assert "RuntimeError" in result["pipeline_error"]
    assert result["verdict"]["status"].startswith("pipeline_error")
    for suffix in (".mp4", ".gt.npz", ".json"):
        assert not (tmp_path / f"bad{suffix}").exists()
    assert not (tmp_path / "bad_ink_full.npy").exists()


def test_run_trial_pipeline_crash_keeps_artifacts(tmp_path, monkeypatch):
    """Com keep_artifacts o vídeo que derruba o pipeline fica
    preservado — é a evidência do defeito, não lixo de retry."""
    import lab.falsify as fz

    def fake_export(capture, output):
        output.write_bytes(b"x")
        output.with_suffix(".gt.npz").write_bytes(b"x")
        return {"frames": 1}

    monkeypatch.setattr(fz, "export_video", fake_export)
    monkeypatch.setattr(
        fz,
        "audit",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("auditoria falhou")),
    )
    cfg = {"text": "ab", "seed": 1, "params": {"fps": 8, "settle_seconds": 0.3}}
    result = fz.run_trial(cfg, tmp_path, "bad", models={}, keep_artifacts=True)
    assert result["score"] == fz.PIPELINE_CRASH_SCORE
    assert result["artifacts"] == str(tmp_path / "bad.mp4")
    assert (tmp_path / "bad.mp4").exists()


def test_run_trial_config_failure_still_raises(tmp_path, monkeypatch):
    """Falha de configuração/export — cenário que nem chegou ao
    pipeline — continua propagando para o score -1 do chamador."""
    import lab.falsify as fz

    monkeypatch.setattr(
        fz,
        "SyntheticCapture",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("parâmetro impossível")
        ),
    )
    cfg = {"text": "ab", "seed": 1, "params": {"fps": 8}}
    with pytest.raises(ValueError):
        fz.run_trial(cfg, tmp_path, "bad", models={}, keep_artifacts=False)


def test_score_report_weights_documented_failures():
    perfect = {
        "verdict": {"final_iou": 0.9},
        "checkpoints": [{"iou": 0.9}],
    }
    score, comps = score_report(perfect)
    assert score == pytest.approx(0.1 + 0.5 * 0.1)
    assert comps["no_canvas"] == 0.0

    lost = {"verdict": {"final_iou": None, "status": "canvas ausente"}}
    score, comps = score_report(lost)
    assert comps["no_canvas"] == 1.0 and score >= 2.0

    ghost = {
        "verdict": {
            "final_iou": 0.8,
            "residual_confirmed_in_erased_px": 8000,
            "redraw_confirmed_recall": 0.2,
        },
        "checkpoints": [{"iou": 0.8}],
    }
    score, comps = score_report(ghost)
    assert comps["erase_residual"] == 1.0
    assert comps["redraw_loss"] == pytest.approx(0.8)
    worst = {
        "verdict": {"final_iou": 0.8},
        "checkpoints": [{"iou": 0.8}, {"iou": 0.1}],
    }
    assert score_report(worst)[1]["worst_phase_loss"] == pytest.approx(0.9)


def test_score_report_ausencia_final_nao_penaliza():
    from lab.falsify import score_report

    report = {
        "verdict": {"status": "ausência (folha fora de cena)", "absence": True},
        "checkpoints": [
            {"frame": 10, "iou": 0.9},
            {"frame": 20, "status": "ausência (folha fora de cena)", "absence": True},
        ],
    }
    score, comp = score_report(report)
    assert comp["no_canvas"] == 0.0
    assert comp["iou_loss"] == 0.0
    assert comp["worst_phase_loss"] == pytest.approx(0.1)


def test_score_report_sem_canvas_real_ainda_reprova():
    from lab.falsify import score_report

    report = {
        "verdict": {"status": "canvas ausente no evento final"},
        "checkpoints": [{"frame": 20, "status": "sem âncora/canvas"}],
    }
    score, comp = score_report(report)
    assert comp["no_canvas"] == 1.0
    assert comp["iou_loss"] == 1.0
    assert comp["worst_phase_loss"] == 1.0


def test_audit_models_default_aceito():
    # models=None é o default; audit com tupla vazia quebrava no .get.
    import inspect

    from lab import eval_scenario

    sig = inspect.signature(eval_scenario.audit)
    assert sig.parameters["models"].default is None


def test_final_checkpoint_propaga_absence():
    # O dict reconstruido sem iou dropava a flag: verdict["absence"]
    # saia sempre False e o score_report cobrava a ausencia.
    from lab.eval_scenario import _final_checkpoint

    checkpoints = [
        {"frame": 10, "iou": 0.9},
        {
            "frame": 20,
            "phase": "lift_out",
            "status": "ausência (folha fora de cena)",
            "absence": True,
            "confirmed_px": 500,
        },
    ]
    events = [{"end": 11}, {"kind": "lift_out", "end": 21}]
    final = _final_checkpoint(checkpoints, events)
    assert final["absence"] is True
    assert final["confirmed_px"] == 500
    score, comp = score_report({"verdict": final, "checkpoints": checkpoints})
    assert comp["no_canvas"] == 0.0
    assert comp["iou_loss"] == 0.0
