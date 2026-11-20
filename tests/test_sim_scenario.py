"""Cenário escrever→apagar→redesenhar→mover da fonte sintética."""

import json

import cv2
import numpy as np
import pytest
from lab import sim_source
from lab.sim_source import SyntheticCapture, export_video


def drain(capture):
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            return frames
        frames += 1


def scenario(**kwargs):
    options = dict(
        fps=10,
        stroke_seconds=0.2,
        warmup_seconds=0.2,
        settle_seconds=0.3,
        hold_seconds=0.4,
        erase_stroke=-1,
        redraw_text="ui",
        erase_seconds=0.3,
    )
    options.update(kwargs)
    return SyntheticCapture("Draw ok", **options)


def kinds(capture):
    return [phase["kind"] for phase in capture.phases]


def test_default_timeline_unchanged():
    capture = SyntheticCapture("ab", fps=4, stroke_seconds=0.5, settle_seconds=0.5)
    assert kinds(capture) == ["warmup", "write", "settle"]
    assert capture.frame_count == capture.phases[-1]["end"]
    assert capture.erase_rect is None and not capture.redraw_specs
    assert np.array_equal(capture._quad_now, capture.quad)


def test_scenario_timeline_order_and_bounds():
    capture = scenario(move=(20, 10, 1.5))
    assert kinds(capture) == [
        "warmup",
        "write",
        "hold",
        "erase",
        "hold",
        "write",
        "hold",
        "move",
        "settle",
    ]
    start = 0
    for phase in capture.phases:
        assert phase["start"] == start and phase["end"] > start
        start = phase["end"]
    assert capture.frame_count == start


def test_erase_validations():
    with pytest.raises(ValueError, match="alternativas"):
        SyntheticCapture("a", erase_stroke=0, erase_rect=(10, 10, 20, 20))
    with pytest.raises(ValueError, match="exige"):
        SyntheticCapture("a", redraw_text="b")
    with pytest.raises(ValueError, match="fora"):
        SyntheticCapture("ab", erase_stroke=99)
    with pytest.raises(ValueError, match="fora"):
        SyntheticCapture("ab", erase_rect=(-5, 0, 10, 10))
    with pytest.raises(ValueError, match="move"):
        SyntheticCapture("ab", move=(1, 2, 3, 4))


def drain_until(capture, frame_index):
    """Lê até o frame inclusive; útil para parar numa fronteira de fase."""
    while capture.get(cv2.CAP_PROP_POS_FRAMES) <= frame_index:
        ok, _ = capture.read()
        if not ok:
            break


def _raster(bundle, indices):
    mask = np.zeros(bundle["page_ink_full"].shape, np.uint8)
    points, offsets = bundle["points"], bundle["offsets"]
    for index in indices:
        geo = points[offsets[index] : offsets[index + 1]]
        cv2.polylines(
            mask, [geo.astype(np.int32)], False, 255, int(bundle["thickness"][index])
        )
    return mask


def test_erasure_removes_ink_only_inside_rect():
    capture = scenario()
    assert capture.erase_rect is not None
    x, y, w, h = capture.erase_rect
    inside_before = int(capture.page_ink_full[y : y + h, x : x + w].sum())
    assert inside_before > 0
    erase_end = next(p["end"] for p in capture.phases if p["kind"] == "erase")
    drain_until(capture, erase_end - 1)
    assert not capture._ink[y : y + h, x : x + w].any()
    # fora do retângulo, a tinta dos traços sobrevive
    outside = capture._ink.copy()
    outside[y : y + h, x : x + w] = 0
    assert outside.any()


def test_redraw_lands_on_erased_area():
    capture = scenario()
    assert capture.redraw_specs
    points = np.concatenate([spec[3] for spec in capture.redraw_specs])
    x, y, w, h = capture.erase_rect
    cx, cy = points.mean(axis=0)
    assert x - w <= cx <= x + 2 * w and y - h <= cy <= y + 2 * h
    drain(capture)
    # GT por fase: plano inclui todos os traços; estado final sem o retângulo
    assert set(np.unique(capture.stroke_phase)) == {0, 1}
    assert capture.stroke_erased.any()


def test_move_shifts_final_quad():
    moved = scenario(move=(30, 15, 2.0))
    drain(moved)
    assert not np.allclose(moved._quad_now, moved.quad)
    still = scenario(move=None)
    drain(still)
    assert np.array_equal(still._quad_now, still.quad)


def test_ground_truth_bundle_phase_fields(tmp_path):
    capture = scenario(move=(20, 10, 1.5))
    drain(capture)
    gt = tmp_path / "scenario.gt.npz"
    capture.save_ground_truth(gt)
    with np.load(gt, allow_pickle=False) as bundle:
        for key in (
            "page_ink_final",
            "stroke_phase",
            "stroke_erased",
            "erase_rect",
            "events",
            "quad_final",
            "page_to_scene_final",
        ):
            assert key in bundle.files
        assert bundle["page_ink_final"].shape == bundle["page_ink_full"].shape
        assert int(bundle["stroke_phase"].max()) == 1
        events = json.loads(str(bundle["events"]))
        # estado final: traço apagado some, redesenho fica no lugar
        final = bundle["page_ink_final"] > 0
        full = bundle["page_ink_full"] > 0
        assert not np.array_equal(final, full)
        erased_mask = _raster(bundle, np.nonzero(bundle["stroke_erased"])[0]) > 0
        redraw_mask = _raster(bundle, np.nonzero(bundle["stroke_phase"] == 1)[0]) > 0
        redraw_mask = cv2.dilate(redraw_mask.astype(np.uint8), np.ones((3, 3))) > 0
        x, y, w, h = (int(v) for v in bundle["erase_rect"])
        leftover = np.zeros_like(final)
        leftover[y : y + h, x : x + w] = True
        assert not (final & erased_mask & leftover & ~redraw_mask).any()
        assert (final & redraw_mask).any()
    assert [e["kind"] for e in events] == kinds(capture)


def test_scenario_is_deterministic():
    a, b = scenario(), scenario()
    assert a.erase_rect == b.erase_rect
    assert len(a.redraw_specs) == len(b.redraw_specs)
    for sa, sb in zip(a.redraw_specs, b.redraw_specs, strict=False):
        assert np.array_equal(sa[3], sb[3])


def test_export_manifest_marks_scenario(tmp_path):
    capture = scenario()
    manifest = export_video(capture, tmp_path / "s.mp4")
    assert manifest["pose"] == "scenario"
    assert manifest["strokes_redraw"] == len(capture.redraw_specs)
    assert manifest["strokes_erased"] == int(capture.stroke_erased.sum())
    assert [p["kind"] for p in manifest["phases"]] == kinds(capture)
    fixed = export_video(
        SyntheticCapture("ab", fps=4, settle_seconds=0.5), tmp_path / "f.mp4"
    )
    assert fixed["pose"] == "fixed"


def test_lift_return_timeline_and_absent_scene():
    """lift & return: a folha some de cena por inteiro (só mesa nos
    frames de absent) e volta na pose declarada por return_pose."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        move=None,
        lift=(300, 200),
        return_pose=(-40, 30, 18),
        absent_seconds=0.8,
        lift_seconds=0.4,
        return_seconds=0.4,
    )
    assert kinds(capture) == [
        "warmup",
        "write",
        "lift_out",
        "absent",
        "return",
        "settle",
    ]
    masks = {}
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        if capture.last_scene is not None:
            masks[capture._position - 1] = capture.last_scene["mask"]
    import lab.gen_synthetic as gs

    for phase in capture.phases:
        end = phase["end"] - 1
        if phase["kind"] == "absent":
            # durante a ausência, nada de papel nem tinta na cena
            assert not (masks[end] == gs.PAPER).any()
            assert not (masks[end] == gs.INK).any()
        elif phase["kind"] == "return":
            assert (masks[end] == gs.PAPER).sum() > 0
    # a volta obedece a return_pose: pose final difere da inicial
    assert not np.allclose(capture._quad_now, capture.quad)


def test_lift_return_ground_truth(tmp_path):
    """GT declara as fases novas e a pose final rodada."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        move=None,
        lift=(200, 150),
        return_pose=(0, 0, 25),
        absent_seconds=0.5,
    )
    drain(capture)
    gt = tmp_path / "lift.gt.npz"
    capture.save_ground_truth(gt)
    with np.load(gt, allow_pickle=False) as bundle:
        events = json.loads(str(bundle["events"]))
        assert [e["kind"] for e in events] == kinds(capture)
        assert "absent" in [e["kind"] for e in events]
        # a tinta sobrevive à saída: estado final = plano completo
        np.testing.assert_array_equal(
            bundle["page_ink_final"] > 0, bundle["page_ink_full"] > 0
        )
        assert not np.allclose(bundle["quad_final"], bundle["quad"])
        assert not np.allclose(bundle["page_to_scene_final"], bundle["page_to_scene"])


def test_events_carry_quad_end_matching_simulated_pose(tmp_path):
    """quad_end declara a pose no ÚLTIMO frame de cada fase geométrica
    — o avaliador reconstrói poses intermediárias (move+lift+return),
    não só inicial/final."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        move=(40, 20, 8),
        lift=(200, 150),
        return_pose=(0, 0, 10),
        lift_seconds=0.3,
        absent_seconds=0.3,
        return_seconds=0.3,
    )
    quads = {}
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        quads[capture._position - 1] = capture._quad_now.copy()
    gt = tmp_path / "q.gt.npz"
    capture.save_ground_truth(gt)
    with np.load(gt, allow_pickle=False) as bundle:
        events = json.loads(str(bundle["events"]))
        for event in events:
            if event["kind"] in ("move", "lift_out", "return"):
                quad_end = np.asarray(event["quad_end"], np.float32)
                assert quad_end.shape == (4, 2)
                assert np.isfinite(quad_end).all()
                np.testing.assert_allclose(quad_end, quads[event["end"] - 1], atol=1e-4)
            else:
                assert "quad_end" not in event
        # a última fase geométrica termina na pose final do bundle
        np.testing.assert_allclose(
            np.asarray(events[-2]["quad_end"], np.float32),
            bundle["quad_final"],
            atol=1e-4,
        )


def test_lift_validations():
    with pytest.raises(ValueError):
        SyntheticCapture("ab", return_pose=(10, 0, 15))
    with pytest.raises(ValueError):
        SyntheticCapture("ab", lift=(0, 0))
    with pytest.raises(ValueError):
        SyntheticCapture("ab", lift=(10, 0, 0))


def test_lift_starts_from_post_move_pose():
    """Combinação move+lift: a saída parte da pose DEPOIS do move —
    o quad_at_lift é composto em __init__, não pode saltar de volta
    para a pose inicial quando o lift_out começa."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        move=(60, 30, 5),
        lift=(300, 200),
        return_pose=(0, 0, 0),
        absent_seconds=0.4,
        lift_seconds=0.4,
        return_seconds=0.4,
    )
    lift_out = next(p for p in capture.phases if p["kind"] == "lift_out")
    move_end = next(p for p in capture.phases if p["kind"] == "move")
    # o quad_from do lift é a pose pós-move: distinto da pose inicial
    # (que é o quad_from do move) e igual ao quad que o move produz
    # no seu último frame — sem salto na transição
    assert not np.allclose(lift_out["quad_from"], move_end["quad_from"])
    moved_quads = {}
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        moved_quads[capture._position - 1] = capture._quad_now.copy()
    end_move = moved_quads[move_end["end"] - 1]
    np.testing.assert_allclose(lift_out["quad_from"], end_move, atol=1.0)


def test_rest_hand_and_sweep_timeline():
    """Palma apoiada após cada hold + sweep antes da geometria. As
    fases são neutras de GT: a tinta não muda com a mão por cima."""
    capture = scenario(
        move=(40, 20, 0),
        rest_hand_seconds=0.5,
        sweep_seconds=0.6,
    )
    assert kinds(capture) == [
        "warmup",
        "write",
        "hold",
        "rest_hand",
        "erase",
        "hold",
        "rest_hand",
        "write",
        "hold",
        "rest_hand",
        "sweep",
        "move",
        "settle",
    ]
    ink_at = {}
    hand_at = {}
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        idx = capture._position - 1
        ink_at[idx] = capture._ink.copy()
        import lab.gen_synthetic as gs

        hand_at[idx] = (capture.last_scene["mask"] == gs.HAND).sum()
    rest = next(p for p in capture.phases if p["kind"] == "rest_hand")
    sweep = next(p for p in capture.phases if p["kind"] == "sweep")
    # a palma pousa sobre o canvas e o sweep a mantém em cena
    assert hand_at[rest["end"] - 1] > 0
    assert hand_at[sweep["start"] + 1] > 0
    assert hand_at[sweep["end"] - 1] > 0
    # mão não muda a tinta: GT antes/depois do rest é idêntico
    np.testing.assert_array_equal(ink_at[rest["start"] - 1], ink_at[rest["end"] - 1])


def test_rest_hand_without_erase_still_rests():
    """Sem borracha não há hold natural: rest_hand_seconds cria o
    repouso após a escrita — pousar depois de escrever é o gesto mais
    comum da mesa."""
    capture = SyntheticCapture(
        "ab",
        fps=10,
        stroke_seconds=0.2,
        warmup_seconds=0.2,
        settle_seconds=0.3,
        rest_hand_seconds=0.4,
    )
    assert kinds(capture) == ["warmup", "write", "hold", "rest_hand", "settle"]


def test_grab_hand_rides_the_sheet():
    """A mão que segura a folha acompanha o quad no move/lift/return
    e some de cena com ela no absent."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        move=None,
        lift=(400, 300),
        return_pose=(0, 0, 10),
        absent_seconds=0.4,
        grab_hand=True,
    )
    import lab.gen_synthetic as gs

    hand_at = {}
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        hand_at[capture._position - 1] = (capture.last_scene["mask"] == gs.HAND).sum()
    lift_out = next(p for p in capture.phases if p["kind"] == "lift_out")
    absent = next(p for p in capture.phases if p["kind"] == "absent")
    ret = next(p for p in capture.phases if p["kind"] == "return")
    # pegada na folha no início da saída e no fim do retorno
    assert hand_at[lift_out["start"]] > 0
    assert hand_at[ret["end"] - 1] > 0
    # ausência: nem folha nem mão
    assert hand_at[absent["end"] - 1] == 0


def test_hand_mass_validations():
    with pytest.raises(ValueError):
        SyntheticCapture("ab", rest_hand_seconds=-1)
    with pytest.raises(ValueError):
        SyntheticCapture("ab", sweep_seconds=-0.5)
    with pytest.raises(ValueError):
        SyntheticCapture("ab", hand_scale=0)
    with pytest.raises(ValueError):
        SyntheticCapture("ab", hand_scale=99)


def test_hand_mass_ground_truth_neutral(tmp_path):
    """As fases de mão exportam kind/start/end e o GT não muda —
    palma e sweep não são tinta."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        move=None,
        lift=None,
        rest_hand_seconds=0.3,
        sweep_seconds=0.4,
    )
    drain(capture)
    gt = tmp_path / "hand.gt.npz"
    capture.save_ground_truth(gt)
    with np.load(gt, allow_pickle=False) as bundle:
        events = json.loads(str(bundle["events"]))
        assert [e["kind"] for e in events] == kinds(capture)
        assert "rest_hand" in [e["kind"] for e in events]
        assert "sweep" in [e["kind"] for e in events]
        np.testing.assert_array_equal(
            bundle["page_ink_final"] > 0, bundle["page_ink_full"] > 0
        )


def test_return_lerp_never_collapses_the_sheet():
    """Retorno de meia volta (180 graus) com numero par de quadros:
    o lerp de cantos colapsava a folha no ponto medio — cantos
    opostos cruzam o centro. Transicoes que degeneram interpolam
    a POSE rigida (rotacao em torno do centro + translacao)."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        lift=(300, 200),
        return_pose=(0, 0, 180),
        absent_seconds=0.2,
        lift_seconds=0.2,
        return_seconds=0.4,  # 4 quadros a 10 fps — par
    )
    phase = next(p for p in capture.phases if p["kind"] == "return")
    n = phase["end"] - phase["start"]
    assert n % 2 == 0
    endpoint_area = min(
        abs(cv2.contourArea(phase["quad_from"].astype(np.float32))),
        abs(cv2.contourArea(phase["quad_to"].astype(np.float32))),
    )
    mid_local = n // 2 - 1  # t = 0.5 exato
    capture._step_quad_lerp(phase, mid_local)
    mid_area = abs(cv2.contourArea(capture._quad_now.astype(np.float32)))
    assert mid_area > 0.5 * endpoint_area
    # O ponto medio rigido e congruente com as pontas: mesmos
    # lados, area cheia.
    first = phase["quad_from"]
    assert np.isclose(
        np.linalg.norm(first[1] - first[0]),
        np.linalg.norm(capture._quad_now[1] - capture._quad_now[0]),
        rtol=0.05,
    )


def test_page_velocity_counts_pure_rotation():
    """O blur da folha usava a media vetorial dos cantos, que
    cancela rotacao pura em torno do centro: move=(0,0,20) saia
    sem blur. A estimativa agora e a media de |deslocamento| por
    canto."""
    import math as _math

    from lab.sim_source import _quad_step_velocity

    center = np.float32([320, 320])
    base = np.float32([[100, -40], [40, 100], [-100, 40], [-40, -100]])
    rad = _math.radians(20)
    rot = np.float32(
        [[_math.cos(rad), -_math.sin(rad)], [_math.sin(rad), _math.cos(rad)]]
    )
    delta = (base @ rot.T + center) - (base + center)
    vx, vy = _quad_step_velocity(delta)
    # Rotacao pura: a media vetorial seria ~0; a media de
    # |deslocamento| por canto e positiva.
    assert _math.hypot(vx, vy) > 1.0
    # Translacao pura sai igual ao deslocamento medio de antes.
    shift = np.full((4, 2), [3.0, -2.0], dtype=np.float32)
    assert _quad_step_velocity(shift) == (3.0, -2.0)


def test_erase_touching_only_the_stroke_edge_erases_it():
    """Apagamento que toca só a espessura do traço (não a linha
    central) também marca stroke_erased — traço meio apagado não
    pode sair do GT como intacto."""
    from lab.sim_source import _polyline_rect_overlap

    # Linha horizontal em y=10, espessura 6 (raio 3): um retangulo
    # que começa em y=12 toca a borda do traço sem tocar o centro.
    line = np.array([[20, 10], [80, 10]], dtype=np.int32)
    assert _polyline_rect_overlap(line, (0, 12, 100, 20), 6)
    assert not _polyline_rect_overlap(line, (0, 14, 100, 20), 6)
    # Sem espessura informada, o comportamento anterior se mantém.
    assert not _polyline_rect_overlap(line, (0, 12, 100, 20))
    # Retângulo roçando o traço entre vértices (canto na vizinhança
    # de um segmento) também conta.
    assert _polyline_rect_overlap(line, (48, 7, 4, 4), 6)


def test_redraw_text_of_only_spaces_yields_no_specs():
    """Texto so de espacos nao planeja traço nenhum: _translate_specs
    chamava np.concatenate numa lista vazia."""
    capture = scenario(erase_stroke=0, redraw_text="   ")
    assert capture.redraw_specs == []


def test_non_collapsing_lerp_keeps_corner_interpolation():
    """Transicao sem colapso (lift_out translada) segue no lerp de
    cantos de sempre — mesma saida de antes, bit a bit."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        lift=(300, 200),
        absent_seconds=0.2,
        lift_seconds=0.4,
        return_seconds=0.4,
    )
    phase = next(p for p in capture.phases if p["kind"] == "lift_out")
    local = 1
    t = min(1.0, (local + 1) / (phase["end"] - phase["start"]))
    expected = phase["quad_from"] + t * (phase["quad_to"] - phase["quad_from"])
    capture._step_quad_lerp(phase, local)
    np.testing.assert_allclose(capture._quad_now, expected)


def test_polyline_rect_overlap_exclusive_border_not_erased():
    """A regiao apagada e [x, x+w) x [y, y+h): o ultimo pixel
    apagado e x+w-1. Um traco cuja linha central so encosta na
    borda exclusiva nao foi apagado — o rotulo nao pode
    contradizer a tinta restante."""
    rect = (10, 5, 20, 10)  # apaga x 10..29, y 5..14
    # Linha central na borda exclusiva direita: intacta.
    edge = np.array([[30, 5], [30, 15]], dtype=np.float32)
    assert not sim_source._polyline_rect_overlap(edge, rect, 1)
    edge_bottom = np.array([[10, 15], [29, 15]], dtype=np.float32)
    assert not sim_source._polyline_rect_overlap(edge_bottom, rect, 1)
    # Invadindo a regiao por um pixel: apagado.
    inside = np.array([[29, 5], [29, 15]], dtype=np.float32)
    assert sim_source._polyline_rect_overlap(inside, rect, 1)
    # So a espessura alcancando a regiao (dist == thickness/2):
    # apagado, como na correcao anterior.
    by_thickness = np.array([[31, 5], [31, 15]], dtype=np.float32)
    assert sim_source._polyline_rect_overlap(by_thickness, rect, 4)


def test_page_expected_matches_simulator_final(tmp_path):
    """O gabarito do avaliador segue a ordem do simulador: a borracha
    correu antes do redesenho, então traço de fase 1 dentro do
    retângulo apagado é tinta esperada — não falso positivo."""
    from lab.eval_scenario import _page_expected

    capture = scenario(move=(20, 10, 1.5))
    drain(capture)
    gt = tmp_path / "scenario.gt.npz"
    capture.save_ground_truth(gt)
    with np.load(gt, allow_pickle=False) as bundle:
        # fase de redraw aplicada e borracha já corrida: o esperado
        # do checkpoint final tem que ser exatamente o estado da folha
        expected = _page_expected(bundle, int(bundle["stroke_phase"].size), erased=True)
        np.testing.assert_array_equal(expected > 0, bundle["page_ink_final"] > 0)
        # e o redesenho DENTRO do retângulo apagado sobrevive no
        # gabarito (era o que o apagar-em-bloco zerava)
        x, y, w, h = (int(v) for v in bundle["erase_rect"])
        redraw = _raster(bundle, np.nonzero(bundle["stroke_phase"] == 1)[0]) > 0
        inside = np.zeros_like(redraw)
        inside[y : y + h, x : x + w] = True
        assert (expected > 0)[inside & redraw].all()


def test_final_checkpoint_terminal_canvas_loss():
    """Canvas perdido no evento final não empresta o IoU de fases
    anteriores: o veredito expõe o status em vez de métricas."""
    from lab.eval_scenario import _final_checkpoint

    events = [{"kind": "hold", "end": 10}, {"kind": "settle", "end": 20}]
    checkpoints = [
        {"frame": 9, "phase": "hold", "iou": 0.9, "confirmed_px": 100},
        {"frame": 19, "phase": "settle", "status": "sem âncora/canvas"},
    ]
    final = _final_checkpoint(checkpoints, events)
    assert "iou" not in final
    assert "canvas ausente" in final["status"]


def test_final_checkpoint_missing_terminal_event():
    """Vídeo truncado antes do último evento: veredito sem métrica,
    com status explicando — nunca o sucesso de uma fase anterior."""
    from lab.eval_scenario import _final_checkpoint

    events = [{"kind": "hold", "end": 10}, {"kind": "settle", "end": 20}]
    checkpoints = [{"frame": 9, "phase": "hold", "iou": 0.9}]
    final = _final_checkpoint(checkpoints, events)
    assert "iou" not in final
    assert "evento final" in final["status"]


def test_pipeline_summary_eh_json_serializavel():
    """anchors/registrations carregam ndarrays — sem o resumo o
    json.dumps do relatório quebrava e a auditoria não produzia
    saída nenhuma."""
    from lab.eval_scenario import _pipeline_summary

    quad = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
    reg = {"anchor": quad, "quad": quad, "submitted": quad, "frame": 3}
    pipeline = {
        "mask": np.ones((4, 4), np.uint8),
        "corners": quad,
        "anchors": [quad.copy()],
        "registrations": [reg],
        "registration_history": [reg, dict(reg, frame=7)],
        "canvas_shape": [60, 80],
        "frames": 10,
        "status": "ok",
    }
    summary = _pipeline_summary(pipeline)
    json.dumps(summary)  # não levanta
    assert summary["anchor_count"] == 1
    assert summary["registration_count"] == 1
    assert summary["registration_history_count"] == 2
    assert summary["anchor_corners"] == [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert summary["final_confirmed_px"] == 16
    assert "mask" not in summary and "registrations" not in summary
    assert "anchors" not in summary and "corners" not in summary
    assert "registration_history" not in summary
    assert pipeline["mask"] is not None  # não muta o original


def test_project_via_registrations_identidade():
    from lab.eval_scenario import _project_via_registrations

    height, width = 60, 80
    mask = np.zeros((30, 30), np.uint8)
    mask[5:15, 5:15] = 255
    anchor = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    reg = {"anchor": anchor, "quad": anchor, "submitted": None, "frame": None}
    out = _project_via_registrations(mask, [reg], 5, canvas_hw=(height, width))
    # âncora==quad → registro identidade; sem submetido/frame cai na
    # escala página→grade — mesmo resultado do caminho anterior.
    expected = cv2.warpPerspective(
        mask,
        np.diag([width / 30, height / 30, 1.0]),
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    assert np.array_equal(out > 0, expected > 0)


def test_project_via_registrations_filtra_por_frame():
    from lab.eval_scenario import _project_via_registrations

    mask = np.zeros((30, 30), np.uint8)
    mask[5:15, 5:15] = 255
    anchor = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    late = {"anchor": anchor, "quad": anchor, "submitted": None, "frame": 10}
    # Passada de frame posterior ao checkpoint não projeta.
    assert _project_via_registrations(mask, [late], 5, canvas_hw=(60, 80)) is None
    assert _project_via_registrations(mask, [late], 10, canvas_hw=(60, 80)) is not None
    # E sem registro nenhum também cai fora (fallback da pose).
    assert _project_via_registrations(mask, [], 10, canvas_hw=(60, 80)) is None
    assert _project_via_registrations(mask, ["lixo"], 10, canvas_hw=(60, 80)) is None


def test_project_via_registrations_unia_duas_poses():
    """Re-âncora sem reset: a tinta antiga fica onde a cadeia dela
    a depositou; a união cobre as duas posições da grade."""
    from lab.eval_scenario import _project_via_registrations

    mask = np.zeros((30, 30), np.uint8)
    mask[5:15, 5:15] = 255
    anchor_a = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    anchor_b = anchor_a + np.float32([8, 0])
    regs = [
        {"anchor": anchor_a, "quad": anchor_a, "submitted": None, "frame": None},
        # A segunda passada registrou a geometria de anchor_a sob a
        # grade da âncora nova — o conteúdo desloca na grade.
        {"anchor": anchor_b, "quad": anchor_a, "submitted": None, "frame": None},
    ]
    union = _project_via_registrations(mask, regs, 9, canvas_hw=(60, 80)) > 0
    single = _project_via_registrations(mask, regs[:1], 9, canvas_hw=(60, 80)) > 0
    assert union.sum() > single.sum()


def test_project_via_registrations_usa_quad_submetido_e_pose_real():
    """sub≠reg com a pose verdadeira do bundle: a cadeia é
    P(âncora)·P(reg)⁻¹·P(sub)·H — composição manual igual."""
    from lab.eval_scenario import _project_via_registrations

    from draw.paper import perspective_matrix

    height, width = 60, 80
    mask = np.zeros((30, 30), np.uint8)
    mask[5:15, 5:15] = 255
    anchor = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    sub = anchor + np.float32([2, 0])  # warp visto com viés
    quads = np.stack([anchor, anchor + np.float32([8, 0])])
    reg = {"anchor": anchor, "quad": anchor, "submitted": sub, "frame": 1}
    out = _project_via_registrations(
        mask, [reg], 9, canvas_hw=(height, width), quads=quads
    )
    destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    page_corners = np.float32([[0, 0], [30, 0], [30, 30], [0, 30]])
    chain = (
        perspective_matrix(anchor, destination)
        @ np.linalg.inv(perspective_matrix(anchor, destination))
        @ perspective_matrix(sub, destination)
        @ perspective_matrix(page_corners, quads[1])
    )
    expected = cv2.warpPerspective(
        mask, chain, (width, height), flags=cv2.INTER_NEAREST
    )
    assert np.array_equal(out > 0, expected > 0)
    # E diverge do caminho sem a pose submetida.
    naive = cv2.warpPerspective(
        mask,
        np.diag([width / 30, height / 30, 1.0]),
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    assert not np.array_equal(out > 0, naive > 0)


def test_project_via_registrations_min_frame_descarta_registros_antigos():
    """Traço de fase 1 projetado só por passadas que já o podiam
    ver: registro anterior à segunda escrita não entra na união."""
    from lab.eval_scenario import _project_via_registrations

    mask = np.zeros((30, 30), np.uint8)
    mask[5:15, 5:15] = 255
    anchor_a = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    anchor_b = anchor_a + np.float32([8, 0])
    early = {"anchor": anchor_a, "quad": anchor_a, "submitted": None, "frame": 2}
    late = {"anchor": anchor_b, "quad": anchor_a, "submitted": None, "frame": 6}
    union = (
        _project_via_registrations(
            mask, [early, late], 9, canvas_hw=(60, 80), min_frame=4
        )
        > 0
    )
    only_late = _project_via_registrations(mask, [late], 9, canvas_hw=(60, 80)) > 0
    # O registro de frame 2 ficou de fora da união com min_frame=4.
    assert np.array_equal(union, only_late)
    # E se nada passa do corte, cai fora para o fallback da pose.
    assert (
        _project_via_registrations(mask, [early], 9, canvas_hw=(60, 80), min_frame=4)
        is None
    )


def test_project_via_registrations_curvo_sem_pose_cai_no_fallback():
    """Bundle curvo sem histórico de quads (GT salvo antes de
    consumir o vídeo): a escala plana página→grade mentiria pior que
    o mapa denso do checkpoint — o registro é pulado e a função cai
    fora para o _project de sempre."""
    from lab.eval_scenario import _project_via_registrations

    mask = np.zeros((30, 30), np.uint8)
    mask[5:15, 5:15] = 255
    anchor = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    reg = {"anchor": anchor, "quad": anchor, "submitted": None, "frame": None}
    assert (
        _project_via_registrations(mask, [reg], 5, canvas_hw=(60, 80), curved=True)
        is None
    )
    # Sem a marca de curvatura o mesmo registro projeta plano.
    assert _project_via_registrations(mask, [reg], 5, canvas_hw=(60, 80)) is not None


def test_project_via_registrations_filtra_por_epoca():
    """Checkpoint de época 2 não projeta pelos registros da época 1:
    depois do reset do acumulador a história anterior descreve tinta
    que não existe mais — e o checkpoint anterior ao reset não pode
    ver os registros novos."""
    from lab.eval_scenario import _project_via_registrations

    mask = np.zeros((30, 30), np.uint8)
    mask[5:15, 5:15] = 255
    anchor_a = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    anchor_b = anchor_a + np.float32([8, 0])
    old = {"anchor": anchor_a, "quad": anchor_a, "submitted": None, "frame": 2}
    old["epoch"] = 1
    new = {"anchor": anchor_b, "quad": anchor_a, "submitted": None, "frame": 6}
    new["epoch"] = 2
    union = (
        _project_via_registrations(mask, [old, new], 9, canvas_hw=(60, 80), epoch=2) > 0
    )
    only_new = _project_via_registrations(mask, [new], 9, canvas_hw=(60, 80)) > 0
    assert np.array_equal(union, only_new)
    # Checkpoint da época 1 (antes do reset) vê só a história dela.
    old_only = (
        _project_via_registrations(mask, [old, new], 9, canvas_hw=(60, 80), epoch=1) > 0
    )
    assert np.array_equal(
        old_only,
        _project_via_registrations(mask, [old], 9, canvas_hw=(60, 80)) > 0,
    )
    # Registro sem marca de época não é excluído por filtro nenhum.
    legacy = {"anchor": anchor_a, "quad": anchor_a, "submitted": None, "frame": 1}
    assert (
        _project_via_registrations(mask, [legacy], 9, canvas_hw=(60, 80), epoch=3)
        is not None
    )


def _timeline_bundle():
    """Dois traços de fase 0 + um de fase 1, com relógio write
    determinístico: 8 frames para 2 traços → 4 frames por traço."""
    points = np.array(
        [
            [2, 4],
            [8, 4],
            [14, 4],
            [20, 4],
            [26, 4],  # traço 0: 5 pontos na horizontal
            [2, 12],
            [14, 12],
            [26, 12],  # traço 1: 3 pontos
            [4, 20],
            [16, 20],
            [28, 20],  # traço 2 (fase 1): 3 pontos
        ],
        np.int32,
    )
    bundle = {
        "page_ink_full": np.zeros((30, 30), np.uint8),
        "points": points,
        "offsets": np.array([0, 5, 8, 11]),
        "thickness": np.array([2, 2, 2]),
        "stroke_phase": np.array([0, 0, 1]),
        "erase_rect": np.array([0, 0, 12, 16], np.int32),
    }
    events = [
        {"kind": "write", "start": 10, "end": 18},
        {"kind": "erase", "start": 20, "end": 28, "rect": (0, 0, 12, 16)},
        {"kind": "write", "start": 30, "end": 34},
    ]
    return bundle, events


def test_stroke_reveals_por_traco():
    from lab.eval_scenario import _stroke_reveals

    bundle, events = _timeline_bundle()
    reveals = _stroke_reveals(bundle, events)
    # write 1: slots 0 e 1 → (10, 13) e (14, 17); write 2: (30, 33).
    assert reveals[0] == (10, 13, 4.0)
    assert reveals[1] == (14, 17, 4.0)
    assert reveals[2] == (30, 33, 4.0)


def test_revealed_page_mask_traco_parcial_e_borracha():
    from lab.eval_scenario import _revealed_page_mask, _stroke_reveals

    bundle, events = _timeline_bundle()
    reveals = _stroke_reveals(bundle, events)
    erase = events[1]
    rect = (0, 0, 12, 16)

    # Antes da escrita: folha vazia.
    assert not _revealed_page_mask(bundle, reveals, 9, erase, rect).any()
    # Meio do traço 0 (fração 2/4): prefixo até ~metade dos pontos
    # (x≤14, o resto da polilinha ainda não existe).
    partial = _revealed_page_mask(bundle, reveals, 11, erase, rect) > 0
    assert partial[:, 2:13].any() and not partial[:, 17:].any()
    # Traço 0 completo, traço 1 ainda não começou.
    mask = _revealed_page_mask(bundle, reveals, 13, erase, rect) > 0
    assert mask[4, 24:27].any() and not mask[12].any()
    # Meio da borracha (t=5/8 → frente ~x=8): traço 1 visível à
    # direita da frente, apagado à esquerda.
    mask = _revealed_page_mask(bundle, reveals, 24, erase, rect) > 0
    assert not mask[:16, :8].any() and mask[12, 14:27].any()
    # Redesenho (fase 1) dentro do retângulo não é apagado.
    mask = _revealed_page_mask(bundle, reveals, 33, erase, rect) > 0
    assert mask[20, 4:10].any()
    # Depois da borracha a faixa dos traços de fase 0 zera de vez.
    assert not mask[:16, :12].any()


def test_project_via_registrations_conteudo_por_frame_sem_fantasma():
    """O caso do achado: âncora A numa passada de página em branco,
    âncora B antes do traço tardio — a união não pode marcar o
    traço na posição de A, onde nenhuma passada o depositou."""
    from lab.eval_scenario import _project_via_registrations

    mask = np.zeros((30, 30), np.uint8)
    mask[5:15, 5:15] = 255
    blank = np.zeros_like(mask)
    anchor_a = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    anchor_b = anchor_a + np.float32([8, 0])
    early = {"anchor": anchor_a, "quad": anchor_a, "submitted": None, "frame": 2}
    late = {"anchor": anchor_b, "quad": anchor_a, "submitted": None, "frame": 6}

    def revealed(frame):
        # A passada do frame 2 viu página em branco; a do 6, o traço.
        return mask if frame >= 4 else blank

    truth = (
        _project_via_registrations(
            mask,
            [early, late],
            9,
            canvas_hw=(60, 80),
            content_by_frame=revealed,
        )
        > 0
    )
    only_late = _project_via_registrations(mask, [late], 9, canvas_hw=(60, 80)) > 0
    assert np.array_equal(truth, only_late)
    # Sem conteúdo por frame a passada cega cria a cópia fantasma.
    naive = _project_via_registrations(mask, [early, late], 9, canvas_hw=(60, 80)) > 0
    assert naive.sum() > truth.sum()


def test_project_via_registrations_apagado_sai_da_verdade():
    """Conteúdo de passada ∩ esperado do checkpoint: a passada viu
    dois traços, a borracha removeu um — só o sobrevivente pode
    continuar projetando como verdade na grade."""
    from lab.eval_scenario import _project_via_registrations

    content = np.zeros((30, 30), np.uint8)
    content[5:15, 5:15] = 255  # traço que a borracha vai remover
    content[20:25, 20:25] = 255  # traço que sobrevive
    expected = np.zeros_like(content)
    expected[20:25, 20:25] = 255
    anchor = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    reg = {"anchor": anchor, "quad": anchor, "submitted": None, "frame": 6}
    out = (
        _project_via_registrations(
            expected,
            [reg],
            9,
            canvas_hw=(60, 80),
            content_by_frame=lambda frame: content,
        )
        > 0
    )
    survivor = _project_via_registrations(expected, [reg], 9, canvas_hw=(60, 80)) > 0
    assert np.array_equal(out, survivor)
    # Sem a interseção o traço apagado também entraria na verdade.
    naive = _project_via_registrations(content, [reg], 9, canvas_hw=(60, 80)) > 0
    assert naive.sum() > out.sum()


def test_unobserved_truth_marca_escrita_nunca_vista():
    """Nenhum registro depois da revelação do traço: a fatia esperada
    continua exigência (projetada pelo chamador pela pose do
    checkpoint) — canvas vazio não pode ler como captura perfeita."""
    from lab.eval_scenario import _unobserved_truth

    expected = np.zeros((30, 30), np.uint8)
    expected[5:15, 5:15] = 255
    anchor = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    early = {"anchor": anchor, "quad": anchor, "submitted": None, "frame": 2}
    late = {"anchor": anchor, "quad": anchor, "submitted": None, "frame": 8}

    def revealed(frame):
        # A escrita só aparece a partir do frame 5.
        return expected if frame >= 5 else np.zeros_like(expected)

    # Só a passada cega (frame 2): a máscara inteira é não observada.
    unreached = _unobserved_truth(expected, [early], 9, None, revealed)
    assert np.array_equal(unreached, expected)
    # Com a passada tardia vendo o traço, nada falta observar.
    unreached = _unobserved_truth(expected, [early, late], 9, None, revealed)
    assert not unreached.any()
    # Época errada não conta como observação.
    late_ep = dict(late, epoch=2)
    unreached = _unobserved_truth(expected, [early, late_ep], 9, 1, revealed)
    assert np.array_equal(unreached, expected)
    # Registro sem frame assume visão plena (contrato antigo).
    timeless = {"anchor": anchor, "quad": anchor, "submitted": None, "frame": None}
    unreached = _unobserved_truth(expected, [timeless], 9, None, revealed)
    assert not unreached.any()


def test_polyline_rect_overlap_matches_scalar_reference():
    import math as _math

    def point_seg_dist(point, a, b):
        ab = b - a
        denom = float(ab @ ab)
        if denom == 0:
            return float(_math.hypot(*(point - a)))
        t = max(0.0, min(1.0, float(((point - a) @ ab) / denom)))
        closest = a + t * ab
        return float(_math.hypot(*(point - closest)))

    def scalar_overlap(points, rect, thickness=0):
        if len(points) == 0:
            return False
        x, y, w, h = rect
        inside = (
            (points[:, 0] >= x)
            & (points[:, 0] < x + w)
            & (points[:, 1] >= y)
            & (points[:, 1] < y + h)
        )
        if inside.any():
            return True
        if thickness <= 0:
            return False
        radius = thickness / 2.0
        x1, y1 = x + w - 1, y + h - 1
        dx = np.maximum.reduce(
            [x - points[:, 0], points[:, 0] - x1, np.zeros(len(points))]
        )
        dy = np.maximum.reduce(
            [y - points[:, 1], points[:, 1] - y1, np.zeros(len(points))]
        )
        if (np.hypot(dx, dy) <= radius).any():
            return True
        corners = np.float32([[x, y], [x1, y], [x1, y1], [x, y1]])
        for a, b in zip(points[:-1], points[1:], strict=False):
            for corner in corners:
                if point_seg_dist(corner, a, b) <= radius:
                    return True
        return False

    rng = np.random.default_rng(12345)
    for _ in range(120):
        n_pts = int(rng.integers(0, 12))
        pts = rng.integers(-10, 80, size=(n_pts, 2), dtype=np.int32)
        rect = (
            int(rng.integers(0, 40)),
            int(rng.integers(0, 40)),
            int(rng.integers(1, 30)),
            int(rng.integers(1, 30)),
        )
        thick = int(rng.choice([0, 1, 2, 4, 7, 12]))
        assert sim_source._polyline_rect_overlap(pts, rect, thick) is scalar_overlap(
            pts, rect, thick
        )
    # Segmento degenerado (comprimento zero) e ponto unico perto da borda:
    same = np.array([[20, 20], [20, 20]], dtype=np.int32)
    assert sim_source._polyline_rect_overlap(same, (22, 18, 10, 10), 4)
    assert not sim_source._polyline_rect_overlap(same, (23, 18, 10, 10), 4)
    single = np.array([[19, 20]], dtype=np.int32)
    assert sim_source._polyline_rect_overlap(single, (21, 18, 10, 10), 4)


def test_static_warp_cache_is_byte_identical_across_static_and_optic_phases():
    kwargs = dict(
        fps=8,
        stroke_seconds=0.375,
        warmup_seconds=0.5,
        settle_seconds=0.5,
        hold_seconds=0.5,
        erase_stroke=-1,
        redraw_text="c",
        erase_seconds=0.25,
        move=(40, -25, 10),
        move_seconds=0.375,
        lift=(300, 200),
        return_pose=(-20, 15, 10),
        absent_seconds=0.5,
        lift_seconds=0.25,
        return_seconds=0.25,
        rest_hand_seconds=0.5,
        motion_blur=True,
        ink_sheen=0.5,
        page_curl=0.4,
    )
    cached = SyntheticCapture("ab", **kwargs)
    uncached = SyntheticCapture("ab", **kwargs)
    while True:
        uncached._static_cache = None
        ok_a, frame_a = cached.read()
        ok_b, frame_b = uncached.read()
        assert ok_a == ok_b
        if not ok_a:
            break
        assert np.array_equal(frame_a, frame_b)
        for key in ("rgb", "mask", "ink_full"):
            assert np.array_equal(cached.last_scene[key], uncached.last_scene[key])
        assert np.array_equal(cached.page_to_scene, uncached.page_to_scene)
    assert cached._static_cache is not None
