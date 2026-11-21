"""Contratos de src/draw/contracts.py: validação e imutabilidade."""

import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from draw.contracts import (
    DetectionResult,
    FramePacket,
    GateInfo,
    InkDisplaySnapshot,
    InkSnapshot,
    InkState,
    PageIdentity,
    SheetQuad,
    WarpedPage,
)

QUAD = np.array(
    [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]],
    dtype=np.float32,
)


def _frame(shape=(4, 4, 3), dtype=np.uint8):
    return np.zeros(shape, dtype=dtype)


# --- FramePacket --------------------------------------------------------


def test_frame_packet_valid():
    packet = FramePacket(image=_frame(), captured_at=1.5, sequence=3)
    assert packet.image.shape == (4, 4, 3)
    assert packet.sequence == 3


@pytest.mark.parametrize(
    "image",
    [
        _frame((4, 4)),  # 2D
        _frame((4,)),  # 1D
        _frame((4, 4, 1)),  # canal único
        _frame((4, 4, 4)),  # RGBA
        _frame(dtype=np.float32),  # dtype errado
        [[0, 1], [2, 3]],  # não é ndarray
    ],
)
def test_frame_packet_rejects_bad_image(image):
    with pytest.raises(ValueError):
        FramePacket(image=image, captured_at=0.0, sequence=0)


def test_frame_packet_rejects_negative_sequence():
    with pytest.raises(ValueError, match="sequence"):
        FramePacket(image=_frame(), captured_at=0.0, sequence=-1)


def test_frame_packet_source_metadata_optional():
    packet = FramePacket(image=_frame(), captured_at=1.0, sequence=0)
    assert packet.source_frame_id is None
    assert packet.source_ts_ms is None
    packet = FramePacket(
        image=_frame(),
        captured_at=1.0,
        sequence=0,
        source_frame_id=7,
        source_ts_ms=42_000,
    )
    assert packet.source_frame_id == 7
    assert packet.source_ts_ms == 42_000


@pytest.mark.parametrize(
    "field",
    [{"source_frame_id": -1}, {"source_ts_ms": -5}],
    ids=["frame_id_negativo", "ts_negativo"],
)
def test_frame_packet_rejects_negative_source_metadata(field):
    with pytest.raises(ValueError, match="source_"):
        FramePacket(image=_frame(), captured_at=0.0, sequence=0, **field)


def test_frame_packet_is_frozen():
    packet = FramePacket(image=_frame(), captured_at=0.0, sequence=0)
    with pytest.raises(FrozenInstanceError):
        packet.sequence = 9


# --- SheetQuad ------------------------------------------------------------


def test_sheet_quad_valid_and_present():
    quad = SheetQuad(corners=QUAD, verdict="paper", detector="neural")
    assert quad.present
    assert quad.corners.dtype == np.float32
    assert quad.corners.shape == (4, 2)


def test_sheet_quad_absent():
    quad = SheetQuad(corners=None, verdict=None)
    assert not quad.present
    assert quad.detector == ""


def test_sheet_quad_normalizes_and_copies_corners():
    source = QUAD.astype(np.float64)
    quad = SheetQuad(corners=source, verdict="paper")
    assert quad.corners.dtype == np.float32
    source[0, 0] = -999.0
    assert quad.corners[0, 0] == 100.0, "o contrato não pode ver o array externo"


@pytest.mark.parametrize(
    "corners",
    [
        np.zeros((3, 2), dtype=np.float32),
        np.zeros((4, 3), dtype=np.float32),
        np.zeros((8,), dtype=np.float32),
    ],
)
def test_sheet_quad_rejects_bad_corners(corners):
    with pytest.raises(ValueError, match="4, 2"):
        SheetQuad(corners=corners, verdict="paper")


def test_sheet_quad_rejects_bad_verdict():
    with pytest.raises(ValueError, match="verdict"):
        SheetQuad(corners=QUAD, verdict="table")


def test_sheet_quad_accepts_blank_maybe():
    """O rastreador confirma com o veredito que sustentou a folha:
    "blank_maybe" sustentado entra confirmado com esse rótulo."""
    quad = SheetQuad(corners=QUAD, verdict="blank_maybe", detector="classic")
    assert quad.verdict == "blank_maybe"


def test_sheet_quad_is_frozen():
    quad = SheetQuad(corners=QUAD, verdict="paper")
    with pytest.raises(FrozenInstanceError):
        quad.verdict = None


# --- PageIdentity ---------------------------------------------------------


def test_page_identity_valid():
    ident = PageIdentity(generation=2, decision="new_page", recall=0.5)
    assert ident.decision == "new_page"
    assert ident.containment is None


@pytest.mark.parametrize("decision", ["same", "new_page", "defer"])
def test_page_identity_decisions(decision):
    assert PageIdentity(generation=0, decision=decision).decision == decision


def test_page_identity_rejects_bad_decision():
    with pytest.raises(ValueError, match="decision"):
        PageIdentity(generation=0, decision="maybe")


@pytest.mark.parametrize("name", ["recall", "containment"])
@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_page_identity_rejects_evidence_out_of_range(name, value):
    with pytest.raises(ValueError, match=name):
        PageIdentity(generation=0, decision="same", **{name: value})


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_page_identity_accepts_evidence_at_bounds(value):
    ident = PageIdentity(
        generation=0, decision="defer", recall=value, containment=value
    )
    assert ident.recall == value


def test_page_identity_is_frozen():
    ident = PageIdentity(generation=0, decision="same")
    with pytest.raises(FrozenInstanceError):
        ident.decision = "defer"


# --- InkState -------------------------------------------------------------


def _mask(shape=(8, 8), dtype=np.uint8):
    return np.zeros(shape, dtype=dtype)


def test_ink_state_valid():
    state = InkState(
        confirmed=_mask(), provisional=_mask(), generation=1, anchor_corners=QUAD
    )
    assert state.generation == 1
    assert state.anchor_corners.dtype == np.float32


def test_ink_state_allows_none_masks():
    state = InkState(confirmed=None, provisional=None, generation=0)
    assert state.anchor_corners is None


@pytest.mark.parametrize("name", ["confirmed", "provisional"])
@pytest.mark.parametrize(
    "mask",
    [
        _mask((8, 8, 3)),  # 3D
        _mask(dtype=np.float32),  # dtype errado
        _mask(dtype=np.bool_),  # dtype errado
    ],
)
def test_ink_state_rejects_bad_masks(name, mask):
    kwargs = {"confirmed": None, "provisional": None, "generation": 0}
    kwargs[name] = mask
    with pytest.raises(ValueError, match=name):
        InkState(**kwargs)


def test_ink_state_rejects_bad_anchor_corners():
    with pytest.raises(ValueError):
        InkState(
            confirmed=None,
            provisional=None,
            generation=0,
            anchor_corners=np.zeros((3, 2), dtype=np.float32),
        )


def test_ink_state_is_frozen():
    state = InkState(confirmed=None, provisional=None, generation=0)
    with pytest.raises(FrozenInstanceError):
        state.generation = 7


def test_sheet_quad_rejects_corners_without_verdict():
    with pytest.raises(ValueError, match="juntos"):
        SheetQuad(corners=QUAD, verdict=None)


def test_sheet_quad_rejects_verdict_without_corners():
    with pytest.raises(ValueError, match="juntos"):
        SheetQuad(corners=None, verdict="paper")


def test_sheet_quad_corners_are_read_only():
    quad = SheetQuad(corners=QUAD, verdict="paper")
    with pytest.raises(ValueError):
        quad.corners[0, 0] = 99.0
    # a fonte do chamador continua gravável
    source = QUAD.copy()
    SheetQuad(corners=source, verdict="paper")
    source[0, 0] = 1.0


def test_frame_packet_image_is_read_only_view():
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    pkt = FramePacket(image=frame, captured_at=0.0, sequence=0)
    with pytest.raises(ValueError):
        pkt.image[0, 0, 0] = 7
    frame[0, 0, 0] = 7  # o buffer original segue gravável


def test_ink_state_masks_and_anchor_are_read_only():
    confirmed = np.zeros((3, 3), dtype=np.uint8)
    anchor = QUAD.copy()
    state = InkState(
        confirmed=confirmed,
        provisional=None,
        generation=0,
        anchor_corners=anchor,
    )
    with pytest.raises(ValueError):
        state.confirmed[0, 0] = 255
    with pytest.raises(ValueError):
        state.anchor_corners[0, 0] = 0.0
    confirmed[0, 0] = 255  # a fonte segue gravável


# --- WarpedPage / InkSnapshot / InkDisplaySnapshot -------------------------


def _warped_page(**overrides):
    fields = {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "generation": 2,
    }
    fields.update(overrides)
    return WarpedPage(**fields)


def test_warped_page_read_only_fields():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    corners = QUAD.copy()
    visible = np.ones((8, 8), dtype=bool)
    coverage = np.zeros((8, 8), dtype=np.uint8)
    page = _warped_page(
        image=image, corners=corners, visible=visible, coverage=coverage
    )
    assert page.generation == 2
    np.testing.assert_array_equal(page.corners, QUAD)
    for field in (page.image, page.corners, page.visible, page.coverage):
        assert not field.flags.writeable
    with pytest.raises(ValueError):
        page.visible[0, 0] = False
    image[0, 0, 0] = 9  # a fonte segue gravável


def test_warped_page_rejects_bad_fields():
    with pytest.raises(ValueError):
        _warped_page(image=np.zeros((8, 8, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        _warped_page(image=np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        _warped_page(generation=-1)
    with pytest.raises(ValueError):
        _warped_page(corners=np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        _warped_page(visible=np.ones((8, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        _warped_page(coverage=np.ones((8, 8, 3), dtype=bool))
    # Sem corners a mascara segue a grade do warp (com corners ela
    # e de camera e o tamanho e o do frame — fora do contrato).
    with pytest.raises(ValueError):
        _warped_page(visible=np.ones((6, 8), dtype=bool))
    with pytest.raises(ValueError):
        _warped_page(coverage=np.zeros((8, 6), dtype=np.uint8))
    page = _warped_page(corners=QUAD.copy(), visible=np.ones((6, 8), bool))
    assert page.visible.shape == (6, 8)


def test_warped_page_accepts_2d_and_float_maps():
    """image 2D aceita qualquer dtype (mapa de probabilidade
    float32 dos page_clean falsos e warps de canal único)."""
    page = _warped_page(image=np.zeros((8, 8), dtype=np.float32))
    assert page.image.dtype == np.float32
    assert page.corners is None and page.visible is None


def test_warped_page_is_frozen():
    page = _warped_page()
    with pytest.raises(FrozenInstanceError):
        page.generation = 3


def test_ink_snapshot_copies_and_freezes():
    source = np.zeros((4, 4), dtype=np.uint8)
    snap = InkSnapshot(version=7, confirmed=source)
    assert snap.version == 7
    assert not snap.confirmed.flags.writeable
    source[0, 0] = 255  # mutação posterior não atravessa a fronteira
    assert snap.confirmed[0, 0] == 0
    with pytest.raises(FrozenInstanceError):
        snap.version = 8
    with pytest.raises(ValueError):
        InkSnapshot(version=1, confirmed=np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        InkSnapshot(version=1, confirmed=np.zeros((4, 4), dtype=np.float32))


def test_ink_display_snapshot_same_shape_read_only():
    confirmed = np.ones((4, 4), dtype=np.uint8)
    provisional = np.zeros((4, 4), dtype=np.uint8)
    snap = InkDisplaySnapshot(version=3, confirmed=confirmed, provisional=provisional)
    assert not snap.confirmed.flags.writeable
    assert not snap.provisional.flags.writeable
    with pytest.raises(ValueError):
        InkDisplaySnapshot(
            version=3,
            confirmed=np.zeros((5, 5), dtype=np.uint8),
            provisional=provisional,
        )
    with pytest.raises(ValueError):
        snap.confirmed[0, 0] = 0


# --- fronteira de dependências --------------------------------------------


def test_contracts_import_leaves_out_heavy_dependencies():
    """Importar draw.contracts não pode puxar torch, pandas ou cv2."""
    code = (
        "import sys\n"
        "import draw.contracts\n"
        "for name in ('torch', 'pandas', 'cv2'):\n"
        "    assert name not in sys.modules, name\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# --- DetectionResult ----------------------------------------------------


def _detection_result(**overrides):
    fields = {
        "sequence": 1,
        "corners": QUAD.copy(),
        "verdict": "paper",
        "captured_at": 12.5,
        "detect_ms": 36.7,
    }
    fields.update(overrides)
    return DetectionResult(**fields)


def test_detection_result_corners_read_only_copy():
    source = QUAD.copy()
    result = _detection_result(corners=source)
    np.testing.assert_array_equal(result.corners, QUAD)
    assert not result.corners.flags.writeable
    assert result.corners.dtype == np.float32
    with pytest.raises(ValueError):
        result.corners[0, 0] = 99.0
    source[0, 0] = 99.0  # a fonte segue gravável
    assert result.corners[0, 0] == QUAD[0, 0]


def test_detection_result_rejects_bad_corners_shape():
    with pytest.raises(ValueError):
        _detection_result(corners=np.zeros((3, 2), dtype=np.float32))


def test_detection_result_allows_paper_without_quad():
    """'paper' sem quadrilátero é legítimo (presença sem posição):
    diferente de SheetQuad, aqui os dois não precisam vir juntos."""
    result = _detection_result(corners=None, verdict="paper")
    assert result.corners is None and result.verdict == "paper"


def test_detection_result_accepts_every_legacy_verdict():
    """Todo veredito que o pipeline legado produz/propaga sai do
    contrato: "paper", "blank_maybe" (candidato branco demais para
    afirmar papel) e "none" (detecção fraca rebaixada pelo portão
    de primeira confirmação do laço), com ou sem quadrilátero."""
    for verdict in (None, "paper", "blank_maybe", "none"):
        result = _detection_result(verdict=verdict)
        assert result.verdict == verdict
    without_quad = _detection_result(corners=None, verdict="blank_maybe")
    assert without_quad.corners is None and without_quad.verdict == "blank_maybe"


def test_detection_result_rejects_bad_verdict_and_negative_ms():
    with pytest.raises(ValueError):
        _detection_result(corners=None, verdict="maybe")
    with pytest.raises(ValueError):
        _detection_result(detect_ms=-1.0)
    with pytest.raises(ValueError):
        _detection_result(sequence=-1)


def test_detection_result_masks_are_read_only_views():
    coverage = np.zeros((8, 8), dtype=np.uint8)
    hand = np.ones((8, 8), dtype=np.uint8)
    result = _detection_result(coverage_mask=coverage, hand_mask=hand)
    assert not result.coverage_mask.flags.writeable
    assert not result.hand_mask.flags.writeable
    with pytest.raises(ValueError):
        result.coverage_mask[0, 0] = 1
    coverage[0, 0] = 1  # a fonte segue gravável
    with pytest.raises(ValueError):
        _detection_result(coverage_mask=np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        _detection_result(hand_mask=np.zeros((8, 8), dtype=np.float32))


def test_detection_result_gate_info_and_frozen():
    gate = GateInfo(source="neural", presence_logit=0.32, edge_in=0.001)
    result = _detection_result(gate_info=gate)
    assert result.gate_info is gate
    with pytest.raises(FrozenInstanceError):
        result.detect_ms = 1.0
