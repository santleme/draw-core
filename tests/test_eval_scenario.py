"""lab/eval_scenario.py: o hop flag -> verify do CLI."""

import pytest


def test_main_propagates_allow_unverified_to_audit(tmp_path, monkeypatch):
    """O hop flag -> verify: sem a flag audit recebe "manifest";
    com --allow-unverified, o opt-out None explícito."""
    import lab.eval_scenario as es

    seen = []

    class _Bail(Exception):
        pass

    def _audit(*_a, verify=None, **_kw):
        seen.append(verify)
        raise _Bail

    monkeypatch.setattr(es, "audit", _audit)

    base = ["--video", str(tmp_path / "v.mp4"), "--gt", str(tmp_path / "gt.npy")]
    with pytest.raises(_Bail):
        es.main(base)
    with pytest.raises(_Bail):
        es.main([*base, "--allow-unverified"])

    assert seen == ["manifest", None]
