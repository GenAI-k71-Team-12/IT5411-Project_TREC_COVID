"""Test kiểm tra lệch giữa kết quả tuning (script 07) và configs/fusion.yaml."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_fusion_experiments.py"
_spec = importlib.util.spec_from_file_location("fusion_experiments", SCRIPT)
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

CFG = {"active": "convex", "weighted_rrf": {"w_bm25": 0.4}, "convex": {"alpha": 0.3}}


def test_khop_config_thi_khong_canh_bao():
    assert exp.config_drift(CFG, w_bm25=0.4, alpha=0.3, active="convex") == []


def test_lech_alpha_hoac_cong_thuc_thi_canh_bao_tung_muc():
    drift = exp.config_drift(CFG, w_bm25=0.4, alpha=0.4, active="weighted_rrf")
    assert any("convex.alpha" in d and "0.3" in d and "0.4" in d for d in drift)
    assert any("active" in d for d in drift)
    assert len(drift) == 2
