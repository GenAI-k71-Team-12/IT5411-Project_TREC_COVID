"""Test bảng ablation, đo latency và manifest tái lập."""

from __future__ import annotations

import math
import time

import pytest

from trec_covid.evaluate.ablation import ablation_table
from trec_covid.evaluate.latency import time_stages
from trec_covid.evaluate.reproduce import compare_manifests, run_manifest
from trec_covid.evaluate.run_writer import write_run
from trec_covid.fuse.ranking import to_ranked

QRELS = {"1": {"d1": 2, "d2": 1, "d3": 0}}


def _write(tmp_path, name, order):
    run = {"1": to_ranked({u: float(len(order) - i) for i, u in enumerate(order)}, name)}
    return write_run(run, tmp_path / f"{name}.txt", name)


class TestAblation:
    def test_giu_thu_tu_tang_va_delta_so_voi_tang_truoc(self, tmp_path):
        stages = [
            ("BM25", _write(tmp_path, "a", ["x", "d3", "d2", "d1"])),
            ("+Rerank", _write(tmp_path, "b", ["d1", "d2", "x", "d3"])),
        ]
        table = ablation_table(stages, QRELS, n_branches=1)
        assert list(table["stage"]) == ["BM25", "+Rerank"]
        assert math.isnan(table.loc[0, "delta_ndcg10"])
        assert table.loc[1, "delta_ndcg10"] == pytest.approx(
            table.loc[1, "ndcg_cut_10"] - table.loc[0, "ndcg_cut_10"]
        )
        for col in ("ndcg10_condensed", "judged_10", "delta_condensed", "n_topics", "bpref"):
            assert col in table.columns
        assert table.loc[1, "ndcg10_condensed"] == pytest.approx(1.0)

    def test_nhanh_song_song_khong_co_delta_fusion_so_voi_nhanh_tot_nhat(self, tmp_path):
        # Dense không xếp chồng lên BM25: Δ của nó vô nghĩa. Fusion so với nhánh đơn
        # tốt nhất THEO TỪNG độ đo, không phải với dòng ngay trước.
        stages = [
            ("BM25", _write(tmp_path, "a", ["d1", "x", "d2"])),
            ("Dense", _write(tmp_path, "b", ["x", "y", "d1"])),
            ("Fusion", _write(tmp_path, "c", ["d1", "d2", "x"])),
            ("+Rerank", _write(tmp_path, "d", ["d1", "d2", "d3"])),
        ]
        table = ablation_table(stages, QRELS, n_branches=2).set_index("stage")
        assert math.isnan(table.loc["BM25", "delta_ndcg10"])
        assert math.isnan(table.loc["Dense", "delta_ndcg10"])
        best = max(table.loc["BM25", "ndcg_cut_10"], table.loc["Dense", "ndcg_cut_10"])
        assert table.loc["Fusion", "delta_ndcg10"] == pytest.approx(
            table.loc["Fusion", "ndcg_cut_10"] - best
        )
        assert table.loc["+Rerank", "delta_ndcg10"] == pytest.approx(
            table.loc["+Rerank", "ndcg_cut_10"] - table.loc["Fusion", "ndcg_cut_10"]
        )


class TestLatency:
    def test_median_p95_va_bo_warmup(self):
        calls = []

        def slow(q):
            calls.append(q)
            time.sleep(0.01)

        table = time_stages({"slow": slow}, ["q1", "q2", "q3"], warmup=2)
        row = table.iloc[0]
        assert row["stage"] == "slow"
        assert row["n"] == 3
        assert 8 <= row["median_ms"] <= 60
        assert row["p95_ms"] >= row["median_ms"]
        assert len(calls) == 5  # 2 warmup + 3 đo


class TestReproduce:
    def test_manifest_khop_khi_cung_thu_tu_lech_khi_doi_thu_tu(self, tmp_path):
        a = run_manifest({"r": _write(tmp_path, "a", ["d1", "d2", "d3"])}, QRELS)
        same = run_manifest({"r": _write(tmp_path, "b", ["d1", "d2", "d3"])}, QRELS)
        swapped = run_manifest({"r": _write(tmp_path, "c", ["d2", "d1", "d3"])}, QRELS)
        assert compare_manifests(a, same) == []
        problems = compare_manifests(a, swapped)
        assert any("thứ tự" in p for p in problems)
        assert any("ndcg_cut_10" in p for p in problems)

    def test_thieu_run_thi_bao(self, tmp_path):
        a = run_manifest({"r": _write(tmp_path, "a", ["d1"])}, QRELS)
        assert compare_manifests(a, {"runs": {}}) == ["thiếu run r ở manifest thứ hai"]
