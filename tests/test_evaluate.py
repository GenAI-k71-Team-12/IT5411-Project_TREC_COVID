"""Test run writer và chấm điểm (pytrec_eval = mã C của trec_eval)."""

from __future__ import annotations

import math

import pytest

from trec_covid.evaluate.run_writer import read_run, write_run
from trec_covid.evaluate.trec_eval import MEASURES, evaluate_run, load_qrels, summarize
from trec_covid.schemas import ScoredDocument

QRELS_CSV = """topic-id,iteration,cord-id,judgement
1,0.5,d1,2
1,1.0,d2,0
1,1.5,d3,1
2,2.0,d9,1
"""


def _ranked(*uids: str) -> list[ScoredDocument]:
    n = len(uids)
    return [ScoredDocument(u, float(n - i), rank=i + 1) for i, u in enumerate(uids)]


@pytest.fixture
def qrels(tmp_path):
    path = tmp_path / "qrels.csv"
    path.write_text(QRELS_CSV, encoding="utf-8")
    return load_qrels(path)


class TestRunWriter:
    def test_ghi_dung_6_cot(self, tmp_path):
        path = write_run({1: _ranked("d1", "d3")}, tmp_path / "r.txt", "tag")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines == ["1 Q0 d1 1 2.000000 tag", "1 Q0 d3 2 1.000000 tag"]

    def test_ghi_roi_doc_lai_giu_nguyen(self, tmp_path):
        run = {"1": _ranked("d1", "d3"), "2": _ranked("d9")}
        back = read_run(write_run(run, tmp_path / "r.txt", "tag"))
        assert list(back) == ["1", "2"]
        assert [(d.cord_uid, d.score, d.rank) for d in back["1"]] == [
            ("d1", 2.0, 1),
            ("d3", 1.0, 2),
        ]
        assert back["2"][0].source == "tag"

    def test_diem_tang_thi_khong_ghi(self, tmp_path):
        bad = [ScoredDocument("a", 1.0, rank=1), ScoredDocument("b", 2.0, rank=2)]
        with pytest.raises(ValueError):
            write_run({"1": bad}, tmp_path / "r.txt", "tag")


class TestLoadQrels:
    def test_bo_cot_iteration(self, qrels):
        assert qrels == {"1": {"d1": 2, "d2": 0, "d3": 1}, "2": {"d9": 1}}


class TestEvaluateRun:
    def test_thu_tu_ly_tuong_ndcg_bang_1(self, qrels):
        per_topic = evaluate_run({"1": _ranked("d1", "d3", "d2")}, qrels)
        assert per_topic.loc["1", "ndcg_cut_10"] == pytest.approx(1.0)
        assert list(per_topic.columns) == list(MEASURES)

    def test_dao_thu_tu_thi_thap_hon(self, qrels):
        per_topic = evaluate_run({"1": _ranked("d2", "d3", "d1")}, qrels)
        assert per_topic.loc["1", "ndcg_cut_10"] < 1.0

    def test_khop_ndcg_tinh_tay(self, qrels):
        # trec_eval dùng gain TUYẾN TÍNH (gain = nhãn), không phải 2^rel - 1.
        # run [d3, d1]: DCG = 1/log2(2) + 2/log2(3); IDCG = 2/log2(2) + 1/log2(3)
        per_topic = evaluate_run({"1": _ranked("d3", "d1")}, qrels)
        expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
        assert abs(per_topic.loc["1", "ndcg_cut_10"] - expected) < 1e-9

    def test_topic_khong_co_qrels_khong_keo_trung_binh(self, qrels):
        # Giống trec_eval không có -c: topic 36 (không có phán định) bị bỏ qua.
        run = {"1": _ranked("d1", "d3", "d2"), "36": _ranked("x")}
        summary = summarize(evaluate_run(run, qrels))
        assert summary["n_topics"] == 1
        assert summary["ndcg_cut_10"] == pytest.approx(1.0)


class TestResultsCsv:
    def test_dong_dau_ghi_commit_va_config(self, tmp_path):
        import pandas as pd

        from trec_covid.evaluate.results import write_results_csv

        df = pd.DataFrame({"run": ["bm25"], "ndcg_cut_10": [0.5]})
        path = write_results_csv(df, tmp_path / "r.csv", {"config": "fusion"})
        first, header = path.read_text(encoding="utf-8").splitlines()[:2]
        assert first.startswith("# commit=")
        assert "config=fusion" in first
        assert header == "run,ndcg_cut_10"
        back = pd.read_csv(path, comment="#")
        assert back.loc[0, "ndcg_cut_10"] == 0.5
