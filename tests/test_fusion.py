"""Test ba công thức dung hợp (mock data theo docs/DATA_CONTRACTS.md §3)."""

from __future__ import annotations

import pytest

from trec_covid.fuse.ranking import check_ranking
from trec_covid.fuse.rrf import rrf, weighted_rrf
from trec_covid.schemas import ScoredDocument

MOCK_BM25_RESULTS = [
    ScoredDocument(cord_uid="doc_001", score=18.5, rank=1, source="bm25"),
    ScoredDocument(cord_uid="doc_002", score=16.2, rank=2, source="bm25"),
    ScoredDocument(cord_uid="doc_003", score=14.1, rank=3, source="bm25"),
]

MOCK_DENSE_RESULTS = [
    ScoredDocument(cord_uid="doc_002", score=0.88, rank=1, source="dense"),
    ScoredDocument(cord_uid="doc_001", score=0.82, rank=2, source="dense"),
    ScoredDocument(cord_uid="doc_004", score=0.79, rank=3, source="dense"),
]


def _uids(docs: list[ScoredDocument]) -> list[str]:
    return [d.cord_uid for d in docs]


class TestRrf:
    def test_diem_dung_cong_thuc_k60(self):
        out = {d.cord_uid: d.score for d in rrf(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS)}
        assert out["doc_001"] == pytest.approx(1 / 61 + 1 / 62)
        assert out["doc_002"] == pytest.approx(1 / 62 + 1 / 61)
        assert out["doc_003"] == pytest.approx(1 / 63)
        assert out["doc_004"] == pytest.approx(1 / 63)

    def test_thu_tu_hoa_diem_theo_cord_uid_giam(self):
        out = rrf(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS)
        assert _uids(out) == ["doc_002", "doc_001", "doc_004", "doc_003"]
        check_ranking(out)
        assert all(d.source == "rrf" for d in out)

    def test_dung_rank_cua_input_khong_dung_vi_tri(self):
        # rank là hợp đồng 1-based; nếu code lấy vị trí 0-based thì doc_x = 1/60.
        out = rrf([ScoredDocument("x", 1.0, rank=5)], [])
        assert out[0].score == pytest.approx(1 / 65)

    def test_metadata_ghi_rank_va_diem_tung_nhanh(self):
        out = {d.cord_uid: d.metadata for d in rrf(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS)}
        assert out["doc_003"] == {
            "bm25_rank": 3,
            "dense_rank": None,
            "bm25_score": 14.1,
            "dense_score": None,
        }

    def test_mot_nhanh_rong_thi_giu_thu_tu_nhanh_kia(self):
        assert _uids(rrf([], MOCK_DENSE_RESULTS)) == _uids(MOCK_DENSE_RESULTS)
        assert _uids(rrf(MOCK_BM25_RESULTS, [])) == _uids(MOCK_BM25_RESULTS)

    def test_ca_hai_rong(self):
        assert rrf([], []) == []


class TestWeightedRrf:
    def test_trong_so_bang_nhau_cung_thu_tu_rrf(self):
        out = weighted_rrf(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, w_bm25=0.5, w_dense=0.5)
        assert _uids(out) == _uids(rrf(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS))
        assert all(d.source == "weighted_rrf" for d in out)

    def test_chi_bm25_thi_theo_thu_tu_bm25(self):
        out = weighted_rrf(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, w_bm25=1.0, w_dense=0.0)
        assert _uids(out)[:3] == _uids(MOCK_BM25_RESULTS)

    def test_diem_nhan_trong_so(self):
        out = {
            d.cord_uid: d.score
            for d in weighted_rrf(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, 0.3, 0.7, k=60)
        }
        assert out["doc_004"] == pytest.approx(0.7 / 63)
