"""Test ba công thức dung hợp (mock data theo docs/DATA_CONTRACTS.md §3)."""

from __future__ import annotations

import pytest

from trec_covid.fuse.ranking import check_ranking
from trec_covid.fuse.rrf import convex, fuse, rrf, weighted_rrf
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


class TestConvex:
    def test_minmax_theo_danh_sach_cua_truy_van(self):
        # MedCPT co cụm ~[0.7, 0.95]: phải chuẩn hoá theo min/max thật, KHÔNG theo [-1, 1].
        dense = [
            ScoredDocument("a", 0.95, rank=1),
            ScoredDocument("b", 0.90, rank=2),
            ScoredDocument("c", 0.70, rank=3),
        ]
        out = {d.cord_uid: d.score for d in convex([], dense, alpha=0.0)}
        assert out == pytest.approx({"a": 1.0, "b": 0.8, "c": 0.0})

    def test_alpha_1_theo_bm25_alpha_0_theo_dense(self):
        # Bài cuối mỗi danh sách bị min-max đưa về 0, hoà với bài vắng mặt (cũng 0),
        # nên chỉ so được các bài trên đáy; đáy hoà thì phá theo cord_uid giảm.
        out1 = {d.cord_uid: d.score for d in convex(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, 1.0)}
        assert out1["doc_001"] > out1["doc_002"] > out1["doc_003"] == out1["doc_004"] == 0.0
        out0 = convex(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, alpha=0.0)
        assert _uids(out0)[:2] == _uids(MOCK_DENSE_RESULTS)[:2]
        assert all(d.source == "convex" for d in out0)

    def test_doc_vang_mot_nhanh_nhan_0_o_nhanh_do(self):
        out = {d.cord_uid: d.score for d in convex(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, 0.5)}
        # doc_004 chỉ có ở dense, đứng cuối dense -> 0 ở cả hai nhánh.
        assert out["doc_004"] == pytest.approx(0.0)
        # doc_001: bm25 = 1.0, dense = (0.82 - 0.79) / (0.88 - 0.79)
        assert out["doc_001"] == pytest.approx(0.5 + 0.5 * (0.03 / 0.09))

    def test_moi_diem_bang_nhau_khong_chia_cho_0(self):
        same = [ScoredDocument("a", 0.9, rank=1), ScoredDocument("b", 0.9, rank=2)]
        out = {d.cord_uid: d.score for d in convex(same, [], alpha=1.0)}
        assert out == {"a": 1.0, "b": 1.0}
        one = convex([ScoredDocument("x", 3.0, rank=1)], [], alpha=0.5)
        assert one[0].score == pytest.approx(0.5)


class TestFuse:
    def test_chon_cong_thuc_theo_active(self):
        cfg = {"active": "convex", "convex": {"alpha": 0.3}}
        out = fuse(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, cfg)
        assert _uids(out) == _uids(convex(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, alpha=0.3))
        cfg = {"active": "weighted_rrf", "weighted_rrf": {"k": 60, "w_bm25": 0.2, "w_dense": 0.8}}
        assert fuse(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, cfg)[0].source == "weighted_rrf"
        assert (
            fuse(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, {"active": "rrf", "rrf": {"k": 60}})[
                0
            ].source
            == "rrf"
        )

    def test_active_la_thi_raise(self):
        with pytest.raises(ValueError, match="active"):
            fuse(MOCK_BM25_RESULTS, MOCK_DENSE_RESULTS, {"active": "combsum"})
