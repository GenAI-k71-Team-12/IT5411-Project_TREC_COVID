"""Test bất biến xếp hạng dùng chung cho fusion, rerank, postfilter và run writer."""

from __future__ import annotations

import pytest

from trec_covid.fuse.ranking import check_ranking, to_ranked
from trec_covid.schemas import ScoredDocument


def _docs(*pairs: tuple[str, float]) -> list[ScoredDocument]:
    return [ScoredDocument(uid, s, rank=i + 1) for i, (uid, s) in enumerate(pairs)]


class TestToRanked:
    def test_sap_giam_theo_diem_hoa_thi_cord_uid_giam(self):
        # trec_eval phá hoà bằng docid giảm dần -> cột rank phải khớp thứ tự đó.
        out = to_ranked({"a": 1.0, "b": 3.0, "c": 3.0}, "x")
        assert [d.cord_uid for d in out] == ["c", "b", "a"]
        assert [d.rank for d in out] == [1, 2, 3]
        assert all(d.source == "x" for d in out)

    def test_gan_metadata_theo_cord_uid(self):
        out = to_ranked({"a": 1.0}, "x", metadata={"a": {"k": 1}})
        assert out[0].metadata == {"k": 1}

    def test_rong_thi_rong(self):
        assert to_ranked({}, "x") == []


class TestCheckRanking:
    def test_hop_le_khong_raise(self):
        check_ranking(_docs(("a", 3.0), ("b", 2.0), ("c", 2.0)))
        check_ranking([])

    def test_diem_tang_thi_raise(self):
        with pytest.raises(ValueError, match="đơn điệu"):
            check_ranking(_docs(("a", 2.0), ("b", 3.0)))

    def test_trung_docid_thi_raise(self):
        with pytest.raises(ValueError, match="trùng"):
            check_ranking(_docs(("a", 3.0), ("a", 2.0)))

    def test_rank_khong_lien_mach_thi_raise(self):
        docs = [ScoredDocument("a", 3.0, rank=1), ScoredDocument("b", 2.0, rank=3)]
        with pytest.raises(ValueError, match="rank"):
            check_ranking(docs)
