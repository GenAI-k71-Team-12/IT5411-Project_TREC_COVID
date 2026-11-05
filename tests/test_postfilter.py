"""Test postfilter 3 chế độ: off | docids | judged (ADR 0005, ADR 0008)."""

from __future__ import annotations

import pytest

from trec_covid.config import load_config
from trec_covid.fuse.ranking import to_ranked
from trec_covid.postfilter.residual import apply_postfilter, load_allowed_docids, load_judged

QRELS = {"1": {"a": 2, "c": 0}, "2": {"b": 1}}


def _run():
    return to_ranked({"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}, "rerank")


def _uids_ranks(docs):
    return [(d.cord_uid, d.rank) for d in docs]


def test_off_giu_nguyen():
    run = _run()
    assert apply_postfilter(run, mode="off", topic_id="1") == run


def test_docids_giu_bai_trong_danh_sach_duoc_phep_va_danh_lai_rank():
    out = apply_postfilter(_run(), mode="docids", topic_id="1", allowed=frozenset({"a", "c", "d"}))
    assert _uids_ranks(out) == [("a", 1), ("c", 2), ("d", 3)]
    assert [d.score for d in out] == [4.0, 2.0, 1.0]


def test_judged_chi_bo_bai_da_cham_cua_dung_topic():
    judged = load_judged(QRELS)
    out = apply_postfilter(_run(), mode="judged", topic_id="1", judged=judged)
    # a, c đã chấm ở topic 1 -> bỏ; b đã chấm ở topic 2 -> vẫn giữ ở topic 1.
    assert _uids_ranks(out) == [("b", 1), ("d", 2)]


def test_judged_topic_khong_co_qrels_thi_khong_bo_gi():
    out = apply_postfilter(_run(), mode="judged", topic_id="36", judged=load_judged(QRELS))
    assert len(out) == 4


def test_thieu_du_lieu_cho_che_do_thi_raise():
    with pytest.raises(ValueError, match="allowed"):
        apply_postfilter(_run(), mode="docids", topic_id="1")
    with pytest.raises(ValueError, match="judged"):
        apply_postfilter(_run(), mode="judged", topic_id="1")


def test_che_do_la_thi_raise():
    with pytest.raises(ValueError, match="mode"):
        apply_postfilter(_run(), mode="residual", topic_id="1")


def test_doc_docids_bo_dong_trong_va_dong_lap(tmp_path):
    path = tmp_path / "docids.txt"
    path.write_text("a\n\nb\na\n", encoding="utf-8")
    assert load_allowed_docids(path) == frozenset({"a", "b"})


def test_config_mac_dinh_tat():
    assert load_config("postfilter")["mode"] == "off"
