"""Test rerank top-k bằng cross-encoder (scorer giả, không tải model) và fuse_and_rerank."""

from __future__ import annotations

import pytest

from trec_covid.fuse.ranking import check_ranking, to_ranked
from trec_covid.rerank.cross_encoder import rerank
from trec_covid.rerank.pipeline import fuse_and_rerank
from trec_covid.schemas import ScoredDocument

DOCS = {f"d{i}": (f"title {i}", f"abstract {i}") for i in range(1, 7)}
DOCS["d6"] = ("title only 6", "")


def _fused(n: int = 6) -> list[ScoredDocument]:
    return to_ranked({f"d{i}": 1.0 - i / 100 for i in range(1, n + 1)}, "convex")


def _fixed_scorer(scores: dict[str, float]):
    """Scorer giả: tra điểm theo số hiệu doc nằm trong văn bản (title i ...)."""

    def scorer(query, articles):
        return [scores[f"d{a.split()[1]}"] if a.split()[0] == "title" else 0.0 for a in articles]

    return scorer


def test_dao_thu_tu_phan_dau_giu_nguyen_phan_duoi():
    out = rerank("q", _fused(), DOCS, _fixed_scorer({"d1": 0.1, "d2": 0.5, "d3": 0.9}), top_k=3)
    assert [d.cord_uid for d in out] == ["d3", "d2", "d1", "d4", "d5", "d6"]
    assert all(d.source == "rerank" for d in out)
    check_ranking(out)


def test_phan_dau_luon_cao_hon_phan_duoi_ke_ca_logit_am_va_hoa():
    out = rerank("q", _fused(), DOCS, _fixed_scorer({"d1": -5, "d2": -7, "d3": -7}), top_k=3)
    head, tail = out[:3], out[3:]
    assert min(d.score for d in head) > max(d.score for d in tail)
    # Hoà logit -> phá theo cord_uid giảm (giống trec_eval): d3 trước d2.
    assert [d.cord_uid for d in head] == ["d1", "d3", "d2"]
    assert head[0].metadata["ce_score"] == -5
    assert head[0].metadata["fused_rank"] == 1
    check_ranking(out)


def test_it_ung_vien_hon_top_k_va_bai_khong_abstract():
    seen = []

    def scorer(query, articles):
        seen.extend(articles)
        return [float(len(a)) for a in articles]

    fused = to_ranked({"d5": 0.9, "d6": 0.8}, "convex")
    out = rerank("q", fused, DOCS, scorer, top_k=50)
    assert seen == ["title 5 abstract 5", "title only 6"]
    assert [d.cord_uid for d in out] == ["d5", "d6"]
    check_ranking(out)


def test_scorer_tra_sai_so_luong_thi_raise():
    with pytest.raises(ValueError, match="scorer"):
        rerank("q", _fused(), DOCS, lambda q, a: [0.0], top_k=3)


def test_danh_sach_rong():
    assert rerank("q", [], DOCS, _fixed_scorer({}), top_k=50) == []


def test_fuse_and_rerank_theo_hop_dong():
    bm25 = [ScoredDocument("d1", 9.0, 1, "bm25"), ScoredDocument("d2", 8.0, 2, "bm25")]
    dense = [ScoredDocument("d3", 0.9, 1, "dense"), ScoredDocument("d1", 0.8, 2, "dense")]
    out = fuse_and_rerank(
        "q",
        bm25,
        dense,
        rerank_top_k=2,
        fusion_config={"active": "rrf", "rrf": {"k": 60}},
        docs=DOCS,
        scorer=_fixed_scorer({"d1": 0.0, "d2": 0.0, "d3": 1.0}),
    )
    # RRF: d1 (2 nhánh) đứng đầu, rồi d3, d2 -> rerank top-2 {d1, d3}: d3 lên đầu.
    assert [d.cord_uid for d in out] == ["d3", "d1", "d2"]
    check_ranking(out)


def test_medcpt_that_cham_bai_lien_quan_cao_hon():
    # Chỉ chạy khi model đã có trong cache HF, để pytest không tải 400 MB trên máy khác.
    hub = pytest.importorskip("huggingface_hub")
    if not isinstance(hub.try_to_load_from_cache("ncbi/MedCPT-Cross-Encoder", "config.json"), str):
        pytest.skip("MedCPT-Cross-Encoder chưa có trong cache HF")
    from trec_covid.rerank.cross_encoder import CrossEncoderScorer

    scorer = CrossEncoderScorer("ncbi/MedCPT-Cross-Encoder", device="cpu", fp16=False)
    rel, irr = scorer(
        "remdesivir clinical trial",
        [
            "Remdesivir in adults with severe COVID-19: a randomised trial",
            "Seasonal weather patterns and humidity in coastal cities",
        ],
    )
    assert rel > irr
