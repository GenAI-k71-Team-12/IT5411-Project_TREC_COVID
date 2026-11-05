"""Ba công thức dung hợp, chạy trên CÙNG một tập ứng viên (hợp hai danh sách top-1000).

    (1) RRF chuẩn       RRF(d) = sum 1 / (k + rank(d)),        k = 60
    (2) Weighted RRF    RRF(d) = sum w / (k + rank(d))
    (3) Convex          s(d)   = a*bm25(d) + (1-a)*dense(d),   đã min-max

Vai trò — đây là thiết kế thí nghiệm, không phải ba lựa chọn ngang nhau:

    (1) đường cơ sở: chỉ dùng thứ hạng, không tinh chỉnh gì ngoài k
    (2) kiểm tra một nhánh có đáng tin hơn nhánh kia không
    (3) kiểm tra biên độ điểm có mang thông tin mà thứ hạng làm mất không

`rank(d)` lấy từ trường `rank` (1-based) của đầu vào, không lấy từ vị trí trong list.
Mỗi kết quả ghi `bm25_rank`, `dense_rank`, `bm25_score`, `dense_score` vào metadata
(None nếu vắng ở nhánh đó) cho chế độ so sánh trên UI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trec_covid.fuse.ranking import to_ranked
from trec_covid.schemas import ScoredDocument


def _branch_metadata(
    bm25: Sequence[ScoredDocument], dense: Sequence[ScoredDocument]
) -> dict[str, dict]:
    """Rank và điểm gốc của từng nhánh cho mọi doc trong hợp hai danh sách."""
    meta: dict[str, dict] = {}
    for name, docs in (("bm25", bm25), ("dense", dense)):
        for d in docs:
            entry = meta.setdefault(
                d.cord_uid,
                {"bm25_rank": None, "dense_rank": None, "bm25_score": None, "dense_score": None},
            )
            entry[f"{name}_rank"] = d.rank
            entry[f"{name}_score"] = d.score
    return meta


def weighted_rrf(
    bm25: Sequence[ScoredDocument],
    dense: Sequence[ScoredDocument],
    w_bm25: float,
    w_dense: float,
    k: int = 60,
    *,
    source: str = "weighted_rrf",
) -> list[ScoredDocument]:
    """RRF có trọng số: `sum_nhánh w / (k + rank)`."""
    scores: dict[str, float] = {}
    for weight, docs in ((w_bm25, bm25), (w_dense, dense)):
        for d in docs:
            scores[d.cord_uid] = scores.get(d.cord_uid, 0.0) + weight / (k + d.rank)
    return to_ranked(scores, source, _branch_metadata(bm25, dense))


def rrf(
    bm25: Sequence[ScoredDocument], dense: Sequence[ScoredDocument], k: int = 60
) -> list[ScoredDocument]:
    """RRF chuẩn (Cormack et al., SIGIR 2009): mọi nhánh trọng số 1."""
    return weighted_rrf(bm25, dense, 1.0, 1.0, k, source="rrf")


def _minmax(docs: Sequence[ScoredDocument]) -> dict[str, float]:
    """Min-max trên CHÍNH danh sách của truy vấn này (không dùng hằng số [-1, 1]).

    Điểm MedCPT co cụm ~[0.7, 0.95]; min-max theo thang lý thuyết sẽ nén mọi chênh lệch.
    Mọi điểm bằng nhau (max = min) thì mọi doc có mặt nhận 1.0.
    """
    if not docs:
        return {}
    lo = min(d.score for d in docs)
    hi = max(d.score for d in docs)
    if hi == lo:
        return {d.cord_uid: 1.0 for d in docs}
    return {d.cord_uid: (d.score - lo) / (hi - lo) for d in docs}


def convex(
    bm25: Sequence[ScoredDocument], dense: Sequence[ScoredDocument], alpha: float
) -> list[ScoredDocument]:
    """Convex: `alpha * bm25_norm + (1 - alpha) * dense_norm`; doc vắng ở nhánh nào nhận 0."""
    nb, nd = _minmax(bm25), _minmax(dense)
    scores = {
        uid: alpha * nb.get(uid, 0.0) + (1.0 - alpha) * nd.get(uid, 0.0)
        for uid in nb.keys() | nd.keys()
    }
    return to_ranked(scores, "convex", _branch_metadata(bm25, dense))


def drop_unmatched(bm25: Sequence[ScoredDocument]) -> list[ScoredDocument]:
    """Bỏ bài BM25 điểm <= 0: điểm 0 nghĩa là không khớp từ nào, không phải hoà điểm.

    bm25s vẫn trả đủ top-k với điểm 0 khi truy vấn không khớp (typo, từ ngoài từ vựng).
    Giữ lại thì min-max cho chúng 1.0 và RRF xen chúng vào ngay từ hạng 1. Bài còn lại
    giữ nguyên `rank` gốc (điểm dương luôn đứng trước điểm 0).
    """
    return [d for d in bm25 if d.score > 0]


def fuse(
    bm25: Sequence[ScoredDocument],
    dense: Sequence[ScoredDocument],
    config: Mapping[str, Any],
) -> list[ScoredDocument]:
    """Dung hợp theo `config["active"]` (khối `configs/fusion.yaml`), cắt còn `top_k`.

    Hợp hai top-1000 có tới ~1.800 bài; hợp đồng (DATA_CONTRACTS §2.3) giao top-1000.
    """
    bm25 = drop_unmatched(bm25)
    active = config.get("active")
    if active == "rrf":
        fused = rrf(bm25, dense, k=int(config["rrf"]["k"]))
    elif active == "weighted_rrf":
        c = config["weighted_rrf"]
        fused = weighted_rrf(bm25, dense, float(c["w_bm25"]), float(c["w_dense"]), int(c["k"]))
    elif active == "convex":
        fused = convex(bm25, dense, alpha=float(config["convex"]["alpha"]))
    else:
        raise ValueError(f"fusion.active không hợp lệ: {active!r} (rrf | weighted_rrf | convex)")
    return fused[: int(config.get("top_k", 1000))]
