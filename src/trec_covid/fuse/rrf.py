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

from collections.abc import Sequence

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
