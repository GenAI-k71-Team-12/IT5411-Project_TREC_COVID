"""Xếp hạng và bất biến xếp hạng dùng chung cho fusion, rerank, postfilter, run file.

trec_eval sắp lại run theo `score` giảm dần, hoà điểm thì theo docid **giảm dần**,
và bỏ qua cột `rank`. Vì vậy mọi tầng của Người 3 xếp theo đúng khoá đó, để thứ
hạng mình tính trùng với thứ hạng trec_eval nhìn thấy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from trec_covid.schemas import ScoredDocument


def to_ranked(
    scores: Mapping[str, float],
    source: str,
    metadata: Mapping[str, dict] | None = None,
) -> list[ScoredDocument]:
    """Đổi `{cord_uid: score}` thành danh sách `ScoredDocument` đã xếp hạng (rank từ 1)."""
    ordered = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    meta = metadata or {}
    return [
        ScoredDocument(
            cord_uid=uid,
            score=float(score),
            rank=i + 1,
            source=source,
            metadata=dict(meta.get(uid, {})),
        )
        for i, (uid, score) in enumerate(ordered)
    ]


def check_ranking(results: Sequence[ScoredDocument]) -> None:
    """Kiểm tra bất biến; sai thì raise `ValueError`, KHÔNG tự sort lại để che lỗi.

    Bất biến: rank là 1..n liền mạch, không trùng `cord_uid`, `score` không tăng.
    """
    seen: set[str] = set()
    for i, doc in enumerate(results):
        if doc.rank != i + 1:
            raise ValueError(f"rank sai ở vị trí {i}: {doc.rank} (phải là {i + 1})")
        if doc.cord_uid in seen:
            raise ValueError(f"cord_uid trùng: {doc.cord_uid}")
        seen.add(doc.cord_uid)
        if i and doc.score > results[i - 1].score:
            raise ValueError(
                f"score không đơn điệu giảm ở rank {doc.rank}: {results[i - 1].score} < {doc.score}"
            )
