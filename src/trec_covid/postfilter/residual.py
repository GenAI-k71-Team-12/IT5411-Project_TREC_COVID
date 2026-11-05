"""Postfilter sau khi đã có thứ hạng cuối: ba chế độ, mặc định TẮT (ADR 0005).

    off     không lọc (mặc định cho đánh giá offline và UI)
    docids  chỉ giữ bài nằm trong `docids-rnd3.txt`, là DANH SÁCH ĐƯỢC PHÉP của vòng 3
            (ADR 0008). Đúng thể lệ, nhưng no-op với Doc Store bản 19/05/2020
    judged  residual đúng nghĩa TREC: với mỗi topic, bỏ bài đã có phán định ở vòng trước
            (lấy từ qrels). Chấm cục bộ bằng chính qrels đó thì sẽ xoá bài có nhãn

Lọc TRƯỚC khi ghi run file. Sau khi lọc, rank đánh lại 1..n; score giữ nguyên nên
vẫn đơn điệu giảm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from trec_covid.fuse.ranking import check_ranking
from trec_covid.schemas import ScoredDocument

MODES: tuple[str, ...] = ("off", "docids", "judged")


def load_allowed_docids(path: str | Path) -> frozenset[str]:
    """Đọc `docids-rnd3.txt` (một id mỗi dòng; file gốc có dòng lặp)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy danh sách docids: {p}")
    with p.open(encoding="utf-8") as fh:
        return frozenset(line.strip() for line in fh if line.strip())


def load_judged(qrels: Mapping[str, Mapping[str, int]]) -> dict[str, frozenset[str]]:
    """`{topic_id: tập bài đã có phán định (mọi nhãn, kể cả 0)}`."""
    return {str(t): frozenset(docs) for t, docs in qrels.items()}


def apply_postfilter(
    results: Sequence[ScoredDocument],
    *,
    mode: str,
    topic_id: str,
    allowed: frozenset[str] | None = None,
    judged: Mapping[str, frozenset[str]] | None = None,
) -> list[ScoredDocument]:
    """Lọc danh sách của một topic theo `mode`, đánh lại rank, kiểm tra bất biến."""
    if mode not in MODES:
        raise ValueError(f"postfilter mode không hợp lệ: {mode!r} (chỉ có {MODES})")
    if mode == "off":
        return list(results)
    if mode == "docids":
        if allowed is None:
            raise ValueError("mode=docids cần `allowed` (load_allowed_docids)")
        keep = [d for d in results if d.cord_uid in allowed]
    else:
        if judged is None:
            raise ValueError("mode=judged cần `judged` (load_judged)")
        drop = judged.get(str(topic_id), frozenset())
        keep = [d for d in results if d.cord_uid not in drop]
    out = [
        ScoredDocument(d.cord_uid, d.score, i + 1, d.source, dict(d.metadata))
        for i, d in enumerate(keep)
    ]
    check_ranking(out)
    return out
