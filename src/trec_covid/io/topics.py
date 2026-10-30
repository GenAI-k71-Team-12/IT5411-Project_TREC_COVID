"""Đọc file topic của cuộc thi (`topics-rnd3.csv`).

Topic TREC-COVID có ba trường văn bản khác nhau, độ dài và độ nhiễu tăng dần:

    query      "coronavirus origin"                     — vài từ, rất ngắn
    question   "what is the origin of COVID-19"          — một câu hỏi
    narrative  "Seeking information about the origin ..." — một đoạn mô tả

Chọn trường nào là **một siêu tham số**, không phải chi tiết kỹ thuật: narrative
dài làm loãng BM25 (nhiều từ chung), nhưng bổ sung ngữ cảnh cho encoder. Vì vậy
trường được chọn nằm trong `configs/base.yaml: competition.query_field`, không
hard-code trong script — để cả nhóm đo cả ba rồi chốt bằng số liệu.

Module này **không** chuẩn hoá văn bản: việc đó thuộc
`preprocess.text.normalize_query()` và do tầng truy xuất gọi, để mọi nhánh đi
qua đúng một làn xử lý.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: Ba trường văn bản có thể dùng làm truy vấn.
TOPIC_FIELDS: tuple[str, ...] = ("query", "question", "narrative")

#: Tên cột có thể gặp ở các bản export khác nhau -> tên chuẩn.
#: (`topic-id` và `topicId` là biến thể có dấu gạch nối/dấu hoa thường gặp
#: trong file export gốc của TREC; phải nhận cả hai).
_ALIASES: dict[str, tuple[str, ...]] = {
    "topic_id": ("topic_id", "topic-id", "topicId", "topic", "id", "number"),
    "query": ("query", "title", "topic_title"),
    "question": ("question", "questions"),
    "narrative": ("narrative", "description", "narratives"),
}


@dataclass(frozen=True)
class Topic:
    """Một topic của cuộc thi."""

    topic_id: int
    query: str = ""
    question: str = ""
    narrative: str = ""

    def text(self, field: str = "query") -> str:
        """Trả văn bản của trường `field`, đã strip. Raise nếu tên trường sai."""
        if field not in TOPIC_FIELDS:
            raise ValueError(f"Trường topic không hợp lệ: {field!r} (chỉ có {TOPIC_FIELDS})")
        return getattr(self, field).strip()


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột theo `_ALIASES`, không phân biệt hoa/thường."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            actual = lower.get(alias)
            if actual is not None:
                rename[actual] = canonical
                break
    return df.rename(columns=rename)


def load_topics(path: str | Path, *, n_topics: int | None = None) -> list[Topic]:
    """Đọc `topics-rnd3.csv` thành danh sách `Topic`.

    Args:
        path: đường dẫn file CSV.
        n_topics: chỉ lấy N topic đầu (dùng khi chạy thử). Mặc định lấy hết.

    Returns:
        Danh sách `Topic`, sắp tăng dần theo `topic_id`.

    Raises:
        FileNotFoundError: không thấy file.
        ValueError: thiếu cột `topic_id` hoặc `query`.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy file topic: {p}")

    # utf-8-sig: file export từ Windows/Kaggle thường có BOM, đọc utf-8 thuần sẽ
    # ra cột tên "\ufefftopic_id" và mọi tra cứu sau đó trượt.
    df = pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    df = _rename_columns(df)

    if "topic_id" not in df.columns:
        raise ValueError(f"{p.name} thiếu cột topic_id (có: {list(df.columns)})")
    if "query" not in df.columns:
        raise ValueError(f"{p.name} thiếu cột query (có: {list(df.columns)})")

    topics: list[Topic] = []
    for record in df.to_dict("records"):
        raw_id = str(record["topic_id"]).strip()
        if not raw_id:
            continue
        topics.append(
            Topic(
                topic_id=int(raw_id),
                query=str(record.get("query", "")),
                question=str(record.get("question", "")),
                narrative=str(record.get("narrative", "")),
            )
        )

    topics.sort(key=lambda t: t.topic_id)
    if n_topics is not None:
        topics = topics[: int(n_topics)]
    if not topics:
        raise ValueError(f"{p.name} không có topic nào")
    return topics


def build_queries(topics: Sequence[Topic], field: str = "query") -> list[tuple[int, str]]:
    """Sinh danh sách `(topic_id, văn bản truy vấn)` để chạy truy xuất hàng loạt."""
    out: list[tuple[int, str]] = []
    for topic in topics:
        text = topic.text(field)
        if text:
            out.append((topic.topic_id, text))
    return out
