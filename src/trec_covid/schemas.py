"""Quy ước kiểu dữ liệu (Data Contracts / Schemas) dùng chung cho toàn bộ pipeline TREC-COVID.

Tất cả các module (BM25, Dense, Fusion, Reranker, Snippet, UI Web)
PHẢI tuân thủ các dataclass trong file này để đảm bảo khả năng tích hợp trơn tru.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class DocumentRecord:
    """Bản ghi bài báo được lưu trong Doc Store (doc_store.parquet)."""
    cord_uid: str
    title: str
    abstract: str
    authors: str = ""
    journal: str = ""
    publish_time: str = ""
    doi: str = ""
    pmcid: str = ""

    @property
    def full_text_representation(self) -> str:
        """Đại diện văn bản dùng cho BM25 hoặc Dense Retrieval (title + abstract)."""
        if not self.abstract:
            return self.title
        return f"{self.title} {self.abstract}".strip()


@dataclass
class ScoredDocument:
    """Kết quả truy xuất của một tài liệu từ một nhánh (BM25 hoặc Dense) hoặc sau Fusion/Rerank."""
    cord_uid: str
    score: float
    rank: int                # 1-indexed, bắt buộc truyền: 1 = hạng cao nhất
    source: str = ""         # 'bm25', 'dense', 'rrf', 'rerank'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank phải >= 1 (1-indexed), nhận được {self.rank}")


@dataclass
class CandidateForRerank:
    """Đầu vào cho mô hình Cross-Encoder Reranker."""
    cord_uid: str
    query: str
    title: str
    abstract: str
    original_rank: int
    original_score: float


@dataclass
class SnippetResult:
    """Kết quả trích dẫn trích đoạn mức câu cho một bài báo."""
    cord_uid: str
    sentence: str            # Câu nguyên văn được chọn từ abstract
    sentence_index: int      # Vị trí câu trong abstract (0-indexed)
    score: float             # Điểm tương đồng giữa câu và query
    highlighted_html: str    # Câu nguyên văn đã được bọc thẻ <mark class="hl">...</mark>
    has_abstract: bool = True


@dataclass
class SearchResultItem:
    """Một bài báo hoàn chỉnh hiển thị trên giao diện người dùng (UI Web)."""
    cord_uid: str
    rank: int
    score: float
    title: str
    authors: str
    journal: str
    publish_time: str
    doi: str
    abstract: str
    snippet_html: str
    snippet_text: str
    has_abstract: bool = True


@dataclass
class SearchResponse:
    """Toàn bộ phản hồi tìm kiếm trả về cho trang Web UI."""
    query: str
    total_hits: int
    page: int
    page_size: int
    total_pages: int
    latency_ms: float
    results: List[SearchResultItem]
