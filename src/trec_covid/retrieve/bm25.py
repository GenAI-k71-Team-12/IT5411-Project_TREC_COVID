"""Nhánh từ khoá — truy vấn chỉ mục BM25.

Mạnh: tên thuốc, mã gen, mã thử nghiệm, biến thể virus (trùng khớp chính xác).
Yếu: viết tắt, từ đồng nghĩa.

Trả về `list[ScoredDocument]` theo đúng hợp đồng `docs/DATA_CONTRACTS.md`:
`score` giảm dần, `rank` 1-indexed, `source = "bm25"`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trec_covid.preprocess.text import normalize_query
from trec_covid.schemas import ScoredDocument

DEFAULT_TOP_K = 1000


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------


def _load_ids_from_corpus(corpus_path: Path) -> list[str]:
    """Đọc danh sách `cord_uid` theo đúng thứ tự dòng của chỉ mục.

    Dùng `corpus.jsonl` thay vì `docid_order.parquet` để module này không phụ
    thuộc vào Doc Store — chỉ mục tự mô tả được thứ tự của chính nó.
    """
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Không thấy corpus: {corpus_path}")
    ids: list[str] = []
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)["id"]))
    return ids


def _to_scored_documents(
    cord_uids: Sequence[str],
    scores: Sequence[float],
    top_k: int,
) -> list[ScoredDocument]:
    """Đóng gói kết quả thô thành `ScoredDocument` đúng hợp đồng."""
    results: list[ScoredDocument] = []
    for i, (cord_uid, score) in enumerate(zip(cord_uids, scores, strict=False)):
        if i >= top_k:
            break
        results.append(
            ScoredDocument(
                cord_uid=str(cord_uid),
                score=float(score),
                rank=i + 1,
                source="bm25",
            )
        )
    return results


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Bm25Retriever:
    """Nạp chỉ mục BM25 **một lần** rồi dùng lại cho mọi truy vấn.

    Khởi động JVM của pyserini tốn ~2–5 s, nên tạo retriever một lần ở đầu
    tiến trình, không tạo mỗi truy vấn.
    """

    def __init__(
        self,
        index_dir: str | Path,
        *,
        k1: float | None = None,
        b: float | None = None,
        backend: str | None = None,
    ) -> None:
        from trec_covid.index.bm25_index import (
            CORPUS_FILE,
            META_FILE,
            index_dir_for,
            load_index_meta,
        )

        self.index_root = Path(index_dir)
        if not (self.index_root / META_FILE).is_file():
            raise FileNotFoundError(
                f"Chưa dựng chỉ mục BM25 tại {self.index_root} "
                "(chạy scripts/02_build_bm25_index.py)"
            )

        self.meta: dict[str, Any] = load_index_meta(self.index_root)
        self.backend = (backend or self.meta.get("backend") or "bm25s").lower()
        self.k1 = float(k1 if k1 is not None else self.meta.get("k1", 0.9))
        self.b = float(b if b is not None else self.meta.get("b", 0.4))
        self._index_path = index_dir_for(self.index_root, self.meta)
        self._cord_uids = _load_ids_from_corpus(self.index_root / CORPUS_FILE)
        self._stopwords = self.meta.get("stopwords", "en")

        if self.backend == "pyserini":
            self._searcher = self._open_lucene()
        elif self.backend == "bm25s":
            self._retriever = self._open_bm25s()
        else:
            raise ValueError(f"Backend không hỗ trợ: {self.backend}")

    # -- khởi tạo backend ---------------------------------------------------

    def _open_lucene(self):
        from pyserini.search.lucene import LuceneSearcher

        searcher = LuceneSearcher(str(self._index_path))
        # BẮT BUỘC gọi trước search(); quên là Lucene lặng lẽ dùng 1.2 / 0.75.
        searcher.set_bm25(self.k1, self.b)
        return searcher

    def _open_bm25s(self):
        import bm25s

        # bm25s lưu k1/b NGAY TRONG chỉ mục: truyền k1/b lúc nạp không có tác
        # dụng. Tuning k1/b bắt buộc phải dựng lại (scripts/02 --k1 --b --force).
        meta_k1 = self.meta.get("k1")
        meta_b = self.meta.get("b")
        if (meta_k1 is not None and abs(float(meta_k1) - self.k1) > 1e-9) or (
            meta_b is not None and abs(float(meta_b) - self.b) > 1e-9
        ):
            print(
                f"[bm25] CẢNH BÁO: chỉ mục bm25s đang dùng k1={meta_k1}, b={meta_b}; "
                f"tham số yêu cầu k1={self.k1}, b={self.b} KHÔNG được áp dụng. "
                "Dựng lại chỉ mục nếu muốn đổi tham số."
            )
        return bm25s.BM25.load(str(self._index_path), load_corpus=False)

    # -- truy vấn -----------------------------------------------------------

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[ScoredDocument]:
        """Tìm `top_k` bài cho `query`.

        Args:
            query: chuỗi truy vấn thô (sẽ được chuẩn hoá ở tầng dưới).
            top_k: số kết quả tối đa.

        Returns:
            Danh sách `ScoredDocument`, `score` giảm dần, `rank` từ 1.
        """
        if not query or not query.strip():
            return []

        # Chuẩn hoá an toàn (NBSP, gạch nối Unicode, entity...) trước khi đưa
        # vào analyzer. Không escape ký tự Lucene: Pyserini tự dựng BooleanQuery
        # từ token đã phân tích, escape sẽ làm hỏng truy vấn.
        query = normalize_query(query)
        if not query:
            return []

        k = max(1, min(int(top_k), len(self._cord_uids)))

        if self.backend == "pyserini":
            hits = self._searcher.search(query, k=k)
            return _to_scored_documents(
                [h.docid for h in hits], [h.score for h in hits], top_k
            )

        from trec_covid.index.bm25_index import _bm25s_tokenize

        tokens = _bm25s_tokenize([query], stopwords=self._stopwords)
        if not tokens or not tokens[0]:
            # Toàn stopword / toàn ký tự đặc biệt -> không có tín hiệu truy vấn.
            # bm25s sẽ trả k bài đầu bảng với điểm 0, gây hiểu nhầm là "có kết quả".
            return []
        indices, scores = self._retriever.retrieve(tokens, k=k, show_progress=False)
        row_idx = indices[0].tolist() if hasattr(indices[0], "tolist") else list(indices[0])
        row_scores = scores[0].tolist() if hasattr(scores[0], "tolist") else list(scores[0])
        cord_uids = [self._cord_uids[i] for i in row_idx]
        return _to_scored_documents(cord_uids, row_scores, top_k)

    # -- tiện ích -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cord_uids)

    def __repr__(self) -> str:
        return (
            f"Bm25Retriever(backend={self.backend!r}, n_docs={len(self)}, "
            f"k1={self.k1}, b={self.b})"
        )


# ---------------------------------------------------------------------------
# Hàm tiện dụng cho người dùng cuối
# ---------------------------------------------------------------------------

_CACHE: dict[tuple[str, float, float], Bm25Retriever] = {}


def search_bm25(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    index_dir: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[ScoredDocument]:
    """Tìm kiếm BM25 một phát, tự cache retriever theo `(index_dir, k1, b)`.

    Args:
        query: chuỗi truy vấn.
        top_k: số kết quả.
        index_dir: thư mục chỉ mục; mặc định lấy `paths.lucene_index` trong config.
        config: dict cấu hình đã merge (xem `trec_covid.config.load_config`).
    """
    if config is None:
        from trec_covid.config import load_config

        config = load_config("bm25")

    if index_dir is None:
        from trec_covid.config import resolve_path

        index_dir = resolve_path(config, "lucene_index")

    k1 = float(config.get("k1", 0.9))
    b = float(config.get("b", 0.4))
    key = (str(Path(index_dir).resolve()), k1, b)
    if key not in _CACHE:
        _CACHE[key] = Bm25Retriever(index_dir, k1=k1, b=b)
    return _CACHE[key].search(query, top_k)
