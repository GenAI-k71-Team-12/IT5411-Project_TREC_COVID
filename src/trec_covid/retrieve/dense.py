"""Nhánh vector — encode query bằng MedCPT Query-Encoder rồi tìm trong FAISS.

Trả về `list[ScoredDocument]` đúng hợp đồng docs/DATA_CONTRACTS.md:
`score` = cosine (IndexFlatIP trên vector đã chuẩn hoá L2), giảm dần, `rank` từ 1,
`source = "dense"`.

LƯU Ý: dùng Query-Encoder (không dùng Article-Encoder) cho câu hỏi, và phải là cặp
checkpoint tương ứng với lúc encode corpus (ghi trong `articles.meta.json`).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trec_covid.preprocess.text import normalize_query
from trec_covid.schemas import ScoredDocument

DEFAULT_TOP_K = 1000


class DenseRetriever:
    """Nạp chỉ mục + Query Encoder **một lần** rồi dùng lại cho mọi truy vấn."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        query_model: str | None = None,
        max_length: int = 64,
        device: str = "auto",
        fp16: bool = True,
        encoder: Any | None = None,   # cho phép tiêm encoder giả khi test
    ) -> None:
        from trec_covid.index.dense_index import MedCPTEncoder, load_dense_index

        self.index, self._cord_uids, self.meta = load_dense_index(index_path)
        model = query_model or self.meta["query_encoder"]
        if encoder is None and model != self.meta["query_encoder"]:
            raise ValueError(
                f"Query encoder {model!r} khác lúc dựng chỉ mục "
                f"({self.meta['query_encoder']!r}): vector ở hai không gian khác nhau."
            )
        self.encoder = encoder or MedCPTEncoder(
            model, max_length=max_length, batch_size=1, device=device, fp16=fp16
        )

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[ScoredDocument]:
        from trec_covid.index.dense_index import l2_normalize

        query = normalize_query(query)
        if not query or top_k < 1:
            return []
        k = min(int(top_k), self.index.ntotal)
        qvec = l2_normalize(self.encoder.encode([query]))
        scores, ids = self.index.search(qvec, k)

        out: list[ScoredDocument] = []
        for score, pos in zip(scores[0], ids[0], strict=True):
            if pos < 0:                                    # FAISS: không đủ kết quả
                continue
            out.append(
                ScoredDocument(
                    cord_uid=self._cord_uids[int(pos)],
                    score=float(score),
                    rank=len(out) + 1,
                    source="dense",
                )
            )
        return out

    def __len__(self) -> int:
        return self.index.ntotal


_CACHE: dict[tuple[str, str], DenseRetriever] = {}


def search_dense(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    index_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[ScoredDocument]:
    """Chữ ký chuẩn: `search_dense(query, top_k=1000) -> list[ScoredDocument]`."""
    if config is None:
        from trec_covid.config import load_config

        config = load_config("dense")
    if index_path is None:
        from trec_covid.config import resolve_path

        index_path = resolve_path(config, "faiss_index")

    key = (str(Path(index_path).resolve()), config["model"]["query_encoder"])
    if key not in _CACHE:
        _CACHE[key] = DenseRetriever(
            index_path,
            query_model=config["model"]["query_encoder"],
            max_length=int(config["embedding"]["query_max_length"]),
            device=config["encode"]["device"],
            fp16=bool(config["encode"]["fp16"]),
        )
    return _CACHE[key].search(query, top_k)
