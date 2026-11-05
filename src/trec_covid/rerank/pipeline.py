"""`fuse_and_rerank`: đầu ra của Người 3 theo đúng hợp đồng docs/DATA_CONTRACTS.md §2.3.

Chữ ký chuẩn giữ nguyên bốn tham số đầu. Các tham số keyword-only để test và ablation
truyền cấu hình, Doc Store hay scorer khác vào; bỏ trống thì nạp từ `configs/` một lần
rồi cache lại cho mọi truy vấn sau (model cross-encoder nạp mất vài giây).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from trec_covid.fuse.rrf import fuse
from trec_covid.rerank.cross_encoder import CrossEncoderScorer, Scorer, rerank
from trec_covid.schemas import ScoredDocument


@lru_cache(maxsize=1)
def default_docs() -> dict[str, tuple[str, str]]:
    """`cord_uid -> (title, abstract)` từ Doc Store trong `configs/base.yaml`."""
    from trec_covid.config import load_config, resolve_path
    from trec_covid.io.doc_store import load_doc_store

    df = load_doc_store(
        resolve_path(load_config("base"), "doc_store"), columns=["cord_uid", "title", "abstract"]
    )
    return dict(zip(df["cord_uid"], zip(df["title"], df["abstract"], strict=True), strict=True))


@lru_cache(maxsize=1)
def default_scorer() -> CrossEncoderScorer:
    """Cross-encoder theo `configs/rerank.yaml`."""
    from trec_covid.config import load_config

    cfg = load_config("rerank")
    return CrossEncoderScorer(
        cfg["model"],
        max_length=int(cfg["input"]["max_length"]),
        batch_size=int(cfg["batch_size"]),
        device=str(cfg.get("device", "auto")),
    )


def fuse_and_rerank(
    query: str,
    bm25_results: list[ScoredDocument],
    dense_results: list[ScoredDocument],
    rerank_top_k: int = 50,
    *,
    fusion_config: Mapping[str, Any] | None = None,
    docs: Mapping[str, tuple[str, str]] | None = None,
    scorer: Scorer | None = None,
) -> list[ScoredDocument]:
    """Dung hợp hai nhánh theo `configs/fusion.yaml` rồi rerank `rerank_top_k` bài đầu."""
    if fusion_config is None:
        from trec_covid.config import load_config

        fusion_config = load_config("fusion")
    fused = fuse(bm25_results, dense_results, fusion_config)
    return rerank(
        query,
        fused,
        docs if docs is not None else default_docs(),
        scorer if scorer is not None else default_scorer(),
        top_k=rerank_top_k,
    )
