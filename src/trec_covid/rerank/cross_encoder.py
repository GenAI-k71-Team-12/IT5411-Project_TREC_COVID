"""MedCPT Cross-Encoder: checkpoint thứ ba, khác hai checkpoint bi-encoder.

Input: cặp (query, title + abstract), cắt 512 token.
Output: MỘT điểm cho mỗi cặp (logit, có thể âm; KHÔNG phải xác suất 0-1), nên không
cache theo vector được.

Chấm top-50, không phải top-1000 (170k cặp sẽ mất hàng giờ).

BẮT BUỘC: ghi điểm đơn điệu giảm theo thứ hạng cuối. Nếu lẫn thang điểm logit với
điểm fusion trong cùng cột score, trec_eval sẽ đảo thứ hạng. `rerank()` dời phần đầu
lên trên hẳn phần đuôi rồi kiểm tra bằng `check_ranking()`, không tự sort lại.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from trec_covid.fuse.ranking import check_ranking, to_ranked
from trec_covid.schemas import ScoredDocument

#: (query, danh sách văn bản bài) -> một điểm cho mỗi bài, cùng thứ tự.
Scorer = Callable[[str, Sequence[str]], Sequence[float]]


def article_text(title: str, abstract: str) -> str:
    """Văn bản bài đưa vào cross-encoder: `title abstract`, chỉ `title` nếu không có abstract."""
    return f"{title} {abstract}".strip() if abstract else title.strip()


class CrossEncoderScorer:
    """Nạp MedCPT Cross-Encoder MỘT lần, chấm theo batch."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        batch_size: int = 32,
        device: str = "auto",
        fp16: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.model = model.half() if fp16 and device == "cuda" else model

    def __call__(self, query: str, articles: Sequence[str]) -> list[float]:
        import torch

        scores: list[float] = []
        for start in range(0, len(articles), self.batch_size):
            batch = articles[start : start + self.batch_size]
            enc = self.tokenizer(
                [[query, a] for a in batch],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.squeeze(-1).float().cpu()
            scores.extend(logits.tolist())
        return scores


def rerank(
    query: str,
    fused: Sequence[ScoredDocument],
    docs: Mapping[str, tuple[str, str]],
    scorer: Scorer,
    top_k: int = 50,
) -> list[ScoredDocument]:
    """Chấm lại `top_k` bài đầu bằng cross-encoder, phần đuôi giữ thứ tự fusion.

    Điểm phần đầu = `tail_max + 1 + (ce - ce_min)` nên luôn lớn hơn hẳn phần đuôi, kể
    cả khi logit âm. Logit gốc nằm ở `metadata["ce_score"]`, hạng trước rerank ở
    `metadata["fused_rank"]`.
    """
    if not fused:
        return []
    head, tail = list(fused[:top_k]), list(fused[top_k:])
    ce = list(scorer(query, [article_text(*docs[d.cord_uid]) for d in head]))
    if len(ce) != len(head):
        raise ValueError(f"scorer trả {len(ce)} điểm cho {len(head)} bài")

    tail_max = tail[0].score if tail else 0.0
    ce_min = min(ce)
    head_scores = {d.cord_uid: tail_max + 1.0 + (s - ce_min) for d, s in zip(head, ce, strict=True)}
    meta = {
        d.cord_uid: {**d.metadata, "ce_score": s, "fused_rank": d.rank}
        for d, s in zip(head, ce, strict=True)
    }
    out = to_ranked(head_scores, "rerank", meta)
    out += [
        ScoredDocument(d.cord_uid, d.score, len(head) + i + 1, "rerank", dict(d.metadata))
        for i, d in enumerate(tail)
    ]
    check_ranking(out)
    return out
