"""Chấm run bằng `pytrec_eval` (bọc đúng mã C của trec_eval, cài bằng pip, xem ADR 0009).

Các bẫy đã biết (ADR 0002), giữ nguyên khi dùng pytrec_eval:
    - trec_eval mặc định KHÔNG tính nDCG -> yêu cầu tường minh `ndcg_cut.10,20`
    - không dùng `-c`: chỉ tính trung bình trên topic có trong cả run lẫn qrels,
      topic 36-40 (không có phán định) không kéo trung bình về 0
    - qrels chỉ có topic 1-35 -> điểm trung bình trên 35 topic
    - trec_eval tự sắp lại theo score (hoà thì docid giảm), bỏ qua cột rank
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytrec_eval

from trec_covid.fuse.ranking import to_ranked
from trec_covid.schemas import ScoredDocument

#: Độ đo báo cáo. nDCG@10 là độ đo chính của TREC-COVID.
MEASURES: tuple[str, ...] = ("ndcg_cut_10", "ndcg_cut_20", "map", "P_5", "bpref")

#: Tên tham số gửi cho trec_eval để ra đúng các khoá trong MEASURES.
_TREC_EVAL_ARGS = {"ndcg_cut.10,20", "map", "P.5", "bpref"}


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Đọc `qrels.csv` của cuộc thi (`topic-id,iteration,cord-id,judgement`).

    Returns:
        `{topic_id: {cord_uid: nhãn}}`. Cột iteration (vòng chấm) bị bỏ.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy qrels: {p}")
    df = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    qrels: dict[str, dict[str, int]] = {}
    for tid, uid, rel in zip(df["topic-id"], df["cord-id"], df["judgement"], strict=True):
        qrels.setdefault(str(tid).strip(), {})[str(uid).strip()] = int(rel)
    return qrels


def evaluate_run(
    run: Mapping[str, Sequence[ScoredDocument]],
    qrels: Mapping[str, Mapping[str, int]],
    measures: Sequence[str] = MEASURES,
) -> pd.DataFrame:
    """Chấm từng topic. Trả DataFrame index `topic_id`, cột theo `measures`.

    Chỉ những topic có trong cả run lẫn qrels mới xuất hiện (trec_eval không `-c`).
    """
    evaluator = pytrec_eval.RelevanceEvaluator(
        {t: dict(d) for t, d in qrels.items()}, _TREC_EVAL_ARGS
    )
    scores = {str(t): {d.cord_uid: d.score for d in docs} for t, docs in run.items()}
    result = evaluator.evaluate(scores)
    df = pd.DataFrame.from_dict(result, orient="index")
    if df.empty:
        return pd.DataFrame(columns=list(measures))
    df = df[list(measures)]
    df.index.name = "topic_id"
    return df.sort_index(key=lambda idx: idx.astype(int))


def summarize(per_topic: pd.DataFrame) -> dict[str, float]:
    """Trung bình từng độ đo trên các topic đã chấm, kèm `n_topics`."""
    out: dict[str, float] = {m: float(per_topic[m].mean()) for m in per_topic.columns}
    out["n_topics"] = len(per_topic)
    return out


def judged_at_k(
    run: Mapping[str, Sequence[ScoredDocument]],
    qrels: Mapping[str, Mapping[str, int]],
    k: int = 10,
) -> pd.Series:
    """Tỉ lệ bài trong top-k CÓ phán định (kể cả nhãn 0), theo từng topic có qrels.

    qrels của cuộc thi chỉ có vòng 1-2, nên phần lớn bài vòng 3 chưa được chấm và bị
    coi là không liên quan. judged@10 thấp nghĩa là nDCG@10 đang đánh giá thấp nhánh đó;
    đọc kèm bpref (ít nhạy với bài chưa chấm) trước khi kết luận nhánh nào kém.
    """
    out = {
        str(t): sum(d.cord_uid in qrels[str(t)] for d in docs[:k]) / k
        for t, docs in run.items()
        if str(t) in qrels
    }
    return pd.Series(out, name=f"judged_{k}", dtype=float)


def condense(
    run: Mapping[str, Sequence[ScoredDocument]],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, list[ScoredDocument]]:
    """Run "condensed" (Sakai 2007): bỏ bài chưa có phán định rồi xếp lại, theo từng topic.

    Chấm nDCG@10 trên run condensed (nDCG'@10) là cách so sánh công bằng khi qrels không
    đầy đủ: nhánh truy hồi nhiều bài mới (chưa ai chấm) không bị phạt oan. Topic không có
    qrels bị bỏ. Đây là độ đo dùng để tuning fusion (ADR 0010); nDCG@10 thô vẫn được báo.
    """
    return {
        str(t): to_ranked(
            {d.cord_uid: d.score for d in docs if d.cord_uid in qrels[str(t)]},
            docs[0].source if docs else "",
        )
        for t, docs in run.items()
        if str(t) in qrels
    }
