"""Bảng ablation: mỗi dòng là một tầng, cùng 35 topic, theo đúng thứ tự pipeline.

    BM25 | Dense | Fusion | +Rerank | +Residual

Ngoài độ đo trec_eval còn có nDCG@10 condensed và judged@10 (ADR 0010): qrels chỉ có
vòng 1-2 nên nDCG@10 thường thiên vị các tầng giống pool cũ. Commit + config ghi ở dòng
đầu CSV (`evaluate/results.py`).

Δ (đóng góp của từng tầng): `n_branches` dòng đầu là các nhánh SONG SONG (BM25, Dense), không
xếp chồng lên nhau nên không có Δ. Dòng dung hợp so với nhánh đơn tốt nhất theo TỪNG độ đo.
Các dòng sau so với dòng ngay trước.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from trec_covid.evaluate.run_writer import read_run
from trec_covid.evaluate.trec_eval import condense, evaluate_run, judged_at_k, summarize


def ablation_table(
    stages: Sequence[tuple[str, str | Path]],
    qrels: Mapping[str, Mapping[str, int]],
    n_branches: int = 2,
) -> pd.DataFrame:
    """Chấm từng run theo thứ tự `stages` = [(tên tầng, đường dẫn run), ...]."""
    rows = []
    for name, path in stages:
        run = read_run(path)
        condensed = evaluate_run(condense(run, qrels), qrels)["ndcg_cut_10"]
        rows.append(
            {
                "stage": name,
                "run": Path(path).name,
                **summarize(evaluate_run(run, qrels)),
                "ndcg10_condensed": float(condensed.mean()) if len(condensed) else 0.0,
                "judged_10": float(judged_at_k(run, qrels).mean()),
            }
        )
    table = pd.DataFrame(rows)
    for metric, delta in (("ndcg_cut_10", "delta_ndcg10"), ("ndcg10_condensed", "delta_condensed")):
        prev = table[metric].shift()
        if n_branches < len(table):
            prev.iloc[n_branches] = table[metric].iloc[:n_branches].max()
        prev.iloc[:n_branches] = float("nan")
        table[delta] = table[metric] - prev
    return table
