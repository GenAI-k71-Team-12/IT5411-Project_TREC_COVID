"""Bảng ablation: mỗi dòng là một tầng, cùng 35 topic, theo đúng thứ tự pipeline.

    BM25 | Dense | Fusion | +Rerank | +Residual

Ngoài độ đo trec_eval còn có nDCG@10 condensed và judged@10 (ADR 0010): qrels chỉ có
vòng 1-2 nên nDCG@10 thường thiên vị các tầng giống pool cũ. Mỗi dòng có Δ so với dòng
ngay trước để thấy đóng góp của từng tầng. Commit + config ghi ở dòng đầu CSV
(`evaluate/results.py`).
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
    table["delta_ndcg10"] = table["ndcg_cut_10"].diff()
    table["delta_condensed"] = table["ndcg10_condensed"].diff()
    return table
