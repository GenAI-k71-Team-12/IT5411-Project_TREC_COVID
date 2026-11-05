"""Chấm một hoặc nhiều run file (nDCG@10, nDCG@20, MAP, P@5, bpref) trên các topic có qrels.

Ví dụ:

    python scripts/05_evaluate.py --run runs/bm25.txt
    python scripts/05_evaluate.py --run runs/bm25.txt --run runs/rrf.txt \
        --out results/2026-11-05_bm25-vs-rrf.csv --per-topic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import pandas as pd  # noqa: E402

from trec_covid.config import load_config, resolve_any_path  # noqa: E402
from trec_covid.evaluate.results import write_results_csv  # noqa: E402
from trec_covid.evaluate.run_writer import read_run  # noqa: E402
from trec_covid.evaluate.trec_eval import evaluate_run, load_qrels, summarize  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chấm run file bằng trec_eval (pytrec_eval)")
    p.add_argument("--run", type=Path, action="append", required=True, help="lặp lại được")
    p.add_argument("--qrels", type=Path, default=None, help="mặc định competition.qrels")
    p.add_argument("--out", type=Path, default=None, help="ghi bảng tổng ra CSV")
    p.add_argument("--per-topic", action="store_true", help="ghi thêm bảng từng topic")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config("base")
    qrels_path = args.qrels or resolve_any_path(cfg, "qrels", "competition")
    qrels = load_qrels(qrels_path)

    rows, per_topic_frames = [], []
    for run_path in args.run:
        per_topic = evaluate_run(read_run(run_path), qrels)
        rows.append({"run": run_path.stem, **summarize(per_topic)})
        per_topic_frames.append(per_topic.assign(run=run_path.stem).reset_index())

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.out:
        prov = {"qrels": Path(qrels_path).name, "runs": ";".join(p.name for p in args.run)}
        write_results_csv(table, args.out, prov)
        print(f"[eval] Ghi {args.out}")
        if args.per_topic:
            per_topic_out = args.out.with_name(args.out.stem + "_per-topic.csv")
            write_results_csv(pd.concat(per_topic_frames), per_topic_out, prov)
            print(f"[eval] Ghi {per_topic_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
