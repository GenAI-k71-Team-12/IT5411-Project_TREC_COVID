"""Đo latency từng tầng (median, p95) trên các truy vấn của cuộc thi.

    bm25      search_bm25(query, 1000)            (Người 1)
    dense     search_dense(query, 1000)           (Người 2; chưa có thì ghi "chưa có")
    fuse      fuse(bm25, dense) theo configs/fusion.yaml
    rerank    cross-encoder top-k theo configs/rerank.yaml
    pipeline  bm25 + dense + fuse + rerank, đo liền một mạch

Đầu vào của fuse/rerank lấy từ kết quả bm25 thật và `runs/dense.txt` khi chưa có
search_dense, để vẫn đo được hai tầng của Người 3.

    python scripts/09_measure_latency.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import pandas as pd  # noqa: E402

from trec_covid.config import load_config, resolve_any_path, resolve_path  # noqa: E402
from trec_covid.evaluate.latency import time_stages  # noqa: E402
from trec_covid.evaluate.results import write_results_csv  # noqa: E402
from trec_covid.evaluate.run_writer import read_run  # noqa: E402
from trec_covid.fuse.rrf import fuse  # noqa: E402
from trec_covid.io.topics import build_queries, load_topics  # noqa: E402
from trec_covid.rerank.cross_encoder import rerank  # noqa: E402
from trec_covid.rerank.pipeline import default_docs, default_scorer  # noqa: E402
from trec_covid.retrieve.bm25 import search_bm25  # noqa: E402


def load_search_dense():
    """`search_dense` của Người 2, hoặc None nếu chưa cài đặt."""
    try:
        from trec_covid.retrieve.dense import search_dense
    except ImportError:
        return None
    return search_dense


def device_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Đo latency từng tầng")
    p.add_argument("--dense-run", type=Path, default=Path("runs/dense.txt"))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--tag", default=date.today().isoformat())
    return p.parse_args(argv)


def utf8_console() -> None:
    """In tiếng Việt an toàn khi stdout bị redirect trên Windows (mặc định cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    fusion_cfg, rerank_cfg = load_config("fusion"), load_config("rerank")
    topics = load_topics(resolve_any_path(fusion_cfg, "topics", "competition"))
    pairs = build_queries(topics, field=fusion_cfg["competition"]["query_field"])
    queries = [q for _, q in pairs]
    top_k = int(rerank_cfg["top_k"])
    docs, scorer, search_dense = default_docs(), default_scorer(), load_search_dense()

    dense_run = read_run(args.dense_run)
    dense_of = {q: dense_run.get(str(t), []) for t, q in pairs}
    bm25_of = {q: search_bm25(q, 1000) for q in queries}
    fused_of = {q: fuse(bm25_of[q], dense_of[q], fusion_cfg) for q in queries}

    stages = {
        "bm25": lambda q: search_bm25(q, 1000),
        "fuse": lambda q: fuse(bm25_of[q], dense_of[q], fusion_cfg),
        "rerank": lambda q: rerank(q, fused_of[q], docs, scorer, top_k=top_k),
    }
    dense_fn = (lambda q: search_dense(q, 1000)) if search_dense else (lambda q: dense_of[q])
    if search_dense:
        stages["dense"] = dense_fn
    stages["pipeline"] = lambda q: rerank(
        q, fuse(search_bm25(q, 1000), dense_fn(q), fusion_cfg), docs, scorer, top_k=top_k
    )

    table = time_stages(stages, queries, warmup=args.warmup)
    table["note"] = ""
    if not search_dense:
        missing = {"stage": "dense", "median_ms": float("nan"), "p95_ms": float("nan"),
                   "n": 0, "note": "chưa có search_dense (Người 2)"}  # fmt: skip
        table = pd.concat([table, pd.DataFrame([missing])], ignore_index=True)
        table.loc[table["stage"] == "pipeline", "note"] = "chưa gồm dense (đọc từ run file)"
    prov = {"device": device_name().replace(" ", "_"), "rerank_top_k": top_k,
            "fusion": fusion_cfg["active"], "queries": len(queries)}  # fmt: skip
    out = resolve_path(fusion_cfg, "results") / f"{args.tag}_latency.csv"
    write_results_csv(table, out, prov)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"[latency] Ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
