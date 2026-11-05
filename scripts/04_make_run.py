"""Hai run nhánh -> fusion -> rerank -> run file.

Đọc run file của hai nhánh (bàn giao của Người 1 và Người 2) thay vì gọi retriever,
để tầng sau chạy lại được mà không cần chỉ mục. Tham số lấy từ `configs/fusion.yaml`
và `configs/rerank.yaml`.

    python scripts/04_make_run.py --out runs/rerank.txt
    python scripts/04_make_run.py --no-rerank --out runs/fused.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from trec_covid.config import load_config  # noqa: E402
from trec_covid.evaluate.run_writer import read_run, write_run  # noqa: E402
from trec_covid.fuse.rrf import fuse  # noqa: E402
from trec_covid.io.topics import load_topics  # noqa: E402
from trec_covid.rerank.cross_encoder import rerank  # noqa: E402
from trec_covid.rerank.pipeline import default_docs, default_scorer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusion + rerank từ hai run nhánh")
    p.add_argument("--bm25", type=Path, default=Path("runs/bm25.txt"))
    p.add_argument("--dense", type=Path, default=Path("runs/dense.txt"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", default=None, help="run-tag, mặc định là tên file")
    p.add_argument("--no-rerank", action="store_true", help="chỉ fusion")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fusion_cfg, rerank_cfg = load_config("fusion"), load_config("rerank")
    bm25, dense = read_run(args.bm25), read_run(args.dense)
    topics = {str(t.topic_id): t for t in load_topics(fusion_cfg["competition"]["topics"])}
    field = fusion_cfg["competition"]["query_field"]
    top_k = int(rerank_cfg["top_k"])
    docs = None if args.no_rerank else default_docs()
    scorer = None if args.no_rerank else default_scorer()

    run, latency_ms = {}, []
    for tid in sorted(bm25.keys() | dense.keys(), key=int):
        fused = fuse(bm25.get(tid, []), dense.get(tid, []), fusion_cfg)
        if scorer is not None:
            start = time.perf_counter()
            fused = rerank(topics[tid].text(field), fused, docs, scorer, top_k=top_k)
            latency_ms.append((time.perf_counter() - start) * 1000)
        run[tid] = fused

    write_run(run, args.out, args.tag or args.out.stem)
    print(f"[make-run] fusion={fusion_cfg['active']} rerank={'off' if args.no_rerank else top_k}")
    if latency_ms:
        print(
            f"[make-run] rerank top-{top_k}: median {statistics.median(latency_ms):.0f} ms, "
            f"max {max(latency_ms):.0f} ms / truy vấn ({len(latency_ms)} topic)"
        )
    print(f"[make-run] Ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
