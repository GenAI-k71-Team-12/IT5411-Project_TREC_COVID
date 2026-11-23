"""Chạy nhánh dense trên mọi topic -> run file 6 cột (baseline runs/dense.txt).

    python scripts/11_make_dense_run.py
    python scripts/11_make_dense_run.py --field question --out runs/dense_question.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from trec_covid.config import load_config, resolve_any_path  # noqa: E402
from trec_covid.evaluate.run_writer import write_run  # noqa: E402
from trec_covid.io.topics import build_queries, load_topics  # noqa: E402
from trec_covid.retrieve.dense import search_dense  # noqa: E402


def utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sinh run file cho nhánh dense")
    p.add_argument("--out", type=Path, default=Path("runs/dense.txt"))
    p.add_argument("--field", default=None, help="query | question | narrative (mặc định: config)")
    p.add_argument("--index-path", type=Path, default=None, help="vd smoke.faiss")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--tag", default=None, help="run-tag, mặc định là tên file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    cfg = load_config("dense")
    field = args.field or cfg["competition"]["query_field"]
    top_k = args.top_k or int(cfg["top_k"])

    topics = load_topics(resolve_any_path(cfg, "topics", "competition"))
    queries = build_queries(topics, field)
    print(f"[dense-run] {len(queries)} topic, field={field}, top_k={top_k}")

    # nạp model/chỉ mục ngoài phần đo giờ
    search_dense("warm up", 10, config=cfg, index_path=args.index_path)
    run, latency_ms = {}, []
    for tid, text in queries:
        start = time.perf_counter()
        run[str(tid)] = search_dense(text, top_k, config=cfg, index_path=args.index_path)
        latency_ms.append((time.perf_counter() - start) * 1000)

    write_run(run, args.out, args.tag or args.out.stem)
    print(f"[dense-run] latency median {statistics.median(latency_ms):.0f} ms, "
          f"max {max(latency_ms):.0f} ms / truy vấn")
    print(f"[dense-run] Ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
