"""Hai run nhánh -> fusion -> rerank -> postfilter -> run file.

Đọc run file của hai nhánh (bàn giao của Người 1 và Người 2) thay vì gọi retriever,
để tầng sau chạy lại được mà không cần chỉ mục. Tham số lấy từ `configs/fusion.yaml`,
`configs/rerank.yaml` và `configs/postfilter.yaml`.

    python scripts/04_make_run.py --out runs/rerank.txt
    python scripts/04_make_run.py --no-rerank --out runs/fused.txt
    python scripts/04_make_run.py --from-run runs/rerank.txt --postfilter judged \
        --out runs/rerank_judged.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from trec_covid.config import load_config, resolve_any_path  # noqa: E402
from trec_covid.evaluate.run_writer import Run, read_run, write_run  # noqa: E402
from trec_covid.evaluate.trec_eval import load_qrels  # noqa: E402
from trec_covid.fuse.rrf import fuse  # noqa: E402
from trec_covid.io.topics import load_topics  # noqa: E402
from trec_covid.postfilter.residual import (  # noqa: E402
    MODES,
    apply_postfilter,
    load_allowed_docids,
    load_judged,
)
from trec_covid.rerank.cross_encoder import rerank  # noqa: E402
from trec_covid.rerank.pipeline import default_docs, default_scorer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusion + rerank + postfilter từ hai run nhánh")
    p.add_argument("--bm25", type=Path, default=Path("runs/bm25.txt"))
    p.add_argument("--dense", type=Path, default=Path("runs/dense.txt"))
    p.add_argument("--from-run", type=Path, default=None, help="bỏ qua fusion/rerank, chỉ lọc")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", default=None, help="run-tag, mặc định là tên file")
    p.add_argument("--no-rerank", action="store_true", help="chỉ fusion")
    p.add_argument("--postfilter", choices=MODES, default=None, help="mặc định theo config")
    return p.parse_args(argv)


def fuse_and_rerank_all(args: argparse.Namespace) -> Run:
    """Fusion (+ rerank) cho mọi topic, in latency rerank."""
    fusion_cfg, rerank_cfg = load_config("fusion"), load_config("rerank")
    bm25, dense = read_run(args.bm25), read_run(args.dense)
    topics = load_topics(resolve_any_path(fusion_cfg, "topics", "competition"))
    queries = {str(t.topic_id): t.text(fusion_cfg["competition"]["query_field"]) for t in topics}
    top_k = int(rerank_cfg["top_k"])
    docs = None if args.no_rerank else default_docs()
    scorer = None if args.no_rerank else default_scorer()

    run, latency_ms = {}, []
    for tid in sorted(bm25.keys() | dense.keys(), key=int):
        fused = fuse(bm25.get(tid, []), dense.get(tid, []), fusion_cfg)
        if scorer is not None:
            start = time.perf_counter()
            fused = rerank(queries[tid], fused, docs, scorer, top_k=top_k)
            latency_ms.append((time.perf_counter() - start) * 1000)
        run[tid] = fused

    print(f"[make-run] fusion={fusion_cfg['active']} rerank={'off' if args.no_rerank else top_k}")
    if latency_ms:
        print(
            f"[make-run] rerank top-{top_k}: median {statistics.median(latency_ms):.0f} ms, "
            f"max {max(latency_ms):.0f} ms / truy vấn ({len(latency_ms)} topic)"
        )
    return run


def postfilter_all(run: Run, mode: str) -> Run:
    """Áp postfilter cho mọi topic; chỉ nạp dữ liệu mà chế độ cần."""
    cfg = load_config("base")
    allowed = judged = None
    if mode == "docids":
        allowed = load_allowed_docids(resolve_any_path(cfg, "docids", "competition"))
    if mode == "judged":
        judged = load_judged(load_qrels(resolve_any_path(cfg, "qrels", "competition")))
    out = {
        t: apply_postfilter(docs, mode=mode, topic_id=t, allowed=allowed, judged=judged)
        for t, docs in run.items()
    }
    removed = sum(len(run[t]) - len(out[t]) for t in run)
    print(f"[make-run] postfilter={mode}: bỏ {removed:,} dòng")
    return out


def utf8_console() -> None:
    """In tiếng Việt an toàn khi stdout bị redirect trên Windows (mặc định cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    run = read_run(args.from_run) if args.from_run else fuse_and_rerank_all(args)
    mode = args.postfilter or str(load_config("postfilter")["mode"])
    run = postfilter_all(run, mode)
    write_run(run, args.out, args.tag or args.out.stem)
    print(f"[make-run] Ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
