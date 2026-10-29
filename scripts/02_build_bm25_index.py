"""Doc Store -> chỉ mục Lucene BM25.

Ví dụ:

    python scripts/02_build_bm25_index.py
    python scripts/02_build_bm25_index.py --backend bm25s
    python scripts/02_build_bm25_index.py --k1 1.2 --b 0.75 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from trec_covid.config import load_config, resolve_path  # noqa: E402
from trec_covid.index.bm25_index import build_index  # noqa: E402
from trec_covid.io.doc_store import load_doc_store  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dựng chỉ mục BM25 từ Doc Store")
    p.add_argument("--config", default="bm25", help="tầng cấu hình (mặc định: bm25)")
    p.add_argument("--doc-store", type=Path, default=None, help="đường dẫn doc_store.parquet")
    p.add_argument("--out", type=Path, default=None, help="thư mục chỉ mục đầu ra")
    p.add_argument("--backend", choices=["pyserini", "bm25s"], default=None)
    p.add_argument("--k1", type=float, default=None)
    p.add_argument("--b", type=float, default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--force", action="store_true", help="dựng lại dù đã có chỉ mục")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    doc_store_path = args.doc_store or resolve_path(cfg, "doc_store")
    out_dir = args.out or resolve_path(cfg, "lucene_index")

    k1 = float(args.k1 if args.k1 is not None else cfg.get("k1", 0.9))
    b = float(args.b if args.b is not None else cfg.get("b", 0.4))
    title_weight = float(cfg.get("fields", {}).get("title", 2.0))

    print(f"[bm25] Doc Store: {doc_store_path}")
    df = load_doc_store(doc_store_path, columns=["cord_uid", "title", "abstract"])
    print(f"[bm25] {len(df):,} bài")

    meta = build_index(
        df,
        out_dir,
        k1=k1,
        b=b,
        title_weight=title_weight,
        backend=args.backend or cfg.get("implementation"),
        threads=args.threads,
        force=args.force,
    )
    print(f"[bm25] meta: {meta['backend']}, {meta['n_docs']:,} bài, k1={meta['k1']}, b={meta['b']}")
    print(f"[bm25] Chỉ mục tại: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
