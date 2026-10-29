"""CORD-19 (metadata.csv) -> Doc Store (parquet).

Hai đợt bàn giao, **cùng một đường dẫn** `data/processed/doc_store.parquet`
(xem docs/DATA_CONTRACTS.md mục 2.0):

    --minimal   ĐỢT 1 (mốc chặn ngày 2): 3 trường cord_uid, title, abstract.
                Người 2 dùng ngay để bắt đầu encode ~170k bài, không chờ đủ
                8 trường.
    (mặc định)  ĐỢT 2: đủ 8 trường. Ghi đè lên file tối giản — thứ tự dòng
                không đổi nên KHÔNG phải encode lại.

`docid_order.parquet` (`position` -> `cord_uid`) được sinh ở **cả hai đợt**:
FAISS chỉ trả vị trí số nguyên, nên Người 2 cần bảng này ngay từ đợt 1.

Ví dụ:

    python scripts/01_build_docstore.py --minimal
    python scripts/01_build_docstore.py
    python scripts/01_build_docstore.py --metadata data/raw/.../metadata.csv --nrows 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from trec_covid.config import load_config, resolve_path  # noqa: E402
from trec_covid.io.doc_store import (  # noqa: E402
    CORE_COLUMNS,
    build_doc_store,
    build_docid_order,
    doc_store_stats,
    load_cord19,
    save_doc_store,
    validate_doc_store,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dựng Doc Store từ metadata.csv của CORD-19")
    p.add_argument("--config", default="base", help="tầng cấu hình (mặc định: base)")
    p.add_argument("--metadata", type=Path, default=None, help="đường dẫn metadata.csv")
    p.add_argument("--out", type=Path, default=None, help="đường dẫn parquet đầu ra")
    p.add_argument(
        "--minimal",
        action="store_true",
        help="chỉ 3 trường lõi — đợt bàn giao 1 (mốc chặn ngày 2)",
    )
    p.add_argument("--no-docid-order", action="store_true", help="không sinh docid_order.parquet")
    p.add_argument("--nrows", type=int, default=None, help="chỉ đọc N dòng đầu (smoke test)")
    p.add_argument(
        "--force",
        action="store_true",
        help="cho phép --minimal ghi đè lên Doc Store đã có đủ 8 trường",
    )
    return p.parse_args(argv)


def _existing_columns(path: Path) -> list[str]:
    """Đọc tên cột của Doc Store đã có (rỗng nếu chưa có file)."""
    if not path.is_file():
        return []
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    metadata = args.metadata or resolve_path(cfg, "data_raw") / "metadata.csv"
    out_path = args.out or resolve_path(cfg, "doc_store")

    if args.minimal:
        existing = _existing_columns(out_path)
        if len(existing) > len(CORE_COLUMNS) and not args.force:
            print(
                f"[docstore] TỪ CHỐI: {out_path} đang có {len(existing)} cột. "
                "Chạy --minimal bây giờ sẽ xoá 5 trường metadata.\n"
                "           Dùng --force nếu thật sự muốn quay lại đợt 1.",
                file=sys.stderr,
            )
            return 2

    print(f"[docstore] Đọc {metadata}")
    raw = load_cord19(metadata, nrows=args.nrows)
    print(f"[docstore] {len(raw):,} dòng thô, {len(raw.columns)} cột")

    df = build_doc_store(raw, minimal=args.minimal)
    validate_doc_store(df, minimal=args.minimal)

    save_doc_store(df, out_path)
    kind = "tối giản (đợt 1)" if args.minimal else "đầy đủ (đợt 2)"
    print(f"[docstore] Ghi {out_path}  [{kind}, {len(df.columns)} cột]")

    stats = doc_store_stats(df)
    for key, value in stats.items():
        print(f"           {key:20s} {value}")

    if not args.no_docid_order:
        order_path = out_path.with_name("docid_order.parquet")
        build_docid_order(df, order_path)
        print(f"[docstore] Ghi {order_path}  ({len(df):,} dòng, position -> cord_uid)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
