"""Doc Store -> encode MedCPT -> FAISS IndexFlatIP (~2-3 giờ GPU).

    python scripts/03_build_faiss_index.py --limit 2000      # smoke test, ghi smoke.faiss
    python scripts/03_build_faiss_index.py                   # chạy thật (làm tiếp nếu bị ngắt)
    python scripts/03_build_faiss_index.py --force           # xoá chunk, encode lại từ đầu
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from trec_covid.config import load_config, resolve_path  # noqa: E402
from trec_covid.index.dense_index import (  # noqa: E402
    MedCPTEncoder,
    build_faiss_index,
    encode_corpus,
    save_dense_index,
)
from trec_covid.io.doc_store import load_doc_store, load_docid_order  # noqa: E402


def utf8_console() -> None:
    """In tiếng Việt an toàn khi stdout bị redirect trên Windows (mặc định cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dựng chỉ mục FAISS từ Doc Store bằng MedCPT")
    p.add_argument("--config", default="dense")
    p.add_argument("--limit", type=int, default=None, help="chỉ N bài đầu (smoke test)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="bỏ chunk cũ, encode lại")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    cfg = load_config(args.config)
    enc_cfg, emb_cfg = cfg["encode"], cfg["embedding"]

    out_path = args.out or resolve_path(cfg, "faiss_index")
    chunk_dir = resolve_path(cfg, "embeddings")
    if args.limit:                                   # smoke test KHÔNG được ghi đè chỉ mục thật
        out_path = out_path.with_name("smoke.faiss")
        chunk_dir = chunk_dir.with_name(chunk_dir.name + "_smoke")
    if args.force and chunk_dir.exists():
        shutil.rmtree(chunk_dir)

    df = load_doc_store(resolve_path(cfg, "doc_store"), columns=["cord_uid", "title", "abstract"])
    docids = load_docid_order(resolve_path(cfg, "docid_order"))
    # Bất biến: Doc Store và docid_order cùng thứ tự, nếu không position sẽ trỏ sai bài.
    if df["cord_uid"].astype(str).tolist() != docids:
        raise SystemExit("[dense] docid_order.parquet KHÔNG khớp thứ tự Doc Store — dựng lại bằng "
                         "scripts/01_build_docstore.py, tuyệt đối không sort lại Doc Store.")
    if args.limit:
        df, docids = df.iloc[: args.limit], docids[: args.limit]
    print(f"[dense] {len(df):,} bài; model={cfg['model']['article_encoder']}")

    encoder = MedCPTEncoder(
        cfg["model"]["article_encoder"],
        max_length=int(emb_cfg["article_max_length"]),
        batch_size=args.batch_size or int(enc_cfg["batch_size"]),
        device=args.device or enc_cfg["device"],
        fp16=bool(enc_cfg["fp16"]),
    )
    print(f"[dense] device={encoder.device}")

    vectors = encode_corpus(
        df["title"].fillna("").tolist(),
        df["abstract"].fillna("").tolist(),
        encoder,
        chunk_dir,
        chunk_size=int(enc_cfg["chunk_size"]),
        signature={
            "model": cfg["model"]["article_encoder"],
            "max_length": int(emb_cfg["article_max_length"]),
            "fp16": bool(enc_cfg["fp16"]),
        },
    )
    index = build_faiss_index(vectors, docids)       # chuẩn hoá L2 nằm trong hàm này
    save_dense_index(
        index,
        docids,
        out_path,
        {
            "article_encoder": cfg["model"]["article_encoder"],
            "query_encoder": cfg["model"]["query_encoder"],
            "article_max_length": int(emb_cfg["article_max_length"]),
            "fp16": bool(enc_cfg["fp16"]),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    print(f"[dense] ntotal={index.ntotal:,} dim={index.d} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
