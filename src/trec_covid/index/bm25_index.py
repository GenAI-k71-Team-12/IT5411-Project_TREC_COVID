"""Chỉ mục BM25 — nhánh từ khoá (lexical / sparse retrieval).

Hai backend, cùng một giao diện:

    pyserini  Lucene thật, nhanh, chuẩn TREC-COVID. Cần JDK 21.
    bm25s     thuần Python, không cần Java. Chậm hơn nhưng chạy được mọi nơi.

Cả hai đều đánh chỉ mục **một trường văn bản** dựng bằng
`preprocess.text.weighted_text(title, abstract, title_weight)` — title được lặp
`title_weight` lần để xấp xỉ trọng số trường (`title² + abstract` trong README).

Bố cục thư mục sinh ra:

    indexes/lucene/
      meta.json            # backend, k1, b, n_docs, tham số đã dùng
      corpus.jsonl         # văn bản đã ghép, đúng thứ tự Doc Store
      lucene/              # (backend pyserini)
      bm25s/               # (backend bm25s)
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trec_covid.preprocess.text import weighted_text

META_FILE = "meta.json"
CORPUS_FILE = "corpus.jsonl"
LUCENE_SUBDIR = "lucene"
BM25S_SUBDIR = "bm25s"

#: Backend mặc định khi backend ưu tiên không dùng được.
FALLBACK_BACKEND = "bm25s"


# ---------------------------------------------------------------------------
# Chọn backend
# ---------------------------------------------------------------------------


def pyserini_available() -> bool:
    """Pyserini cài được chưa (chưa kiểm tra JVM)."""
    try:
        import pyserini  # noqa: F401
    except Exception:
        return False
    return True


def java_available() -> bool:
    """Có `java` trong PATH không."""
    return shutil.which("java") is not None


def detect_backend(preferred: str | None = None) -> str:
    """Chọn backend thực tế dùng được.

    Không bao giờ raise: nếu thiếu pyserini hoặc thiếu Java thì tự lùi về
    `bm25s` và in cảnh báo, để cả nhóm không bị chặn vì môi trường.
    """
    want = (preferred or "pyserini").lower()
    if want == "bm25s":
        return "bm25s"
    if want != "pyserini":
        raise ValueError(f"Backend không hợp lệ: {preferred!r} (chỉ có 'pyserini' | 'bm25s')")

    if pyserini_available() and java_available():
        return "pyserini"

    reasons = []
    if not pyserini_available():
        reasons.append("chưa cài pyserini")
    if not java_available():
        reasons.append("không thấy java trong PATH")
    print(f"[bm25] Không dùng được pyserini ({', '.join(reasons)}) -> lùi về bm25s")
    return FALLBACK_BACKEND


# ---------------------------------------------------------------------------
# Dựng corpus
# ---------------------------------------------------------------------------


def build_corpus(doc_store: pd.DataFrame, *, title_weight: float = 2.0) -> list[dict[str, str]]:
    """Sinh danh sách bản ghi `{"id": cord_uid, "contents": văn bản}`.

    Thứ tự phần tử **trùng khớp tuyệt đối** với thứ tự dòng của Doc Store.
    """
    docs: list[dict[str, str]] = []
    for cord_uid, title, abstract in zip(
        doc_store["cord_uid"], doc_store["title"], doc_store["abstract"], strict=True
    ):
        docs.append(
            {
                "id": str(cord_uid),
                "contents": weighted_text(title, abstract, title_weight),
            }
        )
    return docs


def write_corpus_jsonl(docs: list[dict[str, str]], path: str | Path) -> Path:
    """Ghi corpus ra JSONL (mỗi dòng một bài) cho pyserini đọc."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return out


def _sha256_of_texts(docs: list[dict[str, str]]) -> str:
    h = hashlib.sha256()
    for doc in docs:
        h.update(doc["id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(doc["contents"].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Backend bm25s
# ---------------------------------------------------------------------------


def _bm25s_tokenize(texts: list[str], *, stopwords: str | None):
    import bm25s

    return bm25s.tokenize(texts, stopwords=stopwords, show_progress=False, return_ids=False)


def build_bm25s_index(
    docs: list[dict[str, str]],
    out_dir: Path,
    *,
    k1: float,
    b: float,
    stopwords: str | None = "en",
) -> Path:
    """Dựng chỉ mục bm25s (thuần Python) và lưu xuống đĩa."""
    import bm25s

    tokenized = _bm25s_tokenize([d["contents"] for d in docs], stopwords=stopwords)
    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(tokenized, show_progress=False)

    target = out_dir / BM25S_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    retriever.save(str(target), show_progress=False)
    return target


# ---------------------------------------------------------------------------
# Backend pyserini / Lucene
# ---------------------------------------------------------------------------


def build_lucene_index(
    corpus_path: Path,
    out_dir: Path,
    *,
    threads: int = 4,
    language: str = "en",
) -> Path:
    """Gọi `pyserini.index.lucene` để dựng chỉ mục Lucene.

    Dùng subprocess thay vì API Python vì đường dẫn CLI ổn định hơn qua các
    bản pyserini. `--storePositions --storeDocvectors --storeRaw` là bộ cờ
    chuẩn để sau này còn tách snippet / kiểm tra postings.
    """
    target = out_dir / LUCENE_SUBDIR
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        # JsonCollection nhận THƯ MỤC và đánh chỉ mục mọi *.jsonl bên trong,
        # nên thư mục chỉ mục phải sạch: chỉ có corpus.jsonl (không để file
        # jsonl khác trong indexes/lucene/).
        "--input", str(corpus_path.parent),
        "--index", str(target),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--language", language,
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    print("[bm25] " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(
            "Dựng chỉ mục Lucene thất bại:\n" + "\n".join(tail)
        )
    return target


# ---------------------------------------------------------------------------
# API chung
# ---------------------------------------------------------------------------


def build_index(
    doc_store: pd.DataFrame,
    out_dir: str | Path,
    *,
    k1: float = 0.9,
    b: float = 0.4,
    title_weight: float = 2.0,
    backend: str | None = None,
    analyzer: str = "EnglishAnalyzer",
    threads: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    """Dựng chỉ mục BM25 và ghi `meta.json`.

    Args:
        doc_store: Doc Store đã sạch, thứ tự dòng là thứ tự đánh chỉ mục.
        out_dir: thư mục đích (thường `indexes/lucene`).
        k1, b: tham số BM25.
        title_weight: số lần lặp title.
        backend: `pyserini` | `bm25s` | None (tự dò).
        force: dựng lại dù đã có.

    Returns:
        dict `meta` đã ghi ra đĩa.
    """
    out = Path(out_dir)
    meta_path = out / META_FILE
    if meta_path.is_file() and not force:
        print(f"[bm25] Đã có chỉ mục tại {out} — bỏ qua (dùng force=True để dựng lại)")
        return load_index_meta(out)

    actual = detect_backend(backend)
    docs = build_corpus(doc_store, title_weight=title_weight)
    if not docs:
        raise ValueError("Doc Store rỗng, không có gì để đánh chỉ mục")

    out.mkdir(parents=True, exist_ok=True)
    corpus_path = write_corpus_jsonl(docs, out / CORPUS_FILE)

    if actual == "pyserini":
        build_lucene_index(corpus_path, out, threads=threads)
        index_subdir = LUCENE_SUBDIR
    else:
        build_bm25s_index(docs, out, k1=k1, b=b)
        index_subdir = BM25S_SUBDIR

    meta: dict[str, Any] = {
        "backend": actual,
        "index_subdir": index_subdir,
        "n_docs": len(docs),
        "k1": float(k1),
        "b": float(b),
        "title_weight": float(title_weight),
        "analyzer": analyzer,
        "stopwords": "en" if actual == "bm25s" else None,
        "corpus_sha256": _sha256_of_texts(docs),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[bm25] Xong: {len(docs):,} bài, backend={actual}, k1={k1}, b={b}")
    return meta


def load_index_meta(index_dir: str | Path) -> dict[str, Any]:
    """Đọc `meta.json` của chỉ mục đã dựng."""
    path = Path(index_dir) / META_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy {path} — chưa dựng chỉ mục?")
    return json.loads(path.read_text(encoding="utf-8"))


def index_dir_for(index_dir: str | Path, meta: Mapping[str, Any]) -> Path:
    """Đường dẫn thư mục con chứa chỉ mục thật của backend đang dùng."""
    return Path(index_dir) / str(meta.get("index_subdir", BM25S_SUBDIR))
