"""Doc Store — nguồn sự thật cho mọi tầng phía sau.

Doc Store là **một bảng parquet tĩnh**, mỗi dòng một bài báo, khoá chính
`cord_uid`. Không phải database: chỉ đọc, không UPDATE, nạp cả bảng vào RAM.

Cột bắt buộc: cord_uid, title, abstract, authors, journal, publish_time, doi, pmcid.

LƯU Ý THIẾT KẾ:
    Giao HAI ĐỢT (xem docs/DATA_CONTRACTS.md mục 2.0):
      Đợt 1 (2 ngày đầu): cord_uid, title, abstract — mốc chặn cho Người 2 encode.
      Đợt 2: 5 trường metadata còn lại + docid_order.parquet (position, cord_uid).

    Giữ `abstract` nguyên một chuỗi. KHÔNG lưu `sentences[]` cho 170k bài —
    việc tách câu làm lúc truy vấn, chỉ cho 10 bài hiển thị (ADR 0003).

    Giữ cả bài chỉ có title, không có abstract: vẫn có thể nằm trong qrels.

    Thứ tự dòng phải TẤT ĐỊNH: FAISS trả vị trí nguyên, nên không ai được
    sort lại Doc Store sau khi đã encode.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from trec_covid.preprocess.text import normalize_document

#: Ba trường tối thiểu — đợt bàn giao 1.
CORE_COLUMNS: tuple[str, ...] = ("cord_uid", "title", "abstract")

#: Năm trường metadata bổ sung — đợt bàn giao 2.
METADATA_COLUMNS: tuple[str, ...] = ("authors", "journal", "publish_time", "doi", "pmcid")

ALL_COLUMNS: tuple[str, ...] = CORE_COLUMNS + METADATA_COLUMNS

#: Tên cột trong `metadata.csv` của CORD-19 -> tên cột chuẩn của Doc Store.
_CORD19_ALIASES: dict[str, tuple[str, ...]] = {
    "cord_uid": ("cord_uid",),
    "title": ("title",),
    "abstract": ("abstract",),
    "authors": ("authors",),
    "journal": ("journal",),
    "publish_time": ("publish_time", "publish_date", "year"),
    "doi": ("doi",),
    "pmcid": ("pmcid",),
}

#: Cột văn bản được chuẩn hoá (các cột còn lại chỉ strip).
_TEXT_COLUMNS: tuple[str, ...] = ("title", "abstract")


# ---------------------------------------------------------------------------
# Đọc CORD-19
# ---------------------------------------------------------------------------


def load_cord19(
    metadata_csv: str | Path,
    *,
    columns: Sequence[str] = ALL_COLUMNS,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Đọc `metadata.csv` của CORD-19, chỉ lấy các cột cần.

    Chỉ đọc 8/22 cột: `metadata.csv` ~200 MB nhưng phần lớn cột không dùng
    (`sha`, `license`, `url`, `fulltext_file`, ...).

    Args:
        metadata_csv: đường dẫn tới `metadata.csv`.
        columns: cột muốn lấy (theo tên chuẩn của Doc Store).
        nrows: giới hạn số dòng, dùng cho test/smoke.

    Returns:
        DataFrame với các cột có mặt trong file, đã đổi về tên chuẩn.

    Raises:
        FileNotFoundError: không thấy file.
        ValueError: thiếu cột `cord_uid`.
    """
    path = Path(metadata_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy metadata.csv: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    header_lower = {c.lower(): c for c in header}

    rename: dict[str, str] = {}
    wanted: list[str] = []
    for canonical in columns:
        for alias in _CORD19_ALIASES[canonical]:
            actual = header_lower.get(alias.lower())
            if actual is not None:
                wanted.append(actual)
                rename[actual] = canonical
                break

    if "cord_uid" not in rename.values():
        raise ValueError(f"{path.name} không có cột cord_uid — không phải metadata CORD-19")

    # keep_default_na=False: CORD-19 có abstract rỗng, để pandas biến thành NaN
    # là mầm lỗi về sau (schema yêu cầu chuỗi rỗng).
    df = pd.read_csv(
        path,
        usecols=wanted,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        nrows=nrows,
        low_memory=False,
    )
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Dựng Doc Store
# ---------------------------------------------------------------------------


def build_doc_store(
    df: pd.DataFrame,
    *,
    minimal: bool = False,
    sort: bool = True,
) -> pd.DataFrame:
    """Chuẩn hoá DataFrame thô thành Doc Store.

    Việc làm:
      1. Bổ sung cột thiếu bằng chuỗi rỗng, ép mọi cột về `str`
      2. Chuẩn hoá `title` và `abstract` bằng `normalize_document()`
      3. Bỏ dòng không có `cord_uid`
      4. Khử trùng `cord_uid`, ưu tiên bản có abstract dài hơn
      5. Sắp xếp theo `cord_uid` (mergesort) để thứ tự dòng **tất định**
      6. Đánh lại index 0..N-1 liên tục

    Args:
        df: DataFrame thô từ `load_cord19`.
        minimal: `True` thì chỉ giữ 3 cột lõi (đợt bàn giao 1).
        sort: sắp xếp theo `cord_uid`. Chỉ đặt `False` khi test thứ tự.

    Returns:
        Doc Store đã sạch, sẵn sàng ghi parquet.
    """
    if "cord_uid" not in df.columns:
        raise ValueError("DataFrame thiếu cột cord_uid")

    target = list(CORE_COLUMNS if minimal else ALL_COLUMNS)
    # Bỏ index gốc: người dùng có thể truyền DataFrame đã lọc/sample (index lỗ
    # chỗ hoặc trùng), gán cột theo index đó sẽ lệch dòng.
    df = df.reset_index(drop=True)
    out = pd.DataFrame(index=df.index)

    for col in target:
        if col in df.columns:
            series = df[col]
        else:
            series = pd.Series([""] * len(df), index=df.index, dtype="object")
        # NaN/None -> "" ; số (nếu có) -> str
        out[col] = series.where(series.notna(), "").astype(str).str.strip()

    for col in _TEXT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(normalize_document)

    out = out[out["cord_uid"] != ""]

    if out.empty:
        raise ValueError("Doc Store rỗng sau khi bỏ dòng thiếu cord_uid")

    # Ưu tiên bản ghi có abstract khi trùng cord_uid. Sắp thêm theo chính
    # `abstract` để hoà độ dài vẫn chọn được CÙNG một dòng dù đầu vào bị xáo
    # trộn -> Doc Store tái lập được từ bất kỳ thứ tự CSV nào.
    if "abstract" in out.columns:
        out = out.assign(_abs_len=out["abstract"].str.len())
        out = out.sort_values(
            ["cord_uid", "_abs_len", "abstract"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        out = out.drop_duplicates(subset="cord_uid", keep="first").drop(columns="_abs_len")
    else:
        out = out.drop_duplicates(subset="cord_uid", keep="first")

    if sort:
        out = out.sort_values("cord_uid", kind="mergesort")

    out = out.reset_index(drop=True)
    return out[target]


def validate_doc_store(df: pd.DataFrame, *, minimal: bool = False) -> None:
    """Kiểm tra bất biến của Doc Store. Raise nếu vi phạm.

    Không tự sửa — sửa ngầm sẽ che mất lỗi ở tầng dựng dữ liệu.
    """
    required = CORE_COLUMNS if minimal else ALL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Doc Store thiếu cột: {missing}")

    if len(df) == 0:
        raise ValueError("Doc Store rỗng")

    if df["cord_uid"].duplicated().any():
        dup = df.loc[df["cord_uid"].duplicated(), "cord_uid"].head(3).tolist()
        raise ValueError(f"cord_uid bị trùng, ví dụ: {dup}")

    for col in required:
        if df[col].isna().any():
            raise ValueError(f"Cột {col} còn NaN — phải là chuỗi rỗng")

    if (df["cord_uid"] == "").any():
        raise ValueError("Có dòng cord_uid rỗng")


# ---------------------------------------------------------------------------
# Ghi / đọc
# ---------------------------------------------------------------------------


def save_doc_store(df: pd.DataFrame, path: str | Path, *, compression: str = "zstd") -> Path:
    """Ghi Doc Store ra parquet (nén zstd — abstract là văn bản tự nhiên nên ~1,5–2x)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression=compression)
    return out


def load_doc_store(path: str | Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Đọc Doc Store từ parquet."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy Doc Store: {p}")
    return pd.read_parquet(p, columns=list(columns) if columns else None)


def build_docid_order(df: pd.DataFrame, path: str | Path | None = None) -> pd.DataFrame:
    """Sinh bảng ánh xạ `position -> cord_uid` cho FAISS.

    FAISS `IndexFlatIP` chỉ trả **vị trí dòng** (0, 1, 2, ...), không biết
    `cord_uid`. Bảng này là cầu nối, phải sinh **cùng lúc** với Doc Store.

    Returns:
        DataFrame hai cột `position` (int64) và `cord_uid` (str).
    """
    order = pd.DataFrame(
        {
            "position": range(len(df)),
            "cord_uid": df["cord_uid"].to_numpy(),
        }
    )
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        order.to_parquet(out, index=False, compression="zstd")
    return order


def load_docid_order(path: str | Path) -> list[str]:
    """Đọc `docid_order.parquet`, trả danh sách `cord_uid` theo đúng thứ tự vị trí."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy docid_order: {p}")
    order = pd.read_parquet(p).sort_values("position", kind="mergesort")
    return order["cord_uid"].astype(str).tolist()


def doc_store_stats(df: pd.DataFrame) -> dict[str, int | float]:
    """Vài con số để in ra log và đưa vào báo cáo."""
    n = len(df)
    has_abs = int((df["abstract"].str.len() > 0).sum()) if "abstract" in df else 0
    return {
        "n_docs": n,
        "n_with_abstract": has_abs,
        "n_title_only": n - has_abs,
        "pct_with_abstract": round(100.0 * has_abs / n, 2) if n else 0.0,
        "n_unique_cord_uid": int(df["cord_uid"].nunique()),
    }


def iter_texts(df: pd.DataFrame, columns: Iterable[str] = _TEXT_COLUMNS) -> Iterable[str]:
    """Sinh văn bản đã ghép để encode/đánh chỉ mục, theo đúng thứ tự dòng."""
    cols = [c for c in columns if c in df.columns]
    for row in df[cols].itertuples(index=False, name=None):
        yield " ".join(part for part in row if part)
