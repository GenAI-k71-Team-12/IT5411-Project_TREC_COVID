"""Doc Store — nguồn sự thật cho mọi tầng phía sau.

Interface dự kiến:

    def load_cord19(metadata_csv: Path) -> pd.DataFrame: ...
    def build_doc_store(df: pd.DataFrame) -> pd.DataFrame: ...
    def load_doc_store(path: Path) -> pd.DataFrame: ...

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

Chưa cài đặt.
"""
