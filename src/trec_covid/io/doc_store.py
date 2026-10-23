"""Doc Store — nguồn sự thật cho mọi tầng phía sau.

Interface dự kiến:

    def load_cord19(metadata_csv: Path) -> pd.DataFrame: ...
    def build_doc_store(df: pd.DataFrame) -> pd.DataFrame: ...
    def load_doc_store(path: Path) -> pd.DataFrame: ...

Cột bắt buộc: cord_uid, title, abstract, authors, journal, year, doi, pmcid.

LƯU Ý THIẾT KẾ:
    Giữ `abstract` nguyên một chuỗi. KHÔNG lưu `sentences[]` cho 170k bài —
    việc tách câu làm lúc truy vấn, chỉ cho 10 bài hiển thị (ADR 0003).

    Giữ cả bài chỉ có title, không có abstract: vẫn có thể nằm trong qrels.

Chưa cài đặt.
"""
