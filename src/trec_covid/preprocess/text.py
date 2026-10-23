"""Chuẩn hoá văn bản dùng chung cho cả hai nhánh truy xuất.

Phải là MỘT hàm duy nhất: nếu hai nhánh chuẩn hoá khác nhau thì query đến
BM25 và query đến encoder sẽ lệch nhau, làm hỏng phép so sánh.

Interface dự kiến:

    def normalize_query(q: str) -> str: ...
    def query_terms(q: str) -> list[str]: ...   # đã stemming, đã bỏ stopword

Chưa cài đặt.
"""
